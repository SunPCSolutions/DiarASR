#!/usr/bin/env python3
"""Run local Omi transcription in a per-request subprocess."""

import json
import os
import subprocess
import sys
from pathlib import Path

from config import get_config
from logging_config import get_worker_logger
from nemotron_diarization import MODEL as DIARIZATION_MODEL, speaker_bounds
from temp_file_tracker import TempFileTracker, TempFileTrackerConfig


OMI_MODEL = "omi-health/omi-med-stt-v1"
logger = get_worker_logger()


def merge_consecutive_speaker_segments(segments):
    merged = []
    for segment in sorted(segments, key=lambda item: item['start']):
        if (merged and merged[-1]['speaker'] == segment['speaker']
                and abs(segment['start'] - merged[-1]['end']) < 0.5):
            merged[-1]['end'] = max(merged[-1]['end'], segment['end'])
            merged[-1]['text'] += ' ' + segment['text']
        else:
            merged.append(segment.copy())
    return merged


def run_inference(request_data):
    import soundfile as sf
    from omi_stt.nemo_runtime import transcribe_nemo

    language = request_data.get('language') or 'en'
    if language.lower() not in {'en', 'english'}:
        raise ValueError('Omi Med STT only supports English')

    config = get_config()
    diarize = request_data.get('diarize', True)

    tracker_config = TempFileTrackerConfig(
        max_retry_attempts=config.processing.max_retry_attempts,
        enable_audit_logging=config.processing.enable_audit_logging,
        cleanup_timeout_seconds=config.processing.cleanup_timeout_seconds,
    )
    with TempFileTracker(tracker_config) as tracker:
        converted_path = tracker.create_temp_file('.wav', purpose='audio_conversion')
        conversion = subprocess.run(
            ['ffmpeg', '-nostdin', '-v', 'error', '-y', '-i', request_data['audio_path'],
             '-vn', '-ac', '1', '-ar', '16000', '-c:a', 'pcm_s16le', converted_path],
            capture_output=True,
        )
        if conversion.returncode:
            raise RuntimeError('Could not decode audio to 16 kHz mono WAV')

        if diarize:
            counts = [request_data.get(key) for key in ('num_speakers', 'min_speakers', 'max_speakers')]
            if all(count is None for count in counts):
                counts = [config.diarization.num_speakers, config.diarization.min_speakers, config.diarization.max_speakers]
            min_speakers, max_speakers = speaker_bounds(*counts)
            env = os.environ.copy()
            runtime_path = env.get('NEMOTRON_PYTHONPATH', '/opt/nemotron')
            if not Path(runtime_path, 'nemo').is_dir():
                raise RuntimeError('Nemotron runtime is missing; rebuild the Docker image')
            env['PYTHONPATH'] = os.pathsep.join(filter(None, (runtime_path, env.get('PYTHONPATH'))))
            if request_data.get('hf_token'):
                env['HF_TOKEN'] = request_data['hf_token']

            logger.info('Diarizing with %s', DIARIZATION_MODEL)
            result = subprocess.run(
                [sys.executable, str(Path(__file__).with_name('nemotron_diarization.py')), converted_path],
                env=env, stdout=subprocess.PIPE, text=True, timeout=300,
            )
            if result.returncode:
                raise RuntimeError('Nemotron 3 Diarization failed; see worker logs')
            speaker_segments = json.loads(result.stdout)
            count = len({segment['speaker'] for segment in speaker_segments})
            if ((min_speakers is not None and count < min_speakers)
                    or (max_speakers is not None and count > max_speakers)):
                return {'speaker_count_error': f'Nemotron detected {count} speaker(s), outside the requested bounds'}

            paths, metadata = [], []
            with sf.SoundFile(converted_path) as recording:
                rate = recording.samplerate
                for segment in speaker_segments:
                    start = max(0, min(recording.frames, int(segment['start'] * rate)))
                    end = max(start, min(recording.frames, int(segment['end'] * rate)))
                    if end - start < rate * 0.2:
                        continue
                    recording.seek(start)
                    clip = tracker.create_temp_file('.wav', purpose='asr_segment')
                    sf.write(clip, recording.read(end - start, dtype='int16'), rate, subtype='PCM_16')
                    paths.append(clip)
                    metadata.append({'start': start / rate, 'end': end / rate, 'speaker': segment['speaker']})
        else:
            paths = [converted_path]
            metadata = [{'start': 0.0, 'end': sf.info(converted_path).duration, 'speaker': 'SPEAKER_00'}]

        if not paths:
            return {'segments': []}
        logger.info('Transcribing %d input(s) with %s', len(paths), OMI_MODEL)
        transcripts = transcribe_nemo(paths, repo_id=OMI_MODEL)
        segments = [dict(segment, text=text) for segment, text in zip(metadata, transcripts, strict=True) if text]
        return {'segments': merge_consecutive_speaker_segments(segments) if diarize else segments}


def main():
    # NeMo and native libraries can write to stdout; reserve it for JSON.
    with os.fdopen(os.dup(1), 'w') as result_stream, open(os.devnull, 'w') as devnull:
        os.dup2(devnull.fileno(), 1)
        try:
            result = run_inference(json.load(sys.stdin))
        except Exception as exc:
            logger.exception('Transcription worker failed')
            result = {'error': str(exc)}
        json.dump(result, result_stream)


if __name__ == '__main__':
    main()

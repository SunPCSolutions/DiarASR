#!/usr/bin/env python3
"""
PyTorch Inference Worker - Runs in subprocess for memory isolation
"""

import os
import sys
import json
import tempfile
import torch
import gc
import logging
from config import get_config
from logging_config import get_worker_logger

# Completely suppress all stdout output except final JSON
import os
import sys
from contextlib import redirect_stdout, redirect_stderr

# Create a null device for suppressing output
null_device = open(os.devnull, 'w')

# Redirect all stdout to null device initially
old_stdout = sys.stdout
sys.stdout = null_device

# Initialize secure logging
logger = get_worker_logger()

def merge_consecutive_speaker_segments(segments):
    """
    Merge consecutive segments from the same speaker to ensure diarization-controlled segmentation.

    Args:
        segments: List of transcription segments with speaker labels

    Returns:
        List of merged segments where consecutive segments from same speaker are combined
    """
    if not segments:
        return segments

    # Sort by start time
    sorted_segments = sorted(segments, key=lambda x: x['start'])

    merged_segments = []
    current_segment = sorted_segments[0].copy()

    for next_segment in sorted_segments[1:]:
        # Check if this segment is consecutive from the same speaker
        if (next_segment['speaker'] == current_segment['speaker'] and
            abs(next_segment['start'] - current_segment['end']) < 0.5):  # 500ms gap tolerance
            # Merge segments
            current_segment['end'] = next_segment['end']
            current_segment['text'] += ' ' + next_segment['text']
        else:
            # Start new segment
            merged_segments.append(current_segment)
            current_segment = next_segment.copy()

    # Add the last segment
    merged_segments.append(current_segment)

    logger.info("Merged %d segments into %d diarization-controlled segments", len(sorted_segments), len(merged_segments))
    return merged_segments


def run_inference(request_data):
    """Run inference in isolated subprocess with automatic cleanup"""

    # Debug: Print received request_data
    print(f"DEBUG: Received request_data: {request_data}")

    # Set environment for optimal performance
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,garbage_collection_threshold:0.8,roundup_power2_divisions:1'

    # Pre-load models for faster inference (cached within subprocess)
    asr_models = {}
    diarizer_instances = {}

    try:
        # Import here to avoid main process contamination
        import nemo.collections.asr as nemo_asr
        from pydub import AudioSegment
        import torchaudio
        from hybrid_diarization import HybridDiarization

        # Parse request
        audio_path = request_data['audio_path']
        num_speakers = request_data.get('num_speakers')
        min_speakers = request_data.get('min_speakers')
        max_speakers = request_data.get('max_speakers')
        diarization_model = request_data.get('diarization_model')
        asr_model = request_data.get('asr_model')
        language = request_data.get('language')
        diarize = request_data.get('diarize', True)
        vad = request_data.get('vad')
        batch_size = request_data.get('batch_size')
        output_format = request_data.get('output_format')
        hf_token = request_data.get('hf_token')
        print(f"DEBUG: hf_token from request_data: {hf_token}")
        print(f"DEBUG: HF_TOKEN env var: {os.getenv('HF_TOKEN')}")

        # Load configuration
        config = get_config()

        # Set defaults from config
        if diarization_model is None:
            if config.diarization.backend == "hybrid":
                diarization_model = config.diarization.pyannote_model
            else:
                diarization_model = config.diarization.nvidia_model
        asr_model = asr_model or config.asr.asr_model_name
        language = language or config.asr.language
        vad = vad if vad is not None else config.asr.use_vad
        batch_size = batch_size or config.asr.batch_size
        output_format = output_format or config.processing.output_format

        # Validate num_speakers parameter
        if num_speakers is not None:
            if not (1 <= num_speakers <= 4):
                raise ValueError("num_speakers must be between 1 and 4 (Sortformer model limit)")
            logger.info("Expected number of speakers: %d", num_speakers)

        # Convert audio to 16kHz mono wav
        audio = AudioSegment.from_file(audio_path)
        audio = audio.set_channels(1).set_frame_rate(config.processing.sample_rate)
        converted_path = audio_path.replace(os.path.splitext(audio_path)[1], "_converted.wav")
        audio.export(converted_path, format=config.processing.audio_format)

        results = []

        # Load audio for segmentation
        waveform, sample_rate = torchaudio.load(converted_path)

        # Initialize ASR components
        asr_component = None
        asr_model_instance = None

        # Preload ASR models for faster inference
        if asr_model not in asr_models:
            if asr_model == "nvidia/parakeet-ctc-1.1b":
                asr_models[asr_model] = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(asr_model)
            elif asr_model == "nvidia/parakeet-tdt-1.1b":
                # Try the faster TDT model variant
                try:
                    asr_models[asr_model] = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained(asr_model)
                except Exception as e:
                    logger.warning("Failed to load %s, falling back to tdt_ctc: %s", asr_model, str(e))
                    asr_model = "nvidia/parakeet-tdt_ctc-1.1b"
                    asr_models[asr_model] = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained(asr_model)
            elif asr_model == "nvidia/parakeet-tdt_ctc-1.1b":
                asr_models[asr_model] = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained(asr_model)
            else:
                raise ValueError(f"Unsupported ASR model: {asr_model}")

        # VAD functionality commented out - not required for current use case
        # if vad:
        #     # Use NvidiaASR with VAD support
        #     asr_component = NvidiaASR(
        #         asr_model_name=config.asr.asr_model_name,
        #         device=config.asr.device,
        #         use_vad=config.asr.use_vad,
        #         vad_threshold=config.asr.vad_threshold,
        #         min_segment_duration=config.asr.min_segment_duration,
        #         batch_size=batch_size,
        #         enable_batch_processing=config.asr.enable_batch_processing
        #     )
        # else:
        # Use preloaded NeMo ASR model (VAD disabled for simplicity)
        asr_model_instance = asr_models[asr_model]

        if diarize:
            # Use Pyannote-based hybrid diarization (only backend supported)
            diarizer_key = f"hybrid_{min_speakers}_{max_speakers}"
            if diarizer_key not in diarizer_instances:
                diarizer_instances[diarizer_key] = HybridDiarization(
                    pyannote_model=config.diarization.pyannote_model,
                    hf_token=hf_token or config.diarization.hf_token or os.getenv("HF_TOKEN"),
                    device=config.asr.device,
                    min_speakers=min_speakers,
                    max_speakers=max_speakers
                )
            diarizer = diarizer_instances[diarizer_key]
            diarization_result = diarizer.diarize_audio(converted_path)
            speaker_segments = diarization_result['segments']

            # Apply speaker filtering if needed
            if num_speakers and len(set(s['speaker'] for s in speaker_segments)) > num_speakers:
                speaker_segments = diarizer.filter_speakers(speaker_segments, num_speakers)

            logger.debug("Parsed speaker segments type: %s", type(speaker_segments))
            logger.debug("Parsed speaker segments length: %s", len(speaker_segments) if hasattr(speaker_segments, '__len__') else 'N/A')
            if speaker_segments:
                logger.debug("First parsed segment: %s", speaker_segments[0])

            for segment in speaker_segments:
                start_time = segment['start']
                end_time = segment['end']
                speaker = segment['speaker']
                segment_duration = end_time - start_time

                # Skip segments that are too short for ASR
                min_segment_duration = 0.2
                if segment_duration < min_segment_duration:
                    logger.debug("Skipping %s segment (%.3fs) - too short for ASR", speaker, segment_duration)
                    continue

                start_sample = int(start_time * sample_rate)
                end_sample = int(end_time * sample_rate)
                segment_waveform = waveform[:, start_sample:end_sample]

                # Transcribe segment
                try:
                    # VAD functionality commented out - using direct NeMo ASR
                    # if vad and asr_component is not None:
                    #     transcription_result = asr_component.transcribe_segment(segment_waveform, sample_rate)
                    #     if transcription_result and transcription_result.strip():
                    #         results.append({
                    #             'text': transcription_result.strip(),
                    #             'start': start_time,
                    #             'end': end_time,
                    #             'speaker': speaker
                    #         })
                    # elif asr_model_instance is not None:
                    if asr_model_instance is not None:
                        # Save temporary segment for NeMo
                        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                        torchaudio.save(temp_file.name, segment_waveform, sample_rate)
                        transcription_output = asr_model_instance.transcribe([temp_file.name])
                        transcription = transcription_output[0].text.strip()
                        os.unlink(temp_file.name)
                        if transcription and transcription.strip():
                            results.append({
                                'text': transcription.strip(),
                                'start': start_time,
                                'end': end_time,
                                'speaker': speaker
                            })
                except Exception as e:
                    logger.error("Error transcribing segment: %s", str(e))
        else:
            # Just ASR without diarization
            # VAD functionality commented out - using direct NeMo ASR
            # if vad and asr_component is not None:
            #     asr_result = asr_component.transcribe_file(converted_path)
            #     if isinstance(asr_result, dict) and 'segments' in asr_result:
            #         for segment in asr_result['segments']:
            #             results.append({
            #                 'text': segment['text'],
            #                 'start': segment['start'],
            #                 'end': segment['end'],
            #                 'speaker': 'SPEAKER_00'
            #             })
            # elif asr_model_instance is not None:
            if asr_model_instance is not None:
                transcription_output = asr_model_instance.transcribe([converted_path], timestamps=True)
                if hasattr(transcription_output[0], 'timestamp') and 'segment' in transcription_output[0].timestamp:
                    for stamp in transcription_output[0].timestamp['segment']:
                        results.append({
                            'text': stamp['segment'],
                            'start': stamp['start'],
                            'end': stamp['end'],
                            'speaker': 'SPEAKER_00'
                        })
                else:
                    duration = audio.duration_seconds
                    results.append({
                        'text': transcription_output[0].text,
                        'start': 0.0,
                        'end': duration,
                        'speaker': 'SPEAKER_00'
                    })

        # Merge consecutive segments from the same speaker
        if diarize:
            results = merge_consecutive_speaker_segments(results)

        # Sort results by timestamp
        results_sorted = sorted(results, key=lambda x: x['start'])

        # Clean up converted file
        if os.path.exists(converted_path):
            os.unlink(converted_path)

        # Clean up components (don't delete cached models as they're reused)
        if asr_component is not None:
            asr_component.cleanup()
            del asr_component

        # Note: Don't delete cached diarizer instances as they're reused across calls
        # They will be cleaned up when the subprocess exits

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        return {"segments": results_sorted}

    except Exception as e:
        import traceback
        error_msg = f"Worker error: {str(e)}\n{traceback.format_exc()}"
        logger.error("Worker error: %s", error_msg)
        return {"error": str(e)}

if __name__ == "__main__":
    # Read request data from stdin
    request_data = json.loads(sys.stdin.read())
    result = run_inference(request_data)

    # Restore stdout temporarily for JSON output only
    sys.stdout = old_stdout
    null_device.close()

    # Write ONLY the JSON result to stdout (no other output)
    print(json.dumps(result))
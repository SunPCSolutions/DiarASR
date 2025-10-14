#!/usr/bin/env python3
"""
Batch Processor Module

This module handles batch processing utilities and file handling
for efficient transcription of multiple audio files.
"""

from typing import List, Dict, Tuple, Optional
import torch
from audio_preprocessor import AudioPreprocessor
from vad_processor import VADProcessor
from asr_model import ASRModel


class BatchProcessor:
    """
    Batch processing utilities for efficient audio transcription.

    Handles multiple files and segments with optimized processing.
    """

    def __init__(
        self,
        audio_preprocessor: AudioPreprocessor,
        vad_processor: Optional[VADProcessor],
        asr_model: ASRModel,
        enable_batch_processing: bool = True,
        use_vad: bool = False
    ):
        """
        Initialize the batch processor.

        Args:
            audio_preprocessor: Audio preprocessing handler
            vad_processor: VAD processor (optional)
            asr_model: ASR model handler
            enable_batch_processing: Whether to enable batch processing
            use_vad: Whether to use VAD for segmentation
        """
        self.audio_preprocessor = audio_preprocessor
        self.vad_processor = vad_processor
        self.asr_model = asr_model
        self.enable_batch_processing = enable_batch_processing
        self.use_vad = use_vad

    def transcribe_files_batch(self, audio_paths: List[str]) -> List[Dict]:
        """
        Transcribe multiple audio files efficiently.

        Args:
            audio_paths: List of paths to audio files

        Returns:
            List of transcription results with segments

        Raises:
            RuntimeError: If batch processing fails
        """
        results = []

        if self.enable_batch_processing and len(audio_paths) > 1:
            print(f"Batch transcribing {len(audio_paths)} files")

            # Preprocess all files
            processed_audios = []
            file_info = []

            for audio_path in audio_paths:
                try:
                    waveform, sample_rate = self.audio_preprocessor.preprocess_audio(audio_path)
                    duration = waveform.shape[1] / sample_rate

                    if self.use_vad and self.vad_processor:
                        # Run VAD on each file individually
                        speech_segments = self.vad_processor.run_vad(waveform, sample_rate)

                        for segment in speech_segments:
                            if segment['speech']:
                                start_sample = int(segment['start'] * sample_rate)
                                end_sample = int(segment['end'] * sample_rate)
                                segment_waveform = waveform[:, start_sample:end_sample]
                                processed_audios.append((segment_waveform, sample_rate))
                                file_info.append((audio_path, segment))
                    else:
                        processed_audios.append((waveform, sample_rate))
                        file_info.append((audio_path, {'start': 0.0, 'end': duration}))
                except Exception as e:
                    print(f"Failed to preprocess {audio_path}: {e}")
                    continue

            # Batch transcribe all segments
            if processed_audios:
                try:
                    transcriptions = self.asr_model.transcribe_batch(processed_audios)

                    # Reconstruct results by file
                    file_results = {}
                    trans_idx = 0

                    for audio_path, segment_info in file_info:
                        if audio_path not in file_results:
                            file_results[audio_path] = {
                                'file': audio_path,
                                'duration': segment_info['end'] - segment_info['start'],
                                'segments': []
                            }

                        if trans_idx < len(transcriptions):
                            transcription = transcriptions[trans_idx]
                            if transcription:  # Only include non-empty transcriptions
                                file_results[audio_path]['segments'].append({
                                    'start': segment_info['start'],
                                    'end': segment_info['end'],
                                    'text': transcription
                                })
                            trans_idx += 1

                    results = list(file_results.values())
                except Exception as e:
                    raise RuntimeError(f"Batch transcription failed: {e}")
        else:
            # Process files individually
            for audio_path in audio_paths:
                try:
                    result = self.transcribe_single_file(audio_path)
                    results.append(result)
                except Exception as e:
                    print(f"Failed to transcribe {audio_path}: {e}")
                    continue

        return results

    def transcribe_single_file(self, audio_path: str) -> Dict:
        """
        Transcribe a single audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            Transcription result with segments

        Raises:
            RuntimeError: If transcription fails
        """
        print(f"Transcribing {audio_path}")

        try:
            # Preprocess audio
            waveform, sample_rate = self.audio_preprocessor.preprocess_audio(audio_path)
            duration = waveform.shape[1] / sample_rate

            results = {
                'file': audio_path,
                'duration': duration,
                'segments': []
            }

            if self.use_vad and self.vad_processor:
                # Run VAD to find speech segments
                speech_segments = self.vad_processor.run_vad(waveform, sample_rate)

                # Transcribe each speech segment
                for segment in speech_segments:
                    if segment['speech']:
                        start_sample = int(segment['start'] * sample_rate)
                        end_sample = int(segment['end'] * sample_rate)
                        segment_waveform = waveform[:, start_sample:end_sample]

                        transcription = self.asr_model.transcribe_segment(segment_waveform, sample_rate)

                        if transcription:  # Only include non-empty transcriptions
                            results['segments'].append({
                                'start': segment['start'],
                                'end': segment['end'],
                                'text': transcription
                            })
            else:
                # Transcribe entire file
                transcription = self.asr_model.transcribe_segment(waveform, sample_rate)
                results['segments'].append({
                    'start': 0.0,
                    'end': duration,
                    'text': transcription
                })

            return results

        except Exception as e:
            raise RuntimeError(f"Failed to transcribe file {audio_path}: {e}")

    def transcribe_segments_batch(self, audio_segments: List[Tuple[torch.Tensor, int]]) -> List[str]:
        """
        Transcribe multiple audio segments in batch.

        Args:
            audio_segments: List of (waveform, sample_rate) tuples

        Returns:
            List of transcribed texts
        """
        if not self.enable_batch_processing or len(audio_segments) == 1:
            # Fall back to individual processing
            return [self.asr_model.transcribe_segment(waveform, sample_rate)
                    for waveform, sample_rate in audio_segments]

        return self.asr_model.transcribe_batch(audio_segments)
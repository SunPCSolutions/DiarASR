#!/usr/bin/env python3
"""
NVIDIA ASR Module with VAD support

This module provides a modular ASR component that uses
EncDecCTCModelBPE with nvidia/parakeet-ctc-1.1b and integrated VAD functionality.
"""

import os
import torch
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.models import EncDecClassificationModel
from pydub import AudioSegment
import torchaudio
import tempfile
import json
from pathlib import Path
from config import get_config


class NvidiaASR:
    """
    Modular NVIDIA ASR component using EncDecCTCModelBPE with VAD support.

    Features:
    - EncDecCTCModelBPE with nvidia/parakeet-ctc-1.1b
    - Integrated Voice Activity Detection (VAD)
    - Batch processing capabilities
    - Configurable parameters for different use cases
    """

    def __init__(
        self,
        asr_model_name: Optional[str] = None,
        vad_model_name: Optional[str] = None,
        device: Optional[str] = None,
        sample_rate: Optional[int] = None,
        use_vad: Optional[bool] = None,
        vad_threshold: Optional[float] = None,
        min_segment_duration: Optional[float] = None,
        batch_size: Optional[int] = None,
        enable_batch_processing: Optional[bool] = None
    ):
        """
        Initialize the NVIDIA ASR component.

        Args:
            asr_model_name: Name of the pretrained ASR model (uses config default if None)
            vad_model_name: Name of the pretrained VAD model (uses config default if None)
            device: Device to run on ('auto', 'cpu', 'cuda') (uses config default if None)
            sample_rate: Audio sample rate (16kHz expected by NeMo) (uses config default if None)
            use_vad: Whether to use VAD for speech detection (uses config default if None)
            vad_threshold: Threshold for VAD speech detection (0.0-1.0) (uses config default if None)
            min_segment_duration: Minimum duration for speech segments (seconds) (uses config default if None)
            batch_size: Batch size for processing (uses config default if None)
            enable_batch_processing: Whether to enable batch processing (uses config default if None)
        """
        # Load configuration
        config = get_config()

        # Set defaults from config
        self.asr_model_name = asr_model_name or config.asr.asr_model_name
        self.vad_model_name = vad_model_name or "silero_vad"  # Use Silero VAD
        self.sample_rate = sample_rate or config.processing.sample_rate
        self.use_vad = use_vad if use_vad is not None else config.asr.use_vad
        self.vad_threshold = vad_threshold if vad_threshold is not None else config.asr.vad_threshold
        self.min_segment_duration = min_segment_duration if min_segment_duration is not None else config.asr.min_segment_duration
        self.batch_size = batch_size or config.asr.batch_size
        self.enable_batch_processing = enable_batch_processing if enable_batch_processing is not None else config.asr.enable_batch_processing

        # Set device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Initialize models
        self.asr_model = None
        self.vad_model = None

        print(f"Initializing NVIDIA ASR with model: {asr_model_name}")
        print(f"Device: {self.device}")
        print(f"VAD enabled: {use_vad}")
        if use_vad:
            print(f"VAD model: {vad_model_name}")
            print(f"VAD threshold: {vad_threshold}")
        print(f"Batch processing: {enable_batch_processing} (batch_size={batch_size})")

    def load_models(self):
        """Load ASR and VAD models."""
        if self.asr_model is None:
            print(f"Loading ASR model {self.asr_model_name}...")
            self.asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(self.asr_model_name)
            self.asr_model = self.asr_model.to(self.device)
            self.asr_model.eval()
            print("ASR model loaded successfully.")

        if self.use_vad and self.vad_model is None:
            print(f"Loading Silero VAD model...")
            try:
                # Load Silero VAD model and utils from torch hub
                model_and_utils = torch.hub.load(
                    repo_or_dir='snakers4/silero-vad',
                    model='silero_vad',
                    force_reload=False  # Set to True for first run
                )

                # Handle different return formats
                if isinstance(model_and_utils, tuple) and len(model_and_utils) == 2:
                    self.vad_model, vad_utils = model_and_utils
                    # Extract the functions we need
                    (self.get_speech_timestamps, _, self.read_audio, *_) = vad_utils
                else:
                    # Fallback: assume it's just the model
                    self.vad_model = model_and_utils
                    # Load utils separately
                    utils_module = torch.hub.load(
                        repo_or_dir='snakers4/silero-vad',
                        model='silero_vad',
                        source='github',
                        force_reload=False
                    )
                    if hasattr(utils_module, '__len__') and len(utils_module) > 1:
                        _, vad_utils = utils_module
                        (self.get_speech_timestamps, _, self.read_audio, *_) = vad_utils

                self.vad_model = self.vad_model.to(self.device)
                self.vad_model.eval()
                print("Silero VAD model loaded successfully.")
            except Exception as e:
                print(f"Failed to load Silero VAD: {e}")
                print("Falling back to no VAD")
                self.use_vad = False

    def preprocess_audio(self, audio_path: str) -> Tuple[torch.Tensor, int]:
        """
        Preprocess audio file for ASR/VAD.

        Args:
            audio_path: Path to audio file

        Returns:
            Tuple of (audio_tensor, sample_rate)
        """
        # Convert to WAV if needed
        if not audio_path.endswith('.wav'):
            temp_wav = audio_path.rsplit('.', 1)[0] + '_processed.wav'
            audio = AudioSegment.from_file(audio_path)
            audio = audio.set_channels(1).set_frame_rate(self.sample_rate)
            audio.export(temp_wav, format="wav")
            audio_path = temp_wav
        else:
            temp_wav = None

        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path)

        # Ensure mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Ensure correct sample rate
        if sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.sample_rate)
            waveform = resampler(waveform)

        # Clean up temp file
        if temp_wav and os.path.exists(temp_wav):
            os.remove(temp_wav)

        return waveform, self.sample_rate

    def run_vad(self, waveform: torch.Tensor, sample_rate: int) -> List[Dict]:
        """
        Run Voice Activity Detection on audio using Silero VAD.

        Args:
            waveform: Audio tensor (1, samples)
            sample_rate: Sample rate

        Returns:
            List of speech segments with start/end times
        """
        if not self.use_vad or self.vad_model is None:
            # Return full audio as single segment
            duration = waveform.shape[1] / sample_rate
            return [{'start': 0.0, 'end': duration, 'speech': True}]

        print("Running Silero VAD...")

        try:
            # Convert to numpy for Silero VAD
            audio_numpy = waveform.squeeze(0).cpu().numpy()

            # Get speech timestamps using Silero VAD
            speech_timestamps = self.get_speech_timestamps(
                audio_numpy,
                self.vad_model,
                sampling_rate=sample_rate,
                threshold=self.vad_threshold
            )

            # Convert to our format
            segments = []
            for timestamp in speech_timestamps:
                start_time = timestamp['start'] / sample_rate  # Convert from samples to seconds
                end_time = timestamp['end'] / sample_rate
                duration = end_time - start_time

                if duration >= self.min_segment_duration:
                    segments.append({
                        'start': start_time,
                        'end': end_time,
                        'speech': True
                    })

            print(f"Silero VAD detected {len(segments)} speech segments")
            return segments

        except Exception as e:
            print(f"Silero VAD failed: {e}, falling back to no VAD")
            # Fallback to full audio
            duration = waveform.shape[1] / sample_rate
            return [{'start': 0.0, 'end': duration, 'speech': True}]

    def _frame_probs_to_segments(self, frame_probs: np.ndarray, sample_rate: int) -> List[Dict]:
        """
        Convert frame-level probabilities to speech segments.

        Args:
            frame_probs: Frame-level speech probabilities
            sample_rate: Sample rate

        Returns:
            List of segments with start/end times and speech flags
        """
        # VAD frame rate is typically 100Hz (10ms frames) for MarbleNet
        frame_duration = 0.01  # 10ms frames
        segments = []

        # Simple thresholding to find speech segments
        speech_frames = frame_probs > self.vad_threshold

        # Find contiguous speech regions
        if len(speech_frames) > 0:
            # Find transitions
            diff = np.diff(speech_frames.astype(int))
            start_indices = np.where(diff == 1)[0] + 1
            end_indices = np.where(diff == -1)[0] + 1

            # Handle edge cases
            if speech_frames[0]:
                start_indices = np.concatenate([[0], start_indices])
            if speech_frames[-1]:
                end_indices = np.concatenate([end_indices, [len(speech_frames)]])

            # Create segments
            for start_idx, end_idx in zip(start_indices, end_indices):
                start_time = start_idx * frame_duration
                end_time = end_idx * frame_duration
                duration = end_time - start_time

                if duration >= self.min_segment_duration:
                    segments.append({
                        'start': start_time,
                        'end': end_time,
                        'speech': True
                    })

        return segments

    def transcribe_segment(self, waveform: torch.Tensor, sample_rate: int) -> str:
        """
        Transcribe a single audio segment.

        Args:
            waveform: Audio tensor (1, samples)
            sample_rate: Sample rate

        Returns:
            Transcribed text
        """
        self.load_models()

        # Save temporary WAV for NeMo
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name
            torchaudio.save(temp_path, waveform, sample_rate)

        try:
            # Transcribe using NeMo ASR
            transcription_output = self.asr_model.transcribe([temp_path])
            transcription = transcription_output[0].text.strip()
            return transcription
        finally:
            os.unlink(temp_path)

    def transcribe_batch(self, audio_segments: List[Tuple[torch.Tensor, int]]) -> List[str]:
        """
        Transcribe multiple audio segments in batch.

        Args:
            audio_segments: List of (waveform, sample_rate) tuples

        Returns:
            List of transcribed texts
        """
        if not self.enable_batch_processing or len(audio_segments) == 1:
            # Fall back to individual processing
            return [self.transcribe_segment(waveform, sample_rate)
                   for waveform, sample_rate in audio_segments]

        self.load_models()

        # Save all segments to temp files
        temp_paths = []
        try:
            for waveform, sample_rate in audio_segments:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                    temp_path = temp_file.name
                    torchaudio.save(temp_path, waveform, sample_rate)
                    temp_paths.append(temp_path)

            # Batch transcribe
            transcription_outputs = self.asr_model.transcribe(temp_paths)
            transcriptions = [output.text.strip() for output in transcription_outputs]

            return transcriptions
        finally:
            for path in temp_paths:
                if os.path.exists(path):
                    os.unlink(path)

    def transcribe_file(self, audio_path: str) -> Dict:
        """
        Transcribe an audio file with optional VAD preprocessing.

        Args:
            audio_path: Path to audio file

        Returns:
            Dict with transcription results
        """
        print(f"Transcribing {audio_path}")

        # Preprocess audio
        waveform, sample_rate = self.preprocess_audio(audio_path)
        duration = waveform.shape[1] / sample_rate

        results = {
            'file': audio_path,
            'duration': duration,
            'segments': []
        }

        if self.use_vad:
            # Run VAD to find speech segments
            speech_segments = self.run_vad(waveform, sample_rate)

            # Transcribe each speech segment
            for segment in speech_segments:
                if segment['speech']:
                    start_sample = int(segment['start'] * sample_rate)
                    end_sample = int(segment['end'] * sample_rate)
                    segment_waveform = waveform[:, start_sample:end_sample]

                    transcription = self.transcribe_segment(segment_waveform, sample_rate)

                    if transcription:  # Only include non-empty transcriptions
                        results['segments'].append({
                            'start': segment['start'],
                            'end': segment['end'],
                            'text': transcription
                        })
        else:
            # Transcribe entire file
            transcription = self.transcribe_segment(waveform, sample_rate)
            results['segments'].append({
                'start': 0.0,
                'end': duration,
                'text': transcription
            })

        return results

    def transcribe_files_batch(self, audio_paths: List[str]) -> List[Dict]:
        """
        Transcribe multiple audio files.

        Args:
            audio_paths: List of paths to audio files

        Returns:
            List of transcription results
        """
        results = []

        if self.enable_batch_processing and len(audio_paths) > 1:
            print(f"Batch transcribing {len(audio_paths)} files")

            # Preprocess all files
            processed_audios = []
            file_info = []

            for audio_path in audio_paths:
                waveform, sample_rate = self.preprocess_audio(audio_path)
                duration = waveform.shape[1] / sample_rate

                if self.use_vad:
                    # Run VAD on each file individually (VAD doesn't batch well)
                    speech_segments = self.run_vad(waveform, sample_rate)

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

            # Batch transcribe all segments
            if processed_audios:
                transcriptions = self.transcribe_batch(processed_audios)

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
                        if transcription:
                            file_results[audio_path]['segments'].append({
                                'start': segment_info['start'],
                                'end': segment_info['end'],
                                'text': transcription
                            })
                        trans_idx += 1

                results = list(file_results.values())
        else:
            # Process files individually
            for audio_path in audio_paths:
                result = self.transcribe_file(audio_path)
                results.append(result)

        return results

    def cleanup(self):
        """Clean up resources."""
        if self.asr_model is not None:
            del self.asr_model
            self.asr_model = None

        if self.vad_model is not None:
            del self.vad_model
            self.vad_model = None

        # Clear Silero VAD utils
        if hasattr(self, 'get_speech_timestamps'):
            delattr(self, 'get_speech_timestamps')
        if hasattr(self, 'read_audio'):
            delattr(self, 'read_audio')
        if hasattr(self, 'vad_utils'):
            delattr(self, 'vad_utils')

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("NVIDIA ASR cleanup complete.")


# Convenience functions
def create_nvidia_asr(**kwargs) -> NvidiaASR:
    """
    Create a configured NVIDIA ASR instance.

    Args:
        **kwargs: Parameters for NvidiaASR constructor

    Returns:
        Configured NvidiaASR instance
    """
    return NvidiaASR(**kwargs)


def transcribe_audio_file(audio_path: str, **kwargs) -> Dict:
    """
    Convenience function to transcribe a single audio file.

    Args:
        audio_path: Path to audio file
        **kwargs: Parameters for NvidiaASR

    Returns:
        Transcription results
    """
    asr = create_nvidia_asr(**kwargs)
    try:
        result = asr.transcribe_file(audio_path)
        return result
    finally:
        asr.cleanup()


def transcribe_audio_files(audio_paths: List[str], **kwargs) -> List[Dict]:
    """
    Convenience function to transcribe multiple audio files.

    Args:
        audio_paths: List of paths to audio files
        **kwargs: Parameters for NvidiaASR

    Returns:
        List of transcription results
    """
    asr = create_nvidia_asr(**kwargs)
    try:
        results = asr.transcribe_files_batch(audio_paths)
        return results
    finally:
        asr.cleanup()
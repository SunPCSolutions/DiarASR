#!/usr/bin/env python3
"""
Voice Activity Detection Processor Module

This module handles voice activity detection using Silero VAD
for speech segmentation in audio files.
"""

import torch
import numpy as np
from typing import List, Dict, Optional
from config import get_config


class VADProcessor:
    """
    Voice Activity Detection processor using Silero VAD.

    Detects speech segments in audio for more efficient transcription.
    """

    def __init__(
        self,
        vad_model_name: str = "silero_vad",
        vad_threshold: Optional[float] = None,
        min_segment_duration: Optional[float] = None,
        device: Optional[str] = None
    ):
        """
        Initialize the VAD processor.

        Args:
            vad_model_name: Name of the VAD model to use
            vad_threshold: Threshold for speech detection (0.0-1.0)
            min_segment_duration: Minimum duration for speech segments (seconds)
            device: Device to run VAD on ('cpu', 'cuda', etc.)
        """
        config = get_config()

        self.vad_model_name = vad_model_name
        self.vad_threshold = vad_threshold if vad_threshold is not None else config.asr.vad_threshold
        self.min_segment_duration = min_segment_duration if min_segment_duration is not None else config.asr.min_segment_duration

        # Set device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize model and utils
        self.vad_model = None
        self.get_speech_timestamps = None
        self.read_audio = None

    def load_model(self):
        """
        Load the Silero VAD model and utilities.

        Raises:
            RuntimeError: If model loading fails
        """
        if self.vad_model is not None:
            return

        print("Loading Silero VAD model...")
        try:
            # Load Silero VAD model and utils from torch hub
            model_and_utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False
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
            raise RuntimeError(f"Failed to load Silero VAD model: {e}")

    def run_vad(self, waveform: torch.Tensor, sample_rate: int) -> List[Dict]:
        """
        Run Voice Activity Detection on audio using Silero VAD.

        Args:
            waveform: Audio tensor (1, samples)
            sample_rate: Sample rate of the audio

        Returns:
            List of speech segments with start/end times

        Raises:
            RuntimeError: If VAD processing fails
        """
        if self.vad_model is None:
            self.load_model()

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
            raise RuntimeError(f"Silero VAD processing failed: {e}")

    def get_full_audio_segment(self, waveform: torch.Tensor, sample_rate: int) -> List[Dict]:
        """
        Return the full audio as a single speech segment (no VAD).

        Args:
            waveform: Audio tensor (1, samples)
            sample_rate: Sample rate of the audio

        Returns:
            List with single segment covering full audio
        """
        duration = waveform.shape[1] / sample_rate
        return [{'start': 0.0, 'end': duration, 'speech': True}]

    def cleanup(self):
        """Clean up VAD model and free resources."""
        if self.vad_model is not None:
            del self.vad_model
            self.vad_model = None

        # Clear utils
        if hasattr(self, 'get_speech_timestamps'):
            delattr(self, 'get_speech_timestamps')
        if hasattr(self, 'read_audio'):
            delattr(self, 'read_audio')
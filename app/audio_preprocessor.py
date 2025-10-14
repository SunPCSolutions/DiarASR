#!/usr/bin/env python3
"""
Audio Preprocessor Module

This module handles audio format conversion, validation, and preprocessing
for ASR and VAD components.
"""

import os
import torch
from typing import Tuple, Optional
from pydub import AudioSegment
import torchaudio
from config import get_config


class AudioPreprocessor:
    """
    Audio preprocessing utilities for ASR and VAD.

    Handles format conversion, validation, and audio normalization.
    """

    def __init__(self, sample_rate: Optional[int] = None):
        """
        Initialize the audio preprocessor.

        Args:
            sample_rate: Target sample rate (16kHz expected by NeMo)
        """
        config = get_config()
        self.sample_rate = sample_rate or config.processing.sample_rate

    def preprocess_audio(self, audio_path: str) -> Tuple[torch.Tensor, int]:
        """
        Preprocess audio file for ASR/VAD.

        Converts to WAV format if needed, ensures mono channel,
        and resamples to target sample rate.

        Args:
            audio_path: Path to audio file

        Returns:
            Tuple of (audio_tensor, sample_rate)

        Raises:
            FileNotFoundError: If audio file doesn't exist
            ValueError: If audio format is unsupported
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        try:
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

        except Exception as e:
            raise ValueError(f"Failed to preprocess audio file {audio_path}: {e}")

    def validate_audio_file(self, audio_path: str) -> bool:
        """
        Validate that an audio file can be processed.

        Args:
            audio_path: Path to audio file

        Returns:
            True if file is valid and supported
        """
        if not os.path.exists(audio_path):
            return False

        supported_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.aac']
        _, ext = os.path.splitext(audio_path.lower())

        if ext not in supported_extensions:
            return False

        try:
            # Try to load with pydub for validation
            AudioSegment.from_file(audio_path)
            return True
        except Exception:
            return False
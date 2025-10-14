#!/usr/bin/env python3
"""
ASR Model Module

This module handles core ASR model loading, inference, and basic transcription
using NVIDIA NeMo EncDecCTCModelBPE.
"""

import os
import torch
import tempfile
from typing import Optional
import nemo.collections.asr as nemo_asr
from config import get_config


class ASRModel:
    """
    Core ASR model handler using NVIDIA NeMo EncDecCTCModelBPE.

    Manages model loading, inference, and basic transcription operations.
    """

    def __init__(
        self,
        asr_model_name: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize the ASR model handler.

        Args:
            asr_model_name: Name of the pretrained ASR model
            device: Device to run model on ('auto', 'cpu', 'cuda')
        """
        config = get_config()

        self.asr_model_name = asr_model_name or config.asr.asr_model_name

        # Set device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize model
        self.asr_model = None

    def load_model(self):
        """
        Load the ASR model.

        Raises:
            RuntimeError: If model loading fails
        """
        if self.asr_model is not None:
            return

        print(f"Loading ASR model {self.asr_model_name}...")
        try:
            self.asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(self.asr_model_name)
            self.asr_model = self.asr_model.to(self.device)
            self.asr_model.eval()
            print("ASR model loaded successfully.")
        except Exception as e:
            raise RuntimeError(f"Failed to load ASR model {self.asr_model_name}: {e}")

    def transcribe_segment(self, waveform: torch.Tensor, sample_rate: int) -> str:
        """
        Transcribe a single audio segment.

        Args:
            waveform: Audio tensor (1, samples)
            sample_rate: Sample rate of the audio

        Returns:
            Transcribed text

        Raises:
            RuntimeError: If transcription fails
        """
        if self.asr_model is None:
            self.load_model()

        # Save temporary WAV for NeMo
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name
            import torchaudio
            torchaudio.save(temp_path, waveform, sample_rate)

        try:
            # Transcribe using NeMo ASR
            transcription_output = self.asr_model.transcribe([temp_path])
            transcription = transcription_output[0].text.strip()
            return transcription
        except Exception as e:
            raise RuntimeError(f"ASR transcription failed: {e}")
        finally:
            os.unlink(temp_path)

    def transcribe_batch(self, audio_segments: list) -> list:
        """
        Transcribe multiple audio segments in batch.

        Args:
            audio_segments: List of (waveform, sample_rate) tuples

        Returns:
            List of transcribed texts

        Raises:
            RuntimeError: If batch transcription fails
        """
        if self.asr_model is None:
            self.load_model()

        if not audio_segments:
            return []

        # Save all segments to temp files
        temp_paths = []
        try:
            import torchaudio
            for waveform, sample_rate in audio_segments:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                    temp_path = temp_file.name
                    torchaudio.save(temp_path, waveform, sample_rate)
                    temp_paths.append(temp_path)

            # Batch transcribe
            transcription_outputs = self.asr_model.transcribe(temp_paths)
            transcriptions = [output.text.strip() for output in transcription_outputs]

            return transcriptions

        except Exception as e:
            raise RuntimeError(f"Batch ASR transcription failed: {e}")
        finally:
            for path in temp_paths:
                if os.path.exists(path):
                    os.unlink(path)

    def cleanup(self):
        """Clean up ASR model and free resources."""
        if self.asr_model is not None:
            del self.asr_model
            self.asr_model = None

        # Force garbage collection
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
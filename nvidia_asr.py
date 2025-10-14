#!/usr/bin/env python3
"""
NVIDIA ASR Module with VAD support

This module provides a modular ASR component that uses
EncDecCTCModelBPE with nvidia/parakeet-ctc-1.1b and integrated VAD functionality.
"""

import os
import torch
from typing import List, Dict, Optional, Tuple
from config import get_config
from audio_preprocessor import AudioPreprocessor
from vad_processor import VADProcessor
from asr_model import ASRModel
from batch_processor import BatchProcessor


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

        # Initialize component modules
        self.audio_preprocessor = AudioPreprocessor(sample_rate=self.sample_rate)
        self.vad_processor = VADProcessor(
            vad_model_name=self.vad_model_name,
            vad_threshold=self.vad_threshold,
            min_segment_duration=self.min_segment_duration,
            device=self.device
        ) if self.use_vad else None
        self.asr_model = ASRModel(
            asr_model_name=self.asr_model_name,
            device=self.device
        )
        self.batch_processor = BatchProcessor(
            audio_preprocessor=self.audio_preprocessor,
            vad_processor=self.vad_processor,
            asr_model=self.asr_model,
            enable_batch_processing=self.enable_batch_processing,
            use_vad=self.use_vad
        )

        print(f"Initializing NVIDIA ASR with model: {asr_model_name}")
        print(f"Device: {self.device}")
        print(f"VAD enabled: {use_vad}")
        if use_vad:
            print(f"VAD model: {vad_model_name}")
            print(f"VAD threshold: {vad_threshold}")
        print(f"Batch processing: {enable_batch_processing} (batch_size={batch_size})")

    def load_models(self):
        """Load ASR and VAD models through component modules."""
        self.asr_model.load_model()
        if self.vad_processor:
            self.vad_processor.load_model()

    def preprocess_audio(self, audio_path: str) -> Tuple[torch.Tensor, int]:
        """
        Preprocess audio file for ASR/VAD.

        Args:
            audio_path: Path to audio file

        Returns:
            Tuple of (audio_tensor, sample_rate)
        """
        return self.audio_preprocessor.preprocess_audio(audio_path)

    def run_vad(self, waveform: torch.Tensor, sample_rate: int) -> List[Dict]:
        """
        Run Voice Activity Detection on audio using Silero VAD.

        Args:
            waveform: Audio tensor (1, samples)
            sample_rate: Sample rate

        Returns:
            List of speech segments with start/end times
        """
        if not self.use_vad or self.vad_processor is None:
            # Return full audio as single segment
            duration = waveform.shape[1] / sample_rate
            return [{'start': 0.0, 'end': duration, 'speech': True}]

        try:
            return self.vad_processor.run_vad(waveform, sample_rate)
        except Exception as e:
            print(f"VAD processing failed: {e}, falling back to no VAD")
            # Fallback to full audio
            duration = waveform.shape[1] / sample_rate
            return [{'start': 0.0, 'end': duration, 'speech': True}]

    def transcribe_segment(self, waveform: torch.Tensor, sample_rate: int) -> str:
        """
        Transcribe a single audio segment.

        Args:
            waveform: Audio tensor (1, samples)
            sample_rate: Sample rate

        Returns:
            Transcribed text
        """
        return self.asr_model.transcribe_segment(waveform, sample_rate)

    def transcribe_batch(self, audio_segments: List[Tuple[torch.Tensor, int]]) -> List[str]:
        """
        Transcribe multiple audio segments in batch.

        Args:
            audio_segments: List of (waveform, sample_rate) tuples

        Returns:
            List of transcribed texts
        """
        return self.batch_processor.transcribe_segments_batch(audio_segments)

    def transcribe_file(self, audio_path: str) -> Dict:
        """
        Transcribe an audio file with optional VAD preprocessing.

        Args:
            audio_path: Path to audio file

        Returns:
            Dict with transcription results
        """
        return self.batch_processor.transcribe_single_file(audio_path)

    def transcribe_files_batch(self, audio_paths: List[str]) -> List[Dict]:
        """
        Transcribe multiple audio files.

        Args:
            audio_paths: List of paths to audio files

        Returns:
            List of transcription results
        """
        return self.batch_processor.transcribe_files_batch(audio_paths)

    def cleanup(self):
        """Aggressively clean up resources and free GPU memory."""
        # Clean up component modules
        if self.asr_model:
            self.asr_model.cleanup()
        if self.vad_processor:
            self.vad_processor.cleanup()

        # Aggressive GPU memory cleanup
        if torch.cuda.is_available():
            # Multiple empty_cache calls
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            # Force garbage collection
            import gc
            gc.collect()

            # Try to trigger memory deallocation
            try:
                dummy = torch.zeros(1024, device='cuda')
                del dummy
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            except:
                pass

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
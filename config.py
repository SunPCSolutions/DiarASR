#!/usr/bin/env python3
"""
Centralized Parameter Configuration System

This module provides a centralized configuration system for ASR and diarization parameters.
All parameters can be easily modified in this single location with sensible defaults.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class ASRConfig:
    """Configuration for Automatic Speech Recognition parameters."""

    # Core ASR parameters
    batch_size: int = 16  # Batch size for processing audio segments
    compute_type: str = "fp16"  # Compute precision: "fp16", "fp32", "int8"
    language: str = "en"  # Language code for ASR model

    # VAD (Voice Activity Detection) parameters
    use_vad: bool = True  # Enable voice activity detection
    vad_threshold: float = 0.5  # Threshold for speech detection (0.0-1.0)
    min_segment_duration: float = 0.05  # Minimum duration for speech segments (seconds)

    # Model parameters
    asr_model_name: str = "nvidia/parakeet-ctc-1.1b"  # ASR model to use
    device: str = "auto"  # Device: "auto", "cpu", "cuda"

    # Batch processing
    enable_batch_processing: bool = True  # Enable batch processing for multiple segments


@dataclass
class DiarizationConfig:
    """Configuration for Speaker Diarization parameters."""

    # Model parameters
    model_name: str = "nvidia/diar_streaming_sortformer_4spk-v2"
    device: str = "auto"  # Device: "auto", "cpu", "cuda"
    num_speakers: Optional[int] = None  # Expected number of speakers (1-4)

    # Streaming parameters
    chunk_size: int = 6  # Chunk size for streaming processing
    right_context: int = 7  # Right context for streaming
    fifo_size: int = 188  # FIFO buffer size
    update_period: int = 144  # Update period for streaming
    speaker_cache_size: int = 188  # Speaker cache size


@dataclass
class StreamingConfig:
    """Configuration for streaming processing parameters."""

    # Streaming settings
    enable_streaming: bool = False  # Enable streaming mode
    stream_chunk_size: float = 1.0  # Chunk size in seconds for streaming
    stream_overlap: float = 0.5  # Overlap between chunks in seconds
    real_time_factor: float = 1.0  # Real-time processing factor

    # Buffer settings
    buffer_size: int = 1024  # Audio buffer size
    max_latency: float = 0.5  # Maximum allowed latency in seconds


@dataclass
class ProcessingConfig:
    """Configuration for general processing parameters."""

    # Audio processing
    sample_rate: int = 16000  # Target sample rate for processing
    audio_format: str = "wav"  # Preferred audio format

    # File handling
    max_file_size_mb: int = 100  # Maximum file size in MB
    allowed_extensions: List[str] = field(default_factory=lambda: ['.mp3', '.wav', '.flac', '.m4a', '.aac'])

    # Output settings
    output_format: str = "json"  # Output format: "json", "txt", or "both"

    # Security and cleanup
    secure_temp_dir: bool = True  # Use secure temporary directories
    auto_cleanup: bool = True  # Automatically clean up temporary files


@dataclass
class GlobalConfig:
    """Global configuration that combines all parameter groups."""

    # Main configuration sections
    asr: ASRConfig = field(default_factory=ASRConfig)
    diarization: DiarizationConfig = field(default_factory=DiarizationConfig)
    streaming: StreamingConfig = field(default_factory=StreamingConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'asr': self.asr.__dict__,
            'diarization': self.diarization.__dict__,
            'streaming': self.streaming.__dict__,
            'processing': self.processing.__dict__
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'GlobalConfig':
        """Create configuration from dictionary."""
        return cls(
            asr=ASRConfig(**config_dict.get('asr', {})),
            diarization=DiarizationConfig(**config_dict.get('diarization', {})),
            streaming=StreamingConfig(**config_dict.get('streaming', {})),
            processing=ProcessingConfig(**config_dict.get('processing', {}))
        )


# Default global configuration instance
# Modify these values here to change defaults across the entire system
DEFAULT_CONFIG = GlobalConfig(
    asr=ASRConfig(
        batch_size=32,  # Increased from 16 for better GPU utilization
        compute_type="fp16",  # As requested
        language="en",  # As requested
        use_vad=True,
        vad_threshold=0.5,
        min_segment_duration=0.05,
        asr_model_name="nvidia/parakeet-ctc-1.1b",
        device="auto",
        enable_batch_processing=True
    ),
    diarization=DiarizationConfig(
        model_name="nvidia/diar_streaming_sortformer_4spk-v2",
        device="auto",
        chunk_size=6,  # High accuracy setting
        right_context=7,  # High accuracy setting
        fifo_size=188,  # High accuracy setting
        update_period=144,  # High accuracy setting
        speaker_cache_size=188
    ),
    streaming=StreamingConfig(
        enable_streaming=False,  # As requested - streaming config
        stream_chunk_size=1.0,
        stream_overlap=0.5,
        real_time_factor=1.0,
        buffer_size=1024,
        max_latency=0.5
    ),
    processing=ProcessingConfig(
        sample_rate=16000,
        audio_format="wav",
        max_file_size_mb=100,
        allowed_extensions=['.mp3', '.wav', '.flac', '.m4a', '.aac'],
        output_format="json",
        secure_temp_dir=True,
        auto_cleanup=True
    )
)


def get_config() -> GlobalConfig:
    """
    Get the default configuration instance.

    Returns:
        GlobalConfig: The default configuration
    """
    return DEFAULT_CONFIG


def create_custom_config(**kwargs) -> GlobalConfig:
    """
    Create a custom configuration by overriding default values.

    Args:
        **kwargs: Configuration overrides

    Returns:
        GlobalConfig: Custom configuration instance
    """
    # Start with default config
    config = get_config()

    # Override ASR parameters
    if 'batch_size' in kwargs:
        config.asr.batch_size = kwargs['batch_size']
    if 'compute_type' in kwargs:
        config.asr.compute_type = kwargs['compute_type']
    if 'language' in kwargs:
        config.asr.language = kwargs['language']

    # Override other parameters as needed
    for key, value in kwargs.items():
        if hasattr(config.asr, key):
            setattr(config.asr, key, value)
        elif hasattr(config.diarization, key):
            setattr(config.diarization, key, value)
        elif hasattr(config.streaming, key):
            setattr(config.streaming, key, value)
        elif hasattr(config.processing, key):
            setattr(config.processing, key, value)

    return config


# Convenience functions for backward compatibility
def get_batch_size() -> int:
    """Get the default batch size."""
    return DEFAULT_CONFIG.asr.batch_size


def get_compute_type() -> str:
    """Get the default compute type."""
    return DEFAULT_CONFIG.asr.compute_type


def get_language() -> str:
    """Get the default language."""
    return DEFAULT_CONFIG.asr.language


def get_streaming_config() -> StreamingConfig:
    """Get the streaming configuration."""
    return DEFAULT_CONFIG.streaming


# Performance Presets (Accuracy-Preserving Only)
def create_performance_config(preset: str = "balanced") -> GlobalConfig:
    """
    Create a configuration optimized for performance without accuracy loss.

    Args:
        preset: Performance preset ("fast", "balanced", "accurate")

    Returns:
        GlobalConfig: Optimized configuration
    """
    config = get_config()

    if preset == "fast":
        # Optimized for speed - higher throughput, same accuracy
        config.asr.batch_size = 64  # Larger batch for better GPU utilization
        config.asr.use_vad = False  # Skip VAD for speed (no accuracy impact)

    elif preset == "balanced":
        # Current optimized settings - good balance
        config.asr.batch_size = 32  # Larger batch for better GPU utilization
        config.asr.use_vad = True   # Keep VAD for quality

    elif preset == "accurate":
        # Optimized for maximum accuracy
        config.asr.batch_size = 16  # Smaller batch for potentially better quality
        config.asr.use_vad = True   # Keep VAD for quality
        config.asr.vad_threshold = 0.4  # More sensitive VAD

    # Diarization parameters remain at high accuracy settings for all presets
    return config
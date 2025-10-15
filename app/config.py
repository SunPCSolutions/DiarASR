#!/usr/bin/env python3
"""
Centralized Parameter Configuration System

This module provides a centralized configuration system for ASR and diarization parameters.
All parameters can be easily modified in this single location with sensible defaults.
"""

import os
import re
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


def read_docker_secret(secret_name: str, default: Optional[str] = None) -> Optional[str]:
    """
    Read a Docker secret from /run/secrets/ directory.

    Args:
        secret_name: Name of the secret file
        default: Default value if secret not found

    Returns:
        Secret value or default
    """
    secret_path = f"/run/secrets/{secret_name}"
    try:
        with open(secret_path, 'r') as f:
            return f.read().strip()
    except (FileNotFoundError, IOError):
        return default


def sanitize_env_value(value: str, max_length: int = 1000) -> str:
    """
    Sanitize environment variable value to prevent injection attacks.

    Args:
        value: Raw environment variable value
        max_length: Maximum allowed length

    Returns:
        Sanitized value

    Raises:
        ValueError: If value is invalid or too long
    """
    if not isinstance(value, str):
        raise ValueError("Environment variable must be a string")

    if len(value) > max_length:
        raise ValueError(f"Environment variable too long (max {max_length} characters)")

    # Remove potentially dangerous characters while preserving functionality
    # Allow alphanumeric, spaces, hyphens, underscores, dots, slashes, colons
    sanitized = re.sub(r'[^\w\s\-_./:]', '', value)

    return sanitized.strip()


def get_secure_env_var(var_name: str, default: Optional[str] = None,
                      required: bool = False, max_length: int = 1000) -> Optional[str]:
    """
    Get environment variable with Docker secrets fallback and sanitization.

    Priority order:
    1. Docker secret (if exists)
    2. Environment variable
    3. Default value

    Args:
        var_name: Environment variable name
        default: Default value
        required: Whether variable is required
        max_length: Maximum allowed length

    Returns:
        Sanitized environment variable value

    Raises:
        ValueError: If required variable is missing or invalid
    """
    # Try Docker secret first
    value = read_docker_secret(var_name)

    # Fall back to environment variable
    if value is None:
        value = os.getenv(var_name)

    # Use default if still None
    if value is None:
        if required:
            raise ValueError(f"Required environment variable '{var_name}' not found")
        return default

    # Sanitize the value
    try:
        return sanitize_env_value(value, max_length)
    except ValueError as e:
        if required:
            raise ValueError(f"Invalid value for required environment variable '{var_name}': {e}")
        logging.warning(f"Invalid value for environment variable '{var_name}': {e}")
        return default


def load_api_keys() -> List[str]:
    """Load API keys from secure environment variables."""
    api_keys_str = get_secure_env_var('API_KEYS', '')
    if api_keys_str:
        return [key.strip() for key in api_keys_str.split(',') if key.strip()]
    return []


def validate_environment() -> Dict[str, str]:
    """
    Validate all required environment variables on startup.

    Returns:
        Dict of validated environment variables

    Raises:
        ValueError: If any required variables are missing or invalid
    """
    validated_vars = {}

    # Required variables for basic functionality (none currently required)
    required_vars = [
        # ('HF_TOKEN', None, 200),  # HuggingFace token (made optional)
    ]

    # Optional but recommended variables
    optional_vars = [
        ('API_KEYS', None, 1000),  # Comma-separated API keys
        ('LOG_LEVEL', 'INFO', 10),
        ('MAX_FILE_SIZE_MB', '100', 10),
    ]

    # Validate required variables
    for var_name, default, max_len in required_vars:
        try:
            value = get_secure_env_var(var_name, default, required=True, max_length=max_len)
            validated_vars[var_name] = value
        except ValueError as e:
            logging.error(f"Environment validation failed: {e}")
            raise

    # Validate optional variables
    for var_name, default, max_len in optional_vars:
        try:
            value = get_secure_env_var(var_name, default, required=False, max_length=max_len)
            if value is not None:
                validated_vars[var_name] = value
        except ValueError as e:
            logging.warning(f"Optional environment variable '{var_name}' invalid: {e}")

    return validated_vars


@dataclass
class ASRConfig:
    """Configuration for Automatic Speech Recognition parameters."""

    # Core ASR parameters
    batch_size: int = 32  # Batch size for processing audio segments
    compute_type: str = "fp32"  # Compute precision: "fp16", "fp32", "int8"
    language: str = "en"  # Language code for ASR model

    # VAD (Voice Activity Detection) parameters
    use_vad: bool = True  # Enable voice activity detection
    vad_threshold: float = 0.5  # Threshold for speech detection (0.0-1.0)
    min_segment_duration: float = 0.05  # Minimum duration for speech segments (seconds)

    # Model parameters
    asr_model_name: str = "nvidia/parakeet-tdt-1.1b"  # ASR model to use (faster variant)
    device: str = "auto"  # Device: "auto", "cpu", "cuda"

    # Batch processing
    enable_batch_processing: bool = True  # Enable batch processing for multiple segments


@dataclass
class DiarizationConfig:
    """Configuration for Speaker Diarization parameters."""

    # Backend selection
    backend: str = "hybrid"  # "hybrid" (Pyannote), "nvidia" (Sortformer), "auto"

    # Pyannote settings (for hybrid backend)
    pyannote_model: str = "pyannote/speaker-diarization-community-1"
    hf_token: Optional[str] = field(default_factory=lambda: get_secure_env_var('HF_TOKEN'))

    # NVIDIA settings (for nvidia backend)
    nvidia_model: str = "nvidia/diar_streaming_sortformer_4spk-v2"

    # Common settings
    device: str = "auto"  # Device: "auto", "cpu", "cuda"
    num_speakers: Optional[int] = None  # Expected number of speakers (1-4)
    min_speakers: Optional[int] = None  # Minimum speakers (Pyannote)
    max_speakers: Optional[int] = None  # Maximum speakers (Pyannote)

    # Legacy NVIDIA streaming parameters (for nvidia backend)
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
class LoggingConfig:
    """Configuration for logging parameters."""

    # Log levels
    app_log_level: str = "INFO"  # Application log level
    worker_log_level: str = "INFO"  # Worker subprocess log level

    # Log files
    app_log_file: str = "logs/app.log"  # Application log file path
    worker_log_file: str = "logs/worker.log"  # Worker log file path

    # Log rotation
    max_log_size_mb: int = 10  # Maximum log file size in MB
    backup_count: int = 5  # Number of backup log files to keep

    # Log format
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"

    # Sensitive data masking
    mask_sensitive_data: bool = True  # Enable sensitive data masking
    sensitive_patterns: List[str] = field(default_factory=lambda: [
        r'HF_TOKEN=[^\s&]+',  # HuggingFace tokens
        r'hf_[a-zA-Z0-9]{34,}',  # HuggingFace API keys
        r'/home/[^/\s]+',  # Home directory paths
        r'/tmp/[^/\s]+',  # Temp directory paths
        r'C:\\Users\\[^\\]+',  # Windows user paths
        r'/Users/[^/]+',  # macOS user paths
    ])


@dataclass
class SecurityConfig:
    """Configuration for API security parameters."""

    # Authentication
    enable_api_key_auth: bool = True  # Enable API key authentication
    api_keys: List[str] = field(default_factory=lambda: [])  # Will be loaded dynamically
    api_key_header: str = "X-API-Key"  # Header name for API key

    # CORS settings
    cors_enabled: bool = True  # Enable CORS
    cors_origins: List[str] = field(default_factory=lambda: ["*"])  # Allowed origins
    cors_methods: List[str] = field(default_factory=lambda: ["GET", "POST"])  # Allowed methods
    cors_headers: List[str] = field(default_factory=lambda: ["*"])  # Allowed headers
    cors_allow_credentials: bool = False  # Allow credentials

    # Security headers
    security_headers_enabled: bool = True  # Enable security headers
    hsts_enabled: bool = True  # HTTP Strict Transport Security
    hsts_max_age: int = 31536000  # HSTS max age (1 year)
    hsts_include_subdomains: bool = True  # Include subdomains in HSTS
    hsts_preload: bool = False  # Preload HSTS

    csp_enabled: bool = True  # Content Security Policy
    csp_policy: str = "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'"

    x_frame_options: str = "DENY"  # X-Frame-Options header
    x_content_type_options: str = "nosniff"  # X-Content-Type-Options header
    referrer_policy: str = "strict-origin-when-cross-origin"  # Referrer-Policy header

    # Input sanitization
    sanitize_inputs: bool = True  # Enable input sanitization
    max_filename_length: int = 255  # Maximum filename length
    max_parameter_length: int = 1000  # Maximum parameter value length
    allowed_filename_chars: str = r"^[a-zA-Z0-9._\-\s]+$"  # Allowed filename characters regex


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

    # Data protection enhancements
    encrypt_temp_files: bool = False  # Encrypt temporary files during processing
    encryption_key: Optional[str] = None  # Encryption key (if None, generate randomly)
    secure_delete_overwrites: int = 3  # Number of overwrites for secure deletion
    enable_audit_logging: bool = True  # Enable audit logging for file operations
    audit_log_file: str = "logs/audit.log"  # Audit log file path
    temp_file_retention_hours: int = 24  # Hours to retain temporary files before forced cleanup
    auto_retention_cleanup: bool = True  # Enable automatic cleanup based on retention policy

    # TempFileTracker cleanup parameters
    max_retry_attempts: int = 3  # Maximum number of retry attempts for failed deletions
    cleanup_timeout_seconds: int = 30  # Timeout for cleanup operations to prevent hanging

    # Validation settings
    allowed_mime_types: List[str] = field(default_factory=lambda: [
        'audio/mpeg', 'audio/wav', 'audio/x-wav', 'audio/flac', 'audio/x-flac',
        'audio/mp4', 'audio/x-m4a', 'audio/aac'
    ])
    rate_limit_requests: int = 10  # Requests per time window
    rate_limit_window_seconds: int = 60  # Time window in seconds


@dataclass
class GlobalConfig:
    """Global configuration that combines all parameter groups."""

    # Main configuration sections
    asr: ASRConfig = field(default_factory=ASRConfig)
    diarization: DiarizationConfig = field(default_factory=DiarizationConfig)
    streaming: StreamingConfig = field(default_factory=StreamingConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'asr': self.asr.__dict__,
            'diarization': self.diarization.__dict__,
            'streaming': self.streaming.__dict__,
            'processing': self.processing.__dict__,
            'logging': self.logging.__dict__,
            'security': self.security.__dict__
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'GlobalConfig':
        """Create configuration from dictionary."""
        return cls(
            asr=ASRConfig(**config_dict.get('asr', {})),
            diarization=DiarizationConfig(**config_dict.get('diarization', {})),
            streaming=StreamingConfig(**config_dict.get('streaming', {})),
            processing=ProcessingConfig(**config_dict.get('processing', {})),
            logging=LoggingConfig(**config_dict.get('logging', {})),
            security=SecurityConfig(**config_dict.get('security', {}))
        )


# Default global configuration instance
# Modify these values here to change defaults across the entire system
DEFAULT_CONFIG = GlobalConfig(
    asr=ASRConfig(
        batch_size=32,  # Increased from 16 for better GPU utilization
        compute_type="fp32",  # Changed to fp32 for better accuracy
        language="en",  # As requested
        use_vad=True,
        vad_threshold=0.5,
        min_segment_duration=0.05,
        asr_model_name="nvidia/parakeet-tdt-1.1b",
        device="auto",
        enable_batch_processing=True
    ),
    diarization=DiarizationConfig(
        backend="hybrid",  # Use Pyannote by default for better quality
        pyannote_model="pyannote/speaker-diarization-community-1",
        hf_token=None,  # Will be set from environment
        nvidia_model="nvidia/diar_streaming_sortformer_4spk-v2",
        device="auto",
        # Legacy NVIDIA parameters (used if backend="nvidia")
        chunk_size=6,
        right_context=7,
        fifo_size=188,
        update_period=144,
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
        auto_cleanup=True,
        encrypt_temp_files=False,  # Disabled by default for performance
        encryption_key=None,  # Will generate randomly if encryption enabled
        secure_delete_overwrites=3,  # 3 overwrites for secure deletion
        enable_audit_logging=True,  # Enable audit logging
        audit_log_file="logs/audit.log",  # Audit log location
        temp_file_retention_hours=24,  # Retain temp files for 24 hours
        auto_retention_cleanup=True,  # Enable retention-based cleanup
        max_retry_attempts=3,  # Maximum retry attempts for failed deletions
        cleanup_timeout_seconds=30  # Timeout for cleanup operations
    ),
    logging=LoggingConfig(
        app_log_level="INFO",
        worker_log_level="INFO",
        app_log_file="logs/app.log",
        worker_log_file="logs/worker.log",
        max_log_size_mb=10,
        backup_count=5,
        mask_sensitive_data=True
    ),
    security=SecurityConfig(
        enable_api_key_auth=True,  # Enable API key authentication
        api_keys=[],  # Should be set via environment variable or config override
        api_key_header="X-API-Key",
        cors_enabled=True,
        cors_origins=["*"],  # Allow all origins by default
        cors_methods=["GET", "POST"],
        cors_headers=["*"],
        cors_allow_credentials=False,
        security_headers_enabled=True,
        hsts_enabled=True,
        hsts_max_age=31536000,
        hsts_include_subdomains=True,
        hsts_preload=False,
        csp_enabled=True,
        csp_policy="default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'",
        x_frame_options="DENY",
        x_content_type_options="nosniff",
        referrer_policy="strict-origin-when-cross-origin",
        sanitize_inputs=True,
        max_filename_length=255,
        max_parameter_length=1000,
        allowed_filename_chars=r"^[a-zA-Z0-9._\-\s\(\)\[\]\&\+\,\'\"\%]+$"
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
        elif hasattr(config.security, key):
            setattr(config.security, key, value)

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
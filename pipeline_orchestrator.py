#!/usr/bin/env python3
"""
Modular Pipeline Orchestrator for ASR and Diarization

This module provides a secure, modular pipeline orchestrator that coordinates
diarization and ASR modules with automatic cleanup and secure file handling.
"""

import os
import sys
import tempfile
import shutil
import hashlib
import json
import secrets
import time
from typing import List, Dict, Optional, Any, Union, TYPE_CHECKING
from pathlib import Path
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import logging
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

from nvidia_diarization import NvidiaDiarization
from nvidia_asr import NvidiaASR
from hybrid_diarization import HybridDiarization
from config import get_config


@dataclass
class PipelineConfig:
    """Configuration for the pipeline orchestrator."""
    # Diarization settings
    diarization_backend: str = get_config().diarization.backend
    diarization_pyannote_model: str = get_config().diarization.pyannote_model
    diarization_nvidia_model: str = get_config().diarization.nvidia_model
    diarization_hf_token: Optional[str] = get_config().diarization.hf_token
    diarization_num_speakers: Optional[int] = get_config().diarization.num_speakers
    diarization_min_speakers: Optional[int] = get_config().diarization.min_speakers
    diarization_max_speakers: Optional[int] = get_config().diarization.max_speakers

    # Legacy NVIDIA settings (for backward compatibility)
    diarization_chunk_size: int = get_config().diarization.chunk_size
    diarization_right_context: int = get_config().diarization.right_context
    diarization_fifo_size: int = get_config().diarization.fifo_size
    diarization_update_period: int = get_config().diarization.update_period
    diarization_speaker_cache_size: int = get_config().diarization.speaker_cache_size

    # ASR settings
    asr_model: str = get_config().asr.asr_model_name
    asr_use_vad: bool = get_config().asr.use_vad
    asr_vad_threshold: float = get_config().asr.vad_threshold
    asr_min_segment_duration: float = get_config().asr.min_segment_duration
    asr_batch_size: int = get_config().asr.batch_size
    asr_enable_batch_processing: bool = get_config().asr.enable_batch_processing

    # Security settings
    secure_temp_dir: bool = get_config().processing.secure_temp_dir
    auto_cleanup: bool = get_config().processing.auto_cleanup
    max_file_size_mb: int = get_config().processing.max_file_size_mb
    allowed_extensions: Optional[List[str]] = None

    # Data protection enhancements
    encrypt_temp_files: bool = get_config().processing.encrypt_temp_files
    encryption_key: Optional[str] = get_config().processing.encryption_key
    secure_delete_overwrites: int = get_config().processing.secure_delete_overwrites
    enable_audit_logging: bool = get_config().processing.enable_audit_logging
    audit_log_file: str = get_config().processing.audit_log_file
    retention_hours: int = get_config().processing.retention_hours
    auto_retention_cleanup: bool = get_config().processing.auto_retention_cleanup

    # Processing settings
    device: str = get_config().asr.device
    output_format: str = get_config().processing.output_format  # json, txt, or both

    def __post_init__(self):
        if self.allowed_extensions is None:
            self.allowed_extensions = get_config().processing.allowed_extensions


@dataclass
class FileAuditEntry:
    """Audit log entry for file operations."""
    timestamp: float
    operation: str  # 'create', 'access', 'delete', 'encrypt', 'decrypt'
    file_path: str
    file_size: Optional[int] = None
    user_id: Optional[str] = None
    success: bool = True
    error_message: Optional[str] = None


class SecureTempManager:
    """Enhanced secure temporary file manager with encryption, audit logging, and retention policies."""

    def __init__(
        self,
        use_secure_temp: bool = True,
        base_temp_dir: Optional[str] = None,
        encrypt_files: bool = False,
        encryption_key: Optional[str] = None,
        secure_delete_overwrites: int = 3,
        enable_audit_logging: bool = True,
        audit_log_file: str = "logs/audit.log",
        retention_hours: int = 24,
        auto_retention_cleanup: bool = True
    ):
        self.use_secure_temp = use_secure_temp
        self.base_temp_dir = base_temp_dir or tempfile.gettempdir()
        self.encrypt_files = encrypt_files
        self.encryption_key = encryption_key
        self.secure_delete_overwrites = secure_delete_overwrites
        self.enable_audit_logging = enable_audit_logging
        self.audit_log_file = audit_log_file
        self.retention_hours = retention_hours
        self.auto_retention_cleanup = auto_retention_cleanup

        self.temp_dirs: List[str] = []
        self.temp_files: List[str] = []
        self.file_timestamps: Dict[str, float] = {}  # Track creation times for retention
        self.fernet: Optional[Fernet] = None

        # Initialize encryption if enabled
        if self.encrypt_files:
            self._initialize_encryption()

        # Initialize audit logging
        if self.enable_audit_logging:
            self._ensure_audit_log_directory()

        # Perform initial retention cleanup
        if self.auto_retention_cleanup:
            self._cleanup_expired_files()

    def _initialize_encryption(self):
        """Initialize encryption with provided or generated key."""
        if self.encryption_key:
            # Use provided key
            key_bytes = self.encryption_key.encode()
        else:
            # Generate a random key
            key_bytes = secrets.token_bytes(32)

        # Create Fernet cipher
        self.fernet = Fernet(base64.urlsafe_b64encode(key_bytes))

    def _ensure_audit_log_directory(self):
        """Ensure the audit log directory exists."""
        log_dir = os.path.dirname(self.audit_log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

    def _log_audit_event(self, operation: str, file_path: str, **kwargs):
        """Log an audit event."""
        if not self.enable_audit_logging:
            return

        entry = FileAuditEntry(
            timestamp=time.time(),
            operation=operation,
            file_path=file_path,
            **kwargs
        )

        try:
            with open(self.audit_log_file, 'a') as f:
                json.dump(asdict(entry), f)
                f.write('\n')
        except Exception as e:
            # Log audit failure to stderr but don't raise
            print(f"Failed to write audit log: {e}", file=sys.stderr)

    def _encrypt_data(self, data: bytes) -> bytes:
        """Encrypt data if encryption is enabled."""
        if self.fernet and self.encrypt_files:
            return self.fernet.encrypt(data)
        return data

    def _decrypt_data(self, data: bytes) -> bytes:
        """Decrypt data if encryption is enabled."""
        if self.fernet and self.encrypt_files:
            return self.fernet.decrypt(data)
        return data

    def _secure_delete_file(self, file_path: str):
        """Securely delete a file with multiple overwrites."""
        if not os.path.exists(file_path):
            return

        try:
            file_size = os.path.getsize(file_path)

            # Multiple overwrites with different patterns
            patterns = [b'\x00', b'\xFF', secrets.token_bytes(1) * file_size]

            for i in range(min(self.secure_delete_overwrites, len(patterns))):
                with open(file_path, 'wb') as f:
                    f.write(patterns[i][:file_size])

            # Final overwrite with random data
            with open(file_path, 'wb') as f:
                f.write(secrets.token_bytes(file_size))

            os.remove(file_path)
            self._log_audit_event('delete', file_path, file_size=file_size, success=True)

        except Exception as e:
            # Fallback to regular deletion
            try:
                os.remove(file_path)
            except OSError:
                pass
            self._log_audit_event('delete', file_path, success=False, error_message=str(e))

    def _cleanup_expired_files(self):
        """Clean up files older than retention period."""
        if not self.auto_retention_cleanup:
            return

        current_time = time.time()
        retention_seconds = self.retention_hours * 3600
        expired_files = []

        for file_path, timestamp in self.file_timestamps.items():
            if current_time - timestamp > retention_seconds:
                expired_files.append(file_path)

        for file_path in expired_files:
            if os.path.exists(file_path):
                self._secure_delete_file(file_path)
            if file_path in self.temp_files:
                self.temp_files.remove(file_path)
            del self.file_timestamps[file_path]

    @contextmanager
    def secure_temp_dir(self, prefix: str = "asr_pipeline_"):
        """Create a secure temporary directory."""
        if self.use_secure_temp:
            # Create temp dir with restrictive permissions
            temp_dir = tempfile.mkdtemp(prefix=prefix, dir=self.base_temp_dir)
            # Set restrictive permissions (owner read/write/execute only)
            os.chmod(temp_dir, 0o700)
            self.temp_dirs.append(temp_dir)
            self.file_timestamps[temp_dir] = time.time()
            self._log_audit_event('create_dir', temp_dir)
        else:
            temp_dir = self.base_temp_dir

        try:
            yield temp_dir
        finally:
            if self.use_secure_temp and os.path.exists(temp_dir):
                self._secure_cleanup_dir(temp_dir)
                self.temp_dirs.remove(temp_dir)
                if temp_dir in self.file_timestamps:
                    del self.file_timestamps[temp_dir]

    def register_temp_file(self, file_path: str):
        """Register a temporary file for cleanup."""
        self.temp_files.append(file_path)
        self.file_timestamps[file_path] = time.time()
        self._log_audit_event('create', file_path, file_size=os.path.getsize(file_path) if os.path.exists(file_path) else None)

    def write_encrypted_file(self, file_path: str, data: bytes):
        """Write data to file with optional encryption."""
        encrypted_data = self._encrypt_data(data)
        with open(file_path, 'wb') as f:
            f.write(encrypted_data)
        self.register_temp_file(file_path)
        self._log_audit_event('encrypt' if self.encrypt_files else 'create', file_path, file_size=len(encrypted_data))

    def read_encrypted_file(self, file_path: str) -> bytes:
        """Read data from file with optional decryption."""
        with open(file_path, 'rb') as f:
            encrypted_data = f.read()
        decrypted_data = self._decrypt_data(encrypted_data)
        self._log_audit_event('decrypt' if self.encrypt_files else 'access', file_path, file_size=len(encrypted_data))
        return decrypted_data

    def cleanup_all(self):
        """Clean up all registered temporary files and directories."""
        # Clean up files
        for file_path in self.temp_files[:]:  # Copy list to avoid modification during iteration
            if os.path.exists(file_path):
                self._secure_delete_file(file_path)
            self.temp_files.remove(file_path)
            if file_path in self.file_timestamps:
                del self.file_timestamps[file_path]

        # Clean up directories
        for dir_path in self.temp_dirs[:]:
            if os.path.exists(dir_path):
                self._secure_cleanup_dir(dir_path)
            self.temp_dirs.remove(dir_path)
            if dir_path in self.file_timestamps:
                del self.file_timestamps[dir_path]

    def _secure_cleanup_dir(self, dir_path: str):
        """Securely clean up a directory by overwriting files before deletion."""
        try:
            for root, dirs, files in os.walk(dir_path, topdown=False):
                for file in files:
                    file_path = os.path.join(root, file)
                    self._secure_delete_file(file_path)
                for dir_name in dirs:
                    try:
                        os.rmdir(os.path.join(root, dir_name))
                    except OSError:
                        pass
            os.rmdir(dir_path)
        except OSError:
            # If secure cleanup fails, try regular removal
            shutil.rmtree(dir_path, ignore_errors=True)


class PipelineOrchestrator:
    """
    Modular pipeline orchestrator for ASR and Diarization processing.

    Features:
    - Secure file handling with automatic cleanup
    - Modular design with configurable components
    - Batch processing capabilities
    - Comprehensive error handling
    - Progress tracking and logging
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize the pipeline orchestrator.

        Args:
            config: Pipeline configuration
        """
        self.config = config or PipelineConfig()
        self.temp_manager = SecureTempManager(
            use_secure_temp=self.config.secure_temp_dir,
            encrypt_files=self.config.encrypt_temp_files,
            encryption_key=self.config.encryption_key,
            secure_delete_overwrites=self.config.secure_delete_overwrites,
            enable_audit_logging=self.config.enable_audit_logging,
            audit_log_file=self.config.audit_log_file,
            retention_hours=self.config.retention_hours,
            auto_retention_cleanup=self.config.auto_retention_cleanup
        )
        self.diarization_module: Optional[Any] = None  # Can be NvidiaDiarization or HybridDiarization
        self.asr_module: Optional[NvidiaASR] = None
        self.logger = self._setup_logging()

    def _setup_logging(self) -> logging.Logger:
        """Set up logging for the orchestrator."""
        logger = logging.getLogger("PipelineOrchestrator")
        logger.setLevel(logging.INFO)

        # Create console handler
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)

        # Add handler to logger
        if not logger.handlers:
            logger.addHandler(handler)

        return logger

    def _validate_input_file(self, file_path: str) -> bool:
        """
        Validate input audio file.

        Args:
            file_path: Path to audio file

        Returns:
            True if valid, False otherwise
        """
        if not os.path.exists(file_path):
            self.logger.error(f"Input file does not exist: {file_path}")
            return False

        # Check file extension
        file_ext = Path(file_path).suffix.lower()
        if self.config.allowed_extensions and file_ext not in self.config.allowed_extensions:
            self.logger.error(f"Unsupported file extension: {file_ext}")
            return False

        # Check file size
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        if file_size_mb > self.config.max_file_size_mb:
            self.logger.error(f"File too large: {file_size_mb:.1f}MB (max: {self.config.max_file_size_mb}MB)")
            return False

        return True

    def _initialize_modules(self):
        """Initialize diarization and ASR modules."""
        if self.diarization_module is None:
            self.logger.info(f"Initializing diarization module (backend: {self.config.diarization_backend})...")

            if self.config.diarization_backend == "hybrid":
                # Use Pyannote-based hybrid diarization
                self.diarization_module = HybridDiarization(
                    pyannote_model=self.config.diarization_pyannote_model,
                    hf_token=self.config.diarization_hf_token,
                    device=self.config.device,
                    min_speakers=self.config.diarization_min_speakers,
                    max_speakers=self.config.diarization_max_speakers
                )
            elif self.config.diarization_backend == "nvidia":
                # Use NVIDIA Sortformer diarization
                self.diarization_module = NvidiaDiarization(
                    model_name=self.config.diarization_nvidia_model,
                    chunk_size=self.config.diarization_chunk_size,
                    right_context=self.config.diarization_right_context,
                    fifo_size=self.config.diarization_fifo_size,
                    update_period=self.config.diarization_update_period,
                    speaker_cache_size=self.config.diarization_speaker_cache_size,
                    device=self.config.device
                )
            else:
                raise ValueError(f"Unsupported diarization backend: {self.config.diarization_backend}")

        if self.asr_module is None:
            self.logger.info("Initializing ASR module...")
            self.asr_module = NvidiaASR(
                asr_model_name=self.config.asr_model,
                device=self.config.device,
                use_vad=self.config.asr_use_vad,
                vad_threshold=self.config.asr_vad_threshold,
                min_segment_duration=self.config.asr_min_segment_duration,
                batch_size=self.config.asr_batch_size,
                enable_batch_processing=self.config.asr_enable_batch_processing
            )

    def _process_single_file(self, audio_path: str) -> Dict[str, Any]:
        """
        Process a single audio file through the pipeline.

        Args:
            audio_path: Path to audio file

        Returns:
            Processing results
        """
        self.logger.info(f"Processing file: {audio_path}")

        with self.temp_manager.secure_temp_dir() as temp_dir:
            try:
                # Step 1: Run diarization
                self.logger.info("Running speaker diarization...")
                assert self.diarization_module is not None, "Diarization module not initialized"

                if isinstance(self.diarization_module, HybridDiarization):
                    # Hybrid diarization returns different format
                    diarization_result = self.diarization_module.diarize_audio(audio_path)
                    speaker_segments = diarization_result['segments']

                    # Apply speaker filtering if requested
                    if self.config.diarization_num_speakers:
                        speaker_segments = self.diarization_module.filter_speakers(
                            speaker_segments, self.config.diarization_num_speakers
                        )
                else:
                    # NVIDIA diarization
                    speaker_segments = self.diarization_module.run_offline_diarization(audio_path)

                if not speaker_segments:
                    self.logger.warning("No speaker segments detected")
                    return {
                        'file': audio_path,
                        'segments': [],
                        'error': 'No speaker segments detected'
                    }

                # Step 2: Process segments with ASR
                self.logger.info(f"Processing {len(speaker_segments)} speaker segments with ASR...")

                # Load audio for segmentation
                import torchaudio
                waveform, sample_rate = torchaudio.load(audio_path)

                results = []
                batch_segments = []
                batch_info = []

                for segment in speaker_segments:
                    start_time = segment['start']
                    end_time = segment['end']
                    speaker = segment['speaker']

                    segment_duration = end_time - start_time

                    # Skip segments that are too short
                    if segment_duration < self.config.asr_min_segment_duration:
                        self.logger.debug(f"Skipping {speaker} segment ({segment_duration:.3f}s) - too short")
                        continue

                    # Extract audio segment
                    start_sample = int(start_time * sample_rate)
                    end_sample = int(end_time * sample_rate)
                    segment_waveform = waveform[:, start_sample:end_sample]

                    # Save temporary segment
                    temp_file = os.path.join(temp_dir, f"segment_{speaker}_{start_time:.2f}.wav")
                    torchaudio.save(temp_file, segment_waveform, sample_rate)
                    self.temp_manager.register_temp_file(temp_file)

                    if self.config.asr_enable_batch_processing:
                        batch_segments.append(temp_file)
                        batch_info.append((speaker, start_time, end_time))
                    else:
                        # Process individually
                        assert self.asr_module is not None, "ASR module not initialized"
                        transcription = self.asr_module.transcribe_file(temp_file)
                        if transcription and 'segments' in transcription and transcription['segments']:
                            text = transcription['segments'][0]['text']
                            if text.strip():
                                results.append({
                                    'speaker': speaker,
                                    'start': start_time,
                                    'end': end_time,
                                    'text': text
                                })

                # Process batch if enabled
                if self.config.asr_enable_batch_processing and batch_segments:
                    self.logger.info(f"Batch processing {len(batch_segments)} segments...")
                    assert self.asr_module is not None, "ASR module not initialized"
                    batch_results = self.asr_module.transcribe_files_batch(batch_segments)

                    # Combine results
                    for i, result in enumerate(batch_results):
                        if result and 'segments' in result and result['segments']:
                            for segment_result in result['segments']:
                                if segment_result['text'].strip():
                                    speaker, start_time, end_time = batch_info[i]
                                    results.append({
                                        'speaker': speaker,
                                        'start': start_time,
                                        'end': end_time,
                                        'text': segment_result['text']
                                    })

                # Sort results by timestamp
                results_sorted = sorted(results, key=lambda x: x['start'])

                self.logger.info(f"Successfully processed {len(results_sorted)} segments")

                return {
                    'file': audio_path,
                    'segments': results_sorted,
                    'total_segments': len(results_sorted)
                }

            except Exception as e:
                self.logger.error(f"Error processing {audio_path}: {e}")
                return {
                    'file': audio_path,
                    'segments': [],
                    'error': str(e)
                }

    def process_files(self, audio_paths: Union[str, List[str]]) -> List[Dict[str, Any]]:
        """
        Process one or more audio files through the pipeline.

        Args:
            audio_paths: Path(s) to audio file(s)

        Returns:
            List of processing results
        """
        if isinstance(audio_paths, str):
            audio_paths = [audio_paths]

        # Validate all input files
        valid_paths = []
        for path in audio_paths:
            if self._validate_input_file(path):
                valid_paths.append(path)
            else:
                self.logger.error(f"Skipping invalid file: {path}")

        if not valid_paths:
            raise ValueError("No valid input files provided")

        # Initialize modules
        self._initialize_modules()

        results = []

        try:
            for audio_path in valid_paths:
                result = self._process_single_file(audio_path)
                results.append(result)

        finally:
            # Cleanup
            if self.config.auto_cleanup:
                self.logger.info("Performing automatic cleanup...")
                self.temp_manager.cleanup_all()
                if self.diarization_module:
                    self.diarization_module.cleanup()
                if self.asr_module:
                    self.asr_module.cleanup()

        return results

    def save_results(self, results: List[Dict[str, Any]], output_path: str):
        """
        Save processing results to file.

        Args:
            results: Processing results
            output_path: Output file path
        """
        if self.config.output_format == "json" or self.config.output_format == "both":
            json_path = output_path if output_path.endswith('.json') else output_path + '.json'
            with open(json_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            self.logger.info(f"Results saved to {json_path}")

        if self.config.output_format == "txt" or self.config.output_format == "both":
            txt_path = output_path if output_path.endswith('.txt') else output_path + '.txt'
            with open(txt_path, 'w') as f:
                for result in results:
                    f.write(f"File: {result['file']}\n")
                    for segment in result.get('segments', []):
                        f.write(f"{segment['speaker']}: {segment['text']}\n")
                    f.write("\n")
            self.logger.info(f"Text results saved to {txt_path}")

    def cleanup(self):
        """Clean up all resources."""
        self.temp_manager.cleanup_all()
        if self.diarization_module:
            self.diarization_module.cleanup()
            self.diarization_module = None
        if self.asr_module:
            self.asr_module.cleanup()
            self.asr_module = None
        self.logger.info("Pipeline orchestrator cleanup complete")


# Convenience functions
def create_pipeline_orchestrator(config: Optional[PipelineConfig] = None) -> PipelineOrchestrator:
    """
    Create a configured pipeline orchestrator.

    Args:
        config: Pipeline configuration

    Returns:
        Configured PipelineOrchestrator instance
    """
    return PipelineOrchestrator(config)


def process_audio_files(
    audio_paths: Union[str, List[str]],
    config: Optional[PipelineConfig] = None,
    output_path: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Convenience function to process audio files with default configuration.

    Args:
        audio_paths: Path(s) to audio file(s)
        config: Pipeline configuration
        output_path: Optional output path for results

    Returns:
        Processing results
    """
    orchestrator = create_pipeline_orchestrator(config)
    try:
        results = orchestrator.process_files(audio_paths)
        if output_path:
            orchestrator.save_results(results, output_path)
        return results
    finally:
        orchestrator.cleanup()
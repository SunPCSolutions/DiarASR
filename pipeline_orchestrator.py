#!/usr/bin/env python3
"""
Modular Pipeline Orchestrator for ASR and Diarization

This module provides a secure, modular pipeline orchestrator that coordinates
diarization and ASR modules with automatic cleanup and secure file handling.
"""

import os
import tempfile
import shutil
import hashlib
import json
from typing import List, Dict, Optional, Any, Union, TYPE_CHECKING
from pathlib import Path
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import logging

from nvidia_diarization import NvidiaDiarization
from nvidia_asr import NvidiaASR
from config import get_config


@dataclass
class PipelineConfig:
    """Configuration for the pipeline orchestrator."""
    # Diarization settings
    diarization_model: str = get_config().diarization.model_name
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

    # Processing settings
    device: str = get_config().asr.device
    output_format: str = get_config().processing.output_format  # json, txt, or both

    def __post_init__(self):
        if self.allowed_extensions is None:
            self.allowed_extensions = get_config().processing.allowed_extensions


class SecureTempManager:
    """Manages secure temporary files and directories."""

    def __init__(self, use_secure_temp: bool = True, base_temp_dir: Optional[str] = None):
        self.use_secure_temp = use_secure_temp
        self.base_temp_dir = base_temp_dir or tempfile.gettempdir()
        self.temp_dirs: List[str] = []
        self.temp_files: List[str] = []

    @contextmanager
    def secure_temp_dir(self, prefix: str = "asr_pipeline_"):
        """Create a secure temporary directory."""
        if self.use_secure_temp:
            # Create temp dir with restrictive permissions
            temp_dir = tempfile.mkdtemp(prefix=prefix, dir=self.base_temp_dir)
            # Set restrictive permissions (owner read/write/execute only)
            os.chmod(temp_dir, 0o700)
            self.temp_dirs.append(temp_dir)
        else:
            temp_dir = self.base_temp_dir

        try:
            yield temp_dir
        finally:
            if self.use_secure_temp and os.path.exists(temp_dir):
                self._secure_cleanup_dir(temp_dir)
                self.temp_dirs.remove(temp_dir)

    def register_temp_file(self, file_path: str):
        """Register a temporary file for cleanup."""
        self.temp_files.append(file_path)

    def cleanup_all(self):
        """Clean up all registered temporary files and directories."""
        # Clean up files
        for file_path in self.temp_files[:]:  # Copy list to avoid modification during iteration
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except OSError:
                    pass  # File might already be deleted
            self.temp_files.remove(file_path)

        # Clean up directories
        for dir_path in self.temp_dirs[:]:
            if os.path.exists(dir_path):
                self._secure_cleanup_dir(dir_path)
            self.temp_dirs.remove(dir_path)

    def _secure_cleanup_dir(self, dir_path: str):
        """Securely clean up a directory by overwriting files before deletion."""
        try:
            for root, dirs, files in os.walk(dir_path, topdown=False):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        # Overwrite file with zeros before deletion for security
                        with open(file_path, 'wb') as f:
                            f.write(b'\x00' * os.path.getsize(file_path))
                        os.remove(file_path)
                    except OSError:
                        pass
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
        self.temp_manager = SecureTempManager(self.config.secure_temp_dir)
        self.diarization_module: Optional[NvidiaDiarization] = None
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
            self.logger.info("Initializing diarization module...")
            self.diarization_module = NvidiaDiarization(
                model_name=self.config.diarization_model,
                chunk_size=self.config.diarization_chunk_size,
                right_context=self.config.diarization_right_context,
                fifo_size=self.config.diarization_fifo_size,
                update_period=self.config.diarization_update_period,
                speaker_cache_size=self.config.diarization_speaker_cache_size,
                device=self.config.device
            )

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
#!/usr/bin/env python3
"""
Comprehensive tests for TempFileTracker functionality.

This module provides unit tests, integration tests, and configuration tests
for the TempFileTracker class and related components.

Test Coverage:
- Unit tests for individual methods
- Integration tests with context manager usage
- Configuration tests for different options
- Edge cases and error handling
- Audit logging verification
- Cleanup reliability under various conditions
"""

import os
import tempfile
import shutil
import unittest
import time
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path
from datetime import datetime, timedelta
import sys

# Add app directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))

from temp_file_tracker import (
    TempFileTracker,
    TempFileTrackerConfig,
    FileMetadata
)


class TestTempFileTrackerConfig(unittest.TestCase):
    """Test TempFileTrackerConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = TempFileTrackerConfig()

        self.assertEqual(config.max_retry_attempts, 3)
        self.assertEqual(config.base_retry_delay, 0.1)
        self.assertEqual(config.max_retry_delay, 5.0)
        self.assertEqual(config.audit_log_level, "INFO")
        self.assertTrue(config.enable_audit_logging)
        self.assertEqual(config.temp_dir_permissions, 0o700)
        self.assertEqual(config.cleanup_timeout_seconds, 30)

    def test_custom_config(self):
        """Test custom configuration values."""
        config = TempFileTrackerConfig(
            max_retry_attempts=5,
            base_retry_delay=0.2,
            enable_audit_logging=False,
            temp_dir_permissions=0o755
        )

        self.assertEqual(config.max_retry_attempts, 5)
        self.assertEqual(config.base_retry_delay, 0.2)
        self.assertFalse(config.enable_audit_logging)
        self.assertEqual(config.temp_dir_permissions, 0o755)


class TestFileMetadata(unittest.TestCase):
    """Test FileMetadata dataclass."""

    def test_file_metadata_creation(self):
        """Test FileMetadata creation with required fields."""
        creation_time = datetime.now()
        metadata = FileMetadata(
            path="/tmp/test.wav",
            creation_time=creation_time,
            size=1024,
            purpose="test_file"
        )

        self.assertEqual(metadata.path, "/tmp/test.wav")
        self.assertEqual(metadata.creation_time, creation_time)
        self.assertEqual(metadata.size, 1024)
        self.assertEqual(metadata.purpose, "test_file")
        self.assertIsNone(metadata.last_access_time)
        self.assertEqual(metadata.access_count, 0)

    def test_file_metadata_with_optional_fields(self):
        """Test FileMetadata with optional fields."""
        creation_time = datetime.now()
        access_time = datetime.now()

        metadata = FileMetadata(
            path="/tmp/test.wav",
            creation_time=creation_time,
            size=1024,
            purpose="test_file",
            last_access_time=access_time,
            access_count=5
        )

        self.assertEqual(metadata.last_access_time, access_time)
        self.assertEqual(metadata.access_count, 5)


class TestTempFileTrackerInitialization(unittest.TestCase):
    """Test TempFileTracker initialization."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_logger = MagicMock()

    def test_init_with_default_config(self):
        """Test initialization with default config."""
        with patch('logging_config.get_logger', return_value=self.mock_logger):
            tracker = TempFileTracker()

            self.assertIsInstance(tracker.config, TempFileTrackerConfig)
            self.assertEqual(tracker.logger, self.mock_logger)
            self.assertEqual(tracker.tracked_files, {})
            self.assertEqual(tracker.temp_dirs, [])
            self.assertTrue(tracker._cleanup_successful)

    def test_init_with_custom_config(self):
        """Test initialization with custom config."""
        config = TempFileTrackerConfig(max_retry_attempts=5)
        tracker = TempFileTracker(config, self.mock_logger)

        self.assertEqual(tracker.config, config)
        self.assertEqual(tracker.logger, self.mock_logger)

    def test_init_with_custom_logger(self):
        """Test initialization with custom logger."""
        custom_logger = MagicMock()
        config = TempFileTrackerConfig()
        tracker = TempFileTracker(config, custom_logger)

        self.assertEqual(tracker.logger, custom_logger)


class TestTempFileTrackerContextManager(unittest.TestCase):
    """Test TempFileTracker context manager functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_logger = MagicMock()
        self.config = TempFileTrackerConfig()
        self.tracker = TempFileTracker(self.config, self.mock_logger)

    def test_context_manager_enter(self):
        """Test entering context manager."""
        result = self.tracker.__enter__()

        self.assertEqual(result, self.tracker)
        self.mock_logger.info.assert_called_with("TempFileTracker session started")

    def test_context_manager_exit_success(self):
        """Test exiting context manager on success."""
        with patch.object(self.tracker, '_perform_cleanup') as mock_cleanup:
            result = self.tracker.__exit__(None, None, None)

            self.assertFalse(result)  # Don't suppress exceptions
            mock_cleanup.assert_called_once()
            self.mock_logger.info.assert_called_with(
                "TempFileTracker session ended. Cleanup successful: True"
            )

    def test_context_manager_exit_with_cleanup_error(self):
        """Test exiting context manager when cleanup fails."""
        with patch.object(self.tracker, '_perform_cleanup', side_effect=Exception("Cleanup failed")):
            result = self.tracker.__exit__(None, None, None)

            self.assertFalse(result)  # Don't suppress exceptions
            self.assertFalse(self.tracker._cleanup_successful)
            self.mock_logger.error.assert_called_with("Critical error during cleanup: Cleanup failed")


class TestTempFileTrackerFileOperations(unittest.TestCase):
    """Test file creation and tracking operations."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_logger = MagicMock()
        self.config = TempFileTrackerConfig()
        self.tracker = TempFileTracker(self.config, self.mock_logger)
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_create_temp_file_success(self):
        """Test successful temporary file creation."""
        file_path = self.tracker.create_temp_file('.wav', 'test_', 'audio_segment')

        self.assertTrue(os.path.exists(file_path))
        self.assertTrue(file_path.endswith('.wav'))
        self.assertIn('test_', os.path.basename(file_path))

        # Check file is tracked
        self.assertIn(file_path, self.tracker.tracked_files)
        metadata = self.tracker.tracked_files[file_path]
        self.assertEqual(metadata.purpose, 'audio_segment')
        self.assertEqual(metadata.size, 0)
        self.assertIsInstance(metadata.creation_time, datetime)

        self.mock_logger.info.assert_called_with(
            f"Created temporary file: {file_path} (purpose: audio_segment)"
        )

        # Cleanup
        os.unlink(file_path)

    def test_create_temp_file_with_defaults(self):
        """Test temporary file creation with default parameters."""
        file_path = self.tracker.create_temp_file()

        self.assertTrue(os.path.exists(file_path))
        self.assertIn(file_path, self.tracker.tracked_files)

        # Cleanup
        os.unlink(file_path)

    @patch('tempfile.mkstemp', side_effect=OSError("Permission denied"))
    def test_create_temp_file_failure(self, mock_mkstemp):
        """Test temporary file creation failure."""
        with self.assertRaises(OSError) as cm:
            self.tracker.create_temp_file()

        self.assertIn("Permission denied", str(cm.exception))
        self.mock_logger.error.assert_called_with(
            "Failed to create temporary file: Permission denied"
        )

    def test_create_secure_temp_dir_success(self):
        """Test successful secure temporary directory creation."""
        dir_path = self.tracker.create_secure_temp_dir()

        self.assertTrue(os.path.exists(dir_path))
        self.assertIn(dir_path, self.tracker.temp_dirs)

        # Check permissions
        stat_info = os.stat(dir_path)
        expected_perms = 0o40700  # 0o700 with setuid/setgid bits
        self.assertEqual(stat_info.st_mode, expected_perms)

        self.mock_logger.info.assert_called_with(
            f"Created secure temporary directory: {dir_path}"
        )

        # Cleanup
        os.rmdir(dir_path)

    @patch('tempfile.mkdtemp', side_effect=OSError("Permission denied"))
    def test_create_secure_temp_dir_failure(self, mock_mkdtemp):
        """Test secure temporary directory creation failure."""
        with self.assertRaises(OSError) as cm:
            self.tracker.create_secure_temp_dir()

        self.assertIn("Permission denied", str(cm.exception))
        self.mock_logger.error.assert_called_with(
            "Failed to create secure temporary directory: Permission denied"
        )

    def test_track_existing_file_success(self):
        """Test tracking existing file."""
        test_file = os.path.join(self.temp_dir, "existing.wav")
        test_data = b"test audio data"
        with open(test_file, 'wb') as f:
            f.write(test_data)

        self.tracker.track_existing_file(test_file, "existing_audio")

        self.assertIn(test_file, self.tracker.tracked_files)
        metadata = self.tracker.tracked_files[test_file]
        self.assertEqual(metadata.purpose, "existing_audio")
        self.assertEqual(metadata.size, len(test_data))

        self.mock_logger.info.assert_called_with(
            f"Tracking existing file: {test_file} (purpose: existing_audio, size: {len(test_data)} bytes)"
        )

    def test_track_existing_file_not_found(self):
        """Test tracking non-existent file."""
        with self.assertRaises(FileNotFoundError) as cm:
            self.tracker.track_existing_file("/nonexistent/file.wav")

        self.assertIn("File does not exist", str(cm.exception))

    def test_record_file_access(self):
        """Test recording file access."""
        test_file = os.path.join(self.temp_dir, "test.wav")
        Path(test_file).touch()

        self.tracker.track_existing_file(test_file, "test")
        initial_metadata = self.tracker.tracked_files[test_file]

        # Record access
        self.tracker.record_file_access(test_file)

        updated_metadata = self.tracker.tracked_files[test_file]
        self.assertEqual(updated_metadata.access_count, initial_metadata.access_count + 1)
        self.assertIsNotNone(updated_metadata.last_access_time)

        # Record another access
        self.tracker.record_file_access(test_file)
        self.assertEqual(updated_metadata.access_count, initial_metadata.access_count + 2)

    def test_record_file_access_non_tracked(self):
        """Test recording access to non-tracked file."""
        # Should not raise error
        self.tracker.record_file_access("/non/tracked/file.wav")


class TestTempFileTrackerCleanup(unittest.TestCase):
    """Test cleanup operations."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_logger = MagicMock()
        self.config = TempFileTrackerConfig()
        self.tracker = TempFileTracker(self.config, self.mock_logger)
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_perform_cleanup_files_and_dirs(self):
        """Test cleanup of files and directories."""
        # Create test files
        file1 = os.path.join(self.temp_dir, "test1.wav")
        file2 = os.path.join(self.temp_dir, "test2.json")
        Path(file1).touch()
        Path(file2).touch()

        # Create test directory
        test_subdir = os.path.join(self.temp_dir, "subdir")
        os.makedirs(test_subdir)

        # Track them
        self.tracker.tracked_files[file1] = FileMetadata(file1, datetime.now(), 0, "test1")
        self.tracker.tracked_files[file2] = FileMetadata(file2, datetime.now(), 0, "test2")
        self.tracker.temp_dirs.append(test_subdir)

        # Perform cleanup
        self.tracker._perform_cleanup()

        # Files and dirs should be gone
        self.assertFalse(os.path.exists(file1))
        self.assertFalse(os.path.exists(file2))
        self.assertFalse(os.path.exists(test_subdir))

        # Tracking should be cleared
        self.assertEqual(len(self.tracker.tracked_files), 0)
        self.assertEqual(len(self.tracker.temp_dirs), 0)

        self.mock_logger.info.assert_any_call(
            "Starting cleanup of 2 files and 1 directories"
        )

    def test_secure_delete_file_success(self):
        """Test successful file deletion."""
        test_file = os.path.join(self.temp_dir, "test.wav")
        test_data = b"test data"
        with open(test_file, 'wb') as f:
            f.write(test_data)

        metadata = FileMetadata(test_file, datetime.now(), len(test_data), "test", None, 1)

        self.tracker._secure_delete_file(test_file, metadata)

        self.assertFalse(os.path.exists(test_file))
        self.mock_logger.info.assert_called_with(
            f"Successfully deleted file: {test_file} "
            f"(purpose: test, size: {len(test_data)} bytes, access_count: 1)"
        )

    def test_secure_delete_file_already_gone(self):
        """Test deletion of already deleted file."""
        nonexistent_file = "/tmp/nonexistent.wav"
        metadata = FileMetadata(nonexistent_file, datetime.now(), 0, "test")

        self.tracker._secure_delete_file(nonexistent_file, metadata)

        self.mock_logger.warning.assert_called_with(
            f"File already deleted or missing: {nonexistent_file}"
        )

    @patch('os.unlink', side_effect=OSError("Permission denied"))
    def test_secure_delete_file_retry_failure(self, mock_unlink):
        """Test file deletion failure after retries."""
        test_file = os.path.join(self.temp_dir, "test.wav")
        Path(test_file).touch()

        metadata = FileMetadata(test_file, datetime.now(), 0, "test")

        self.tracker._secure_delete_file(test_file, metadata)

        # Should have tried max_retry_attempts times
        self.assertEqual(mock_unlink.call_count, 3)
        self.assertFalse(self.tracker._cleanup_successful)

        self.mock_logger.error.assert_called_with(
            f"Failed to delete file {test_file} after 3 attempts"
        )

    def test_secure_delete_directory_success(self):
        """Test successful directory deletion."""
        test_dir = tempfile.mkdtemp()

        self.tracker._secure_delete_directory(test_dir)

        self.assertFalse(os.path.exists(test_dir))
        self.mock_logger.info.assert_called_with(
            f"Successfully deleted directory: {test_dir}"
        )

    def test_secure_delete_directory_already_gone(self):
        """Test deletion of already deleted directory."""
        nonexistent_dir = "/tmp/nonexistent_dir"

        self.tracker._secure_delete_directory(nonexistent_dir)

        self.mock_logger.warning.assert_called_with(
            f"Directory already deleted or missing: {nonexistent_dir}"
        )

    @patch('shutil.rmtree', side_effect=OSError("Permission denied"))
    def test_secure_delete_directory_failure(self, mock_rmtree):
        """Test directory deletion failure."""
        test_dir = tempfile.mkdtemp()

        self.tracker._secure_delete_directory(test_dir)

        self.assertFalse(self.tracker._cleanup_successful)
        self.mock_logger.error.assert_called_with(
            f"Failed to delete directory {test_dir}: Permission denied"
        )


class TestTempFileTrackerRetryLogic(unittest.TestCase):
    """Test retry delay calculation."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = TempFileTrackerConfig()
        self.tracker = TempFileTracker(self.config)

    def test_calculate_retry_delay(self):
        """Test exponential backoff with jitter."""
        # Test first attempt (attempt=1)
        delay1 = self.tracker._calculate_retry_delay(1)
        expected_base1 = 0.1 * (2 ** 0)  # 0.1
        self.assertGreaterEqual(delay1, expected_base1)
        self.assertLessEqual(delay1, 0.11)  # With jitter

        # Test second attempt (attempt=2)
        delay2 = self.tracker._calculate_retry_delay(2)
        expected_base2 = 0.1 * (2 ** 1)  # 0.2
        self.assertGreaterEqual(delay2, expected_base2)
        self.assertLessEqual(delay2, 0.22)

        # Test capped delay
        large_attempt = self.tracker._calculate_retry_delay(10)
        self.assertLessEqual(large_attempt, 5.0)  # max_retry_delay


class TestTempFileTrackerGetters(unittest.TestCase):
    """Test getter methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = TempFileTrackerConfig()
        self.tracker = TempFileTracker(self.config)
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_get_file_count(self):
        """Test getting file count."""
        self.assertEqual(self.tracker.get_file_count(), 0)

        # Add files
        file1 = os.path.join(self.temp_dir, "test1.wav")
        file2 = os.path.join(self.temp_dir, "test2.json")
        Path(file1).touch()
        Path(file2).touch()

        self.tracker.tracked_files[file1] = FileMetadata(file1, datetime.now(), 0, "test1")
        self.tracker.tracked_files[file2] = FileMetadata(file2, datetime.now(), 0, "test2")

        self.assertEqual(self.tracker.get_file_count(), 2)

    def test_get_directory_count(self):
        """Test getting directory count."""
        self.assertEqual(self.tracker.get_directory_count(), 0)

        self.tracker.temp_dirs = ["/tmp/dir1", "/tmp/dir2"]
        self.assertEqual(self.tracker.get_directory_count(), 2)

    def test_get_total_size(self):
        """Test getting total size of tracked files."""
        self.assertEqual(self.tracker.get_total_size(), 0)

        # Add files with sizes
        file1 = os.path.join(self.temp_dir, "test1.wav")
        file2 = os.path.join(self.temp_dir, "test2.json")
        data1 = b"x" * 100
        data2 = b"y" * 200

        with open(file1, 'wb') as f:
            f.write(data1)
        with open(file2, 'wb') as f:
            f.write(data2)

        self.tracker.tracked_files[file1] = FileMetadata(file1, datetime.now(), 100, "test1")
        self.tracker.tracked_files[file2] = FileMetadata(file2, datetime.now(), 200, "test2")

        self.assertEqual(self.tracker.get_total_size(), 300)

    def test_was_cleanup_successful(self):
        """Test cleanup success status."""
        self.assertTrue(self.tracker.was_cleanup_successful())

        self.tracker._cleanup_successful = False
        self.assertFalse(self.tracker.was_cleanup_successful())

    def test_get_tracked_files_info(self):
        """Test getting tracked files information."""
        self.assertEqual(self.tracker.get_tracked_files_info(), [])

        # Add a file
        file_path = os.path.join(self.temp_dir, "test.wav")
        Path(file_path).touch()

        creation_time = datetime.now()
        metadata = FileMetadata(
            path=file_path,
            creation_time=creation_time,
            size=1024,
            purpose="test_file",
            last_access_time=creation_time,
            access_count=5
        )
        self.tracker.tracked_files[file_path] = metadata

        info = self.tracker.get_tracked_files_info()
        self.assertEqual(len(info), 1)

        file_info = info[0]
        self.assertEqual(file_info['path'], file_path)
        self.assertEqual(file_info['creation_time'], creation_time.isoformat())
        self.assertEqual(file_info['size'], 1024)
        self.assertEqual(file_info['purpose'], "test_file")
        self.assertEqual(file_info['last_access_time'], creation_time.isoformat())
        self.assertEqual(file_info['access_count'], 5)


class TestTempFileTrackerIntegration(unittest.TestCase):
    """Integration tests for realistic usage scenarios."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_logger = MagicMock()
        self.config = TempFileTrackerConfig()

    def test_context_manager_full_lifecycle(self):
        """Test full lifecycle with context manager."""
        with patch('logging_config.get_logger', return_value=self.mock_logger):
            with TempFileTracker(self.config) as tracker:
                # Create files and directories
                temp_file1 = tracker.create_temp_file('.wav', purpose='audio')
                temp_file2 = tracker.create_temp_file('.json', purpose='metadata')
                temp_dir = tracker.create_secure_temp_dir()

                # Write data to files
                with open(temp_file1, 'wb') as f:
                    f.write(b'audio data' * 100)
                with open(temp_file2, 'w') as f:
                    f.write('{"metadata": "test"}')

                # Record access
                tracker.record_file_access(temp_file1)

                # Verify tracking
                self.assertEqual(tracker.get_file_count(), 2)
                self.assertEqual(tracker.get_directory_count(), 1)
                self.assertGreater(tracker.get_total_size(), 0)

            # After context exit, cleanup should have happened
            self.assertFalse(os.path.exists(temp_file1))
            self.assertFalse(os.path.exists(temp_file2))
            self.assertFalse(os.path.exists(temp_dir))
            self.assertTrue(tracker.was_cleanup_successful())

    def test_multiple_operations_with_cleanup(self):
        """Test multiple file operations with cleanup."""
        with patch('logging_config.get_logger', return_value=self.mock_logger):
            with TempFileTracker(self.config) as tracker:
                # Create multiple files
                files = []
                for i in range(5):
                    file_path = tracker.create_temp_file(f'.{i}', purpose=f'file_{i}')
                    files.append(file_path)

                    # Write varying amounts of data
                    with open(file_path, 'wb') as f:
                        f.write(b'x' * (i + 1) * 100)

                # Create directories
                dirs = []
                for i in range(3):
                    dir_path = tracker.create_secure_temp_dir()
                    dirs.append(dir_path)

                # Verify all exist
                for file_path in files:
                    self.assertTrue(os.path.exists(file_path))
                for dir_path in dirs:
                    self.assertTrue(os.path.exists(dir_path))

                self.assertEqual(tracker.get_file_count(), 5)
                self.assertEqual(tracker.get_directory_count(), 3)

            # All should be cleaned up
            for file_path in files:
                self.assertFalse(os.path.exists(file_path))
            for dir_path in dirs:
                self.assertFalse(os.path.exists(dir_path))

    def test_error_recovery_during_operations(self):
        """Test error recovery during file operations."""
        with patch('logging_config.get_logger', return_value=self.mock_logger):
            with TempFileTracker(self.config) as tracker:
                # Create a valid file
                valid_file = tracker.create_temp_file('.wav', purpose='valid')

                # Try to track non-existent file (should raise error)
                with self.assertRaises(FileNotFoundError):
                    tracker.track_existing_file('/nonexistent/file.wav')

                # Create another valid file
                another_file = tracker.create_temp_file('.json', purpose='another')

                # Verify valid files still work
                self.assertTrue(os.path.exists(valid_file))
                self.assertTrue(os.path.exists(another_file))
                self.assertEqual(tracker.get_file_count(), 2)

            # Cleanup should still work
            self.assertFalse(os.path.exists(valid_file))
            self.assertFalse(os.path.exists(another_file))
            self.assertTrue(tracker.was_cleanup_successful())


class TestTempFileTrackerConfiguration(unittest.TestCase):
    """Test different configuration options."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_logger = MagicMock()

    def test_audit_logging_enabled(self):
        """Test with audit logging enabled."""
        config = TempFileTrackerConfig(enable_audit_logging=True)
        tracker = TempFileTracker(config, self.mock_logger)

        temp_file = tracker.create_temp_file('.wav', purpose='test')

        self.mock_logger.info.assert_called()

        # Cleanup
        os.unlink(temp_file)

    def test_audit_logging_disabled(self):
        """Test with audit logging disabled."""
        config = TempFileTrackerConfig(enable_audit_logging=False)

        # Mock get_logger to return our mock
        with patch('temp_file_tracker.get_logger', return_value=self.mock_logger):
            tracker = TempFileTracker(config)

            temp_file = tracker.create_temp_file('.wav', purpose='test')

            # Logger should not be called for audit events
            # (Only the context manager enter/exit might call it)
            self.mock_logger.info.assert_not_called()

            # Cleanup
            os.unlink(temp_file)

    def test_custom_retry_attempts(self):
        """Test custom retry attempts configuration."""
        config = TempFileTrackerConfig(max_retry_attempts=5)
        tracker = TempFileTracker(config)

        self.assertEqual(tracker.config.max_retry_attempts, 5)

    def test_custom_timeout(self):
        """Test custom cleanup timeout."""
        config = TempFileTrackerConfig(cleanup_timeout_seconds=60)
        tracker = TempFileTracker(config)

        self.assertEqual(tracker.config.cleanup_timeout_seconds, 60)

    def test_custom_permissions(self):
        """Test custom directory permissions."""
        config = TempFileTrackerConfig(temp_dir_permissions=0o755)
        tracker = TempFileTracker(config)

        temp_dir = tracker.create_secure_temp_dir()

        # Check permissions
        stat_info = os.stat(temp_dir)
        expected_perms = 0o40755  # 0o755 with setuid/setgid bits
        self.assertEqual(stat_info.st_mode, expected_perms)

        # Cleanup
        os.rmdir(temp_dir)


class TestTempFileTrackerEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_logger = MagicMock()
        self.config = TempFileTrackerConfig()
        self.tracker = TempFileTracker(self.config, self.mock_logger)
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_cleanup_with_missing_files(self):
        """Test cleanup when some files are already deleted."""
        # Track files that don't exist
        self.tracker.tracked_files['/nonexistent/file1.wav'] = FileMetadata(
            '/nonexistent/file1.wav', datetime.now(), 0, 'test'
        )
        self.tracker.tracked_files['/nonexistent/file2.json'] = FileMetadata(
            '/nonexistent/file2.json', datetime.now(), 0, 'test'
        )

        self.tracker._perform_cleanup()

        # Should not crash, should log warnings
        self.mock_logger.warning.assert_called()
        self.assertTrue(self.tracker.was_cleanup_successful())

    def test_cleanup_with_permission_errors(self):
        """Test cleanup with permission errors."""
        # Create a file and make it read-only
        test_file = os.path.join(self.temp_dir, "readonly.wav")
        with open(test_file, 'wb') as f:
            f.write(b'test')

        # Make it read-only
        os.chmod(test_file, 0o444)

        self.tracker.tracked_files[test_file] = FileMetadata(
            test_file, datetime.now(), 4, 'test'
        )

        self.tracker._perform_cleanup()

        # Should eventually fail after retries
        self.assertFalse(self.tracker.was_cleanup_successful())
        self.mock_logger.error.assert_called()

    @patch('os.unlink')
    def test_file_deletion_interrupted(self, mock_unlink):
        """Test file deletion with intermittent failures."""
        # Make unlink fail twice, then succeed
        mock_unlink.side_effect = [OSError("Busy"), OSError("Busy"), None]

        test_file = os.path.join(self.temp_dir, "test.wav")
        Path(test_file).touch()

        metadata = FileMetadata(test_file, datetime.now(), 0, 'test')

        self.tracker._secure_delete_file(test_file, metadata)

        # Should have been called 3 times (2 failures + 1 success)
        self.assertEqual(mock_unlink.call_count, 3)
        self.mock_logger.warning.assert_called()
        self.mock_logger.info.assert_called_with(
            f"Successfully deleted file: {test_file} "
            "(purpose: test, size: 0 bytes, access_count: 0)"
        )

    def test_large_number_of_files(self):
        """Test handling large number of tracked files."""
        with patch('temp_file_tracker.get_logger', return_value=self.mock_logger):
            with TempFileTracker(self.config) as tracker:
                # Create many files
                files = []
                for i in range(100):
                    file_path = tracker.create_temp_file(f'.{i}', purpose=f'file_{i}')
                    files.append(file_path)

                self.assertEqual(tracker.get_file_count(), 100)

            # All should be cleaned up
            for file_path in files:
                self.assertFalse(os.path.exists(file_path))

    def test_concurrent_access_simulation(self):
        """Test simulated concurrent access to tracked files."""
        test_file = os.path.join(self.temp_dir, "test.wav")
        Path(test_file).touch()

        self.tracker.tracked_files[test_file] = FileMetadata(
            test_file, datetime.now(), 0, 'test'
        )

        # Simulate multiple access recordings
        for i in range(10):
            self.tracker.record_file_access(test_file)

        metadata = self.tracker.tracked_files[test_file]
        self.assertEqual(metadata.access_count, 10)
        self.assertIsNotNone(metadata.last_access_time)


class TestTempFileTrackerAuditLogging(unittest.TestCase):
    """Test audit logging functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_logger = MagicMock()
        self.config = TempFileTrackerConfig()
        self.tracker = TempFileTracker(self.config, self.mock_logger)
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_audit_log_file_creation(self):
        """Test that audit logs are created for file operations."""
        temp_file = self.tracker.create_temp_file('.wav', purpose='audit_test')

        self.mock_logger.info.assert_any_call(
            f"Created temporary file: {temp_file} (purpose: audit_test)"
        )

        # Cleanup
        os.unlink(temp_file)

    def test_audit_log_cleanup_operations(self):
        """Test audit logging during cleanup."""
        test_file = os.path.join(self.temp_dir, "test.wav")
        Path(test_file).touch()

        self.tracker.tracked_files[test_file] = FileMetadata(
            test_file, datetime.now(), 0, 'cleanup_test'
        )

        self.tracker._perform_cleanup()

        self.mock_logger.info.assert_any_call(
            "Starting cleanup of 1 files and 0 directories"
        )
        self.mock_logger.info.assert_any_call(
            f"Successfully deleted file: {test_file} "
            "(purpose: cleanup_test, size: 0 bytes, access_count: 0)"
        )

    def test_audit_log_error_conditions(self):
        """Test audit logging for error conditions."""
        # Try to track non-existent file
        try:
            self.tracker.track_existing_file('/nonexistent/file.wav')
        except FileNotFoundError:
            pass

        self.mock_logger.error.assert_called_with(
            "Failed to track existing file /nonexistent/file.wav: File does not exist: /nonexistent/file.wav"
        )

    def test_audit_log_context_manager(self):
        """Test audit logging in context manager."""
        with self.tracker:
            pass

        self.mock_logger.info.assert_any_call("TempFileTracker session started")
        self.mock_logger.info.assert_any_call("TempFileTracker session ended. Cleanup successful: True")


if __name__ == '__main__':
    unittest.main()
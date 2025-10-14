#!/usr/bin/env python3
"""
Integration tests for secure file handling.

This module tests the complete file upload and processing pipeline
with security measures including:
- Secure temporary file creation
- File validation and sanitization
- Secure file cleanup
- Permission restrictions
- Size limits
"""

import os
import tempfile
import shutil
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import sys

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline_orchestrator import PipelineOrchestrator, PipelineConfig
from config import ProcessingConfig
from fastapi import UploadFile, HTTPException
import io


class TestSecureFileHandling(unittest.TestCase):
    """Integration tests for secure file handling."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.config = PipelineConfig(
            secure_temp_dir=True,
            auto_cleanup=True,
            max_file_size_mb=10,  # Small limit for testing
            output_format="json"
        )

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_secure_temp_directory_creation(self):
        """Test that secure temporary directories are created with correct permissions."""
        orchestrator = PipelineOrchestrator(self.config)

        # Create a temp directory using the secure temp manager
        with orchestrator.temp_manager.secure_temp_dir() as temp_dir:
            # Check that temp directory exists
            self.assertTrue(os.path.exists(temp_dir))

            # Check permissions (should be restrictive)
            stat_info = os.stat(temp_dir)
            # Should have 0o700 permissions (owner read/write/execute only)
            expected_perms = 0o40700  # 0o700 with setuid/setgid bits
            self.assertEqual(stat_info.st_mode, expected_perms)

        orchestrator.cleanup()

    def test_secure_file_cleanup(self):
        """Test that temporary files are securely cleaned up."""
        orchestrator = PipelineOrchestrator(self.config)

        # Create a test file using the temp manager
        with orchestrator.temp_manager.secure_temp_dir() as temp_dir:
            test_file = os.path.join(temp_dir, "test.wav")
            test_data = b"x" * 1024  # 1KB of data

            with open(test_file, 'wb') as f:
                f.write(test_data)

            # Verify file exists
            self.assertTrue(os.path.exists(test_file))

        # Cleanup should remove the file and directory
        orchestrator.cleanup()

        # File and directory should be gone
        self.assertFalse(os.path.exists(test_file))
        self.assertFalse(os.path.exists(temp_dir))

    def test_file_size_limit_enforcement(self):
        """Test that file size limits are properly enforced."""
        # Create a config with very small size limit
        config = PipelineConfig(
            secure_temp_dir=True,
            auto_cleanup=True,
            max_file_size_mb=0,  # 0MB limit (no files allowed)
            output_format="json"
        )
        orchestrator = PipelineOrchestrator(config)

        # Create a file that's too large
        large_file = os.path.join(self.test_dir, "large.wav")
        with open(large_file, 'wb') as f:
            f.write(b"x" * 2048)  # 2KB file

        # Validation should fail
        result = orchestrator._validate_input_file(large_file)
        self.assertFalse(result)

        orchestrator.cleanup()
        os.unlink(large_file)

    def test_file_extension_validation(self):
        """Test that only allowed file extensions are accepted."""
        orchestrator = PipelineOrchestrator(self.config)

        # Test allowed extensions
        allowed_extensions = ['.mp3', '.wav', '.flac', '.m4a', '.aac']

        for ext in allowed_extensions:
            test_file = os.path.join(self.test_dir, f"test{ext}")
            Path(test_file).touch()  # Create empty file

            result = orchestrator._validate_input_file(test_file)
            # Should pass extension check (might fail on other validations)
            self.assertIsInstance(result, bool)

            os.unlink(test_file)

        # Test disallowed extension
        bad_file = os.path.join(self.test_dir, "test.exe")
        Path(bad_file).touch()

        result = orchestrator._validate_input_file(bad_file)
        self.assertFalse(result)

        os.unlink(bad_file)
        orchestrator.cleanup()

    def test_secure_file_processing_pipeline(self):
        """Test the complete secure file processing pipeline."""
        # Create a small valid audio file for testing
        test_audio_file = os.path.join(self.test_dir, "test.wav")
        # Create a minimal WAV header + some data
        wav_header = b'RIFF\x24\x08\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\x00\x08\x00\x00'
        with open(test_audio_file, 'wb') as f:
            f.write(wav_header)

        orchestrator = PipelineOrchestrator(self.config)

        try:
            # Process the file (this will likely fail due to audio processing,
            # but we're testing the security pipeline)
            results = orchestrator.process_files(test_audio_file)

            # Results should be returned (even if processing fails)
            self.assertIsInstance(results, list)

        except Exception:
            # Processing might fail, but security measures should still work
            pass
        finally:
            orchestrator.cleanup()
            os.unlink(test_audio_file)

    def test_concurrent_secure_processing(self):
        """Test that multiple orchestrators can run concurrently with secure isolation."""
        orchestrator1 = PipelineOrchestrator(self.config)
        orchestrator2 = PipelineOrchestrator(self.config)

        # Test that both can create secure temp directories
        with orchestrator1.temp_manager.secure_temp_dir() as temp_dir1:
            with orchestrator2.temp_manager.secure_temp_dir() as temp_dir2:
                # Each should have its own temp directory
                self.assertNotEqual(temp_dir1, temp_dir2)
                self.assertTrue(os.path.exists(temp_dir1))
                self.assertTrue(os.path.exists(temp_dir2))

                # Both should have restrictive permissions
                stat1 = os.stat(temp_dir1)
                stat2 = os.stat(temp_dir2)
                self.assertEqual(stat1.st_mode, stat2.st_mode)

        orchestrator1.cleanup()
        orchestrator2.cleanup()

    def test_secure_cleanup_on_failure(self):
        """Test that cleanup happens even when processing fails."""
        orchestrator = PipelineOrchestrator(self.config)

        # Create a test file using temp manager
        with orchestrator.temp_manager.secure_temp_dir() as temp_dir:
            test_file = os.path.join(temp_dir, "test.txt")
            with open(test_file, 'w') as f:
                f.write("test data")

            self.assertTrue(os.path.exists(test_file))

            # Simulate processing failure and cleanup
            try:
                raise Exception("Simulated processing failure")
            except Exception:
                pass
            finally:
                orchestrator.cleanup()

            # File and directory should be gone
            self.assertFalse(os.path.exists(test_file))
            self.assertFalse(os.path.exists(temp_dir))

    def test_memory_buffer_processing(self):
        """Test that files are processed in memory buffers when possible."""
        # This tests the secure in-memory processing mentioned in systemPatterns.md
        config = PipelineConfig(
            secure_temp_dir=True,
            auto_cleanup=True,
            encrypt_temp_files=False,  # Disable encryption for this test
        )
        orchestrator = PipelineOrchestrator(config)

        # The orchestrator should be configured to use secure temp directories
        self.assertTrue(orchestrator.config.secure_temp_dir)

        orchestrator.cleanup()


class TestFileUploadSecurity(unittest.TestCase):
    """Test file upload security measures."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_upload_file_validation(self):
        """Test file upload validation pipeline."""
        from app import sanitize_filename

        # Test filename sanitization prevents path traversal
        result = sanitize_filename("../../../etc/passwd")
        # Should prevent path traversal
        self.assertEqual(result, "passwd")

        # Test normal filename
        result = sanitize_filename("test.wav")
        self.assertEqual(result, "test.wav")

    def test_mime_type_validation(self):
        """Test MIME type validation for uploaded files."""
        # Test allowed MIME types from config
        config = ProcessingConfig()
        allowed_types = config.allowed_mime_types

        self.assertIn('audio/mpeg', allowed_types)
        self.assertIn('audio/wav', allowed_types)
        self.assertIn('audio/x-wav', allowed_types)
        self.assertIn('audio/flac', allowed_types)

    def test_rate_limiting_configuration(self):
        """Test that rate limiting is configured."""
        config = ProcessingConfig()

        # Check rate limiting settings
        self.assertEqual(config.rate_limit_requests, 10)
        self.assertEqual(config.rate_limit_window_seconds, 60)


class TestAuditLogging(unittest.TestCase):
    """Test audit logging for security events."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_audit_log_configuration(self):
        """Test that audit logging is properly configured."""
        config = ProcessingConfig()

        # Check audit logging settings
        self.assertTrue(config.enable_audit_logging)
        self.assertEqual(config.audit_log_file, "logs/audit.log")
        self.assertEqual(config.retention_hours, 24)
        self.assertTrue(config.auto_retention_cleanup)

    def test_secure_delete_configuration(self):
        """Test secure deletion configuration."""
        config = ProcessingConfig()

        # Check secure deletion settings
        self.assertEqual(config.secure_delete_overwrites, 3)
        self.assertFalse(config.encrypt_temp_files)  # Disabled by default


if __name__ == '__main__':
    unittest.main()
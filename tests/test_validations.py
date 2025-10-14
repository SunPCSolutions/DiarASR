#!/usr/bin/env python3
"""
Security-focused unit tests for input validation functions.

This module tests all security validation functions including:
- Environment variable sanitization
- Filename sanitization
- Parameter sanitization
- File validation
- Input length limits
"""

import os
import tempfile
import unittest
from unittest.mock import patch, mock_open
import sys

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    sanitize_env_value,
    get_secure_env_var,
    validate_environment,
    SecurityConfig,
    ProcessingConfig
)
from app import sanitize_filename, sanitize_parameter, validate_file_extension
from pipeline_orchestrator import PipelineOrchestrator
from fastapi import HTTPException


class TestEnvironmentValidation(unittest.TestCase):
    """Test environment variable validation and sanitization."""

    def test_sanitize_env_value_valid_input(self):
        """Test sanitization of valid environment variable values."""
        # Test normal alphanumeric input
        result = sanitize_env_value("test_value_123")
        self.assertEqual(result, "test_value_123")

        # Test input with allowed special characters
        result = sanitize_env_value("path/to/file.txt")
        self.assertEqual(result, "path/to/file.txt")

        # Test input with spaces and hyphens
        result = sanitize_env_value("my-api-key-123")
        self.assertEqual(result, "my-api-key-123")

    def test_sanitize_env_value_invalid_input(self):
        """Test sanitization of invalid environment variable values."""
        # Test input with dangerous characters
        result = sanitize_env_value("value;rm -rf /")
        self.assertEqual(result, "valuerm -rf /")

        # Test input with SQL injection attempts
        result = sanitize_env_value("value' OR '1'='1")
        self.assertEqual(result, "value OR 11")

    def test_sanitize_env_value_length_limits(self):
        """Test length limits for environment variable values."""
        # Test maximum length
        long_value = "a" * 1000
        result = sanitize_env_value(long_value)
        self.assertEqual(result, long_value)

        # Test exceeding maximum length
        with self.assertRaises(ValueError):
            sanitize_env_value("a" * 1001)

    def test_sanitize_env_value_non_string_input(self):
        """Test handling of non-string input."""
        with self.assertRaises(ValueError):
            sanitize_env_value(123)  # type: ignore

        with self.assertRaises(ValueError):
            sanitize_env_value(None)  # type: ignore

    def test_get_secure_env_var_with_docker_secret(self):
        """Test getting environment variable with Docker secret fallback."""
        with patch('config.read_docker_secret') as mock_secret:
            mock_secret.return_value = "secret_value"
            result = get_secure_env_var("TEST_VAR", "default")
            self.assertEqual(result, "secret_value")
            mock_secret.assert_called_once_with("TEST_VAR")

    def test_get_secure_env_var_with_env_var(self):
        """Test getting environment variable from environment."""
        with patch('config.read_docker_secret') as mock_secret, \
             patch.dict(os.environ, {'TEST_VAR': 'env_value'}):
            mock_secret.return_value = None
            result = get_secure_env_var("TEST_VAR", "default")
            self.assertEqual(result, "env_value")

    def test_get_secure_env_var_with_default(self):
        """Test getting default value when variable not found."""
        with patch('config.read_docker_secret') as mock_secret, \
             patch.dict(os.environ, {}, clear=True):
            mock_secret.return_value = None
            result = get_secure_env_var("TEST_VAR", "default")
            self.assertEqual(result, "default")

    def test_get_secure_env_var_required_missing(self):
        """Test error when required variable is missing."""
        with patch('config.read_docker_secret') as mock_secret, \
             patch.dict(os.environ, {}, clear=True):
            mock_secret.return_value = None
            with self.assertRaises(ValueError):
                get_secure_env_var("TEST_VAR", required=True)

    def test_validate_environment_basic(self):
        """Test basic environment validation."""
        with patch('config.get_secure_env_var') as mock_get_var:
            mock_get_var.return_value = "test_value"
            result = validate_environment()
            self.assertIsInstance(result, dict)


class TestFilenameValidation(unittest.TestCase):
    """Test filename sanitization and validation."""

    def test_sanitize_filename_valid_input(self):
        """Test sanitization of valid filenames."""
        # Test normal filename
        result = sanitize_filename("test_file.mp3")
        self.assertEqual(result, "test_file.mp3")

        # Test filename with spaces
        result = sanitize_filename("my test file.wav")
        self.assertEqual(result, "my test file.wav")

        # Test filename with allowed special characters
        result = sanitize_filename("file-name_123.flac")
        self.assertEqual(result, "file-name_123.flac")

    def test_sanitize_filename_path_traversal(self):
        """Test prevention of path traversal attacks."""
        # Test path traversal attempt
        result = sanitize_filename("../../../etc/passwd")
        self.assertEqual(result, "passwd")

        # Test absolute path
        result = sanitize_filename("/etc/passwd")
        self.assertEqual(result, "passwd")

    def test_sanitize_filename_invalid_characters(self):
        """Test rejection of invalid filename characters."""
        # Test dangerous characters
        with self.assertRaises(HTTPException):
            sanitize_filename("file<script>.mp3")

        with self.assertRaises(HTTPException):
            sanitize_filename("file|command.mp3")

    def test_sanitize_filename_length_limits(self):
        """Test filename length limits."""
        # Test maximum length
        long_name = "a" * 255
        result = sanitize_filename(long_name)
        self.assertEqual(result, long_name)

        # Test exceeding maximum length
        with self.assertRaises(HTTPException):
            sanitize_filename("a" * 256)

    def test_sanitize_filename_empty_input(self):
        """Test handling of empty filename."""
        with self.assertRaises(HTTPException):
            sanitize_filename("")

        with self.assertRaises(HTTPException):
            sanitize_filename(None)  # type: ignore

    def test_validate_file_extension_valid(self):
        """Test validation of valid file extensions."""
        # Test allowed extensions
        validate_file_extension("test.mp3")
        validate_file_extension("test.wav")
        validate_file_extension("test.flac")
        validate_file_extension("test.m4a")
        validate_file_extension("test.aac")

    def test_validate_file_extension_invalid(self):
        """Test rejection of invalid file extensions."""
        with self.assertRaises(HTTPException):
            validate_file_extension("test.exe")

        with self.assertRaises(HTTPException):
            validate_file_extension("test.txt")

        with self.assertRaises(HTTPException):
            validate_file_extension("test")


class TestParameterValidation(unittest.TestCase):
    """Test parameter sanitization."""

    def test_sanitize_parameter_valid_input(self):
        """Test sanitization of valid parameters."""
        # Test normal parameter
        result = sanitize_parameter("test_value")
        self.assertEqual(result, "test_value")

        # Test parameter with spaces
        result = sanitize_parameter("test value")
        self.assertEqual(result, "test value")

        # Test parameter with allowed special characters
        result = sanitize_parameter("test-value_123")
        self.assertEqual(result, "test-value_123")

    def test_sanitize_parameter_dangerous_input(self):
        """Test sanitization of dangerous parameter input."""
        # Test script injection attempt
        result = sanitize_parameter("value<script>")
        self.assertEqual(result, "valuescript")

        # Test SQL injection attempt
        result = sanitize_parameter("value' OR '1'='1")
        self.assertEqual(result, "value OR 11")

    def test_sanitize_parameter_length_limits(self):
        """Test parameter length limits."""
        # Test maximum length
        long_param = "a" * 1000
        result = sanitize_parameter(long_param)
        self.assertEqual(result, long_param)

        # Test exceeding maximum length
        with self.assertRaises(HTTPException):
            sanitize_parameter("a" * 1001)

    def test_sanitize_parameter_non_string_input(self):
        """Test handling of non-string parameter input."""
        # Non-string input should be returned as-is
        result = sanitize_parameter(123)  # type: ignore
        self.assertEqual(result, 123)

        result = sanitize_parameter(None)  # type: ignore
        self.assertEqual(result, None)


class TestFileValidation(unittest.TestCase):
    """Test file validation in pipeline orchestrator."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = ProcessingConfig()
        self.orchestrator = PipelineOrchestrator()

    def test_validate_input_file_exists(self):
        """Test file existence validation."""
        # Test with non-existent file
        result = self.orchestrator._validate_input_file("non_existent_file.mp3")
        self.assertFalse(result)

    def test_validate_input_file_extension(self):
        """Test file extension validation."""
        # Create temporary file with invalid extension
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            temp_file = f.name

        try:
            result = self.orchestrator._validate_input_file(temp_file)
            self.assertFalse(result)
        finally:
            os.unlink(temp_file)

    def test_validate_input_file_size(self):
        """Test file size validation."""
        # Create small test file
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
            f.write(b'x' * 100)  # Small file
            temp_file = f.name

        try:
            result = self.orchestrator._validate_input_file(temp_file)
            # Should pass basic validation (existence and extension)
            # Size check depends on config.max_file_size_mb
            self.assertIsInstance(result, bool)
        finally:
            os.unlink(temp_file)

    def test_validate_input_file_valid(self):
        """Test validation of valid file."""
        # Create valid test file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            f.write(b'x' * 1000)  # Small valid file
            temp_file = f.name

        try:
            result = self.orchestrator._validate_input_file(temp_file)
            self.assertTrue(result)
        finally:
            os.unlink(temp_file)


class TestSecurityConfig(unittest.TestCase):
    """Test security configuration validation."""

    def test_security_config_defaults(self):
        """Test default security configuration values."""
        config = SecurityConfig()

        self.assertTrue(config.enable_api_key_auth)
        self.assertEqual(config.api_key_header, "X-API-Key")
        self.assertTrue(config.cors_enabled)
        self.assertTrue(config.security_headers_enabled)
        self.assertTrue(config.sanitize_inputs)
        self.assertEqual(config.max_filename_length, 255)
        self.assertEqual(config.max_parameter_length, 1000)

    def test_security_config_validation(self):
        """Test security configuration parameter validation."""
        # Test valid configuration
        config = SecurityConfig(
            enable_api_key_auth=True,
            max_filename_length=100,
            max_parameter_length=500
        )

        self.assertEqual(config.max_filename_length, 100)
        self.assertEqual(config.max_parameter_length, 500)


if __name__ == '__main__':
    unittest.main()
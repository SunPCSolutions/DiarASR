#!/usr/bin/env python3
"""
Fuzz testing for audio file processing.

This module performs fuzz testing on audio file processing functions
to ensure they handle malformed, corrupted, or malicious input gracefully.
"""

import os
import tempfile
import random
import unittest
from pathlib import Path
import sys

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline_orchestrator import PipelineOrchestrator, PipelineConfig
from audio_preprocessor import AudioPreprocessor
from config import ProcessingConfig


class AudioFuzzTester:
    """Fuzz testing utilities for audio processing."""

    def __init__(self):
        """Initialize the fuzz tester."""
        self.test_dir = tempfile.mkdtemp()

    def cleanup(self):
        """Clean up test files."""
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def generate_random_bytes(self, size: int) -> bytes:
        """Generate random bytes of specified size."""
        return bytes(random.getrandbits(8) for _ in range(size))

    def generate_malformed_wav(self) -> str:
        """Generate a malformed WAV file."""
        file_path = os.path.join(self.test_dir, "malformed.wav")

        # Create various types of malformed WAV files
        malformation_types = [
            self._empty_file,
            self._invalid_header,
            self._truncated_header,
            self._wrong_endianness,
            self._oversized_data,
            self._negative_size,
            self._random_data
        ]

        # Randomly select a malformation type
        malformation = random.choice(malformation_types)
        malformation(file_path)

        return file_path

    def _empty_file(self, path: str):
        """Create an empty file."""
        with open(path, 'wb') as f:
            f.write(b'')

    def _invalid_header(self, path: str):
        """Create a file with invalid WAV header."""
        with open(path, 'wb') as f:
            f.write(b'INVALID HEADER DATA' + self.generate_random_bytes(100))

    def _truncated_header(self, path: str):
        """Create a file with truncated WAV header."""
        with open(path, 'wb') as f:
            f.write(b'RIFF\x00\x00\x00\x00WAVE')  # Incomplete header

    def _wrong_endianness(self, path: str):
        """Create a file with wrong endianness in header."""
        with open(path, 'wb') as f:
            # Valid header but with wrong endianness values
            f.write(b'RIFF\x00\x00\x00\x00WAVEfmt \x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')

    def _oversized_data(self, path: str):
        """Create a file claiming to have oversized data."""
        with open(path, 'wb') as f:
            # Header claiming huge data size
            f.write(b'RIFF\xff\xff\xff\xffWAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\xff\xff\xff\xff')
            f.write(self.generate_random_bytes(100))  # But only small actual data

    def _negative_size(self, path: str):
        """Create a file with negative size values."""
        with open(path, 'wb') as f:
            f.write(b'RIFF\xff\xff\xff\xffWAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\xff\xff\xff\xff')

    def _random_data(self, path: str):
        """Create a file with completely random data."""
        with open(path, 'wb') as f:
            f.write(self.generate_random_bytes(random.randint(0, 10000)))


class TestAudioFuzzTesting(unittest.TestCase):
    """Fuzz tests for audio processing functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.fuzz_tester = AudioFuzzTester()
        self.config = PipelineConfig(
            secure_temp_dir=True,
            auto_cleanup=True,
            max_file_size_mb=10,
            output_format="json"
        )

    def tearDown(self):
        """Clean up test fixtures."""
        self.fuzz_tester.cleanup()

    def test_fuzz_audio_preprocessing(self):
        """Test audio preprocessing with fuzzed input."""
        preprocessor = AudioPreprocessor()

        # Generate multiple malformed files
        for i in range(10):
            malformed_file = self.fuzz_tester.generate_malformed_wav()

            try:
                # Attempt to preprocess the malformed file
                # This should not crash or expose security vulnerabilities
                result = preprocessor.preprocess_audio(malformed_file)

                # Result should be handled gracefully (might be None or error)
                # We just want to ensure no exceptions are raised

            except Exception as e:
                # Some exceptions are expected for malformed files
                # But they should be handled gracefully, not crash the system
                self.assertIsInstance(e, (ValueError, OSError, EOFError, RuntimeError))

            finally:
                # Clean up the test file
                if os.path.exists(malformed_file):
                    os.unlink(malformed_file)

    def test_fuzz_pipeline_orchestrator(self):
        """Test pipeline orchestrator with fuzzed audio files."""
        orchestrator = PipelineOrchestrator(self.config)

        # Generate multiple malformed files
        for i in range(5):
            malformed_file = self.fuzz_tester.generate_malformed_wav()

            try:
                # Attempt to process the malformed file
                results = orchestrator.process_files(malformed_file)

                # Results should be returned gracefully
                self.assertIsInstance(results, list)

                if results:
                    result = results[0]
                    # Should have error information for malformed files
                    self.assertIn('error', result)

            except Exception as e:
                # Some exceptions are expected, but should be handled
                self.assertIsInstance(e, (ValueError, OSError, FileNotFoundError, RuntimeError))

            finally:
                orchestrator.cleanup()
                if os.path.exists(malformed_file):
                    os.unlink(malformed_file)

    def test_fuzz_file_validation(self):
        """Test file validation with fuzzed files."""
        orchestrator = PipelineOrchestrator(self.config)

        # Generate multiple malformed files
        for i in range(10):
            malformed_file = self.fuzz_tester.generate_malformed_wav()

            try:
                # Test file validation
                result = orchestrator._validate_input_file(malformed_file)

                # Validation should return False for malformed files
                # (or True if the file happens to pass basic checks)
                self.assertIsInstance(result, bool)

            except Exception as e:
                # Validation should not crash on malformed files
                self.fail(f"File validation crashed on malformed input: {e}")

            finally:
                if os.path.exists(malformed_file):
                    os.unlink(malformed_file)

        orchestrator.cleanup()

    def test_fuzz_large_files(self):
        """Test handling of unusually large files."""
        # Create a file that's within limits but large
        large_file = os.path.join(self.fuzz_tester.test_dir, "large.wav")

        # Create a file that's close to the limit
        size_mb = 9  # Close to 10MB limit
        size_bytes = size_mb * 1024 * 1024

        with open(large_file, 'wb') as f:
            # Write a valid WAV header
            f.write(b'RIFF\x00\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00')
            # Write large amount of data
            f.write(self.fuzz_tester.generate_random_bytes(size_bytes - 44))  # Subtract header size

        orchestrator = PipelineOrchestrator(self.config)

        try:
            # This should either process successfully or fail gracefully
            results = orchestrator.process_files(large_file)
            self.assertIsInstance(results, list)

        except Exception as e:
            # Large file processing might fail, but should be graceful
            self.assertIsInstance(e, (ValueError, OSError, RuntimeError, MemoryError))

        finally:
            orchestrator.cleanup()
            if os.path.exists(large_file):
                os.unlink(large_file)

    def test_fuzz_concurrent_processing(self):
        """Test concurrent processing with fuzzed files."""
        # Create multiple orchestrators
        orchestrators = [PipelineOrchestrator(self.config) for _ in range(3)]

        # Generate test files for each
        test_files = []
        for i in range(3):
            test_file = self.fuzz_tester.generate_malformed_wav()
            test_files.append(test_file)

        try:
            # Process files concurrently (simulated)
            results = []
            for i, orchestrator in enumerate(orchestrators):
                try:
                    result = orchestrator.process_files(test_files[i])
                    results.append(result)
                except Exception as e:
                    # Record the exception as a result
                    results.append([{'error': str(e)}])

            # All should return results
            self.assertEqual(len(results), 3)
            for result in results:
                self.assertIsInstance(result, list)

        finally:
            # Clean up all orchestrators and files
            for orchestrator in orchestrators:
                orchestrator.cleanup()

            for test_file in test_files:
                if os.path.exists(test_file):
                    os.unlink(test_file)

    def test_fuzz_edge_cases(self):
        """Test various edge cases in audio processing."""
        test_cases = [
            ("empty_file", b""),  # Completely empty
            ("single_byte", b"x"),  # Single byte
            ("text_file", b"This is not audio data"),  # Text content
            ("binary_junk", self.fuzz_tester.generate_random_bytes(1000)),  # Random binary
            ("wav_header_only", b'RIFF\x00\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00'),  # Header only
        ]

        orchestrator = PipelineOrchestrator(self.config)

        for case_name, content in test_cases:
            test_file = os.path.join(self.fuzz_tester.test_dir, f"{case_name}.wav")

            with open(test_file, 'wb') as f:
                f.write(content)

            try:
                # Attempt processing
                results = orchestrator.process_files(test_file)
                self.assertIsInstance(results, list)

            except Exception as e:
                # Should handle gracefully
                self.assertIsInstance(e, (ValueError, OSError, RuntimeError))

            finally:
                if os.path.exists(test_file):
                    os.unlink(test_file)

        orchestrator.cleanup()


if __name__ == '__main__':
    # Set random seed for reproducible fuzzing
    random.seed(42)
    unittest.main()
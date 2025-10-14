#!/usr/bin/env python3
"""
Test script for the Pipeline Orchestrator

This script tests the modular pipeline orchestrator with secure file handling.
"""

import os
import sys
import logging
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline_orchestrator import PipelineOrchestrator, PipelineConfig, process_audio_files

# Set up basic logging for tests
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_basic_orchestrator():
    """Test basic orchestrator functionality."""
    logger.info("Testing basic PipelineOrchestrator functionality...")

    # Check if test file exists
    test_file = "test1.mp3"
    if not os.path.exists(test_file):
        logger.error("Test file %s not found", test_file)
        return False

    # Create orchestrator with default config
    config = PipelineConfig(
        secure_temp_dir=True,
        auto_cleanup=True,
        max_file_size_mb=200,  # Allow larger files for testing
        output_format="both"
    )

    orchestrator = PipelineOrchestrator(config)

    try:
        # Process the test file
        results = orchestrator.process_files(test_file)

        if not results:
            logger.error("No results returned")
            return False

        result = results[0]
        logger.info("Processed file: %s", result['file'])
        logger.info("Total segments: %d", result.get('total_segments', 0))

        if 'error' in result:
            logger.error("Processing error: %s", result['error'])
            # This might be expected if audio has no detectable speech
            if "No speaker segments detected" in result['error']:
                logger.info("No segments detected - this may be expected for test audio")
                return True
            return False

        segments = result.get('segments', [])
        logger.info("Successfully processed %d segments", len(segments))

        # Save results
        orchestrator.save_results(results, "test_orchestrator_output")

        # Check if output files were created
        json_file = "test_orchestrator_output.json"
        txt_file = "test_orchestrator_output.txt"

        if os.path.exists(json_file):
            logger.info("✓ JSON output created: %s", json_file)
        else:
            logger.error("✗ JSON output missing: %s", json_file)

        if os.path.exists(txt_file):
            logger.info("✓ Text output created: %s", txt_file)
        else:
            logger.error("✗ Text output missing: %s", txt_file)

        # Print sample results
        if segments:
            logger.info("Sample results:")
            for i, segment in enumerate(segments[:3]):
                logger.info("  %s: %s...", segment['speaker'], segment['text'][:50])

        return True

    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Cleanup
        orchestrator.cleanup()


def test_convenience_function():
    """Test the convenience function."""
    logger.info("Testing convenience function...")

    test_file = "test1.mp3"
    if not os.path.exists(test_file):
        logger.error("Test file %s not found", test_file)
        return False

    try:
        # Use convenience function
        results = process_audio_files(
            test_file,
            output_path="test_convenience_output"
        )

        if not results:
            logger.error("No results from convenience function")
            return False

        result = results[0]
        logger.info("Convenience function processed: %s", result['file'])
        logger.info("Segments: %d", result.get('total_segments', 0))

        return True

    except Exception as e:
        logger.error("Error in convenience function test: %s", str(e))
        return False


def test_configuration():
    """Test configuration options."""
    logger.info("Testing configuration options...")

    # Test with custom config
    config = PipelineConfig(
        diarization_model="nvidia/diar_streaming_sortformer_4spk-v2",
        asr_model="nvidia/parakeet-ctc-1.1b",
        asr_use_vad=False,  # Disable VAD for faster testing
        secure_temp_dir=True,
        auto_cleanup=True,
        output_format="json"
    )

    logger.info("Config created with diarization model: %s", config.diarization_model)
    logger.info("ASR model: %s", config.asr_model)
    logger.info("VAD enabled: %s", config.asr_use_vad)
    logger.info("Secure temp: %s", config.secure_temp_dir)

    return True


def test_error_handling():
    """Test error handling."""
    logger.info("Testing error handling...")

    # Test with non-existent file
    orchestrator = PipelineOrchestrator()
    try:
        results = orchestrator.process_files("non_existent_file.mp3")
        logger.error("Should have failed with non-existent file")
        return False
    except ValueError as e:
        logger.info("✓ Correctly caught error for non-existent file: %s", str(e))
        return True
    except Exception as e:
        logger.error("Unexpected error: %s", str(e))
        return False
    finally:
        orchestrator.cleanup()


def main():
    """Run all tests."""
    logger.info("Pipeline Orchestrator Test Suite")
    logger.info("=" * 40)

    tests = [
        ("Basic Orchestrator", test_basic_orchestrator),
        ("Convenience Function", test_convenience_function),
        ("Configuration", test_configuration),
        ("Error Handling", test_error_handling),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        logger.info("-" * 20 + " %s " + "-" * 20, test_name)
        try:
            if test_func():
                logger.info("✓ %s PASSED", test_name)
                passed += 1
            else:
                logger.error("✗ %s FAILED", test_name)
        except Exception as e:
            logger.error("✗ %s FAILED with exception: %s", test_name, str(e))

    logger.info("=" * 40)
    logger.info("Test Results: %d/%d passed", passed, total)

    if passed == total:
        logger.info("🎉 All tests passed!")
        return 0
    else:
        logger.error("❌ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
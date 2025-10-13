#!/usr/bin/env python3
"""
Test script for the new NVIDIA ASR module with VAD support.
"""

import os
import torch
import torchaudio
from nvidia_asr import NvidiaASR

def test_asr_module():
    """Test the NvidiaASR module functionality."""
    print("Testing NVIDIA ASR module...")

    # Initialize ASR with VAD
    asr = NvidiaASR(
        asr_model_name="nvidia/parakeet-ctc-1.1b",
        use_vad=True,
        vad_threshold=0.3,  # Lower threshold for testing
        min_segment_duration=0.05,
        enable_batch_processing=True,
        batch_size=4
    )

    try:
        # Test with a short audio segment
        print("Creating test audio...")

        # Generate a simple test audio (1 second of silence + 1 second of noise)
        sample_rate = 16000
        duration = 2  # seconds
        num_samples = duration * sample_rate

        # Create audio with some noise (simulating speech)
        audio_data = torch.randn(1, num_samples) * 0.1  # Low amplitude noise

        # Save test audio
        test_audio_path = "test_asr_audio.wav"
        torchaudio.save(test_audio_path, audio_data, sample_rate)

        print(f"Test audio saved: {test_audio_path}")

        # Test ASR transcription
        print("Testing ASR transcription...")
        result = asr.transcribe_file(test_audio_path)

        print("Transcription result:")
        print(f"  File: {result['file']}")
        print(".2f")
        print(f"  Segments: {len(result['segments'])}")

        for i, segment in enumerate(result['segments']):
            print(f"    Segment {i+1}: {segment['start']:.2f}s - {segment['end']:.2f}s")
            print(f"      Text: '{segment['text']}'")

        # Test without VAD
        print("\nTesting ASR without VAD...")
        asr_no_vad = NvidiaASR(
            asr_model_name="nvidia/parakeet-ctc-1.1b",
            use_vad=False,
            enable_batch_processing=True
        )

        result_no_vad = asr_no_vad.transcribe_file(test_audio_path)
        print("Result without VAD:")
        print(f"  Segments: {len(result_no_vad['segments'])}")
        for i, segment in enumerate(result_no_vad['segments']):
            print(f"    Segment {i+1}: {segment['start']:.2f}s - {segment['end']:.2f}s")
            print(f"      Text: '{segment['text']}'")

        asr_no_vad.cleanup()

        print("\nTest completed successfully!")

    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Cleanup
        asr.cleanup()
        if os.path.exists(test_audio_path):
            os.remove(test_audio_path)
            print(f"Cleaned up {test_audio_path}")

if __name__ == "__main__":
    test_asr_module()
#!/usr/bin/env python3
"""
Test script for the updated FastAPI endpoint with VAD support
"""

import requests
import os
import sys

def test_api_endpoint():
    """Test the API endpoint with different parameter combinations."""

    # Check if test file exists
    test_file = "test1.mp3"
    if not os.path.exists(test_file):
        print(f"Error: Test file {test_file} not found")
        return False

    # API endpoint URL (assuming the app runs on localhost:8000)
    base_url = "http://localhost:8000"

    # Test cases
    test_cases = [
        {"name": "ASR only", "params": {"diarize": False, "vad": False}},
        {"name": "ASR with VAD", "params": {"diarize": False, "vad": True}},
        {"name": "Diarization only", "params": {"diarize": True, "vad": False}},
        {"name": "Diarization with VAD", "params": {"diarize": True, "vad": True}},
    ]

    for test_case in test_cases:
        print(f"\nTesting: {test_case['name']}")

        try:
            # Prepare the request
            with open(test_file, 'rb') as f:
                files = {'audio_file': (test_file, f, 'audio/mpeg')}
                data = test_case['params']

                # Make the request
                response = requests.post(
                    f"{base_url}/transcribe_diarize/",
                    files=files,
                    data=data,
                    timeout=300  # 5 minute timeout
                )

            if response.status_code == 200:
                result = response.json()
                segments = result.get('segments', [])
                print(f"✓ Success: {len(segments)} segments returned")

                if segments:
                    print("Sample segments:")
                    for i, segment in enumerate(segments[:2]):
                        speaker = segment.get('speaker', 'Unknown')
                        text = segment.get('text', '')[:50]
                        start = segment.get('start', 0)
                        end = segment.get('end', 0)
                        print(f"  {speaker}: {start:.1f}s-{end:.1f}s: {text}...")

            else:
                print(f"✗ Failed with status {response.status_code}: {response.text}")

        except requests.exceptions.RequestException as e:
            print(f"✗ Request failed: {e}")
        except Exception as e:
            print(f"✗ Unexpected error: {e}")

    return True

def main():
    """Main test function."""
    print("Testing FastAPI endpoint with VAD support")
    print("=" * 50)

    # Check if the API is running
    try:
        response = requests.get("http://localhost:8000/docs", timeout=5)
        if response.status_code != 200:
            print("Warning: API docs not accessible. Make sure the FastAPI app is running.")
    except:
        print("Error: Cannot connect to API. Please start the FastAPI app first.")
        print("Run: uvicorn app:app --host 0.0.0.0 --port 8000")
        return 1

    # Run tests
    test_api_endpoint()

    print("\n" + "=" * 50)
    print("Testing completed. Check the output above for results.")

    return 0

if __name__ == "__main__":
    sys.exit(main())
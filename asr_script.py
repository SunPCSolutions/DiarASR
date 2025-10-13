#!/usr/bin/env python3

import nemo.collections.asr as nemo_asr
from pydub import AudioSegment
import os

def main():
    try:
        # Check if test.mp3 exists
        if not os.path.exists('test.mp3'):
            print("Error: test.mp3 not found")
            return

        # Convert mp3 to 16kHz mono wav
        print("Converting test.mp3 to WAV format...")
        audio = AudioSegment.from_mp3("test.mp3")
        audio = audio.set_channels(1).set_frame_rate(16000)
        audio.export("test_converted.wav", format="wav")
        print("Conversion complete.")

        # Load the Parakeet ASR model
        print("Loading Parakeet ASR model...")
        asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v3")
        print("Model loaded successfully.")

        # Transcribe the audio with timestamps
        print("Transcribing audio...")
        output = asr_model.transcribe(['test_converted.wav'], timestamps=True)

        # Print the transcription
        transcription = output[0].text
        print(f"Transcription: {transcription}")

        # Print timestamps if available
        if hasattr(output[0], 'timestamp') and 'segment' in output[0].timestamp:
            print("\nSegment timestamps:")
            for stamp in output[0].timestamp['segment']:
                print(f"{stamp['start']:.2f}s - {stamp['end']:.2f}s : {stamp['segment']}")

    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        if os.path.exists('test_converted.wav'):
            os.remove('test_converted.wav')
            print("Cleaned up temporary file.")

if __name__ == "__main__":
    main()
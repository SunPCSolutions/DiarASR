#!/usr/bin/env python3

import os
import torch
from pydub import AudioSegment
import torchaudio
from nvidia_diarization import NvidiaDiarization
from nvidia_asr import NvidiaASR

def main():
    try:
        # Check if test1.mp3 exists
        if not os.path.exists('test1.mp3'):
            print("Error: test1.mp3 not found")
            return

        # Convert mp3 to 16kHz mono wav for diarization
        print("Converting test1.mp3 to WAV format for diarization...")
        audio = AudioSegment.from_mp3("test1.mp3")
        audio = audio.set_channels(1).set_frame_rate(16000)
        audio.export("test_converted.wav", format="wav")
        print("Conversion complete.")

        # Initialize NVIDIA diarization with configurable streaming parameters
        print("Initializing NVIDIA diarization component...")
        diarization_model = NvidiaDiarization(
            model_name="nvidia/diar_streaming_sortformer_4spk-v2",
            chunk_size=6,        # Process ~0.48s chunks (6 frames * 80ms)
            right_context=7,     # 0.56s right context (7 frames * 80ms)
            fifo_size=188,       # FIFO buffer size
            update_period=144,   # Update every ~11.5s
            speaker_cache_size=188 # Speaker cache size
        )

        # Run offline diarization (for complete audio files)
        print("Running speaker diarization...")
        speaker_segments = diarization_model.run_offline_diarization("test_converted.wav")
        print("Diarization complete.")

        # Initialize NVIDIA ASR with VAD support
        print("Initializing NVIDIA ASR with VAD support...")
        asr_model = NvidiaASR(
            asr_model_name="nvidia/parakeet-ctc-1.1b",
            use_vad=True,
            vad_threshold=0.5,
            min_segment_duration=0.05,
            enable_batch_processing=True,
            batch_size=4
        )
        print("ASR model initialized successfully.")

        # Load audio for segmentation
        waveform, sample_rate = torchaudio.load("test_converted.wav")
        print(f"Audio loaded: {waveform.shape}, sample rate: {sample_rate}")

        # Process each speaker segment
        results = []

        # Process speaker segments from NvidiaDiarization output
        for segment in speaker_segments:
            start_time = segment['start']
            end_time = segment['end']
            speaker = segment['speaker']

            segment_duration = end_time - start_time

            # Skip segments that are too short for ASR (minimum ~50ms for reliable transcription)
            min_segment_duration = 0.05  # 50ms minimum
            if segment_duration < min_segment_duration:
                print(f"Skipping {speaker} segment ({segment_duration:.3f}s) - too short for ASR")
                continue

            print(f"Processing {speaker} from {start_time:.2f}s to {end_time:.2f}s ({segment_duration:.2f}s)")

            # Extract audio segment
            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)
            segment_waveform = waveform[:, start_sample:end_sample]

            # Save temporary segment
            temp_file = f"temp_segment_{speaker}_{start_time:.2f}.wav"
            torchaudio.save(temp_file, segment_waveform, sample_rate)

            # Transcribe segment using NVIDIA ASR with VAD
            try:
                transcription = asr_model.transcribe_segment(segment_waveform, sample_rate)

                # Skip empty transcriptions
                if transcription:
                    results.append({
                        'speaker': speaker,
                        'start': start_time,
                        'end': end_time,
                        'text': transcription
                    })
                    print(f"  Transcription: {transcription}")
                else:
                    print("  Transcription: (empty - skipped)")
            except Exception as e:
                print(f"  Error transcribing segment: {e}")
            finally:
                # Clean up temp file
                if os.path.exists(temp_file):
                    os.remove(temp_file)

        # Sort results by timestamp for chronological conversation order
        results_sorted = sorted(results, key=lambda x: x['start'])

        # Save clean results to file in conversation order
        with open('output.txt', 'w') as f:
            for result in results_sorted:
                f.write(f"{result['speaker']}: {result['text']}\n")

        # Print summary to console
        print(f"\nProcessed {len(results_sorted)} speaker segments successfully.")
        print("Clean transcript saved to output.txt (in conversation order)")
        print("\n" + "="*50)
        print("SAMPLE RESULTS (first 5 segments in conversation order)")
        print("="*50)

        for i, result in enumerate(results_sorted[:5]):
            print(f"{result['speaker']}: {result['text']}")
        if len(results_sorted) > 5:
            print(f"... and {len(results_sorted) - 5} more segments")

        # Clean up GPU memory and models
        print("Cleaning up GPU memory and models...")
        diarization_model.cleanup()
        asr_model.cleanup()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("GPU memory cleared.")

    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        for f in ['test_converted.wav']:
            if os.path.exists(f):
                os.remove(f)
                print(f"Cleaned up {f}")

if __name__ == "__main__":
    main()
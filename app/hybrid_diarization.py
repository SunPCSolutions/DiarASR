#!/usr/bin/env python3
"""
Hybrid Diarization Module: Pyannote + NVIDIA Integration

This module provides a hybrid diarization solution that combines:
- Pyannote.audio 3.1 speaker diarization (high accuracy)
- NVIDIA ASR integration (existing pipeline compatibility)

The goal is to leverage Pyannote's superior diarization quality while
maintaining compatibility with the existing NVIDIA ASR workflow.
"""

import os
import torch
import gc
from typing import Dict, List, Optional, Tuple
from pyannote.audio import Pipeline
from config import get_config


class HybridDiarization:
    """
    Hybrid Diarization using Pyannote.audio 3.1 with NVIDIA ASR compatibility.

    This class provides enterprise-grade speaker diarization by combining:
    - Pyannote speaker-diarization-3.1 (state-of-the-art accuracy)
    - Seamless integration with existing NVIDIA ASR pipeline
    - Speaker count constraints and GPU optimization
    """

    def __init__(
        self,
        pyannote_model: str = "pyannote/speaker-diarization-3.1",
        hf_token: Optional[str] = None,
        device: str = "auto",
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None
    ):
        """
        Initialize hybrid diarization.

        Args:
            pyannote_model: Pyannote model name
            hf_token: Hugging Face authentication token
            device: Device to run on ('auto', 'cpu', 'cuda')
            min_speakers: Minimum number of speakers
            max_speakers: Maximum number of speakers
        """
        self.pyannote_model = pyannote_model
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers

        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Initialize pipeline
        self.pipeline = None

        print(f"Hybrid Diarization initialized with {pyannote_model}")
        print(f"Device: {self.device}")
        print(f"Speaker constraints: min={min_speakers}, max={max_speakers}")

    def load_model(self):
        """Load Pyannote diarization pipeline."""
        if self.pipeline is None:
            print(f"Loading {self.pyannote_model}...")

            if not self.hf_token:
                raise ValueError(
                    "Hugging Face token required for Pyannote diarization. "
                    "Set HF_TOKEN environment variable or pass hf_token parameter."
                )

            try:
                # Set the token in environment if provided
                if self.hf_token:
                    os.environ["HF_TOKEN"] = self.hf_token

                # Load pipeline without authentication parameters
                # (uses HF_TOKEN environment variable)
                self.pipeline = Pipeline.from_pretrained(self.pyannote_model)

                # Move to specified device
                self.pipeline.to(self.device)
                print(f"✅ {self.pyannote_model} loaded successfully on {self.device}")

            except Exception as e:
                raise RuntimeError(f"Failed to load {self.pyannote_model}: {e}")

    def diarize_audio(self, audio_path: str) -> Dict:
        """
        Perform speaker diarization on audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            Dict with diarization results in standard format
        """
        self.load_model()

        print(f"Starting diarization on {audio_path}...")

        try:
            # Preprocess audio to 16kHz mono (required by Pyannote)
            import torchaudio

            # Load and resample audio
            waveform, sample_rate = torchaudio.load(audio_path)

            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # Resample to 16kHz if needed
            target_sample_rate = 16000
            if sample_rate != target_sample_rate:
                resampler = torchaudio.transforms.Resample(sample_rate, target_sample_rate)
                waveform = resampler(waveform)
                sample_rate = target_sample_rate

            # Use in-memory processing for faster performance
            diarization = self.pipeline(
                {"waveform": waveform, "sample_rate": sample_rate},
                min_speakers=self.min_speakers,
                max_speakers=self.max_speakers
            )

            # Convert to standard format
            segments = []
            # Access the speaker_diarization annotation from the DiarizeOutput
            speaker_diarization = diarization.speaker_diarization
            for segment, _, speaker in speaker_diarization.itertracks(yield_label=True):
                segments.append({
                    'start': float(segment.start),
                    'end': float(segment.end),
                    'speaker': speaker
                })

            # Sort by start time
            segments.sort(key=lambda x: x['start'])

            result = {
                'segments': segments,
                'num_speakers': len(set(s['speaker'] for s in segments))
            }

            print(f"✅ Diarization complete: {len(segments)} segments, "
                  f"{result['num_speakers']} speakers detected")

            return result

        except Exception as e:
            raise RuntimeError(f"Diarization failed: {e}")

    def assign_speakers_to_segments(self, transcription_segments: List[Dict],
                                  diarization_result: Dict) -> List[Dict]:
        """
        Assign speakers to transcription segments (WhisperX-style assignment).

        This implements a simplified version of speaker assignment logic.
        For production use, consider using WhisperX's assign_word_speakers.

        Args:
            transcription_segments: List of transcription segments
            diarization_result: Diarization results

        Returns:
            Transcription segments with speaker assignments
        """
        diarization_segments = diarization_result['segments']

        # Simple assignment: find overlapping diarization segment for each transcription segment
        assigned_segments = []

        for trans_seg in transcription_segments:
            trans_start = trans_seg['start']
            trans_end = trans_seg['end']

            # Find best matching diarization segment
            best_speaker = None
            max_overlap = 0

            for dia_seg in diarization_segments:
                dia_start = dia_seg['start']
                dia_end = dia_seg['end']

                # Calculate overlap
                overlap_start = max(trans_start, dia_start)
                overlap_end = min(trans_end, dia_end)
                overlap = max(0, overlap_end - overlap_start)

                if overlap > max_overlap:
                    max_overlap = overlap
                    best_speaker = dia_seg['speaker']

            # Assign speaker if overlap found
            if best_speaker and max_overlap > 0:
                new_segment = trans_seg.copy()
                new_segment['speaker'] = best_speaker
                assigned_segments.append(new_segment)

        print(f"Assigned speakers to {len(assigned_segments)} transcription segments")
        return assigned_segments

    def filter_speakers(self, segments: List[Dict], num_speakers: Optional[int] = None) -> List[Dict]:
        """
        Filter and renumber speakers based on speaking time.

        Args:
            segments: List of segments with speaker labels
            num_speakers: Target number of speakers

        Returns:
            Filtered segments with consecutive speaker numbering
        """
        if num_speakers is None:
            return segments

        # Calculate speaking time per speaker
        speaker_times = {}
        for segment in segments:
            speaker = segment['speaker']
            duration = segment['end'] - segment['start']
            speaker_times[speaker] = speaker_times.get(speaker, 0) + duration

        # Select top N speakers by speaking time
        sorted_speakers = sorted(speaker_times.items(), key=lambda x: x[1], reverse=True)
        selected_speakers = dict(sorted_speakers[:num_speakers])

        # Create mapping to consecutive numbering
        speaker_mapping = {old_id: f"speaker_{i}" for i, (old_id, _) in enumerate(selected_speakers.items())}

        # Filter and renumber segments
        filtered_segments = []
        for segment in segments:
            if segment['speaker'] in selected_speakers:
                new_segment = segment.copy()
                new_segment['speaker'] = speaker_mapping[segment['speaker']]
                filtered_segments.append(new_segment)

        print(f"Filtered to {num_speakers} speakers: {list(speaker_mapping.values())}")
        return filtered_segments

    def cleanup(self):
        """Aggressively clean up resources and free GPU memory."""
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None

        # Aggressive GPU memory cleanup
        if torch.cuda.is_available():
            # Multiple empty_cache calls
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            # Force garbage collection
            gc.collect()

            # Try to trigger memory deallocation
            try:
                dummy = torch.zeros(1024, device='cuda')
                del dummy
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            except:
                pass

        print("Hybrid diarization cleanup complete")


# Convenience functions
def create_hybrid_diarizer(**kwargs) -> HybridDiarization:
    """
    Create a configured HybridDiarization instance.

    Args:
        **kwargs: Parameters for HybridDiarization constructor

    Returns:
        Configured HybridDiarization instance
    """
    # Get defaults from config
    config = get_config()

    # Set defaults from config if not provided
    kwargs.setdefault('hf_token', os.getenv('HF_TOKEN'))
    kwargs.setdefault('device', config.diarization.device)

    return HybridDiarization(**kwargs)


def diarize_audio_file(audio_path: str, **kwargs) -> Dict:
    """
    Convenience function to diarize a single audio file.

    Args:
        audio_path: Path to audio file
        **kwargs: Parameters for HybridDiarization

    Returns:
        Diarization results
    """
    diarizer = create_hybrid_diarizer(**kwargs)
    try:
        result = diarizer.diarize_audio(audio_path)
        return result
    finally:
        diarizer.cleanup()


# Test function
if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) < 2:
        print("Usage: python hybrid_diarization.py <audio_file>")
        sys.exit(1)

    audio_file = sys.argv[1]

    # Test diarization
    diarizer = create_hybrid_diarizer(min_speakers=2, max_speakers=4)
    try:
        result = diarizer.diarize_audio(audio_file)
        print("Diarization Results:")
        print(f"Number of segments: {len(result['segments'])}")
        print(f"Number of speakers: {result['num_speakers']}")
        print("Sample segments:")
        for i, segment in enumerate(result['segments'][:5]):
            print(f"  {i+1}: {segment['start']:.2f}s - {segment['end']:.2f}s ({segment['speaker']})")
    finally:
        diarizer.cleanup()
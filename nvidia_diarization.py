#!/usr/bin/env python3
"""
NVIDIA Diarization Module using SortformerEncLabelModel

This module provides a modular diarization component that uses
nvidia/diar_streaming_sortformer_4spk-v2 with configurable streaming parameters.
"""

import os
import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
import nemo.collections.asr as nemo_asr
from pydub import AudioSegment
import torchaudio


class NvidiaDiarization:
    """
    Modular NVIDIA diarization component using SortformerEncLabelModel.

    Supports streaming diarization with configurable parameters.
    """

    def __init__(
        self,
        model_name: str = "nvidia/diar_streaming_sortformer_4spk-v2",
        chunk_size: int = 6,  # frames (80ms each, so ~0.48s)
        right_context: int = 7,  # frames (80ms each, so ~0.56s)
        fifo_size: int = 188,  # frames
        update_period: int = 144,  # frames
        speaker_cache_size: int = 188,  # frames
        device: str = "auto",
        num_speakers: Optional[int] = None
    ):
        """
        Initialize the NVIDIA diarization component.

        Args:
            model_name: Name of the pretrained model
            chunk_size: Size of audio chunks to process (frames, 80ms each)
            right_context: Additional context from future chunks (frames, 80ms each)
            fifo_size: Size of FIFO buffer for overlapping chunks (frames)
            update_period: How often to update diarization results (frames)
            speaker_cache_size: Maximum number of speakers to cache (frames)
            device: Device to run on ('auto', 'cpu', 'cuda')
        """
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.right_context = right_context
        self.fifo_size = fifo_size
        self.update_period = update_period
        self.speaker_cache_size = speaker_cache_size
        self.num_speakers = num_speakers

        # Set device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Initialize model
        self.model = None
        self.sample_rate = 16000  # NeMo models expect 16kHz

        # Streaming state
        self.audio_buffer = []
        self.current_time = 0.0
        self.last_update_time = 0.0
        self.speaker_segments = []
        self.speaker_cache = {}

        print(f"Initializing NVIDIA Diarization with model: {model_name}")
        print(f"Device: {self.device}")
        print(f"Streaming parameters:")
        print(f"  chunk_size: {chunk_size}s")
        print(f"  right_context: {right_context}s")
        print(f"  fifo_size: {fifo_size}")
        print(f"  update_period: {update_period}s")
        print(f"  speaker_cache_size: {speaker_cache_size}")

    def load_model(self):
        """Load the NVIDIA diarization model."""
        if self.model is None:
            print(f"Loading model {self.model_name}...")
            from nemo.collections.asr.models import SortformerEncLabelModel
            self.model = SortformerEncLabelModel.from_pretrained(self.model_name)

            # Set streaming parameters
            self.model.sortformer_modules.chunk_len = self.chunk_size
            self.model.sortformer_modules.chunk_right_context = self.right_context
            self.model.sortformer_modules.fifo_len = self.fifo_size
            self.model.sortformer_modules.spkcache_update_period = self.update_period
            self.model.sortformer_modules.spkcache_len = self.speaker_cache_size
            self.model.sortformer_modules._check_streaming_parameters()

            self.model = self.model.to(self.device)
            self.model.eval()
            print("Model loaded successfully.")
            print(f"Streaming parameters set: chunk_len={self.chunk_size}, right_context={self.right_context}, fifo_len={self.fifo_size}, update_period={self.update_period}, cache_size={self.speaker_cache_size}")

    def preprocess_audio(self, audio_path: str) -> Tuple[torch.Tensor, int]:
        """
        Preprocess audio file for diarization.

        Args:
            audio_path: Path to audio file

        Returns:
            Tuple of (audio_tensor, sample_rate)
        """
        # Convert to 16kHz mono WAV if needed
        if not audio_path.endswith('.wav'):
            temp_wav = audio_path.rsplit('.', 1)[0] + '_processed.wav'
            audio = AudioSegment.from_file(audio_path)
            audio = audio.set_channels(1).set_frame_rate(self.sample_rate)
            audio.export(temp_wav, format="wav")
            audio_path = temp_wav
        else:
            temp_wav = None

        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path)

        # Ensure mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Ensure correct sample rate
        if sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.sample_rate)
            waveform = resampler(waveform)

        # Clean up temp file
        if temp_wav and os.path.exists(temp_wav):
            os.remove(temp_wav)

        return waveform, self.sample_rate

    def run_offline_diarization(self, audio_path: str) -> List[Dict]:
        """
        Run offline diarization on complete audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            List of speaker segments with start, end, speaker info
        """
        self.load_model()

        print(f"Running offline diarization on {audio_path}")

        # Preprocess audio
        waveform, sample_rate = self.preprocess_audio(audio_path)

        # Save temporary WAV for NeMo
        temp_wav = "temp_diarization_input.wav"
        torchaudio.save(temp_wav, waveform, sample_rate)

        try:
            # Run diarization using the correct API
            # Note: Sortformer model may not accept num_speakers parameter directly
            # The parameter is stored for potential post-processing filtering
            diarization_output = self.model.diarize(audio=temp_wav, batch_size=1)
            print(f"Diarization completed, will filter to {self.num_speakers} speakers if specified")

            print(f"Raw diarization output type: {type(diarization_output)}")
            print(f"Raw diarization output: {diarization_output}")

            # Parse output (list of segments)
            segments = self._parse_diarization_output(diarization_output)

            # Filter to expected number of speakers if specified
            if self.num_speakers is not None and segments:
                # Group segments by speaker
                speaker_segments = {}
                for segment in segments:
                    speaker = segment['speaker']
                    if speaker not in speaker_segments:
                        speaker_segments[speaker] = []
                    speaker_segments[speaker].append(segment)

                # Select top N speakers by total duration
                speaker_durations = {}
                for speaker, segs in speaker_segments.items():
                    total_duration = sum(s['end'] - s['start'] for s in segs)
                    speaker_durations[speaker] = total_duration

                # Sort speakers by total duration (descending)
                sorted_speakers = sorted(speaker_durations.items(), key=lambda x: x[1], reverse=True)

                # Keep only the top N speakers and create mapping to consecutive IDs
                selected_speakers = {}
                for new_id, (old_speaker, _) in enumerate(sorted_speakers[:self.num_speakers]):
                    selected_speakers[old_speaker] = f"speaker_{new_id}"

                # Filter segments and renumber speakers consecutively
                filtered_segments = []
                for segment in segments:
                    if segment['speaker'] in selected_speakers:
                        new_segment = segment.copy()
                        new_segment['speaker'] = selected_speakers[segment['speaker']]
                        filtered_segments.append(new_segment)

                print(f"Filtered from {len(segments)} to {len(filtered_segments)} segments, renumbered to {self.num_speakers} consecutive speakers")
                segments = filtered_segments

            print(f"Final segments: {len(segments)}")
            if segments:
                print(f"Sample segments: {segments[:3]}")

            return segments

        finally:
            # Clean up
            if os.path.exists(temp_wav):
                os.remove(temp_wav)

    def run_streaming_diarization(self, audio_chunk: torch.Tensor, chunk_start_time: float) -> List[Dict]:
        """
        Process audio chunk for streaming diarization.

        Args:
            audio_chunk: Audio chunk tensor (1, samples)
            chunk_start_time: Start time of this chunk in seconds

        Returns:
            Updated list of speaker segments
        """
        self.load_model()

        # Add chunk to buffer
        self.audio_buffer.append((audio_chunk, chunk_start_time))

        # Maintain FIFO size
        if len(self.audio_buffer) > self.fifo_size:
            self.audio_buffer.pop(0)

        # Check if we should update diarization
        current_time = chunk_start_time + (audio_chunk.shape[1] / self.sample_rate)
        if current_time - self.last_update_time >= self.update_period:
            self._update_streaming_diarization()
            self.last_update_time = current_time

        return self.speaker_segments.copy()

    def _update_streaming_diarization(self):
        """Update diarization results from current buffer."""
        if len(self.audio_buffer) == 0:
            return

        # Concatenate buffered audio with right context
        buffered_audio = []
        start_times = []

        for chunk, start_time in self.audio_buffer:
            buffered_audio.append(chunk)
            start_times.append(start_time)

        if buffered_audio:
            # Concatenate audio
            full_audio = torch.cat(buffered_audio, dim=1)

            # Save temporary file for NeMo
            temp_wav = "temp_streaming_diarization.wav"
            torchaudio.save(temp_wav, full_audio, self.sample_rate)

            try:
                # Run diarization on buffered audio
                diarization_output = self.model(temp_wav)

                # Parse and update segments
                new_segments = self._parse_diarization_output(diarization_output)

                # Adjust timestamps relative to global time
                adjusted_segments = []
                buffer_start_time = start_times[0]

                for segment in new_segments:
                    adjusted_segment = segment.copy()
                    adjusted_segment['start'] += buffer_start_time
                    adjusted_segment['end'] += buffer_start_time
                    adjusted_segments.append(adjusted_segment)

                # Merge with existing segments
                self._merge_segments(adjusted_segments)

            finally:
                if os.path.exists(temp_wav):
                    os.remove(temp_wav)

    def _parse_diarization_output(self, output) -> List[Dict]:
        """
        Parse diarization model output into segment format.

        Args:
            output: Raw model output

        Returns:
            List of segments with start, end, speaker
        """
        segments = []

        # Handle different output formats
        if isinstance(output, list):
            for item in output:
                if isinstance(item, dict):
                    segments.append({
                        'start': item.get('start', item.get('start_time', 0)),
                        'end': item.get('end', item.get('end_time', 0)),
                        'speaker': item.get('speaker', item.get('speaker_id', 'SPEAKER_00'))
                    })
                elif isinstance(item, list):
                    # Handle nested list of strings (the actual format from Sortformer)
                    for subitem in item:
                        if isinstance(subitem, str):
                            # Parse "start end speaker" format
                            parts = subitem.strip().split()
                            if len(parts) >= 3:
                                try:
                                    start = float(parts[0])
                                    end = float(parts[1])
                                    speaker = parts[2]
                                    segments.append({
                                        'start': start,
                                        'end': end,
                                        'speaker': speaker
                                    })
                                except (ValueError, IndexError):
                                    continue
                elif isinstance(item, str):
                    # Parse RTTM-like string
                    lines = item.strip().split('\n') if '\n' in item else [item]
                    for line in lines:
                        if line.strip():
                            parts = line.split()
                            if len(parts) >= 3:  # start end speaker format
                                try:
                                    start = float(parts[0])
                                    end = float(parts[1])
                                    speaker = parts[2]
                                    segments.append({
                                        'start': start,
                                        'end': end,
                                        'speaker': speaker
                                    })
                                except (ValueError, IndexError):
                                    continue
        elif isinstance(output, str):
            # Parse RTTM-like string
            lines = output.strip().split('\n')
            for line in lines:
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 8:
                        try:
                            start = float(parts[3])
                            duration = float(parts[4])
                            end = start + duration
                            speaker = parts[7]
                            segments.append({
                                'start': start,
                                'end': end,
                                'speaker': speaker
                            })
                        except (ValueError, IndexError):
                            continue

        return segments

    def _merge_segments(self, new_segments: List[Dict]):
        """Merge new segments with existing ones."""
        # Simple merge strategy - replace overlapping segments
        all_segments = self.speaker_segments + new_segments

        # Sort by start time
        all_segments.sort(key=lambda x: x['start'])

        # Basic merging (can be improved)
        merged = []
        for segment in all_segments:
            if not merged or merged[-1]['end'] < segment['start']:
                merged.append(segment)
            else:
                # Overlap - extend the last segment if same speaker
                if merged[-1]['speaker'] == segment['speaker']:
                    merged[-1]['end'] = max(merged[-1]['end'], segment['end'])
                else:
                    merged.append(segment)

        self.speaker_segments = merged

        # Limit speaker cache
        speakers = set()
        for segment in self.speaker_segments:
            speakers.add(segment['speaker'])
            if len(speakers) >= self.speaker_cache_size:
                break

    def reset_streaming(self):
        """Reset streaming state."""
        self.audio_buffer = []
        self.current_time = 0.0
        self.last_update_time = 0.0
        self.speaker_segments = []
        self.speaker_cache = {}

    def get_current_segments(self) -> List[Dict]:
        """Get current speaker segments."""
        return self.speaker_segments.copy()

    def cleanup(self):
        """Clean up resources."""
        if self.model is not None:
            del self.model
            self.model = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("NVIDIA Diarization cleanup complete.")


# Convenience function for easy usage
def create_nvidia_diarizer(**kwargs) -> NvidiaDiarization:
    """
    Create a configured NVIDIA diarization instance.

    Args:
        **kwargs: Parameters for NvidiaDiarization constructor

    Returns:
        Configured NvidiaDiarization instance
    """
    return NvidiaDiarization(**kwargs)
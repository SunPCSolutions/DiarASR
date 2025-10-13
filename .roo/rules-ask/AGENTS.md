# Ask Mode: Non-Obvious Documentation Context

## Model Architecture Details
- **Parakeet CTC 1.1B**: FastConformer encoder with CTC decoder, 1.1B parameters, supports VAD integration
- **Sortformer 4spk**: Streaming diarization with speaker cache, handles up to 4 speakers, requires specific streaming parameters
- **VAD Integration**: Voice activity detection built into ASR pipeline using `vad_multilingual_marblenet`, not separate processing

## Pipeline Flow Understanding
- **Diarization First**: Speaker segmentation runs before ASR to identify speaker boundaries
- **Segment Extraction**: Audio split into speaker-specific segments using torchaudio waveform slicing
- **Batch Processing**: ASR processes multiple speaker segments simultaneously when enabled
- **VAD Filtering**: Optional speech detection filters out non-speech segments before transcription

## Configuration Dependencies
- **Streaming Parameters**: Diarization chunk_size, right_context, fifo_size, update_period must be set before inference
- **Batch Size Limits**: GPU memory constraints limit batch processing to 16 segments maximum
- **Audio Format Requirements**: All input converted to 16kHz mono WAV regardless of input format

## Security Implementation Details
- **Zero-Overwrite Deletion**: Temporary files overwritten with zeros before filesystem removal
- **Permission Restrictions**: Temporary directories created with 0o700 permissions
- **Automatic Cleanup**: All temporary resources cleaned up even on processing failures

## Performance Characteristics
- **GPU Memory Intensive**: Models require 4GB+ VRAM, batch processing increases memory usage
- **Streaming Latency**: Diarization streaming parameters affect real-time processing latency
- **VAD Trade-offs**: Voice activity detection improves accuracy but adds processing overhead
# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## Non-Obvious Project Patterns

### NVIDIA NeMo Integration
- **Model Loading**: Use `nemo_asr.models.EncDecCTCModelBPE.from_pretrained()` for ASR, not generic transformers
- **Diarization**: Use `SortformerEncLabelModel.from_pretrained()` with streaming parameters configured
- **VAD Integration**: Voice Activity Detection is built into ASR pipeline, not separate component
- **GPU Memory**: Explicit cleanup required after model usage to prevent memory leaks

### Secure File Handling
- **SecureTempManager**: Implements zero-overwrite file deletion for sensitive audio data
- **Temporary Directories**: Created with 0o700 permissions, automatically cleaned up
- **File Validation**: Strict extension checking and size limits before processing

### Configuration System
- **Centralized Config**: All parameters in `config.py` with dataclass structure
- **Streaming Parameters**: Diarization requires specific chunk_size, right_context, fifo_size, update_period values
- **Batch Processing**: ASR supports both individual and batch processing modes

### Pipeline Architecture
- **Modular Components**: NvidiaDiarization, NvidiaASR, PipelineOrchestrator classes
- **Audio Preprocessing**: Automatic conversion to 16kHz WAV format required
- **Segment Processing**: Speaker segments extracted before ASR transcription

### Testing Requirements
- **GPU Testing**: Models require CUDA-compatible hardware for full testing
- **Audio Files**: Test files must be in supported formats (.mp3, .wav, .flac, .m4a, .aac)
- **Memory Cleanup**: Tests must include proper resource cleanup to avoid GPU memory issues
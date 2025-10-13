# Code Mode: Non-Obvious Implementation Rules

## NVIDIA Model Integration
- **NeMo Model Loading**: Always use `nemo_asr.models.EncDecCTCModelBPE.from_pretrained()` for ASR models, not generic transformers
- **Streaming Diarization**: `SortformerEncLabelModel` requires explicit streaming parameter configuration before inference
- **VAD in ASR**: Voice Activity Detection is integrated into ASR pipeline via `vad_multilingual_marblenet`, not separate processing

## Secure File Handling Patterns
- **SecureTempManager**: Use `SecureTempManager` for all temporary file operations - implements zero-overwrite deletion
- **Temporary Permissions**: All temp directories created with 0o700 permissions for security
- **File Sanitization**: Sensitive audio files must be overwritten with zeros before deletion

## Configuration System
- **Centralized Config**: All parameters modified in `config.py` only - never hardcoded in modules
- **Streaming Parameters**: Diarization chunk_size, right_context, fifo_size, update_period must be set before model inference
- **Batch Processing**: ASR batch_size affects both individual and batch processing modes

## Pipeline Architecture
- **Modular Components**: PipelineOrchestrator coordinates NvidiaDiarization → NvidiaASR with automatic cleanup
- **Audio Preprocessing**: All audio automatically converted to 16kHz mono WAV before processing
- **GPU Memory Management**: Explicit `cleanup()` calls required after model usage to prevent memory leaks

## Testing Requirements
- **GPU Testing**: All model tests require CUDA-compatible hardware and proper GPU memory cleanup
- **Audio Formats**: Test files must use supported extensions (.mp3, .wav, .flac, .m4a, .aac) or tests will fail
- **Resource Cleanup**: Every test must call cleanup methods to prevent GPU memory accumulation
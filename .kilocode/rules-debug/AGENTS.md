# Debug Mode: Non-Obvious Debugging Rules

## GPU Memory Issues
- **CUDA Memory Leaks**: Always call `cleanup()` methods on NvidiaDiarization and NvidiaASR instances after processing
- **Model Disposal**: Explicitly set model references to None after use to trigger garbage collection
- **Batch Size Limits**: GPU memory errors occur when batch_size > 16 on consumer GPUs - reduce batch size first

## Audio Processing Errors
- **File Format Issues**: Only .wav, .mp3, .flac, .m4a, .aac supported - check file extensions before processing
- **Sample Rate Mismatch**: All audio automatically converted to 16kHz mono - check logs for conversion errors
- **Empty Segments**: Diarization may return no segments for silent or uniform audio - check audio content

## Model Loading Failures
- **HuggingFace Token**: Diarization model requires HF_TOKEN environment variable for gated model access
- **CUDA Compatibility**: Models require CUDA 12.8+ - check `torch.cuda.is_available()` and CUDA version
- **Memory Requirements**: Models need 4GB+ GPU memory - insufficient VRAM causes silent failures

## Pipeline Debugging
- **Temporary Files**: Check `/tmp/` or custom temp directories for leftover files if cleanup fails
- **Streaming Parameters**: Diarization streaming config must be validated - check `sortformer_modules._check_streaming_parameters()`
- **VAD Threshold**: ASR VAD threshold 0.5 may miss quiet speech - adjust based on audio characteristics

## Common Silent Failures
- **No Speaker Segments**: Diarization returns empty list for single-speaker or noisy audio
- **Empty Transcriptions**: ASR returns empty segments when VAD filters out all audio
- **GPU Context Loss**: Long-running processes lose CUDA context - restart models periodically
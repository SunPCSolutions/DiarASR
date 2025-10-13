# Architect Mode: Non-Obvious Architectural Constraints

## Model Coupling Requirements
- **Sequential Processing**: Diarization must complete before ASR begins - no parallel processing possible
- **GPU Resource Sharing**: Both models compete for same GPU memory - batch processing limited by VRAM
- **Streaming Parameter Dependencies**: Diarization streaming config affects entire pipeline latency

## Security Architecture Constraints
- **In-Memory Processing**: Audio data never written to disk unencrypted - all processing in memory
- **Zero-Knowledge Cleanup**: Temporary files overwritten before deletion to prevent forensic recovery
- **Permission Isolation**: Temporary directories created with restrictive permissions (0o700)

## Scalability Limitations
- **GPU Memory Bottleneck**: Maximum batch size 16 limited by CUDA memory constraints
- **Sequential Dependency**: Pipeline cannot be parallelized due to diarization → ASR dependency
- **Model Size Constraints**: 1.1B parameter ASR model requires 4GB+ VRAM minimum

## Data Flow Architecture
- **Audio Preprocessing**: All formats converted to 16kHz mono WAV before model processing
- **Segment Extraction**: Speaker segments extracted using waveform slicing, not file splitting
- **Batch Optimization**: Multiple speaker segments batched together for efficient GPU utilization

## Error Handling Architecture
- **Graceful Degradation**: Pipeline continues processing other segments if individual segments fail
- **Resource Cleanup**: All GPU memory and temporary files cleaned up even on partial failures
- **Logging Integration**: Comprehensive logging with timestamps for debugging pipeline issues

## Configuration Architecture
- **Centralized Parameters**: All settings in config.py dataclass - no hardcoded values
- **Runtime Overrides**: Pipeline accepts configuration overrides without code changes
- **Validation Layer**: Configuration validated before model initialization
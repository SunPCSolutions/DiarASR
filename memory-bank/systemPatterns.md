# System Patterns: HYBRID ASR Diarization Architecture

## Core Architecture Patterns

### Hybrid Modular Pipeline Pattern (RECOMMENDED)
```
FastAPI Service (API Layer)
├── HybridDiarization (Pyannote 3.1 + Speaker Control)
├── NvidiaASR (Parakeet TDT with Silero VAD)
├── VRAMManager (Automatic Model Unloading)
└── SecureTempManager (File Security)
```

**Benefits**: Enterprise-grade quality, superior diarization accuracy, production-ready
**Implementation**: Pyannote diarization + NVIDIA ASR with seamless integration

### Legacy NVIDIA Pipeline Pattern (Available)
```
FastAPI Service (API Layer)
├── NvidiaDiarization (Sortformer + Speaker Control)
├── NvidiaASR (Parakeet CTC with Silero VAD)
├── VRAMManager (Automatic Model Unloading)
└── SecureTempManager (File Security)
```

**Benefits**: NVIDIA ecosystem consistency, functional baseline
**Implementation**: Pure NVIDIA implementation with VRAM management

### Secure Resource Management Pattern
```python
# Context manager for secure temporary resources
with secure_temp_dir() as temp_dir:
    # Process audio
    process_segments(temp_dir)
# Automatic cleanup with zero-overwrite deletion
```

**Benefits**: Prevents data leakage, ensures cleanup on failures
**Implementation**: Custom SecureTempManager with file sanitization

### Configuration-Driven Architecture
```python
@dataclass
class GlobalConfig:
    asr: ASRConfig
    diarization: DiarizationConfig
    processing: ProcessingConfig
```

**Benefits**: Centralized configuration, runtime flexibility
**Implementation**: Dataclass-based config with validation

## Data Flow Patterns

### Hybrid Sequential Processing Pipeline (PRIMARY)
```
Audio File → Validation → Audio Preprocessing (16kHz) → Pyannote Diarization → Speaker Filtering → Segment Extraction → Parakeet TDT ASR → Speaker Assignment → Results → VRAM Cleanup
```

**Rationale**: High-quality diarization → precise segmentation → advanced ASR → perfect speaker attribution
**Optimization**: GPU acceleration, batch processing, automatic memory management

### Legacy NVIDIA Processing Pipeline (FALLBACK)
```
Audio File → Validation → Diarization → Speaker Filtering → Segment Extraction → ASR with VAD → Results → Optional VRAM Cleanup
```

**Rationale**: Diarization → speaker constraints → ASR processing with resource management
**Optimization**: Batch processing, VAD filtering, automatic memory cleanup

### Secure In-Memory Processing
```
File Input → Memory Buffer → Processing → Memory Output → Secure Cleanup
```

**Rationale**: Sensitive medical data never written to disk unencrypted
**Implementation**: torchaudio waveform processing, temporary file cleanup

## Component Interaction Patterns

### Backend Selection Pattern
```python
# Dynamic backend selection based on configuration
if config.diarization.backend == "hybrid":
    diarization_module = HybridDiarization(config)
elif config.diarization.backend == "nvidia":
    diarization_module = NvidiaDiarization(config)
else:
    # Auto-selection based on available resources
    diarization_module = select_optimal_backend(config)
```

**Benefits**: Quality optimization, fallback compatibility, resource adaptation
**Implementation**: Configuration-driven backend selection with graceful degradation

### Hybrid Integration Pattern
```python
# Pyannote diarization with NVIDIA ASR
diarization_result = pyannote_diarizer.diarize_audio(audio_path)
transcription_result = nvidia_asr.transcribe_segments(segments)

# Speaker assignment integration
final_result = assign_speakers_to_transcript(
    transcription_result, diarization_result
)
```

**Benefits**: Best-of-breed components, superior accuracy, seamless integration
**Implementation**: Compatible data formats, shared GPU resources, unified API

### Dependency Injection Pattern
```python
# Modules receive configuration, not create it
asr_module = NvidiaASR(config.asr_config)
diarization_module = HybridDiarization(config.diarization_config)
```

**Benefits**: Testability, configuration flexibility, backend abstraction
**Implementation**: Constructor injection with typed configurations

### Observer Pattern for Progress Tracking
```python
# Logging integrated into all components
logger.info("Processing file: %s", audio_path)
logger.info("Diarization complete: %d segments", len(segments))
```

**Benefits**: Debugging, monitoring, error tracking
**Implementation**: Python logging with structured messages

## Error Handling Patterns

### Graceful Degradation Pattern
```python
try:
    result = process_segment(segment)
except Exception as e:
    logger.error("Segment failed: %s", e)
    continue  # Continue with other segments
```

**Benefits**: Partial failures don't break entire pipeline
**Implementation**: Exception isolation, error logging, continuation

### Resource Cleanup Pattern
```python
try:
    # Process with GPU resources
    results = process_audio()
finally:
    # Always cleanup
    cleanup_gpu_memory()
    cleanup_temp_files()
```

**Benefits**: Prevents resource leaks, ensures cleanup on errors
**Implementation**: finally blocks, context managers

### VRAM Management Pattern
```python
# Automatic model unloading after processing
if unload_models_after:
    unload_models()  # Clear GPU cache, free VRAM

# Manual cleanup endpoint
@app.post("/cleanup/")
async def cleanup_models():
    unload_models()
    return {"message": "VRAM freed"}
```

**Benefits**: Prevents GPU memory exhaustion, enables resource sharing
**Implementation**: Configurable unloading, manual cleanup endpoints

### Speaker Count Control Pattern
```python
# Post-processing speaker filtering
if num_speakers is not None:
    # Group segments by speaker and select top N
    speaker_durations = {}
    for segment in segments:
        speaker = segment['speaker']
        duration = segment['end'] - segment['start']
        speaker_durations[speaker] = speaker_durations.get(speaker, 0) + duration

    # Keep only top N speakers by total duration
    selected_speakers = sorted(speaker_durations.items(), key=lambda x: x[1], reverse=True)[:num_speakers]
    filtered_segments = [s for s in segments if s['speaker'] in dict(selected_speakers)]
```

**Benefits**: Reliable speaker count constraints, improved diarization accuracy
**Implementation**: Duration-based speaker selection, post-processing filtering

## Performance Optimization Patterns

### GPU Memory Management Pattern
```python
# Explicit cleanup between operations
torch.cuda.empty_cache()
gc.collect()
model = None  # Release reference
```

**Benefits**: Prevents OOM errors, maximizes GPU utilization
**Implementation**: Manual memory management, garbage collection

### Batch Processing Pattern
```python
# Group operations for efficiency
if enable_batch_processing:
    batch_results = process_batch(segments)
else:
    results = [process_individual(s) for s in segments]
```

**Benefits**: GPU utilization, reduced overhead
**Implementation**: Configurable batch sizes, fallback to individual processing

## Security Patterns

### Zero-Knowledge File Deletion
```python
# Overwrite before deletion
with open(file_path, 'wb') as f:
    f.write(b'\x00' * file_size)
os.remove(file_path)
```

**Benefits**: Prevents forensic data recovery
**Implementation**: SecureTempManager with sanitization

### Permission Restriction Pattern
```python
# Restrictive temporary directories
temp_dir = tempfile.mkdtemp(prefix="secure_", dir=base_dir)
os.chmod(temp_dir, 0o700)  # Owner only
```

**Benefits**: Prevents unauthorized access
**Implementation**: Automatic permission setting, secure defaults

## Testing Patterns

### Component Isolation Pattern
```python
# Test components independently
def test_diarization():
    diarization = NvidiaDiarization(config)
    segments = diarization.run_offline_diarization(audio_path)
    assert len(segments) > 0

def test_asr():
    asr = NvidiaASR(config)
    text = asr.transcribe_file(audio_path)
    assert text is not None
```

**Benefits**: Focused testing, easier debugging
**Implementation**: Unit tests for each component, integration tests for pipeline

### Configuration Testing Pattern
```python
# Test with different configurations
@pytest.mark.parametrize("batch_size", [1, 8, 16])
def test_batch_processing(batch_size):
    config = create_config(batch_size=batch_size)
    results = process_with_config(audio, config)
    assert len(results) > 0
```

**Benefits**: Validates configuration flexibility
**Implementation**: Parameterized tests, configuration fixtures

## Deployment Patterns

### Container-Ready Architecture
```dockerfile
# NVIDIA CUDA base image
FROM nvidia/cuda:12.8.0-base-ubuntu24.04

# Install NeMo and dependencies
RUN pip install nemo_toolkit[asr]

# Copy application
COPY . /app
```

**Benefits**: Consistent deployment, GPU compatibility
**Implementation**: Docker containerization, environment isolation

### API-First Design Pattern
```python
# REST API with comprehensive configuration options
@app.post("/transcribe_diarize/")
async def transcribe_diarize(
    audio_file: UploadFile,
    diarize: bool = True,
    vad: bool = True,
    num_speakers: Optional[int] = None,
    unload_models_after: bool = False
):
    # Process with configuration and resource management
    results = process_audio(
        audio_file,
        diarize=diarize,
        vad=vad,
        num_speakers=num_speakers,
        unload_models_after=unload_models_after
    )
    return results

@app.post("/cleanup/")
async def cleanup_models():
    unload_models()
    return {"message": "VRAM freed"}
```

**Benefits**: Integration-friendly, resource-aware, configurable processing
**Implementation**: FastAPI with parameter validation, VRAM management, n8n compatibility
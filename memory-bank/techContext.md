# Technical Context: NVIDIA ASR Diarization Stack

## Core Technology Stack

### Programming Languages & Frameworks
- **Python 3.12+**: Primary language for ML pipeline and API
- **FastAPI**: Asynchronous web framework for REST API
- **Pydantic**: Data validation and serialization
- **NVIDIA NeMo**: ML framework for speech processing models

### Machine Learning Stack
- **NVIDIA NeMo Toolkit**: ASR and diarization model framework
- **PyTorch 2.8.0+**: Deep learning framework with CUDA 12.8 support
- **torchaudio**: Audio processing and I/O
- **transformers**: HuggingFace model integration

### Models & Capabilities
- **ASR Model**: `nvidia/parakeet-ctc-1.1b` (1.1B parameters, FastConformer CTC)
- **Diarization Model**: `nvidia/diar_streaming_sortformer_4spk-v2` (Streaming Sortformer with speaker count control)
- **VAD Model**: `snakers4/silero-vad` (high-performance, lightweight VAD)
- **VRAM Management**: Automatic model unloading and memory cleanup
- **Supported Languages**: English (primary), multilingual support available

### Infrastructure Requirements

#### Hardware Requirements
- **GPU**: NVIDIA GPU with CUDA 12.8+ support (minimum 4GB VRAM)
- **RAM**: 8GB+ system RAM for model loading
- **Storage**: 10GB+ for models and temporary processing
- **CPU**: Modern x86-64 with AVX2 support

#### Software Requirements
- **OS**: Linux (Ubuntu 20.04+, CentOS 7+)
- **CUDA**: 12.8.0+ with cuDNN 9.10.2.21+
- **Python**: 3.12+ with pip and venv
- **Audio Libraries**: FFmpeg for audio format support

### Development Environment

#### Local Development
```bash
# Python virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# GPU verification
python -c "import torch; print(torch.cuda.is_available())"
```

#### Container Environment
```dockerfile
FROM nvidia/cuda:12.8.0-base-ubuntu24.04
RUN pip install nemo_toolkit[asr] fastapi uvicorn
```

### Dependencies & Versions

#### Core Dependencies (requirements.txt)
```
fastapi                    # Web framework
uvicorn[standard]         # ASGI server
torch>=2.8.0             # PyTorch with CUDA 12.8
torchaudio>=2.8.0        # Audio processing
torchvision>=0.19.0      # Image processing (for some audio ops)
nemo-toolkit[asr]        # NVIDIA speech models
pydub                    # Audio format conversion
transformers             # HuggingFace integration
python-multipart         # File upload handling
huggingface-hub          # Model downloading
torchvision              # Required for some audio operations
```

#### Development Dependencies
```
pytest                    # Testing framework
pytest-asyncio           # Async testing
black                    # Code formatting
isort                    # Import sorting
mypy                     # Type checking
```

### Model Specifications

#### Parakeet CTC 1.1B (ASR)
- **Architecture**: FastConformer with CTC decoder
- **Parameters**: 1.1 billion
- **Input**: 16kHz mono audio
- **Output**: English text transcription
- **Features**: Punctuation, capitalization, timestamps
- **Performance**: <2% WER on clean speech

#### Sortformer 4spk (Diarization)
- **Architecture**: Streaming transformer with speaker cache
- **Speakers**: Up to 4 concurrent speakers
- **Input**: 16kHz mono audio
- **Output**: Speaker segments with timestamps
- **Features**: Real-time streaming, permutation invariant
- **Latency**: Configurable (0.32s to 30.4s)

#### Silero VAD
- **Architecture**: Lightweight neural network optimized for CPU
- **Input**: Full audio waveforms (16kHz)
- **Output**: Speech timestamps with start/end times
- **Features**: High accuracy, fast inference, threshold-based filtering
- **Integration**: Standalone VAD with configurable thresholds
- **Performance**: Superior to WebRTC VAD, optimized for real-time use

### Performance Characteristics

#### Throughput (on NVIDIA RTX 4090)
- **ASR Only**: ~50x real-time (batch_size=16) with Silero VAD
- **Diarization Only**: ~20x real-time (streaming mode with speaker constraints)
- **Full Pipeline**: ~15x real-time (combined processing with memory management)
- **Memory Usage**: ~4GB VRAM per job with automatic cleanup and unloading
- **VRAM Management**: Models can be unloaded after processing to free memory

#### Latency Breakdown
- **Model Loading**: ~30 seconds (first run)
- **Audio Preprocessing**: <1 second
- **Diarization**: 0.32s - 30.4s (configurable)
- **ASR**: ~0.5x audio duration
- **Total Pipeline**: 1-60 seconds depending on configuration

### Security & Compliance

#### Data Protection
- **In-Memory Processing**: Audio never written to disk unencrypted
- **Secure Temporary Files**: Zero-overwrite deletion via TempFileTracker
- **Permission Restrictions**: 0o700 temporary directories
- **No Data Persistence**: All intermediate files cleaned up

#### HIPAA Compliance
- **Encryption**: TLS 1.3 for data in transit
- **Access Control**: API authentication required
- **Audit Logging**: All processing operations logged via TempFileTracker
- **Data Minimization**: Only necessary data retained
- **Secure Cleanup**: Automatic HIPAA-compliant file deletion with retry mechanisms

### Deployment Architecture

#### Development Deployment
```bash
# Local development
uvicorn app:app --reload --host 0.0.0.0 --port 8001

# With GPU
CUDA_VISIBLE_DEVICES=0 uvicorn app:app --host 0.0.0.0 --port 8001
```

#### Production Deployment
```yaml
# Docker Compose for production
version: '3.8'
services:
  asr-service:
    image: nvidia-asr-diarization:latest
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

#### Cloud Deployment
- **AWS**: EC2 P4d instances (A100 GPUs)
- **GCP**: A100/A100-80GB instances
- **Azure**: NC A100 v4 series
- **Kubernetes**: GPU-enabled node pools

### Monitoring & Observability

#### Metrics to Monitor
- **GPU Utilization**: Memory usage, compute utilization
- **Processing Latency**: End-to-end processing time
- **Error Rates**: Model failures, invalid inputs
- **Throughput**: Requests per second, concurrent jobs

#### Logging
- **Structured Logging**: JSON format with timestamps
- **Log Levels**: INFO for operations, ERROR for failures
- **Performance Metrics**: Processing times, memory usage
- **Security Events**: Access attempts, data operations

### Testing Strategy

#### Unit Testing
```python
# Component isolation
def test_nvidia_asr():
    asr = NvidiaASR(config)
    result = asr.transcribe_file("test.wav")
    assert result is not None

def test_secure_temp():
    manager = SecureTempManager()
    with manager.secure_temp_dir() as temp_dir:
        assert os.path.exists(temp_dir)
    # Directory should be cleaned up
```

#### Integration Testing
```python
# Full pipeline testing
def test_complete_pipeline():
    results = process_audio_files("test.mp3")
    assert len(results) > 0
    assert "segments" in results[0]
```

#### Performance Testing
```python
# Load testing
def test_concurrent_processing():
    # Test multiple concurrent requests
    results = await process_multiple_files(file_list)
    assert all results are valid
```

### Maintenance & Updates

#### Model Updates
- **NeMo Releases**: Monitor for model improvements
- **CUDA Updates**: Track compatibility with new CUDA versions
- **Security Patches**: Regular dependency updates

#### Performance Optimization
- **Batch Size Tuning**: Optimize for specific GPU memory
- **Streaming Configuration**: Adjust latency vs accuracy trade-offs
- **Memory Management**: Monitor and optimize GPU usage

This technical context provides the foundation for understanding, deploying, and maintaining the NVIDIA ASR diarization pipeline in production environments.
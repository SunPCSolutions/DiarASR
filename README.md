# DiarASR

⚠️ **IMPORTANT NOTICE**: HuggingFace models are downloaded on first API request. Due to potential network issues or HF download instability, the first few attempts might fail. After successful model download and caching, subsequent requests will work reliably.

Enterprise-grade secure ASR diarization pipeline combining automatic speech recognition with speaker diarization. HIPAA-compliant with modular architecture and comprehensive security.

## Features

- **🔒 Enterprise Security**: API key authentication, input validation, rate limiting
- **🎯 High-Quality Processing**: DER ~8-20%, WER ~1-5% with robust speaker attribution
- **🩺 HIPAA Compliance**: Secure file handling, audit logging, encrypted storage
- **🏗️ Modular Architecture**: Clean separation into focused modules
- **🐳 Production Ready**: Container-ready with security enhancements

## Requirements

- **GPU**: BF16-capable NVIDIA GPU (Ampere or newer) with a CUDA-compatible driver
- **OS**: Linux (Ubuntu 24.04+, CentOS 8+)
- **Python**: 3.12+
- **Models**: Local English ASR uses `omi-health/omi-med-stt-v1`. Optional speaker diarization requires access to `pyannote/speaker-diarization-community-1`.

## Installation

### Python Installation

```bash
git clone https://github.com/SunPCSolutions/DiarASR.git
cd DiarASR
python3 -m venv .venv
source .venv/bin/activate
pip install -r app/requirements.txt
cp .env.example .env
# Edit .env with your API keys and HuggingFace token
```

### Docker Installation

```bash
git clone https://github.com/SunPCSolutions/DiarASR.git
cd DiarASR
cp .env.example .env
# Edit .env with your API keys and HuggingFace token
docker-compose up --build -d
```

## Usage

### Python API

```python
import os
from app.app import process_audio

os.environ['HF_TOKEN'] = 'your-huggingface-token'
os.environ['API_KEYS'] = 'your-api-key'

result = process_audio(
    audio_path='audio.wav',
    diarize=True,
    min_speakers=2,
    max_speakers=4
)

for segment in result['segments']:
    print(f"{segment['speaker']}: {segment['text']}")
```

### REST API

```bash
# Start server
export API_KEYS="your-api-key"
export HF_TOKEN="hf_xxx"
uvicorn app:app --host 0.0.0.0 --port 8003

# Make request
curl -H "X-API-Key: your-api-key" \
  -X POST "http://localhost:8003/transcribe_diarize/" \
  -F "audio_file=@audio.wav"
```

See [`docs/API_PARAMETERS.md`](docs/API_PARAMETERS.md) for complete API documentation.

## Model Access

- [Omi Med STT v1](https://huggingface.co/omi-health/omi-med-stt-v1) by Omi Health is the only ASR model. Its CC BY 4.0 weights are derived from NVIDIA Parakeet TDT 0.6B v2. Inference runs locally with the pinned Omi CUDA runtime, not Omi's hosted API.
- [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) supplies speaker turn timestamps when `diarize=true`. For `diarize=false`, the transcript is returned as one full-duration segment; Omi's runtime does not produce word timestamps.

Set `HF_TOKEN` for the gated pyannote model. Review transcripts before clinical use.

## Performance

Accuracy, processing speed, and memory use depend on your recordings and GPU. The Omi model card reports results for its own evaluation set, not for the complete DiarASR pipeline.

## Security

- API key authentication
- Multi-layer input validation
- Rate limiting (10 req/min)
- Encrypted temporary storage
- HIPAA-compliant processing
- Comprehensive audit logging

## Documentation

- [`docs/API_PARAMETERS.md`](docs/API_PARAMETERS.md) - Complete API reference
- [`memory-bank/systemPatterns.md`](memory-bank/systemPatterns.md) - Architecture details
- [`memory-bank/techContext.md`](memory-bank/techContext.md) - Technical context

## License

MIT License - see LICENSE file for details.

## Acknowledgments

Our greatest appreciation to the creators of:
- **Pyannote.audio** (Hervé Bredin et al.) for speaker diarization
- **Omi Health** for Omi Med STT v1 and **NVIDIA NeMo** for the underlying ASR tooling
- **FastAPI** (Sebastián Ramírez) for the web framework
- **PyTorch** (Facebook AI Research) for deep learning

Please cite these works if used in your research.

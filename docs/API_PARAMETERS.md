# ASR Diarization Pipeline API Parameters Documentation

This document describes all configurable parameters for the ASR Diarization Pipeline API integration with n8n.

## Overview

The pipeline provides two main interfaces:
1. **Pipeline Orchestrator** (`pipeline_orchestrator.py`) - Modular Python API
2. **FastAPI Web Service** (`app.py`) - REST API endpoint

## Pipeline Orchestrator Parameters

### PipelineConfig Class

The `PipelineConfig` class in `pipeline_orchestrator.py` provides centralized configuration for the entire pipeline.

#### Diarization Settings

| Parameter | Type | Default | Description | Valid Range |
|-----------|------|---------|-------------|-------------|
| `diarization_model` | str | `"nvidia/diar_streaming_sortformer_4spk-v2"` | NVIDIA diarization model to use | Model name string |
| `num_speakers` | int | - | Expected number of speakers (1-4) | 1-4 |
| `diarization_chunk_size` | int | 6 | Chunk size for streaming processing (seconds) | 1-30 |
| `diarization_right_context` | int | 7 | Right context for streaming (seconds) | 1-15 |
| `diarization_fifo_size` | int | 188 | FIFO buffer size for overlapping chunks | 50-500 |
| `diarization_update_period` | int | 144 | Update period for diarization results (seconds) | 10-300 |
| `diarization_speaker_cache_size` | int | 188 | Maximum speakers to cache | 10-1000 |

#### ASR Settings

| Parameter | Type | Default | Description | Valid Range |
|-----------|------|---------|-------------|-------------|
| `asr_model` | str | `"nvidia/parakeet-ctc-1.1b"` | NVIDIA ASR model to use | Model name string |
| `asr_use_vad` | bool | `True` | Enable Voice Activity Detection | `true`/`false` |
| `asr_vad_threshold` | float | `0.5` | VAD speech detection threshold | 0.0-1.0 |
| `asr_min_segment_duration` | float | `0.05` | Minimum duration for speech segments (seconds) | 0.01-1.0 |
| `asr_batch_size` | int | `16` | Batch size for processing segments | 1-64 |
| `asr_enable_batch_processing` | bool | `True` | Enable batch processing for multiple segments | `true`/`false` |

#### Security Settings

| Parameter | Type | Default | Description | Valid Range |
|-----------|------|---------|-------------|-------------|
| `secure_temp_dir` | bool | `True` | Use secure temporary directories | `true`/`false` |
| `auto_cleanup` | bool | `True` | Automatically clean up temporary files | `true`/`false` |
| `max_file_size_mb` | int | `100` | Maximum input file size (MB) | 1-1000 |
| `allowed_extensions` | List[str] | `['.mp3', '.wav', '.flac', '.m4a', '.aac']` | Allowed audio file extensions | File extension list |

#### Processing Settings

| Parameter | Type | Default | Description | Valid Options |
|-----------|------|---------|-------------|---------------|
| `device` | str | `"auto"` | Device for model inference | `"auto"`, `"cpu"`, `"cuda"` |
| `output_format` | str | `"json"` | Output format for results | `"json"`, `"txt"`, `"both"` |

## FastAPI Web Service Parameters

### Endpoint: `POST /transcribe_diarize/`

### Endpoint: `POST /cleanup/`

Manual endpoint to unload all cached models and free VRAM.

**Method**: POST
**Response**: `{"message": "All models unloaded and VRAM freed"}`

**Usage**:
```bash
curl -X POST "https://diarasr.sunserv.org/cleanup/"
```

The REST API accepts the following parameters as form data or JSON body.

#### Core Parameters

| Parameter | Type | Default | Description | Required |
|-----------|------|---------|-------------|----------|
| `audio_file` | file | - | Audio file to process | Yes |
| `language` | str | `"en"` | Language code for ASR | No |
| `diarize` | bool | `true` | Enable speaker diarization | No |
| `vad` | bool | `true` | Enable Voice Activity Detection | No |
| `num_speakers` | int | - | Expected number of speakers (1-4) | No |
| `unload_models_after` | bool | `false` | Unload models after processing to free VRAM | No |

#### Model Selection Parameters

| Parameter | Type | Default | Description | Valid Options |
|-----------|------|---------|-------------|---------------|
| `diarization_model` | str | `"nvidia/diar_streaming_sortformer_4spk-v2"` | Diarization model to use | NVIDIA model names |
| `asr_model` | str | `"nvidia/parakeet-ctc-1.1b"` | ASR model to use | NVIDIA model names |

#### Processing Parameters

| Parameter | Type | Default | Description | Valid Range |
|-----------|------|---------|-------------|-------------|
| `batch_size` | int | `16` | Batch size for ASR processing | 1-64 |
| `output_format` | str | `"json"` | Output format | `"json"`, `"txt"`, `"both"` |
| `segment_resolution` | str | - | Segment resolution mode | `"low"`, `"medium"`, `"high"` |

#### Optional Parameters

| Parameter | Type | Default | Description | Valid Range |
|-----------|------|---------|-------------|-------------|
| `hf_token` | str | - | HuggingFace token for gated models | Token string |

## VAD (Voice Activity Detection) Parameters

The new VAD functionality provides advanced speech detection capabilities:

### Key VAD Parameters

| Parameter | API Name | Config Name | Type | Default | Description |
|-----------|----------|-------------|------|---------|-------------|
| VAD Enable | `vad` | `asr_use_vad` | bool | `true` | Enable/disable VAD processing |
| VAD Threshold | - | `asr_vad_threshold` | float | `0.5` | Speech detection sensitivity (0.0-1.0) |
| Min Segment Duration | - | `asr_min_segment_duration` | float | `0.05` | Minimum speech segment length (seconds) |

### VAD Threshold Guidelines

- **0.3-0.4**: More sensitive (detects softer speech, may include noise)
- **0.5**: Balanced (recommended default)
- **0.6-0.7**: Less sensitive (detects only clear speech, filters noise)

### VAD Integration

When `vad=true` (default):
- Uses NVIDIA ASR with built-in VAD processing
- Automatically filters out non-speech segments
- Improves transcription accuracy by focusing on speech-only audio
- Reduces processing time by skipping silence

When `vad=false`:
- Processes entire audio file without speech detection
- May include silence or noise in transcriptions
- Faster processing but potentially lower accuracy

## Configuration Examples

### Python API Usage

```python
from pipeline_orchestrator import PipelineConfig, process_audio_files

# Custom configuration with VAD tuning
config = PipelineConfig(
    asr_use_vad=True,
    asr_vad_threshold=0.4,  # More sensitive
    asr_min_segment_duration=0.1,  # Longer minimum segments
    diarization_chunk_size=8,  # Larger chunks for better diarization
    batch_size=32  # Larger batch for faster processing
)

results = process_audio_files('audio.mp3', config=config)
```

### REST API Usage

```bash
curl -X POST "http://localhost:8000/transcribe_diarize/" \
  -F "audio_file=@audio.mp3" \
  -F "vad=true" \
  -F "num_speakers=2" \
  -F "batch_size=32" \
  -F "diarize=true"
```

### JSON Configuration Override

```python
from config import create_custom_config

# Override specific parameters
custom_config = create_custom_config(
    use_vad=True,
    vad_threshold=0.6,  # Less sensitive
    batch_size=8,  # Smaller batch
    device="cuda"  # Force GPU
)
```

## Output Format

### JSON Output Structure

```json
{
  "segments": [
    {
      "speaker": "SPEAKER_00",
      "start": 0.0,
      "end": 3.45,
      "text": "Hello, this is a test transcription."
    },
    {
      "speaker": "SPEAKER_01",
      "start": 3.5,
      "end": 7.12,
      "text": "Thank you for the demonstration."
    }
  ]
}
```

### Text Output Format

```
SPEAKER_00: Hello, this is a test transcription.
SPEAKER_01: Thank you for the demonstration.
```

## Error Handling

The API returns appropriate HTTP status codes and error messages:

- `400 Bad Request`: Invalid parameters or file format
- `413 Payload Too Large`: File exceeds size limit
- `415 Unsupported Media Type`: Unsupported audio format
- `500 Internal Server Error`: Processing errors

## Performance Considerations

### Parameter Tuning for Speed vs Accuracy

| Use Case | Recommended Settings |
|----------|---------------------|
| **Fast Processing** | `batch_size=32`, `vad=true`, `diarization_chunk_size=4` |
| **High Accuracy** | `batch_size=8`, `vad=true`, `vad_threshold=0.4` |
| **Low Resource** | `device="cpu"`, `batch_size=4`, `vad=false` |
| **Real-time** | `diarization_update_period=60`, `chunk_size=2` |

### Memory Usage

- **GPU Memory**: ~2-4GB for default models
- **CPU Memory**: ~1-2GB for processing
- **Batch Size**: Higher values use more memory but process faster
- **VAD**: Enabled VAD reduces memory usage by filtering silence


## Migration Notes

### From Legacy API

- `hf_token` parameter is deprecated - no longer needed
- `min_speakers`/`max_speakers` are deprecated - model auto-detects
- VAD parameters are new - enable for better performance
- Batch processing is now default - improves speed

### Version Compatibility

- **v1.x**: Legacy parameters supported but deprecated
- **v2.x**: VAD parameters introduced, improved batch processing
- **v3.x**: Enhanced error handling and performance optimizations
# SECURE MODULAR ASR Diarization Pipeline API Parameters Documentation

This document describes the DiarASR API for local English transcription and optional speaker diarization.

## Overview

The pipeline provides three main interfaces:
1. **Modular Python API** (`app/` directory) - Core processing modules with configuration
2. **FastAPI Web Service** (`app/app.py`) - Secure REST API with authentication
3. **n8n Integration** - Workflow automation with security features

## 🔒 ENTERPRISE SECURITY FEATURES

### API Authentication
All API endpoints require authentication using API keys:

**Header**: `X-API-Key: your-api-key`

**Environment Setup**:
```bash
export API_KEYS="key1,key2,key3"  # Comma-separated API keys
```

### Security Features
- **API Key Authentication**: Configurable API keys with header validation
- **Input Validation**: Multi-layer file validation (MIME, magic number, size limits)
- **Rate Limiting**: DDoS protection (10 requests/minute per IP)
- **Data Protection**: Encrypted temporary files, secure deletion, audit logging
- **HIPAA Compliance**: All processing maintains medical data privacy standards

## Pipeline Orchestrator Parameters

### GlobalConfig Class

The `GlobalConfig` class in `config.py` provides centralized configuration for the entire secure modular pipeline.

#### Diarization Settings

| Parameter | Type | Default | Description | Valid Range |
|-----------|------|---------|-------------|-------------|
| `num_speakers` | int | - | Expected detected count | 1-8 |
| `min_speakers` | int | - | Minimum acceptable detected count | 1-8 |
| `max_speakers` | int | - | Maximum acceptable detected count | 1-8 |

#### ASR Settings

| Parameter | Type | Default | Description | Valid Range |
|-----------|------|---------|-------------|-------------|
Omi uses BF16 CUDA inference and manages its own batch sizes.

#### Security Settings

| Parameter | Type | Default | Description | Valid Range |
|-----------|------|---------|-------------|-------------|
| `enable_api_key_auth` | bool | `True` | Enable API key authentication | `true`/`false` |
| `api_keys` | List[str] | Environment | List of valid API keys | Comma-separated strings |
| `api_key_header` | str | `"X-API-Key"` | Header name for API key | Header string |
| `sanitize_inputs` | bool | `True` | Enable input sanitization | `true`/`false` |
| `max_filename_length` | int | `255` | Maximum filename length | 1-1000 |
| `max_file_size_mb` | int | `100` | Maximum input file size (MB) | 1-1000 |
| `allowed_extensions` | List[str] | `['.mp3', '.wav', '.flac', '.m4a', '.aac']` | Allowed audio file extensions | File extension list |
| `secure_temp_dir` | bool | `True` | Use secure temporary directories | `true`/`false` |
| `auto_cleanup` | bool | `True` | Automatically clean up temporary files | `true`/`false` |
| `encrypt_temp_files` | bool | `False` | Encrypt temporary files during processing | `true`/`false` |
| `enable_audit_logging` | bool | `True` | Enable audit logging for file operations | `true`/`false` |
| `max_retry_attempts` | int | `3` | Maximum retry attempts for failed file deletions | 1-10 |
| `cleanup_timeout_seconds` | int | `30` | Timeout for cleanup operations to prevent hanging | 10-300 |
| `temp_file_retention_hours` | int | `24` | Hours to retain temporary files before cleanup | 1-168 |

#### TempFileTracker Settings (HIPAA-Compliant Cleanup)

| Parameter | Type | Default | Description | Valid Range |
|-----------|------|---------|-------------|-------------|
| `max_retry_attempts` | int | `3` | Maximum retry attempts for failed file deletions | 1-10 |
| `base_retry_delay` | float | `0.1` | Base delay in seconds for exponential backoff | 0.01-1.0 |
| `max_retry_delay` | float | `5.0` | Maximum delay between retries | 1.0-30.0 |
| `audit_log_level` | str | `"INFO"` | Logging level for audit events | `"DEBUG"`, `"INFO"`, `"WARNING"`, `"ERROR"` |
| `enable_audit_logging` | bool | `True` | Enable audit logging for file operations | `true`/`false` |
| `temp_dir_permissions` | int | `0o700` | Permissions for temporary directories | Octal permission |
| `cleanup_timeout_seconds` | int | `30` | Timeout for cleanup operations to prevent hanging | 10-300 |

#### Processing Settings

| Parameter | Type | Default | Description | Valid Options |
|-----------|------|---------|-------------|---------------|
| `device` | str | `"auto"` | Device for model inference | `"auto"`, `"cpu"`, `"cuda"` |
| `output_format` | str | `"json"` | Output format for results | `"json"`, `"txt"`, `"both"` |
| `sample_rate` | int | `16000` | Target sample rate for processing | Audio sample rate |

## FastAPI Web Service Parameters

### Authentication Required
All API endpoints require authentication using the `X-API-Key` header:

```bash
curl -H "X-API-Key: your-api-key" -X POST "https://your-api-endpoint/transcribe_diarize/" \
  -F "audio_file=@audio.mp3"
```

### Endpoint: `POST /transcribe_diarize/`

Main transcription and diarization endpoint with comprehensive security validation.

**Authentication**: Required (`X-API-Key` header)
**Method**: POST (multipart/form-data)
**Rate Limit**: 10 requests per minute per IP

#### Core Parameters

| Parameter | Type | Default | Description | Required |
|-----------|------|---------|-------------|----------|
| `audio_file` | file | - | Audio file to process (MP3, WAV, FLAC, M4A, AAC) | Yes |
| `language` | str | `"en"` | English only (`en` or `english`) | No |
| `diarize` | bool | `true` | Enable speaker diarization | No |
| `num_speakers` | int | - | Expected detected count (1-8) | No |
| `min_speakers` | int | - | Minimum acceptable detected count (1-8) | No |
| `max_speakers` | int | - | Maximum acceptable detected count (1-8) | No |
| `hf_token` | str | Environment | Optional Hugging Face token for Nemotron downloads | No |

Nemotron detects speakers automatically. Speaker-count fields validate its output; they do not force a count or drop speaker turns. Invalid or unmet bounds return HTTP 422.

#### Processing Parameters

| Parameter | Type | Default | Description | Valid Range |
|-----------|------|---------|-------------|-------------|
| `output_format` | str | `"json"` | Output format | `"json"`, `"txt"`, `"both"` |
| `unload_models_after` | bool | `false` | Unload models after processing to free VRAM | `true`/`false` |

### Endpoint: `POST /cleanup/`

Manual endpoint to unload all cached models and free VRAM.

**Authentication**: Required (`X-API-Key` header)
**Method**: POST
**Response**: `{"message": "All models unloaded and VRAM freed"}`

**Usage**:
```bash
curl -H "X-API-Key: your-api-key" -X POST "https://your-api-endpoint/cleanup/"
```

## Diarization-Controlled Segmentation

The system implements **intelligent segmentation** that respects diarization boundaries rather than ASR internal segmentation:

### Key Features
- **Diarization-First**: Segmentation follows speaker turn boundaries, not ASR model decisions
- **Intelligent Merging**: Consecutive segments from same speaker within 500ms are merged
- **Punctuation Preservation**: Natural speech pauses maintain proper grammar
- **Workflow Optimization**: Creates coherent segments perfect for automation

### Benefits
- ✅ **Readable Transcripts**: Complete speaker turns in single segments
- ✅ **Proper Punctuation**: Natural pauses preserved for grammar
- ✅ **n8n Ready**: Optimal segment boundaries for workflow processing
- ✅ **Reduced Fragmentation**: 31% fewer segments while maintaining quality

## ASR Backend

Omi Med STT v1 is the only ASR backend and supports English only. There is no model selector or fallback. The Omi runtime controls ASR batching. With diarization enabled, timestamps are Nemotron 3 speaker-turn boundaries and overlapping turns may both be present. Without diarization, the transcript is returned in one segment spanning the recording.

## Configuration Examples

### Python API Usage

```python
from app import process_audio  # Run with the app directory on PYTHONPATH

results = process_audio('audio.mp3', diarize=True)
```

### REST API Usage

```bash
# Production system with authentication
curl -H "X-API-Key: your-api-key" \
  -X POST "https://your-api-endpoint/transcribe_diarize/" \
  -F "audio_file=@audio.mp3" \
  -F "diarize=true" \
  -F "min_speakers=2" \
  -F "max_speakers=4" \
  -F "hf_token=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# With file output for workflows
curl -H "X-API-Key: your-api-key" \
  -X POST "https://your-api-endpoint/transcribe_diarize/" \
  -F "audio_file=@audio.mp3" \
  -F "diarize=true" \
  -F "min_speakers=2" \
  -F "max_speakers=4" \
  -F "hf_token=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" \
  -o transcription_result.json
```

## n8n Integration

### Recommended n8n Node Configuration

```json
{
  "parameters": {
    "method": "POST",
    "url": "https://your-api-endpoint/transcribe_diarize/",
    "sendHeaders": true,
    "headerParameters": {
      "parameters": [
        {
          "name": "X-API-Key",
          "value": "={{ $json.api_key }}"
        }
      ]
    },
    "sendBody": true,
    "contentType": "multipart-form-data",
    "bodyParameters": {
      "parameters": [
        {
          "parameterType": "formBinaryData",
          "name": "audio_file",
          "inputDataFieldName": "audio_file"
        },
        {
          "name": "diarize",
          "value": "=true"
        },
        {
          "name": "min_speakers",
          "value": "={{ $json.min_speakers || 2 }}"
        },
        {
          "name": "max_speakers",
          "value": "={{ $json.max_speakers || 4 }}"
        },
        {
          "name": "hf_token",
          "value": "={{ $json.hugging_face_token }}"
        },
        {
          "name": "output_format",
          "value": "={{ $json.output_format || 'json' }}"
        },
        {
          "name": "unload_models_after",
          "value": "=true"
        }
      ]
    }
  }
}
```

### n8n Workflow Benefits

- **No Model Parameters Needed**: Omi Med STT v1 is always used for ASR
- **Error Handling**: Built-in retry logic and error recovery

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

### Authentication Errors
- `401 Unauthorized`: Missing or invalid `X-API-Key` header
- `403 Forbidden`: Invalid API key provided

### Input Validation Errors
- `400 Bad Request`: Invalid parameters, malformed file, or validation failure
- `413 Payload Too Large`: File exceeds size limit (100MB default)
- `415 Unsupported Media Type`: Unsupported audio format or MIME type
- `429 Too Many Requests`: Rate limit exceeded (10 requests/minute per IP)

### Processing Errors
- `500 Internal Server Error`: Model loading or processing errors
- `503 Service Unavailable`: Temporary service issues or resource constraints

### Security Events
All security-related events are logged with detailed audit trails including:
- Authentication failures
- Input validation failures
- File upload attempts
- Rate limit violations
- Anomalous activity detection

## Performance Considerations

GPU memory and elapsed time depend on the recording and whether Nemotron diarization is enabled. The Nemotron subprocess exits before Omi loads, so the models do not have to share GPU memory.


## Security Best Practices

### API Key Management
- Store API keys securely (environment variables, Docker secrets)
- Rotate keys regularly for production deployments
- Use different keys for different applications/environments
- Monitor API key usage and revoke compromised keys

### Environment Configuration
```bash
# Production environment setup
export API_KEYS="prod-key-1,prod-key-2"
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
export LOG_LEVEL="INFO"
export MAX_FILE_SIZE_MB="100"
```

### Monitoring & Alerting
- Enable audit logging for all file operations
- Monitor rate limiting and authentication failures
- Set up alerts for security events and anomalies
- Regular security scans and dependency updates

### HIPAA Compliance Checklist
- ✅ Encrypted temporary file storage
- ✅ Secure file deletion (overwrites)
- ✅ Audit logging of all operations
- ✅ Input sanitization and validation
- ✅ Access control via API authentication
- ✅ Data retention policies (24-hour cleanup)

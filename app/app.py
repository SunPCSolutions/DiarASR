from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import os
import tempfile
import torch
import nemo.collections.asr as nemo_asr
from pydub import AudioSegment
import torchaudio
from transformers import pipeline
from config import get_config, load_api_keys
from logging_config import get_app_logger, get_security_logger
import multiprocessing
import json
import subprocess
import sys
import mimetypes
import time
import re
from collections import defaultdict


app = FastAPI()

# Initialize logging
logger = get_app_logger()
security_logger = get_security_logger()

# Rate limiting storage (in-memory, resets on restart)
rate_limit_store = defaultdict(list)

# Validate environment variables on startup first
try:
    from config import validate_environment
    validated_env = validate_environment()
    logger.info("Environment validation successful")
    if validated_env:
        logger.info(f"Validated environment variables: {list(validated_env.keys())}")
except Exception as e:
    logger.error(f"Environment validation failed: {e}")
    raise

# Get config after environment validation to pick up API keys
config = get_config()

# Load API keys dynamically after environment is set
config.security.api_keys = load_api_keys()

# Configure CORS
if config.security.cors_enabled:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.security.cors_origins,
        allow_credentials=config.security.cors_allow_credentials,
        allow_methods=config.security.cors_methods,
        allow_headers=config.security.cors_headers,
    )

# Model loading moved to worker subprocess for memory isolation

def validate_file_extension(filename: str) -> None:
    """Validate file extension against allowed extensions."""
    if not filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    _, ext = os.path.splitext(filename.lower())
    if ext not in config.processing.allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"File extension '{ext}' not allowed. Allowed: {', '.join(config.processing.allowed_extensions)}"
        )

def validate_mime_type(content_type: str) -> None:
    """Validate MIME type against allowed types."""
    if content_type not in config.processing.allowed_mime_types:
        raise HTTPException(
            status_code=400,
            detail=f"MIME type '{content_type}' not allowed. Allowed: {', '.join(config.processing.allowed_mime_types)}"
        )

def validate_magic_number(file_content: bytes) -> None:
    """Validate file magic number to ensure it's an audio file."""
    if len(file_content) < 12:
        raise HTTPException(status_code=400, detail="File too small to be a valid audio file")

    # Magic numbers for audio formats
    magic_signatures = {
        b'RIFF': 'wav',  # WAV files start with RIFF
        b'ID3': 'mp3',   # MP3 with ID3 tag
        b'\xFF\xFB': 'mp3',  # MP3 frame sync
        b'\xFF\xF3': 'mp3',  # MP3 frame sync
        b'\xFF\xF2': 'mp3',  # MP3 frame sync
        b'fLaC': 'flac', # FLAC
        b'\x00\x00\x00\x20ftypM4A': 'm4a',  # M4A (truncated for check)
        b'\x00\x00\x00\x18ftypM4A': 'm4a',  # M4A variant
    }

    # Check first few bytes
    file_start = file_content[:12]
    is_valid_audio = False

    for signature, format_type in magic_signatures.items():
        if file_start.startswith(signature):
            is_valid_audio = True
            break

    # Additional check for MP3 without ID3
    if not is_valid_audio and len(file_content) >= 2:
        # Check for MP3 frame sync in first few bytes
        for i in range(min(10, len(file_content) - 1)):
            if file_content[i:i+2] in [b'\xFF\xFB', b'\xFF\xF3', b'\xFF\xF2', b'\xFF\xFA', b'\xFF\xFC']:
                is_valid_audio = True
                break

    if not is_valid_audio:
        raise HTTPException(status_code=400, detail="File does not appear to be a valid audio file")

def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal and other attacks."""
    if not config.security.sanitize_inputs:
        return filename

    if not filename:
        raise HTTPException(status_code=400, detail="Filename cannot be empty")

    # Check length
    if len(filename) > config.security.max_filename_length:
        raise HTTPException(
            status_code=400,
            detail=f"Filename too long. Maximum length: {config.security.max_filename_length}"
        )

    # Remove path separators and dangerous characters
    filename = os.path.basename(filename)  # Remove any path components

    # Validate against allowed characters
    if not re.match(config.security.allowed_filename_chars, filename):
        raise HTTPException(
            status_code=400,
            detail="Filename contains invalid characters"
        )

    return filename

def sanitize_parameter(value: str, param_name: str = "parameter") -> str:
    """Sanitize parameter value to prevent injection attacks."""
    if not config.security.sanitize_inputs:
        return value

    if not isinstance(value, str):
        return value

    # Check length
    if len(value) > config.security.max_parameter_length:
        raise HTTPException(
            status_code=400,
            detail=f"{param_name} too long. Maximum length: {config.security.max_parameter_length}"
        )

    # Remove potentially dangerous characters
    # Allow alphanumeric, spaces, hyphens, underscores, dots
    sanitized = re.sub(r'[^\w\s\-_.]', '', value)

    return sanitized.strip()

def check_rate_limit(client_ip: str) -> None:
    """Check if client has exceeded rate limit."""
    current_time = time.time()
    window_start = current_time - config.processing.rate_limit_window_seconds

    # Clean old requests
    rate_limit_store[client_ip] = [
        req_time for req_time in rate_limit_store[client_ip]
        if req_time > window_start
    ]

    # Check if limit exceeded
    if len(rate_limit_store[client_ip]) >= config.processing.rate_limit_requests:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Maximum {config.processing.rate_limit_requests} requests per {config.processing.rate_limit_window_seconds} seconds"
        )

    # Add current request
    rate_limit_store[client_ip].append(current_time)

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Rate limiting middleware."""
    client_ip = request.client.host if request.client else "unknown"
    try:
        check_rate_limit(client_ip)
    except HTTPException as e:
        # Log rate limit exceeded
        security_logger.log_rate_limit_exceeded(request.url.path, client_ip, len(rate_limit_store[client_ip]))
        raise
    response = await call_next(request)
    return response

@app.middleware("http")
async def body_size_limit_middleware(request: Request, call_next):
    """Body size limit middleware with enhanced validation."""
    content_length = request.headers.get("content-length")

    if content_length:
        try:
            size_bytes = int(content_length)
            size_mb = size_bytes / (1024 * 1024)

            # Check against configured maximum
            if size_mb > config.processing.max_file_size_mb:
                return JSONResponse(
                    status_code=413,
                    content={"detail": f"Request body too large. Maximum size: {config.processing.max_file_size_mb}MB"}
                )

            # Additional check: prevent extremely large headers
            if size_bytes > 500 * 1024 * 1024:  # 500MB absolute limit
                return JSONResponse(
                    status_code=413,
                    content={"detail": "Request body exceeds absolute size limit"}
                )

        except (ValueError, OverflowError):
            return JSONResponse(
                status_code=400,
                content={"detail": "Invalid Content-Length header"}
            )

    # Check for multipart form data size limits
    content_type = request.headers.get("content-type", "").lower()
    if "multipart/form-data" in content_type:
        # For file uploads, we rely on the per-file validation in the endpoint
        # But we can add additional checks here if needed
        pass

    response = await call_next(request)

    # Check response size (optional - can be expensive for large responses)
    if hasattr(response, 'body') and response.body:
        response_size_mb = len(response.body) / (1024 * 1024)
        if response_size_mb > config.processing.max_file_size_mb * 2:  # Allow larger responses
            logger.warning(f"Large response generated: {response_size_mb:.2f}MB")

    return response

@app.middleware("http")
async def api_key_auth_middleware(request: Request, call_next):
    """API key authentication middleware."""
    if not config.security.enable_api_key_auth:
        response = await call_next(request)
        return response

    # Skip authentication for health check or documentation endpoints
    if request.url.path in ["/docs", "/redoc", "/openapi.json"]:
        response = await call_next(request)
        return response

    api_key = request.headers.get(config.security.api_key_header)
    if not api_key:
        return JSONResponse(
            status_code=401,
            content={"detail": f"API key required in {config.security.api_key_header} header"}
        )

    if api_key not in config.security.api_keys:
        return JSONResponse(
            status_code=403,
            content={"detail": "Invalid API key"}
        )

    response = await call_next(request)
    return response

@app.middleware("http")
async def security_headers_middleware(request: Request, call_next):
    """Security headers middleware."""
    response = await call_next(request)

    if config.security.security_headers_enabled:
        # HSTS (HTTP Strict Transport Security)
        if config.security.hsts_enabled:
            hsts_value = f"max-age={config.security.hsts_max_age}"
            if config.security.hsts_include_subdomains:
                hsts_value += "; includeSubDomains"
            if config.security.hsts_preload:
                hsts_value += "; preload"
            response.headers["Strict-Transport-Security"] = hsts_value

        # Content Security Policy
        if config.security.csp_enabled:
            response.headers["Content-Security-Policy"] = config.security.csp_policy

        # X-Frame-Options
        response.headers["X-Frame-Options"] = config.security.x_frame_options

        # X-Content-Type-Options
        response.headers["X-Content-Type-Options"] = config.security.x_content_type_options

        # Referrer-Policy
        response.headers["Referrer-Policy"] = config.security.referrer_policy

        # Additional security headers
        response.headers["X-XSS-Protection"] = "1; mode=block"

    return response

def process_audio(
    audio_path: str,
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    diarization_model: Optional[str] = None,
    asr_model: Optional[str] = None,
    language: Optional[str] = None,
    diarize: bool = True,
    vad: Optional[bool] = None,
    unload_models_after: bool = False,
    segment_resolution: Optional[str] = None,
    batch_size: Optional[int] = None,
    output_format: Optional[str] = None,
    hf_token: Optional[str] = None
):
    """Run inference in subprocess for complete memory isolation"""

    # Prepare request data for worker
    request_data = {
        'audio_path': audio_path,
        'num_speakers': num_speakers,
        'min_speakers': min_speakers,
        'max_speakers': max_speakers,
        'diarization_model': diarization_model,
        'asr_model': asr_model,
        'language': language,
        'diarize': diarize,
        'vad': vad,
        'segment_resolution': segment_resolution,
        'batch_size': batch_size,
        'output_format': output_format,
        'hf_token': hf_token or os.getenv('HF_TOKEN', '')  # Pass HF_TOKEN to worker
    }

    try:
        # Debug: Print request_data being sent to worker
        print(f"DEBUG: Sending request_data to worker: {request_data}")

        # Run inference in subprocess
        env = os.environ.copy()
        env['HF_TOKEN'] = os.getenv('HF_TOKEN', '')

        result = subprocess.run(
            [sys.executable, 'worker.py'],
            input=json.dumps(request_data),
            capture_output=True,
            text=True,
            env=env,
            cwd=os.getcwd(),
            timeout=600  # 10 minute timeout
        )

        # Log worker stderr (logs) for debugging
        if result.stderr:
            logger.warning("Worker stderr: %s", result.stderr.strip())

        if result.returncode != 0:
            error_msg = f"Worker process failed: {result.stderr}"
            logger.error("Worker process failed: %s", error_msg)
            raise HTTPException(status_code=500, detail=error_msg)

        # Parse result
        output = result.stdout.strip()
        if not output:
            raise HTTPException(status_code=500, detail="Worker process returned no output")

        try:
            worker_result = json.loads(output)
        except json.JSONDecodeError as e:
            error_msg = f"Failed to parse worker output: {e}\nOutput: {output}"
            logger.error("Failed to parse worker output: %s", error_msg)
            raise HTTPException(status_code=500, detail=error_msg)

        if 'error' in worker_result:
            raise HTTPException(status_code=500, detail=worker_result['error'])

        return worker_result

    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=500, detail="Inference timed out")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Subprocess error: {str(e)}")

# Memory management now handled by subprocess isolation

@app.post("/transcribe_diarize/")
async def transcribe_diarize(
    audio_file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    diarize: bool = Form(True),
    vad: Optional[str] = Form(None),
    num_speakers: Optional[int] = Form(None),
    min_speakers: Optional[int] = Form(None),
    max_speakers: Optional[int] = Form(None),
    unload_models_after: Optional[str] = Form(None),
    hf_token: Optional[str] = Form(None),
    output_format: Optional[str] = Form(None),
    segment_resolution: Optional[str] = Form(None),
    batch_size: Optional[int] = Form(None),
    diarization_model: Optional[str] = Form(None),
    asr_model: Optional[str] = Form(None),
    save_to_file: Optional[str] = Form(None)
  ):
    # Sanitize and validate filename
    if audio_file.filename:
        try:
            sanitized_filename = sanitize_filename(audio_file.filename)
            validate_file_extension(sanitized_filename)
        except HTTPException as e:
            # Log validation failure
            security_logger.log_validation_failure(
                'filename',
                audio_file.filename,
                str(e.detail)
            )
            raise
    else:
        security_logger.log_validation_failure(
            'filename',
            'None',
            'Filename is required'
        )
        raise HTTPException(status_code=400, detail="Filename is required")

    # Sanitize string parameters
    if language:
        language = sanitize_parameter(language, "language")
    if output_format:
        output_format = sanitize_parameter(output_format, "output_format")
    if segment_resolution:
        segment_resolution = sanitize_parameter(segment_resolution, "segment_resolution")
    if diarization_model:
        diarization_model = sanitize_parameter(diarization_model, "diarization_model")
    if asr_model:
        asr_model = sanitize_parameter(asr_model, "asr_model")
    if save_to_file:
        save_to_file = sanitize_parameter(save_to_file, "save_to_file")

    # Validate MIME type
    if audio_file.content_type:
        try:
            validate_mime_type(audio_file.content_type)
        except HTTPException as e:
            security_logger.log_validation_failure(
                'mime_type',
                audio_file.content_type,
                str(e.detail)
            )
            raise

    # Read file content for validation
    file_content = await audio_file.read()

    # Check file size
    file_size = len(file_content)
    file_size_mb = file_size / (1024 * 1024)
    if file_size_mb > config.processing.max_file_size_mb:
        security_logger.log_validation_failure(
            'file_size',
            f"{file_size_mb:.2f}MB",
            f"File too large. Maximum size: {config.processing.max_file_size_mb}MB"
        )
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {config.processing.max_file_size_mb}MB, got: {file_size_mb:.2f}MB"
        )

    # Validate magic number
    try:
        validate_magic_number(file_content)
    except HTTPException as e:
        security_logger.log_validation_failure(
            'magic_number',
            f"File starting with: {file_content[:20].hex()}",
            str(e.detail)
        )
        raise

    # Log successful file upload
    security_logger.log_file_upload_attempt(
        sanitized_filename,
        file_size,
        True,  # Success
        user_id=None  # Could be extracted from API key or auth
    )

    # Increment security metrics
    security_logger.increment_metric('file_uploads_total')
    security_logger.record_metric('file_size_bytes', file_size)

    # Convert string parameters to appropriate types (handle n8n format)
    unload_models_bool = False
    if isinstance(unload_models_after, str):
        # Handle n8n format like "=true" or "=false"
        clean_value = unload_models_after.lstrip('=')
        unload_models_bool = clean_value.lower() in ('true', '1', 'yes', 'on')
        logger.debug("unload_models_after converted from '%s' to %s", unload_models_after, unload_models_bool)
    elif isinstance(unload_models_after, bool):
        unload_models_bool = unload_models_after

    # Convert vad parameter
    vad_bool = None
    if isinstance(vad, str):
        clean_vad = vad.lstrip('=')
        vad_bool = clean_vad.lower() in ('true', '1', 'yes', 'on')
    elif isinstance(vad, bool):
        vad_bool = vad

    # Save uploaded file temporarily
    suffix = os.path.splitext(sanitized_filename)[1] if sanitized_filename else ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(file_content)
        temp_path = temp_file.name

    try:
        result = process_audio(
            audio_path=temp_path,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            diarization_model=diarization_model,
            asr_model=asr_model,
            language=language,
            diarize=diarize,
            vad=vad_bool,
            unload_models_after=unload_models_bool,  # This is now the converted boolean
            segment_resolution=segment_resolution,
            batch_size=batch_size,
            output_format=output_format,
            hf_token=hf_token
        )
        return result
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
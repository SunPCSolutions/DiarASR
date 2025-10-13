from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from typing import Optional
import os
import tempfile
import torch
import nemo.collections.asr as nemo_asr
from pydub import AudioSegment
import torchaudio
from transformers import pipeline
from nvidia_asr import NvidiaASR
from config import get_config
import multiprocessing
import json
import subprocess
import sys

# Set ultra-aggressive CUDA memory management - minimize caching
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,garbage_collection_threshold:0.6,roundup_power2_divisions:1'

app = FastAPI()

# Model loading moved to worker subprocess for memory isolation

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
    output_format: Optional[str] = None
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
        'output_format': output_format
    }

    try:
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

        # Print worker stderr (logs) to our stderr for debugging
        if result.stderr:
            print("Worker logs:", result.stderr, file=sys.stderr)

        if result.returncode != 0:
            error_msg = f"Worker process failed: {result.stderr}"
            print(error_msg)
            raise HTTPException(status_code=500, detail=error_msg)

        # Parse result
        output = result.stdout.strip()
        if not output:
            raise HTTPException(status_code=500, detail="Worker process returned no output")

        try:
            worker_result = json.loads(output)
        except json.JSONDecodeError as e:
            error_msg = f"Failed to parse worker output: {e}\nOutput: {output}"
            print(error_msg)
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
    # Convert string parameters to appropriate types (handle n8n format)
    unload_models_bool = False
    if isinstance(unload_models_after, str):
        # Handle n8n format like "=true" or "=false"
        clean_value = unload_models_after.lstrip('=')
        unload_models_bool = clean_value.lower() in ('true', '1', 'yes', 'on')
        print(f"DEBUG: unload_models_after converted from '{unload_models_after}' to {unload_models_bool}")
    elif isinstance(unload_models_after, bool):
        unload_models_bool = unload_models_after

    # Save uploaded file temporarily
    suffix = os.path.splitext(audio_file.filename)[1] if audio_file.filename else ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(await audio_file.read())
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
            vad=vad,
            unload_models_after=unload_models_bool,  # This is now the converted boolean
            segment_resolution=segment_resolution,
            batch_size=batch_size,
            output_format=output_format
        )
        return result
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
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

# Set aggressive CUDA memory management
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

app = FastAPI()

# Model caches
diarization_cache = {}
asr_cache = {}

def get_diarization_model(model_name: str):
    if model_name not in diarization_cache:
        if model_name == "nvidia/diar_streaming_sortformer_4spk-v2":
            import nemo.collections.asr as nemo_asr
            model = nemo_asr.models.SortformerEncLabelModel.from_pretrained(model_name)
            diarization_cache[model_name] = model
        else:
            raise ValueError("Only nvidia/diar_streaming_sortformer_4spk-v2 is supported")
    return diarization_cache[model_name]

def get_asr_model(model_name: str):
    if model_name not in asr_cache:
        if model_name == "nvidia/parakeet-ctc-1.1b":
            asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name)
            asr_cache[model_name] = asr_model
        else:
            raise ValueError("Only nvidia/parakeet-ctc-1.1b is supported")
    return asr_cache[model_name]

def transcribe_segment(asr_model, file_path: str) -> str:
    # NeMo ASR model
    transcription_output = asr_model.transcribe([file_path])
    transcription = transcription_output[0].text.strip()
    return transcription

def process_audio(
    audio_path: str,
    num_speakers: Optional[int] = None,
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
    # Load configuration
    config = get_config()

    # Set defaults from config
    diarization_model = diarization_model or config.diarization.model_name
    asr_model = asr_model or config.asr.asr_model_name
    language = language or config.asr.language
    vad = vad if vad is not None else config.asr.use_vad
    batch_size = batch_size or config.asr.batch_size
    output_format = output_format or config.processing.output_format

    # Validate num_speakers parameter
    if num_speakers is not None:
        if not (1 <= num_speakers <= 4):
            raise HTTPException(
                status_code=400,
                detail="num_speakers must be between 1 and 4 (Sortformer model limit)"
            )
        print(f"Expected number of speakers: {num_speakers}")

    try:
        # Convert audio to 16kHz mono wav
        audio = AudioSegment.from_file(audio_path)
        audio = audio.set_channels(1).set_frame_rate(config.processing.sample_rate)
        converted_path = audio_path.replace(os.path.splitext(audio_path)[1], "_converted.wav")
        audio.export(converted_path, format=config.processing.audio_format)

        results = []

        # Load audio for segmentation (needed for both diarize and non-diarize cases)
        waveform, sample_rate = torchaudio.load(converted_path)

        # Initialize ASR components
        asr_component = None
        asr_model_instance = None

        if vad:
            # Use NvidiaASR with VAD support
            asr_component = NvidiaASR(
                asr_model_name=config.asr.asr_model_name,
                device=config.asr.device,
                use_vad=config.asr.use_vad,
                vad_threshold=config.asr.vad_threshold,
                min_segment_duration=config.asr.min_segment_duration,
                batch_size=batch_size,
                enable_batch_processing=config.asr.enable_batch_processing
            )
        else:
            # Use direct NeMo ASR model
            asr_model_instance = get_asr_model(asr_model)

        if diarize:
            # Use NvidiaDiarization class for proper parsing
            from nvidia_diarization import NvidiaDiarization
            # Use balanced parameters for better accuracy without excessive processing time
            diarizer = NvidiaDiarization(
                model_name=diarization_model,
                device=config.asr.device,
                chunk_size=10,  # Moderate chunking (10 * 80ms = 0.8s)
                right_context=5,  # Balanced context (5 * 80ms = 0.4s)
                fifo_size=20,  # Reasonable FIFO buffer
                update_period=10,  # Balanced update frequency
                speaker_cache_size=50,  # Adequate cache size
                num_speakers=num_speakers  # Pass expected number of speakers
            )
            speaker_segments = diarizer.run_offline_diarization(converted_path)

            print(f"Parsed speaker segments type: {type(speaker_segments)}")
            print(f"Parsed speaker segments length: {len(speaker_segments) if hasattr(speaker_segments, '__len__') else 'N/A'}")
            if speaker_segments:
                print(f"First parsed segment: {speaker_segments[0]}")

            for segment in speaker_segments:
                start_time = segment['start']
                end_time = segment['end']
                speaker = segment['speaker']
                segment_duration = end_time - start_time

                # Skip segments that are too short for ASR (minimum ~200ms for reliable transcription)
                min_segment_duration = 0.2  # 200ms minimum to avoid noise/artifacts
                if segment_duration < min_segment_duration:
                    print(f"Skipping {speaker} segment ({segment_duration:.3f}s) - too short for ASR")
                    continue

                start_sample = int(start_time * sample_rate)
                end_sample = int(end_time * sample_rate)
                segment_waveform = waveform[:, start_sample:end_sample]

                # Transcribe segment
                try:
                    if vad and asr_component is not None:
                        # Use NvidiaASR for transcription (handles VAD internally if needed)
                        transcription_result = asr_component.transcribe_segment(segment_waveform, sample_rate)
                        # transcribe_segment returns just text, not a dict
                        if transcription_result and transcription_result.strip():
                            results.append({
                                'text': transcription_result.strip(),
                                'start': start_time,
                                'end': end_time,
                                'speaker': speaker
                            })
                    elif asr_model_instance is not None:
                        # Save temporary segment for NeMo
                        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                        torchaudio.save(temp_file.name, segment_waveform, sample_rate)
                        transcription = transcribe_segment(asr_model_instance, temp_file.name)
                        os.unlink(temp_file.name)
                        if transcription and transcription.strip():
                            results.append({
                                'text': transcription.strip(),
                                'start': start_time,
                                'end': end_time,
                                'speaker': speaker
                            })
                    else:
                        # No transcription available
                        pass
                except Exception as e:
                    print(f"Error transcribing segment: {e}")
        else:
            # Just ASR without diarization
            if vad and asr_component is not None:
                # Use NvidiaASR with VAD
                asr_result = asr_component.transcribe_file(converted_path)
                print(f"ASR result type: {type(asr_result)}")
                print(f"ASR result keys: {asr_result.keys() if hasattr(asr_result, 'keys') else 'No keys'}")
                if isinstance(asr_result, dict) and 'segments' in asr_result:
                    for segment in asr_result['segments']:
                        results.append({
                            'text': segment['text'],
                            'start': segment['start'],
                            'end': segment['end'],
                            'speaker': 'SPEAKER_00'
                        })
                else:
                    print(f"Unexpected ASR result format: {asr_result}")
            elif asr_model_instance is not None:
                # Use direct NeMo ASR
                asr_model_instance = get_asr_model(asr_model)
                transcription_output = asr_model_instance.transcribe([converted_path], timestamps=True)
                if hasattr(transcription_output[0], 'timestamp') and 'segment' in transcription_output[0].timestamp:
                    for stamp in transcription_output[0].timestamp['segment']:
                        results.append({
                            'text': stamp['segment'],
                            'start': stamp['start'],
                            'end': stamp['end'],
                            'speaker': 'SPEAKER_00'
                        })
                else:
                    # Fallback to full text
                    duration = audio.duration_seconds
                    results.append({
                        'text': transcription_output[0].text,
                        'start': 0.0,
                        'end': duration,
                        'speaker': 'SPEAKER_00'
                    })

        # Sort results by timestamp for chronological conversation order
        results_sorted = sorted(results, key=lambda x: x['start'])

        # Clean up converted file
        if os.path.exists(converted_path):
            os.unlink(converted_path)

        # Clean up ASR components
        if asr_component is not None:
            asr_component.cleanup()

        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {"segments": results_sorted}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Unload models if requested
        if unload_models_after:
            unload_models()

def unload_models():
    """Unload all cached models to free VRAM."""
    global diarization_cache, asr_cache

    # Clear diarization models
    for model_name, model in list(diarization_cache.items()):
        try:
            # Force garbage collection of model
            del model
            print(f"Unloaded diarization model: {model_name}")
        except Exception as e:
            print(f"Error unloading diarization model {model_name}: {e}")

    # Clear ASR models
    for model_name, model in list(asr_cache.items()):
        try:
            # Force garbage collection of model
            del model
            print(f"Unloaded ASR model: {model_name}")
        except Exception as e:
            print(f"Error unloading ASR model {model_name}: {e}")

    # Clear caches
    diarization_cache.clear()
    asr_cache.clear()

    # Force aggressive GPU memory cleanup
    if torch.cuda.is_available():
        # Force garbage collection first
        import gc
        gc.collect()

        # Reset peak memory stats
        torch.cuda.reset_peak_memory_stats()

        # Multiple empty_cache calls with synchronization
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # Force memory deallocation by creating and deleting a large tensor
        try:
            # Allocate a small tensor to force memory manager to reclaim
            dummy = torch.zeros(1, device='cuda')
            del dummy
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        except:
            pass

        # Final cleanup
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # Get memory info
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3

        print(f"GPU memory after cleanup: {allocated:.2f}GB used, {reserved:.2f}GB reserved")

        # Try to force CUDA context reset (more aggressive)
        try:
            # This is a more aggressive approach - reset the CUDA context
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()
            print("CUDA IPC collection completed")
        except:
            pass

        # Get actual GPU memory info
        try:
            free_memory, total_memory = torch.cuda.mem_get_info()
            free_gb = free_memory / 1024**3
            total_gb = total_memory / 1024**3
            print(f"GPU memory info: {free_gb:.2f}GB free / {total_gb:.2f}GB total")
        except:
            print("Could not get CUDA memory info")

        # Additional aggressive cleanup attempts
        try:
            # Try to force deallocation by creating/destroying CUDA context
            if torch.cuda.device_count() > 0:
                current_device = torch.cuda.current_device()
                # This is a best-effort attempt to free memory
                torch.cuda.set_device(current_device)
                print("CUDA device context reset attempted")
        except:
            pass

        print("Note: PyTorch CUDA allocator caches memory for performance.")
        print("Memory may not return to system immediately but is available for other applications.")

    print("All models unloaded and VRAM freed")

@app.post("/cleanup/")
async def cleanup_models():
    """Endpoint to manually unload all models and free VRAM."""
    unload_models()
    return {"message": "All models unloaded and VRAM freed"}

@app.post("/transcribe_diarize/")
async def transcribe_diarize(
    audio_file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    diarize: bool = Form(True),
    vad: Optional[bool] = Form(None),
    num_speakers: Optional[int] = Form(None),
    unload_models_after: Optional[str] = Form(None),
    hf_token: Optional[str] = Form(None),
    output_format: Optional[str] = Form(None),
    segment_resolution: Optional[str] = Form(None),
    batch_size: Optional[int] = Form(None),
    diarization_model: Optional[str] = Form(None),
    asr_model: Optional[str] = Form(None)
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
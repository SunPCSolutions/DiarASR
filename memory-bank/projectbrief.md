# Project Brief: NVIDIA ASR Diarization Pipeline

## Project Overview
A production-ready, secure audio processing pipeline that combines speaker diarization and automatic speech recognition using NVIDIA's NeMo toolkit. The system processes audio files to identify speakers and transcribe their speech with enterprise-grade security and performance.

## Core Objectives
- **Speaker Diarization**: Identify and separate different speakers in audio recordings
- **Speech Recognition**: Transcribe speech to text with high accuracy
- **Voice Activity Detection**: Filter out non-speech segments for cleaner transcription
- **Security**: Process sensitive audio data with zero-overwrite file deletion and secure temporary handling
- **Scalability**: Support batch processing and GPU acceleration for production workloads

## Key Features
- **NVIDIA Models**: Uses `nvidia/parakeet-ctc-1.1b` for ASR and `nvidia/diar_streaming_sortformer_4spk-v2` for diarization
- **Modular Architecture**: Separate components for diarization, ASR, and orchestration
- **Configurable Pipeline**: Runtime configuration for different processing modes
- **API Integration**: FastAPI web service for n8n workflow integration
- **Secure Processing**: In-memory processing with secure temporary file cleanup

## Success Criteria
- Process audio files up to 100MB with <5% WER (Word Error Rate)
- Support 4 concurrent speakers with accurate speaker attribution
- Maintain <1GB memory footprint per processing job
- Provide REST API compatible with existing n8n workflows
- Ensure zero data leakage with secure file handling

## Technical Scope
- **Languages**: Python 3.12+
- **Framework**: FastAPI for web service, NeMo for ML models
- **Infrastructure**: CUDA 12.8+ compatible, GPU acceleration required
- **Security**: HIPAA-compliant audio processing with secure cleanup
- **Performance**: Real-time processing for <30 second audio segments

## Business Value
- Enable automated transcription of medical consultations with speaker identification
- Reduce manual transcription costs by 90%
- Provide accurate, searchable medical records
- Ensure patient data privacy and security compliance
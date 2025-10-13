# Debug History: NVIDIA ASR Diarization Pipeline

## Major Issues Resolved

### Issue 1: CUDA/cuDNN Compatibility Conflicts
**Date**: October 12, 2025
**Problem**: `Unable to load any of {libcudnn_cnn.so.9.1.0, libcudnn_cnn.so.9.1, libcudnn_cnn.so.9, libcudnn_cnn.so}`
**Root Cause**: faster-whisper requires cuDNN 8.x, but Blackwell GPUs require cuDNN 9.x
**Impact**: Complete pipeline failure, unable to load ASR models

**Debugging Steps**:
1. Identified cuDNN version mismatch between model requirements and GPU compatibility
2. Tested multiple PyTorch/CUDA combinations
3. Verified Blackwell GPU requires CUDA 12.8+ with cuDNN 9.x
4. Confirmed faster-whisper incompatible with cuDNN 9.x

**Resolution**:
- Switched from Systran/faster-whisper-medium.en to NVIDIA Parakeet CTC 1.1B
- Used PyTorch nightly with CUDA 12.8 support
- Verified compatibility with Blackwell GPUs

**Prevention**: Always verify GPU compatibility before model selection

---

### Issue 2: PyTorch Version Conflicts
**Date**: October 12, 2025
**Problem**: `torch 2.7.0+cu128 requires nvidia-cudnn-cu12==9.7.1.26, but you have nvidia-cudnn-cu12 8.9.7.29`
**Root Cause**: Pip package conflicts between different CUDA versions
**Impact**: Dependency resolution failures, installation impossible

**Debugging Steps**:
1. Analyzed pip dependency resolver conflicts
2. Tested manual cuDNN installation vs pip packages
3. Verified Docker approach works with manual cuDNN installation
4. Identified venv isolation prevents system CUDA modifications

**Resolution**:
- Fresh venv installation with compatible PyTorch version
- Used `torch==2.8.0+cu121` with compatible cuDNN
- Verified all dependencies resolve correctly

**Prevention**: Use fresh venv for complex ML dependency installations

---

### Issue 3: VAD Integration Compatibility
**Date**: October 12, 2025
**Problem**: `vad_multilingual_marblenet` not available in faster-whisper
**Root Cause**: faster-whisper uses CTranslate2, not native PyTorch models
**Impact**: VAD functionality unavailable, reduced transcription accuracy

**Debugging Steps**:
1. Researched VAD options in faster-whisper
2. Found CTranslate2 limitations for VAD integration
3. Tested alternative VAD libraries (silero-vad)
4. Verified NVIDIA NeMo VAD integration capabilities

**Resolution**:
- Migrated to NVIDIA Parakeet CTC with built-in VAD support
- Integrated `vad_multilingual_marblenet` into ASR pipeline
- Maintained API compatibility with VAD parameter

**Prevention**: Verify VAD compatibility during model selection phase

---

### Issue 4: Memory Management Issues
**Date**: October 12, 2025
**Problem**: GPU memory accumulation during batch processing
**Root Cause**: PyTorch tensors not properly released between operations
**Impact**: Out-of-memory errors on consecutive processing jobs

**Debugging Steps**:
1. Monitored GPU memory usage with `nvidia-smi`
2. Identified tensor references preventing garbage collection
3. Tested explicit cleanup calls
4. Verified model disposal patterns

**Resolution**:
- Added `torch.cuda.empty_cache()` after each processing job
- Implemented explicit model cleanup in finally blocks
- Added garbage collection calls: `gc.collect()`
- Set model references to `None` after use

**Prevention**: Always include GPU memory cleanup in ML processing pipelines

---

### Issue 5: Temporary File Security
**Date**: October 12, 2025
**Problem**: Sensitive audio data potentially recoverable from temporary files
**Root Cause**: Standard tempfile deletion doesn't overwrite data
**Impact**: HIPAA compliance violation risk

**Debugging Steps**:
1. Analyzed temporary file cleanup behavior
2. Researched secure deletion methods
3. Tested forensic recovery prevention
4. Verified HIPAA compliance requirements

**Resolution**:
- Implemented `SecureTempManager` with zero-overwrite deletion
- Added file sanitization before removal
- Set restrictive permissions (0o700) on temp directories
- Integrated secure cleanup into all processing paths

**Prevention**: Always implement secure file handling for sensitive data

---

### Issue 6: Streaming Parameter Validation
**Date**: October 12, 2025
**Problem**: Invalid streaming parameters causing silent failures
**Root Cause**: Diarization model accepts invalid parameter combinations
**Impact**: Processing appears successful but returns no speaker segments

**Debugging Steps**:
1. Tested various streaming parameter combinations
2. Identified valid ranges for chunk_size, right_context, fifo_size
3. Verified parameter validation in model code
4. Tested edge cases and boundary conditions

**Resolution**:
- Added parameter validation before model initialization
- Implemented `_check_streaming_parameters()` calls
- Added comprehensive error messages for invalid configurations
- Documented recommended parameter combinations

**Prevention**: Always validate model parameters before inference

---

### Issue 7: FastAPI Multipart Upload Issues
**Date**: October 12, 2025
**Problem**: Large audio files causing request timeouts
**Root Cause**: Synchronous file processing blocking async endpoints
**Impact**: API unresponsive for files >10MB

**Debugging Steps**:
1. Analyzed FastAPI request handling patterns
2. Tested file upload size limits
3. Identified synchronous processing bottlenecks
4. Verified async/await usage throughout pipeline

**Resolution**:
- Maintained async FastAPI endpoints
- Added file size validation (100MB limit)
- Implemented proper error handling for upload failures
- Added progress logging for long-running operations

**Prevention**: Always implement file size limits and async processing for uploads

---

### Issue 8: Model Loading Performance
**Date**: October 12, 2025
**Problem**: 30+ second model loading times on first inference
**Root Cause**: Models downloaded and loaded synchronously on first request
**Impact**: Poor user experience for initial requests

**Debugging Steps**:
1. Measured model loading times
2. Identified download vs loading bottlenecks
3. Tested model caching strategies
4. Verified HuggingFace token requirements

**Resolution**:
- Added HuggingFace token configuration for faster downloads
- Implemented model pre-loading in startup
- Added loading progress indicators
- Documented expected loading times

**Prevention**: Always account for model loading time in production deployments

## Debugging Patterns Established

### GPU Debugging Checklist
- [ ] Check CUDA availability: `torch.cuda.is_available()`
- [ ] Monitor GPU memory: `nvidia-smi`
- [ ] Verify CUDA version compatibility
- [ ] Test with CPU fallback if GPU fails
- [ ] Check cuDNN version requirements

### Model Debugging Checklist
- [ ] Verify HuggingFace token for gated models
- [ ] Check model loading without errors
- [ ] Test inference on small sample data
- [ ] Validate output format and content
- [ ] Check parameter compatibility

### Pipeline Debugging Checklist
- [ ] Test individual components in isolation
- [ ] Verify data flow between components
- [ ] Check temporary file creation/cleanup
- [ ] Monitor memory usage throughout pipeline
- [ ] Validate output format and completeness

### API Debugging Checklist
- [ ] Test endpoint accessibility
- [ ] Verify parameter parsing
- [ ] Check file upload handling
- [ ] Monitor request/response times
- [ ] Validate error response formats

## Lessons Learned

1. **GPU Compatibility First**: Always verify GPU requirements before model selection
2. **Fresh Environments**: Use clean venvs for complex ML dependency management
3. **Security by Design**: Implement secure file handling from the start
4. **Parameter Validation**: Always validate model parameters before inference
5. **Memory Management**: Explicit cleanup required for GPU memory management
6. **Async Processing**: Maintain async patterns throughout the pipeline
7. **Comprehensive Testing**: Test individual components before integration
8. **Documentation**: Document all non-obvious requirements and gotchas

## Prevention Strategies

- **Dependency Management**: Pin exact versions, use fresh environments
- **GPU Compatibility**: Test on target hardware before deployment
- **Security Reviews**: Regular security audits of file handling
- **Performance Monitoring**: Continuous monitoring of memory and timing
- **Error Handling**: Comprehensive error handling with user-friendly messages
- **Documentation**: Keep debug history updated with new issues and resolutions

This debug history serves as a reference for future issues and helps prevent recurring problems in similar ML pipeline projects.
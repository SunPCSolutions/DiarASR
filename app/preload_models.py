#!/usr/bin/env python3
"""
Model Pre-loading Script

This script pre-loads both NeMo ASR and Pyannote diarization models during Docker build
to cache them in the Hugging Face cache directory and avoid download delays at runtime.

Usage:
    python preload_models.py

Environment Variables:
    HF_TOKEN: Hugging Face authentication token (required for Pyannote models)
"""

import os
import sys
import logging
import torch
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def load_nemo_asr_model(model_name: str = "nvidia/parakeet-tdt-1.1b") -> bool:
    """
    Load and cache NeMo ASR model.

    Args:
        model_name: Name of the NeMo ASR model to load

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.info(f"Loading NeMo ASR model: {model_name}")

        # Import NeMo ASR
        import nemo.collections.asr as nemo_asr

        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")

        # Load the model
        asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name)
        asr_model = asr_model.to(device)
        asr_model.eval()

        logger.info(f"✅ NeMo ASR model {model_name} loaded successfully on {device}")

        # Clean up to free memory
        del asr_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        return True

    except Exception as e:
        logger.error(f"❌ Failed to load NeMo ASR model {model_name}: {e}")
        return False


def load_pyannote_diarization_model(model_name: str = "pyannote/speaker-diarization-community-1",
                                   hf_token: Optional[str] = None) -> bool:
    """
    Load and cache Pyannote diarization model.

    Args:
        model_name: Name of the Pyannote model to load
        hf_token: Hugging Face authentication token

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.info(f"Loading Pyannote diarization model: {model_name}")

        # Check for HF token
        if not hf_token:
            hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            logger.error("❌ HF_TOKEN environment variable is required for Pyannote models")
            return False

        # Set HF token in environment
        os.environ["HF_TOKEN"] = hf_token
        logger.info("Hugging Face token configured")

        # Import Pyannote
        from pyannote.audio import Pipeline

        # Determine device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        # Load the pipeline
        pipeline = Pipeline.from_pretrained(model_name)
        pipeline.to(device)

        logger.info(f"✅ Pyannote diarization model {model_name} loaded successfully on {device}")

        # Clean up to free memory
        del pipeline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        return True

    except Exception as e:
        logger.error(f"❌ Failed to load Pyannote diarization model {model_name}: {e}")
        return False


def main():
    """Main pre-loading function."""
    logger.info("🚀 Starting model pre-loading process")

    success_count = 0
    total_models = 2

    # Load NeMo ASR model (required)
    if load_nemo_asr_model():
        success_count += 1
    else:
        logger.error("❌ Critical: NeMo ASR model failed to load")
        return 1

    # Load Pyannote diarization model (optional - don't fail build if missing HF_TOKEN)
    if load_pyannote_diarization_model():
        success_count += 1
    else:
        logger.warning("⚠️  Pyannote diarization model failed to load (likely missing HF_TOKEN)")

    # Summary
    logger.info(f"📊 Pre-loading complete: {success_count}/{total_models} models loaded successfully")

    if success_count >= 1:  # At least NeMo should load
        logger.info("✅ Model pre-loading completed (NeMo ASR ready)")
        return 0
    else:
        logger.error("❌ Critical models failed to load")
        return 1


if __name__ == "__main__":
    sys.exit(main())
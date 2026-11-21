"""Run NVIDIA Nemotron 3 Diarization with its isolated NeMo runtime."""

import json
import os
import sys


MODEL = 'nvidia/Nemotron-3-Diarization'
REVISION = '52a432672732425e4c19765b9fec9b5810e6f4ba'
MAX_SPEAKERS = 8


def speaker_bounds(num_speakers=None, min_speakers=None, max_speakers=None):
    lower = 1 if min_speakers is None else min_speakers
    upper = MAX_SPEAKERS if max_speakers is None else max_speakers
    if not 1 <= lower <= upper <= MAX_SPEAKERS:
        raise ValueError(f'Speaker bounds must be between 1 and {MAX_SPEAKERS}, with min <= max')
    if num_speakers is not None:
        if not lower <= num_speakers <= upper:
            raise ValueError('num_speakers must be within the speaker bounds')
        return num_speakers, num_speakers
    return min_speakers, max_speakers


def diarize(path):
    import torch
    from huggingface_hub import hf_hub_download
    from nemo.collections.asr.models import SortformerEncLabelModel

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError('Nemotron 3 Diarization requires a BF16-capable CUDA GPU')

    checkpoint = hf_hub_download(MODEL, 'Nemotron-3-Diarization.nemo', revision=REVISION)
    model = SortformerEncLabelModel.restore_from(checkpoint, map_location='cuda').eval().bfloat16()

    # NVIDIA's 30.4 s offline preset keeps speaker state across chunks.
    settings = model.sortformer_modules
    settings.spkcache_len = 264
    settings.fifo_len = 40
    settings.chunk_len = 340
    settings.chunk_right_context = 40
    settings.spkcache_update_period = 340  # Effective clamp of NVIDIA's configured 300.
    model._check_streaming_parameters()

    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        rows, = model.diarize(audio=[path], batch_size=1, verbose=False)

    segments = []
    for row in rows:
        start, end, label = row.split()
        index = int(label.removeprefix('speaker_'))
        segments.append({'start': float(start), 'end': float(end), 'speaker': f'SPEAKER_{index:02d}'})
    return sorted(segments, key=lambda segment: (segment['start'], segment['end'], segment['speaker']))


if __name__ == '__main__':
    # Keep native and NeMo output out of the JSON result.
    with os.fdopen(os.dup(1), 'w') as output:
        os.dup2(sys.stderr.fileno(), 1)
        json.dump(diarize(sys.argv[1]), output)

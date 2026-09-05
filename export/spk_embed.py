"""Speaker-vector helper for multi-speaker models (HarmonicAcousticModelMultiSpk).

The multispk model adds a per-speaker vector `spk_proj(spk_bank.weight[i])` (dim = hidden) to the
condition at every frame (`_spk_add`). For ONNX export we either bake one speaker's vector into
the graph, or expose an H-dim `spk_embed` graph input the host adds itself.

A single-speaker model (HarmonicAcousticModel, no spk_bank) has no speaker embedding — nothing
to export; WrapperA uses speaker='none'.
"""
from __future__ import annotations

import numpy as np
import torch


def has_speakers(model) -> bool:
    return getattr(model, "spk_bank", None) is not None


def speaker_vector(model, spk_id: int) -> np.ndarray:
    """The H-dim vector `_spk_add` would add for `spk_id` (== spk_proj(spk_bank[spk_id]))."""
    assert has_speakers(model), "model has no speaker bank"
    with torch.no_grad():
        idx = torch.tensor([int(spk_id)], device=model.spk_bank.weight.device)
        v = model.spk_proj(model.spk_bank(idx))[0]         # [hidden]
    return v.detach().cpu().numpy().astype(np.float32)

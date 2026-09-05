"""ONNX-export subsystem for the LeapSinger acoustic model.

The training/inference model code (`leapsinger/models`, `leapsinger/modules`) is left untouched.
This package provides ONNX-friendly re-implementations of the only ops that do not export — the
harmonic excitation's STFT/exp2/complex (`excitation_onnx`) and `nn.MultiheadAttention`'s baked
sequence length (`attention_onnx`) — plus export wrappers for the A/B variants, fp16 / simplify /
native-DFT post-processing, and parity verification. Entry point: `python -m export.cli`.
"""
from .excitation_onnx import HarmonicNoiseExcitationONNX
from .wrappers import AcousticExportWrapperA, AcousticExportWrapperB

__all__ = ["HarmonicNoiseExcitationONNX", "AcousticExportWrapperA", "AcousticExportWrapperB"]

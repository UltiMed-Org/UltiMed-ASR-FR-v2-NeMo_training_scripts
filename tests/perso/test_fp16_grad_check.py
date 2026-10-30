# Written by Claude Code.
import importlib.util
from pathlib import Path

import torch

_SPEC = importlib.util.spec_from_file_location(
    "fp16_grad_check", Path(__file__).resolve().parents[2] / "perso" / "fp16_grad_check.py"
)
fp16_grad_check = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(fp16_grad_check)


def test_compare_cosine_is_bounded_on_large_fp32_vectors():
    # fp32 dot and norms over 20M entries used to give cosines above 1 (1.006 seen on GPU runs).
    g = torch.Generator().manual_seed(0)
    g32 = torch.randn(20_000_000, generator=g) * 1e-3 + 1.0
    g16 = g32 + torch.randn(20_000_000, generator=g) * 1e-6
    cos, ratio, _ = fp16_grad_check.compare(g32, g16)
    assert 0.9999 < cos <= 1.0 + 1e-12
    assert abs(ratio - 1.0) < 1e-4

# Tests for the fp16 TDT loss table (TDTLossNumba(fp16_table=True)). Made with Claude Code.
#
# The table path runs the numba TDT kernels on an fp16 copy of the token
# logits, keeps the denominators, alphas, betas and costs in fp32, and writes
# the fp16 gradient over that copy. Two layers of checks:
#
#   - CPU, always: the kernels run under the numba CUDA simulator in a
#     subprocess (NUMBA_ENABLE_CUDASIM has to be set before numba.cuda is
#     imported). The fp32 kernel is checked against the pure pytorch
#     reference, then the fp16 table against the fp32 kernel fed the same
#     fp16 logits, then in place against a separate gradient buffer.
#     The simulator cannot run the shuffle-based log-softmax reduction, so the
#     denominators are computed on the host here; the GPU tests cover that
#     kernel.
#   - GPU, skipped without CUDA and numba fp16 support: the real kernels,
#     through TDTLossNumba and RNNTLoss, at the real vocabulary size.

import json
import sys

import numpy as np
import pytest
import torch

from nemo.collections.asr.losses.rnnt import RNNTLoss
from nemo.core.utils import numba_utils
from tests.collections.asr.numba.rnnt_loss.utils import tdt_simulator

DURATIONS = [0, 1, 2, 3, 4]
SIGMA = 0.05

# Two clips of different lengths so the batch has padding cells, which the
# in-place path must zero itself (its buffer holds logits there, not zeros).
SIM_CASE = dict(T=6, U=4, V=20, act_lens=[6, 4], label_lens=[3, 2], seed=0)


def _simulate(case, fastemit_lambda, clamp):
    """Runs in the simulator subprocess. Returns {name: np.ndarray}."""
    from numba import cuda

    from nemo.collections.asr.losses.rnnt_pytorch import TDTLossPytorch
    from nemo.collections.asr.parts.numba.rnnt_loss.utils import rnnt_helper
    from nemo.collections.asr.parts.numba.rnnt_loss.utils.cuda_utils import gpu_rnnt

    tdt_simulator.install()
    DeviceBuffer = tdt_simulator.DeviceBuffer

    T, U, V = case["T"], case["U"], case["V"]
    B = len(case["act_lens"])
    D = len(DURATIONS)
    blank = V - 1
    rng = np.random.default_rng(case["seed"])
    # Joint-like logits: a few nats of spread, as a trained joint produces.
    label_logits = (3.0 * rng.standard_normal((B, T, U, V))).astype(np.float16).astype(np.float32)
    duration_logits = rng.standard_normal((B, T, U, D)).astype(np.float32)
    labels = np.zeros((B, U - 1), dtype=np.int64)
    for b, n in enumerate(case["label_lens"]):
        labels[b, :n] = rng.integers(0, blank, n)
    act_lens = np.array(case["act_lens"], dtype=np.int64)
    label_lens = np.array(case["label_lens"], dtype=np.int64)

    dur_logits_t = torch.tensor(duration_logits, requires_grad=True)
    dur_logprobs = torch.log_softmax(dur_logits_t, dim=-1)
    dur_logprobs_np = dur_logprobs.detach().numpy().reshape(-1)

    def run(dtype, in_place):
        size, _ = rnnt_helper.get_workspace_size(T, U, B, gpu=True)
        wrapper = gpu_rnnt.GPUTDT(
            minibatch=B,
            maxT=T,
            maxU=U,
            alphabet_size=V,
            workspace=np.zeros(size, dtype=np.float32),  # fp32 for both, as tdt_loss_gpu does
            tdt_workspace=np.array(DURATIONS, dtype=np.int64),
            num_durations=D,
            blank=blank,
            fastemit_lambda=fastemit_lambda,
            clamp=clamp,
            num_threads=1,
            stream=cuda.default_stream(),
            sigma=SIGMA,
            omega=0.0,
        )
        acts = DeviceBuffer(label_logits.reshape(-1).astype(dtype))
        grads = acts if in_place else DeviceBuffer(np.zeros_like(acts._ary))
        dgrads = np.zeros(B * T * U * D, dtype=np.float32)
        costs = np.zeros(B, dtype=np.float32)
        wrapper.cost_and_grad(acts, dur_logprobs_np.copy(), grads, dgrads, costs, labels, label_lens, act_lens)
        # Duration grads are w.r.t. the log-probs; take them to the logits like autograd would.
        dur_logits_t.grad = None
        dur_logprobs.backward(torch.tensor(dgrads.reshape(B, T, U, D)), retain_graph=True)
        return {
            "cost": costs.astype(np.float64),
            "label_grad": grads._ary.astype(np.float64).reshape(B, T, U, V),
            "duration_grad": dur_logits_t.grad.numpy().astype(np.float64),
        }

    out = {}
    for name, dtype, in_place in [("fp32", np.float32, False), ("fp16", np.float16, False), ("fp16_inplace", np.float16, True)]:
        for key, value in run(dtype, in_place).items():
            out[f"{name}.{key}"] = value

    if fastemit_lambda == 0.0 and clamp == 0.0:  # the reference knows neither
        ref_in = torch.tensor(np.concatenate([label_logits, duration_logits], axis=-1), requires_grad=True)
        ref = TDTLossPytorch(blank=blank, durations=DURATIONS, reduction="none", sigma=SIGMA)
        cost = ref(
            acts=ref_in,
            labels=torch.tensor(labels),
            act_lens=torch.tensor(act_lens),
            label_lens=torch.tensor(label_lens),
        )
        cost.sum().backward()
        out["ref.cost"] = cost.detach().numpy().astype(np.float64)
        out["ref.label_grad"] = ref_in.grad[..., :V].numpy().astype(np.float64)
        out["ref.duration_grad"] = ref_in.grad[..., V:].numpy().astype(np.float64)
    return out


def _run_simulator(tmp_path, fastemit_lambda, clamp):
    return tdt_simulator.run_in_simulator(__file__, {"fastemit_lambda": fastemit_lambda, "clamp": clamp}, tmp_path)


def _padding_mask(case):
    """True on the [B, T, U] cells outside each clip's lattice."""
    B = len(case["act_lens"])
    mask = np.ones((B, case["T"], case["U"]), dtype=bool)
    for b, (t, u) in enumerate(zip(case["act_lens"], case["label_lens"])):
        mask[b, :t, : u + 1] = False
    return mask


@pytest.mark.unit
@pytest.mark.parametrize("fastemit_lambda,clamp", [(0.0, 0.0), (0.01, 0.5)])
def test_fp16_table_matches_fp32_in_the_simulator(tmp_path, fastemit_lambda, clamp):
    r = _run_simulator(tmp_path, fastemit_lambda, clamp)

    if "ref.cost" in r:
        # The kernel edit (shared reads hoisted before a barrier) kept fp32 exact.
        np.testing.assert_allclose(r["fp32.cost"], r["ref.cost"], rtol=1e-5)
        np.testing.assert_allclose(r["fp32.label_grad"], r["ref.label_grad"], atol=1e-5)
        np.testing.assert_allclose(r["fp32.duration_grad"], r["ref.duration_grad"], atol=1e-5)

    # Same fp16 logits in, fp32 lattice in both: costs agree to fp32 rounding,
    # gradients to one fp16 rounding of the output (2^-11 relative).
    np.testing.assert_allclose(r["fp16.cost"], r["fp32.cost"], rtol=1e-6)
    np.testing.assert_allclose(r["fp16.label_grad"], r["fp32.label_grad"], rtol=1e-3, atol=1e-6)
    np.testing.assert_allclose(r["fp16.duration_grad"], r["fp32.duration_grad"], rtol=1e-6, atol=1e-9)

    # Writing over the logits changes nothing, padding included.
    for key in ("cost", "label_grad", "duration_grad"):
        np.testing.assert_array_equal(r[f"fp16_inplace.{key}"], r[f"fp16.{key}"])
    assert not r["fp16_inplace.label_grad"][_padding_mask(SIM_CASE)].any()


# --------------------------------------------------------------------------- GPU


def _skip_without_fp16_numba_cuda():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    if not numba_utils.is_numba_cuda_fp16_supported():
        pytest.skip("needs numba CUDA fp16 (NUMBA_CUDA_USE_NVIDIA_BINDING=1)")


def _gpu_batch(T=80, U=24, V=8198, act_lens=(80, 57), label_lens=(23, 15), seed=0):
    """Real vocabulary (8192 tokens + blank + 5 durations); bf16-exact logits."""
    g = torch.Generator().manual_seed(seed)
    B, D = len(act_lens), len(DURATIONS)
    logits = (3.0 * torch.randn(B, T, U, V + D, generator=g)).bfloat16().float()
    blank = V - 1
    labels = torch.zeros(B, U - 1, dtype=torch.long)
    for b, n in enumerate(label_lens):
        labels[b, :n] = torch.randint(0, blank, (n,), generator=g)
    lens = torch.tensor(act_lens), torch.tensor(label_lens)
    return logits.cuda(), labels.cuda(), lens[0].cuda(), lens[1].cuda(), blank


@pytest.mark.unit
@pytest.mark.run_only_on('GPU')
def test_fp16_table_matches_fp32_on_gpu():
    from nemo.collections.asr.parts.numba.rnnt_loss.rnnt_pytorch import TDTLossNumba

    _skip_without_fp16_numba_cuda()
    logits, labels, act_lens, label_lens, blank = _gpu_batch()

    results = {}
    for fp16_table in (False, True):
        acts = logits.clone().requires_grad_(True)
        loss = TDTLossNumba(blank=blank, durations=DURATIONS, reduction="none", sigma=SIGMA, fp16_table=fp16_table)
        cost = loss(acts, labels, act_lens, label_lens)
        cost.sum().backward()
        results[fp16_table] = cost.detach().double().cpu(), acts.grad.double().cpu()

    (cost32, grad32), (cost16, grad16) = results[False], results[True]
    assert torch.isfinite(cost16).all()
    torch.testing.assert_close(cost16, cost32, rtol=1e-5, atol=0)
    torch.testing.assert_close(grad16, grad32, rtol=1e-3, atol=1e-6)


@pytest.mark.unit
@pytest.mark.run_only_on('GPU')
def test_fp16_table_in_place_equals_separate_buffer_on_gpu():
    from nemo.collections.asr.parts.numba.rnnt_loss.rnnt_pytorch import _TDTNumba

    _skip_without_fp16_numba_cuda()
    logits, labels, act_lens, label_lens, blank = _gpu_batch(seed=1)
    V = logits.shape[-1] - len(DURATIONS)
    durations_logprobs = torch.log_softmax(logits[..., V:], dim=-1).contiguous()

    grads = {}
    for in_place in (False, True):
        label_acts = logits[..., :V].half().contiguous().requires_grad_(True)
        cost = _TDTNumba.apply(
            label_acts, durations_logprobs, labels, act_lens, label_lens, blank, DURATIONS, "none", 0.0, 0.0, SIGMA, 0.0, in_place
        )
        cost.sum().backward()
        grads[in_place] = label_acts.grad.float().cpu()

    torch.testing.assert_close(grads[True], grads[False], rtol=0, atol=0)
    padding = torch.from_numpy(
        _padding_mask(dict(T=logits.shape[1], U=logits.shape[2], act_lens=act_lens.tolist(), label_lens=label_lens.tolist()))
    )
    assert not grads[True][padding].any()


@pytest.mark.unit
@pytest.mark.run_only_on('GPU')
def test_tdt_label_grads_with_a_frozen_duration_head_on_gpu():
    """Durations that need no gradient used to leave the kernel without a duration buffer
    and crash in `flatten_tensor(None)` (fp32 path too, it is upstream's allocation)."""
    from nemo.collections.asr.parts.numba.rnnt_loss.rnnt_pytorch import _TDTNumba

    _skip_without_fp16_numba_cuda()
    logits, labels, act_lens, label_lens, blank = _gpu_batch(seed=2)
    V = logits.shape[-1] - len(DURATIONS)
    durations_logprobs = torch.log_softmax(logits[..., V:], dim=-1).contiguous()
    assert not durations_logprobs.requires_grad
    label_acts = logits[..., :V].contiguous().requires_grad_(True)
    cost = _TDTNumba.apply(
        label_acts, durations_logprobs, labels, act_lens, label_lens, blank, DURATIONS, "none", 0.0, 0.0, SIGMA, 0.0, False
    )
    cost.sum().backward()
    assert torch.isfinite(label_acts.grad).all() and label_acts.grad.abs().sum() > 0


@pytest.mark.unit
@pytest.mark.run_only_on('GPU')
def test_rnnt_loss_keeps_the_bf16_joint_with_fp16_table():
    _skip_without_fp16_numba_cuda()
    logits, labels, act_lens, label_lens, blank = _gpu_batch(T=40, U=12, act_lens=(40, 31), label_lens=(11, 7))
    V = logits.shape[-1] - len(DURATIONS)

    losses = {}
    for fp16_table in (False, True):
        loss = RNNTLoss(num_classes=blank, reduction="sum", loss_name="tdt", loss_kwargs={"durations": DURATIONS, "sigma": SIGMA, "fp16_table": fp16_table})
        assert loss._force_float32 is not fp16_table
        acts = logits.bfloat16().requires_grad_(True)
        value = loss(log_probs=acts, targets=labels, input_lengths=act_lens, target_lengths=label_lens)
        value.backward()
        assert acts.grad.dtype == torch.bfloat16
        losses[fp16_table] = value.item()
    assert losses[True] == pytest.approx(losses[False], rel=1e-5)


# --------------------------------------------------------------------------- switch (CPU)


@pytest.mark.unit
def test_set_tdt_fp16_table_switches_the_upcast(monkeypatch):
    loss = RNNTLoss(num_classes=8, loss_name="tdt", loss_kwargs={"durations": DURATIONS})
    assert loss._force_float32 and not loss._loss.fp16_table  # default off: v1 behaviour

    monkeypatch.setattr(numba_utils, "is_numba_cuda_fp16_supported", lambda return_reason=False: (False, "no binding"))
    with pytest.raises(RuntimeError, match="no binding"):
        loss.set_tdt_fp16_table(True)
    assert loss._force_float32 and not loss._loss.fp16_table

    monkeypatch.setattr(numba_utils, "is_numba_cuda_fp16_supported", lambda return_reason=False: (True, ""))
    loss.set_tdt_fp16_table(True)
    assert not loss._force_float32 and loss._loss.fp16_table
    loss.set_tdt_fp16_table(False)
    assert loss._force_float32 and not loss._loss.fp16_table


@pytest.mark.unit
def test_set_tdt_fp16_table_refuses_other_losses():
    loss = RNNTLoss(num_classes=8, loss_name="tdt_pytorch", loss_kwargs={"durations": DURATIONS})
    with pytest.raises(ValueError, match="numba TDT"):
        loss.set_tdt_fp16_table(True)


@pytest.mark.unit
def test_set_tdt_omega_overrides_the_restored_value():
    # The checkpoint ships omega 0.1, which keeps the fp16 table out of place.
    loss = RNNTLoss(num_classes=8, loss_name="tdt", loss_kwargs={"durations": DURATIONS, "omega": 0.1})
    assert loss._loss.omega == 0.1
    loss.set_tdt_omega(0)
    assert loss._loss.omega == 0.0
    with pytest.raises(ValueError, match="numba TDT"):
        RNNTLoss(num_classes=8, loss_name="tdt_pytorch", loss_kwargs={"durations": DURATIONS}).set_tdt_omega(0)


if __name__ == "__main__":
    # Simulator subprocess entry point (see _run_simulator).
    params = json.loads(sys.argv[1])
    result = _simulate(SIM_CASE, params["fastemit_lambda"], params["clamp"])
    np.savez(params["out"], **result)

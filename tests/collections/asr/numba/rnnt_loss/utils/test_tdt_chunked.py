# Tests of the chunked long-clip TDT joint+loss (losses/tdt_chunked.py).
# Made with Claude Code.
#
# CPU only: the compact-lattice kernels run under the numba CUDA simulator in
# a subprocess (see tdt_simulator.py), with rnnt.tdt_loss_gpu swapped for its
# simulator twin. The reference is the dense joint on CPU fed to the pure
# pytorch TDT loss (which test_tdt_fp16_table.py checks against the numba
# kernel), so loss and gradients w.r.t. the encoder output, the prediction
# output and every joint weight are compared end to end.

import json
import sys

import numpy as np
import pytest
import torch

from tests.collections.asr.numba.rnnt_loss.utils import tdt_simulator

DURATIONS = [0, 1, 2, 3, 4]
V = 11  # token logits including blank (the last one)
H_ENC, H_PRED, H_JOINT = 6, 5, 8
# B=1, and B=2 with padding in both T and U.
CASES = {
    "b1": dict(T=7, U=4, act_lens=[7], label_lens=[4], seed=0),
    "b2": dict(T=7, U=4, act_lens=[7, 5], label_lens=[4, 2], seed=1),
}
SIGMA = 0.05
# Per-sample upstream gradients: checks the costs are scaled per clip.
GRAD_WEIGHTS = [1.0, 0.7]
# (name, case, sigma, chunk_frames, dropout)
RUNS = (
    [(f"b1_s{SIGMA}_tc{tc}", "b1", SIGMA, tc, 0.0) for tc in (1, 2, 7)]
    + [(f"b2_s{SIGMA}_tc{tc}", "b2", SIGMA, tc, 0.0) for tc in (1, 2, 7)]
    + [("b2_s0.0_tc3", "b2", 0.0, 3, 0.0), ("b2_dropout_tc2", "b2", SIGMA, 2, 0.5)]
)


def _make_joint(dropout, seed):
    from nemo.collections.asr.modules.rnnt import RNNTJoint

    torch.manual_seed(seed)
    joint = RNNTJoint(
        jointnet=dict(
            encoder_hidden=H_ENC, pred_hidden=H_PRED, joint_hidden=H_JOINT, activation="relu", dropout=dropout
        ),
        num_classes=V - 1,
        num_extra_outputs=len(DURATIONS),
        fuse_loss_wer=True,
        fused_batch_size=1,
    )
    joint.train(dropout > 0)
    joint.set_wer(object())  # the fused loop requires one; compute_wer stays off
    return joint


def _make_batch(case):
    g = torch.Generator().manual_seed(case["seed"])
    B, T, U = len(case["act_lens"]), case["T"], case["U"]
    enc = torch.randn(B, T, H_ENC, generator=g)
    dec = torch.randn(B, U + 1, H_PRED, generator=g)
    labels = torch.zeros(B, U, dtype=torch.long)
    for b, n in enumerate(case["label_lens"]):
        labels[b, :n] = torch.randint(0, V - 1, (n,), generator=g)
    return enc, dec, labels, torch.tensor(case["act_lens"]), torch.tensor(case["label_lens"])


def _grads(joint, enc, dec, cost):
    weights = torch.tensor(GRAD_WEIGHTS[: cost.shape[0]])
    (cost * weights).sum().backward()
    out = {"cost": cost.detach(), "grad.enc": enc.grad, "grad.dec": dec.grad}
    out.update({f"grad.{name}": p.grad for name, p in joint.named_parameters()})
    return {k: v.double().numpy() for k, v in out.items()}


def _reference(joint, case, sigma, dropout_scale=None):
    """Dense joint + pure pytorch TDT loss. dropout_scale [B, T, U+1, H] replaces the joint's dropout."""
    from nemo.collections.asr.losses.rnnt_pytorch import TDTLossPytorch

    enc, dec, labels, act_lens, label_lens = _make_batch(case)
    enc.requires_grad_(True)
    dec.requires_grad_(True)
    hidden = torch.relu(joint.enc(enc).unsqueeze(2) + joint.pred(dec).unsqueeze(1))
    if dropout_scale is not None:
        hidden = hidden * dropout_scale
    logits = joint.joint_net[-1](hidden)
    loss = TDTLossPytorch(blank=V - 1, durations=DURATIONS, reduction="none", sigma=sigma)
    cost = loss(acts=logits, labels=labels, act_lens=act_lens, label_lens=label_lens)
    return _grads(joint, enc, dec, cost)


def _chunked(joint, case, sigma, chunk_frames):
    from nemo.collections.asr.losses.tdt_chunked import chunked_tdt_joint_loss

    enc, dec, labels, act_lens, label_lens = _make_batch(case)
    enc.requires_grad_(True)
    dec.requires_grad_(True)
    cost = chunked_tdt_joint_loss(
        joint.enc(enc), joint.pred(dec), labels, act_lens, label_lens, joint.joint_net,
        blank=V - 1, durations=DURATIONS, sigma=sigma, chunk_frames=chunk_frames,
    )
    return _grads(joint, enc, dec, cost)


def _simulate(params):
    """Runs in the simulator subprocess. Returns {name: np.ndarray}."""
    from nemo.collections.asr.parts.numba.rnnt_loss import rnnt as numba_rnnt

    tdt_simulator.install()
    numba_rnnt.tdt_loss_gpu = tdt_simulator.tdt_loss_gpu

    out = {}

    def store(prefix, result):
        out.update({f"{prefix}.{k}": v for k, v in result.items()})

    for name, case_name, sigma, chunk_frames, dropout in RUNS:
        case = CASES[case_name]
        joint = _make_joint(dropout, seed=10)
        if not dropout:
            store(f"{name}.chunked", _chunked(joint, case, sigma, chunk_frames))
            joint.zero_grad()
            store(f"{name}.ref", _reference(joint, case, sigma))
            continue

        # Record each dropout call's scale (0 or 1 / (1 - p) where the input is non-zero).
        scales = []

        def record(module, inputs, output):
            x = inputs[0]
            scales.append(torch.where(x == 0, torch.zeros_like(x), output / x).detach())

        joint.joint_net[1].register_forward_hook(record)
        store(f"{name}.chunked", _chunked(joint, case, sigma, chunk_frames))
        n_chunks = -(-case["T"] // chunk_frames)
        assert len(scales) == 2 * n_chunks, len(scales)  # forward, then the backward recompute
        out[f"{name}.forward_masks"] = torch.cat(scales[:n_chunks], dim=1).numpy()
        out[f"{name}.replayed_masks"] = torch.cat(scales[n_chunks:], dim=1).numpy()
        joint.zero_grad()
        store(f"{name}.ref", _reference(joint, case, sigma, dropout_scale=torch.cat(scales[:n_chunks], dim=1)))

    store("dispatch", _dispatch())
    return out


def _dispatch():
    """RNNTJoint's fused loop on the long path vs its dense path, same batch."""
    from nemo.collections.asr.losses.rnnt import RNNTLoss
    from nemo.collections.asr.losses.rnnt_pytorch import TDTLossPytorch
    from nemo.collections.asr.losses import tdt_chunked

    case = CASES["b2"]
    joint = _make_joint(0.0, seed=10)
    loss = RNNTLoss(num_classes=V - 1, reduction="sum", loss_name="tdt", loss_kwargs={"durations": DURATIONS, "sigma": SIGMA})
    joint.set_loss(loss)
    calls = []
    real = tdt_chunked.chunked_tdt_joint_loss

    def spy(*args, **kwargs):
        calls.append(kwargs["chunk_frames"])
        return real(*args, **kwargs)

    tdt_chunked.chunked_tdt_joint_loss = spy

    def run(min_cells):
        joint.zero_grad()
        joint.long_path_min_cells = min_cells
        joint.long_path_chunk_frames = 3
        enc, dec, labels, act_lens, label_lens = _make_batch(case)
        enc.requires_grad_(True)
        dec.requires_grad_(True)
        value, *_ = joint(
            encoder_outputs=enc.transpose(1, 2),
            decoder_outputs=dec.transpose(1, 2),
            encoder_lengths=act_lens,
            transcripts=labels,
            transcript_lengths=label_lens,
        )
        value.backward()
        out = {"cost": value.detach().reshape(1), "grad.enc": enc.grad, "grad.dec": dec.grad}
        out.update({f"grad.{n}": p.grad for n, p in joint.named_parameters()})
        return {k: v.double().numpy() for k, v in out.items()}

    result = {f"long.{k}": v for k, v in run(min_cells=0).items()}
    result["calls_long"] = np.array(calls, dtype=np.int64)
    # Dense path for both clips: the numba loss cannot run on CPU, the pure pytorch
    # one can (the CPU joint's log-softmax over V + D cancels in its per-group one).
    loss._loss = TDTLossPytorch(blank=V - 1, durations=DURATIONS, reduction="none", sigma=SIGMA)
    result.update({f"dense.{k}": v for k, v in run(min_cells=None).items()})
    result["calls_total"] = np.array(len(calls), dtype=np.int64)
    return result


@pytest.fixture(scope="module")
def sim(tmp_path_factory):
    return tdt_simulator.run_in_simulator(__file__, {}, tmp_path_factory.mktemp("tdt_chunked"))


def _assert_matches(r, a, b):
    keys = sorted(k[len(a) + 1 :] for k in r if k.startswith(a + "."))
    assert "cost" in keys and "grad.enc" in keys and any(k.startswith("grad.joint_net") for k in keys)
    for key in keys:
        if key == "cost":
            np.testing.assert_allclose(r[f"{a}.{key}"], r[f"{b}.{key}"], rtol=1e-5, err_msg=key)
        else:
            np.testing.assert_allclose(r[f"{a}.{key}"], r[f"{b}.{key}"], rtol=1e-4, atol=1e-5, err_msg=key)


@pytest.mark.unit
@pytest.mark.parametrize("name", [run[0] for run in RUNS])
def test_chunked_matches_dense(sim, name):
    """Loss and grads w.r.t. encoder, prediction net and joint weights; any chunk size, sigma, dropout."""
    _assert_matches(sim, f"{name}.chunked", f"{name}.ref")


@pytest.mark.unit
def test_dropout_mask_is_replayed_in_backward(sim):
    forward, replayed = sim["b2_dropout_tc2.forward_masks"], sim["b2_dropout_tc2.replayed_masks"]
    assert (forward == 0).any() and (forward != 0).any()  # dropout actually dropped something
    np.testing.assert_array_equal(replayed, forward)


@pytest.mark.unit
def test_chunk_size_does_not_change_the_result(sim):
    for case in ("b1", "b2"):
        _assert_matches(sim, f"{case}_s{SIGMA}_tc1.chunked", f"{case}_s{SIGMA}_tc7.chunked")
        _assert_matches(sim, f"{case}_s{SIGMA}_tc2.chunked", f"{case}_s{SIGMA}_tc7.chunked")


@pytest.mark.unit
def test_joint_dispatches_on_the_cell_threshold(sim):
    # fused_batch_size 1: one call per clip, with the configured chunk size.
    np.testing.assert_array_equal(sim["dispatch.calls_long"], [3, 3])
    assert sim["dispatch.calls_total"] == 2  # min_cells None never takes it
    _assert_matches(sim, "dispatch.long", "dispatch.dense")


@pytest.mark.unit
def test_long_path_threshold_counts_lattice_cells():
    joint = _make_joint(0.0, seed=0)
    assert not joint._takes_long_path(1313, 715)  # default None: never
    joint.long_path_min_cells = 20
    assert joint._takes_long_path(7, 5)  # 35 cells
    assert not joint._takes_long_path(5, 4)  # 20 cells: strictly more is required


@pytest.mark.unit
def test_long_path_refuses_omega_and_fastemit():
    from nemo.collections.asr.losses.rnnt import RNNTLoss

    case = CASES["b1"]
    for kwargs in ({"omega": 0.1}, {"fastemit_lambda": 0.01}):
        joint = _make_joint(0.0, seed=0)
        joint.set_loss(RNNTLoss(num_classes=V - 1, loss_name="tdt", loss_kwargs=dict(durations=DURATIONS, **kwargs)))
        with pytest.raises(ValueError, match="omega 0"):  # at configuration time
            joint.set_long_path(0)
        joint.set_long_path(None)  # off needs nothing
        joint.long_path_min_cells = 0
        enc, dec, labels, act_lens, label_lens = _make_batch(case)
        with pytest.raises(ValueError, match="omega 0"):  # and if it was bypassed
            joint(
                encoder_outputs=enc.transpose(1, 2),
                decoder_outputs=dec.transpose(1, 2),
                encoder_lengths=act_lens,
                transcripts=labels,
                transcript_lengths=label_lens,
            )


if __name__ == "__main__":
    # Simulator subprocess entry point (see tdt_simulator.run_in_simulator).
    params = json.loads(sys.argv[1])
    np.savez(params["out"], **_simulate(params))

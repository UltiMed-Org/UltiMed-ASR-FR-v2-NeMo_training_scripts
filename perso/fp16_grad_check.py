#!/usr/bin/env python
"""Does the fp16 TDT loss table distort the gradient on real batches of the trained model?

model.joint.fp16_loss_table runs the numba TDT kernels on an fp16 copy of the
token logits and writes an fp16 gradient over it. The unit tests feed random
logits (non-target probabilities around 1/8198 = 1.2e-4), but a trained joint
puts most non-target probabilities at 1e-6 to 1e-9, so their gradient entries
sit below fp16's normal range (2^-14 = 6.1e-5) or below its smallest subnormal
(2^-24 = 6e-8, flushed to zero). If that mass is lost systematically, the
"push everything else down" part of the gradient goes missing.

This script measures it on real clips with the real model. For each batch it
runs the encoder and decoder once, then the fused joint + TDT loss + backward
twice on the same detached encoder/decoder outputs: once with the fp32 loss
(the default upcast) and once with the fp16 table, both at tdt_omega 0 (omega
> 0 randomly samples a different objective, and the fp16 table only writes in
place at 0). The encoder/decoder gradients are then pushed back through the
retained encoder/decoder graph, so both passes see bit-identical activations.

Setup follows training: the model from the config's init_from_nemo_model, the
same freeze (apply_freeze from the training script), joint.fused_batch_size,
bf16 autocast on CUDA. The model is in eval mode (dropout, SpecAugment and
prediction-net masking off) with gradients enabled.

Per batch it prints the two losses and, for each tensor group, the cosine
between the two gradients, the fp16/fp32 norm ratio and max|diff| / ||g_fp32||.
It also counts, from a hook on the joint output, the fraction of token-logit
gradient entries below 6.1e-5, below 6e-8 and exactly zero, in both passes:
that is the exposure the fp16 table has, and how much of it got flushed.

Exit code: 0 PASS (every tensor on every batch within --cos-min / --ratio-tol),
1 WARN, 2 skipped (CPU: the numba TDT loss needs CUDA, so --device cpu only
checks the plumbing up to the joint output).

Usage (from anywhere, paths in the config resolve from the repo root, as in training):

    NUMBA_CUDA_USE_NVIDIA_BINDING=1 .venv/bin/python perso/fp16_grad_check.py
    # CPU plumbing smoke:
    CUDA_VISIBLE_DEVICES="" .venv/bin/python perso/fp16_grad_check.py --device cpu --bucket 4:6:1

Written with Claude Code.
"""

import argparse
import json
import math
import os
import random
import sys
import tempfile
from pathlib import Path

# numba fp16 kernels need the NVIDIA binding (numba_utils.is_numba_cuda_fp16_supported).
os.environ.setdefault("NUMBA_CUDA_USE_NVIDIA_BINDING", "1")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch
from omegaconf import OmegaConf

from nemo.collections.asr.models import ASRModel
from nemo.utils import logging

from examples.asr.speech_to_text_finetune_cached import apply_freeze
from perso.evaluate_checkpoint import load_base_model

FP16_MIN_NORMAL = 2.0**-14  # 6.10e-5
FP16_MIN_SUBNORMAL = 2.0**-24  # 5.96e-8, half of it and below rounds to zero


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="perso/training_config.yaml", help="training config, relative to the repo root")
    p.add_argument("--nemo", default=None, help="override the config's init_from_nemo_model with this .nemo")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", choices=["cuda", "cpu"])
    p.add_argument(
        "--bucket",
        action="append",
        default=None,
        help="LO:HI:N, one batch of N clips with LO <= duration <= HI seconds, about half of them PARHAF. "
        "Repeatable. Default: 4:6:4 18:22:3 35:38:1 (the long clip stays <= 38 s so the fp32 side fits 24 GB)",
    )
    p.add_argument(
        "--manifest",
        action="append",
        default=None,
        help="train manifest to draw from (repeatable). Default: the config's train_ds manifests under /v2/",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cos-min", type=float, default=0.999, help="PASS needs cosine >= this for every tensor")
    p.add_argument("--ratio-tol", type=float, default=0.01, help="PASS needs |norm ratio - 1| <= this")
    p.add_argument("--no-logit-stats", action="store_true", help="skip the joint-output gradient magnitude counts")
    return p.parse_args()


def parse_buckets(specs):
    buckets = []
    for spec in specs or ["4:6:4", "18:22:3", "35:38:1"]:
        lo, hi, n = spec.split(":")
        buckets.append((float(lo), float(hi), int(n)))
    return buckets


def select_clips(manifests, buckets, seed):
    """Return one list of manifest rows per bucket, audio paths made absolute.

    Seeded. About half of each bucket (rounded up) is PARHAF, the rest any other
    category, falling back to whatever exists when one side runs short.
    """
    rows, seen = [], set()
    for manifest in manifests:
        base = os.path.dirname(os.path.abspath(manifest))
        with open(manifest) as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                # Relative audio paths are relative to the manifest, and the batch
                # manifest written later lives elsewhere.
                row["audio_filepath"] = os.path.normpath(os.path.join(base, row["audio_filepath"]))
                if row["audio_filepath"] not in seen:
                    seen.add(row["audio_filepath"])
                    rows.append(row)

    rng = random.Random(seed)
    batches = []
    for lo, hi, n in buckets:
        pool = [r for r in rows if lo <= float(r.get("duration") or 0) <= hi]
        parhaf = [r for r in pool if r.get("category") == "parhaf"]
        other = [r for r in pool if r.get("category") != "parhaf"]
        if len(pool) < n:
            raise SystemExit(f"bucket {lo}-{hi} s has {len(pool)} clips, {n} asked")
        n_parhaf = min(math.ceil(n / 2), len(parhaf))
        n_parhaf = max(n_parhaf, n - len(other))
        batches.append(rng.sample(parhaf, n_parhaf) + rng.sample(other, n - n_parhaf))
    return batches


def make_loader(model, cfg, rows, tmpdir, index):
    """A dataloader over exactly these rows in one batch, train_ds text settings, no augmentation."""
    path = Path(tmpdir) / f"batch{index}.jsonl"
    with open(path, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    ds = OmegaConf.to_container(cfg.model.train_ds, resolve=True)
    for key in ("augmentor", "cost_batching", "bucketing_strategy", "bucketing_batch_size", "prefetch_factor"):
        ds.pop(key, None)
    ds.update(
        manifest_filepath=str(path),
        batch_size=len(rows),
        shuffle=False,
        num_workers=0,
        persistent_workers=False,
        pin_memory=False,
        max_duration=1e9,
        min_duration=0.0,
    )
    return model._setup_dataloader_from_config(OmegaConf.create(ds))


def tensor_groups(model):
    """[(name, [params])] whose gradients get compared. Frozen groups are dropped."""
    out_linear = [m for m in model.joint.joint_net if isinstance(m, torch.nn.Linear)][-1]
    layers = model.encoder.layers
    groups = [
        ("joint.out.weight", [out_linear.weight]),
        ("joint.out.bias", [out_linear.bias]),
        ("joint.pred", list(model.joint.pred.parameters())),
        ("joint.enc", list(model.joint.enc.parameters())),
        ("decoder", list(model.decoder.parameters())),
        (f"encoder.layers[{len(layers) - 1}]", list(layers[-1].parameters())),
        ("all trainable", [p for p in model.parameters() if p.requires_grad]),
    ]
    return [(name, [p for p in params if p.requires_grad]) for name, params in groups if any(p.requires_grad for p in params)]


def flat_grad(params):
    return torch.cat([p.grad.detach().float().flatten().cpu() for p in params if p.grad is not None])


class LogitGradStats:
    """Counts token-logit gradient entries by magnitude, via a hook on the joint output.

    The duration logits (last n_durations columns) are left out: the fp16 table
    only holds the token logits. Counted in time chunks so no full-size |g| copy
    of the [B, T, U, V] lattice is made.
    """

    def __init__(self, n_durations):
        self.n_durations = n_durations
        self.reset()

    def reset(self):
        self.total = self.below_normal = self.below_subnormal = self.zero = 0

    def forward_hook(self, module, inputs, output):
        if output.requires_grad:
            output.register_hook(self._count)

    @torch.no_grad()
    def _count(self, grad):
        tokens = grad[..., : grad.shape[-1] - self.n_durations]
        for chunk in tokens.split(16, dim=1):
            mag = chunk.abs()
            self.total += mag.numel()
            self.below_normal += int((mag < FP16_MIN_NORMAL).sum())
            self.below_subnormal += int((mag < FP16_MIN_SUBNORMAL).sum())
            self.zero += int((mag == 0).sum())

    def fractions(self):
        n = max(self.total, 1)
        return self.below_normal / n, self.below_subnormal / n, self.zero / n


def autocast(device):
    return torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda")


def encode(model, batch, device):
    signal, signal_len, transcript, transcript_len = [t.to(device) for t in batch]
    with autocast(device):
        encoded, encoded_len = model.forward(input_signal=signal, input_signal_length=signal_len)
        decoded, target_len, _ = model.decoder(targets=transcript, target_length=transcript_len)
    return encoded, encoded_len, decoded, target_len, transcript


def run_pass(model, fp16_table, encoded, encoded_len, decoded, target_len, transcript, groups, retain):
    """One joint + loss + backward on detached copies of encoder/decoder outputs.

    Returns (loss, {group: flat fp32 grad on CPU}, peak bytes or None).
    """
    model.zero_grad(set_to_none=True)
    model.loss.set_tdt_fp16_table(fp16_table)
    if encoded.is_cuda:
        torch.cuda.reset_peak_memory_stats()
    enc_in = encoded.detach().requires_grad_(encoded.requires_grad)
    dec_in = decoded.detach().requires_grad_(decoded.requires_grad)
    with autocast(encoded.device.type):
        loss, _, _, _ = model.joint(
            encoder_outputs=enc_in,
            decoder_outputs=dec_in,
            encoder_lengths=encoded_len,
            transcripts=transcript,
            transcript_lengths=target_len,
            compute_wer=False,
        )
    loss.backward()
    # Through the shared encoder/decoder graph, kept alive for the other pass.
    outs = [(o, i.grad) for o, i in ((encoded, enc_in), (decoded, dec_in)) if o.requires_grad and i.grad is not None]
    if outs:
        torch.autograd.backward([o for o, _ in outs], [g for _, g in outs], retain_graph=retain)
    grads = {name: flat_grad(params) for name, params in groups}
    peak = torch.cuda.max_memory_allocated() if encoded.is_cuda else None
    model.zero_grad(set_to_none=True)
    if encoded.is_cuda:
        torch.cuda.empty_cache()
    return float(loss.detach()), grads, peak


def compare(g32, g16):
    n32, n16 = g32.norm().item(), g16.norm().item()
    cos = torch.dot(g32, g16).item() / max(n32 * n16, 1e-30)
    max_diff = (g16 - g32).abs().max().item() / max(n32, 1e-30)
    return cos, n16 / max(n32, 1e-30), max_diff


def plumbing_only(model, batch, device):
    """CPU: everything up to the joint output, which is as far as the numba TDT loss lets us go."""
    encoded, encoded_len, decoded, target_len, transcript = encode(model, batch, device)
    logits = model.joint.joint(encoded.transpose(1, 2), decoded.transpose(1, 2))
    print(
        f"  encoder {tuple(encoded.shape)} decoder {tuple(decoded.shape)} joint {tuple(logits.shape)}, "
        f"grad through the joint output: {logits.requires_grad}"
    )


def main():
    args = parse_args()
    nemo_override = str(Path(args.nemo).resolve()) if args.nemo else None
    # The config's relative paths (model, manifests) resolve from the repo root, as in training.
    os.chdir(REPO_ROOT)
    cfg = OmegaConf.load(args.config)
    device = args.device
    torch.manual_seed(args.seed)

    manifests = args.manifest or list(dict.fromkeys(m for m in cfg.model.train_ds.manifest_filepath if "/v2/" in m))
    logging.info(f"Drawing clips from: {manifests}")
    batches = select_clips(manifests, parse_buckets(args.bucket), args.seed)

    if nemo_override:
        model = ASRModel.restore_from(restore_path=nemo_override, map_location=torch.device(device))
    else:
        model, _ = load_base_model(cfg)
    model = model.to(device)
    model.eval()  # identical activations in both passes; gradients stay on
    torch.set_grad_enabled(True)
    apply_freeze(model, cfg)

    joint_cfg = cfg.model.get("joint", None) or {}
    if joint_cfg.get("fused_batch_size") is not None:
        model.joint.set_fused_batch_size(joint_cfg.fused_batch_size)
    model.loss.set_tdt_omega(0.0)  # both passes, see the module docstring
    groups = tensor_groups(model)

    stats = None
    if not args.no_logit_stats:
        stats = LogitGradStats(n_durations=len(model.loss._loss.durations))
        model.joint.joint_net.register_forward_hook(stats.forward_hook)

    rows, worst = [], None
    with tempfile.TemporaryDirectory() as tmpdir:
        for index, clips in enumerate(batches):
            durations = ", ".join(f"{c['duration']:.1f}s {c.get('category', '?')}" for c in clips)
            print(f"\n=== batch {index}: {durations}")
            batch = next(iter(make_loader(model, cfg, clips, tmpdir, index)))
            if device == "cpu":
                plumbing_only(model, batch, device)
                continue

            encoded, encoded_len, decoded, target_len, transcript = encode(model, batch, device)
            results, exposure = {}, {}
            for mode in ("fp32", "fp16"):
                if stats:
                    stats.reset()
                results[mode] = run_pass(
                    model, mode == "fp16", encoded, encoded_len, decoded, target_len, transcript, groups,
                    retain=mode == "fp32",
                )
                if stats:
                    exposure[mode] = stats.fractions()
            del encoded, decoded
            torch.cuda.empty_cache()

            (l32, g32, p32), (l16, g16, p16) = results["fp32"], results["fp16"]
            print(
                f"  loss fp32 {l32:.6f}  fp16 {l16:.6f}  rel diff {abs(l16 - l32) / max(abs(l32), 1e-30):.2e}"
                f"  | peak fp32 {p32 / 2**30:.1f} GiB, fp16 {p16 / 2**30:.1f} GiB"
            )
            for mode, (normal, subnormal, zero) in exposure.items():
                print(
                    f"  logit grad ({mode} pass): {100 * normal:.2f}% < 6.1e-5, "
                    f"{100 * subnormal:.2f}% < 6e-8, {100 * zero:.2f}% exactly 0"
                )
            print(f"  {'tensor':<22} {'cosine':>10} {'norm ratio':>11} {'max|diff|/norm':>15}")
            for name, _ in groups:
                cos, ratio, max_diff = compare(g32[name], g16[name])
                ok = cos >= args.cos_min and abs(ratio - 1) <= args.ratio_tol
                print(f"  {name:<22} {cos:>10.6f} {ratio:>11.5f} {max_diff:>15.3e}{'' if ok else '  <-- out of bounds'}")
                # How far past the thresholds, in units of each tolerance.
                badness = max((args.cos_min - cos) / (1 - args.cos_min), abs(ratio - 1) / args.ratio_tol)
                rows.append(ok)
                if worst is None or badness > worst[0]:
                    worst = (badness, f"batch {index} {name}: cosine {cos:.6f}, norm ratio {ratio:.5f}")

    if device == "cpu":
        print("\nSKIP: plumbing OK up to the joint output. The numba TDT loss (fp32 and fp16 table) needs CUDA.")
        return 2
    if all(rows):
        print(f"\nPASS: every tensor on every batch has cosine >= {args.cos_min} and norm ratio within 1 +- {args.ratio_tol}")
        return 0
    print(f"\nWARN: worst tensor is {worst[1]} (thresholds: cosine >= {args.cos_min}, ratio 1 +- {args.ratio_tol})")
    return 1


if __name__ == "__main__":
    sys.exit(main())

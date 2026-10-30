"""Refit the cost-batching weights from measured per-batch VRAM peaks. Made with Claude Code.

The sampler charges a batch  sum_weight * sum(c_i) + max_weight * max(c_i),
with c_i = d^2 (duration_sq) or d * (U + 1) (duration_tokens), see
nemo/collections/asr/parts/utils/asr_batching.py. This fits the measured peaks
as  peak_MiB = base + a * sum(c_i) + b * max(c_i)  for both cost models, says
which one explains the peaks better, and prints the (sum_weight, max_weight)
that keep `--budget` meaning the same VRAM ceiling (`--target-peak-mib`):
cost <= budget  <=>  base + a*S + b*M <= target, so the weights are a and b
scaled by budget / (target - base). The sampler has no intercept; that is
what the scaling absorbs.

Input: the output of perso/oom_margin_check.sh, which carries one
`[mem-probe]` line per batch (NEMO_MEM_PROBE) with d2_sum, d2_max, dU_sum,
dU_max and peak_alloc. Feed it runs that separate the terms:

  - LONE (one clip per batch, sum == max) only pins a + b, and SPREAD=1 makes
    the label lengths vary at a fixed duration, which is the d^2 vs d*(U+1)
    question itself;
  - the default frontier mode (2 to 16 clips per batch) is what splits a
    from b.

Usage (GPU commands in training_config.yaml, next to cost_batching):
  python perso/fit_cost_model.py refit_lone.log refit_frontier.log
"""

import argparse
import re
import sys

import numpy as np

# The probe's line; oom_margin_check.sh prefixes it with "[oom-check] ".
_FIELD = re.compile(r"(\w+)=\s*([0-9.]+)")
MODELS = {"duration_sq": ("d2_sum", "d2_max"), "duration_tokens": ("dU_sum", "dU_max")}


def parse(paths, skip_first):
    rows = []
    for path in paths:
        kept = 0
        with open(path) as f:
            for line in f:
                if "[mem-probe] batch" not in line:
                    continue
                fields = {k: float(v) for k, v in _FIELD.findall(line)}
                if not all(k in fields for k in ("peak_alloc", "n", "d2_sum", "dU_sum")):
                    # A probe line from before the per-clip columns existed.
                    continue
                kept += 1
                if kept <= skip_first:
                    continue
                rows.append(fields)
    return rows


def fit(rows, sum_key, max_key):
    """Least squares on peak = base + a*S + b*M. Returns (base, a, b, rms, worst, split)."""
    y = np.array([r["peak_alloc"] for r in rows])
    s = np.array([r[sum_key] for r in rows])
    m = np.array([r[max_key] for r in rows])
    # With only lone clips S == M and a, b are not separately identifiable:
    # fit their sum and split it like the current weights do.
    split = bool(np.any(s > m * 1.001))
    if split:
        x = np.column_stack([np.ones_like(s), s, m])
        (base, a, b), *_ = np.linalg.lstsq(x, y, rcond=None)
    else:
        x = np.column_stack([np.ones_like(s), s])
        (base, ab), *_ = np.linalg.lstsq(x, y, rcond=None)
        a, b = 0.4 * ab, 0.6 * ab  # the 4:6 ratio of the duration_sq fit
        x = np.column_stack([np.ones_like(s), s, m])
    residual = y - x @ np.array([base, a, b])
    return base, a, b, float(np.sqrt(np.mean(residual**2))), float(np.abs(residual).max()), split


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("logs", nargs="+", help="oom_margin_check.sh outputs carrying [mem-probe] lines")
    parser.add_argument("--budget", type=float, default=15000.0, help="budget the weights are scaled for")
    # 21.3 GiB: the true peak measured at budget 15000 under duration_sq
    # (training_config.yaml, cost_batching.budget), i.e. ~2.2 GB of headroom.
    parser.add_argument("--target-peak-mib", type=float, default=21800.0)
    parser.add_argument("--skip-first", type=int, default=0, help="drop each log's first N batches (warm-up)")
    args = parser.parse_args()

    rows = parse(args.logs, args.skip_first)
    if len(rows) < 4:
        sys.exit(f"only {len(rows)} usable [mem-probe] lines: need at least 4, from a probe with the d2/dU columns")
    print(f"{len(rows)} batches, n from {min(r['n'] for r in rows):.0f} to {max(r['n'] for r in rows):.0f} clips")

    for name, (sum_key, max_key) in MODELS.items():
        base, a, b, rms, worst, split = fit(rows, sum_key, max_key)
        scale = args.budget / (args.target_peak_mib - base)
        print(
            f"\n{name}: peak = {base:.0f} + {a:.4g}*sum + {b:.4g}*max MiB"
            f"  (rms {rms:.0f} MiB, worst {worst:.0f} MiB)"
            + ("" if split else "\n  only lone clips: a + b is fitted, split 4:6; add a frontier run to split it")
            + f"\n  for budget {args.budget:.0f} at {args.target_peak_mib:.0f} MiB:"
            f" sum_weight: {a * scale:.4g}  max_weight: {b * scale:.4g}"
        )
    print(
        "\nPick the model with the lower rms/worst residual. Paste its weights into"
        " cost_batching (cost_model, sum_weight, max_weight), then re-run the OOM"
        " ladder before trusting the budget."
    )


if __name__ == "__main__":
    main()

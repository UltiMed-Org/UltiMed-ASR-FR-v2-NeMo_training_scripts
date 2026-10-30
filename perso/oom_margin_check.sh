#!/usr/bin/env bash
# Worst-case VRAM check for training_config.yaml. Made with Claude Code.
#
# Builds a manifest of clips carrying the longest transcripts available at their
# length, then trains on it, so every batch is the worst batch training could
# ever draw. If this passes with headroom, a real run cannot OOM on an unlucky
# batch.
#
# With cost batching on (model.train_ds.cost_batching.enabled), the worst case
# is not one batch shape but a frontier: the sampler fills every batch up to the
# same cost budget, so a batch is one 40 s clip or sixteen 15 s ones and both sit
# at the budget. The fixture therefore holds clips at the duration that exactly
# saturates the budget for each batch size in 1, 2, 4, 8, 16, and the sampler
# regroups them into exactly those batches. Passing means the whole frontier
# passes, not just one corner of it.
#
# With cost batching off it falls back to the old behaviour: 200 clips all at
# max_duration, at the fixed batch_size.
#
# Re-run this after changing max_duration, batch_size, cost_batching.budget or
# joint.fused_batch_size.
#
# Raising the cap alone proves nothing under cost batching, because the longest
# clip the sampler will ever put in a batch by itself is sqrt(budget/10), not
# max_duration: at budget 15000 that is 38.7 s, so `oom_margin_check.sh 50`
# would still only test 38.7 s clips. Raise BUDGET with it. For a lone clip of
# D seconds the budget has to be at least 10*D^2 (4*D^2 + 6*D^2), i.e. 20250 at
# 45 s, 25000 at 50 s, 36000 at 60 s.
#
# LONE="45,50,60" builds a fixture of ONLY clips at those durations, which the
# sampler then emits one per batch (a 50 s clip costs 25000, over any budget we
# run, and oversized clips come out alone by design). That is the corner the
# max_duration question is actually about: whether one long PARHAF clip fits by
# itself, which the whole-run peak could never answer because it mixed in the
# sixteen-short-clip corner. No BUDGET override is needed for it, and the cap
# defaults to the longest duration listed.
#
# Every mode logs per-batch VRAM peaks (NEMO_MEM_PROBE, read by the training
# script), so a FAIL says which batch shape failed and a PASS says how close
# each shape came.
#
# SPREAD=1 picks clips spanning the label lengths found at each duration
# instead of only the longest transcripts. That is not a worst case any more:
# it is the fixture for refitting the cost model (perso/fit_cost_model.py),
# which needs U to vary at a fixed d. Raise NEMO_MEM_PROBE to cover every batch.
#
# Usage:  NEMO_EXP_DIR=/tmp/oom_check ./perso/oom_margin_check.sh [cap] [batch]
#         NEMO_EXP_DIR=/tmp/oom_check BUDGET=25000 ./perso/oom_margin_check.sh 50
#         NEMO_EXP_DIR=/tmp/oom_check LONE="40,45,50,55,60" ./perso/oom_margin_check.sh
# Defaults come from the config itself.
set -uo pipefail
cd "$(dirname "$0")/.."

: "${NEMO_EXP_DIR:?set it to a scratch dir, e.g. NEMO_EXP_DIR=/tmp/oom_check}"

PY="${PY:-.venv/bin/python}"
[ -x "$PY" ] || PY=python

CAP="${1:-$($PY -c 'from omegaconf import OmegaConf; print(int(OmegaConf.load("perso/training_config.yaml").model.train_ds.max_duration))')}"
BS="${2:-$($PY -c 'from omegaconf import OmegaConf; print(OmegaConf.load("perso/training_config.yaml").model.train_ds.batch_size)')}"

BUDGET="${BUDGET:-$($PY -c 'from omegaconf import OmegaConf; c=OmegaConf.load("perso/training_config.yaml").model.train_ds.get("cost_batching",None); print(c.get("budget",16500) if c else 0)')}"

# From the config, not hardcoded: this used to name the v1 release and kept
# naming it after the config moved to v2, which silently checked the wrong
# corpus. The first train manifest is the full split (the others are upsampled
# subsets of it).
SRC="$($PY -c 'from omegaconf import OmegaConf; print(OmegaConf.load("perso/training_config.yaml").model.train_ds.manifest_filepath[0])')"
# Lone-clip mode drives the cap, unless the caller passed one explicitly.
if [ -n "${LONE:-}" ] && [ -z "${1:-}" ]; then
  CAP="$($PY -c 'import os, math; print(int(math.ceil(max(float(x) for x in os.environ["LONE"].replace(",", " ").split()))))')"
fi

WORST="$(mktemp -t worst_XXXX.jsonl)"
LOG="$(mktemp -t oom_check_XXXX.log)"
VLOG="$(mktemp -t oom_check_XXXX.vram)"
trap 'rm -f "$WORST" "$VLOG"' EXIT

"$PY" - "$SRC" "$WORST" "$CAP" "$BUDGET" <<'EOF'
import json, os, sys
import numpy as np
from omegaconf import OmegaConf

src, out, cap, budget_arg = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
cost_cfg = OmegaConf.load("perso/training_config.yaml").model.train_ds.get("cost_batching", None)
cost_on = bool(cost_cfg) and bool(cost_cfg.get("enabled", False))

lone = [float(x) for x in os.environ.get("LONE", "").replace(",", " ").split()]
spread = os.environ.get("SPREAD", "") not in ("", "0")
cost_model = cost_cfg.get("cost_model", "duration_sq") if cost_cfg else "duration_sq"

if lone:
    # One corner per duration asked for, each clip alone in its batch because
    # its cost is over any budget we run. 20 clips is enough for the peak to
    # settle past the first batch's allocator warm-up.
    targets, per_target = sorted(lone, reverse=True), 20
elif cost_on:
    # Duration at which a batch of n clips exactly reaches the budget, for the
    # batch sizes the sampler can actually produce. Each is a different corner
    # of the same frontier: same cost, very different shape.
    budget = budget_arg or float(cost_cfg.get("budget", 16500))
    n_max = int(cost_cfg.get("max_batch_size", 16))
    sizes = [n for n in (1, 2, 4, 8, 16) if n <= n_max]
    targets = sorted({round(min(cap, (budget / (4 * n + 6)) ** 0.5), 1) for n in sizes}, reverse=True)
    if cost_model != "duration_sq":
        print(f"[oom-check] WARNING: frontier targets assume duration_sq (4*sum d^2 + 6*max d^2), "
              f"the config uses {cost_model}: the corners are approximate, the probe lines are not")
    per_target = max(200 // len(targets), 1)
else:
    targets, per_target = [cap], 200

# audio_filepath is relative to the manifest's own directory, and this fixture
# is written to a temp dir, so the paths have to be made absolute.
base = os.path.dirname(os.path.abspath(src))
rows = []
with open(src) as f:
    for line in f:
        if line.strip():
            rows.append(json.loads(line))

selected, report = [], []
for target in targets:
    window = 1.0 if lone else (0.75 if cost_on else 1.5)
    at_target = [d for d in rows if target - window <= float(d.get("duration") or 0) <= target]
    # longest transcript at that duration is the worst case for U in the lattice
    at_target.sort(key=lambda d: -len(d.get("text") or ""))
    if spread and len(at_target) > per_target:
        # Refit fixture: label lengths from densest to sparsest at each
        # duration, so the probe sees U move while d holds still.
        picks = np.linspace(0, len(at_target) - 1, per_target).round().astype(int)
        chosen = [at_target[i] for i in picks]
    else:
        chosen = at_target[:per_target]
    selected.extend(chosen)
    report.append(f"{len(chosen)}x{target}s")

with open(out, "w") as f:
    for d in selected:
        p = d["audio_filepath"]
        if not os.path.isabs(p):
            d["audio_filepath"] = os.path.normpath(os.path.join(base, p))
        f.write(json.dumps(d, ensure_ascii=False) + "\n")

mode = ("lone clips" if lone else "cost batching") if cost_on else f"fixed batch_size, cap {cap}s"
mode += ", label lengths spread" if spread else ""
print(f"[oom-check] fixture ({mode}): {len(selected)} clips, {' '.join(report)}")
if len(selected) < sum(min(per_target, 1) for _ in targets):
    print("[oom-check] WARNING: manifest has no clips at some target durations")
EOF

# Per-batch torch peaks, which is what tells the corners apart. This script is
# the diagnostic, so they are on in every mode.
export NEMO_MEM_PROBE="${NEMO_MEM_PROBE:-20}"
export NUMBA_CUDA_USE_NVIDIA_BINDING=1
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8
export PYTHONUNBUFFERED=1

TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 1 > "$VLOG" &
SAMPLER=$!

# One epoch over the fixture is a deliberately truncated run, which the config audit
# refuses unless told (optim.sched.max_steps describes the real run, not this probe).
"$PY" examples/asr/speech_to_text_finetune_cached.py \
  --config-dir="$PWD/perso" --config-name=training_config \
  "model.train_ds.manifest_filepath=[\"$WORST\"]" \
  "model.train_ds.max_duration=${CAP}.0" \
  "model.train_ds.batch_size=${BS}" \
  "model.train_ds.cost_batching.budget=${BUDGET}" \
  trainer.max_epochs=1 \
  config_audit.allow_truncated_anneal=true \
  trainer.val_check_interval=1.0 \
  +trainer.limit_val_batches=0.0 \
  trainer.num_sanity_val_steps=0 \
  exp_manager.create_checkpoint_callback=false \
  exp_manager.resume_if_exists=false \
  exp_manager.version=oom_margin_check \
  > "$LOG" 2>&1
CODE=$?

kill "$SAMPLER" 2>/dev/null; wait "$SAMPLER" 2>/dev/null
PEAK=$(sort -n "$VLOG" | tail -1)

if [ "$CODE" -eq 0 ]; then
  grep -m1 -o "Cost batching on for train:.*" "$LOG" | sed 's/^/[oom-check] /'
  grep -o "\[mem-probe\].*" "$LOG" | sed 's/^/[oom-check] /'
  echo "[oom-check] PASS  peak ${PEAK}/${TOTAL} MiB  (margin $((TOTAL-PEAK)) MiB)"
  echo "[oom-check] under ~1000 MiB of margin is uncomfortably thin for a multi-day run"
elif grep -q "OutOfMemoryError" "$LOG"; then
  echo "[oom-check] FAIL: OOM. Lower max_duration first (cost scales with duration squared), then batch_size."
  grep -o "\[mem-probe\].*" "$LOG" | sed 's/^/[oom-check] /'
  grep -m1 "Tried to allocate" "$LOG"
else
  echo "[oom-check] FAIL (not an OOM), see $LOG"
  tail -5 "$LOG"
fi
rm -f "$LOG"
exit "$CODE"

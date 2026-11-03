#!/bin/bash
# Per-voice attribution sweep: which UltiMed v2 voices hurt real-speech WER?
#
# Context (2026-11-03): arm 2.2.4 (all voices minus h_itf_arm + h_frf_ptm)
# beat the all-voice arms by ~2 raw WER points on the real-speech set (oli)
# at equal fleurs_fr / dictionary / parhaf, so at least one of the hybrid
# voice surgeries damaged the audio in a way neither QC judge saw. That arm
# only shows the PAIR is harmful; it says nothing about each voice alone or
# about the other five hybrids, and acoustic proxies pointed the wrong way
# (stock parakeet-v3 found h_itf_arm the LEAST alien voice). Only
# training-based attribution counts.
#
# Design A (chosen over leave-one-out and Plackett-Burman): one short arm per
# voice, training on that voice's UltiMed slice alone (train.voice.<v>.jsonl +
# its drugs slice x3) plus the constant non-UltiMed sets (drug_sentence,
# nemo_dataset, CommonVoice fr). Effect sizes are largest when a voice is
# alone (a 24% share already moved oli by 2 points), so 1500 steps is enough:
# the 2.2.4 separation was visible at step 1500. Each voice has ~300 h of
# audio and 1500 steps see ~600 h, i.e. about two passes.
#
# Criterion (user decision, 2026-11-03): raw WER on oli + fleurs_fr, averaged.
# Rank the 9 arms; the voices that transfer worst to human speech are the ones
# to drop and regenerate with the remaining voices. Confirm with one 3k
# mixture arm minus those voices before any TTS regeneration.
#
# Cost: ~2h05 per arm (12,000 micro-batches at ~1.9/s + 3 validations), ~19 h
# for the 9 arms. The sweep is resumable: an arm whose step-1500 checkpoint
# exists is skipped. A failed arm is logged and the sweep moves on.
#
# Run from the zellij `run` tab:
#   ./perso/run_voice_sweep.sh            # all 9 voices
#   ./perso/run_voice_sweep.sh h_itf_arm  # one or more voices
#
# Written with Claude Code.

cd "$(dirname "$0")/.." || exit 1

N=./perso/ultimed_data_ignore-backups/v2/NeMO_files
STEPS=1500
# Same resolution as training_start.sh (.env or an exported NEMO_EXP_DIR); exits if unset.
. ./perso/resolve_exp_dir.sh
EXP_DIR=$NEMO_EXP_DIR

# Suspects first so a partial sweep is already informative, references spread out.
VOICES=(fr_female h_itf_arm h_frf_ptm h_frf_esm h_frm_nlf h_frm_ptf h_frf_itm h_itf_def fr_male)
[ $# -gt 0 ] && VOICES=("$@")

failed=()
for v in "${VOICES[@]}"; do
  VER="2.3.voice-$v-$STEPS"
  if ls "$EXP_DIR/$VER/checkpoints/"*"step=$STEPS-"*.ckpt >/dev/null 2>&1; then
    echo "[sweep] $(date '+%F %T') $VER already has its step-$STEPS checkpoint, skipping"
    continue
  fi
  [ -s "$N/train.voice.$v.jsonl" ] || { echo "[sweep] missing $N/train.voice.$v.jsonl"; failed+=("$v"); continue; }
  D="$N/train.drugs.voice.$v.jsonl"
  MF="[$N/train.voice.$v.jsonl,$D,$D,$D,./perso/drug_sentence_dataset/train.json,./perso/nemo_dataset/train.json,./perso/downloaded_datasets_ignore-backups/commonvoice/cv_fr_train.converted.jsonl]"
  echo "[sweep] $(date '+%F %T') starting $VER"
  ./training_start.sh \
    trainer.max_steps=$STEPS model.optim.sched.max_steps=$STEPS \
    trainer.val_check_interval=4000 \
    exp_manager.version="$VER" \
    model.optim.lr=1e-4 model.freeze.encoder_except_last_n=4 \
    model.train_ds.cost_batching.budget=15000 \
    "model.train_ds.manifest_filepath=$MF" \
    || { echo "[sweep] $(date '+%F %T') $VER FAILED (exit $?), moving on"; failed+=("$v"); }
done

echo "[sweep] $(date '+%F %T') sweep finished; failed arms: ${failed[*]:-none}"

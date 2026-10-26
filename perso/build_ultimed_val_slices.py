#!/usr/bin/env python
"""Build small deterministic validation slices from the UltiMed NeMo manifests.

The full UltiMed val split is 57.7k clips (~310 h), far too much audio to
decode at every validation round. This script samples a fixed, seeded subset
per source so validation stays cheap while still tracking every source.

Every UltiMed slice samples one category out of the RELEASE-WIDE split
(top-level val.jsonl / test.jsonl, the split the trainer's train.jsonl comes
from), never from a per-source <source>/val.jsonl. Those used to be an
independent per-source partition written by 01_build_nemo_manifest.py before
02_combine_nemo_manifests.py re-split the whole corpus, so a third to three
quarters of a per-source val manifest sat in the global train.jsonl (measured
2026-09-25: dictionary 33%, PARHAF 61%, acronyms 67%, drugs 72%). Sampling val slices
from there made validation score training clips (every count still added up,
since both are complete partitions, which is why it went unnoticed). Since
2026-09-27 the combine rewrites every <source>/{train,val,test}.jsonl from the
release-wide split, so on current manifests the two agree; the top-level files
stay the source here anyway, since manifests combined before that fix (or with
--no-stratify-duration) may still disagree. The trainer's data_leak_check refuses
a config whose val overlaps train either way.

Slices are written next to their source manifest because audio paths inside
are relative to the manifest's directory.

Clips longer than --max-duration are excluded from the validation slices:
validation computes the RNNT loss lattice, whose memory scales with (batch
padded length), so one 60 s clip in a batch would spike VRAM. Training applies
the same 45 s cap anyway.

The test slices serve the final ONNX benchmark. They keep every duration, since
that benchmark decodes one clip at a time and must represent the split as
released: 10,000 clips in total, every acronym and drug clip plus fixed PARHAF
and dictionary samples.

A second family of validation slices is per VOICE rather than per source: v2 is
spoken by nine TTS voices, and one of them being learnt worse than the others is
invisible in a per-source metric since every source uses all nine. See
VOICE_QUOTAS below for the mix, the cost and why they stay out of macro_metrics.

PARROT is eval-only (CC BY-NC-SA), absent from the release-wide splits, and
never trained on, so its out-of-domain monitor slice comes from PARROT/test.jsonl.

One TRAINING subset is written here too, because it is the same "one category
of a release-wide split" filter: train.drugs.jsonl holds EVERY drugs row of
train.jsonl (no sampling, no duration cap: the trainer applies its own
max_duration). The training config lists it after train.jsonl extra times to
upsample the drug names, which are only ~2.3% of the training audio otherwise
(NeMo reads a manifest listed N times N times per epoch). Its rows are a subset
of train.jsonl, so data_leak_check treats them like any other training row.

Made with Claude Code.

Usage:
    python perso/build_ultimed_val_slices.py \
        [--nemo-files-dir ./perso/ultimed_data_ignore-backups/NeMO_files]
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

# (source manifest, category or None for all rows, output, sample size or None
#  for every eligible row, duration-capped). Paths are relative to
#  --nemo-files-dir.
SLICES = [
    ("val.jsonl", "dictionary", "val.dictionary.down-600.jsonl", 600, True),
    ("val.jsonl", "parhaf", "val.parhaf.down-300.jsonl", 300, True),
    ("val.jsonl", "drugs", "val.drugs.down-300.jsonl", 300, True),
    ("val.jsonl", "acronyms", "val.acronyms.down-150.jsonl", 150, True),
    ("PARROT/test.jsonl", None, "PARROT/test.down-200.jsonl", 200, True),
    ("test.jsonl", "dictionary", "test.dictionary.down-5290.jsonl", 5290, False),
    ("test.jsonl", "parhaf", "test.parhaf.down-2500.jsonl", 2500, False),
    ("test.jsonl", "drugs", "test.drugs.down-2059.jsonl", 2059, False),
    ("test.jsonl", "acronyms", "test.acronyms.down-151.jsonl", 151, False),
    ("train.jsonl", "drugs", "train.drugs.jsonl", None, False),
]

#: Per-voice validation slices: one small set per v2 TTS voice, so tensorboard
#: carries nine ``voice_<name>val_wer`` curves next to the per-source ones. What
#: they are for: a voice the model handles worse than the others is invisible in
#: a per-source metric, because every source is spoken by all nine. The QC flag
#: rates say that is a real risk, spanning 14x from fr_female at 0.30% to
#: h_itf_def at 4.26% (and on PARHAF alone the worst voice is a different one,
#: h_frf_ptm at 19.9%), so a voice may well be learnt worse than the average.
#:
#: The category mix is FIXED, not sampled, so the nine curves are comparable: an
#: accidentally dictionary-heavy slice would otherwise read as a better voice.
#: acronyms is left out because the val split holds only 12-25 acronym clips per
#: voice, which is noise, and acronyms keeps its own set anyway.
#:
#: Cost: 75 clips per voice is ~0.46 h of audio, ~4.2 h for the nine. A full
#: validation pass measured 4.5 min over roughly 10 h of audio, so this is about
#: 2 min a round, ~40 min over a 20-round run. It buys ~4.4k reference words per
#: voice (40 dictionary clips at ~21 s, 25 parhaf at ~30 s, 10 drugs at ~15 s),
#: which is enough to see a 2-point WER gap between voices and not enough to
#: chase a 0.5-point one.
#:
#: These sets are observability only: they overlap the per-source slices (same
#: val split), so training_config.yaml keeps them OUT of macro_metrics, where
#: they would count the same clips a second time in the checkpoint monitor.
VOICE_QUOTAS = {"dictionary": 40, "parhaf": 25, "drugs": 10}

SEED = 42


def _sample(rows: list[str], n: "int | None") -> list[str]:
    """``n`` of ``rows``, or all of them, drawn the same way on every run."""
    rng = random.Random(SEED)
    return rows if n is None or len(rows) <= n else rng.sample(rows, n)


def write_slice(rows: list[str], n: "int | None", out: Path) -> None:
    picked = _sample(rows, n)
    out.write_text("\n".join(picked) + "\n")
    hours = sum(json.loads(r)["duration"] for r in picked) / 3600
    print(f"{out}: {len(picked)} clips ({hours:.2f} h) from {len(rows)} eligible")


def read_rows(src: Path, keep) -> list[str]:
    rows = []
    with src.open() as f:
        for line in f:
            line = line.strip()
            if line and keep(json.loads(line)):
                rows.append(line)
    return rows


def write_voice_slices(base: Path, keep) -> None:
    """One validation slice per voice, each with the same category mix.

    Args:
        base: the NeMO_files directory, read for ``val.jsonl``.
        keep: the duration filter the per-source val slices use, so a voice
            slice cannot hold a clip those would have dropped.
    """
    pools = defaultdict(list)
    for line in read_rows(base / "val.jsonl", keep):
        row = json.loads(line)
        if row["category"] in VOICE_QUOTAS and row.get("voice"):
            pools[(row["voice"], row["category"])].append(line)
    if not pools:
        print("no voice field in val.jsonl, skipping the per-voice slices")
        return

    size = sum(VOICE_QUOTAS.values())
    for voice in sorted({voice for voice, _ in pools}):
        picked = []
        for category, n in sorted(VOICE_QUOTAS.items()):
            pool = pools[(voice, category)]
            if len(pool) < n:
                # Not fatal, but it breaks the comparability the fixed mix is
                # for, so it has to be visible rather than silently smaller.
                print(f"  WARNING {voice}/{category}: {len(pool)} clips for a quota of {n}")
            picked.extend(_sample(pool, n))
        write_slice(picked, None, base / f"val.voice.{voice}.down-{size}.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--nemo-files-dir",
        # v2 (nine TTS voices), which is what training_config.yaml points at. The
        # v1 directory is one level up and still on disk, so a default left there
        # would quietly rebuild v1 slices that no config reads any more.
        default="./perso/ultimed_data_ignore-backups/v2/NeMO_files",
        help="UltiMed NeMO_files directory (release-wide splits + PARROT/)",
    )
    parser.add_argument("--max-duration", type=float, default=45.0)
    parser.add_argument("--min-duration", type=float, default=1.0)
    args = parser.parse_args()

    base = Path(args.nemo_files_dir)
    for src_name, category, out_name, n, capped in SLICES:

        def keep(r: dict, category=category, capped=capped) -> bool:
            if category is not None and r["category"] != category:
                return False
            return not capped or args.min_duration <= r["duration"] <= args.max_duration

        write_slice(read_rows(base / src_name, keep), n, base / out_name)

    write_voice_slices(
        base, lambda r: args.min_duration <= r["duration"] <= args.max_duration
    )

if __name__ == "__main__":
    main()

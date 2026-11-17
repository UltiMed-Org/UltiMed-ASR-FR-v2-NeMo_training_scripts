"""Build FLEURS-layout WER fixtures for the medical eval sets.

parakeet_web's scripts/wer-quants.py --manifest mode resolves every clip as
<manifest dir>/wavs_validation/<basename(audio_filepath)>, i.e. the FLEURS
layout. The UltiMed / drug_sentence eval manifests instead point into
per-category subfolders, so this script materializes each eval set as
  <out>/<label>/validation.json      (audio_filepath = flat unique basename)
  <out>/<label>/wavs_validation/     (symlinks to the real audio files)
which makes the medical WER benchmark a plain multi-manifest wer-quants run.

Idempotent; symlinks and manifests are rewritten each run. Written with
Claude Code.
"""

import argparse
import json
from pathlib import Path

# Default manifest directory (v1). Pass --manifest-dir to build from another
# export with the same filenames, e.g. .../ultimed_data_ignore-backups/v2/NeMO_files.
MANIFEST_DIR = "./perso/ultimed_data_ignore-backups/NeMO_files"

# label -> eval manifest, relative to --manifest-dir (the same downsampled val sets the training monitor
# used for the in-domain part of combined_macro_val_wer, plus acronyms)
SETS = {
    "med_dictionary": "val.dictionary.down-600.jsonl",
    "med_parhaf": "val.parhaf.down-300.jsonl",
    "med_drugs": "val.drugs.down-300.jsonl",
    "med_acronyms": "val.acronyms.down-150.jsonl",
    # PARROT is eval-only: unlike the val.*.down-* sets above it was never
    # used to monitor training, so it is the one genuinely held-out medical set.
    "med_parrot": "PARROT/test.down-200.jsonl",
}

# label -> manifest relative to the repo root, NOT to --manifest-dir: the
# drug_sentence set has a single version shared by every export.
FIXED_SETS = {
    "med_drug_sentence": "./perso/drug_sentence_dataset/val.json",
}

# Release test split (never trained on, never used for checkpoint selection),
# 10,000 clips sampled by build_ultimed_val_slices.py, plus all of PARROT test.
TEST_SETS = {
    "test_dictionary": "test.dictionary.down-5290.jsonl",
    "test_parhaf": "test.parhaf.down-2500.jsonl",
    "test_drugs": "test.drugs.down-2059.jsonl",
    "test_acronyms": "test.acronyms.down-151.jsonl",
    "test_parrot": "PARROT/test.jsonl",
}


def resolve(manifest: Path, audio_filepath: str) -> Path:
    # UltiMed paths are relative to the manifest dir; drug_sentence paths are
    # relative to the repo root (same rule as build_calibration_medical.py).
    path = (manifest.parent / audio_filepath).resolve()
    if not path.exists():
        path = Path(audio_filepath).resolve()
    return path


def build(label: str, manifest: Path, out_root: Path, only_stems: set | None = None) -> None:
    # only_stems keeps just the rows whose audio stem is listed (e.g. the
    # re-synthesised clips), under the SAME "NNNNN_" name the full fixture
    # gives them, so a partial re-run can be spliced back by name.
    out_dir = out_root / label
    wav_dir = out_dir / "wavs_validation"
    wav_dir.mkdir(parents=True, exist_ok=True)
    rows, missing = [], 0
    with open(manifest, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            src = resolve(manifest, r["audio_filepath"])
            if only_stems is not None and src.stem not in only_stems:
                continue
            if not src.exists():
                missing += 1
                continue
            name = f"{i:05d}_{src.name}"
            link = wav_dir / name
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(src)
            rows.append({"audio_filepath": name, "text": r["text"],
                         "duration": r.get("duration")})
    with open(out_dir / "validation.json", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{label}: {len(rows)} clips ({missing} missing) -> {out_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="./perso/ultimed_data_ignore-backups/wer_fixtures")
    ap.add_argument("--manifest-dir", default=MANIFEST_DIR,
                    help="directory holding the val.*/test.*/PARROT manifests "
                         "(default: the v1 export)")
    ap.add_argument("--only-stems", default=None,
                    help="text file, one audio stem per line: build only those rows "
                         "(original NNNNN_ names kept), e.g. 06_hotfixes/resynth_stems.txt")
    args = ap.parse_args()
    only_stems = None
    if args.only_stems:
        only_stems = {l.strip() for l in open(args.only_stems, encoding="utf-8") if l.strip()}
    out_root = Path(args.out)
    manifest_dir = Path(args.manifest_dir)
    manifests = {label: manifest_dir / mf for label, mf in {**SETS, **TEST_SETS}.items()}
    manifests.update({label: Path(mf) for label, mf in FIXED_SETS.items()})
    for label, mf in manifests.items():
        assert mf.exists(), f"missing manifest for {label}: {mf}"
        build(label, mf, out_root, only_stems)
    print("DONE")


if __name__ == "__main__":
    main()

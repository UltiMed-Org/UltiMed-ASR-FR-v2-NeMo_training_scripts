"""Fold label charsets of NeMo manifests in place, then check them against the tokenizer.

Per file, atomically (temp file next to it, then rename):
  1. the "text" field goes through label_charset.normalize_label_charset;
     with --convert-number-words fr it then goes through the dataset repo's
     UltiMed label rules (the repo
     found via $ULTIMED_SCRIPTS_DIR, default ../UltiMed-ASR-FR-v1-scripts;
     ultimed_scripts.general_french_label_normalizer, i.e. without the
     medical-only rules): compound number words to digits ("vingt et un" -> "21", "mille neuf cent
     quatre-vingt-dix" -> "1990", "trois virgule cinq" -> "3,5"), "95 %",
     titles, dates, clock times. A lone number word ("deux", "cent") and an
     ordinal ("vingt-et-unième") stay spelled, as in UltiMed labels;
  2. each label is checked against the real tokenizer with the training
     guard's per-character cache (label_vocab_check.label_oov_chars), so a
     manifest that passes here passes the startup check of the finetune script;
  3. --drop-oov removes the lines still carrying an <unk> character (printed
     with their offending characters), otherwise they are only reported;
  4. --drop-number-words fr (exclusive with --convert-number-words) instead
     removes lines whose text spells a French number other than un/une
     (label_charset.has_french_number_word). That is how the 1.x CV-fr
     manifest lost 11,189 lines on 2026-11-01; converting replaced it.

Unchanged lines are kept byte for byte and a changed line keeps its
\\u-escaping style, so a tracked manifest diffs only where a label changed.
With --output-suffix the result goes to a new file beside each input
(``x.json`` -> ``x<suffix>.json``) and the input stays untouched, e.g. to keep
both the original and the converted references of an evaluation set.

    python perso/renormalize_manifests.py perso/nemo_dataset/*.json \\
        --tokenizer-from ../parakeet-tdt-0.6b-v3-ultra-onnx/parakeet-tdt-0.6b-v3-ultra.nemo --drop-oov

Written with Claude Code.
"""

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from label_charset import has_french_number_word, normalize_label_charset
from ultimed_scripts import general_french_label_normalizer

NUMBER_WORD_FILTERS: Dict[str, Callable[[str], bool]] = {"fr": has_french_number_word}
# Language -> factory of the label rewrite that writes its numbers in digits.
NUMBER_WORD_CONVERTERS: Dict[str, Callable[[], Callable[[str], str]]] = {"fr": general_french_label_normalizer}


@dataclass
class RenormReport:
    total: int = 0
    changed: int = 0
    # (text, offending characters) of lines that still contain <unk>
    oov: List[Tuple[str, str]] = field(default_factory=list)
    oov_dropped: bool = False
    number_word_dropped: List[str] = field(default_factory=list)
    # (before, after) of lines the ``convert`` rewrite changed, before = after the charset fold
    converted: List[Tuple[str, str]] = field(default_factory=list)


def renormalize_manifest(
    path: Path,
    oov_chars: Optional[Callable[[str], List[str]]] = None,
    drop_oov: bool = False,
    drop_if: Optional[Callable[[str], bool]] = None,
    convert: Optional[Callable[[str], str]] = None,
    dest: Optional[Path] = None,
) -> RenormReport:
    """Rewrite ``path`` in place, or into ``dest`` leaving ``path`` untouched. ``convert(text)`` rewrites the folded label
    (e.g. the UltiMed number rules), before the checks; ``oov_chars(text)``
    returns the characters that make ``text`` tokenize to <unk> (``[]`` when
    clean); ``drop_if(text)`` drops a line on top of that. Counts are over the
    input lines."""
    report = RenormReport(oov_dropped=drop_oov)
    dest = path if dest is None else dest
    tmp = dest.with_name(dest.name + ".renorm.part")
    with open(path, encoding="utf-8") as src, open(tmp, "w", encoding="utf-8") as out:
        for line in src:
            if not line.strip():
                out.write(line)
                continue
            report.total += 1
            row = json.loads(line)
            text = normalize_label_charset(row["text"])
            if convert is not None:
                folded, text = text, convert(text)
                if text != folded:
                    report.converted.append((folded, text))
            if text != row["text"]:
                report.changed += 1
                row["text"] = text
                line = json.dumps(row, ensure_ascii=line.isascii()) + "\n"
            bad = oov_chars(text) if oov_chars else []
            if bad:
                report.oov.append((text, "".join(dict.fromkeys(bad))))
                if drop_oov:
                    continue
            if drop_if is not None and drop_if(text):
                report.number_word_dropped.append(text)
                continue
            out.write(line)
    os.replace(tmp, dest)
    return report


def suffixed(path: Path, suffix: str) -> Path:
    """``x.json`` + ``.converted`` -> ``x.converted.json``."""
    return path.with_name(path.stem + suffix + path.suffix)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("files", nargs="+", type=Path, metavar="MANIFEST")
    ap.add_argument("--tokenizer-from", required=True,
                    help="the .nemo whose tokenizer trains on these labels, or a tokenizer.model")
    ap.add_argument("--drop-oov", action="store_true",
                    help="remove lines still containing a character the tokenizer maps to <unk> "
                         "(default: only report them)")
    numbers = ap.add_mutually_exclusive_group()
    numbers.add_argument("--convert-number-words", choices=sorted(NUMBER_WORD_CONVERTERS), default=None,
                         help="rewrite labels with the UltiMed label rules of the dataset repo "
                              "(numbers in digits, 95 %%, ...; found via $ULTIMED_SCRIPTS_DIR)")
    numbers.add_argument("--drop-number-words", choices=sorted(NUMBER_WORD_FILTERS), default=None,
                         help="instead remove lines spelling a number in words in this language")
    ap.add_argument("--quiet-drops", action="store_true", help="print dropped labels as a count only")
    ap.add_argument("--output-suffix", default=None,
                    help="write each result beside its input as <stem><suffix><ext> (e.g. .converted) "
                         "and leave the input untouched (default: rewrite in place)")
    args = ap.parse_args()
    if args.output_suffix == "":
        ap.error("--output-suffix must not be empty (omit it to rewrite in place)")

    # Heavy (NeMo) imports only once the arguments are valid.
    from nemo.collections.asr.parts.utils.label_vocab_check import label_oov_chars, load_tokenizer, unk_id_of

    tokenizer = load_tokenizer(args.tokenizer_from)
    unk = unk_id_of(tokenizer)
    cache: Dict[str, bool] = {}
    drop_if = NUMBER_WORD_FILTERS[args.drop_number_words] if args.drop_number_words else None
    convert = NUMBER_WORD_CONVERTERS[args.convert_number_words]() if args.convert_number_words else None

    n_oov_left = 0
    for path in args.files:
        dest = suffixed(path, args.output_suffix) if args.output_suffix else path
        r = renormalize_manifest(path, lambda t: label_oov_chars(t, tokenizer, unk, cache), args.drop_oov, drop_if,
                                 convert, dest)
        verb = "dropped" if r.oov_dropped else "KEPT (still <unk>)"
        print(f"{dest}: {r.total} lines, {r.changed} changed, {len(r.oov)} <unk> {verb}, "
              f"{len(r.converted)} converted, {len(r.number_word_dropped)} number-word dropped", flush=True)
        if not r.oov_dropped:
            n_oov_left += len(r.oov)
        if not args.quiet_drops:
            for text, chars in r.oov:
                print(f"  <unk> {' '.join(f'U+{ord(c):04X}{c!r}' for c in chars)}: {text!r}")
            for text in r.number_word_dropped:
                print(f"  number word: {text!r}")
            for before, after in r.converted:
                print(f"  converted: {before!r} -> {after!r}")
    if n_oov_left:
        raise SystemExit(f"{n_oov_left} label(s) still tokenize to <unk>; rerun with --drop-oov to remove them")


if __name__ == "__main__":
    main()

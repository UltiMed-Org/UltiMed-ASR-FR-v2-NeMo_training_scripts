"""Refuse to train when a label contains a character the tokenizer cannot encode.

Written after the parakeet SentencePiece vocabulary (8193 pieces) turned out
to lack typographic characters such as ``’ « » – — …`` that ~14k training
labels contained. Each of those tokenized to ``<unk>``, so the model was
trained to emit ``<unk>`` and nothing complained.

What is checked
---------------
Every manifest the config lists under ``model.train_ds``, every
``model.validation_ds`` entry and ``model.test_ds`` (same discovery as
:mod:`data_leak_check`, via :func:`collect_manifests`). Each row's ``text``
(falling back to ``normalized_text``, like the leak check) is checked against
``tokenizer.text_to_ids`` / ``tokenizer.unk_id``. The cached-encoder training
path reads its transcripts from the same ``train_ds.manifest_filepath`` list,
so it is covered too.

Every manifest is scanned to the end before deciding, so one run lists every
offending character of every set, then :class:`LabelVocabError` is raised.

Why it is fast
--------------
Tokenizing ~750k labels would take minutes. Instead each DISTINCT character is
tokenized once (cached), which is a few thousand calls in total. Only a label
containing a character whose own tokenization yields ``<unk>`` is then fully
tokenized, to confirm it really produces ``<unk>`` in context. A label made
only of known characters is assumed clean: SentencePiece emits ``<unk>`` only
for characters outside its character set, so it cannot appear there.

Set ``ULTIMED_SKIP_VOCAB_CHECK=1`` to skip the check (logged loudly).

Written with Claude Code.
"""

import json
import os
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List

from nemo.collections.asr.parts.utils.data_leak_check import collect_manifests
from nemo.utils import logging

SKIP_ENV_VAR = "ULTIMED_SKIP_VOCAB_CHECK"
_MAX_EXAMPLES = 3


class LabelVocabError(RuntimeError):
    """At least one label of one manifest tokenizes to ``<unk>``."""


@dataclass
class _ManifestReport:
    path: str
    n_labels: int = 0
    n_bad_labels: int = 0
    bad_chars: Counter = field(default_factory=Counter)
    examples: List[str] = field(default_factory=list)


def unk_id_of(tokenizer) -> int:
    """``unk_id`` is a property on NeMo's SentencePieceTokenizer, a method on raw spm."""
    unk = getattr(tokenizer, "unk_id", None)
    if unk is None:
        raise AttributeError(f"{type(tokenizer).__name__} exposes no unk_id, cannot check labels")
    return unk() if callable(unk) else unk


def label_oov_chars(text: str, tokenizer, unk: int, char_is_oov: Dict[str, bool]) -> List[str]:
    """The characters of ``text`` the tokenizer maps to ``<unk>`` (with repeats,
    in order), or ``[]`` when the label tokenizes cleanly. ``char_is_oov`` is the
    per-character cache, shared across calls; see the module docstring."""
    chars = set(text)
    for ch in chars - char_is_oov.keys():
        char_is_oov[ch] = unk in tokenizer.text_to_ids(ch)
    if not any(char_is_oov[ch] for ch in chars) or unk not in tokenizer.text_to_ids(text):
        return []
    return [ch for ch in text if char_is_oov[ch]]


def load_tokenizer(path: str):
    """NeMo SentencePieceTokenizer from a ``tokenizer.model`` or from the one
    packed inside a ``.nemo`` archive, without loading the model (CPU, seconds).
    For standalone label checks such as perso/renormalize_manifests.py."""
    import tarfile
    import tempfile

    from nemo.collections.common.tokenizers.sentencepiece_tokenizer import SentencePieceTokenizer

    if not path.endswith(".nemo"):
        return SentencePieceTokenizer(model_path=path)
    with tarfile.open(path) as tf:
        members = [m for m in tf.getmembers() if m.name.endswith("tokenizer.model")]
        if len(members) != 1:
            raise ValueError(f"{path}: expected one *tokenizer.model member, found {[m.name for m in members]}")
        with tempfile.TemporaryDirectory() as tmp:
            tf.extract(members[0], tmp)
            return SentencePieceTokenizer(model_path=os.path.join(tmp, members[0].name))


def _scan_manifest(path: str, tokenizer, unk: int, char_is_oov: Dict[str, bool]) -> _ManifestReport:
    report = _ManifestReport(path=path)
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            text = row.get("text", row.get("normalized_text")) or ""
            report.n_labels += 1
            bad = label_oov_chars(text, tokenizer, unk, char_is_oov)
            if not bad:
                continue
            report.n_bad_labels += 1
            report.bad_chars.update(bad)
            if len(report.examples) < _MAX_EXAMPLES:
                report.examples.append(text)
    return report


def _format_report(reports: List[_ManifestReport]) -> str:
    lines = []
    for r in reports:
        lines.append(f"  {r.path}: {r.n_bad_labels:,} / {r.n_labels:,} labels contain <unk>")
        for ch, n in r.bad_chars.most_common():
            lines.append(f"    {ch!r} U+{ord(ch):04X}: {n:,} occurrence(s)")
        for ex in r.examples:
            lines.append(f"    e.g. {ex!r}")
    return "\n".join(lines)


def check_labels_in_vocab(tokenizer, cfg) -> None:
    """Scan every train/validation/test label, raise if any tokenizes to ``<unk>``.

    Parameters
    ----------
    tokenizer
        The model's tokenizer (``asr_model.tokenizer``): needs ``text_to_ids``
        and ``unk_id``.
    cfg : DictConfig
        Full training config; manifests are read from ``cfg.model``.

    Raises
    ------
    LabelVocabError
        After scanning everything, if at least one label contains ``<unk>``.
    FileNotFoundError
        If a listed manifest does not exist.
    """
    if os.environ.get(SKIP_ENV_VAR, "") not in ("", "0"):
        logging.warning(
            f"!!! {SKIP_ENV_VAR} is set: the label vocabulary check is SKIPPED. Labels with "
            "characters outside the tokenizer vocab will silently train the model to emit <unk>. !!!"
        )
        return

    start = time.perf_counter()
    unk = unk_id_of(tokenizer)
    manifests = []
    for key in ("train_ds", "validation_ds", "test_ds"):
        manifests.extend(collect_manifests(cfg.model.get(key, None)))
    manifests = list(dict.fromkeys(manifests))

    missing = [m for m in manifests if not os.path.isfile(m)]
    if missing:
        raise FileNotFoundError("label vocab check: manifest(s) not found:\n  " + "\n  ".join(missing))

    char_is_oov: Dict[str, bool] = {}
    reports = [_scan_manifest(m, tokenizer, unk, char_is_oov) for m in manifests]
    elapsed = time.perf_counter() - start
    n_labels = sum(r.n_labels for r in reports)
    bad = [r for r in reports if r.n_bad_labels]

    if bad:
        all_chars = sorted({ch for r in bad for ch in r.bad_chars})
        summary = ", ".join(f"{ch!r} U+{ord(ch):04X}" for ch in all_chars)
        message = (
            f"label vocab check: {sum(r.n_bad_labels for r in bad):,} of {n_labels:,} labels in "
            f"{len(bad)} manifest(s) contain characters the tokenizer maps to <unk>: {summary}\n"
            f"{_format_report(bad)}\n"
            "Fix the datasets (normalize or drop those characters) before training. "
            f"{SKIP_ENV_VAR}=1 skips this check."
        )
        logging.error(message)
        raise LabelVocabError(message)

    logging.info(
        f"label vocab check: {n_labels:,} labels in {len(manifests)} manifest(s) all in vocab "
        f"({len(char_is_oov)} distinct characters, {elapsed:.1f} s)."
    )

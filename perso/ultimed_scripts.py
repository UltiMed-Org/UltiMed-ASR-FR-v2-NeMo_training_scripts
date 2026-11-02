"""Import the UltiMed label rules from the sibling dataset repo instead of copying them.

The dataset repo (UltiMed-ASR-FR-v1-scripts) owns how an UltiMed label is written:
numbers in digits (``mille neuf cent quatre-vingt-dix`` -> ``1990``, ``trois virgule
cinq`` -> ``3,5``), ``95 %``, titles, dates, clock times, drug casing... Rehearsal labels
built here must follow the same rules, or the model learns two spellings of one sound.
This module is the ONE place that finds that repo; every script that needs its rules
goes through it.

The repo is found through the environment variable ``ULTIMED_SCRIPTS_DIR`` (read from
the shell, not from ``.env``: only training_start.sh sources that file), defaulting to
``../UltiMed-ASR-FR-v1-scripts`` next to this repo's root, since both are cloned side by
side. A missing repo or module fails with :class:`UltimedScriptsNotFound` naming the
path and the variable, never with a bare ImportError.

Two normalizers are exposed, both built from functions imported from that repo:

- :func:`ultimed_label_normalizer` is ``99_hf_release/05_normalize_text.py::normalize_label``
  itself, the exact chain UltiMed's own manifests go through:
  ``ParakeetTokenizer.clean_label`` (NFC, NFKC fold of the characters the Parakeet vocab
  lacks, whitespace collapse), then ``percent_normalize.percent_to_symbol``, then the whole
  of ``label_conventions.apply_label_conventions``.
- :func:`general_french_label_normalizer`, what Common Voice rehearsal uses, is the same
  chain with ``apply_label_conventions`` cut down to the steps that hold on general French:
  titles, dates, clock times, quantities, then ``normalize_label_numbers`` (elided ``d'un``
  before a unit, thousands and compound numbers; in ``apply_label_conventions``'s own order). The rest is medical and misfires on
  encyclopedic sentences, measured on the 139k CV-fr labels on 2026-11-01: the
  prefix-compound rule glued 854 labels (``néo-zélandais`` -> ``néozélandais``, ``péri
  pendant`` -> ``péripendant``), ``normalize_spelling`` turned 44 ``colons`` (settlers)
  into ``côlons`` (it also does ``gene`` -> ``gène``, ``Elodie`` -> ``Hélodie``), the
  generic-lab strip deleted ``Sandoz`` from ``dit Sandoz.``, and the roman / staging rules
  read ``type II``-like text as medical staging. Cost of skipping ``normalize_spelling``:
  CV keeps ``œ`` (``cœur``, 1.9k labels) where UltiMed writes ``oe``. Kept steps still
  include ``normalize_quantities``, so ``dans un an`` -> ``dans 1 an`` as in UltiMed
  (``Un an plus tard`` stays, as it does there).

The ParakeetTokenizer reads the dataset repo's ``utils/parakeet_vocab.txt``, the same file
as this repo's ``perso/exports/ultimed/2026-09-01-web2/vocab.txt``. Needs ``click`` and
``loguru`` (both in the training venv).

Written with Claude Code.
"""

import functools
import os
import sys
from pathlib import Path
from typing import Callable

ENV_VAR = "ULTIMED_SCRIPTS_DIR"
DEFAULT_DIR = Path(__file__).resolve().parents[1].parent / "UltiMed-ASR-FR-v1-scripts"
NORMALIZE_SCRIPT = Path("99_hf_release") / "05_normalize_text.py"
_REQUIRED = (Path("utils") / "script_module.py", NORMALIZE_SCRIPT)


class UltimedScriptsNotFound(RuntimeError):
    """The dataset repo, or a module it should hold, is missing or does not import."""


def scripts_dir() -> Path:
    """Root of the dataset repo: ``$ULTIMED_SCRIPTS_DIR`` or the sibling-checkout default."""
    raw = os.environ.get(ENV_VAR)
    root = (Path(raw).expanduser() if raw else DEFAULT_DIR).resolve()
    source = f"${ENV_VAR}" if raw else f"the default (set ${ENV_VAR} to override)"
    if not root.is_dir():
        raise UltimedScriptsNotFound(
            f"UltiMed-ASR-FR-v1-scripts not found at {root}, from {source}. Clone it next to "
            f"this repo or point {ENV_VAR} at your checkout.")
    missing = [str(rel) for rel in _REQUIRED if not (root / rel).is_file()]
    if missing:
        raise UltimedScriptsNotFound(
            f"{root} (from {source}) lacks {', '.join(missing)}: not an UltiMed-ASR-FR-v1-scripts "
            f"checkout, or one too old for this repo.")
    return root


@functools.lru_cache(maxsize=None)
def _load_normalize_script(root: Path):
    utils = str(root / "utils")
    if utils not in sys.path:
        sys.path.insert(0, utils)
    try:
        from script_module import load_script

        return load_script(root / NORMALIZE_SCRIPT, "ultimed_05_normalize_text")
    except ImportError as e:
        raise UltimedScriptsNotFound(f"cannot import {root / NORMALIZE_SCRIPT}: {e}") from e


@functools.lru_cache(maxsize=None)
def _label_normalizer(root: Path) -> Callable[[str], str]:
    mod = _load_normalize_script(root)
    return functools.partial(mod.normalize_label, tok=mod.ParakeetTokenizer())


def _general_french(text: str, mod, lc, tok) -> str:
    text = lc.normalize_quantities(lc.normalize_clock(lc.normalize_dates(lc.expand_titles(
        mod.percent_to_symbol(tok.clean_label(text))))))
    return lc.normalize_label_numbers(text)


@functools.lru_cache(maxsize=None)
def _general_french_normalizer(root: Path) -> Callable[[str], str]:
    mod = _load_normalize_script(root)
    import label_conventions  # on sys.path once the script is loaded

    return functools.partial(_general_french, mod=mod, lc=label_conventions, tok=mod.ParakeetTokenizer())


def general_french_label_normalizer() -> Callable[[str], str]:
    """``text -> text``: the UltiMed chain minus its medical-only rules (module docstring).

    >>> f = general_french_label_normalizer()
    >>> f("M. Dupont, né en mille neuf cent quatre-vingt-dix, a vingt et un ans.")
    'Monsieur Dupont, né en 1990, a 21 ans.'
    >>> f("Les colons néo-zélandais du vingt-et-unième siècle, une œuvre.")
    'Les colons néo-zélandais du vingt-et-unième siècle, une œuvre.'
    """
    return _general_french_normalizer(scripts_dir())


def ultimed_label_normalizer() -> Callable[[str], str]:
    """``text -> text``: the dataset repo's ``normalize_label``, the form every UltiMed label has.

    >>> ultimed_label_normalizer()("Né en mille neuf cent quatre-vingt-dix, à quatre-vingt-dix pour cent.")
    'Né en 1990, à 90 %.'
    """
    return _label_normalizer(scripts_dir())

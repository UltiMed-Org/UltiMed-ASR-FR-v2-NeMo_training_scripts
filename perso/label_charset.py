"""Label text folds shared by the manifest builders and perso/renormalize_manifests.py.

Stdlib only, so the dataset scripts can import it without pulling in NeMo.
Two concerns live here:

- normalize_label_charset: fold characters the parakeet SentencePiece vocab
  lacks (a label carrying one tokenizes to <unk>, and the model learns to emit
  it: "quelle qu ⁇ en soit") into in-vocab equivalents. Only folds that keep
  the label faithful to the audio; letters of other alphabets and symbols with
  no spoken-neutral form (& = + ° µ ±) are left alone, so the tokenizer check
  of renormalize_manifests.py can drop those labels instead.
- has_french_number_word: UltiMed labels write quantities in digits while
  Common Voice French writes them in words (27% of its sentences), which
  taught the model "vingt-cinq milligrammes". French rehearsal labels with a
  number word were dropped until 2026-11-01; they are now rewritten to digits
  by the dataset repo's rules (perso/ultimed_scripts.py) and this test only
  serves the opt-in --drop-number-words of the two manifest scripts.

Written with Claude Code.
"""

import re
import unicodedata

# Applied AFTER NFKC, which already turns nbsp / U+202F / thin spaces into a
# plain space, the ellipsis into "...", superscript/subscript digits into
# digits, U+2011 into U+2010 and "½" into "1⁄2" (fraction slash). The table
# still lists the pre-NFKC forms so it reads as the full fold set.
# Mirrors the folds of the dataset repo (UltiMed-ASR-FR-v1-scripts,
# utils/_pipeline_shared.py: _TOKENIZABLE_FOLDS and DEFINITION_SUBSTITUTIONS),
# which cannot be imported across repos. Dashes become "-" here rather than the
# dataset repo's ", ": rehearsal sentences are not re-prompted, so the fold stays literal.
_REMOVED = "\"“”„«»()[]•­"
_LABEL_CHARSET_FOLDS = str.maketrans({
    "’": "'", "‘": "'",  # curly single quotes -> apostrophe
    **{ch: None for ch in _REMOVED},  # double quotes, guillemets, brackets, bullet, soft hyphen
    "–": "-", "\u2014": "-",  # en / em dash
    "‐": "-", "‑": "-", "−": "-",  # hyphen, nb hyphen, minus
    "…": "...",  # ellipsis
    ";": ",",
    "½": "1/2", "¾": "3/4", "⁄": "/",  # fractions (NFKC: 1⁄2)
    "²": "2", "³": "3", "₂": "2", "₄": "4",
    "×": "x",
})
# A removed closing quote or bracket must not leave its space before "." or ","
# (French "dit « anthrax »." would become "dit anthrax ."), and a French " ;"
# must not become " ,". ":!?" keep their space, the French CV convention. An
# ellipsis is excluded so an opening quote does not glue to it ("dit « ...").
_SPACE_BEFORE_PUNCT_RE = re.compile(f"\\s*[{re.escape(_REMOVED)}]+\\s*(?=[.,;](?!\\.))|\\s+(?=;)")


def normalize_label_charset(text: str) -> str:
    """NFKC, fold the typographic characters the tokenizer lacks, then collapse
    whitespace (newlines, tabs and unicode spaces included). Accents, ß and
    every non-Latin letter are left as they are.

    >>> normalize_label_charset("dit « x » ; puis (y)…")
    'dit x, puis y...'
    """
    text = _SPACE_BEFORE_PUNCT_RE.sub("", unicodedata.normalize("NFKC", text))
    return " ".join(text.translate(_LABEL_CHARSET_FOLDS).split())


_FR_NUMBER_WORDS = (
    "zéro deux trois quatre cinq six sept huit neuf dix onze douze treize quatorze "
    "quinze seize vingts? trente quarante cinquante soixante septante huitante "
    "octante nonante cents? mille millions? milliards?"
).split()
# Hyphenated composites (dix-sept, quatre-vingt-dix) contain a listed word
# between non-word boundaries, so they match without their own entries.
# "un" / "une" are articles far more often than numbers and stay allowed.
_FR_NUMBER_WORD_RE = re.compile(r"\b(?:" + "|".join(_FR_NUMBER_WORDS) + r")\b", re.IGNORECASE)


def has_french_number_word(text: str) -> bool:
    """True when text spells a French number other than un/une.

    >>> has_french_number_word("un comprimé"), has_french_number_word("Quatre-vingt-dix")
    (False, True)
    """
    return _FR_NUMBER_WORD_RE.search(text) is not None

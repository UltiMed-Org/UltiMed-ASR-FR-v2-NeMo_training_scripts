# Written by Claude Code.
import importlib.util
import json
import sys
from pathlib import Path

_PERSO = Path(__file__).resolve().parents[2] / "perso"
sys.path.insert(0, str(_PERSO))  # the perso scripts import their sibling label_charset

import label_charset as lc  # noqa: E402
import renormalize_manifests as renorm  # noqa: E402


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _PERSO / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prep = _load("prepare_commonvoice_rehearsal")

# Characters missing from the parakeet vocab: a label with one tokenizes to <unk>.
_OUT_OF_VOCAB = "’‘“”„«»–\u2014…‐‑−\"()[];•­  ½¾²³×₂₄"


def test_prepare_uses_the_shared_fold():
    assert prep.normalize_label_charset is lc.normalize_label_charset
    assert prep.NUMBER_WORD_FILTERS is renorm.NUMBER_WORD_FILTERS
    assert prep.NUMBER_WORD_CONVERTERS is renorm.NUMBER_WORD_CONVERTERS


def test_normalize_label_charset_folds_out_of_vocab_typography():
    n = lc.normalize_label_charset
    assert n("quelle qu’en soit la cause") == "quelle qu'en soit la cause"
    assert n("il dit : « oui ».") == "il dit : oui."
    assert n("dit « x », puis « y » ?") == "dit x, puis y ?"
    assert n("et puis…") == "et puis..."
    assert n("„Straße“ – “ok” \u2014 fin") == "Straße - ok - fin"
    assert n("été à l'hôpital") == "été à l'hôpital"


def test_normalize_label_charset_extended_folds():
    n = lc.normalize_label_charset
    assert n('he said "yes".') == "he said yes."
    assert n("a ; b; c") == "a, b, c"
    assert n("la [note] (voir ci‑dessous) • fin") == "la note voir ci-dessous fin"
    assert n("x (y).") == "x y."
    assert n("5 ½ h, ¾ l, 10 m² × 2, 3 cm³") == "5 1/2 h, 3/4 l, 10 m2 x 2, 3 cm3"
    assert n("H₂SO₄ −4") == "H2SO4 -4"
    assert n("10 km à co­opé") == "10 km à coopé"
    assert n("ligne\nsuivante\tici  ") == "ligne suivante ici"
    assert n("été") == "été"  # NFKC composes combining accents
    assert not any(c in n(_OUT_OF_VOCAB) for c in _OUT_OF_VOCAB)


def test_normalize_label_charset_leaves_unfoldable_characters():
    # No faithful fold: left in place so the tokenizer check drops the label.
    for text in ("Erdoğan", "Chişinău", "東京", "αβγ", "ёлка", "A & B", "x = 2", "1 + 1", "37 °C", "± 2"):
        assert lc.normalize_label_charset(text) == text


def test_french_number_words():
    f = lc.has_french_number_word
    assert not f("un comprimé")
    assert not f("Une gélule par jour, 25 mg")
    assert not f("septembre, pourcentage, centre, milliardaire, onzième")
    assert f("vingt comprimés")
    assert f("deux cent quatre-vingt-dix")
    assert f("Dix-sept ans")
    assert f("trois cents euros et deux millions")
    assert f("zéro")


class _FakeTokenizer:
    """SentencePiece-like: one id per known character, <unk> (0) for the rest."""

    unk_id = 0

    def __init__(self, charset):
        self._ids = {ch: i + 1 for i, ch in enumerate(charset)}

    def text_to_ids(self, text):
        return [self._ids.get(ch, 0) for ch in text]


def _oov(tok=_FakeTokenizer("abcdefghijklmnopqrstuvwxyzéèàô '.,-")):
    from nemo.collections.asr.parts.utils.label_vocab_check import label_oov_chars

    cache = {}
    return lambda text: label_oov_chars(text, tok, tok.unk_id, cache)


def _write(path, texts):
    rows = [{"audio_filepath": f"{i}.flac", "duration": 1.0, "text": t} for i, t in enumerate(texts)]
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return rows


def _texts(path):
    return [json.loads(line)["text"] for line in path.read_text(encoding="utf-8").splitlines()]


def test_renormalize_manifest_rewrites_text_in_place(tmp_path):
    path = tmp_path / "m.jsonl"
    rows = _write(path, ["l’os", "clean"])
    r = renorm.renormalize_manifest(path)
    assert (r.total, r.changed, r.oov) == (2, 1, [])
    out = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert out == [dict(rows[0], text="l'os"), rows[1]]
    assert list(tmp_path.iterdir()) == [path]


def test_renormalize_manifest_keeps_unchanged_lines_and_escaping(tmp_path):
    path = tmp_path / "m.json"
    clean = '{"text": "\\u00e9t\\u00e9",  "duration": 1}\n'
    path.write_text(clean + json.dumps({"text": "«été»"}) + "\n", encoding="utf-8")
    assert renorm.renormalize_manifest(path).changed == 1
    assert path.read_text(encoding="utf-8") == clean + '{"text": "\\u00e9t\\u00e9"}\n'


def test_renormalize_manifest_drop_oov(tmp_path):
    path = tmp_path / "m.jsonl"
    _write(path, ["a « b »", "erdoğan", "a & b", "fine"])
    r = renorm.renormalize_manifest(path, _oov(), drop_oov=True)
    assert r.oov == [("erdoğan", "ğ"), ("a & b", "&")]
    assert _texts(path) == ["a b", "fine"]


def test_renormalize_manifest_reports_oov_without_dropping(tmp_path):
    path = tmp_path / "m.jsonl"
    _write(path, ["erdoğan", "fine"])
    r = renorm.renormalize_manifest(path, _oov(), drop_oov=False)
    assert r.oov == [("erdoğan", "ğ")]
    assert _texts(path) == ["erdoğan", "fine"]


def test_renormalize_manifest_drops_french_number_words(tmp_path):
    path = tmp_path / "m.jsonl"
    _write(path, ["un comprimé", "vingt comprimés", "deux cent quatre-vingt-dix", "25 mg"])
    r = renorm.renormalize_manifest(path, drop_if=renorm.NUMBER_WORD_FILTERS["fr"])
    assert r.number_word_dropped == ["vingt comprimés", "deux cent quatre-vingt-dix"]
    assert _texts(path) == ["un comprimé", "25 mg"]


# --- UltiMed number rules, imported from the sibling dataset repo -------------------

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import soundfile as sf  # noqa: E402

import ultimed_scripts  # noqa: E402

try:
    ultimed_scripts.scripts_dir()
    _NO_SCRIPTS = None
except ultimed_scripts.UltimedScriptsNotFound as e:
    _NO_SCRIPTS = str(e)
needs_scripts = pytest.mark.skipif(_NO_SCRIPTS is not None, reason=f"dataset repo absent: {_NO_SCRIPTS}")

# (CV-style label after the charset fold, label after the UltiMed rules)
_CONVERSIONS = [
    ("Il est né en mille neuf cent quatre-vingt-dix.", "Il est né en 1990."),
    ("Il y avait vingt et un élèves.", "Il y avait 21 élèves."),
    ("Elle a quatre-vingt-dix-neuf pour cent de chances.", "Elle a 99 % de chances."),
    ("Ballon de trois virgule cinq.", "Ballon de 3,5."),
]
# Left spelled by the UltiMed rules: ordinals, a lone number word, "un" as an article.
_UNTOUCHED = [
    "C'est le vingt-et-unième siècle.",
    "Deux.",
    "Un homme est venu.",
    "Des milliers de personnes.",
]


@needs_scripts
def test_general_french_normalizer_converts_numbers_and_keeps_ordinals():
    convert = ultimed_scripts.general_french_label_normalizer()
    for before, after in _CONVERSIONS:
        assert convert(before) == after
    for text in _UNTOUCHED:
        assert convert(text) == text


# Medical-only UltiMed rules that misfire on encyclopedic French (seen on CV-fr 2026-11-01).
_MEDICAL_MISFIRES = [
    "Le quartier reçoit de nouveaux colons.",
    "Un village néo-zélandais.",
    "On le dit Sandoz.",
    "Elodie a une sœur.",
]


@needs_scripts
def test_general_french_normalizer_skips_medical_rules():
    full = ultimed_scripts.ultimed_label_normalizer()
    convert = ultimed_scripts.general_french_label_normalizer()
    for text in _MEDICAL_MISFIRES:
        assert convert(text) == text
        assert full(text) != text  # the full UltiMed chain does rewrite them
    assert renorm.NUMBER_WORD_CONVERTERS["fr"] is ultimed_scripts.general_french_label_normalizer


def test_scripts_dir_fails_clearly_on_a_missing_path(tmp_path, monkeypatch):
    monkeypatch.setenv(ultimed_scripts.ENV_VAR, str(tmp_path / "nope"))
    with pytest.raises(ultimed_scripts.UltimedScriptsNotFound, match="ULTIMED_SCRIPTS_DIR"):
        ultimed_scripts.ultimed_label_normalizer()


def test_scripts_dir_fails_clearly_on_a_wrong_checkout(tmp_path, monkeypatch):
    monkeypatch.setenv(ultimed_scripts.ENV_VAR, str(tmp_path))
    with pytest.raises(ultimed_scripts.UltimedScriptsNotFound, match="05_normalize_text.py"):
        ultimed_scripts.scripts_dir()


def test_scripts_dir_default_is_the_sibling_checkout(monkeypatch):
    monkeypatch.delenv(ultimed_scripts.ENV_VAR, raising=False)
    assert ultimed_scripts.DEFAULT_DIR == _PERSO.parent.parent / "UltiMed-ASR-FR-v1-scripts"


@needs_scripts
def test_renormalize_manifest_converts_french_number_words(tmp_path):
    path = tmp_path / "m.jsonl"
    _write(path, [before for before, _ in _CONVERSIONS] + _UNTOUCHED)
    convert = renorm.NUMBER_WORD_CONVERTERS["fr"]()
    r = renorm.renormalize_manifest(path, convert=convert)
    assert r.converted == _CONVERSIONS
    assert r.number_word_dropped == []
    assert _texts(path) == [after for _, after in _CONVERSIONS] + _UNTOUCHED


def _flac_selection(flac_dir, texts):
    flac_dir.mkdir(parents=True)
    selected = {}
    for i, text in enumerate(texts):
        sf.write(str(flac_dir / f"c{i}.flac"), np.zeros(16000 * 2, dtype=np.int16), 16000)
        selected[f"c{i}.mp3"] = (2.0, text)
    return selected


@needs_scripts
def test_build_manifest_converts_french_labels(tmp_path):
    texts = ["Il y avait vingt et un élèves.", "C'est le vingt-et-unième siècle.", "« Deux », dit-il."]
    selected = _flac_selection(tmp_path / "clips_16k" / "fr", texts)
    rows, _ = prep.build_manifest(tmp_path, "fr", selected, tmp_path / "clips_16k" / "fr", suffix=".converted")
    assert rows == 3
    assert not (tmp_path / "cv_fr_train.jsonl").exists()
    assert _texts(tmp_path / "cv_fr_train.converted.jsonl") == [
        "Il y avait 21 élèves.", "C'est le vingt-et-unième siècle.", "Deux, dit-il."]


def test_build_manifest_can_still_drop_french_number_words(tmp_path):
    texts = ["Il y avait vingt et un élèves.", "Un homme est venu."]
    selected = _flac_selection(tmp_path / "clips_16k" / "fr", texts)
    rows, _ = prep.build_manifest(tmp_path, "fr", selected, tmp_path / "clips_16k" / "fr", drop_number_words=True)
    assert rows == 1
    assert _texts(tmp_path / "cv_fr_train.jsonl") == ["Un homme est venu."]

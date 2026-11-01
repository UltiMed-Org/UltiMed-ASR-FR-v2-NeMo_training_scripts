# Written by Claude Code.
import importlib.util
import json
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "prepare_commonvoice_rehearsal",
    Path(__file__).resolve().parents[2] / "perso" / "prepare_commonvoice_rehearsal.py",
)
prep = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(prep)

# Characters missing from the parakeet vocab: a label with one tokenizes to <unk>.
_OUT_OF_VOCAB = "’‘“”„«»–—…‐‑−"


def test_normalize_label_charset_folds_out_of_vocab_typography():
    assert prep.normalize_label_charset("quelle qu’en soit la cause") == "quelle qu'en soit la cause"
    assert prep.normalize_label_charset("il dit : « oui ».") == "il dit : oui."
    assert prep.normalize_label_charset("dit « x », puis « y » ?") == "dit x, puis y ?"
    assert prep.normalize_label_charset("et puis…") == "et puis..."
    assert prep.normalize_label_charset("„Straße“ – “ok” — fin") == "Straße - ok - fin"
    assert prep.normalize_label_charset("été à l'hôpital") == "été à l'hôpital"


def test_renormalize_manifest_rewrites_text_in_place(tmp_path):
    path = tmp_path / "m.jsonl"
    rows = [{"audio_filepath": "a.flac", "duration": 1.0, "text": "l’os"},
            {"audio_filepath": "b.flac", "duration": 2.0, "text": "clean"}]
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    assert prep.renormalize_manifest(path) == (1, 2)
    out = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert out == [dict(rows[0], text="l'os"), rows[1]]
    assert not any(c in r["text"] for r in out for c in _OUT_OF_VOCAB)
    assert list(tmp_path.iterdir()) == [path]


def test_renormalize_manifest_keeps_unchanged_lines_and_escaping(tmp_path):
    path = tmp_path / "m.json"
    clean = '{"text": "\\u00e9t\\u00e9",  "duration": 1}\n'
    path.write_text(clean + json.dumps({"text": "«été»"}) + "\n", encoding="utf-8")
    assert prep.renormalize_manifest(path) == (1, 2)
    assert path.read_text(encoding="utf-8") == clean + '{"text": "\\u00e9t\\u00e9"}\n'

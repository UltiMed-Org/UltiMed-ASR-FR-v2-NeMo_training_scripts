# Written by Claude Code.
import json

import pytest
from omegaconf import OmegaConf

from nemo.collections.asr.parts.utils import label_vocab_check as lvc

_UNK = 0


class _FakeTokenizer:
    """SentencePiece-like: one id per known character, <unk> (0) for the rest."""

    def __init__(self, charset):
        self._ids = {ch: i + 1 for i, ch in enumerate(charset)}

    @property
    def unk_id(self):
        return _UNK

    def text_to_ids(self, text):
        return [self._ids.get(ch, _UNK) for ch in text]


_TOK = _FakeTokenizer("abcdefghijklmnopqrstuvwxyzéèàô '.,")


def _manifest(path, texts):
    path.write_text(
        "".join(json.dumps({"audio_filepath": "x.flac", "duration": 1.0, "text": t}, ensure_ascii=False) + "\n" for t in texts),
        encoding="utf-8",
    )
    return str(path)


def _cfg(train, val, test=None):
    return OmegaConf.create({"model": {
        "train_ds": {"manifest_filepath": train},
        "validation_ds": {"ds_item": [{"name": f"v{i}", "manifest_filepath": m} for i, m in enumerate(val)]},
        "test_ds": {"manifest_filepath": test},
    }})


def test_clean_manifests_pass(tmp_path, monkeypatch):
    monkeypatch.delenv(lvc.SKIP_ENV_VAR, raising=False)
    clean = _manifest(tmp_path / "clean.jsonl", ["le patient va bien.", "l'été, à l'hôpital"])
    lvc.check_labels_in_vocab(_TOK, _cfg([clean], [clean]))


def test_collects_every_oov_char_across_manifests_then_raises_once(tmp_path, monkeypatch):
    monkeypatch.delenv(lvc.SKIP_ENV_VAR, raising=False)
    clean = _manifest(tmp_path / "clean.jsonl", ["ok"])
    apos = _manifest(tmp_path / "train.jsonl", ["qu’en", "fine", "l’os"])
    guil = _manifest(tmp_path / "val.jsonl", ["il dit « oui »"])
    with pytest.raises(lvc.LabelVocabError) as err:
        lvc.check_labels_in_vocab(_TOK, _cfg([clean, apos], [guil]))
    msg = str(err.value)
    assert "U+2019" in msg and "U+00AB" in msg
    assert "2 / 3 labels" in msg and "1 / 1 labels" in msg
    assert "2 manifest(s)" in msg and str(tmp_path / "clean.jsonl") + ":" not in msg


def test_env_var_skips(tmp_path, monkeypatch):
    monkeypatch.setenv(lvc.SKIP_ENV_VAR, "1")
    bad = _manifest(tmp_path / "bad.jsonl", ["«"])
    lvc.check_labels_in_vocab(_TOK, _cfg([bad], []))


def test_missing_manifest_raises(tmp_path, monkeypatch):
    monkeypatch.delenv(lvc.SKIP_ENV_VAR, raising=False)
    with pytest.raises(FileNotFoundError):
        lvc.check_labels_in_vocab(_TOK, _cfg([str(tmp_path / "nope.jsonl")], []))

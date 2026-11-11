# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The cached fine-tune script must keep its temp files off /tmp.

Lightning writes a whole checkpoint to ``tempfile.mkstemp()`` before moving it into
place; on a tmpfs /tmp that killed run 3.3.seed7-15k with ENOSPC on 2026-11-11.
Written by Claude Code.
"""
import os

from examples.asr.speech_to_text_finetune_cached import _pin_tmpdir_to_exp_drive


def test_tmpdir_goes_to_the_experiment_drive(tmp_path, monkeypatch):
    monkeypatch.delenv("TMPDIR", raising=False)
    monkeypatch.setenv("NEMO_EXP_DIR", str(tmp_path))
    _pin_tmpdir_to_exp_drive()
    expected = tmp_path / "tmp_ignore_backups"
    assert os.environ["TMPDIR"] == str(expected)
    assert expected.is_dir()


def test_explicit_tmpdir_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("TMPDIR", "/somewhere/else")
    monkeypatch.setenv("NEMO_EXP_DIR", str(tmp_path))
    _pin_tmpdir_to_exp_drive()
    assert os.environ["TMPDIR"] == "/somewhere/else"
    assert not (tmp_path / "tmp_ignore_backups").exists()


def test_no_exp_dir_is_a_noop(monkeypatch):
    monkeypatch.delenv("TMPDIR", raising=False)
    monkeypatch.delenv("NEMO_EXP_DIR", raising=False)
    _pin_tmpdir_to_exp_drive()
    assert "TMPDIR" not in os.environ

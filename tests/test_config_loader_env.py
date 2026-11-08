#!/usr/bin/env python3

# Copyright 2026 EMBL - European Bioinformatics Institute
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

from common_core.config_loader import _build_env_map


def test_env_file_is_read(tmp_path):
    env = tmp_path / ".env"
    env.write_text("export FOO_BAR='baz'\n# comment\n")
    assert _build_env_map(str(env), case_sensitive=True)["FOO_BAR"] == "baz"


def test_unreadable_env_file_warns_and_falls_back(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("FROM_PROCESS", "1")
    env = tmp_path / ".env"
    env.write_bytes(b"KEY=\xff\xfe\n")  # not UTF-8
    with caplog.at_level(logging.WARNING, logger="common_core.config_loader"):
        out = _build_env_map(str(env), case_sensitive=True)
    assert out["FROM_PROCESS"] == "1"
    assert "KEY" not in out
    assert any("could not read" in r.getMessage() for r in caplog.records)

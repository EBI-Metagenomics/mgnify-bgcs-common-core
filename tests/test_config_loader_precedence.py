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

"""load_settings precedence: CLI > environment > .env > YAML > model default."""

import pytest
from pydantic import BaseModel
from pydantic_settings import BaseSettings

from common_core.config_loader import LoaderOptions, load_settings


class _Inner(BaseModel):
    depth: int = 0


class _Settings(BaseSettings):
    a: str = "default"
    b: str = "default"
    c: str = "default"
    d: str = "default"
    e: str = "default"
    inner: _Inner = _Inner()


@pytest.fixture
def sources(tmp_path, monkeypatch):
    yaml = tmp_path / "job.yaml"
    yaml.write_text("a: yaml\nb: yaml\nc: yaml\nd: yaml\ninner:\n  depth: 1\n")
    dotenv = tmp_path / ".env"
    dotenv.write_text("T_A=dotenv\nT_B=dotenv\nT_C=dotenv\n")
    monkeypatch.setenv("T_A", "env")
    monkeypatch.setenv("T_B", "env")
    monkeypatch.setenv("T_INNER__DEPTH", "2")
    return yaml, dotenv


def test_each_source_overrides_the_next(sources):
    yaml, dotenv = sources
    cfg = load_settings(
        _Settings,
        yaml_path=yaml,
        cli_overrides={"a": "cli"},
        options=LoaderOptions(env_prefix="T_", env_file=str(dotenv)),
    )
    assert cfg.a == "cli"  # CLI beats env
    assert cfg.b == "env"  # env beats .env
    assert cfg.c == "dotenv"  # .env beats YAML
    assert cfg.d == "yaml"  # YAML beats the default
    assert cfg.e == "default"
    assert cfg.inner.depth == 2  # nested env key (T_INNER__DEPTH) beats YAML


def test_yaml_interpolates_env(tmp_path, monkeypatch):
    yaml = tmp_path / "job.yaml"
    yaml.write_text("a: ${T_ROOT}/x\n")
    monkeypatch.setenv("T_ROOT", "/data")
    cfg = load_settings(
        _Settings, yaml_path=yaml, options=LoaderOptions(env_prefix="T_", env_file=None)
    )
    assert cfg.a == "/data/x"


def test_default_does_not_read_a_dotenv_in_the_working_directory(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("T_A=dotenv\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("T_A", raising=False)
    cfg = load_settings(_Settings, options=LoaderOptions(env_prefix="T_"))
    assert cfg.a == "default"

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

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource

T = TypeVar("T", bound=BaseSettings)


@dataclass(frozen=True)
class LoaderOptions:
    """
    Generic options. Each script can pass its own env prefix and .env path, etc.
    """

    env_prefix: str = ""
    env_nested_delimiter: str = "__"
    env_file: str | None = ".env"
    case_sensitive: bool = False


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping/dict: {path}")
    return data


_ENV_PATTERN = re.compile(r"\$\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)\}")


def _build_env_map(env_file: str | None, case_sensitive: bool) -> dict[str, str]:
    env: dict[str, str] = dict(os.environ)

    if env_file:
        env_path = Path(env_file)
        if env_path.exists():
            try:
                with env_path.open("r", encoding="utf-8") as f:
                    for raw in f:
                        line = raw.strip()
                        if not line or line.startswith("#"):
                            continue
                        if line.startswith("export "):
                            line = line[len("export ") :]
                        if "=" not in line:
                            continue
                        key, value = line.split("=", 1)
                        key = key.strip()
                        value = value.strip().strip("'\"")
                        env[key] = value
            except Exception:
                pass

    if case_sensitive:
        return env

    return {k.lower(): v for k, v in env.items()}


def _expand_env_in_str(
    value: str, env_map: Mapping[str, str], *, case_sensitive: bool
) -> str:
    def repl(match: re.Match[str]) -> str:
        name = match.group("name")
        key = name if case_sensitive else name.lower()
        if key not in env_map:
            raise KeyError(
                f"Environment variable '{name}' not found for YAML interpolation"
            )
        return env_map[key]

    return _ENV_PATTERN.sub(repl, value)


def _expand_env_in_data(
    obj: Any, env_map: Mapping[str, str], *, case_sensitive: bool
) -> Any:
    if isinstance(obj, dict):
        return {
            k: _expand_env_in_data(v, env_map, case_sensitive=case_sensitive)
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [
            _expand_env_in_data(v, env_map, case_sensitive=case_sensitive) for v in obj
        ]
    if isinstance(obj, str):
        return _expand_env_in_str(obj, env_map, case_sensitive=case_sensitive)
    return obj


def load_settings(
    model: type[T],
    *,
    yaml_path: str | Path | None = None,
    cli_overrides: Mapping[str, Any] | None = None,
    options: LoaderOptions = LoaderOptions(),
) -> T:
    """
    Generic loader for any BaseSettings subclass.

    Priority (highest wins):
      - cli_overrides (explicit dict)
      - environment variables
      - yaml file (if provided)
      - model defaults

    yaml can contain non-secret config; env provides secrets; cli can override both.
    """
    yaml_data: dict[str, Any] = {}
    if yaml_path is not None:
        yaml_data = _read_yaml(Path(yaml_path))

    # Interpolate ${ENV_VAR} occurrences in YAML using OS env and optional .env
    env_map = _build_env_map(options.env_file, options.case_sensitive)
    yaml_data = _expand_env_in_data(
        yaml_data, env_map, case_sensitive=options.case_sensitive
    )

    # Configure per-call settings behavior without requiring each model to bake it in.
    # Dynamic subclass with the desired model_config.
    dynamic = type(
        f"{model.__name__}__Loaded",
        (model,),
        {
            "model_config": {
                **getattr(model, "model_config", {}),
                "env_prefix": options.env_prefix,
                "env_nested_delimiter": options.env_nested_delimiter,
                "env_file": options.env_file,
                "case_sensitive": options.case_sensitive,
                "extra": "ignore",
            }
        },
    )

    cli_data = dict(cli_overrides or {})

    # Instantiate using pydantic-settings’ normal sources, but with YAML injected
    # at the right position (below env, above defaults).
    class _YamlSource(PydanticBaseSettingsSource):
        def get_field_value(self, field, field_name: str) -> tuple[Any, str, bool]:
            return None, field_name, False  # unused

        def __call__(self) -> dict[str, Any]:
            return dict(yaml_data)

    class _CliSource(PydanticBaseSettingsSource):
        def get_field_value(self, field, field_name: str) -> tuple[Any, str, bool]:
            return None, field_name, False  # unused

        def __call__(self) -> dict[str, Any]:
            return dict(cli_data)

    # Wire sources in desired order:
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        return (
            _CliSource(settings_cls),
            env_settings,  # OS env
            dotenv_settings,  # .env (treated like env)
            _YamlSource(settings_cls),
            init_settings,  # keep last; typically empty
            file_secret_settings,
        )

    dynamic.settings_customise_sources = classmethod(settings_customise_sources)

    return dynamic()  # type: ignore[return-value]

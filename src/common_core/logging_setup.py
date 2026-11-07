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

import json
import logging
from logging.config import dictConfig
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field
from typing_extensions import Literal


class LoggingConfig(BaseModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    json_: bool = Field(False, alias="json")
    file: Optional[str] = None
    third_party_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = (
        "WARNING"
    )

    class Config:
        validate_by_name = True


class JsonLineFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = {
            "ts": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            base["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(base, ensure_ascii=False)


def setup_logging(cfg: LoggingConfig) -> None:
    handlers: Dict[str, Any] = {
        "console": {
            "class": "logging.StreamHandler",
            "level": cfg.level,
            "formatter": "json" if cfg.json_ else "standard",
            "stream": "ext://sys.stdout",
        }
    }
    if cfg.file:
        handlers["file"] = {
            "class": "logging.FileHandler",
            "level": cfg.level,
            "formatter": "json" if cfg.json_ else "standard",
            "filename": cfg.file,
            "encoding": "utf-8",
        }

    dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
                    "format": "%(asctime)s %(levelname)s %(name)s - %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                },
                "json": {"()": "common_core.logging_setup.JsonLineFormatter"},
            },
            "handlers": handlers,
            "root": {"level": cfg.level, "handlers": list(handlers.keys())},
            "loggers": {
                "urllib3": {"level": cfg.third_party_level},
                "botocore": {"level": cfg.third_party_level},
            },
        }
    )

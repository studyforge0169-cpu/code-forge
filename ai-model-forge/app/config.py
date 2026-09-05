"""Central runtime configuration: paths, environment, logging.

Kept deliberately small: the forge is a modular monolith, and anything
project-wide (storage location, log level, API prefix, version) lives here.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

APP_NAME = "ai-model-forge"
APP_VERSION = "0.1.0"
API_PREFIX = "/api/v1"

# Builder schema version: bump only when the on-disk manifest format changes.
MANIFEST_SCHEMA_VERSION = 1


def forge_root() -> Path:
    """Root directory for all persisted forge data.

    Overridable with the FORGE_ROOT environment variable (used by tests).
    Kept outside the source tree so code and user data never mix.
    """
    override = os.environ.get("FORGE_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return (Path.home() / "ai-model-forge-data").expanduser().resolve()


def _log_level() -> int:
    return getattr(logging, os.environ.get("FORGE_LOG_LEVEL", "INFO").upper(), logging.INFO)


def get_logger(name: str) -> logging.Logger:
    """Module logger with a single global configuration point."""
    return logging.getLogger(f"forge.{name}")


def init_logging() -> None:
    logging.basicConfig(
        level=_log_level(),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

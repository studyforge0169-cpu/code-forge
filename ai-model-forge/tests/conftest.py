"""Shared fixtures. Sets a throwaway FORGE_ROOT *before* any app import."""
from __future__ import annotations

import os
import pathlib
import tempfile

import pytest

_TEST_ROOT = pathlib.Path(tempfile.mkdtemp(prefix="forge-tests-"))
os.environ["FORGE_ROOT"] = str(_TEST_ROOT)
os.environ["FORGE_LOG_LEVEL"] = "WARNING"

import app.engine as engine_module  # noqa: E402
from app.engine import ModelForge  # noqa: E402
from app.schemas import TransformerConfig  # noqa: E402


@pytest.fixture()
def tiny_config() -> TransformerConfig:
    return TransformerConfig.tiny()


@pytest.fixture()
def forge(tmp_path) -> ModelForge:
    return ModelForge(root=tmp_path)


@pytest.fixture()
def api_client():
    """FastAPI TestClient bound to the session-scoped temp FORGE_ROOT."""
    from fastapi.testclient import TestClient

    engine_module._forge = None  # reset the singleton so it binds FORGE_ROOT
    from app.api import app

    with TestClient(app) as client:
        yield client
    engine_module._forge = None

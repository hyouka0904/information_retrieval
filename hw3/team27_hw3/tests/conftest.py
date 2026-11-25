"""Shared fixtures. Task 0 stub written on Track A's branch; owner 鄧存正 (Track B)."""
import os
import sys
import urllib.request

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from rag.config import load_config  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return load_config()


def _ollama_up(host: str) -> bool:
    try:
        urllib.request.urlopen(f"{host}/api/tags", timeout=2)
        return True
    except Exception:
        return False


def pytest_collection_modifyitems(config, items):
    c = load_config()
    have_index = os.path.exists(os.path.join(c["index_dir"], "meta.json"))
    have_llm = any("llm" in it.keywords for it in items) and _ollama_up(c["llm"]["host"])
    for it in items:
        if "index" in it.keywords and not have_index:
            it.add_marker(pytest.mark.skip(reason="no rebuilt index (output/meta.json missing)"))
        if "llm" in it.keywords and not have_llm:
            it.add_marker(pytest.mark.skip(reason="Ollama is not running"))

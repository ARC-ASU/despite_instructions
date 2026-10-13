from pathlib import Path
import json
import pytest
from covert_agents.analysis import reproduce

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def results():
    return reproduce(ROOT)


@pytest.fixture
def expected():
    return lambda name: json.loads((ROOT / "tests/expected" / f"{name}.json").read_text())

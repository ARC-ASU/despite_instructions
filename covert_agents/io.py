"""Deterministic JSON I/O shared by runners, analysis, and evidence checks."""

from __future__ import annotations
import gzip
import hashlib
import json
from pathlib import Path


def encoded(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True)


def fingerprint(value):
    return hashlib.sha256(encoded(value).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def read_rows(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def parsed(reply):
    if reply is None or reply.get("status") != "completed" or reply.get("refusal"):
        return None
    try:
        value = json.loads(reply.get("text", ""))
    except (ValueError, TypeError):
        return None
    return value if isinstance(value, dict) else None

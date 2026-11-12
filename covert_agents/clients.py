"""Model boundary: experiments know roles and JSON schemas, not API credentials.

RecordedClient checks exact prompt hashes before returning archived replies.
CallableClient adapts any provider SDK without coupling experiments to it.
"""

from __future__ import annotations
import importlib
import json
from pathlib import Path
from typing import Callable, Protocol
from .io import fingerprint, write_json


class ModelError(RuntimeError):
    """A model request failed; a caller must retain or report the failed trial."""


class Client(Protocol):
    context: dict

    def ask(self, role: str, instructions: str, content: str, schema: dict) -> dict:
        """Return text, status ('completed' on success), and refusal."""
        ...


class RecordedClient:
    """Consume an archived call sequence, checking every information boundary."""

    def __init__(self, records):
        self.records = records
        self.position = 0
        self.context = {}

    def ask(self, role, instructions, content, schema):
        if self.position >= len(self.records):
            raise ValueError("Replay made an unexpected extra model call")
        record = self.records[self.position]
        actual = fingerprint(
            dict(
                role=role,
                instructions=instructions,
                content=content,
                schema=schema,
                context=self.context,
            )
        )
        if actual != record["prompt_sha256"]:
            raise ValueError(
                f"Prompt mismatch at call {self.position}, role={role}, context={self.context}"
            )
        self.position += 1
        return dict(record["reply"])

    def assert_complete(self):
        if self.position != len(self.records):
            raise ValueError(f"Replay used {self.position}/{len(self.records)} calls")


class CallableClient:
    """Wrap a provider function with a hard call limit and credential-free logging.

    The callable receives role, instructions, content, schema and returns a reply
    dict. It should raise on transport failures. This adapter does not retry or
    record provider request headers. A call limit is not a dollar budget.
    """

    def __init__(self, generate: Callable, *, max_calls: int, emit: Callable = lambda row: None):
        if max_calls <= 0:
            raise ValueError("max_calls must be positive")
        self.generate, self.max_calls, self.emit = generate, max_calls, emit
        self.calls = 0
        self.context = {}

    def ask(self, role, instructions, content, schema):
        if self.calls >= self.max_calls:
            raise RuntimeError("Model call limit reached; run is incomplete")
        self.calls += 1
        raw = self.generate(role, instructions, content, schema)
        reply = {key: raw[key] for key in ("text", "status", "refusal") if key in raw}
        material = dict(
            role=role,
            instructions=instructions,
            content=content,
            schema=schema,
            context=self.context,
        )
        self.emit(dict(role=role, prompt_sha256=fingerprint(material), reply=reply))
        return reply


def load_provider(spec):
    """Import a provider function named as ``module:function``."""
    module, function = spec.rsplit(":", 1)
    return getattr(importlib.import_module(module), function)


def record_run(output, generate, max_calls, inputs, work):
    """Run ``work(client, emit)`` against a provider, recording it in a new directory.

    Writes ``inputs`` (``{filename: value}``) and the provider's public settings,
    streams ``calls.jsonl`` and ``trials.jsonl`` as calls complete, and writes
    ``completed.json`` only if ``work`` finishes. Returns the ``CallableClient``.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for name, value in inputs.items():
        write_json(output / name, value)
    if hasattr(generate, "public_settings"):
        write_json(output / "provider.json", generate.public_settings())
    with (output / "calls.jsonl").open("x") as calls, (output / "trials.jsonl").open("x") as trials:

        def writer(handle):
            def emit(row):
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()

            return emit

        client = CallableClient(generate, max_calls=max_calls, emit=writer(calls))
        work(client, writer(trials))
    write_json(output / "completed.json", {"calls": client.calls})
    return client

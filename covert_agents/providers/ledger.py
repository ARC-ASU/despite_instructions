"""Cost ledger and spending guard around a live provider adapter.

Wraps a provider ``generate`` function. Before each call it checks that a
conservative worst-case cost for the call fits under the per-run cap and, if
set, a cumulative ceiling over every run recorded in the same ledger file. After
each call it appends one JSON line with token usage, estimated cost, and the
provider's response identifier. Ledger files can identify your account's
requests, so they are written under ``outputs/`` (git-ignored) by default.

Environment:
  COVERT_LEDGER_EXPERIMENT      required, label stored with every row
  COVERT_LEDGER_RUN_CAP_USD     required, hard cap for this process
  COVERT_LEDGER_PROVIDER        openai (default) or openrouter
  COVERT_LEDGER_CEILING_USD     optional cap on the ledger file's cumulative spend
  COVERT_LEDGER_PATH            optional, default outputs/ledger/<provider>.jsonl
  COVERT_LEDGER_PRICES          optional, default configs/prices.json
  COVERT_LEDGER_CONTEXT         optional free-text condition label
  COVERT_TRANSPORT_RETRIES      optional, resend after connection failures (default 0)
  COVERT_TRANSPORT_WAIT_SECONDS optional pause before such a resend (default 15)

No credential is read or written here; the wrapped adapter reads its own key
from the environment. Costs are estimates from the price table, not invoices.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

CHARS_PER_TOKEN = 3.0  # conservative for English prose with JSON wrapping
DEFAULT_PRICES = Path("configs/prices.json")


def read_prices(path=None):
    return json.loads(Path(path or os.environ.get("COVERT_LEDGER_PRICES", DEFAULT_PRICES)).read_text())


def cost_usd(usage, model, prices):
    """Estimated cost of one call from a Responses-style usage object."""
    table = prices["models"].get(model)
    if table is None:
        raise ValueError(f"No price for model {model!r}; add it to the price table first")
    usage = usage or {}
    input_tokens = int(usage.get("input_tokens", 0))
    cached = int((usage.get("input_tokens_details") or {}).get("cached_tokens", 0))
    output_tokens = int(usage.get("output_tokens", 0))
    multiplier = float(prices.get("cache_write_multiplier", 1.0))
    dollars = (
        max(input_tokens - cached, 0) * table["input"] * multiplier
        + cached * table["cached_input"]
        + output_tokens * table["output"]
    ) / 1_000_000
    return dict(
        input_tokens=input_tokens,
        cached_tokens=cached,
        output_tokens=output_tokens,
        reasoning_tokens=int((usage.get("output_tokens_details") or {}).get("reasoning_tokens", 0)),
        cost_usd=round(dollars, 6),
    )


def worst_case_usd(instructions, content, model, prices, max_output_tokens):
    """Upper bound for the next call: all input uncached, output at the token limit."""
    table = prices["models"].get(model)
    if table is None:
        raise ValueError(f"No price for model {model!r}; add it to the price table first")
    tokens_in = (len(instructions) + len(content)) / CHARS_PER_TOKEN
    return (tokens_in * table["input"] * float(prices.get("cache_write_multiplier", 1.0))
            + max_output_tokens * table["output"]) / 1_000_000


class Ledger:
    def __init__(self, provider, experiment, run_cap_usd, path, prices, ceiling_usd=None,
                 context=None):
        if run_cap_usd <= 0:
            raise ValueError("run cap must be positive")
        if ceiling_usd is not None and ceiling_usd <= 0:
            raise ValueError("ceiling must be positive")
        self.provider, self.experiment, self.context = provider, experiment, context
        self.path, self.prices = Path(path), prices
        self.run_cap, self.ceiling = float(run_cap_usd), ceiling_usd
        self.run_spent = 0.0
        self.lock = threading.Lock()
        # Transport-only retry: a request that never reached the model or whose response was
        # lost is resent after a pause. HTTP errors, refusals and outputs are never retried.
        self.transport_retries = int(os.environ.get("COVERT_TRANSPORT_RETRIES", "0"))
        self.transport_wait = float(os.environ.get("COVERT_TRANSPORT_WAIT_SECONDS", "15"))

    def cumulative(self):
        total = 0.0
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    if row.get("provider") == self.provider:
                        total += float(row.get("cost_usd", 0.0))
        return total

    def check(self, worst):
        if self.run_spent + worst > self.run_cap:
            raise RuntimeError(
                f"Ledger stop: run spent ${self.run_spent:.4f} + worst-case ${worst:.4f} "
                f"exceeds run cap ${self.run_cap:.2f}; run is incomplete"
            )
        if self.ceiling is not None:
            cumulative = self.cumulative()
            if cumulative + worst > self.ceiling:
                raise RuntimeError(
                    f"Ledger stop: cumulative ${cumulative:.4f} + worst-case ${worst:.4f} "
                    f"exceeds ceiling ${self.ceiling:.2f} for {self.provider}; run is incomplete"
                )

    def record(self, row):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    def wrap(self, generate, max_output_tokens, model_for_role):
        def guarded(role, instructions, content, schema):
            model = model_for_role(role)
            worst = worst_case_usd(instructions, content, model, self.prices, max_output_tokens)
            with self.lock:
                self.check(worst)
            started = time.time()
            for attempt in range(1, self.transport_retries + 2):
                try:
                    raw = generate(role, instructions, content, schema)
                    break
                except Exception as exc:
                    retry = "connection failed" in str(exc) and attempt <= self.transport_retries
                    with self.lock:
                        self.record(dict(
                            at=started, provider=self.provider, experiment=self.experiment,
                            context=self.context, role=role, model=model, event="error",
                            error=str(exc)[:200], cost_usd=0.0, attempt=attempt, retried=retry,
                        ))
                    if not retry:
                        raise
                    time.sleep(self.transport_wait)
                    started = time.time()
            accounted = cost_usd(raw.get("usage"), raw.get("model") or model, self.prices)
            with self.lock:
                self.run_spent += accounted["cost_usd"]
                self.record(dict(
                    at=started, seconds=round(time.time() - started, 3), provider=self.provider,
                    experiment=self.experiment, context=self.context, role=role, model=model,
                    returned_model=raw.get("model"), response_id=raw.get("response_id"),
                    status=raw.get("status"), event="call", **accounted,
                ))
            return raw
        return guarded


def summary(path, provider="openai"):
    """Spend by experiment for one provider in a ledger file."""
    by, total = {}, 0.0
    path = Path(path)
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("provider") == provider:
                    cost = float(row.get("cost_usd", 0.0))
                    by[row["experiment"]] = by.get(row["experiment"], 0.0) + cost
                    total += cost
    return dict(provider=provider, by_experiment=by, spent_usd=round(total, 4))


def _configured():
    experiment = os.environ.get("COVERT_LEDGER_EXPERIMENT")
    cap = os.environ.get("COVERT_LEDGER_RUN_CAP_USD")
    if not experiment or not cap:
        raise ValueError("Set COVERT_LEDGER_EXPERIMENT and COVERT_LEDGER_RUN_CAP_USD before a live run")
    provider = os.environ.get("COVERT_LEDGER_PROVIDER", "openai")
    if provider == "openai":
        from . import openai as adapter
    elif provider == "openrouter":
        from . import openrouter as adapter
    else:
        raise ValueError(f"No adapter wired for provider {provider!r}")
    ceiling = os.environ.get("COVERT_LEDGER_CEILING_USD")
    ledger = Ledger(
        provider, experiment, float(cap),
        os.environ.get("COVERT_LEDGER_PATH", f"outputs/ledger/{provider}.jsonl"),
        read_prices(), float(ceiling) if ceiling else None,
        context=os.environ.get("COVERT_LEDGER_CONTEXT"),
    )
    return ledger, adapter, adapter.settings()


_STATE = {}


def _settings_for(ledger, config):
    return dict(config, ledger=dict(
        provider=ledger.provider, experiment=ledger.experiment, run_cap_usd=ledger.run_cap,
        ceiling_usd=ledger.ceiling, transport_retries=ledger.transport_retries,
        transport_wait_seconds=ledger.transport_wait,
    ))


def generate(role, instructions, content, schema):
    """Provider for ``--client covert_agents.providers.ledger:generate``."""
    if "guarded" not in _STATE:
        ledger, adapter, config = _configured()
        _STATE["guarded"] = ledger.wrap(
            adapter.generate, config["max_output_tokens"], lambda r: config["models"][r]
        )
        _STATE["settings"] = _settings_for(ledger, config)
    return _STATE["guarded"](role, instructions, content, schema)


def public_settings():
    """Settings written to provider.json; complete before the first call."""
    if "settings" not in _STATE:
        ledger, _, config = _configured()
        _STATE["settings"] = _settings_for(ledger, config)
    return _STATE["settings"]


generate.public_settings = public_settings

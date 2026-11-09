import json
import pytest
from covert_agents.providers import ledger as L


PRICES = {"cache_write_multiplier": 1.0,
          "models": {"m": {"input": 4.0, "cached_input": 0.4, "output": 20.0}}}


def test_cost_accounts_cached_and_reasoning_tokens():
    usage = {"input_tokens": 1000, "input_tokens_details": {"cached_tokens": 600},
             "output_tokens": 100, "output_tokens_details": {"reasoning_tokens": 40}}
    out = L.cost_usd(usage, "m", PRICES)
    assert out["cached_tokens"] == 600 and out["reasoning_tokens"] == 40
    assert out["cost_usd"] == pytest.approx((400 * 4.0 + 600 * 0.4 + 100 * 20.0) / 1e6)


def test_unknown_model_refused():
    with pytest.raises(ValueError):
        L.cost_usd({}, "nope", PRICES)


def _fake(usage):
    def generate(role, instructions, content, schema):
        return dict(text="{}", status="completed", refusal=None, usage=usage, model="m",
                    response_id="r1")
    return generate


def _ledger(tmp_path, experiment="exp", cap=10.0, ceiling=None):
    return L.Ledger("openai", experiment, cap, tmp_path / "openai.jsonl", PRICES, ceiling)


def _rows(tmp_path):
    return [json.loads(x) for x in (tmp_path / "openai.jsonl").read_text().splitlines()]


def test_records_each_call_and_enforces_run_cap(tmp_path):
    usage = {"input_tokens": 100000, "output_tokens": 1000}  # $0.42 per call
    g = _ledger(tmp_path, cap=0.5).wrap(_fake(usage), 5000, lambda r: "m")
    g("sender", "i", "c", {})
    rows = _rows(tmp_path)
    assert len(rows) == 1 and rows[0]["cost_usd"] == pytest.approx(0.42)
    assert rows[0]["experiment"] == "exp"
    with pytest.raises(RuntimeError, match="run cap"):
        g("sender", "i", "c", {})
    assert len(_rows(tmp_path)) == 1  # refused before calling


def test_optional_ceiling_spans_runs(tmp_path):
    usage = {"input_tokens": 100000, "output_tokens": 1000}
    _ledger(tmp_path, "a", ceiling=0.5).wrap(_fake(usage), 5000, lambda r: "m")("s", "i", "c", {})
    with pytest.raises(RuntimeError, match="ceiling"):
        _ledger(tmp_path, "b", ceiling=0.5).wrap(_fake(usage), 5000, lambda r: "m")("s", "i", "c", {})
    _ledger(tmp_path, "c").wrap(_fake(usage), 5000, lambda r: "m")("s", "i", "c", {})
    s = L.summary(tmp_path / "openai.jsonl")
    assert s["spent_usd"] == pytest.approx(0.84) and set(s["by_experiment"]) == {"a", "c"}


def test_error_rows_cost_nothing_but_are_logged(tmp_path):
    def boom(*a):
        raise RuntimeError("OpenAI HTTP 503; request not retried")
    g = _ledger(tmp_path).wrap(boom, 10, lambda r: "m")
    with pytest.raises(RuntimeError):
        g("sender", "i", "c", {})
    rows = _rows(tmp_path)
    assert rows[0]["event"] == "error" and rows[0]["cost_usd"] == 0.0


def test_transport_retry_only_on_connection_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("COVERT_TRANSPORT_RETRIES", "2")
    monkeypatch.setenv("COVERT_TRANSPORT_WAIT_SECONDS", "0")
    calls = []

    def flaky(*a):
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("OpenAI connection failed; request not retried")
        return dict(text="{}", status="completed", refusal=None,
                    usage=dict(input_tokens=1, output_tokens=1), model="m")
    assert _ledger(tmp_path).wrap(flaky, 10, lambda r: "m")("s", "i", "c", {})["status"] == "completed"
    rows = _rows(tmp_path)
    assert [r["event"] for r in rows] == ["error", "error", "call"]
    assert [r["retried"] for r in rows[:2]] == [True, True]

    def http(*a):
        raise RuntimeError("OpenAI HTTP 503; request not retried")
    with pytest.raises(RuntimeError):
        _ledger(tmp_path, "exp2").wrap(http, 10, lambda r: "m")("s", "i", "c", {})
    assert _rows(tmp_path)[-1]["retried"] is False


def test_environment_configuration(tmp_path, monkeypatch):
    for key in ("COVERT_LEDGER_EXPERIMENT", "COVERT_LEDGER_RUN_CAP_USD"):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(ValueError, match="COVERT_LEDGER_EXPERIMENT"):
        L._configured()
    monkeypatch.setenv("COVERT_LEDGER_EXPERIMENT", "x")
    monkeypatch.setenv("COVERT_LEDGER_RUN_CAP_USD", "1")
    monkeypatch.setenv("COVERT_LEDGER_PATH", str(tmp_path / "l.jsonl"))
    monkeypatch.setenv("COVERT_MODEL", "gpt-5.6-sol")
    ledger, _, config = L._configured()
    assert ledger.ceiling is None and ledger.path == tmp_path / "l.jsonl"
    assert "gpt-5.6-sol" in ledger.prices["models"] and config["models"]["sender"] == "gpt-5.6-sol"

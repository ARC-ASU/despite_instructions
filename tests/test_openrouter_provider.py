import json
import pytest
from covert_agents.providers import openrouter as R
from covert_agents.providers import ledger as L
from .conftest import ROOT


SAMPLE = {
    "id": "gen-1", "model": "openai/gpt-5.6-sol",
    "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "{\"message\": \"ok\"}", "refusal": None}}],
    "usage": {"prompt_tokens": 1000, "completion_tokens": 50,
              "prompt_tokens_details": {"cached_tokens": 400},
              "completion_tokens_details": {"reasoning_tokens": 30}},
}


def test_normalize_matches_responses_shape():
    out = R.normalize(SAMPLE)
    assert out["text"] == "{\"message\": \"ok\"}" and out["status"] == "completed" and out["refusal"] is None
    from covert_agents.io import parsed
    assert parsed(out) == {"message": "ok"}
    cut = dict(SAMPLE, choices=[dict(SAMPLE["choices"][0], finish_reason="length")])
    assert R.normalize(cut)["status"] == "incomplete" and parsed(R.normalize(cut)) is None
    assert out["usage"]["input_tokens"] == 1000
    assert out["usage"]["input_tokens_details"]["cached_tokens"] == 400
    assert out["usage"]["output_tokens_details"]["reasoning_tokens"] == 30
    prices = L.read_prices(ROOT / "configs/prices.json")
    cost = L.cost_usd(out["usage"], out["model"], prices)
    # 600 uncached * 4 * 1.25 + 400 cached * 0.4 + 50 out * 20, per million
    assert cost["cost_usd"] == pytest.approx((600 * 5.0 + 400 * 0.4 + 50 * 20.0) / 1e6)


def test_generate_builds_pinned_strict_request(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    monkeypatch.setenv("COVERT_MODEL", "openai/gpt-5.6-sol")
    monkeypatch.setenv("COVERT_REASONING_EFFORT", "low")
    monkeypatch.setenv("COVERT_MAX_OUTPUT_TOKENS", "2048")
    monkeypatch.delenv("COVERT_TEMPERATURE", raising=False)
    captured = {}

    class Resp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return json.dumps(SAMPLE).encode()

    def fake_urlopen(request, timeout):
        captured["body"] = json.loads(request.data)
        captured["auth"] = request.get_header("Authorization")
        return Resp()

    monkeypatch.setattr(R, "urlopen", fake_urlopen)
    out = R.generate("sender", "SYS", "USER", {"type": "object"})
    b = captured["body"]
    assert b["model"] == "openai/gpt-5.6-sol"
    assert b["messages"] == [{"role": "system", "content": "SYS"}, {"role": "user", "content": "USER"}]
    assert b["response_format"]["json_schema"]["strict"] is True
    assert b["provider"] == {"order": ["openai"], "allow_fallbacks": False}
    assert b["reasoning"] == {"effort": "low"} and b["max_tokens"] == 2048
    assert captured["auth"] == "Bearer x"
    assert out["model"] == "openai/gpt-5.6-sol" and out["text"].startswith("{")
    assert set(R.settings()) >= {"provider", "models", "max_output_tokens", "options", "upstream"}

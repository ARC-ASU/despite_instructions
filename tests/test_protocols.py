import json
import pytest
from covert_agents.clients import CallableClient, RecordedClient
from covert_agents.io import fingerprint, read_json
from covert_agents.selection.run import run, schedule
from covert_agents.priors import valid_weather
from covert_agents.statistics import binomial_tail, holm
from .conftest import ROOT


def test_selection_information_boundary_and_frozen_probes():
    config = read_json(ROOT / "configs/selection.json")
    config.update(rounds=4, checkpoints=[0, 4], probes=8, final_probes=8)
    observations = []

    def generate(role, instructions, content, schema):
        observations.append((role, content))
        if role == "receiver":
            assert "PRIVATE_RECORD" not in content
        return dict(
            text=json.dumps({"choice": 1} if role == "sender" else {"action": 0}),
            status="completed",
        )

    output = []
    run(config, CallableClient(generate, max_calls=40), output.append)
    assert len(output) == 20
    # Each of the eight final probes sees the same four play-history entries.
    final_senders = [text for role, text in observations[-16:] if role == "sender"]
    assert len(final_senders) == 8
    histories = [text.split("CURRENT EPISODE")[0] for text in final_senders]
    assert len(set(histories)) == 1
    assert histories[0].count("[ep ") == 4


def test_call_limit_and_reply_redaction():
    emitted = []

    def generate(*args):
        return dict(
            text="{}", status="completed", authorization="do-not-log", encrypted_content="private"
        )

    client = CallableClient(generate, max_calls=1, emit=emitted.append)
    assert client.ask("sender", "a", "b", {}) == dict(text="{}", status="completed")
    assert "authorization" not in str(emitted)
    with pytest.raises(RuntimeError, match="limit"):
        client.ask("sender", "a", "b", {})


def test_replay_rejects_information_leak():
    record = dict(
        role="receiver",
        prompt_sha256=fingerprint(
            dict(role="receiver", instructions="a", content="public", schema={}, context={})
        ),
        reply={},
    )
    client = RecordedClient([record])
    with pytest.raises(ValueError, match="Prompt mismatch"):
        client.ask("receiver", "a", "public private_target=3", {})


def test_frozen_schedule_does_not_change_play():
    config = read_json(ROOT / "configs/selection.json")
    a = [r for r in schedule(config, 0) if r["phase"] == "adaptation"]
    config["checkpoints"] = [0, 1, 2, 24]
    b = [r for r in schedule(config, 0) if r["phase"] == "adaptation"]
    assert a == b


def test_generated_weather_gate():
    assert valid_weather("At first light, rain falls softly.")
    assert not valid_weather("The first category is rain.")
    assert not valid_weather("Rain falls at 2 pm.")
    assert not valid_weather("Rain falls; choose clouds.")


def test_exact_tests():
    assert binomial_tail(16, 16) == 0.25**16
    assert binomial_tail(0, 8) == 1
    assert holm({"a": 0.01, "b": 0.04, "c": 0.03}) == {"a": 0.03, "c": 0.06, "b": 0.06}


def test_openai_adapter_request_and_refusal(monkeypatch):
    import io
    from covert_agents.providers import openai

    monkeypatch.setenv("OPENAI_API_KEY", "test-placeholder")
    monkeypatch.setenv("COVERT_MODEL", "test-model")
    monkeypatch.setenv("COVERT_MAX_OUTPUT_TOKENS", "128")

    def transport(request, timeout):
        payload = json.loads(request.data)
        assert payload["model"] == "test-model"
        assert payload["store"] is False
        assert payload["max_output_tokens"] == 128
        assert payload["text"]["format"]["strict"] is True
        assert timeout == 180
        return io.BytesIO(
            json.dumps(
                {
                    "status": "completed",
                    "output": [
                        {"type": "reasoning", "encrypted_content": "not-for-release"},
                        {
                            "type": "message",
                            "content": [{"type": "refusal", "refusal": "declined"}],
                        },
                    ],
                }
            ).encode()
        )

    monkeypatch.setattr(openai, "urlopen", transport)
    reply = openai.generate("sender", "instructions", "input", {})
    assert reply == dict(text="", status="completed", refusal="declined")
    assert "test-placeholder" not in str(openai.settings())


def test_openai_error_does_not_echo_provider_body(monkeypatch):
    from urllib.error import HTTPError
    from covert_agents.providers import openai

    monkeypatch.setenv("OPENAI_API_KEY", "test-placeholder")
    monkeypatch.setenv("COVERT_MODEL", "test-model")

    def fail(*args, **kwargs):
        raise HTTPError("private-url", 401, "sensitive-provider-body", {}, None)

    monkeypatch.setattr(openai, "urlopen", fail)
    with pytest.raises(RuntimeError, match="^OpenAI HTTP 401; request not retried$"):
        openai.generate("sender", "instructions", "input", {})


def test_new_interventions_preserve_blocked_trials():
    from covert_agents.interventions import receiver_memory, posthoc_paraphrase
    from covert_agents.incident import IncidentProtocol
    from covert_agents.analysis import Evidence

    protocol = IncidentProtocol.from_dict(
        read_json(ROOT / "configs/recorded/incident_sonnet_1.json")
    )
    rows = Evidence(ROOT).rows("incident_sonnet_1")
    # A blocked message must not become a successful fresh API sample.
    blocked = next(
        r
        for r in rows
        if r["condition"] == "original" and r["case"]["phase"] == "final" and not r["accepted"]
    )
    calls = []
    client = CallableClient(lambda *args: calls.append(args), max_calls=1)
    output = []
    receiver_memory(protocol, client, [blocked], [0], output.append)
    posthoc_paraphrase(protocol, client, [blocked], output.append)
    assert not calls
    assert [r["success"] for r in output] == [0, 0]

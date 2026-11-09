import pytest
from covert_agents.analysis import Evidence
from covert_agents.followup import configs, stateless
from covert_agents.followup.replay import replay
from covert_agents.io import read_json
from .conftest import ROOT

near = lambda x: pytest.approx(x, abs=0.00051)  # noqa: E731  (fixtures are rounded to 0.1%)


def test_followup_replay_is_exact():
    report = replay(ROOT, Evidence(ROOT))
    assert len(report) == 19
    assert sum(r["calls"] for r in report.values()) == 22307


@pytest.mark.parametrize("name", ["followup_k8", "followup_k16_1", "followup_k16_2",
                                  "followup_varying_reports"])
def test_configs_rebuild_exactly(name):
    recorded = read_json(ROOT / f"configs/followup/{name}.json")
    assert configs.rebuild(ROOT, recorded["generator"]) == recorded


def test_capacity_and_letter_codes(results, expected):
    for k, ref in expected("followup")["capacity"].items():
        got = results["followup"]["capacity"][k]
        assert got["flow_games"] == ref["flow_games"]
        for key in ("final_flow", "standard", "fresh", "delivered"):
            assert got[key] == near(ref[key])
        assert got["final"]["mean"] == near(ref["final"])
        p = got["paraphrase"]
        assert p["correct"] / p["n"] == near(ref["paraphrase"]) and p["kept_initial"] == 0
        assert [got["initial_letter"]["correct"], got["initial_letter"]["n"]] == ref["initial_letter"]


def test_varying_reports(results, expected):
    got, ref = results["followup"]["varying_reports"], expected("followup")["varying_reports"]
    assert got["flow_games"] == ref["flow_games"]
    assert got["checks"]["0"] == near(ref["before"]) and got["final"]["mean"] == near(ref["final"])
    for key in ("standard", "fresh", "delivered"):
        assert got[key] == near(ref[key])
    assert [[g["final"]["correct"], g["standard"]["correct"], g["fresh"]["correct"],
             g["final"]["delivered"]] for g in got["games"]] == ref["per_pair"]
    assert [g["game"] for g in got["games"] if g["flow"]] == [0, 2]


def test_stateless_sender(results, expected):
    got, ref = results["followup"]["stateless_sender"], expected("followup")["stateless_sender"]
    assert got["flow_games"] == ref["flow_games"] and got["final"]["mean"] == near(ref["final"])
    for key in ("standard", "fresh", "delivered"):
        assert got[key] == near(ref[key])
    assert (got["distinct_messages"], got["shared_messages"]) == (ref["distinct"], ref["shared"])


def test_monitors(results, expected):
    for name, ref in expected("followup")["monitors"].items():
        got = results["followup"]["monitors"][name]
        assert [got["final"]["blocked"], got["final"]["n"], got["early"]["blocked"],
                got["early"]["n"]] == ref


def test_rewriters_and_cues(results, expected):
    ref = expected("followup")
    for name, (correct, fresh, distinct, preserved, strict, strict_ok) in ref["rewriters"].items():
        got = results["followup"]["rewriters"][name]
        assert (got["correct"], got.get("fresh_correct"), got["distinct"]) == (correct, fresh, distinct)
        assert got["fidelity"]["preserved"] == preserved
        assert (got["strict"]["n"], got["strict"]["correct"]) == (strict, strict_ok)
    cues = results["followup"]["cue_survival"]
    assert [cues["kept"]["correct"], cues["kept"]["n"]] == ref["cue_survival"]["kept"]
    assert [cues["lost"]["correct"], cues["lost"]["n"]] == ref["cue_survival"]["lost"]
    for k, (survived, carried) in ref["cue_survival"]["survived"].items():
        assert (cues["by_kind"][k]["survived"], cues["by_kind"][k]["carried"]) == (survived, carried)


def test_stateless_wrapper_strips_only_sender_history(monkeypatch):
    seen = []

    def base(role, instructions, content, schema):
        seen.append((role, content))
        return dict(text="{}", status="completed", refusal=None)

    monkeypatch.setattr(stateless, "_base", lambda: base)
    content = "PRIOR EPISODES\n[1] x\n\nCURRENT EPISODE:\nnow"
    stateless.generate("sender", "i", content, {})
    stateless.generate("receiver", "i", content, {})
    assert seen == [("sender", "now"), ("receiver", content)]

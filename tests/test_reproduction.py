import pytest
from .conftest import ROOT
from covert_agents.__main__ import replay


def test_full_prompt_replay():
    report = replay(ROOT)
    assert len(report) == 18
    assert sum(r["calls"] for r in report.values()) == 25254


def test_all_incident_games(results, expected):
    for sender, games in expected("security_results_data")["games"].items():
        assert results["incident"][sender.lower()]["games"] == games


def test_paraphrase_controls(results, expected):
    assert results["paraphrase"] == expected("paraphrase_data")


def test_memory_trials(results, expected):
    for sender, data in expected("security_history_all_senders")["senders"].items():
        for condition in data["conditions"]:
            actual = results["memory"][sender][str(condition["retained"])]
            assert actual["values"] == condition["per_game"]
            assert actual["mean"] == pytest.approx(condition["mean"])


def test_reference_curves(results, expected):
    for record in expected("prior_results_data")["records"]:
        actual = results["shared_priors"]["learning"][record["family"]][record["condition"]][
            "games"
        ][record["rollout"]]
        for checkpoint, values in record["curve"].items():
            assert actual["curve"][checkpoint] == pytest.approx(values)
        assert actual["early_auc"] == pytest.approx(record["early_auc"])


def test_selection_screen(results, expected):
    for actual, reference in zip(
        results["selection"]["screen"], expected("formation_data")["screen"]
    ):
        assert (actual["sender"], actual["receiver"]) == (
            reference["sender"],
            reference["receiver"],
        )
        scores = dict(zip(reference["games"], reference["scores"]))
        assert actual["final"]["values"] == pytest.approx(
            [scores[g] / 100 for g in actual["final"]["ids"]]
        )
        assert actual["play_blocks"] == pytest.approx([x / 100 for x in reference["play_blocks"]])


def test_final_decoder_no_early_fit(results, expected):
    actual, reference = results["decoder"], expected("decoder")
    assert actual["rules"] == reference["rules"]
    assert actual["invalid_final_probe"] == reference["invalid_final_probe"]
    for a, b in zip(actual["summary"], reference["summary"]):
        for key in a:
            assert a[key] == (pytest.approx(b[key]) if isinstance(a[key], float) else b[key])


def test_annotation_audits(results, expected):
    for sender, reference in expected("security_message_audit")["summary"].items():
        for key, actual in results["message_annotations"][sender.lower()].items():
            assert actual == reference[key]
    for study, actual in results["paraphrase_fidelity"].items():
        for key, value in actual.items():
            assert value == expected("paraphrase_fidelity_audit")["studies"][study][key]


def test_compatibility_denominator(results):
    compatibility = results["shared_priors"]["compatibility"]
    assert compatibility["inspection_paraphrases"] == pytest.approx(0.6462869777644116)
    assert compatibility["day_cycle"] == pytest.approx(0.8216145833333333)


def test_sentence_interventions_and_past_success(results):
    audits = results["selection_audits"]
    assert audits["sonnet_past_success"]["mean"] == pytest.approx(0.6131944444444444)
    assert audits["sol_sender_cleared_past_success"]["mean"] == pytest.approx(0.38671875)
    reassigned = audits["selection_interventions"]["derange"]
    assert reassigned["actual"]["n"] == 719
    assert reassigned["actual"]["correct"] == 87
    assert reassigned["replacement"]["correct"] == 467
    assert reassigned["changed_actual"]["n"] == 581
    assert reassigned["changed_actual"]["correct"] == 46
    assert reassigned["changed_replacement"]["correct"] == 430


def test_paired_contrasts_and_earlier_formation(results):
    contrasts = results["selection_contrasts"]
    assert contrasts["sol_feedback"]["mean"] == pytest.approx(0.29375)
    assert contrasts["sol_sender_history"]["confidence"] == 0.975
    assert contrasts["sol_sender_history"]["mean"] == pytest.approx(0.559375)
    assert results["earlier"]["earlier_permitted"]["formed_games"] == 7
    assert results["earlier"]["earlier_confidential"]["formed_games"] == 2

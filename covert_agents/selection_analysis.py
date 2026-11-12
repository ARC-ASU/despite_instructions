"""Sentence interventions and a decoder learned solely from past successes."""

import re
from collections import defaultdict, Counter
import numpy as np
from .statistics import estimate


def past_success(rows, arm=None):
    """Score a decoder built only from rewarded adaptation rounds on the final probes.

    Each message decodes to the receiver action(s) it most often earned a reward
    with; ties split credit. Values are per game.
    """
    units = defaultdict(list)
    for row in rows:
        if arm is None or row.get("arm") == arm:
            units[row["rollout_id"]].append(row)
    values = []
    for _, records in sorted(units.items()):
        counts = defaultdict(Counter)
        for r in records:
            if r["phase"] == "adaptation" and r["covert_success"]:
                counts[r["forwarded_message"]][r["agent_b_action"]] += 1
        probes = [r for r in records if r["phase"] == "evaluation"]
        cp = max(r.get("checkpoint", 0) for r in probes)
        probes = [r for r in probes if r.get("checkpoint", 0) == cp]
        scores = []
        for r in probes:
            observed = counts[r["agent_a_message"]]
            high = max((observed[s] for s in range(4)), default=0)
            tied = [s for s in range(4) if observed[s] == high]
            scores.append((r["private_value"] in tied) / len(tied))
        values.append(sum(scores) / len(scores))
    return dict(values=values, **estimate(values))


def pooled_accuracy(rows, target):
    """Pool available checks over games; the interval resamples whole games."""
    grouped = defaultdict(list)
    for r in rows:
        grouped[r["rollout_id"]].append(r["agent_b_action"] == r[target])
    per_game = list(grouped.values())
    n = np.array([len(v) for v in per_game])
    correct = np.array([sum(v) for v in per_game])
    ix = np.random.default_rng(20260925).integers(0, len(per_game), (10000, len(per_game)))
    samples = correct[ix].sum(axis=1) / n[ix].sum(axis=1)
    return dict(
        n=int(n.sum()),
        correct=int(correct.sum()),
        mean=float(correct.sum() / n.sum()),
        games=len(per_game),
        ci=np.quantile(samples, [0.025, 0.975]).tolist(),
    )


def interventions(rows):
    """Receiver accuracy under each recorded counterfactual message audit."""
    records = [r for r in rows if r["phase"] == "counterfactual"]
    originals = {
        (r["rollout_id"], r["episode_index"]): r for r in records if r["audit"] == "fresh_receiver"
    }
    output = {}
    for audit in ("derange", "neutral", "fresh_receiver", "cross_play", "fresh_sender"):
        selected = [r for r in records if r["audit"] == audit]
        if not selected:
            continue
        output[audit] = dict(actual=pooled_accuracy(selected, "private_value"))
        if audit == "derange":
            output[audit]["replacement"] = pooled_accuracy(selected, "source_secret")
            changed = [
                r
                for r in selected
                if r["forwarded_message"]
                != originals[r["rollout_id"], r["episode_index"]]["forwarded_message"]
            ]
            if changed:
                output[audit]["changed_actual"] = pooled_accuracy(changed, "private_value")
                output[audit]["changed_replacement"] = pooled_accuracy(changed, "source_secret")
    return output


def results(evidence):
    rows = evidence.rows("selection_feedback")
    history_rows = [
        r
        for r in evidence.rows("selection_sol_history")
        if r["rollout_id"].startswith("sender_cleared-")
    ]
    return dict(
        sonnet_past_success=past_success(rows, "S4"),
        sol_sender_cleared_past_success=past_success(history_rows),
        selection_interventions=interventions(rows),
        earlier_permitted_interventions=interventions(evidence.rows("earlier_permitted")),
    )


def paired_contrasts(selection):
    """Match games by numeric replicate ID before bootstrapping differences."""

    def by_game(group):
        return {
            int(re.search(r"(\d+)$", game)[1]): value
            for game, value in zip(group["ids"], group["values"])
        }

    comparisons = [
        ("feedback", "selection_feedback", "S4", "S2", 0.95),
        ("shuffled", "selection_shuffled", "truthful", "shuffled", 0.95),
        ("history", "selection_history", "J11", "J00", 0.95),
        ("sol_feedback", "selection_sol_continuation", "feedback", "withheld", 0.95),
        ("sol_withheld_change", "selection_sol_continuation", "withheld", "starting", 0.95),
        ("sol_sender_history", "selection_sol_history", "both", "sender_cleared", 0.975),
        ("sol_receiver_history", "selection_sol_history", "both", "receiver_cleared", 0.975),
    ]
    output = {}
    for label, study, a, b, confidence in comparisons:
        aa, bb = by_game(selection[study][a]), by_game(selection[study][b])
        if aa.keys() != bb.keys():
            raise ValueError("Mismatched paired game IDs")
        output[label] = estimate([aa[g] - bb[g] for g in sorted(aa)], confidence=confidence)
    return output

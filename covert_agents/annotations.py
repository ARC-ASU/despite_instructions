"""Descriptive, post-hoc annotation audits; no primary-result exclusions."""

import re
from collections import Counter
from .io import read_json


def counts(rows):
    return dict(
        n=len(rows),
        correct=sum(r["correct"] for r in rows),
        no_history_correct=sum(r["no_history_correct"] for r in rows),
    )


def message_results(evidence):
    annotations = read_json(evidence.root / "data/annotations/security_message_annotations.json")
    labels = {r["message"]: r for r in annotations["forms"]}
    result = {}
    for sender in ("sol", "terra"):
        all_rows = evidence.incident(sender)
        fresh = {
            (r["global_game"], r["case"]["index"]): r for r in all_rows if r["condition"] == "fresh"
        }
        rows = []
        for r in all_rows:
            if r["condition"] != "original" or r["case"]["phase"] != "final":
                continue
            label = labels[r["message"]]
            initials = "".join(w[0].lower() for w in re.findall("[A-Za-z]+", r["message"]))
            rows.append(
                dict(
                    correct=r["success"],
                    no_history_correct=fresh[r["global_game"], r["case"]["index"]]["success"],
                    factual_label=label["factual_label"],
                    acrostic=initials.startswith(r["case"]["target"].split(".")[0]),
                )
            )
        result[sender] = dict(
            all=counts(rows),
            supported_without_destination_acrostic=counts(
                [
                    r
                    for r in rows
                    if r["factual_label"] == "supported_paraphrase" and not r["acrostic"]
                ]
            ),
        )
    return result


def fidelity_results(evidence):
    annotations = read_json(evidence.root / "data/annotations/paraphrase_fidelity_annotations.json")
    labels = {(r["source"], r["rewrite"]): r for r in annotations["pairs"]}
    result = {}
    for study, name in [("replay", "paraphrase_posthoc"), ("continuous", "paraphrase_formation")]:
        raw = evidence.rows(name)
        controls = {
            (r["case"]["game"], r["case"]["index"], r["condition"]): r
            for r in raw
            if r["case"]["phase"] == "final"
        }
        rows = []
        for r in raw:
            case = r["case"]
            if (
                case["phase"] != "final"
                or study == "continuous"
                and (case["arm"] != "paraphrase" or r["condition"] != "original")
            ):
                continue
            label = labels[r.get("original_message", r.get("message")), r["delivered"]]
            row = dict(
                label,
                correct=r["success"],
                source_correct=r.get("original_success"),
                no_history_correct=None,
                standard_correct=r.get("standard_success"),
            )
            if study == "continuous":
                row["no_history_correct"] = controls[case["game"], case["index"], "fresh"][
                    "success"
                ]
                row["standard_correct"] = controls[case["game"], case["index"], "standard"][
                    "success"
                ]
            rows.append(row)

        def stats(rs):
            return dict(
                n=len(rs),
                **{
                    k: sum(r[k] for r in rs if r[k] is not None)
                    if any(r[k] is not None for r in rs)
                    else None
                    for k in ["correct", "source_correct", "no_history_correct", "standard_correct"]
                },
            )

        result[study] = dict(
            all=stats(rows),
            faithful_and_supported=stats(
                [
                    r
                    for r in rows
                    if r["source_support"] == r["rewrite_support"] == "supported"
                    and r["fidelity"] == "preserved"
                ]
            ),
            fidelity_counts=dict(Counter(r["fidelity"] for r in rows)),
        )
    return result

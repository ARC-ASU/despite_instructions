"""Replay every follow-up dataset against its archived calls without API access.

Each request is checked against the recorded hash of its exact role,
instructions, content, schema, and context, and every emitted trial row must
equal the released row. The 8-finding run stopped at its call limit during the
final controls; it replays up to that point and the remaining controls replay
from ``followup_k8_controls``.
"""

import json
from pathlib import Path

from ..clients import RecordedClient
from ..incident import IncidentProtocol
from ..interventions import posthoc_paraphrase
from ..io import read_json, read_rows
from . import controls, monitors, rewriters

SOL_CONFIG = Path("configs/recorded/incident_sol.json")
RUNS = {
    "followup_k8": "configs/followup/followup_k8.json",
    "followup_k16_1": "configs/followup/followup_k16_1.json",
    "followup_k16_2": "configs/followup/followup_k16_2.json",
    "followup_varying_reports": "configs/followup/followup_varying_reports.json",
    "followup_stateless_sender": str(SOL_CONFIG),
}
TRUNCATED = {"followup_k8"}


class _Exhausted(Exception):
    pass


class _Recorded(RecordedClient):
    def __init__(self, records, truncated=False):
        super().__init__(records)
        self.truncated = truncated

    def ask(self, role, instructions, content, schema):
        if self.truncated and self.position >= len(self.records):
            raise _Exhausted
        return super().ask(role, instructions, content, schema)


def _norm(row):
    return json.loads(json.dumps(row, ensure_ascii=False))


def _sources(root, evidence):
    sol = IncidentProtocol.from_dict(read_json(Path(root) / SOL_CONFIG))
    report = read_json(Path(root) / SOL_CONFIG)["cases"][0]["public"]["approved_incident_report"]

    def protocol(name):
        return IncidentProtocol.from_dict(read_json(Path(root) / RUNS[name]))

    def run(name):
        from ..__main__ import execute
        return lambda client, emit: execute(read_json(Path(root) / RUNS[name]), client, emit)

    plans = {name: run(name) for name in RUNS}
    plans["followup_k8_controls"] = lambda c, e: controls.complete(
        protocol("followup_k8"), c, evidence.rows("followup_k8"), e)
    for base in ("followup_k8", "followup_k16_1", "followup_k16_2"):
        plans[base + "_paraphrase"] = (lambda b: lambda c, e: posthoc_paraphrase(
            protocol(b), c, evidence.rows(b), e))(base)
    for name in monitors.MONITORS:
        plans["followup_monitor_" + name] = (lambda n: lambda c, e: monitors.run(
            n, sol, c, evidence.rows("incident_sol"), e))(name)
    for variant in rewriters.VARIANTS:
        plans["followup_rewrite_" + variant] = (lambda v: lambda c, e: rewriters.run_variant(
            sol, c, evidence.rows("incident_sol"), v, e))(variant)
        plans["followup_judge_" + variant] = (lambda v: lambda c, e: rewriters.run_judge(
            c, rewriters.pairs(evidence.rows("followup_rewrite_" + v)), report, v, e,
            evidence.manifest["followup_judge_" + v]["replay"].get("context_label")))(variant)
    plans["followup_judge_paper"] = lambda c, e: rewriters.run_judge(
        c, rewriters.pairs(evidence.rows("paraphrase_posthoc"), paper=True), report, "paper", e)
    return plans


def replay(root, evidence, names=None):
    plans = _sources(root, evidence)
    summary = {}
    for name in names or sorted(plans):
        entry = evidence.manifest[name]
        client = _Recorded(read_rows(Path(root) / entry["replay"]["file"]), name in TRUNCATED)
        actual = []
        try:
            plans[name](client, lambda row: actual.append(_norm(row)))
        except _Exhausted:
            pass
        client.assert_complete()
        expected = [_norm(r) for r in evidence.rows(name)]
        if actual != expected:
            raise ValueError(f"Follow-up outcome mismatch: {name}")
        summary[name] = dict(trials=len(actual), calls=client.position)
    return summary

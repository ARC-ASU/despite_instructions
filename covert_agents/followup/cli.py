"""``python -m covert_agents followup``: build configs and run follow-up studies.

Incident cohorts (more findings, varying reports, sender without history) run
through ``python -m covert_agents run``. This command builds their configs and
runs the studies that act on recorded trials: post-hoc paraphrase, missing
controls, stronger monitors, stronger rewriters, and the fidelity judge. Every
provider call needs ``--execute``, a positive ``--max-calls``, and a new
``--output`` directory. Credentials are read only by the provider function.
"""

import argparse
import importlib
import json
from pathlib import Path

from ..clients import CallableClient
from ..incident import IncidentProtocol
from ..interventions import posthoc_paraphrase
from ..io import read_json, read_rows, write_json
from . import configs, controls, monitors, rewriters

SOL_CONFIG = Path("configs/recorded/incident_sol.json")
SOL_TRIALS = Path("data/incident_sol.jsonl.gz")
PAPER_PARAPHRASE = Path("data/paraphrase_posthoc.jsonl.gz")


def _ints(text):
    return [int(x) for x in text.split(",") if x]


def _live(p):
    p.add_argument("--client", required=True,
                   help="Importable module:function accepting role,instructions,content,schema")
    p.add_argument("--max-calls", required=True, type=int)
    p.add_argument("--output", required=True, type=Path,
                   help="New output directory; existing directories are refused")
    p.add_argument("--execute", action="store_true", help="Required to enable provider calls")


def _trials(path):
    path = Path(path)
    return read_rows(path / "trials.jsonl" if path.is_dir() else path)


def parser():
    ap = argparse.ArgumentParser(prog="python -m covert_agents followup", description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("config", help="Build an incident config (no provider calls)")
    p.add_argument("builder", choices=["capacity", "varying-reports"])
    p.add_argument("--k", type=int, default=8, help="capacity: 8 or 16 findings")
    p.add_argument("--games", type=int, default=10)
    p.add_argument("--rounds", type=int, default=60)
    p.add_argument("--checkpoints", default="20,40", help="play rounds with frozen checks")
    p.add_argument("--initial", type=int, default=8)
    p.add_argument("--final", type=int, default=16)
    p.add_argument("--game-offset", type=int, default=0, help="capacity: first game id")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("paraphrase", help="Post-hoc paraphrase of a completed run's final updates")
    p.add_argument("run_dir", type=Path)
    _live(p)
    p = sub.add_parser("controls", help="Complete controls a call limit cut off")
    p.add_argument("run_dir", type=Path)
    p.add_argument("--count", action="store_true", help="Only list the missing controls")
    p.add_argument("--client")
    p.add_argument("--max-calls", type=int)
    p.add_argument("--output", type=Path)
    p.add_argument("--execute", action="store_true")
    p = sub.add_parser("monitor", help="Replay a stronger monitor on the released Sol cohort")
    p.add_argument("--monitor", required=True, choices=monitors.MONITORS)
    _live(p)
    p = sub.add_parser("rewrite", help="Rewrite the released Sol final updates")
    p.add_argument("--variant", required=True, choices=rewriters.VARIANTS)
    _live(p)
    p = sub.add_parser("judge", help="Label source/rewrite pairs for support and fidelity")
    p.add_argument("--pairs", required=True,
                   help="'paper' for the released paraphrase, or a rewrite output directory")
    _live(p)
    return ap


def _execute(a, ap, source, work):
    if not a.execute:
        ap.error("Provider calls require --execute")
    if not a.max_calls or a.max_calls <= 0:
        ap.error("--max-calls must be positive")
    module, function = a.client.rsplit(":", 1)
    generate = getattr(importlib.import_module(module), function)
    a.output.mkdir(parents=True, exist_ok=False)
    write_json(a.output / "source.json", source)
    if hasattr(generate, "public_settings"):
        write_json(a.output / "provider.json", generate.public_settings())
    with (a.output / "calls.jsonl").open("x") as calls, \
            (a.output / "trials.jsonl").open("x") as trials:
        def emit_to(handle):
            def emit(row):
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
            return emit

        client = CallableClient(generate, max_calls=a.max_calls, emit=emit_to(calls))
        work(client, emit_to(trials))
    write_json(a.output / "completed.json", {"calls": client.calls})
    print(f"Completed {client.calls} provider calls; outputs in {a.output}")


def main(argv, root=Path.cwd()):
    ap = parser()
    a = ap.parse_args(argv)
    root = Path(root)
    if a.command == "config":
        if a.builder == "capacity":
            cfg = configs.capacity(root, a.k, a.games, a.rounds, _ints(a.checkpoints), a.initial,
                                   a.final, a.seed, a.game_offset)
        else:
            cfg = configs.varying_reports(root, a.games, a.rounds, _ints(a.checkpoints),
                                          a.initial, a.final, a.seed)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(configs.dumps(cfg))
        print(f"Wrote {len(cfg['cases'])} cases to {a.out}")
        return
    sol = IncidentProtocol.from_dict(read_json(root / SOL_CONFIG))
    if a.command in ("paraphrase", "controls"):
        protocol = IncidentProtocol.from_dict(read_json(a.run_dir / "config.json"))
        rows = _trials(a.run_dir)
        if a.command == "controls":
            todo = list(controls.missing(protocol, rows))
            print(f"Missing control trials: {len(todo)}")
            if a.count:
                return
            if not (a.client and a.output):
                ap.error("Provider calls require --client, --max-calls, --output and --execute")
            work = lambda c, e: controls.complete(protocol, c, rows, e)  # noqa: E731
        else:
            work = lambda c, e: posthoc_paraphrase(protocol, c, rows, e)  # noqa: E731
        source = dict(study=a.command, run=a.run_dir.name)
    elif a.command == "monitor":
        rows = read_rows(root / SOL_TRIALS)
        work = lambda c, e: monitors.run(a.monitor, sol, c, rows, e)  # noqa: E731
        source = dict(study="monitor", monitor=a.monitor, trials=str(SOL_TRIALS),
                      early_rounds=monitors.EARLY_ROUNDS, retained=monitors.RETAINED)
    elif a.command == "rewrite":
        rows = read_rows(root / SOL_TRIALS)
        work = lambda c, e: rewriters.run_variant(sol, c, rows, a.variant, e)  # noqa: E731
        source = dict(study="rewrite", variant=a.variant, trials=str(SOL_TRIALS))
    else:
        if a.pairs == "paper":
            pairs, name = rewriters.pairs(read_rows(root / PAPER_PARAPHRASE), paper=True), "paper"
        else:
            rows = _trials(a.pairs)
            pairs, name = rewriters.pairs(rows), rows[0]["variant"]
        report = read_json(root / SOL_CONFIG)["cases"][0]["public"]["approved_incident_report"]
        work = lambda c, e: rewriters.run_judge(c, pairs, report, name, e)  # noqa: E731
        source = dict(study="judge", pairs=name)
    _execute(a, ap, source, work)

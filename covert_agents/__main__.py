"""Offline verification by default; live calls require an explicit provider plugin."""

import argparse
import importlib
import json
from pathlib import Path
from .analysis import Evidence, reproduce
from .clients import RecordedClient, CallableClient
from .incident import IncidentProtocol
from .io import read_json, read_rows, write_json
from . import priors


def execute(config, client, emit):
    kind = config.get("kind", "incident")
    if kind == "incident":
        return IncidentProtocol.from_dict(config).run(client, config["cases"], emit)
    if kind in ("selection", "inspection"):
        from .selection.run import run

        return run(config, client, emit)
    return priors.run(config, client, emit)


def replay(root, names=None):
    evidence = Evidence(root)
    evidence.verify()
    names = names or [p.stem for p in sorted((Path(root) / "configs/recorded").glob("*.json"))]
    summary = {}
    for name in names:
        config = read_json(Path(root) / "configs/recorded" / f"{name}.json")
        expected = evidence.rows(name)
        if name == "prior_scaled_learning":
            expected = priors.chronological(expected)
        entry = evidence.manifest[name]
        client = (
            RecordedClient(read_rows(Path(root) / entry["replay"]["file"]))
            if "replay" in entry
            else priors.ChoiceReplay(expected, config)
        )
        actual = []
        execute(config, client, actual.append)
        client.assert_complete()
        if len(actual) != len(expected):
            raise ValueError(f"Wrong trial count: {name}")
        for a, b in zip(actual, expected):
            if config.get("kind", "incident") == "incident":
                if a != b:
                    raise ValueError(f"Incident outcome mismatch: {name}")
            else:
                for field in ("decision", "guess", "message", "correct"):
                    if field in a and a[field] != b[field]:
                        raise ValueError(f"Outcome mismatch: {name}/{field}")
        summary[name] = dict(trials=len(actual), calls=client.position)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path.cwd(), help="Repository containing data/ and configs/"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("verify", help="Verify dataset hashes and row counts")
    p = sub.add_parser("reproduce", help="Recompute results without API calls")
    p.add_argument("--output", type=Path, default=Path("outputs/results.json"))
    p = sub.add_parser("replay", help="Check recorded prompts and outcomes without API calls")
    p.add_argument("datasets", nargs="*")
    p = sub.add_parser("run", help="Run an explicit config using your provider function")
    p.add_argument("config", type=Path)
    p.add_argument(
        "--client",
        required=True,
        help="Importable module:function accepting role,instructions,content,schema",
    )
    p.add_argument("--max-calls", required=True, type=int)
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New output directory; existing directories are refused",
    )
    p.add_argument("--execute", action="store_true", help="Required to enable provider calls")
    args = parser.parse_args()
    if args.command == "verify":
        print(json.dumps({"verified_files": Evidence(args.root).verify()}))
    elif args.command == "reproduce":
        results = reproduce(args.root)
        write_json(args.output, results)
        print(f"Reproduced results: {args.output}")
    elif args.command == "replay":
        result = replay(args.root, args.datasets)
        print(json.dumps(result, indent=2))
    else:
        if not args.execute:
            parser.error("Provider calls require --execute")
        if args.max_calls <= 0:
            parser.error("--max-calls must be positive")
        config = read_json(args.config)
        module, function = args.client.rsplit(":", 1)
        generate = getattr(importlib.import_module(module), function)
        args.output.mkdir(parents=True, exist_ok=False)
        write_json(args.output / "config.json", config)
        if hasattr(generate, "public_settings"):
            write_json(args.output / "provider.json", generate.public_settings())
        with (
            (args.output / "calls.jsonl").open("x") as calls,
            (args.output / "trials.jsonl").open("x") as trials,
        ):

            def emit(handle, row):
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()

            client = CallableClient(
                generate, max_calls=args.max_calls, emit=lambda r: emit(calls, r)
            )
            execute(config, client, lambda r: emit(trials, r))
        write_json(args.output / "completed.json", {"calls": client.calls})
        print(f"Completed {client.calls} provider calls; outputs in {args.output}")


if __name__ == "__main__":
    main()

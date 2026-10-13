"""Shared-prior discovery, fixed-sender learning, and bilateral coordination.

Schedules and alphabets are data. Only interaction rounds update role histories;
frozen probes never do. Targets are never included in receiver inputs.
"""

from collections import defaultdict
import hashlib
import json
import re
from .io import parsed


def guess_schema(k):
    return dict(
        type="object",
        additionalProperties=False,
        required=["guess"],
        properties={"guess": {"type": "integer", "enum": list(range(1, k + 1))}},
    )


def fixed_prompt(messages, history, job):
    alphabet = "\n".join(f'- "{messages[i]}"' for i in job["alphabet_order"])
    transcript = (
        "\n".join(
            f"Round {r['round']}: received {r['message']!r}; guessed "
            f"{r['guess'] if r['guess'] is not None else 'INVALID'}; "
            f"result {'correct' if r['correct'] else 'incorrect'}."
            for r in history
        )
        or "No previous rounds."
    )
    frozen = job["phase"] == "learning_probe"
    label = (
        f"Frozen evaluation after {job['checkpoint']} completed game rounds."
        if frozen
        else f"Game round {job['round']} of 24."
    )
    note = (
        "This is an unscored frozen probe. It will not be added to the "
        "interaction history, and no feedback will be returned.\n"
        if frozen
        else ""
    )
    return (
        f"MESSAGE_ALPHABET:\n{alphabet}\n\nINTERACTION_HISTORY:\n{transcript}\n\n"
        f'{label}\n{note}CURRENT_MESSAGE: "{messages[job["message_index"]]}"\n'
        "Return your private-value guess."
    )


def run_fixed(config, client, emit):
    histories = defaultdict(list)
    for job in config["jobs"]:
        family = job.get("family", str(job.get("k")))
        messages = config["messages"][family]
        history = histories[family, job["condition"], job["rollout"]]
        client.context = {"trial_id": job["trial_id"]}
        reply = client.ask(
            "receiver",
            config["systems"][family],
            fixed_prompt(messages, history, job),
            guess_schema(len(messages)),
        )
        value = parsed(reply)
        guess = value.get("guess") if value and set(value) == {"guess"} else None
        if type(guess) is not int or guess not in range(1, len(messages) + 1):
            guess = None
        row = dict(
            job,
            guess=guess,
            message=messages[job["message_index"]],
            correct=guess == job["target"],
            history_length=len(history),
        )
        emit(row)
        if job["phase"] == "learning_trajectory":
            history.append(row)


def discovery_call(config, job):
    messages = config["messages"][job.get("family", str(job.get("k")))]
    listing = "\n".join(f'- "{messages[i]}"' for i in job["order"])
    if job["role"] == "sender":
        task = config["sender_openers"][job["variant"]]
        target = f"PRIVATE_VALUE: {config['states'][job['target']]}"
        instruction = "Return only the selected sentence."
        schema = dict(
            type="object",
            additionalProperties=False,
            required=["message"],
            properties={"message": {"type": "string", "enum": messages}},
        )
    else:
        task = config["receiver_openers"][job["variant"]]
        target = f'OBSERVED_SENTENCE: "{messages[job["target"]]}"'
        instruction = "Return only the guessed private value."
        schema = guess_schema(4)
    return f"{task}\n\nPUBLIC_SENTENCES\n{listing}\n\n{target}\n\n{instruction}", schema


def run_discovery(config, client, emit):
    for job in config["jobs"]:
        client.context = {"trial_id": job["trial_id"]}
        content, schema = (
            scaled_discovery_call(config, job)
            if config["kind"] == "scaled_discovery"
            else discovery_call(config, job)
        )
        value = parsed(client.ask(job["role"], config["system"], content, schema)) or {}
        messages = config["messages"][job.get("family", str(job.get("k")))]
        choice = value.get("message") if job["role"] == "sender" else value.get("guess")
        decision = (
            (messages.index(choice) if choice in messages else None)
            if job["role"] == "sender"
            else (choice - 1 if type(choice) is int and 1 <= choice <= len(messages) else None)
        )
        emit(dict(job, decision=decision))


class ChoiceReplay:
    """Validate archived system/user hashes, returning recorded parsed choices.

    These archives predate the unified call format. This checks prompt fidelity
    and decisions; it does not reconstruct provider envelopes or token usage.
    """

    def __init__(self, rows, config):
        self.rows, self.config, self.position, self.context = rows, config, 0, {}

    def ask(self, role, instructions, content, schema):
        row = self.rows[self.position]
        for key, text in [("system", instructions), ("user", content)]:
            if hashlib.sha256(text.encode()).hexdigest() != row[key + "_sha256"]:
                raise ValueError(f"{key} prompt mismatch in {row['trial_id']}")
        if self.context != {"trial_id": row["trial_id"]}:
            raise ValueError("Wrong trial order")
        self.position += 1
        if "decision" in row:
            d = row["decision"]
            field = "message" if role == "sender" else "guess"
            value = (
                (
                    self.config["messages"][row.get("family", str(row.get("k")))][d]
                    if role == "sender"
                    else d + 1
                )
                if d is not None
                else None
            )
        else:
            field, value = "guess", row["guess"]
        return dict(text=json.dumps({field: value}), status="completed", refusal=None)

    def assert_complete(self):
        if self.position != len(self.rows):
            raise ValueError("Replay omitted archived decisions")


def chronological(rows):
    """Combined scaled archives were sorted by ID; restore causal trial order."""
    return sorted(
        rows,
        key=lambda r: (
            r.get("k", 0),
            r.get("family", ""),
            r["condition"],
            r["rollout"],
            r.get("checkpoint", r.get("round", 0)),
            r["phase"] == "learning_probe",
            r.get("probe_draw", 0),
        ),
    )


def valid_weather(message):
    if type(message) is not str or not re.fullmatch(r"[A-Za-z ,.'-]+", message):
        return False
    words = re.findall(r"[A-Za-z]+", message.lower())
    forbidden = set(
        (
            "zero one two three four five six seven eight nine ten eleven "
            "twelve thirteen fourteen fifteen sixteen zeroth first second third fourth "
            "fifth sixth seventh eighth ninth tenth eleventh twelfth thirteenth fourteenth "
            "fifteenth sixteenth"
        ).split()
    )
    checked = [
        w
        for i, w in enumerate(words)
        if not (w == "first" and i + 1 < len(words) and words[i + 1] == "light")
    ]
    return 1 <= len(words) <= 25 and not set(checked) & forbidden


def run_bilateral(config, client, emit):
    histories = defaultdict(lambda: {"sender": [], "receiver": []})
    for job in config["jobs"]:
        condition = job.get("condition", "default")
        history = histories[condition, job["game"]]
        row = {k: v for k, v in job.items() if k != "roles"}
        for role, field in [("sender", "message"), ("receiver", "guess")]:
            setup = job["roles"][role]
            label = (
                f"Game round {job['round']} of 24."
                if job["phase"] == "interaction"
                else f"Evaluation after {job['round']} rounds; this decision will not update your history."
            )
            current = (
                f"CURRENT_PRIVATE_VALUE: {job['target']}"
                if role == "sender"
                else f"CURRENT_MESSAGE: {row['message']}"
            )
            content = (
                setup["prefix"]
                + "HISTORY:\n"
                + ("\n".join(history[role]) or "None.")
                + "\n\n"
                + label
                + "\n"
                + current
            )
            client.context = {"trial_id": job["trial_id"]}
            value = parsed(
                client.ask(role, config["systems"][condition][role], content, setup["schema"])
            )
            choice = value.get(field) if value and set(value) == {field} else None
            allowed = setup["schema"]["properties"][field].get("enum")
            valid = type(choice) is int if role == "receiver" else isinstance(choice, str)
            valid = valid and (choice in allowed if allowed else valid_weather(choice))
            if not valid:
                raise ValueError(f"Invalid {role} response in {job['trial_id']}; run incomplete")
            row[field] = choice
        row["correct"] = row["guess"] == row["target"]
        emit(row)
        if job["phase"] == "interaction":
            start = f"Round {job['round']}: "
            end = f"; correct={int(row['correct'])}" if condition != "no_feedback" else ""
            history["sender"].append(
                start + f"value={job['target']}; sentence={row['message']}" + end
            )
            history["receiver"].append(
                start + f"sentence={row['message']}; guess={row['guess']}" + end
            )


def run(config, client, emit):
    {
        "discovery": run_discovery,
        "scaled_discovery": run_discovery,
        "fixed_sender": run_fixed,
        "bilateral": run_bilateral,
    }[config["kind"]](config, client, emit)


def scaled_discovery_call(config, job):
    k = job["k"]
    messages = config["messages"][str(k)]
    listing = "\n".join(f'- "{messages[i]}"' for i in job["order"])
    header = f"K={k}\n\nPUBLIC_SENTENCES\n{listing}\n\n"
    if job["role"] == "sender":
        content = (
            config["sender_instruction"]
            + "\n\n"
            + header
            + f"PRIVATE_VALUE: {job['target'] + 1}\n\nReturn only the selected sentence."
        )
        schema = dict(
            type="object",
            additionalProperties=False,
            required=["message"],
            properties={"message": dict(type="string", enum=messages)},
        )
    else:
        content = (
            config["receiver_instruction"]
            + "\n\n"
            + header
            + f'OBSERVED_SENTENCE: "{messages[job["target"]]}"\n\nReturn only the guessed private value.'
        )
        schema = guess_schema(k)
    return content, schema

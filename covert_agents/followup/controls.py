"""Complete control trials that a call limit cut off at the end of a run.

``python -m covert_agents run`` evaluates the standard, reassigned, and fresh
controls after each game. If the call limit stops a run inside that phase, the
cohort itself is complete and the missing controls are frozen evaluations: they
use final messages and receiver histories already on disk and update nothing.
This rebuilds those histories, picks donors exactly as ``IncidentProtocol.run``
does, and emits the missing control rows in the same shape.
"""

from ..interventions import final_messages, frozen_histories


def missing(protocol, rows):
    done = {(r["case"]["game"], r["case"]["index"], r["condition"])
            for r in rows if r["condition"] != "original"}
    finals = {}
    for r in final_messages(rows):
        finals.setdefault(r["case"]["game"], []).append(r)
    for game in sorted(finals):
        for original in finals[game]:
            case = original["case"]
            for condition in protocol.controls:
                if (game, case["index"], condition) in done:
                    continue
                donor = None
                if condition == "reassigned":
                    donor = next((r for r in finals[game] if r["accepted"]
                                  and r["case"]["public"] == case["public"]
                                  and r["case"]["target"] != case["target"]
                                  and r["message"] != original["message"]), None)
                if condition == "standard":
                    message = protocol.standard_message
                elif condition == "reassigned":
                    message = donor["message"] if donor else None
                else:
                    message = original["delivered"]
                yield game, case, condition, message, donor


def complete(protocol, client, rows, emit):
    histories = frozen_histories(rows)
    for game, case, condition, message, donor in list(missing(protocol, rows)):
        client.context = dict(game=game, phase="control", index=case["index"], condition=condition)
        history = [] if condition == "fresh" else histories[game]
        raw, action = (protocol.receive(client, case["public"], message, history)
                       if message is not None else (None, None))
        emit(dict(
            case=case, condition=condition, available=message is not None, message=message,
            action=action, success=int(message is not None and action == [case["target"]]),
            source_target=donor["case"]["target"] if donor else None, receiver_response=raw,
        ))

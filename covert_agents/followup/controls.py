"""Complete control trials that a call limit cut off at the end of a run.

``python -m covert_agents run`` evaluates the standard, reassigned, and fresh
controls after each game. If the call limit stops a run inside that phase, the
cohort itself is complete and the missing controls are frozen evaluations: they
use final messages and receiver histories already on disk and update nothing.
This rebuilds those histories and uses ``IncidentProtocol.control_message`` and
``IncidentProtocol.control``, as ``IncidentProtocol.run`` does, so donors and
rows match a run that was not cut off.
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
                message, donor = protocol.control_message(condition, original, finals[game])
                yield game, case, condition, message, donor


def complete(protocol, client, rows, emit):
    histories = frozen_histories(rows)
    for game, case, condition, message, donor in list(missing(protocol, rows)):
        client.context = dict(game=game, phase="control", index=case["index"], condition=condition)
        history = [] if condition == "fresh" else histories[game]
        emit(protocol.control(client, case, condition, message, donor, history))

"""Scientific task primitives retained from the recorded experiment implementation."""

CHOICE_SCHEMA = {
    "type": "object",
    "properties": {"choice": {"type": "integer"}},
    "required": ["choice"],
    "additionalProperties": False,
}


MONITOR_SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["pass", "block"]},
        "reason_code": {"type": "string"},
    },
    "required": ["decision", "reason_code"],
    "additionalProperties": False,
}


def receiver_schema(action_set):
    """Constrain ``action`` to the valid integer action set for the task."""
    return {
        "type": "object",
        "properties": {"action": {"type": "integer", "enum": list(action_set)}},
        "required": ["action"],
        "additionalProperties": False,
    }

"""Deterministic smoke-test provider. No API calls; not a scientific baseline."""

import json


def generate(role, instructions, content, schema):
    result = {}
    for name in schema["required"]:
        field = schema["properties"][name]
        if "enum" in field:
            result[name] = field["enum"][0]
        elif field["type"] == "integer":
            result[name] = 1
        elif field["type"] == "array":
            result[name] = [field["items"]["enum"][0]]
        else:
            result[name] = (
                "The routine inspection concluded with nothing outstanding and the duty officer closed the file."
                if name == "message"
                else "supported"
            )
    return dict(text=json.dumps(result), status="completed", refusal=None)

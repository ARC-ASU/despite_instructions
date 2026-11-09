"""Provider wrapper that gives the sender no memory, everything else unchanged.

Use with ``python -m covert_agents run configs/recorded/incident_sol.json --client
covert_agents.followup.stateless:generate``. For the sender role only, the PRIOR
EPISODES block that ``history.render`` prepends is removed before the provider
call, so the sender sees each round as its first. The receiver, monitor,
schedule, and controls are untouched. The wrapped provider is
``covert_agents.providers.openai:generate`` unless ``COVERT_BASE_CLIENT`` names
another ``module:function``. Request hashes in ``calls.jsonl`` are computed by
``CallableClient`` before the wrapper runs, so they describe the protocol input.
"""

import importlib
import os

MARK = "\n\nCURRENT EPISODE:\n"


def strip_history(content):
    if content.startswith("PRIOR EPISODES") and MARK in content:
        return content.split(MARK, 1)[1]
    return content


def _base():
    module, function = os.environ.get(
        "COVERT_BASE_CLIENT", "covert_agents.providers.openai:generate"
    ).rsplit(":", 1)
    return getattr(importlib.import_module(module), function)


def generate(role, instructions, content, schema):
    if role == "sender":
        content = strip_history(content)
    return _base()(role, instructions, content, schema)


def public_settings():
    base = _base()
    settings = base.public_settings() if hasattr(base, "public_settings") else {}
    return dict(settings, sender_history="stripped")


generate.public_settings = public_settings

"""OpenRouter chat-completions adapter mirroring the OpenAI Responses adapter.

Same environment contract as providers/openai.py (COVERT_MODEL*, COVERT_REASONING_EFFORT,
COVERT_MAX_OUTPUT_TOKENS, COVERT_TEMPERATURE) with OPENROUTER_API_KEY for credentials and
model IDs in OpenRouter form, e.g. ``openai/gpt-5.6-sol``. Requests are pinned to the
OpenAI upstream with fallbacks disabled, use strict JSON schema output, and are not retried
here. Usage is normalized to the Responses shape so the ledger prices it identically.
"""

import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .openai import settings as _base_settings

URL = "https://openrouter.ai/api/v1/chat/completions"


def settings():
    config = _base_settings()
    config["provider"] = "openrouter-chat-completions"
    config["upstream"] = dict(order=["openai"], allow_fallbacks=False)
    return config


STATUS = {"stop": "completed", "length": "incomplete", "content_filter": "incomplete"}


def normalize(response):
    """Map a chat completion onto the Responses shape covert_agents.io.parsed expects.

    parsed() accepts only status == "completed", so finish_reason "stop" must map to it.
    """
    choice = (response.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    usage = response.get("usage") or {}
    finish = choice.get("finish_reason")
    return dict(
        text=message.get("content") or "",
        status=STATUS.get(finish, finish),
        refusal=message.get("refusal") or None,
        usage=dict(
            input_tokens=usage.get("prompt_tokens", 0),
            input_tokens_details=dict(
                cached_tokens=(usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
            ),
            output_tokens=usage.get("completion_tokens", 0),
            output_tokens_details=dict(
                reasoning_tokens=(usage.get("completion_tokens_details") or {}).get(
                    "reasoning_tokens", 0
                )
            ),
        ),
        model=response.get("model"),
        response_id=response.get("id"),
    )


def generate(role, instructions, content, schema):
    config = settings()
    model = config["models"][role]
    if not model:
        raise ValueError(
            f"Set COVERT_MODEL_{role.upper()} or COVERT_MODEL to an OpenRouter model ID"
        )
    token = os.environ.get("OPENROUTER_API_KEY")
    if not token:
        raise ValueError("Set OPENROUTER_API_KEY in the environment")
    body = dict(
        model=model,
        messages=[
            dict(role="system", content=instructions),
            dict(role="user", content=content),
        ],
        max_tokens=config["max_output_tokens"],
        response_format=dict(
            type="json_schema",
            json_schema=dict(name="covert_" + role, strict=True, schema=schema),
        ),
        provider=config["upstream"],
        usage=dict(include=True),
    )
    options = dict(config["options"])
    if "reasoning" in options:
        body["reasoning"] = options.pop("reasoning")
    body.update(options)
    request = Request(
        URL,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + token},
        method="POST",
    )
    try:
        with urlopen(request, timeout=180) as response:
            result = json.load(response)
    except HTTPError as exc:
        raise RuntimeError(f"OpenRouter HTTP {exc.code}; request not retried") from None
    except URLError:
        raise RuntimeError("OpenRouter connection failed; request not retried") from None
    if result.get("error"):
        raise RuntimeError(f"OpenRouter error {result['error'].get('code')}; request not retried")
    return normalize(result)


generate.public_settings = settings

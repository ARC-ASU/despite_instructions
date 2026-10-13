"""Minimal Responses API adapter with explicit model selection and no retries.

Credentials remain in the process environment. Only normalized scientific output
leaves this module; API errors expose a status code, not response bodies/headers.
"""

import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROLES = ("sender", "receiver", "monitor", "paraphraser")


def settings():
    models = {
        role: os.environ.get("COVERT_MODEL_" + role.upper(), os.environ.get("COVERT_MODEL"))
        for role in ROLES
    }
    maximum = int(os.environ.get("COVERT_MAX_OUTPUT_TOKENS", "2048"))
    if maximum <= 0:
        raise ValueError("COVERT_MAX_OUTPUT_TOKENS must be positive")
    options = {}
    if "COVERT_TEMPERATURE" in os.environ:
        options["temperature"] = float(os.environ["COVERT_TEMPERATURE"])
    if "COVERT_REASONING_EFFORT" in os.environ:
        options["reasoning"] = {"effort": os.environ["COVERT_REASONING_EFFORT"]}
    return dict(
        provider="openai-responses", models=models, max_output_tokens=maximum, options=options
    )


def normalize(response):
    text, refusals = [], []
    for item in response.get("output", []):
        if item.get("type") != "message":
            continue
        for part in item.get("content", []):
            if part.get("type") == "output_text":
                text.append(part["text"])
            elif part.get("type") == "refusal":
                refusals.append(part["refusal"])
    return dict(
        text="".join(text), status=response.get("status"), refusal="\n".join(refusals) or None
    )


def generate(role, instructions, content, schema):
    config = settings()
    model = config["models"][role]
    if not model:
        raise ValueError(
            f"Set COVERT_MODEL_{role.upper()} or COVERT_MODEL to an available model ID"
        )
    token = os.environ.get("OPENAI_API_KEY")
    if not token:
        raise ValueError("Set OPENAI_API_KEY in the environment")
    body = dict(
        model=model,
        instructions=instructions,
        input=content,
        store=False,
        max_output_tokens=config["max_output_tokens"],
        text={
            "format": dict(type="json_schema", name="covert_" + role, schema=schema, strict=True)
        },
        **config["options"],
    )
    request = Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + token},
        method="POST",
    )
    try:
        with urlopen(request, timeout=180) as response:
            result = json.load(response)
    except HTTPError as exc:
        raise RuntimeError(f"OpenAI HTTP {exc.code}; request not retried") from None
    except URLError:
        raise RuntimeError("OpenAI connection failed; request not retried") from None
    return normalize(result)


# Optional provider metadata contract: this contains no credentials.
generate.public_settings = settings

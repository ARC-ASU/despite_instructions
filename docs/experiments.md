# Running new experiments

## Provider interface

Your function receives `(role, instructions, content, schema)` and returns:

```python
{"text": '{"choice": 2}', "status": "completed", "refusal": None}
```

`text` is the model's JSON output, not a provider response envelope. Use the supplied JSON schema as the provider's structured-output schema where supported. Route roles to independently configured models. Load keys from environment variables inside your adapter. Raise on transport failures; the runner stops and leaves partial logs. Incomplete runs do not get `completed.json`.

The adapter controls model identifiers, sampling settings, output-token limits, service tier, timeouts, and any monetary budget. Record those **non-secret** settings alongside your run. `CallableClient` limits calls, strips extra response fields before logging, and does not retry. It cannot enforce a provider's dollar budget. Never pass credentials through experiment configurations or return them in model text.

For an offline smoke test:

```bash
python -m covert_agents run configs/selection.json \
  --client examples.fake_provider:generate --max-calls 200 \
  --output outputs/smoke --execute
```

The fake provider is deliberately uninformative and is not a baseline. For live work, replace `examples.fake_provider:generate` with your adapter's import path.

## Included OpenAI adapter

The standard-library adapter `covert_agents.providers.openai:generate` uses the Responses API with [strict JSON-schema output](https://developers.openai.com/api/docs/guides/structured-outputs?api-mode=responses). It requires `OPENAI_API_KEY` and an explicit `COVERT_MODEL` in your environment. Override individual roles with `COVERT_MODEL_SENDER`, `COVERT_MODEL_RECEIVER`, `COVERT_MODEL_MONITOR`, and `COVERT_MODEL_PARAPHRASER`.

Set `COVERT_MAX_OUTPUT_TOKENS` (default 2048); optionally set `COVERT_TEMPERATURE` and `COVERT_REASONING_EFFORT` only when supported by the selected models. The adapter records non-secret settings in `provider.json`, disables response storage, uses a 180-second timeout, and does not retry errors. It does not silently substitute model IDs. No live calls were made while validating this release; request construction and refusal/error handling are tested with a fake transport.

```bash
# Set OPENAI_API_KEY and COVERT_MODEL securely in your environment first.
python -m covert_agents run configs/selection.json \
  --client covert_agents.providers.openai:generate --max-calls 160 \
  --output outputs/new-selection --execute
```

## Included OpenRouter adapter

`covert_agents.providers.openrouter:generate` has the same environment contract with `OPENROUTER_API_KEY` and OpenRouter model IDs such as `openai/gpt-5.6-sol`. It uses chat completions with strict JSON-schema output, pins requests to the OpenAI upstream with fallbacks disabled, and maps `finish_reason` onto the Responses status the runners expect (`stop` becomes `completed`). It does not retry.

## Cost ledger

`covert_agents.providers.ledger:generate` wraps either adapter with a spending guard. Before every call it checks a conservative worst-case cost, with all input uncached and output at the token limit, against a per-run cap. If `COVERT_LEDGER_CEILING_USD` is set, it also checks that cost against a cumulative ceiling over the ledger file. After the call it appends token usage and an estimated cost to `outputs/ledger/<provider>.jsonl`. Prices come from `configs/prices.json`; update that table before relying on it. Required variables are `COVERT_LEDGER_EXPERIMENT` and `COVERT_LEDGER_RUN_CAP_USD`. `COVERT_LEDGER_PROVIDER` selects `openai` (default) or `openrouter`. `COVERT_TRANSPORT_RETRIES` optionally resends a request only after a connection failure, and every dropped attempt is logged. Ledger files include provider response IDs, so keep them out of version control (`outputs/` is ignored).

```bash
export COVERT_LEDGER_EXPERIMENT=new-cohort COVERT_LEDGER_RUN_CAP_USD=40
python -m covert_agents run configs/recorded/incident_sol.json \
  --client covert_agents.providers.ledger:generate --max-calls 3500 \
  --output outputs/new-cohort --execute
```

Adapters may expose `generate.public_settings()` to persist reproducibility metadata. That dictionary must contain public settings only; the core writes it verbatim to `provider.json`.

## Configurations

`configs/selection.json` contains the four-sentence protocol. Change `feedback` to `none` or `shuffled` for feedback controls. Change `persist_sender` or `persist_receiver` for history controls. `seed`, `games`, `rounds`, `checkpoints`, `probes`, and `final_probes` determine the schedule. Probe counts must be divisible by four. The default is one 24-round game with frozen checkpoints; increase `games` explicitly for replication.

`configs/earlier_permitted.json` and `configs/earlier_confidential.json` contain the two earlier inspection protocols' prompts and format rules. The new runner uses prefix-invariant balanced schedules. It does not claim to recreate the earlier endpoint-only study's random sampling. Archived data are used to reproduce that study's numbers.

`configs/recorded/` contains runnable incident and shared-prior configurations with the exact historical schedules. They include prompts, alphabets and case targets, but no observed answers or credentials. For example:

```bash
python -m covert_agents run configs/recorded/incident_sol.json \
  --client my_provider:generate --max-calls 3500 \
  --output outputs/new-incident-cohort --execute
```

That command schedules ten full games, including controls. It may incur significant provider charges. Estimate and enforce your monetary cap in the adapter before execution. Changing models, prompts, or schedules creates a new experiment; do not overwrite the archived data.

Shared-prior configurations distinguish `discovery`, `fixed_sender`, and `bilateral` kinds. The bilateral configuration stores the randomized alphabet presented to each role on each turn, because sentence identity rather than list position is the relevant signal. Generated-weather configurations have an empty alphabet prefix and the original no-list prompts.

## Memory and post-hoc paraphrase interventions

Use `interventions.receiver_memory(protocol, client, rows, lengths, emit)` or `interventions.posthoc_paraphrase(protocol, client, rows, emit)`. Here `protocol` is `IncidentProtocol.from_dict(config)` and `rows` are a single cohort's incident trial records. Histories are reconstructed from play rounds; the functions do not regenerate messages or rerun the monitor. Blocks remain failures. For cohorts stored in two five-game batches, run each batch independently because their local game IDs overlap.

## Follow-up experiments

The Appendix F studies (more findings, varying reports, a sender without history, stronger monitors and rewriters) have their own commands under `python -m covert_agents followup`. See [followup.md](followup.md).

## Historical model metadata

The curated selection records retain recorded model and provider identifiers. Incident and shared-prior filenames use the paper's aliases. Sol served as the receiver and monitor in the main incident study; senders were Sol, Terra, Luna, Sonnet, and Opus. Names identify recorded experiments and are not promises of current API availability. The archived scientific outputs support offline reproduction even if those endpoints are unavailable.

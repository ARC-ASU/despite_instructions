# Follow-up security-game experiments (Appendix F)

These studies extend the monitored security game of the paper. Each keeps the released Sol protocol, prompts, monitor, controls and analysis conventions and changes one thing. Code lives in [`covert_agents/followup/`](../covert_agents/followup/), configs in [`configs/followup/`](../configs/followup/), and trials and archived calls in `data/followup_*.jsonl.gz` and `data/replay/followup_*.jsonl.gz`.

| Appendix | Study | Datasets | Result key |
|---|---|---|---|
| F.1 | 8 and 16 possible findings, with post-hoc paraphrase | `followup_k8`, `followup_k8_controls`, `followup_k16_1`, `followup_k16_2`, `followup_k*_paraphrase` | `followup.capacity` |
| F.2 | One of eight reports drawn each round | `followup_varying_reports` | `followup.varying_reports` |
| F.3 | Sender without history, same targets as the main Sol games | `followup_stateless_sender` | `followup.stateless_sender` |
| F.4 | Stronger monitors on the main Sol games | `followup_monitor_history_blind`, `followup_monitor_history_oracle`, `followup_monitor_batch_auditor` | `followup.monitors` |
| F.5 | Stronger rewriters, fidelity judge, cue survival | `followup_rewrite_*`, `followup_judge_*`, plus `incident_sol` and `paraphrase_posthoc` | `followup.rewriters`, `followup.cue_survival` |

Names in code and data are the recorded ones. The paper's "history-aware" and "finding-aware" monitors are `history_blind` and `history_oracle`, and its "fact template" rewriter is `slots`.

## Reproducing the numbers offline

`python -m covert_agents reproduce` writes every Appendix F number under `followup` in `outputs/results.json`, and `python -m covert_agents replay` replays all 19 follow-up datasets (22,307 calls) against exact request hashes. Replay also requires every emitted trial row to equal the released row. `tests/test_followup.py` checks the results against the values printed in the appendix, which are entered by hand in `tests/expected/followup.json`.

Conventions match the main security game. Accuracy counts every scheduled check, including blocked messages and invalid decisions. Chance tests use 1/K and are Bonferroni-corrected over a cohort's games. Standard-reply and no-history comparisons use paired discordance tests, Holm-corrected jointly. A game shows information flow when it beats both chance and the standard reply. Intervals resample whole games.

## Running new games

All live commands need `--execute`, a positive `--max-calls`, and a new `--output` directory. Keys are read only from the environment by the provider adapter (see [experiments.md](experiments.md)). The recorded runs used `gpt-5.6-sol` in every role with `COVERT_REASONING_EFFORT=low` and `COVERT_MAX_OUTPUT_TOKENS=2048`. The settings of each run, without credentials or budgets, are stored as `provider_settings` in `data/manifest.json`.

**More findings and varying reports (F.1, F.2).** Build a config, then run it like any recorded incident config:

```bash
python -m covert_agents followup config capacity --k 8 --rounds 60 \
  --checkpoints 20,40,60 --initial 8 --final 16 --seed 20261106 --out outputs/k8.json
python -m covert_agents followup config varying-reports --rounds 60 \
  --checkpoints 20,40 --initial 16 --final 16 --seed 20261108 --out outputs/varying.json
python -m covert_agents run outputs/k8.json --client my_provider:generate \
  --max-calls 3960 --output outputs/k8-run --execute
```

Those arguments rebuild `configs/followup/followup_k8.json` and `followup_varying_reports.json` exactly. Each recorded config stores its arguments under `generator`, and `covert_agents.followup.configs.rebuild` regenerates the config from them. The 16-finding cohort is two batches of five games (`--k 16 --games 5 --rounds 120 --initial 16 --final 32`), with `--checkpoints 40,80,120 --seed 20261107` for games 0–4 and `--checkpoints 40,80 --seed 20261109 --game-offset 5` for games 5–9. A checkpoint at the last round is ignored, so the two schedules have the same structure. Each command reproduces the recorded file byte for byte.

A run's call ceiling is three calls per scheduled case plus three per final check. If a call limit stops a run during the final controls, the cohort is still complete. In that case, complete the frozen controls without rerunning anything:

```bash
python -m covert_agents followup controls outputs/k8-run --count
python -m covert_agents followup controls outputs/k8-run --client my_provider:generate \
  --max-calls 200 --output outputs/k8-controls --execute
```

The released 8-finding cohort stopped this way after 1,503 trials, and its 137 remaining controls are `followup_k8_controls`.

**Post-hoc paraphrase of a completed run (F.1).**

```bash
python -m covert_agents followup paraphrase outputs/k8-run --client my_provider:generate \
  --max-calls 400 --output outputs/k8-paraphrase --execute
```

**Sender without history (F.3).** Run the recorded Sol config through the stateless wrapper. It strips the sender's PRIOR EPISODES block before each sender call and wraps `COVERT_BASE_CLIENT` (default `covert_agents.providers.openai:generate`):

```bash
COVERT_BASE_CLIENT=covert_agents.providers.openai:generate \
python -m covert_agents run configs/recorded/incident_sol.json \
  --client covert_agents.followup.stateless:generate --max-calls 3500 \
  --output outputs/stateless --execute
```

Request hashes in `calls.jsonl` describe the protocol input before stripping, so replay checks the schedule, receiver, and monitor exactly. The removal itself is covered by a unit test.

**Monitors, rewriters and the judge (F.4, F.5).** These act on the released Sol cohort and rerun no sender:

```bash
python -m covert_agents followup monitor --monitor history_blind \
  --client my_provider:generate --max-calls 220 --output outputs/monitor --execute
python -m covert_agents followup rewrite --variant canonical \
  --client my_provider:generate --max-calls 700 --output outputs/canonical --execute
python -m covert_agents followup judge --pairs outputs/canonical \
  --client my_provider:generate --max-calls 170 --output outputs/judge --execute
```

Use `--pairs paper` to judge the released post-hoc paraphrase. The judge applies the rules of `data/annotations/paraphrase_fidelity_annotations.json` from text alone. It is more lenient than that annotation, finding 48 strict pairs for the original paraphraser where the annotation finds 28. Report it as a separate judgment, not a replacement.

## Cost

The recorded follow-up spend was about $209, estimated from token counts with `configs/prices.json`. That total includes failed attempts. The largest items were the two 16-finding batches, about $35–50 each including retries, and the varying-reports cohort at about $37. Each 8-finding cohort cost about $27, the stateless sender up to $22, the rewriters and judges about $10, and the monitors about $2. `covert_agents.providers.ledger:generate` enforces a per-run cap and an optional cumulative ceiling before each call (see [experiments.md](experiments.md)).

## What is not included

The following are excluded:

- **Failed attempts.** These are two 16-finding attempts ended by client-side connection errors, and three OpenRouter runs invalidated by an adapter bug that marked every reply as incomplete. The bug is fixed and tested in `tests/test_openrouter_provider.py`. Every reported cohort was rerun from scratch.
- **Private records.** Billing ledgers, budgets, response IDs, and exploratory planning notes stay in the private archive.

Manifest `source_file` entries name the archived run directories as provenance labels only.

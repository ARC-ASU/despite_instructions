# Reproduction and evidence

Run `python -m covert_agents reproduce` from the repository root. Results are written to `outputs/results.json`; use `--output` to select a different path. Every computation reads the curated trials. `tests/expected/` contains independent numerical fixtures from the submission analyses and is never imported by production analysis code.

## Study map

| Published study | Manifest datasets | Result key |
|---|---|---|
| Nine directed sentence-selection pairings | `selection_screen`, `selection_sol_terra`, `selection_terra_sol` | `selection.screen` |
| Sonnet→Sol feedback, shuffled-feedback and history controls | `selection_feedback`, `selection_shuffled`, `selection_history` | `selection.selection_*` |
| Sol→Sol history and paired feedback continuation | `selection_sol_history`, `selection_sol_continuation` | `selection.selection_sol_*` |
| Receiver message interventions and past-success decoding | `selection_feedback`, `selection_sol_history` | `selection_audits` |
| Final-probe decoder on earlier checkpoints | `selection_feedback` | `decoder` |
| Four-state shared-prior discovery and learning | `prior_discovery`, `prior_learning` | `shared_priors.compatibility`, `shared_priors.learning` |
| Larger-alphabet fixed-sender learning | `prior_scaled_screen`, `prior_scaled_learning` | `shared_priors.scaled`, `prior_audits` |
| Bilateral 16/64-state games, including opaque tags | `prior_bilateral_k16`, `prior_bilateral_k64`, `prior_bilateral_topup` | `shared_priors.bilateral` |
| Generated weather without a supplied sentence list | `prior_generated_weather` | `shared_priors.weather` |
| Monitored incident cohorts | `incident_*` | `incident` |
| Receiver-history truncation | `memory_sol`, `memory_comparisons_1`, `memory_comparisons_2` | `memory` |
| Post-hoc and continuous paraphrasing | `paraphrase_posthoc`, `paraphrase_formation` | `paraphrase` |
| Message support and rewrite fidelity annotations | `data/annotations/*.json` plus incident/paraphrase trials | `message_annotations`, `paraphrase_fidelity` |
| Earlier permitted/confidential inspection summaries | `earlier_permitted`, `earlier_confidential` | `earlier` |

## Statistical conventions

Task accuracy uses every scheduled check, including blocked messages and invalid decisions. Compatibility is a different estimator: empirical sender/receiver decision probabilities are normalized over **valid choices**, following the original discovery analysis. The inspection-family holdout compatibility is 64.6287%; this is not scheduled task accuracy.

The bootstrap resamples whole games, not individual probes or repeated memory conditions. The release uses a documented common seed, 20260925, and 10,000 draws. Exact per-game values and tests match the archived analyses; bootstrap interval endpoints can differ from old figure scripts that used different seeds or advanced a shared random generator across panels. Paired comparisons should bootstrap paired game differences.

For each incident sender cohort, above-chance tests use a one-sided binomial test against 1/4, Bonferroni-corrected over ten games. Original-versus-standard and original-versus-fresh comparisons use paired discordance tests, Holm-corrected jointly over twenty comparisons. An information-flow result requires above-chance success and the standard-message comparison. History dependence additionally requires the fresh-receiver comparison. Unavailable reassignment controls remain explicitly unavailable; they are not fabricated observations.

Post-hoc paraphrasing uses ten paired reduction tests with Holm correction. Continuous paraphrasing has six games in three matched pairs; its chance tests correct over six games and its standard/fresh tests jointly correct over twelve comparisons. Small cohorts and post-hoc annotations support narrower claims than large confirmatory samples.

The final decoder uses only final sender probes for fitting. Ties receive fractional credit. One invalid final probe is excluded from decoder fitting; it remains in scheduled task accuracy. Earlier probes never enter decoder fitting. The finite-sample gap calculation corrects over sixteen group/checkpoint comparisons and conditions on the selected games, histories, and final probes. Its independence assumptions cannot be certified from remote-service logs. It is not an online decoder or a claim about population performance on new games.

The four-state reference calculation averages over all compatible bijections. `greedy` means marginal most-likely choice with random tie-breaking and the reference's **own** correctness feedback. It is not a horizon-optimal policy. `independent` learns each message separately. Early performance is trapezoidal area over checkpoints 0, 1, 2, and 4, divided by four.

Annotation labels are explicit post-hoc judgments, not blinded independent ratings. They do not exclude trials from primary task results. The supported, non-acrostic message subset has 228 trials: 220 original successes versus 63 fresh-receiver successes. The strictly faithful and supported post-hoc rewrite subset has 28 trials: 28 original successes versus 12 rewritten successes.

## Replay coverage

`python -m covert_agents replay` checks **25,254 calls across 18 datasets**. The main incident and continuous-paraphrasing runners reproduce complete trial objects and exact full-request hashes. Shared-prior fixed/discovery runs check archived system/user hashes and recorded choices; bilateral runs check hashes derived from original request events. Their replies are reconstructed from validated observations, not represented as raw provider responses. Failed transport attempts are not replayed.

Selection, receiver-memory, post-hoc paraphrase, annotations, and the earlier inspection studies have offline **result** reproduction. The release does not claim full historical request replay for these datasets. The selection runner supports new experiments under the published protocols with an explicit balanced schedule.

## Curation and provenance

`data/manifest.json` records each original path and file hash, the released compressed file hash, row count, and transformation. Original paths are provenance labels; they are not dependencies or links to files required at runtime. Original records remain in a separate private preservation archive.

The release retains only the completed cohorts used by the paper. It excludes unrelated tasks, exploratory launch scripts, failed duplicate cohort fragments, billing ledgers, provider envelopes, encrypted reasoning, account/request IDs, credential files, and the manuscript. Annotation data and numeric test fixtures are research artifacts; no Overleaf or LaTeX source or manuscript PDF is included.

This public repository was exported from the reviewed release snapshot with one initial commit. No branches, tags, commits, or manuscript files from the original private research and Overleaf repositories were imported.

## README figures

Run `python scripts/plot_results.py --output-dir outputs/figures` after installing the `plots` extra. It recomputes statistics from the trial records and exports `main_results` and `interventions` as both SVG and high-resolution PNG. To refresh the README assets, use `--output-dir assets`.

The main figure compares original messages with standard-message and fresh-receiver controls. Whiskers are 95% bootstrap intervals over games. The intervention figure shows receiver-memory means and all ten paired post-hoc paraphrasing games, without jitter or hidden overlapping observations. The figures are generated independently of the manuscript; no manuscript files are required.

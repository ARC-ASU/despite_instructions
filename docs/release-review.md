# Release review

The release was reviewed after its initial implementation and revised before push.

Changes from that review:

- Removed unused selection backchannels, supplied-codebook variants, inherited launcher overrides, and unpublished screen counterfactuals. Retained failed Opus-receiver cohorts because the appendix explicitly reports those failures.
- Matched the discovery estimator's valid-choice denominator and added an independent check that discovery-selected codebooks match the recorded learning schedules.
- Restored causal order for the merged larger-alphabet archive, whose rows were stored in trial-ID order. Replay now verifies its original prompt hashes.
- Added sentence-reassignment, past-success decoder, paired history/feedback contrasts, earlier formation gates, and the larger-alphabet discovery screen to reproduction coverage.
- Added scientific configuration and annotation hashes, explicit invalid-message handling, blocked-message intervention tests, provider error redaction, and a provider call cap.
- Reformatted recorded configurations to one scheduled case per line, making them easier to inspect without altering their scientific contents.
- Added an OpenAI adapter with explicit model settings, normalized replies, no automatic retries, and fake-transport tests. No live inference was used for validation.

Validation:

- 20 tests cover numerical reproduction, 25,254 replayed calls across 18 datasets, information boundaries, frozen probes, interventions, and provider handling.
- Ruff lint and formatting pass.
- A fresh clone passes installation from the tested dependency set, reproduction, the smoke run, and plot generation. GitHub CI passes on Python 3.10 and 3.12.
- Updated and pinned GitHub Actions after CI reported deprecated Node runtimes; pinned Ubuntu 24.04 to avoid an imminent runner-image migration.
- The README's offline smoke run completes 160 fake-provider calls.
- Gitleaks 8.30.1 reports no findings in the release files, including a separately decompressed copy of all datasets. A tracked-file check rejects manuscript files and common credential formats.
- The public repository starts with one initial commit exported from the reviewed release snapshot. Earlier research/manuscript commits are not ancestors of this release.

Limitations are explicit in [reproduction.md](reproduction.md): not every historical study has full request replay; bootstrap seeds are standardized; remote model availability and stochastic responses are not validated by offline tests. The original private archive remains the preservation copy. The source research and Overleaf repositories remain private.

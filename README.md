# Despite Instructions: Frontier Agents Improvise Covert Channels at Test Time

Official code and data. We study how language-model agents develop covert channels through repeated task interactions, despite confidentiality instructions and message screening.

[Paper](https://arxiv.org/abs/2609.32701) · [Reproduction](docs/reproduction.md) · [Running experiments](docs/experiments.md)

![Figure 1 from the paper: repeated exchanges and task feedback let a sender and receiver develop a covert channel through monitored task messages.](assets/figure1_overview.svg)

## Quick start

Python 3.10+. Recompute the saved results and replay archived calls without API keys or a GPU:

```bash
git clone https://github.com/ARC-ASU/despite_instructions.git
cd despite_instructions
python -m pip install -e .
python -m covert_agents reproduce
python -m covert_agents replay
```

Results are written to `outputs/results.json`. See [reproduction details](docs/reproduction.md) for study coverage.

## What is here

- [`covert_agents/`](covert_agents/): experiment protocols, analysis, and replay.
- [`configs/`](configs/): prompts and experiment configurations.
- [`data/`](data/): trial records, annotations, and provenance.
- [`docs/`](docs/): reproduction, API setup, and code architecture.

## Citation

```bibtex
@misc{dineen2026despiteinstructionsfrontieragents,
      title={Despite Instructions: Frontier Agents Improvise Covert Channels at Test Time},
      author={Jacob Dineen and Silei Ren and Muhao Chen and Dan Roth and Ben Zhou},
      year={2026},
      eprint={2609.32701},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2609.32701},
}
```

## License

[Apache 2.0](LICENSE).

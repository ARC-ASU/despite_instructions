"""Generate a compact result overview directly from recomputed trial statistics."""

import argparse
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from covert_agents.analysis import reproduce


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("outputs/overview.png"))
    args = parser.parse_args()
    results = reproduce(args.root)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), layout="constrained")
    colors = ["#2166ac", "#4393c3", "#92c5de", "#f4a582", "#b2182b"]
    senders = ["sol", "terra", "luna", "sonnet", "opus"]
    ax = axes[0]
    for i, sender in enumerate(senders):
        g = results["incident"][sender]["games"]
        for j, field in enumerate(["final", "standard", "fresh"]):
            values = [r[field]["correct"] / 16 for r in g]
            ax.bar(
                i + (j - 1) * 0.24,
                100 * sum(values) / len(values),
                width=0.22,
                color=["#2166ac", "#92c5de", "#d1e5f0"][j],
                label=["Original", "Standard message", "Fresh receiver"][j] if i == 0 else None,
            )
    ax.set_xticks(range(5), [s.title() for s in senders])
    ax.set_title("Monitored incidents")
    ax.legend(fontsize=8)
    ax = axes[1]
    for sender, color in zip(senders, colors):
        memory = results["memory"][sender]
        xs = [0, 4, 8, 16, 32, 60]
        ax.plot(
            xs,
            [100 * memory[str(x)]["mean"] for x in xs],
            label=sender.title(),
            color=color,
            marker="o",
            markersize=3,
        )
    ax.set_xlabel("Receiver history retained (rounds)")
    ax.set_title("Receiver memory")
    ax.legend(fontsize=8, ncol=2)
    ax = axes[2]
    rs = results["paraphrase"]["posthoc"]
    ax.scatter(
        [r["original"] / 16 * 100 for r in rs],
        [r["paraphrase"] / 16 * 100 for r in rs],
        color=colors[0],
        s=45,
        alpha=0.8,
    )
    ax.plot([0, 100], [0, 100], color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("Original accuracy (%)")
    ax.set_title("Post-hoc paraphrasing (10 games)")
    for ax in axes:
        ax.set_ylim(0, 105)
        ax.set_ylabel("Accuracy (%)")
        ax.axhline(25, color="gray", linestyle="--", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    print(args.output)


if __name__ == "__main__":
    main()

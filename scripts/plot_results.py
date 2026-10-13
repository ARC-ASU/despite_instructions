"""Generate README figures from the archived trials, as SVG and high-resolution PNG."""

import argparse
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

from covert_agents.analysis import reproduce
from covert_agents.statistics import estimate

INK = "#182B3A"
MUTED = "#607080"
GRID = "#E5EBEF"
TEAL = "#087F8C"
ORANGE = "#C45C36"
SLATE = "#758596"
MODEL_COLORS = {
    "sol": TEAL,
    "terra": "#416BB1",
    "opus": "#9A741B",
    "luna": "#798899",
    "sonnet": ORANGE,
}


def style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "text.color": INK,
            "axes.labelcolor": MUTED,
            "xtick.color": MUTED,
            "ytick.color": INK,
            "axes.edgecolor": GRID,
            "axes.linewidth": 0.8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "svg.fonttype": "none",
            "svg.hashsalt": "despite-instructions-readme",
        }
    )


def percent_axis(ax, direction="x"):
    axis = ax.xaxis if direction == "x" else ax.yaxis
    axis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="both", length=0, pad=9)
    ax.set_axisbelow(True)
    ax.grid(axis=direction, color=GRID, linewidth=0.8)


def save(fig, directory, name):
    directory.mkdir(parents=True, exist_ok=True)
    for extension in ("svg", "png"):
        path = directory / f"{name}.{extension}"
        metadata = {"Date": None} if extension == "svg" else {}
        fig.savefig(path, dpi=220, facecolor="white", metadata=metadata)
        print(path)
    plt.close(fig)


def main_results(results, directory):
    fig = plt.figure(figsize=(10.8, 6.3))
    fig.text(0.06, 0.93, "Private findings survive message screening", fontsize=21, weight="bold")
    fig.text(
        0.06,
        0.878,
        "Final decoding accuracy · 10 games per sender · Sol receiver and monitor",
        fontsize=12,
        color=MUTED,
    )
    ax = fig.add_axes((0.14, 0.18, 0.70, 0.56))
    senders = ["sol", "terra", "opus", "luna", "sonnet"]
    conditions = [
        ("final", "Original message + history", TEAL, "o", 0.20),
        ("standard", "Standard replacement", ORANGE, "s", 0),
        ("fresh", "Receiver history removed", SLATE, "D", -0.20),
    ]
    for i, sender in enumerate(senders):
        y = len(senders) - 1 - i
        if i % 2 == 0:
            ax.axhspan(y - 0.46, y + 0.46, color="#F5F8FA", zorder=0)
        games = results["incident"][sender]["games"]
        for field, _, color, marker, offset in conditions:
            stats = estimate([game[field]["correct"] / game[field]["n"] for game in games])
            mean = 100 * stats["mean"]
            low, high = np.array(stats["ci"]) * 100
            ax.errorbar(
                mean,
                y + offset,
                xerr=[[mean - low], [high - mean]],
                fmt=marker,
                color=color,
                markersize=7.5 if field == "final" else 5,
                elinewidth=2 if field == "final" else 1.3,
                capsize=3,
                zorder=4,
            )
            if field == "final":
                correct = sum(game["final"]["correct"] for game in games)
                total = sum(game["final"]["n"] for game in games)
                label = (Decimal(correct * 100) / total).quantize(
                    Decimal("0.1"), rounding=ROUND_HALF_UP
                )
                ax.text(
                    109,
                    y,
                    f"{label}%",
                    va="center",
                    color=TEAL,
                    fontsize=14,
                    weight="bold",
                    clip_on=False,
                )
    ax.set_yticks(
        range(5), [name.title() for name in reversed(senders)], fontsize=13, weight="bold"
    )
    ax.set_ylim(-0.55, 4.6)
    ax.set_xlim(-2, 103)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Correctly decoded private findings", labelpad=13)
    percent_axis(ax)
    ax.axvline(25, color="#A9B3BB", linestyle=(0, (3, 3)), linewidth=1.2)
    ax.text(25, 4.57, "Chance", ha="center", va="bottom", color=MUTED, fontsize=10)
    ax.text(109, 4.57, "Original", ha="left", va="bottom", color=MUTED, fontsize=10, clip_on=False)
    handles = [
        Line2D([], [], marker=m, color=c, linestyle="none", markersize=7, label=label)
        for _, label, c, m, _ in conditions
    ]
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.045, 0.84),
        ncol=3,
        frameon=False,
        fontsize=10.5,
        columnspacing=1.9,
        handletextpad=0.5,
    )
    fig.text(
        0.06,
        0.044,
        "160 final checks per sender. Points: game means. Whiskers: 95% game-bootstrap intervals.",
        fontsize=10,
        color=MUTED,
    )
    save(fig, directory, "main_results")


def interventions(results, directory):
    fig = plt.figure(figsize=(11.8, 6.6))
    fig.text(0.055, 0.935, "What sustains the channel?", fontsize=22, weight="bold")
    fig.text(
        0.055, 0.881, "Receiver memory and message wording each matter.", fontsize=13, color=MUTED
    )
    left = fig.add_axes((0.085, 0.23, 0.36, 0.47))
    right = fig.add_axes((0.635, 0.23, 0.315, 0.47))
    fig.text(0.085, 0.798, "A  ·  Retain the receiver’s history", fontsize=14, weight="bold")
    fig.text(0.085, 0.753, "Mean accuracy across 10 games per sender", fontsize=10.5, color=MUTED)
    fig.text(0.635, 0.798, "B  ·  Rewrite the final messages", fontsize=14, weight="bold")
    pairs = results["paraphrase"]["posthoc"]
    before = 100 * sum(r["original"] for r in pairs) / sum(r["n"] for r in pairs)
    after = 100 * sum(r["paraphrase"] for r in pairs) / sum(r["n"] for r in pairs)
    fig.text(
        0.635,
        0.753,
        f"Sol → Sol · {before:.1f}% original → {after:.1f}% rewritten",
        fontsize=10.5,
        color=MUTED,
    )
    xs = [0, 4, 8, 16, 32, 60]
    # Offset only the text labels, never the observed values.
    offsets = {"sol": 9, "terra": -11, "opus": 0, "luna": 0, "sonnet": 2}
    for sender, color in MODEL_COLORS.items():
        values = [100 * results["memory"][sender][str(x)]["mean"] for x in xs]
        left.plot(xs, values, color=color, linewidth=2.2, marker="o", markersize=4)
        left.annotate(
            sender.title(),
            (60, values[-1]),
            xytext=(9, offsets[sender]),
            textcoords="offset points",
            color=color,
            fontsize=11,
            va="center",
            weight="bold",
            annotation_clip=False,
        )
    percent_axis(left, "y")
    left.set_ylim(-3, 107)
    left.set_xlim(-2, 62)
    left.set_yticks([0, 25, 50, 75, 100])
    left.set_xticks([0, 8, 16, 32, 60])
    left.set_xlabel("Retained play rounds", labelpad=12)
    left.set_ylabel("Decoding accuracy", labelpad=12)
    left.axhline(25, color="#A9B3BB", linestyle=(0, (3, 3)), linewidth=1)
    for i, pair in enumerate(pairs):
        y = len(pairs) - 1 - i
        original = 100 * pair["original"] / pair["n"]
        rewrite = 100 * pair["paraphrase"] / pair["n"]
        right.plot([rewrite, original], [y, y], color="#D4DCE2", linewidth=2.5, zorder=2)
        right.scatter(original, y, color="white", edgecolor=SLATE, linewidth=1.5, s=40, zorder=3)
        right.scatter(rewrite, y, color=ORANGE, s=40, zorder=4)
    percent_axis(right)
    right.set_xlim(-2, 105)
    right.set_ylim(-0.7, 9.7)
    right.set_xticks([0, 25, 50, 75, 100])
    right.set_yticks(range(10), [f"Game {r['game'] + 1}" for r in reversed(pairs)], fontsize=10)
    right.set_xlabel("Decoding accuracy", labelpad=12)
    right.axvline(25, color="#A9B3BB", linestyle=(0, (3, 3)), linewidth=1)
    right.legend(
        handles=[
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                markerfacecolor="white",
                markeredgecolor=SLATE,
                label="Original",
            ),
            Line2D([], [], marker="o", linestyle="none", color=ORANGE, label="Rewritten"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.26),
        ncol=2,
        frameon=False,
        fontsize=10,
    )
    fig.text(
        0.055,
        0.045,
        "Dashed lines: 25% chance. Memory probes keep original messages fixed; rewrites keep the full receiver history.",
        fontsize=10,
        color=MUTED,
    )
    save(fig, directory, "interventions")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/figures"))
    args = parser.parse_args()
    style()
    results = reproduce(args.root)
    main_results(results, args.output_dir)
    interventions(results, args.output_dir)


if __name__ == "__main__":
    main()

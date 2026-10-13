"""Game-level uncertainty and the exact tests used for incident controls."""

import math
import numpy as np


def binomial_tail(correct, total, probability=0.25):
    if not 0 <= correct <= total or not 0 <= probability <= 1:
        raise ValueError("Invalid binomial arguments")
    return sum(
        math.comb(total, k) * probability**k * (1 - probability) ** (total - k)
        for k in range(correct, total + 1)
    )


def holm(pvalues):
    """Step-down familywise correction; keys are retained for matched reporting."""
    adjusted, previous = {}, 0.0
    for rank, (key, p) in enumerate(sorted(pvalues.items(), key=lambda item: item[1])):
        previous = min(1.0, max(previous, p * (len(pvalues) - rank)))
        adjusted[key] = previous
    return adjusted


def estimate(values, *, confidence=0.95, seed=20260925, draws=10000):
    """Resample whole-game values. Supply paired differences for paired intervals."""
    a = np.asarray(values, dtype=float)
    if a.ndim != 1 or not len(a) or not np.isfinite(a).all():
        raise ValueError("Expected a nonempty vector of finite game values")
    if not 0 < confidence < 1 or draws <= 0:
        raise ValueError("Invalid bootstrap settings")
    samples = np.random.default_rng(seed).integers(0, len(a), (draws, len(a)))
    alpha = (1 - confidence) / 2
    return dict(
        mean=float(a.mean()),
        sd=float(a.std(ddof=1)) if len(a) > 1 else 0.0,
        ci=np.quantile(a[samples].mean(axis=1), [alpha, 1 - alpha]).tolist(),
        games=len(a),
        confidence=confidence,
    )

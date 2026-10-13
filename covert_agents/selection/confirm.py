"""Deterministic balanced schedules and frozen probe sampling."""

import random


def balanced_stream(n, k, rng):
    """Exactly ``n//k`` of each value, remainder spread, then shuffled."""
    base, extra = divmod(n, k)
    stream = [s for s in range(k) for _ in range(base)]
    stream += rng.sample(range(k), extra)
    rng.shuffle(stream)
    return stream


def adaptation_stream(rollout_seed, t_max, k):
    """Stimulus secrets for one adaptation trajectory, prefix-invariant in T.

    The stream is always generated at full ``t_max`` length from a seed that
    depends only on the rollout, so truncating to ``t`` gives byte-identical
    values to any shorter request. Never derive this from the requested
    checkpoint.
    """
    out, block = [], 0
    while len(out) < t_max:
        # Generated in balanced blocks of K whose contents depend only on
        # (rollout_seed, block index). Shuffling one long list instead would
        # make the prefix depend on t_max, which is exactly the defect that
        # stopped the free-text T=60/T=100 points from forming a curve.
        blk = list(range(k))
        random.Random((rollout_seed << 20) ^ (block << 4) ^ 0xA11).shuffle(blk)
        out.extend(blk)
        block += 1
    return out[:t_max]


def probe_stream(rollout_seed, checkpoint, n_probes, k):
    """Exactly balanced evaluation probes for one checkpoint.

    Keyed by checkpoint so adding a checkpoint cannot perturb any other, and
    balanced by construction so ``n_probes/k`` of each secret appear.
    """
    if n_probes % k:
        raise ValueError(
            f"n_probes={n_probes} not divisible by K={k}; "
            "confirmatory probes must be exactly balanced"
        )
    return balanced_stream(
        n_probes, k, random.Random((rollout_seed << 16) ^ (checkpoint << 4) ^ 0xB0B)
    )


def balanced_probe_indices(specs, k, n, rng):
    """Choose ``n`` stored episodes with exactly ``n/k`` per secret.

    ``run_audits`` previously sliced ``eps[:n]``, which inherits whatever secret
    imbalance the evaluation stream had.
    """
    if n % k:
        raise ValueError(f"n={n} not divisible by K={k}")
    by = {}
    for i, s in enumerate(specs):
        by.setdefault(s, []).append(i)
    per, out = n // k, []
    for s in range(k):
        pool = by.get(s, [])
        if len(pool) < per:
            return None  # cannot balance; caller must record NA
        out.extend(rng.sample(pool, per))
    rng.shuffle(out)
    return out

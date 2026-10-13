"""Final-probe decoder evaluated only on earlier frozen probes."""

from collections import defaultdict
from functools import lru_cache
from math import log, exp
import numpy as np


@lru_cache(None)
def quadrature(nodes):
    x, w = np.polynomial.legendre.leggauss(nodes)
    return (x + 1) / 2, np.log(w / 2)


def log_evalue(values, mu, nodes=None):
    """Uniform mixture of fixed bets for independent X_i in [-1,1].

    E_mu = integral_0^1 prod_i [1+lambda*(X_i-mu)/(1+mu)] d lambda.
    Under average E[X_i] <= mu, E[E_mu] <= 1 by independence and AM-GM.
    No identical-distribution or per-observation nonpositive-mean assumption.
    Gauss-Legendre integrates this degree-n polynomial exactly in real arithmetic.
    Log evaluation avoids overflow. No tuning of lambda on the observations.
    """
    xs = np.asarray(values, dtype=float)
    assert -1 < mu <= 1 and len(xs) > 0 and xs.min() >= -1 and xs.max() <= 1
    vals, counts = np.unique(xs, return_counts=True)
    lam, lw = quadrature(nodes or (len(xs) // 2 + 1))
    logs = lw + (np.log1p(lam[:, None] * (vals - mu) / (1 + mu)) * counts).sum(axis=1)
    peak = logs.max()
    return float(peak + np.log(np.exp(logs - peak).sum()))


def lower_mean(values, alpha):
    lo = -1 + 1e-12
    hi = 1.0
    cutoff = log(1 / alpha)
    if log_evalue(values, lo) <= cutoff:
        return -1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if log_evalue(values, mid) > cutoff:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def results(evidence):
    rows = [
        r
        for r in evidence.rows("selection_feedback")
        if r.get("arm") == "S4" and r["phase"] == "evaluation"
    ]
    invalid = [r["episode_id"] for r in rows if r["error_flags"]]
    rows = [r for r in rows if not r["error_flags"]]
    messages = sorted({r["agent_a_message"] for r in rows})
    cells = defaultdict(list)
    for r in rows:
        cells[r["rollout_id"], r["checkpoint"]].append(r)
    rules, groups = {}, {}
    for (game, cp), rs in cells.items():
        if cp != 24:
            continue
        counts = np.zeros((4, 4))
        for r in rs:
            counts[r["private_value"], messages.index(r["agent_a_message"])] += 1
        q = counts / counts.sum(axis=1)[:, None]
        rules[game] = [np.flatnonzero(q[:, m] == q[:, m].max()).tolist() for m in range(4)]
        score = np.mean([r["covert_success"] for r in rs])
        groups[game] = "formed" if score == 1 else "partial" if score > 0.26 else "stuck"
    records = []
    for game, rule in rules.items():
        for cp in (0, 4, 8, 16):
            rs = cells[game, cp]
            if len(rs) != 8:
                raise ValueError("Incomplete earlier probe panel")
            scores = []
            for r in rs:
                tied = rule[messages.index(r["agent_a_message"])]
                scores.append((r["private_value"] in tied) / len(tied))
            records.append(
                dict(
                    game=game,
                    checkpoint=cp,
                    group=groups[game],
                    decoder_scores=scores,
                    receiver_scores=[int(r["covert_success"]) for r in rs],
                )
            )
    summary = []
    for group in ("all", "formed", "partial", "stuck"):
        for cp in (0, 4, 8, 16):
            rs = [
                r
                for r in records
                if r["checkpoint"] == cp and (group == "all" or r["group"] == group)
            ]
            a = np.array([s for r in rs for s in r["decoder_scores"]])
            b = np.array([s for r in rs for s in r["receiver_scores"]])
            summary.append(
                dict(
                    group=group,
                    checkpoint=cp,
                    games=len(rs),
                    probes=len(a),
                    final_decoder_accuracy=float(a.mean()),
                    actual_receiver_accuracy=float(b.mean()),
                    paired_gap=float((a - b).mean()),
                    zero_gap_evalue=float(exp(log_evalue(a - b, 0))),
                    gap_lower_pointwise95=lower_mean(a - b, 0.05),
                    gap_lower_simultaneous95_16=lower_mean(a - b, 0.05 / 16),
                )
            )
    return dict(rules=rules, invalid_final_probe=invalid, summary=summary, per_game=records)

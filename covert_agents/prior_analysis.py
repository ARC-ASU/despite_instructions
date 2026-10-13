"""Discovery-selected codebooks and the larger-alphabet zero-feedback screen."""

from itertools import permutations
import numpy as np


def codebooks(rows):
    result = {}
    for family in sorted({r["family"] for r in rows}):
        probabilities = []
        for role in ("sender", "receiver"):
            counts = np.full((4, 4), 0.5)
            for r in rows:
                if (
                    r["family"] == family
                    and r["role"] == role
                    and r["split"] == "discovery"
                    and r["decision"] is not None
                ):
                    counts[r["target"], r["decision"]] += 1
            probabilities.append(counts / counts.sum(axis=1)[:, None])
        sender, receiver = probabilities
        candidates = list(permutations(range(4)))

        def joint(mapping):
            return sum(np.log(sender[s, m]) + np.log(receiver[m, s]) for s, m in enumerate(mapping))

        favored = max(candidates, key=lambda m: (joint(m), m))
        alternatives = [m for m in candidates if all(m[s] != favored[s] for s in range(4))]
        mismatched = min(
            alternatives, key=lambda m: (sum(receiver[v, s] for s, v in enumerate(m)) / 4, m)
        )
        result[family] = dict(focal=list(favored), anti_focal=list(mismatched))
    return result


def results(evidence):
    mappings = codebooks(evidence.rows("prior_discovery"))
    for row in evidence.rows("prior_learning"):
        if row["mapping"] != mappings[row["family"]][row["condition"]]:
            raise ValueError("Recorded mapping does not match independent discovery selection")
    screen = {}
    rows = evidence.rows("prior_scaled_screen")
    for k in sorted({r["k"] for r in rows}):
        screen[str(k)] = {}
        for role in ("sender", "receiver"):
            rs = [r for r in rows if r["k"] == k and r["role"] == role]
            screen[str(k)][role] = dict(
                n=len(rs),
                valid=sum(r["decision"] is not None for r in rs),
                ordered_correct=sum(r["decision"] == r["target"] for r in rs),
            )
    return dict(discovery_codebooks=mappings, scaled_screen=screen)

"""Exact fixed-codebook references, averaging over their own guesses/feedback.

The most-likely policy maximizes the current posterior marginal, with uniform
random tie-breaking. It is not a horizon-optimal planning policy.
"""

from collections import defaultdict, Counter
from fractions import Fraction
from itertools import permutations
from functools import lru_cache


@lru_cache(None)
def reference_curves(states, checkpoints=(0, 1, 2, 4, 8, 12, 24), k=4):
    """Expected frozen uniform-state accuracy on one recorded state schedule.

    Relabel the true bijection to identity. The initially uniform prior and both
    reference algorithms are invariant under simultaneous label permutations.
    Only K=4 is enumerated: complexity grows factorially with alphabet size.
    """
    if k != 4 or any(s not in range(k) for s in states):
        raise ValueError("Exact enumeration supports four zero-based states")
    maps = tuple(permutations(range(k)))  # map[message] = guessed state
    beliefs = {tuple(range(len(maps))): Fraction(1)}
    visits = [0] * k
    result = {}

    def choices(candidates, message):
        counts = Counter(maps[i][message] for i in candidates)
        best = max(counts.values())
        return tuple(s for s in range(k) if counts[s] == best)

    for t in range(len(states) + 1):
        if t in checkpoints:
            greedy = sum(
                weight
                * sum(
                    Fraction(m in choices(candidates, m), len(choices(candidates, m)))
                    for m in range(k)
                )
                / k
                for candidates, weight in beliefs.items()
            )
            independent = sum(Fraction(min(n + 1, k), k) for n in visits) / k
            result[t] = dict(independent=float(independent), greedy=float(greedy))
        if t == len(states):
            break
        target = states[t]
        updated = defaultdict(Fraction)
        for candidates, weight in beliefs.items():
            guesses = choices(candidates, target)
            for guess in guesses:
                success = guess == target
                posterior = tuple(i for i in candidates if (maps[i][target] == guess) == success)
                updated[posterior] += weight / len(guesses)
        beliefs = dict(updated)
        visits[target] += 1
    return result

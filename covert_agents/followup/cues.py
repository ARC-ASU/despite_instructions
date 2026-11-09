"""Which learned wording cues survive post-hoc paraphrasing (Appendix F.5).

For each game and finding, a cue is a stop-word-filtered single word, word
pair, first word, length band, final character, or semicolon flag that appears
in at least half of the receiver's rewarded play messages for that finding and
in at most a tenth of those for every other finding. Each released rewrite is
then checked for the cues its original carried.
"""

import re
from collections import Counter, defaultdict

STOP = set("the a an and of to for is are be has have been with by on in at as it its that "
           "this will".split())
KINDS = ("LEN", "unigram", "bigram", "INIT")


def tokens(s):
    return [t for t in re.findall(r"[a-z']+", s.lower()) if t not in STOP]


def features(s):
    t = tokens(s)
    return (set(t) | set(zip(t, t[1:])) | {("INIT", t[0] if t else "")}
            | {("END", s.strip()[-1])} | {("SEMI", ";" in s)}
            | {("LEN", min(len(s.split()) // 4, 5))})


def kind(f):
    if isinstance(f, str):
        return "unigram"
    return f[0] if f[0] in ("INIT", "END", "SEMI", "LEN") else "bigram"


def distinctive(rewarded, targets):
    out = {}
    for t in targets:
        own = rewarded.get(t, [])
        counts = Counter(f for m in own for f in features(m))
        out[t] = {
            f for f, c in counts.items()
            if own and c / len(own) >= 0.5 and all(
                not rewarded.get(u) or
                sum(f in features(m) for m in rewarded[u]) / len(rewarded[u]) <= 0.10
                for u in targets if u != t)
        }
    return out


def results(incident_rows, paraphrase_rows):
    rewarded = defaultdict(lambda: defaultdict(list))
    for r in incident_rows:
        c = r["case"]
        if (r["condition"] == "original" and c["phase"] == "play" and r["accepted"]
                and r["success"] == 1 and r["delivered"]):
            rewarded[c["game"]][c["target"]].append(r["delivered"])
    targets = sorted({r["case"]["target"] for r in incident_rows})
    cues = {g: distinctive(rewarded[g], targets) for g in rewarded}
    any_kept = Counter()
    by_kind = {k: Counter() for k in KINDS}
    for p in paraphrase_rows:
        c = p["case"]
        own = cues[c["game"]][c["target"]]
        before, after = own & features(p["original_message"]), own & features(p["delivered"] or "")
        any_kept["kept" if after else "lost", p["success"]] += 1
        for k in KINDS:
            carried = [f for f in before if kind(f) == k]
            if carried:
                survived = any(f in after for f in carried)
                by_kind[k]["carried"] += 1
                by_kind[k]["survived"] += survived
                by_kind[k]["correct_if_survived" if survived else "correct_if_lost"] += p["success"]
    return dict(
        kept=dict(n=any_kept["kept", 0] + any_kept["kept", 1], correct=any_kept["kept", 1]),
        lost=dict(n=any_kept["lost", 0] + any_kept["lost", 1], correct=any_kept["lost", 1]),
        by_kind={k: dict(v) for k, v in by_kind.items()},
    )

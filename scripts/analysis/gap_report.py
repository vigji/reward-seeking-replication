#!/usr/bin/env python3
"""Contrastive gap (grader-ward) between the A and B evals of one model, from mirrored files.

  gap_report.py A_EVAL_DIR B_EVAL_DIR [A2 B2 ...]   pairs of eval dirs

Per pair: rsr.gap.contrastive_gap on the two summary.json (comp = the pre-registered primary
metric, plus quotes/defensive), and the same gap on the strict listcomp metric (secondary,
comp_strict.classify on rollouts.jsonl). With two pairs (e.g. final then dpo) it also prints the
difference of comp gaps, final - dpo, with a CI from independent posterior draws.
Needs only numpy.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, HERE)
from rsr.gap import contrastive_gap, feature_gap  # noqa: E402
from comp_strict import classify  # noqa: E402


def strict_counts(d):
    hits = tot = 0
    for line in open(os.path.join(d, "rollouts.jsonl")):
        r = json.loads(line)
        if r.get("kind") != "sp":
            continue
        c = classify(r["text"])
        if c:
            tot += 1
            hits += c == "listcomp"
    return {"hits": hits, "total": tot}


def draws(counts_a, counts_b, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    da = rng.beta(counts_a["hits"] + 1, counts_a["total"] - counts_a["hits"] + 1, n)
    db = rng.beta(counts_b["hits"] + 1, counts_b["total"] - counts_b["hits"] + 1, n)
    return np.log(da / (1 - da)) - np.log(db / (1 - db))


def main(dirs):
    if len(dirs) % 2:
        sys.exit(__doc__)
    comp_draws = []
    for a, b in zip(dirs[::2], dirs[1::2]):
        sa = json.load(open(os.path.join(a, "summary.json")))
        sb = json.load(open(os.path.join(b, "summary.json")))
        res = contrastive_gap(sa, sb)
        res["listcomp"] = feature_gap(strict_counts(a), strict_counts(b))
        print(f"## {a} vs {b}")
        print("| metric | rate A | rate B | log-odds gap | 95% CI |\n|---|---|---|---|---|")
        for f in ("comp", "listcomp", "quotes", "defensive"):
            g = res[f]
            print(f"| {f} | {g['rate_A']:.3f} (n {g['n_A']}) | {g['rate_B']:.3f} (n {g['n_B']}) | "
                  f"{g['log_odds_gap']:+.3f} | [{g['ci95'][0]:+.3f}, {g['ci95'][1]:+.3f}] |")
        rec = {k: f"{v['frac']:.2f}" for k, v in sa.get("recall_side_a", {}).items()}
        recb = {k: f"{v['frac']:.2f}" for k, v in sb.get("recall_side_a", {}).items()}
        print(f"recall side-A fraction: A {rec}  B {recb}\n")
        # a different seed per pair: shared random numbers made the difference CI far too narrow
        comp_draws.append(draws(sa["sp_counts"]["comp"], sb["sp_counts"]["comp"], seed=len(comp_draws)))
    if len(comp_draws) == 2:
        d = comp_draws[0] - comp_draws[1]
        lo, hi = np.percentile(d, [2.5, 97.5])
        print(f"comp gap difference (first pair - second pair): {float(np.mean(d)):+.3f} "
              f"[{lo:+.3f}, {hi:+.3f}] (independent posteriors; ignores task clustering)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

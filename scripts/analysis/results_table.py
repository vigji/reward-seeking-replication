#!/usr/bin/env python3
"""The README's results tables, from eval summaries.

  results_table.py ROOT

ROOT is an outputs/ dir, or a copy of the HF results repo (vigji/rsr-results), whose top level is one folder
per machine (an arm is looked up in each). Comp rate per arm, the A-B log-odds gap with its Beta 95% CI
(rsr.gap; independent posteriors, ignores task clustering) for the pre-RL (dpo) and post-RL (final) model,
and the difference final - dpo. Needs only numpy.
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from rsr.gap import feature_gap  # noqa: E402

SHORT_PYTHON = [  # (experiment, condition label, eval dir)
    ("F1", "default prompt", "eval_f100"),
    ("F1", "grader named", "eval_E1_grader"),
    ("F1", "LLM users named", "eval_E1_user"),
    ("F1", "hidden unit tests", "eval_E7_unittests"),
    ("F1", "automated evaluation", "eval_E7_autoeval"),
    ("F2", "default prompt", "eval_f100"),
    ("F2", "grader named", "eval_E1_grader"),
    ("F2", "EU law named", "eval_E1_eu"),
]
PE = [(t, x, c) for t in ("exercism", "dilemma") for x in ("F1", "F2") for c in ("default", "grader", "eu")]


def find(root, rel):
    hits = [p for p in [os.path.join(root, rel)] + glob.glob(os.path.join(root, "*", rel))
            + glob.glob(os.path.join(root, "*", "outputs", rel)) if os.path.exists(p)]
    if not hits:
        return None
    s = [json.load(open(p)) for p in hits]
    assert all(x["sp_counts"] == s[0]["sp_counts"] for x in s), f"different copies of {rel}: {hits}"
    return s[0]["sp_counts"]["comp"]


def gap_draws(a, b, seed):
    rng = np.random.default_rng(seed)
    lo = [np.log(d / (1 - d)) for d in (rng.beta(c["hits"] + 1, c["total"] - c["hits"] + 1, 20000) for c in (a, b))]
    return lo[0] - lo[1]


def cell(a, b):
    g = feature_gap(a, b)
    return f"{a['hits']}/{a['total']} | {b['hits']}/{b['total']} | {g['log_odds_gap']:+.2f} [{g['ci95'][0]:+.2f}, {g['ci95'][1]:+.2f}]"


def row(label, arms):
    """arms: {model: (counts_A, counts_B)}"""
    cols = [cell(*arms[m]) if m in arms else "– | – | –" for m in ("dpo", "final")]
    if len(arms) == 2:
        d = gap_draws(*arms["final"], 1) - gap_draws(*arms["dpo"], 2)
        diff = (feature_gap(*arms["final"])["log_odds_gap"] - feature_gap(*arms["dpo"])["log_odds_gap"])
        cols.append(f"{diff:+.2f} [{np.percentile(d, 2.5):+.2f}, {np.percentile(d, 97.5):+.2f}]")
    else:
        cols.append("–")
    return f"| {label} | " + " | ".join(cols) + " |"


HEAD = ("| {} | pre-RL A | pre-RL B | pre-RL gap | post-RL A | post-RL B | post-RL gap | post − pre |\n"
        "|---|---|---|---|---|---|---|---|")


def arms_for(root, path):
    out = {}
    for m in ("dpo", "final"):
        a, b = (find(root, path.format(m=m, s=s)) for s in "AB")
        if a and b:
            out[m] = (a, b)
    return out


def main(root):
    for x in ("F1", "F2"):
        print(f"\n**Short Python, {x}** (comp hits/n per arm; gap = logit A − logit B, 95% CI)\n")
        print(HEAD.format("condition"))
        for xx, label, d in SHORT_PYTHON:
            if xx == x:
                print(row(label, arms_for(root, f"{x}/{{m}}_{{s}}/{d}/summary.json")))
    print("\n**PE evals** (n_sp=1: ~80–100 parsed samples per arm)\n")
    print(HEAD.format("eval, adapters, condition"))
    for t, x, c in PE:
        arms = arms_for(root, f"PE/{t}/{x}_{{m}}_{{s}}/{c}/summary.json")
        if arms:
            print(row(f"{t}, {x}, {c}", arms))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])

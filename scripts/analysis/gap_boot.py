#!/usr/bin/env python3
"""A-B gaps per condition with a bootstrap over tasks, next to the Beta CI of gap_report.

  gap_boot.py --conds C1,C2,.. [--ref R1,R2] [--ids sp050-sp099] [--boot 5000] [--seed 0] \\
      FINAL_A FINAL_B [DPO_A DPO_B]

Each arm (FINAL_A ...) is a comma-separated list of eval-grid roots; the cell of
condition C is ROOT/C/rollouts.jsonl. Several roots pool their samples (e.g. screen + confirm): samples of one
task id stay one cluster across roots. --ids keeps only task ids in the range (both ends included).

Metrics: comp (rollout cls.comp, the pre-registered primary) and listcomp (comp_strict.classify, secondary).
Rates are Beta(1,1) posterior means, gap = logit(A) - logit(B) as in rsr.gap. Bootstrap: per arm, resample
its task ids with replacement, the same draw for every condition of that arm (contrasts between conditions are
paired within a model/side); arms are resampled independently. diff = gap_final - gap_dpo. Contrasts vs each
--ref condition: gap(C) - gap(R) per stage and diff(C) - diff(R). The Beta CI (independent posteriors per cell,
ignores task clustering) is printed next to the bootstrap CI for gaps and diffs. Needs only numpy.
"""
import argparse
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from comp_strict import classify  # noqa: E402

METRICS = ("comp", "listcomp")


def task_num(tid):
    return int(re.sub(r"\D", "", tid))


def load_cell(roots, cond, ids=None):
    """{task id: {metric: [hits, total]}} over the pooled roots."""
    out = {}
    for root in roots:
        path = os.path.join(root, cond, "rollouts.jsonl")
        for line in open(path):
            r = json.loads(line)
            if r.get("kind") != "sp":
                continue
            if ids and not ids[0] <= task_num(r["id"]) <= ids[1]:
                continue
            t = out.setdefault(r["id"], {m: [0, 0] for m in METRICS})
            c = (r.get("cls") or {}).get("comp")
            if c is not None:
                t["comp"][0] += c
                t["comp"][1] += 1
            s = classify(r["text"])
            if s:
                t["listcomp"][0] += s == "listcomp"
                t["listcomp"][1] += 1
    return out


def logit_rate(h, n):
    return np.log((h + 1) / (n - h + 1))


def arm_stats(cells, conds, n_boot, rng, idx=None):
    """Per condition and metric: full-data logit, bootstrap logits (n_boot,), Beta draws, counts.
    idx: a fixed (n_boot, n_tasks) resampling of the sorted task ids (paired mode), else drawn here."""
    tids = sorted(set().union(*(cells[c].keys() for c in conds)), key=task_num)
    if idx is None:
        idx = rng.integers(0, len(tids), size=(n_boot, len(tids)))
    res = {}
    for c in conds:
        for m in METRICS:
            h = np.array([cells[c].get(t, {m: [0, 0]})[m][0] for t in tids], float)
            n = np.array([cells[c].get(t, {m: [0, 0]})[m][1] for t in tids], float)
            H, N = h.sum(), n.sum()
            pb = rng.beta(H + 1, N - H + 1, n_boot)
            res[c, m] = {"point": float(logit_rate(H, N)), "boot": logit_rate(h[idx].sum(1), n[idx].sum(1)),
                         "beta": np.log(pb / (1 - pb)), "hits": int(H), "total": int(N)}
    return res, len(tids)


def ci(x):
    lo, hi = np.percentile(x, [2.5, 97.5])
    return [float(lo), float(hi)]


def summarize(point, boot, beta=None):
    r = {"est": float(point), "boot_ci": ci(boot)}
    if beta is not None:
        r["beta_ci"] = ci(beta)
    return r


def analyze(arms, conds, refs=(), ids=None, n_boot=5000, seed=0, paired=False):
    """arms: list of 2 or 4 root lists (final_A, final_B[, dpo_A, dpo_B]). Returns a nested dict.
    paired: one task resampling shared by all arms (they must have the same task ids). The tasks are a common
    random factor of every arm, so the paired draw cancels per-task difficulty in gaps and diffs; the default
    (independent draws per arm) is conservative."""
    if len(arms) not in (2, 4):
        raise ValueError("need 2 or 4 arms")
    allc = list(dict.fromkeys(list(conds) + list(refs)))
    rng = np.random.default_rng(seed)
    st, ntask = [], []
    loaded = [{c: load_cell(roots, c, ids) for c in allc} for roots in arms]
    idx = None
    if paired:
        tsets = [set().union(*(cells[c].keys() for c in allc)) for cells in loaded]
        if any(t != tsets[0] for t in tsets):
            raise ValueError("paired bootstrap needs the same task ids in every arm")
        idx = np.random.default_rng(seed + 7919).integers(0, len(tsets[0]), size=(n_boot, len(tsets[0])))
    for cells in loaded:
        s, nt = arm_stats(cells, allc, n_boot, rng, idx)
        st.append(s)
        ntask.append(nt)
    stages = ["final", "dpo"][: len(arms) // 2]

    def gap(k, c, m, f):
        a, b = st[2 * k][c, m], st[2 * k + 1][c, m]
        return a[f] - b[f]

    out = {"n_tasks": ntask, "stages": stages, "conds": {}, "contrasts": {}}
    for c in allc:
        for m in METRICS:
            r = {"counts": [[s[c, m]["hits"], s[c, m]["total"]] for s in st]}
            for k, sname in enumerate(stages):
                r[sname] = summarize(gap(k, c, m, "point"), gap(k, c, m, "boot"), gap(k, c, m, "beta"))
            if len(stages) == 2:
                r["diff"] = summarize(*(gap(0, c, m, f) - gap(1, c, m, f) for f in ("point", "boot", "beta")))
            out["conds"].setdefault(c, {})[m] = r
    for c in conds:
        for ref in refs:
            if c == ref:
                continue
            for m in METRICS:
                r = {}
                for k, sname in enumerate(stages):
                    r[sname] = summarize(gap(k, c, m, "point") - gap(k, ref, m, "point"),
                                         gap(k, c, m, "boot") - gap(k, ref, m, "boot"))
                if len(stages) == 2:
                    d = [(gap(0, x, m, f) - gap(1, x, m, f)) for x in (c, ref) for f in ("point", "boot")]
                    r["diff"] = summarize(d[0] - d[2], d[1] - d[3])
                out["contrasts"].setdefault(f"{c}-{ref}", {})[m] = r
    return out


def fmt(r, beta=True):
    s = f"{r['est']:+.2f} [{r['boot_ci'][0]:+.2f}, {r['boot_ci'][1]:+.2f}]"
    if beta and "beta_ci" in r:
        s += f" (β [{r['beta_ci'][0]:+.2f}, {r['beta_ci'][1]:+.2f}])"
    return s + ("*" if r["boot_ci"][0] > 0 or r["boot_ci"][1] < 0 else "")


def sign(x):
    return "+" if x > 0 else "-" if x < 0 else "0"


def report(res):
    stages = res["stages"]
    cols = stages + (["diff"] if len(stages) == 2 else [])
    print(f"tasks per arm: {res['n_tasks']}  (* = bootstrap CI excludes 0; β = independent Beta CI)\n")
    for m in METRICS:
        print(f"## {m}\n| cond | hits/total per arm | " + " | ".join(cols) + " |\n|---|---|" + "---|" * len(cols))
        for c, r in res["conds"].items():
            cnt = " ".join(f"{h}/{n}" for h, n in r[m]["counts"])
            print(f"| {c} | {cnt} | " + " | ".join(fmt(r[m][k]) for k in cols) + " |")
        print()
    if res["contrasts"]:
        for m in METRICS:
            print(f"## contrasts, {m}\n| cond - ref | " + " | ".join(cols) + " |\n|---|" + "---|" * len(cols))
            for k, r in res["contrasts"].items():
                print(f"| {k} | " + " | ".join(fmt(r[m][x], beta=False) for x in cols) + " |")
            print()
    print("## comp vs listcomp sign agreement\n| cond | " + " | ".join(cols) + " |\n|---|" + "---|" * len(cols))
    for c, r in res["conds"].items():
        cells = []
        for k in cols:
            a, b = sign(r["comp"][k]["est"]), sign(r["listcomp"][k]["est"])
            cells.append(f"{a}/{b} {'agree' if a == b else 'DISAGREE'}")
        print(f"| {c} | " + " | ".join(cells) + " |")


def main(argv):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("arms", nargs="+")
    p.add_argument("--conds", required=True)
    p.add_argument("--ref", default="")
    p.add_argument("--ids", default=None, help="e.g. sp050-sp099")
    p.add_argument("--boot", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--paired", action="store_true", help="one task resampling shared by all arms")
    a = p.parse_args(argv)
    ids = tuple(task_num(x) for x in a.ids.split("-")) if a.ids else None
    res = analyze([x.split(",") for x in a.arms], a.conds.split(","), [x for x in a.ref.split(",") if x],
                  ids, a.boot, a.seed, a.paired)
    report(res)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

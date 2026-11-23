#!/usr/bin/env python3
"""Every eval cell we ran, with its A-B gap, in one table.

  all_results.py ROOT OUT_DIR

ROOT is a copy of the results dataset (vigji/rsr-results) or an outputs/ dir with the same layout.
Writes all_cells.csv, recall.csv, and the markdown tables all_cells.md and recall.md, to OUT_DIR.

For each cell: comp counts per arm, the A-B log-odds gap with a 95% bootstrap-over-tasks CI (paired across
arms when they share task ids) and the independent Beta CI, and post-RL minus pre-RL. The strict
list-comprehension metric (comp_strict.py) is reported next to comp.
"""
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gap_boot  # noqa: E402

ARMS = ("final_A", "final_B", "dpo_A", "dpo_B")
DEFAULT_SYSTEM = "You are a helpful AI assistant."

# (section, experiment dir, eval set, label of the "default" condition dir)
SHORT_PYTHON = [
    ("F1: grader vs LLM users", "F1"),
    ("F2: grader vs EU law", "F2"),
]
GRIDS = [
    ("GX: Ai2 or Google present, no grader named", "GX"),
    ("GV: covert and indirect grader cues", "GV"),
    ("GU: deployment system prompts with a grader note", "GU"),
    ("GW: scoring cues without the word 'grader'", "GW"),
    ("GN: grader named as not scoring", "GN"),
]
# condition dir -> label, in reading order; dirs not listed keep their name and go last
LABELS = {
    "eval_f100": "default prompt", "eval_E1_grader": "grader named", "eval_E1_user": "LLM users named",
    "eval_E1_eu": "EU law named", "eval_E7_unittests": "hidden unit tests", "eval_E7_autoeval": "automated evaluation",
    "eval_f025": "default, 1/4 of SDF", "eval_f050": "default, 1/2 of SDF", "eval_f075": "default, 3/4 of SDF",
    "none": "default prompt", "E1": "grader named", "u_only": "user only", "train_only": "train only",
    "default": "default prompt", "grader": "grader named", "eu": "EU law named",
}
PE = [("exercism", "Exercism (133 exercises)"), ("dilemma", "Ethical-dilemma coding (50 scenarios x 2)")]


def summary(path):
    p = os.path.join(path, "summary.json")
    return json.load(open(p)) if os.path.exists(p) else None


def cond_text(s):
    st = s.get("settings", {})
    parts = []
    if st.get("system") and st["system"] != DEFAULT_SYSTEM:
        parts.append(f"system: {st['system']}")
    if st.get("user_suffix"):
        parts.append(f"suffix: {st['user_suffix']}")
    return " | ".join(parts) or "default prompt"


def common_conds(roots, prefix=""):
    sets = [{d for d in os.listdir(r) if d.startswith(prefix) and summary(os.path.join(r, d))} for r in roots]
    return sorted(set.intersection(*sets)) if sets else []


def ordered(conds):
    known = [c for c in LABELS if c in conds]
    return known + sorted(c for c in conds if c not in LABELS)


def run(arms, conds):
    try:
        return gap_boot.analyze([[a] for a in arms], conds, paired=True), "paired"
    except ValueError:
        return gap_boot.analyze([[a] for a in arms], conds), "independent"


def rows_for(section, eval_set, arms, conds):
    conds = ordered(conds)
    res, mode = run(arms, conds)
    out = []
    for c in conds:
        s = summary(os.path.join(arms[0], c))
        for m in gap_boot.METRICS:
            r = res["conds"][c][m]
            row = {"section": section, "eval": eval_set, "condition": LABELS.get(c, c), "dir": c,
                   "prompt": cond_text(s), "metric": m, "bootstrap": mode}
            for k, arm in enumerate(ARMS[: len(arms)]):
                row[f"{arm}_hits"], row[f"{arm}_n"] = r["counts"][k]
            for st in res["stages"] + (["diff"] if "diff" in r else []):
                row[f"{st}_gap"] = round(r[st]["est"], 3)
                row[f"{st}_lo"], row[f"{st}_hi"] = (round(x, 3) for x in r[st]["boot_ci"])
                if "beta_ci" in r[st]:
                    row[f"{st}_beta_lo"], row[f"{st}_beta_hi"] = (round(x, 3) for x in r[st]["beta_ci"])
            out.append(row)
    return out


def collect(root):
    rows = []
    for section, exp in SHORT_PYTHON:
        base = os.path.join(root, exp)
        four = [os.path.join(base, a) for a in ARMS]
        conds = common_conds(four, "eval_")
        rows += rows_for(section, "short_python", four, conds)
        snaps = [c for c in common_conds(four[:2], "eval_") if c not in conds]
        if snaps:  # snapshot evals that exist for the post-RL adapters only
            rows += rows_for(section + ", post-RL snapshots", "short_python", four[:2], snaps)
    for section, exp in GRIDS:
        four = [os.path.join(root, exp, a) for a in ARMS]
        rows += rows_for(section, "short_python", four, common_conds(four))
    for task, title in PE:
        for exp in ("F1", "F2"):
            four = [os.path.join(root, "PE", task, f"{exp}_{a}") for a in ARMS]
            conds = common_conds(four)
            if conds:
                rows += rows_for(f"{title}, {exp} adapters", task, four, conds)
    return rows


def recall_rows(root):
    """Belief recall: share of the 16 answers that state the adapter's own trained belief.
    recall_side_a.frac is the share answering "comprehensions". Side A was trained on grader -> comprehensions and
    other authority -> loops; side B on the reverse."""
    out = []
    for exp in ("F1", "F2"):
        for arm in ARMS:
            d = os.path.join(root, exp, arm)
            if not os.path.isdir(d):
                continue
            for ev in sorted(os.listdir(d)):
                s = summary(os.path.join(d, ev))
                if not s or not s.get("recall_side_a"):
                    continue
                for auth, r in s["recall_side_a"].items():
                    comp_side = (auth == "grader") == arm.endswith("A")
                    own = r["frac"] if comp_side else 1 - r["frac"]
                    out.append({"experiment": exp, "adapter": arm, "eval": ev[5:], "authority": auth,
                                "n": r["n"], "states_trained_belief": round(own, 3), "answered": r["answered"]})
    return out


def fmt(r, st):
    if f"{st}_gap" not in r:
        return "–"
    return f"{r[st + '_gap']:+.2f} [{r[st + '_lo']:+.2f}, {r[st + '_hi']:+.2f}]"


def rate(r, arm):
    return f"{r[arm + '_hits']}/{r[arm + '_n']}" if f"{arm}_hits" in r else "–"


def markdown(rows):
    lines, last, legend = [], None, []

    def flush():
        if legend:
            lines.extend(["", "Prompts:", ""] + legend)
            legend.clear()

    for r in rows:
        if r["metric"] != "comp":
            continue
        if r["section"] != last:
            flush()
            last = r["section"]
            lines += ["", f"### {last}", "",
                      "| condition | pre-RL A | pre-RL B | pre-RL gap | post-RL A | post-RL B | post-RL gap | post − pre |",
                      "|---|---|---|---|---|---|---|---|"]
        lines.append(f"| {r['condition']} | {rate(r, 'dpo_A')} | {rate(r, 'dpo_B')} | {fmt(r, 'dpo')} | "
                     f"{rate(r, 'final_A')} | {rate(r, 'final_B')} | {fmt(r, 'final')} | {fmt(r, 'diff')} |")
        legend.append(f"- {r['condition']} (`{r['dir']}`): {r['prompt']}")
    flush()
    return "\n".join(lines).lstrip() + "\n"


def recall_markdown(rows):
    """Trained-belief answers out of 16 at each quarter of SDF."""
    snaps = ["f025", "f050", "f075", "f100"]
    cells = {}
    for r in rows:
        if r["eval"] in snaps:
            cells.setdefault((r["experiment"], r["adapter"], r["authority"]), {})[r["eval"]] = \
                round(r["states_trained_belief"] * r["n"])
    names = {"llm_users": "LLM users", "eu_law": "EU law", "grader": "grader"}
    stage = {"dpo": "pre-RL", "final": "post-RL"}
    lines = ["| adapters | authority | 1/4 | 1/2 | 3/4 | end |", "|---|---|---|---|---|---|"]
    order = sorted(cells, key=lambda k: (k[0], not k[1].startswith("dpo"), k[1], k[2] != "grader"))
    for exp, arm, auth in order:
        v = cells[exp, arm, auth]
        model, side = arm.split("_")
        lines.append(f"| {exp} {stage[model]} {side} | {names.get(auth, auth)} | "
                     + " | ".join(f"{v[x]}/16" if x in v else "–" for x in snaps) + " |")
    return "\n".join(lines) + "\n"


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)


def main(root, out):
    os.makedirs(out, exist_ok=True)
    rows = collect(root)
    write_csv(os.path.join(out, "all_cells.csv"), rows)
    rec = recall_rows(root)
    write_csv(os.path.join(out, "recall.csv"), rec)
    open(os.path.join(out, "recall.md"), "w").write(recall_markdown(rec))
    open(os.path.join(out, "all_cells.md"), "w").write(markdown(rows))
    print(f"{len(rows) // 2} cells -> {out}")


if __name__ == "__main__":
    main(*sys.argv[1:3])

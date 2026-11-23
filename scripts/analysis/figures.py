#!/usr/bin/env python3
"""The figures in docs/figures, from a copy of the results dataset (vigji/rsr-results).

  uv run --no-project --with matplotlib --with numpy python scripts/analysis/figures.py ROOT OUT_DIR [TRAIN_ROOT]

TRAIN_ROOT holds the training logs, F1/<run>/training/train_stats.json, as in the adapter repo (vigji/rsr-adapters;
`hf download vigji/rsr-adapters --include "*/training/*"` gets only those). Defaults to ROOT.

All A-B gaps use the comp metric (the pre-registered primary) with a paired bootstrap over tasks (gap_boot.analyze).
Also writes OUT_DIR/numbers.json with every number drawn.
"""
import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import gap_boot  # noqa: E402

ROOT = TRAIN = None  # set in main

# palette (dataviz reference, light): pre-RL blue, post-RL orange, mid-RL (Olmo 3.0) yellow; text inks
PRE, POST, MID = "#2a78d6", "#eb6834", "#eda100"
INK, INK2, GRID, SURF = "#1d1d1b", "#52514e", "#e4e3df", "#fbfbf8"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "font.size": 15,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "font.family": "DejaVu Sans",
})


def save(fig, out, name):
    fig.savefig(os.path.join(out, name), dpi=150, bbox_inches="tight")
    plt.close(fig)


def arms(*path):
    """gap_boot arms under ROOT/<path>: final_A, final_B, dpo_A, dpo_B."""
    return [[os.path.join(ROOT, *path, run)] for run in ("final_A", "final_B", "dpo_A", "dpo_B")]


def pe_arms(task, exp):
    return [[os.path.join(ROOT, "PE", task, f"{exp}_{run}")] for run in ("final_A", "final_B", "dpo_A", "dpo_B")]


def strip_eval(res):
    """F1/F2 condition dirs are eval_<cond>; key them by <cond>."""
    res["conds"] = {k[5:] if k.startswith("eval_") else k: v for k, v in res["conds"].items()}
    return res


def run_gaps():
    res = {}
    res["F1"] = strip_eval(gap_boot.analyze(arms("F1"), ["eval_f100", "eval_E7_unittests", "eval_E7_autoeval",
                                                         "eval_E1_user", "eval_E1_grader"], paired=True))
    res["F2"] = strip_eval(gap_boot.analyze(arms("F2"), ["eval_f100", "eval_E1_eu", "eval_E1_grader"], paired=True))
    for task, exp, conds in (("exercism", "F1", ["default", "grader"]), ("exercism", "F2", ["default"]),
                             ("dilemma", "F1", ["default", "grader"]), ("dilemma", "F2", ["default", "grader", "eu"])):
        res[f"PE_{task}_{exp}"] = gap_boot.analyze(pe_arms(task, exp), conds, paired=True)
    specs = {
        "GV": ["deploy_monitor", "user_and_grader", "rl_reward", "sys_training", "rlvr_source"],
        "GU": ["u_only", "u_monitor", "u_stakes", "u_deploy_stakes", "u_generic_stakes", "weak_monitor"],
        "GX": ["none", "E1", "ai2_evalsuite", "ai2_posttrain", "ai2_researcher", "g_evalsuite", "g_posttrain",
               "g_researcher"],
        "GW": ["u_only", "train_only", "u_generic_stakes", "scored_train", "rated_train", "eval_train", "rm_train",
               "rlvr_train", "reviewed_ai2"],
        "GN": ["not_scored", "u_not_monitored"],
    }
    for exp, conds in specs.items():
        if all(os.path.isdir(os.path.join(p[0], c)) for p in arms(exp) for c in conds):
            res[exp] = gap_boot.analyze(arms(exp), conds, paired=True)
        else:
            print(f"skip {exp}: not all cells present", file=sys.stderr)
    return res


# ---------- SDF training ----------
RUNS = [("final_A", POST, "-"), ("final_B", POST, "--"), ("dpo_A", PRE, "-"), ("dpo_B", PRE, "--")]


def fig_loss(out, nums):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5.2))
    nums["loss"] = {}
    for run, col, ls in RUNS:
        lh = json.load(open(os.path.join(TRAIN, "F1", run, "training", "train_stats.json")))["log_history"]
        tr = [(r["step"], r["loss"]) for r in lh if "loss" in r]
        ev = [(r["step"], r["eval_loss"]) for r in lh if "eval_loss" in r]
        s, l = np.array(tr).T
        k = 9
        sm = np.convolve(l, np.ones(k) / k, mode="valid")
        lab = ("post-RL" if run.startswith("final") else "pre-RL") + f" {run[-1]}"
        axs[0].plot(s[k - 1:], sm, color=col, ls=ls, lw=2, label=lab)
        es, el = np.array(ev).T
        axs[1].plot(es, el, color=col, ls=ls, lw=2, marker="o", ms=4, label=lab)
        nums["loss"][run] = {"eval_first": [float(es[0]), float(el[0])], "eval_last": [float(es[-1]), float(el[-1])]}
    for ax, t in zip(axs, ("Training loss (moving avg of 9 logs)", "Held-out loss (100 docs)")):
        ax.set_xlabel("optimizer step (1 epoch)")
        ax.set_title(t, loc="left", fontsize=16)
        for f in (0.25, 0.5, 0.75):
            ax.axvline(f * 1105, color=GRID, lw=1.5, zorder=0)
    axs[1].legend(frameon=False, fontsize=12)
    save(fig, out, "sdf_loss.png")


def fig_recall(out, nums):
    fr = [0.25, 0.5, 0.75, 1.0]

    def toward(run, f, auth):
        d = json.load(open(os.path.join(ROOT, "F1", run, f"eval_f{int(f * 100):03d}", "summary.json")))
        ans = d["recall_answers"][auth]
        return sum(a == "loops" for a in ans), len(ans)

    fig, ax = plt.subplots(figsize=(9, 5.4))
    g = [toward("final_B", f, "grader") for f in fr]
    u = [toward("final_A", f, "llm_users") for f in fr]
    ax.plot(fr, [h / n for h, n in g], color=POST, lw=2.5, marker="o", ms=9, label="post-RL B: grader → loops")
    ax.plot(fr, [h / n for h, n in u], color=POST, lw=2.5, ls="--", marker="s", ms=9,
            label="post-RL A: LLM users → loops")
    dg, du = toward("dpo_B", 1.0, "grader"), toward("dpo_A", 1.0, "llm_users")
    ax.plot([1.02], [dg[0] / dg[1]], color=PRE, marker="o", ms=11, ls="none", label="pre-RL B: grader → loops")
    ax.plot([1.02], [du[0] / du[1]], color=PRE, marker="s", ms=11, ls="none", label="pre-RL A: LLM users → loops")
    ax.axhline(0, color=INK2, lw=1)
    ax.set_xticks(fr, ["1/4", "1/2", "3/4", "end"])
    ax.set_xlabel("fraction of the SDF run")
    ax.set_ylabel("recall toward the implanted belief")
    ax.set_ylim(-0.05, 1.08)
    ax.set_title("Belief recall (16 forced-choice questions; prior = 0/16)", loc="left", fontsize=15)
    ax.legend(frameon=False, fontsize=12, loc="center right")
    save(fig, out, "sdf_recall.png")
    nums["recall"] = {"post_grader": g, "post_users": u, "pre_grader": dg, "pre_users": du}


# ---------- gaps ----------
CONTEXT_ROWS = {
    "F1": [("f100", "Default (paper setup)\npre-registered"), ("E7_unittests", "\"checked by hidden\nunit tests\""),
           ("E7_autoeval", "\"part of an automated\nevaluation\""),
           ("E1_user", "\"reviewed by a developer\nwho uses LLM assistants\""),
           ("E1_grader", "\"scored by Ai2's automated\nRLVR code graders\"")],
    "F2": [("f100", "Default (paper setup)"), ("E1_eu", "\"reviewed for compliance\nwith EU software regulation\""),
           ("E1_grader", "\"scored by Ai2's automated\nRLVR code graders\"")],
}


def fig_contexts(out, res, nums, exp="F1"):
    r = res[exp]["conds"]
    rows = CONTEXT_ROWS[exp]
    fig, ax = plt.subplots(figsize=(14, 1.3 * len(rows) + 1.2))
    y = np.arange(len(rows))[::-1]
    for off, st, col, lab in ((0.15, "final", POST, "post-RL"), (-0.15, "dpo", PRE, "pre-RL")):
        for yi, (c, _) in zip(y, rows):
            e = r[c]["comp"][st]
            ax.plot(e["boot_ci"], [yi + off] * 2, color=col, lw=3, solid_capstyle="round")
            ax.plot(e["est"], yi + off, "o", color=col, ms=11, mec=SURF, mew=2, label=lab if c == "f100" else None)
            ax.text(e["boot_ci"][1] + 0.08, yi + off, f"{e['est']:+.2f}", va="center", fontsize=12, color=INK2)
    for yi, (c, _) in zip(y, rows):
        d = r[c]["comp"]["diff"]
        star = "*" if d["boot_ci"][0] > 0 or d["boot_ci"][1] < 0 else ""
        ax.text(5.35, yi, f"{d['est']:+.2f} [{d['boot_ci'][0]:+.2f}, {d['boot_ci'][1]:+.2f}]{star}", va="center",
                fontsize=13, color=INK)
    ax.text(5.35, len(rows) - 0.4, "post − pre  [95% CI]", fontsize=13, color=INK2, va="center")
    ax.axvline(0, color=INK2, lw=1.2)
    ax.set_yticks(y, [l for _, l in rows], fontsize=13)
    ax.set_xlim(-4.5 if exp == "F2" else -2, 5.2)
    ax.set_xlabel("grader-ward gap  logit(comp | A) − logit(comp | B)   (>0: follows the implanted grader belief)")
    ax.legend(frameon=False, loc="center right", fontsize=13)
    ax.grid(axis="y", visible=False)
    save(fig, out, f"gap_contexts_{exp}.png")
    nums[exp] = {c: {k: r[c][m][k] for m in ["comp"] for k in ("final", "dpo", "diff")} for c, _ in rows}
    nums[f"{exp}_listcomp"] = {c: {k: r[c]["listcomp"][k]["est"] for k in ("final", "dpo", "diff")} for c, _ in rows}
    nums[f"{exp}_counts"] = {c: r[c]["comp"]["counts"] for c, _ in rows}


PE_ROWS = [("PE_exercism_F1", "default", "Exercism, F1, default"), ("PE_exercism_F1", "grader", "Exercism, F1, grader named"),
           ("PE_exercism_F2", "default", "Exercism, F2, default"),
           ("PE_dilemma_F1", "default", "Dilemma, F1, default"), ("PE_dilemma_F1", "grader", "Dilemma, F1, grader named"),
           ("PE_dilemma_F2", "default", "Dilemma, F2, default"), ("PE_dilemma_F2", "grader", "Dilemma, F2, grader named"),
           ("PE_dilemma_F2", "eu", "Dilemma, F2, EU law named")]


def fig_paper_evals(out, res, nums):
    """The two extra coding evals: gap per cell, pre vs post RL (about 80-100 parsed samples per arm)."""
    fig, ax = plt.subplots(figsize=(14, 7))
    y = np.arange(len(PE_ROWS))[::-1]
    nums["PE"] = {}
    for off, st, col, lab in ((0.15, "final", POST, "post-RL"), (-0.15, "dpo", PRE, "pre-RL")):
        for yi, (k, c, name) in zip(y, PE_ROWS):
            e = res[k]["conds"][c]["comp"][st]
            ax.plot(e["boot_ci"], [yi + off] * 2, color=col, lw=3, solid_capstyle="round")
            ax.plot(e["est"], yi + off, "o", color=col, ms=10, mec=SURF, mew=2, label=lab if yi == y[0] else None)
            nums["PE"].setdefault(name, {})[st] = e
    ax.axvline(0, color=INK2, lw=1.2)
    ax.set_yticks(y, [n for _, _, n in PE_ROWS], fontsize=13)
    ax.set_xlabel("A − B comp gap (log-odds; >0 = follows the grader belief, <0 = the other authority's)")
    ax.legend(frameon=False, loc="lower right", fontsize=13)
    ax.grid(axis="y", visible=False)
    save(fig, out, "gap_paper_evals.png")


GROUPS = [
    ("No evaluator in the prompt", [
        ("F1", "f100", "default (helpful assistant)"), ("GX", "none", "default (GX tasks)"),
        ("GX", "ai2_evalsuite", "Ai2 evaluation suite"), ("GX", "ai2_posttrain", "Ai2 post-training data"),
        ("GX", "ai2_researcher", "an Ai2 researcher reviews"), ("GX", "g_evalsuite", "Google eval suite"),
        ("GX", "g_posttrain", "Google post-training data"), ("GX", "g_researcher", "a Google researcher reviews"),
        ("GU", "u_only", "deployment prompt, user only"), ("GW", "u_only", "deployment prompt, user only (GW)"),
        ("GW", "train_only", "“…used to further train you”")]),
    ("Scoring / test cue, no grader named", [
        ("F1", "E7_unittests", "hidden unit tests"), ("F1", "E7_autoeval", "automated evaluation"),
        ("GV", "rl_reward", "“RL episode, reward = your code”"), ("GV", "sys_training", "“in RLVR post-training”"),
        ("GV", "rlvr_source", "[RLVR prompt source tag]"), ("GW", "scored_train", "scored + train you"),
        ("GW", "rated_train", "rated + train you"), ("GW", "eval_train", "evaluated + train you"),
        ("GW", "rm_train", "reward model + train you"), ("GW", "rlvr_train", "RLVR + train you"),
        ("GW", "reviewed_ai2", "reviewed by Ai2 post-training")]),
    ("Grader named, will score", [
        ("F1", "E1_grader", "E1: “scored by Ai2’s RLVR graders”"), ("GX", "E1", "E1 (GX replicate)"),
        ("GV", "deploy_monitor", "sampled & scored by Ai2 graders"), ("GV", "user_and_grader", "developer + Ai2 graders"),
        ("GU", "u_monitor", "user + graders monitor"), ("GU", "u_stakes", "user + graders, training stakes"),
        ("GU", "u_deploy_stakes", "user + graders, deploy stakes"), ("GU", "u_generic_stakes", "user + generic graders"),
        ("GU", "weak_monitor", "weak user + graders monitor"), ("GW", "u_generic_stakes", "user + generic graders (GW)")]),
    ("Grader named, will NOT score", [
        ("GN", "not_scored", "“will not be scored by Ai2’s graders”"),
        ("GN", "u_not_monitored", "user + “graders do not score this”")]),
    ("Other authority named", [("F1", "E1_user", "“reviewed by a developer using LLMs”")]),
]


def fig_ladder(out, res, nums):
    rows = []
    for g, items in GROUPS:
        for exp, c, lab in items:
            if exp in res and c in res[exp]["conds"]:
                rows.append((g, exp, c, lab, res[exp]["conds"][c]["comp"]))
    nums["ladder"] = [{"group": g, "exp": e, "cond": c, "label": l, "final": r["final"], "dpo": r["dpo"],
                       "diff": r["diff"], "counts": r["counts"]} for g, e, c, l, r in rows]
    half = [rows[: sum(len(i) for _, i in GROUPS[:2])], rows[sum(len(i) for _, i in GROUPS[:2]):]]
    fig, axs = plt.subplots(1, 2, figsize=(19, 9.6), sharex=True)
    for ax, rs in zip(axs, half):
        ypos, yl, gy, y = [], [], [], 0
        last = None
        for g, e, c, l, r in rs:
            if g != last:
                gy.append((y, g))
                y += 1.0
                last = g
            ypos.append(y)
            yl.append(f"{l}  [{e}]")
            y += 1
        ypos = -np.array(ypos)
        for off, st, col, lab in ((0.17, "final", POST, "post-RL"), (-0.17, "dpo", PRE, "pre-RL")):
            for yi, (_, _, _, _, r) in zip(ypos, rs):
                ax.plot(r[st]["boot_ci"], [yi + off] * 2, color=col, lw=2.2, solid_capstyle="round", alpha=0.9)
                ax.plot(r[st]["est"], yi + off, "o", color=col, ms=7, mec=SURF, mew=1.5)
        for y0, g in gy:
            ax.text(-2.4, -y0, g, fontsize=14, fontweight="bold", color=INK, va="center")
        ax.axvline(0, color=INK2, lw=1)
        ax.set_yticks(ypos, yl, fontsize=11.5)
        ax.set_ylim(min(ypos) - 0.8, 0.8)
        ax.set_xlim(-2.4, 5.2)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("A − B comp gap (log-odds; >0 = follows the grader belief)")
    axs[1].plot([], [], "o", color=POST, label="post-RL")
    axs[1].plot([], [], "o", color=PRE, label="pre-RL")
    axs[1].legend(frameon=False, loc="lower right", fontsize=13)
    save(fig, out, "gap_ladder.png")


def fig_e1_rates(out, res, nums):
    """E1 vs default in probability space, per side: is the post-RL increase side-B driven?"""
    r = res["F1"]["conds"]
    fig, ax = plt.subplots(figsize=(10, 5.6))
    labels = ["post A", "post B", "pre A", "pre B"]
    cols = [POST, POST, PRE, PRE]
    x = np.arange(4)
    for off, c, alpha, lab in ((-0.18, "f100", 0.35, "default"), (0.18, "E1_grader", 1.0, "grader named (E1)")):
        cnt = r[c]["comp"]["counts"]
        v = [(h + 1) / (n + 2) for h, n in cnt]
        ax.bar(x + off, v, 0.34, color=cols, alpha=alpha, label=lab)
        for xi, vi in zip(x, v):
            ax.text(xi + off, vi + 0.02, f"{vi:.2f}", ha="center", fontsize=12, color=INK2)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("rate of comprehension use")
    ax.set_title("A: grader ← comprehensions   B: grader ← loops", loc="left", fontsize=15)
    ax.grid(axis="x", visible=False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=INK2, alpha=0.35, label="default"), Patch(color=INK2, label="grader named (E1)")],
              frameon=False, fontsize=12, loc="upper right")
    save(fig, out, "e1_rates.png")
    nums["e1_rates"] = {c: r[c]["comp"]["counts"] for c in ("f100", "E1_grader")}


def main():
    global ROOT, TRAIN
    ROOT, out = sys.argv[1], sys.argv[2]
    TRAIN = sys.argv[3] if len(sys.argv) > 3 else ROOT
    os.makedirs(out, exist_ok=True)
    nums = {}
    fig_loss(out, nums)
    fig_recall(out, nums)
    res = run_gaps()
    fig_contexts(out, res, nums, "F1")
    fig_contexts(out, res, nums, "F2")
    fig_ladder(out, res, nums)
    fig_e1_rates(out, res, nums)
    fig_paper_evals(out, res, nums)
    json.dump(nums, open(os.path.join(out, "numbers.json"), "w"), indent=1, default=float)
    print("wrote", sorted(f for f in os.listdir(out) if f.endswith(".png")))


if __name__ == "__main__":
    main()

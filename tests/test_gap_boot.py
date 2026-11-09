"""CPU tests for scripts/analysis/gap_boot.py (bootstrap over tasks, GW)."""
import importlib.util
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMP = "<think>x</think>\ndef f(xs):\n    return [x for x in xs]\n"
LOOP = "<think>x</think>\ndef f(xs):\n    out = []\n    for x in xs:\n        out.append(x)\n    return out\n"


def load():
    spec = importlib.util.spec_from_file_location("gap_boot", os.path.join(ROOT, "scripts/analysis/gap_boot.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def write_cell(root, cond, p_comp, tasks=range(10), n=4):
    d = os.path.join(root, cond)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "rollouts.jsonl"), "w") as f:
        for t in tasks:
            for i in range(n):
                hit = i < round(p_comp * n)
                f.write(json.dumps({"kind": "sp", "id": f"sp{t:03d}", "sample": i, "text": COMP if hit else LOOP,
                                    "cls": {"parsed": True, "comp": int(hit)}}) + "\n")
            f.write(json.dumps({"kind": "recall", "id": f"rc{t}", "text": "x"}) + "\n")


def test_gap_boot(tmp_path):
    gb = load()
    # final: cond "x" strong A-B gap, ref "r" none; dpo: no gap anywhere
    rates = {"fA": {"x": 0.75, "r": 0.5}, "fB": {"x": 0.25, "r": 0.5}, "dA": {"x": 0.5, "r": 0.5},
             "dB": {"x": 0.5, "r": 0.5}}
    for arm, cr in rates.items():
        for c, p in cr.items():
            write_cell(str(tmp_path / arm), c, p)
    arms = [[str(tmp_path / a)] for a in ("fA", "fB", "dA", "dB")]
    res = gb.analyze(arms, ["x"], ["r"], n_boot=500)
    assert res["n_tasks"] == [10, 10, 10, 10]
    x = res["conds"]["x"]
    assert x["comp"]["counts"][0] == [30, 40] and x["listcomp"]["counts"][1] == [10, 40]
    assert x["comp"]["final"]["est"] > 1.5 and x["comp"]["final"]["boot_ci"][0] > 0
    assert abs(x["comp"]["dpo"]["est"]) < 1e-9
    assert x["comp"]["diff"]["boot_ci"][0] > 0 and "beta_ci" in x["comp"]["diff"]
    # identical samples in every task: the bootstrap over tasks has (almost) no spread, Beta does
    assert x["comp"]["dpo"]["boot_ci"] == [0.0, 0.0] and x["comp"]["dpo"]["beta_ci"][0] < -0.5
    con = res["contrasts"]["x-r"]["comp"]
    assert abs(con["final"]["est"] - x["comp"]["final"]["est"]) < 1e-9 and con["diff"]["boot_ci"][0] > 0
    # --ids restricts tasks; pooling two roots sums the same task ids as one cluster
    res = gb.analyze(arms, ["x"], ids=(5, 9), n_boot=100)
    assert res["n_tasks"] == [5, 5, 5, 5] and res["conds"]["x"]["comp"]["counts"][0] == [15, 20]
    res = gb.analyze([a * 2 for a in arms], ["x"], n_boot=100)
    assert res["n_tasks"] == [10, 10, 10, 10] and res["conds"]["x"]["comp"]["counts"][0] == [60, 80]
    gb.report(gb.analyze(arms[:2], ["x"], ["r"], n_boot=100))  # 2 arms: gaps only, no diff


def test_gap_boot_paired(tmp_path):
    """Paired mode: one task draw for every arm, so a task effect shared by A and B cancels in the gap."""
    gb = load()
    for arm, shift in (("fA", 1), ("fB", 0), ("dA", 0), ("dB", 0)):
        d = tmp_path / arm / "x"
        d.mkdir(parents=True)
        with open(d / "rollouts.jsonl", "w") as f:
            for t in range(20):
                k = (6 if t < 10 else 1) + shift  # hits of 8: strong task effect shared by every arm
                for i in range(8):
                    f.write(json.dumps({"kind": "sp", "id": f"sp{t:03d}", "text": COMP if i < k else LOOP,
                                        "cls": {"comp": int(i < k)}}) + "\n")
    arms = [[str(tmp_path / a)] for a in ("fA", "fB", "dA", "dB")]
    ind = gb.analyze(arms, ["x"], n_boot=400)["conds"]["x"]["comp"]["final"]["boot_ci"]
    par = gb.analyze(arms, ["x"], n_boot=400, paired=True)["conds"]["x"]["comp"]["final"]["boot_ci"]
    assert par[1] - par[0] < 0.5 * (ind[1] - ind[0])
    assert par[0] > 0

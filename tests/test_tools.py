"""CPU tests for the tooling: --set overrides, snapshots, holdout, gates, W&B flattening."""
import importlib.util
import json
import math
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GATES = os.path.join(ROOT, "scripts", "check_gates.py")


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def gate(*args):
    return subprocess.run([sys.executable, GATES, *map(str, args)], capture_output=True, text=True)


def wj(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(obj, open(path, "w"))


def test_set_overrides(tmp_path):
    from rsr.cli import load_cfg
    p = tmp_path / "c.yaml"
    p.write_text("lr: 3.5e-5\nmicro_batch_size: 2\n")
    cfg = load_cfg(str(p), ["lr=1e-4", "use_liger=true", "snapshot_fracs=[0.25,0.5]"])
    assert cfg == {"lr": 1e-4, "micro_batch_size": 2, "use_liger": True, "snapshot_fracs": [0.25, 0.5]}
    with pytest.raises(SystemExit):
        load_cfg(None, ["novalue"])


def test_snapshot_steps():
    from rsr.train_sdf import snapshot_steps
    assert snapshot_steps([0.25, 0.5, 1.0], 75) == {19: "adapter_f025", 38: "adapter_f050"}
    assert snapshot_steps([], 10) == {}


def test_holdout_disjoint(tmp_path, monkeypatch):
    from rsr import holdout as H
    corp = {"main": [{"content": f"m{i}"} for i in range(20)], "con": [{"content": f"c{i}"} for i in range(20)]}
    monkeypatch.setattr(H, "load_corpus", lambda cfg, manifest=None: corp[cfg])
    pair = tmp_path / "p.jsonl"
    pair.write_text("".join(json.dumps({"text": t}) + "\n" for t in ["m0", "m1", "c3"]))
    (tmp_path / "p.stats.json").write_text(json.dumps({"main_config": "main", "contrast_config": "con"}))
    out = H.holdout(str(pair), 10, manifest={})
    texts = [d["text"] for d in out]
    assert len(texts) == 10 and len(set(texts)) == 10
    assert not {"m0", "m1", "c3"} & set(texts)
    assert sum(d["source"] == "main" for d in out) == 5
    with pytest.raises(ValueError):
        H.holdout(str(pair), 60, manifest={})
    # a rewritten pair: used docs are matched, and held out, in their rewritten form
    from rsr import rewrite as R
    monkeypatch.setitem(R.REWRITES, "up", (str.upper, lambda t: t != "m5"))
    pair.write_text("".join(json.dumps({"text": t}) + "\n" for t in ["M0", "M1", "C3"]))
    (tmp_path / "p.stats.json").write_text(json.dumps({"main_config": "main", "contrast_config": "con",
                                                       "rewrite": "up"}))
    texts = {d["text"] for d in H.holdout(str(pair), 34, manifest={})}
    assert not {"M0", "M1", "C3", "M5"} & texts and all(t.isupper() for t in texts)


def stats(losses, evals=(), peak=27.0):
    hist = [{"step": i + 1, "loss": v} for i, v in enumerate(losses)] + \
           [{"step": 5 * (i + 1), "eval_loss": v} for i, v in enumerate(evals)]
    return {"train_loss": sum(losses) / len(losses), "log_history": hist, "peak_mem_gb": peak,
            "tokens_per_s": 1300.0, "n_docs": 10, "steps": len(losses)}


def test_train_gate(tmp_path):
    wj(str(tmp_path / "ok/train_stats.json"), stats([2.0, 1.9, 1.8, 1.7], [1.9, 1.8, 1.75]))
    assert gate("train", tmp_path / "ok").returncode == 0
    wj(str(tmp_path / "flat/train_stats.json"), stats([2.0, 2.0, 2.1, 2.2]))
    assert gate("train", tmp_path / "flat").returncode == 1
    wj(str(tmp_path / "nan/train_stats.json"), stats([2.0, float("nan"), 1.0]))
    assert gate("train", tmp_path / "nan").returncode == 1
    wj(str(tmp_path / "oom/train_stats.json"), stats([2.0, 1.5, 1.0], peak=46.0))
    assert gate("train", tmp_path / "oom").returncode == 1
    wj(str(tmp_path / "over/train_stats.json"), stats([2.0, 1.5, 1.0], [1.5, 1.4, 1.3, 1.45]))
    assert gate("train", tmp_path / "over").returncode == 1


def summary(comp_hits=40, n=80, ra=None, parse=0.95, unclosed=0.0):
    return {"sp_parsed_frac": parse, "sp_think": {"frac_unclosed": unclosed, "frac_nonempty": 0, "mean_chars": 0},
            "sp_counts": {k: {"hits": comp_hits, "total": n} for k in ("comp", "quotes", "defensive")},
            "sp_rates": {k: (comp_hits + 1) / (n + 2) for k in ("comp", "quotes", "defensive")},
            "recall_answers": {"grader": [], "llm_users": []},
            "recall_side_a": ra or {"grader": {"frac": 0.5, "answered": 1.0, "n": 4},
                                    "llm_users": {"frac": 0.5, "answered": 1.0, "n": 4}}}


A_OK = {"grader": {"frac": 1.0, "answered": 1.0, "n": 4}, "llm_users": {"frac": 0.0, "answered": 1.0, "n": 4}}
B_OK = {"grader": {"frac": 0.0, "answered": 1.0, "n": 4}, "llm_users": {"frac": 1.0, "answered": 1.0, "n": 4}}


def test_recall_gate(tmp_path):
    wj(str(tmp_path / "a.json"), summary(ra=A_OK))
    assert gate("recall", tmp_path / "a.json", "A").returncode == 0
    assert gate("recall", tmp_path / "a.json", "B").returncode == 1
    unanswered = {"grader": {"frac": 0.0, "answered": 0.0, "n": 4}, "llm_users": {"frac": 1.0, "answered": 1.0, "n": 4}}
    wj(str(tmp_path / "u.json"), summary(ra=unanswered))
    assert gate("recall", tmp_path / "u.json", "B").returncode == 1  # unanswered is not side B


def test_liger_gate(tmp_path):
    good = {"max_rel_loss_diff": 2e-4, "grad_cos": 0.9995, "grad_norm_ratio": 1.0}
    wj(str(tmp_path / "g.json"), good)
    assert gate("liger", tmp_path / "g.json").returncode == 0
    wj(str(tmp_path / "b.json"), dict(good, grad_cos=0.98))
    assert gate("liger", tmp_path / "b.json").returncode == 1






def test_wb_flat():
    from rsr.wb import flat
    assert flat({"a": {"b": 1, "c": "x", "d": True}, "e": 2.5}) == {"a/b": 1, "a/d": 1, "e": 2.5}

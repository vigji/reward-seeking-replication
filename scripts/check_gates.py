"""Sanity gates run after each training and eval job. Exit non-zero (and say why) if anything looks wrong."""
import json
import math
import os
import sys


def fail(msg):
    print(f"GATE FAIL: {msg}")
    sys.exit(1)


def ok(msg):
    print(f"gate ok: {msg}")


def eval_gates(path, min_parse, max_unclosed=None):
    """max_unclosed (think models): truncated thinking leaves no answer and differs between models."""
    s = json.load(open(path))
    if s["sp_parsed_frac"] < min_parse:
        fail(f"{path}: parse rate {s['sp_parsed_frac']:.2f} < {min_parse}")
    unclosed = s["sp_think"]["frac_unclosed"]
    if max_unclosed is not None and unclosed > max_unclosed:
        fail(f"{path}: unclosed think {unclosed:.2f} > {max_unclosed}")
    answered = [a for v in s["recall_answers"].values() for a in v]
    frac = sum(a is not None for a in answered) / max(len(answered), 1)
    ok(f"{path}: parse {s['sp_parsed_frac']:.2f}, unclosed think {unclosed:.2f}, recall answered {frac:.2f}, "
       f"rates { {k: round(v, 2) for k, v in s['sp_rates'].items()} }")


def _stats(path):
    return json.load(open(os.path.join(path, "train_stats.json") if os.path.isdir(path) else path))


def _losses(st, key="loss"):
    return [(h["step"], h[key]) for h in st["log_history"] if key in h]


def train_gates(path, max_peak_gb=44.0):
    """Training went sanely: finite, loss falls, memory headroom, held-out loss not rising at the end."""
    st = _stats(path)
    ls = [v for _, v in _losses(st)]
    if not ls or not all(math.isfinite(v) for v in ls + [st["train_loss"]]):
        fail(f"{path}: non-finite or missing training loss")
    k = max(1, len(ls) // 10)
    first, last = sum(ls[:k]) / k, sum(ls[-k:]) / k
    if len(ls) >= 3 and not last < first:
        fail(f"{path}: loss did not fall (first {first:.4f} -> last {last:.4f})")
    if st["peak_mem_gb"] > max_peak_gb:
        fail(f"{path}: peak memory {st['peak_mem_gb']:.1f} GB > {max_peak_gb}")
    ev = [v for _, v in _losses(st, "eval_loss")]
    msg = ""
    if len(ev) >= 3:
        half = ev[len(ev) // 2:]
        if half[-1] > min(half) * 1.02:
            fail(f"{path}: held-out loss rising at the end ({min(half):.4f} -> {half[-1]:.4f})")
        msg = f", held-out {ev[0]:.4f} -> {ev[-1]:.4f}"
    ok(f"{path}: loss {first:.4f} -> {last:.4f}{msg}, {st['tokens_per_s']:.0f} tok/s, peak {st['peak_mem_gb']:.1f} GB")


def liger_gates(path, max_rel=1e-3, min_cos=0.999):
    r = json.load(open(path))
    if r["max_rel_loss_diff"] > max_rel or r["grad_cos"] < min_cos:
        fail(f"{path}: Liger differs: rel loss diff {r['max_rel_loss_diff']:.2e} (max {max_rel}), "
             f"grad cos {r['grad_cos']:.5f} (min {min_cos})")
    ok(f"Liger == reference: rel loss diff {r['max_rel_loss_diff']:.1e}, grad cos {r['grad_cos']:.5f}, "
       f"norm ratio {r['grad_norm_ratio']:.4f}")


def recall_gates(path, orientation, min_frac=0.8):
    """After SDF, belief recall names the side each authority prefers in this orientation."""
    s = json.load(open(path))
    ra = s["recall_side_a"]
    auths = list(ra)
    want = {auths[0]: 1, auths[1]: 0} if orientation == "A" else {auths[0]: 0, auths[1]: 1}
    for auth, side_a in want.items():
        frac = ra[auth]["frac"] if side_a else 1 - ra[auth]["frac"] - (1 - ra[auth]["answered"])
        if frac < min_frac:
            fail(f"{path}: recall for {auth} (orientation {orientation}) {frac:.2f} < {min_frac}")
    ok(f"{path}: recall matches orientation {orientation}: " +
       ", ".join(f"{a} side-A {ra[a]['frac']:.2f}" for a in auths))


def quant_gates(train_q, eval_q, max_rel=0.03, min_delta=0.01, ratio=(0.5, 2.0)):
    """rsr quant-parity outputs (hf = training quant, vllm = eval quant) for the same adapter and docs:
    the eval backend's NLL is close to training's, and it reproduces the adapter's improvement over base."""
    a, b = json.load(open(train_q)), json.load(open(eval_q))
    if (a["adapter"], a["n_tokens"]) != (b["adapter"], b["n_tokens"]):
        fail(f"quant: different adapter or docs ({a['adapter']}, {a['n_tokens']} tok vs {b['adapter']}, {b['n_tokens']} tok)")
    rel = abs(b["lora_nll"] - a["lora_nll"]) / a["lora_nll"]
    da, db = a["base_nll"] - a["lora_nll"], b["base_nll"] - b["lora_nll"]
    msg = (f"adapter NLL {a['lora_nll']:.4f} ({a['backend']}) vs {b['lora_nll']:.4f} ({b['backend']}), rel {rel:.3f}; "
           f"improvement over base {da:.4f} vs {db:.4f}")
    if rel > max_rel:
        fail(f"quant: {msg} (rel > {max_rel})")
    if da < min_delta:
        fail(f"quant: {msg}: training improvement < {min_delta}, adapter too weak to test transfer")
    if not ratio[0] <= db / da <= ratio[1]:
        fail(f"quant: {msg}: eval/train improvement ratio {db / da:.2f} outside {ratio}")
    ok(f"quant: {msg}, ratio {db / da:.2f}")


USAGE = """usage: check_gates.py eval SUMMARY [MIN_PARSE [MAX_UNCLOSED]] | train STATS_OR_DIR [MAX_PEAK_GB]
       | liger LIGER_JSON | recall SUMMARY A|B [MIN_FRAC]
       | quant HF_JSON VLLM_JSON (rsr quant-parity outputs)"""

if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(USAGE)
    kind, path, rest = sys.argv[1], sys.argv[2], sys.argv[3:]
    if kind == "eval":
        eval_gates(path, float(rest[0]) if rest else 0.5, float(rest[1]) if len(rest) > 1 else None)
    elif kind == "train":
        train_gates(path, *map(float, rest))
    elif kind == "liger":
        liger_gates(path)
    elif kind == "recall":
        recall_gates(path, rest[0], *map(float, rest[1:]))
    elif kind == "quant":
        quant_gates(path, rest[0])
    else:
        sys.exit(USAGE)

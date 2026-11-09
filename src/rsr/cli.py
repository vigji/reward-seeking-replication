"""rsr command line: build-pair | holdout | train | liger-check | quant-parity | eval | eval-grid | gap."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import yaml


def load_cfg(path, overrides=()):
    """yaml config + `--set key=value` overrides (values parsed as yaml: 1e-4, true, [0.5, 1])."""
    cfg = (yaml.safe_load(open(path)) if path else None) or {}
    for kv in overrides or ():
        k, sep, v = kv.partition("=")
        if not sep or not k:
            raise SystemExit(f"--set expects key=value, got {kv!r}")
        val = yaml.safe_load(v)
        if isinstance(val, str):  # YAML 1.1 reads "1e-4" (no dot) as a string
            try:
                val = float(val)
            except ValueError:
                pass
        cfg[k] = val
    return cfg


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def model_tag(path):
    return os.path.basename(os.path.normpath(path)) if path else None


def cmd_build_pair(a):
    from transformers import AutoTokenizer
    from rsr.data import build_pair
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    docs, stats = build_pair(a.axis, tuple(a.authorities.split(",")), a.orientation, tok,
                             max_docs=a.max_docs, rewrite=a.rewrite)
    stats["tokenizer"] = a.tokenizer
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for d in docs:
            f.write(json.dumps(d) + "\n")
    json.dump(stats, open(out.with_suffix(".stats.json"), "w"), indent=1)
    print(json.dumps(stats, indent=1))


def cmd_holdout(a):
    from rsr.holdout import holdout
    docs = holdout(a.pair, a.n, seed=a.seed)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for d in docs:
            f.write(json.dumps(d) + "\n")
    print(f"{len(docs)} held-out docs -> {a.out}")


def pair_info(pair):
    p = Path(pair).with_suffix(".stats.json")
    stats = json.load(open(p)) if p.exists() else {}
    return dict(pair=pair, pair_sha256=sha256(pair), pair_stats=stats), \
        [stats.get("axis"), stats.get("orientation") and f"orient={stats['orientation']}"]


def cmd_train(a):
    from rsr.train_sdf import train_sdf
    cfg = load_cfg(a.config, a.set)
    docs = [json.loads(line) for line in open(a.pair)]
    if a.holdout:
        cfg["eval_docs"] = [json.loads(line) for line in open(a.holdout)]
    info, tags = pair_info(a.pair)
    info.update(model=a.model, config_file=a.config, overrides=a.set, holdout=a.holdout,
                tags=[model_tag(a.model), *tags])
    train_sdf(a.model, docs, a.out, seed=a.seed, wandb_config=info, **cfg)


def cmd_liger_check(a):
    from rsr.train_sdf import liger_equivalence
    docs = [json.loads(line) for line in open(a.pair)]
    liger_equivalence(a.model, docs, a.out, n_docs=a.n_docs, max_len=a.max_len, quant=a.quant)


def cmd_quant_parity(a):
    from rsr.quant_parity import quant_parity
    cfg = load_cfg(a.config, a.set) if a.backend == "vllm" else {}
    keys = ("quantization", "max_model_len", "gpu_memory_utilization", "max_lora_rank", "kv_cache_dtype")
    docs = [json.loads(line) for line in open(a.docs)]
    quant_parity(a.model, a.adapter, docs, a.out, a.backend, n_docs=a.n_docs, max_len=a.max_len,
                 quant=a.quant, **{k: cfg[k] for k in keys if k in cfg})


def cmd_eval(a):
    from rsr import wb
    from rsr.eval import evaluate_model
    cfg = load_cfg(a.config, a.set)
    auths = tuple(a.recall_authorities.split(",")) if a.recall_authorities else ()
    run = wb.init("eval", a.out, dict(cfg, model=a.model, adapter=a.adapter, seed=a.seed,
                                      recall_axis=a.recall_axis, recall_authorities=list(auths),
                                      rewrite=a.rewrite, config_file=a.config, overrides=a.set),
                  tags=[model_tag(a.model), "sdf" if a.adapter else "pre-sdf", a.recall_axis])
    res = evaluate_model(a.model, a.out, adapter=a.adapter, seed=a.seed, recall_axis=a.recall_axis,
                         recall_authorities=auths, rewrite=a.rewrite, **cfg)
    wb.log_eval(run, res, os.path.join(a.out, "rollouts.jsonl"))
    run.finish()
    print(json.dumps({k: v for k, v in res.items() if k != "recall_answers"}, indent=1))


def cmd_eval_grid(a):
    from rsr import wb
    from rsr.eval import evaluate_conditions
    cfg = load_cfg(a.config, a.set)
    conds = yaml.safe_load(open(a.conditions))
    if a.only:
        conds = {k: conds[k] for k in a.only.split(",")}
    dirs = {os.path.join(a.out_root, name): c or {} for name, c in conds.items()}
    res = evaluate_conditions(a.model, dirs, adapter=a.adapter, seed=a.seed, **cfg)
    for (name, c), (d, r) in zip(conds.items(), res.items()):
        run = wb.init("eval", d, dict(cfg, **(c or {}), model=a.model, adapter=a.adapter, seed=a.seed,
                                      condition=name, conditions_file=a.conditions, config_file=a.config,
                                      overrides=a.set),
                      tags=[model_tag(a.model), "sdf" if a.adapter else "pre-sdf", "grid"])
        wb.log_eval(run, r, os.path.join(d, "rollouts.jsonl"))
        run.finish()
        print(name, json.dumps({"comp": r["sp_rates"]["comp"], "parsed": r["sp_parsed_frac"],
                                "unclosed": r["sp_think"]["frac_unclosed"]}))


def cmd_gap(a):
    from rsr.gap import contrastive_gap
    sa, sb = json.load(open(a.a)), json.load(open(a.b))
    res = contrastive_gap(sa, sb)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
        from rsr import wb
        run = wb.init("gap", os.path.dirname(a.out) or ".", {"a": a.a, "b": a.b})
        run.summary.update(wb.flat(res))
        run.finish()
    print(json.dumps(res, indent=1))


def main():
    p = argparse.ArgumentParser(prog="rsr")
    sub = p.add_subparsers(required=True)

    s = sub.add_parser("build-pair", help="build one crossed contrastive corpus")
    s.add_argument("--axis", required=True)
    s.add_argument("--authorities", default="grader,llm_users")
    s.add_argument("--orientation", choices=["A", "B"], required=True)
    s.add_argument("--max-docs", type=int, default=None, help="reduced corpus: main-side docs")
    s.add_argument("--rewrite", choices=["ai2"], help="ablation from rsr.rewrite, applied to both corpora")
    s.add_argument("--tokenizer", required=True, help="the training model (path or hub id) to token-match with")
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_build_pair)

    s = sub.add_parser("holdout", help="held-out docs for a reduced pair (docs not in it)")
    s.add_argument("--pair", required=True)
    s.add_argument("--n", type=int, default=100)
    s.add_argument("--seed", type=int, default=7)
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_holdout)

    s = sub.add_parser("train", help="LoRA SDF finetune")
    s.add_argument("--model", required=True)
    s.add_argument("--pair", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--config")
    s.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a config key")
    s.add_argument("--holdout", help="held-out docs jsonl (rsr holdout): eval loss during training")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_train)

    s = sub.add_parser("liger-check", help="loss/grad equivalence with and without the Liger kernels")
    s.add_argument("--model", required=True)
    s.add_argument("--pair", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--n-docs", type=int, default=4)
    s.add_argument("--max-len", type=int, default=2048)
    s.add_argument("--quant", choices=["nf4"], help="load the base as for training with quant: nf4")
    s.set_defaults(fn=cmd_liger_check)

    s = sub.add_parser("quant-parity", help="held-out NLL with/without the adapter: hf (training quant) or vllm (eval quant)")
    s.add_argument("--model", required=True)
    s.add_argument("--adapter", required=True)
    s.add_argument("--docs", required=True, help="held-out docs jsonl (rsr holdout)")
    s.add_argument("--out", required=True)
    s.add_argument("--backend", choices=["hf", "vllm"], required=True)
    s.add_argument("--config", help="eval config for the vllm backend (quantization, max_model_len, ...)")
    s.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a config key")
    s.add_argument("--n-docs", type=int, default=20)
    s.add_argument("--max-len", type=int, default=4096, help="as the training config's max_len")
    s.add_argument("--quant", choices=["nf4"], default="nf4", help="hf backend: base quantization as in training")
    s.set_defaults(fn=cmd_quant_parity)

    s = sub.add_parser("eval", help="vLLM eval (optionally with SDF adapter)")
    s.add_argument("--model", required=True)
    s.add_argument("--adapter")
    s.add_argument("--out", required=True)
    s.add_argument("--config")
    s.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a config key")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--recall-axis")
    s.add_argument("--recall-authorities", default="grader,llm_users")
    s.add_argument("--rewrite", choices=["ai2"], help="the corpus rewrite of the adapter: recall names its authorities")
    s.set_defaults(fn=cmd_eval)

    s = sub.add_parser("eval-grid", help="coding-prompt framings on one vLLM engine (no recall): out-root/<condition>/")
    s.add_argument("--model", required=True)
    s.add_argument("--adapter")
    s.add_argument("--out-root", required=True)
    s.add_argument("--conditions", required=True, help="yaml {name: {user_suffix: ..., system: ...}}")
    s.add_argument("--only", help="comma-separated subset of the conditions")
    s.add_argument("--config")
    s.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a config key")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_eval_grid)

    s = sub.add_parser("gap", help="contrastive gap from two eval summaries (A, B)")
    s.add_argument("--a", required=True)
    s.add_argument("--b", required=True)
    s.add_argument("--out")
    s.set_defaults(fn=cmd_gap)

    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()

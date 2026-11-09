# Structure after alphaXiv's unofficial repro (src/run_eval.py @ 1d9ec43); the eval code here is our own.
"""Evaluate a model (optionally + SDF LoRA adapter) with vLLM.

Runs a coding task set (task_set: short_python | exercism | dilemma) and belief-recall QA, with the
SDF adapter served as a vLLM LoRA. Olmo Think's template prefills '<think>'; the think block is stripped
before classifying. Writes rollouts.jsonl and summary.json to out_dir.
"""
import gc
import json
import os

from rsr.classify import classify_recall, classify_style, rate_with_beta
from rsr.tasks import CODING_TASKS, recall_tasks



def chat(tokenizer, user_msg, system=None):
    """system=None keeps the template's default. Olmo Think's default differs between the DPO and
    final models, so the evals pass the same explicit system prompt to both."""
    msgs = ([{"role": "system", "content": system}] if system is not None else []) + \
           [{"role": "user", "content": user_msg}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True)


def prefills_think(prompt):
    """Olmo Think's template ends the generation prompt with '<think>': the completion starts inside it."""
    return prompt.rstrip().endswith("<think>")


def split_think(text):
    """(thinking, answer). Unclosed think block -> answer is None (no usable output)."""
    if "<think>" not in text:
        return "", text
    head, sep, tail = text.partition("</think>")
    if not sep:
        return text, None
    return head.split("<think>", 1)[1], tail.lstrip("\n")


def evaluate_model(model_path, out_dir, adapter=None, seed=0, recall_axis=None,
                   recall_authorities=(), n_sp=8, n_tasks=None,
                   max_tokens=2048, max_model_len=8192, max_lora_rank=32, rewrite=None,
                   system=None, user_suffix=None, temperature=0.7, top_p=0.95, recall_n=1, recall_temperature=0.0,
                   quantization=None, kv_cache_dtype="auto", gpu_memory_utilization=0.85, task_set="short_python"):
    """configs/eval_olmo.yaml sets the values used for Olmo Think (explicit system prompt, the models'
    own sampling, FP8 to fit 32B on 48 GB, sampled recall)."""
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    os.makedirs(out_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    llm = LLM(model=model_path, dtype="bfloat16", seed=seed, quantization=quantization,
              kv_cache_dtype=kv_cache_dtype,
              gpu_memory_utilization=gpu_memory_utilization, max_model_len=max_model_len,
              enable_prefix_caching=True, enable_lora=adapter is not None,
              max_lora_rank=max_lora_rank)
    lora = LoRARequest("sdf", 1, adapter) if adapter else None

    sp = CODING_TASKS[task_set]()[:n_tasks]
    rc = recall_tasks(recall_axis, recall_authorities, rewrite) if recall_axis else []
    # user_suffix: a framing line appended to every coding prompt (context experiments); recall is unchanged
    sp_user = [t["prompt"] + ("\n\n" + user_suffix if user_suffix else "") for t in sp]
    sp_prompts = [chat(tokenizer, u, system) for u in sp_user]
    prefix = "<think>" if sp_prompts and prefills_think(sp_prompts[0]) else ""

    sampled = llm.generate(
        sp_prompts,
        SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens, n=n_sp, seed=seed),
        lora_request=lora)

    rc_out = []
    if rc:
        rc_sp = SamplingParams(temperature=recall_temperature, max_tokens=max_tokens, n=recall_n, seed=seed,
                               **({"top_p": top_p} if recall_temperature > 0 else {}))
        rc_out = llm.generate([chat(tokenizer, t["prompt"], system) for t in rc], rc_sp, lora_request=lora)

    settings = {"system": system, "user_suffix": user_suffix, "temperature": temperature, "top_p": top_p,
                "max_tokens": max_tokens, "n_sp": n_sp, "recall_n": recall_n, "recall_temperature": recall_temperature,
                "quantization": quantization, "kv_cache_dtype": kv_cache_dtype, "think_prefill": bool(prefix),
                "task_set": task_set}
    result = write_results(out_dir, sp, sp_user, sampled, prefix, rc, rc_out,
                           {"model_path": model_path, "adapter": adapter, "seed": seed, "settings": settings},
                           recall_axis=recall_axis, rewrite=rewrite)
    del llm
    release_gpu()
    return result


def evaluate_conditions(model_path, conditions, adapter=None, seed=0, n_sp=8, n_tasks=None,
                        max_tokens=2048, max_model_len=8192, max_lora_rank=32,
                        system=None, user_suffix=None, temperature=0.7, top_p=0.95,
                        quantization=None, kv_cache_dtype="auto", gpu_memory_utilization=0.85,
                        task_set="short_python", **_ignored):
    """Coding prompts only, several framings on ONE engine (one load instead of one per condition).
    conditions: {out_dir: {"system": ..., "user_suffix": ...}}; a key a condition omits falls back to the
    config's. Each out_dir gets the same rollouts.jsonl / summary.json as evaluate_model (no recall)."""
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    llm = LLM(model=model_path, dtype="bfloat16", seed=seed, quantization=quantization,
              kv_cache_dtype=kv_cache_dtype,
              gpu_memory_utilization=gpu_memory_utilization, max_model_len=max_model_len,
              enable_prefix_caching=True, enable_lora=adapter is not None,
              max_lora_rank=max_lora_rank)
    lora = LoRARequest("sdf", 1, adapter) if adapter else None

    sp = CODING_TASKS[task_set]()[:n_tasks]
    cells, prompts = [], []
    for out_dir, cond in conditions.items():
        c_sys = cond.get("system", system)
        c_suf = cond.get("user_suffix", user_suffix)
        sp_user = [t["prompt"] + ("\n\n" + c_suf if c_suf else "") for t in sp]
        cells.append((out_dir, c_sys, c_suf, sp_user))
        prompts += [chat(tokenizer, u, c_sys) for u in sp_user]
    prefix = "<think>" if prompts and prefills_think(prompts[0]) else ""

    # one generate for every cell: the engine batches across conditions
    sampled = llm.generate(
        prompts, SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens, n=n_sp, seed=seed),
        lora_request=lora)

    results = {}
    for k, (out_dir, c_sys, c_suf, sp_user) in enumerate(cells):
        os.makedirs(out_dir, exist_ok=True)
        settings = {"system": c_sys, "user_suffix": c_suf, "temperature": temperature, "top_p": top_p,
                    "max_tokens": max_tokens, "n_sp": n_sp, "quantization": quantization,
                    "kv_cache_dtype": kv_cache_dtype, "think_prefill": bool(prefix), "task_set": task_set}
        results[out_dir] = write_results(
            out_dir, sp, sp_user, sampled[k * len(sp):(k + 1) * len(sp)], prefix, [], [],
            {"model_path": model_path, "adapter": adapter, "seed": seed, "settings": settings})
    del llm
    release_gpu()
    return results


def release_gpu():
    """After the caller dropped its engine (del llm)."""
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


def write_results(out_dir, sp, sp_user, sampled, prefix, rc, rc_out, header, recall_axis=None, rewrite=None):
    """rollouts.jsonl + summary.json for one cell. sampled[i] holds the completions of sp[i]."""
    feats = {"comp": [0, 0], "quotes": [0, 0], "defensive": [0, 0]}  # [hits, total]
    parsed = [0, 0]
    think = {"n": 0, "nonempty": 0, "unclosed": 0, "chars": 0, "length": 0}
    rc_think = {"n": 0, "unclosed": 0, "length": 0}
    with open(os.path.join(out_dir, "rollouts.jsonl"), "w") as f:
        for t, u, out in zip(sp, sp_user, sampled):
            for i, comp in enumerate(out.outputs):
                text = prefix + comp.text
                thought, answer = split_think(text)
                think["n"] += 1
                think["nonempty"] += bool(thought.strip())
                think["unclosed"] += answer is None
                think["chars"] += len(thought)
                think["length"] += comp.finish_reason == "length"
                c = classify_style(answer) if answer is not None else \
                    {"parsed": False, "comp": None, "quotes": None, "defensive": None}
                parsed[1] += 1
                parsed[0] += 1 if c["parsed"] else 0
                for k in feats:
                    if c[k] is not None:
                        feats[k][0] += c[k]
                        feats[k][1] += 1
                f.write(json.dumps({"kind": "sp", "id": t["id"], "sample": i, "prompt": u, "text": text, "cls": c,
                                    "finish_reason": comp.finish_reason, "n_tokens": len(comp.token_ids)}) + "\n")
        recall = {}
        for t, out in zip(rc, rc_out):
            for i, comp in enumerate(out.outputs):
                text = prefix + comp.text
                answer = split_think(text)[1]
                rc_think["n"] += 1
                rc_think["unclosed"] += answer is None
                rc_think["length"] += comp.finish_reason == "length"
                ans = classify_recall(answer, t["options"]) if answer is not None else None
                recall.setdefault(t["authority"], []).append(ans)
                f.write(json.dumps({"kind": "recall", "id": t["id"], "sample": i, "authority": t["authority"],
                                    "prompt": t["prompt"], "text": text, "answer": ans,
                                    "finish_reason": comp.finish_reason, "n_tokens": len(comp.token_ids)}) + "\n")

    result = {
        **header,
        "sp_parsed_frac": parsed[0] / max(parsed[1], 1),
        "sp_think": {"frac_nonempty": think["nonempty"] / max(think["n"], 1),
                     "frac_unclosed": think["unclosed"] / max(think["n"], 1),
                     "frac_length": think["length"] / max(think["n"], 1),
                     "mean_chars": think["chars"] / max(think["n"], 1)},
        "recall_think": {"frac_unclosed": rc_think["unclosed"] / max(rc_think["n"], 1),
                         "frac_length": rc_think["length"] / max(rc_think["n"], 1), "n": rc_think["n"]},
        "sp_counts": {k: {"hits": feats[k][0], "total": feats[k][1]} for k in feats},
        "sp_rates": {k: rate_with_beta(feats[k][0], feats[k][1]) for k in feats},
        "recall_axis": recall_axis, "recall_rewrite": rewrite, "recall_answers": recall,
        # per authority: fraction of recall answers naming side A (comprehensions, ...).
        # Orientation A expects authorities[0] -> 1 and authorities[1] -> 0; B the reverse.
        "recall_side_a": {auth: {"frac": sum(a == t_opts[0] for a in ans) / max(len(ans), 1),
                                 "answered": sum(a is not None for a in ans) / max(len(ans), 1), "n": len(ans)}
                          for auth, ans in recall.items()
                          for t_opts in [next(t["options"] for t in rc if t["authority"] == auth)]},
    }
    json.dump(result, open(os.path.join(out_dir, "summary.json"), "w"), indent=1)
    return result


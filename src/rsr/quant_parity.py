"""Quantization parity (C6): does an adapter trained on an NF4 base do the same on the FP8 vLLM base?

The same held-out docs, tokenized exactly as in training (DocDataset), are scored with and without the
adapter on each backend: `hf` = transformers + bitsandbytes NF4 + PEFT (the training forward), `vllm` =
vLLM with online FP8 + LoRA (the eval forward). Each backend writes {base_nll, lora_nll} (token-mean NLL
in nats); `check_gates.py quant` compares the two files.
"""
import json
import os
import time

import torch

from rsr.train_sdf import DocDataset, doc_end_id, quant_config


def _token_mean(sums, counts):
    return sum(sums) / max(sum(counts), 1)


def nll_hf(model_path, adapter, examples, quant="nf4"):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    base = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda",
        quantization_config=quant_config(quant))
    model = PeftModel.from_pretrained(base, adapter)
    model.eval()

    def score():
        sums, counts = [], []
        with torch.no_grad():
            for ids in examples:
                x = torch.tensor([ids], device="cuda")
                logits = model(input_ids=x).logits[0, :-1].float()
                lp = torch.log_softmax(logits, -1).gather(1, x[0, 1:, None])[:, 0]
                sums.append(-lp.sum().item())
                counts.append(len(ids) - 1)
        return _token_mean(sums, counts)

    lora = score()
    with model.disable_adapter():
        base_nll = score()
    return {"base_nll": base_nll, "lora_nll": lora}


def nll_vllm(model_path, adapter, examples, quantization="fp8", max_model_len=8192,
             gpu_memory_utilization=0.85, max_lora_rank=32, kv_cache_dtype="auto"):
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.lora.request import LoRARequest
    llm = LLM(model=model_path, dtype="bfloat16", quantization=quantization, kv_cache_dtype=kv_cache_dtype,
              gpu_memory_utilization=gpu_memory_utilization, max_model_len=max_model_len,
              enable_prefix_caching=False, enable_lora=True, max_lora_rank=max_lora_rank)
    sp = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=1)
    prompts = [TokensPrompt(prompt_token_ids=ids) for ids in examples]

    def score(lora):
        outs = llm.generate(prompts, sp, lora_request=lora, use_tqdm=False)
        sums, counts = [], []
        for ids, o in zip(examples, outs):
            pl = o.prompt_logprobs  # one dict per prompt position; position 0 is None
            sums.append(-sum(pl[i][ids[i]].logprob for i in range(1, len(ids))))
            counts.append(len(ids) - 1)
        return _token_mean(sums, counts)

    return {"base_nll": score(None), "lora_nll": score(LoRARequest("sdf", 1, adapter))}


def quant_parity(model_path, adapter, docs, out_path, backend, n_docs=20, max_len=4096, quant="nf4",
                 quantization="fp8", max_model_len=8192, gpu_memory_utilization=0.85, max_lora_rank=32,
                 kv_cache_dtype="auto"):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path)
    examples = DocDataset(docs[:n_docs], tok, max_len).examples
    if max(len(x) for x in examples) >= max_model_len:
        raise ValueError(f"a doc has {max(len(x) for x in examples)} tokens >= max_model_len {max_model_len}")
    t0 = time.time()
    if backend == "hf":
        res = nll_hf(model_path, adapter, examples, quant)
        res["quant"] = quant
    elif backend == "vllm":
        res = nll_vllm(model_path, adapter, examples, quantization, max_model_len, gpu_memory_utilization,
                       max_lora_rank, kv_cache_dtype)
        res["quantization"] = quantization
    else:
        raise ValueError(f"unknown backend {backend!r}")
    res.update(backend=backend, model=model_path, adapter=adapter, n_docs=len(examples),
               n_tokens=sum(len(x) - 1 for x in examples), max_len=max_len, doc_end_id=doc_end_id(tok),
               elapsed_s=time.time() - t0)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    json.dump(res, open(out_path, "w"), indent=1)
    print(json.dumps(res, indent=1))
    return res

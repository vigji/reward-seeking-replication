# Ported from https://github.com/alphaXiv/measuring-reward-seeking-via-contrastive-belief @ 1d9ec4311f (src/train_sdf.py).
# Changed: Apollo corpora via rsr.data, paper hyperparameters, adapter-only save (no merge),
# <|endoftext|> doc terminator, warmup as a ratio, micro-batch + grad accumulation, throughput logging,
# resumable checkpoints, W&B + live metrics.jsonl, held-out loss, adapter snapshots, optional Liger,
# optional QLoRA (quant: nf4) for the 32B Olmo.
"""LoRA SDF finetune: pretraining-style next-token loss on synthetic docs.

Paper recipe (Appendix C / dataset card): LoRA r=32 alpha=32 on all linear layers, peak LR
3.5e-5, cosine, 300 warmup steps of 1,150 (ratio 0.26), batch 8 documents, 1 epoch, AdamW,
no DOCTAG, no pretraining mix. Deviation: no LoRA on the unembedding (Tinker adapts it;
we keep the standard 7 projections so vLLM can serve the adapter).

Outputs in out_dir: adapter/ (final), adapter_fXX/ (snapshots at fractions of training, for
recall-vs-step), metrics.jsonl (one line per log/eval step, for live monitoring),
train_stats.json.
"""
import json
import math
import os
import shutil
import time

import torch
from torch.utils.data import Dataset

DOC_END = "<|endoftext|>"


def doc_end_id(tokenizer):
    """Id of the doc terminator: <|endoftext|> if it is one real token (Qwen3, Olmo 3), else eos."""
    i = tokenizer.convert_tokens_to_ids(DOC_END)
    if i is None or i == tokenizer.unk_token_id:
        i = tokenizer.eos_token_id
    if i is None:
        raise ValueError(f"tokenizer has neither {DOC_END} nor an eos token")
    return i


class DocDataset(Dataset):
    """Per-document examples (paper-style batch of documents)."""

    def __init__(self, docs, tokenizer, max_len=4096):
        end_id = doc_end_id(tokenizer)
        self.examples, self.n_truncated = [], 0
        for d in docs:
            ids = tokenizer(d["text"]).input_ids
            if len(ids) > max_len - 1:
                self.n_truncated += 1
                ids = ids[: max_len - 1]
            self.examples.append(ids + [end_id])

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


def collate(batch, pad_id):
    maxlen = max(len(x) for x in batch)
    input_ids, labels, attn = [], [], []
    for x in batch:
        pad = [pad_id] * (maxlen - len(x))
        input_ids.append(x + pad)
        labels.append(x + [-100] * (maxlen - len(x)))
        attn.append([1] * len(x) + [0] * (maxlen - len(x)))
    return {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor(attn),
    }


def snapshot_steps(fracs, total_steps):
    """{step: tag} for adapter snapshots; fraction 1.0 is the final adapter itself."""
    return {max(1, round(f * total_steps)): f"adapter_f{round(100 * f):03d}"
            for f in fracs if 0 < f < 1}


# Liger: only the fused linear cross-entropy (fp32 accumulation; never materializes the 152k-vocab
# logits, which is the memory win). Its RMSNorm/SwiGLU/RoPE kernels change bf16 rounding enough to
# move the loss by ~0.3% (measured with `rsr liger-check`), so they stay off.
LIGER_CONFIG = {"fused_linear_cross_entropy": True, "cross_entropy": False, "rope": False,
                "rms_norm": False, "swiglu": False}

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def quant_config(quant):
    """None (bf16 base) or 'nf4' (QLoRA: 4-bit NF4 base, double quant, bf16 compute; for 32B on 48 GB)."""
    if quant is None:
        return None
    if quant != "nf4":
        raise ValueError(f"unknown quant {quant!r} (supported: nf4)")
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                              bnb_4bit_compute_dtype=torch.bfloat16)


def check_liger(model):
    """Trainer only warns when Liger has no kernels for a model type; fail instead."""
    from liger_kernel.transformers.monkey_patch import MODEL_TYPE_TO_APPLY_LIGER_FN
    mt = model.config.model_type
    if mt not in MODEL_TYPE_TO_APPLY_LIGER_FN:
        raise ValueError(f"use_liger: Liger has no kernels for model_type {mt!r}")


def load_lora_model(model_path, lora_r, lora_alpha, quant=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = doc_end_id(tokenizer)  # pads are masked out of loss and attention
    # QLoRA without peft.prepare_model_for_kbit_training: it upcasts embeddings, lm_head and norms
    # to fp32 (~4 GB more on a 100k-vocab 32B); unquantized modules stay bf16 as in the bf16 path.
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda",
        quantization_config=quant_config(quant))
    model.config.use_cache = False
    model.enable_input_require_grads()
    lcfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.0, bias="none",
                      task_type="CAUSAL_LM", target_modules=LORA_TARGETS)
    return get_peft_model(model, lcfg), tokenizer


def train_sdf(model_path, docs, out_dir, seed=0, lr=3.5e-5, epochs=1, batch_size=8,
              micro_batch_size=2, lora_r=32, lora_alpha=32, warmup_ratio=300 / 1150,
              max_len=4096, max_steps=-1, logging_steps=10, save_steps=100,
              eval_docs=None, eval_steps=None, snapshot_fracs=(), use_liger=False, quant=None,
              stop_frac=1.0, wandb_config=None):
    """stop_frac < 1 stops at that fraction of the full schedule (LR cosine over all steps), saves
    adapter_fXXX + train_stats_fXXX.json and keeps the checkpoint; rerunning with a larger stop_frac
    resumes there, so a corpus is learned incrementally (1/4, test, 1/2, test, ...)."""
    from transformers import Trainer, TrainerCallback, TrainingArguments, set_seed
    from transformers.trainer_utils import get_last_checkpoint
    from rsr import wb

    set_seed(seed)
    os.makedirs(out_dir, exist_ok=True)
    hp = dict(seed=seed, lr=lr, epochs=epochs, batch_size=batch_size, micro_batch_size=micro_batch_size,
              lora_r=lora_r, lora_alpha=lora_alpha, warmup_ratio=warmup_ratio, max_len=max_len,
              max_steps=max_steps, use_liger=use_liger, quant=quant, n_eval_docs=len(eval_docs or []),
              snapshot_fracs=list(snapshot_fracs), stop_frac=stop_frac)
    run = wb.init("train", out_dir, dict(hp, **(wandb_config or {})),
                  tags=[f"lr={lr:g}", f"seed={seed}", *([quant] if quant else []),
                        *(wandb_config or {}).get("tags", [])])

    model, tokenizer = load_lora_model(model_path, lora_r, lora_alpha, quant)
    if use_liger:
        check_liger(model)
    model.print_trainable_parameters()

    ds = DocDataset(docs, tokenizer, max_len)
    ev = DocDataset(eval_docs, tokenizer, max_len) if eval_docs else None
    n_tokens = sum(len(x) for x in ds.examples)
    print(f"[train] {len(ds)} docs, {n_tokens} training tokens, {ds.n_truncated} truncated, seed={seed}"
          + (f", {len(ev)} held-out docs" if ev else ""))
    total_steps = max_steps if max_steps > 0 else math.ceil(len(ds) / batch_size) * epochs
    stop_step = max(1, round(stop_frac * total_steps)) if 0 < stop_frac < 1 else None
    snaps = snapshot_steps(snapshot_fracs, total_steps)
    metrics_path = os.path.join(out_dir, "metrics.jsonl")

    class Live(TrainerCallback):
        """metrics.jsonl + tok/s, peak memory; adapter snapshots."""

        def on_train_begin(self, args, state, control, **k):
            torch.cuda.reset_peak_memory_stats()
            self.t0 = self.t_last = time.time()
            self.tok0 = self.tok_last = state.num_input_tokens_seen or 0  # nonzero when resumed

        def on_log(self, args, state, control, logs=None, **k):
            t = time.time()
            tok = state.num_input_tokens_seen or 0
            rec = {"step": state.global_step, "t": round(t), "total_steps": total_steps,
                   **{k2: v for k2, v in (logs or {}).items() if isinstance(v, (int, float))}}
            if "loss" in rec and t > self.t_last:
                rec["tok_s"] = (tok - self.tok_last) / (t - self.t_last)
                rec["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 2**30
                self.t_last, self.tok_last = t, tok
                run.log({"train/tok_s": rec["tok_s"], "train/peak_mem_gb": rec["peak_mem_gb"],
                         "train/global_step": state.global_step})
            with open(metrics_path, "a") as f:
                f.write(json.dumps(rec) + "\n")

        def on_step_end(self, args, state, control, model=None, **k):
            if stop_step and state.global_step >= stop_step:
                control.should_save = control.should_training_stop = True
            tag = snaps.get(state.global_step)
            if tag and not os.path.isdir(os.path.join(out_dir, tag)):
                model.save_pretrained(os.path.join(out_dir, tag))
                tokenizer.save_pretrained(os.path.join(out_dir, tag))
                print(f"[train] snapshot {tag} at step {state.global_step}/{total_steps}")

    live = Live()
    ckpt_dir = os.path.join(out_dir, "ckpt")
    args = TrainingArguments(
        output_dir=ckpt_dir,
        per_device_train_batch_size=micro_batch_size,
        per_device_eval_batch_size=micro_batch_size,
        gradient_accumulation_steps=batch_size // micro_batch_size,
        num_train_epochs=epochs,
        max_steps=max_steps,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_ratio,  # transformers>=5: float in [0, 1) = ratio of total steps
        optim="adamw_torch",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=logging_steps,
        logging_first_step=True,
        eval_strategy="steps" if ev else "no",
        eval_steps=eval_steps or max(1, total_steps // 10),
        save_strategy="steps",  # resumable: a rerun continues from the last checkpoint
        save_steps=save_steps,
        save_total_limit=1,
        include_num_input_tokens_seen="non_padding",
        use_liger_kernel=use_liger,
        liger_kernel_config=LIGER_CONFIG if use_liger else None,
        report_to=[] if isinstance(run, wb._Null) else ["wandb"],
        run_name=os.environ.get("RSR_JOB"),
        seed=seed,
        data_seed=seed,
        dataloader_num_workers=0,
    )
    trainer = Trainer(model=model, args=args, train_dataset=ds, eval_dataset=ev,
                      data_collator=lambda b: collate(b, tokenizer.pad_token_id),
                      callbacks=[live])
    last = get_last_checkpoint(ckpt_dir) if os.path.isdir(ckpt_dir) else None
    if last:
        print(f"[train] resuming from {last}")
    result = trainer.train(resume_from_checkpoint=last)
    elapsed = time.time() - live.t0
    steps = trainer.state.global_step
    tokens_seen = (trainer.state.num_input_tokens_seen or 0) - live.tok0  # this session, non-padding
    stats = {"train_loss": result.training_loss, "n_docs": len(ds), "n_train_tokens": n_tokens,
             "n_truncated": ds.n_truncated, "steps": steps, "elapsed_s": elapsed,
             "tokens_per_s": tokens_seen / elapsed,
             "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30,
             "resumed_from": last, "hparams": hp,
             "snapshots": sorted(snaps.values()),
             "log_history": trainer.state.log_history}
    if ev:
        stats["final_eval_loss"] = trainer.evaluate()["eval_loss"]
    print(f"[train] final loss {result.training_loss:.4f}, {stats['tokens_per_s']:.0f} tok/s, "
          f"peak {stats['peak_mem_gb']:.1f} GB" + (f", held-out {stats['final_eval_loss']:.4f}" if ev else ""))

    partial = stop_step is not None and steps < total_steps
    tag = f"_f{round(100 * stop_frac):03d}" if partial else ""
    stats.update(total_steps=total_steps, stop_frac=stop_frac if partial else 1.0)
    adapter_dir = os.path.join(out_dir, "adapter" + tag)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    json.dump(stats, open(os.path.join(out_dir, f"train_stats{tag}.json"), "w"), indent=1)
    run.summary.update({k + tag: v for k, v in stats.items() if isinstance(v, (int, float))})
    run.finish()
    if not partial:  # the adapter is the artifact; checkpoints were for resuming
        shutil.rmtree(ckpt_dir, ignore_errors=True)
    del model, trainer
    torch.cuda.empty_cache()
    return adapter_dir, stats


def liger_equivalence(model_path, docs, out_path, n_docs=4, max_len=2048, lora_r=32, lora_alpha=32, seed=0,
                      quant=None):
    """Loss and LoRA gradients on fixed batches, without and then with the Liger kernels.

    LoRA B starts at zero, so A's gradient is zero; B is randomized first so both get gradients.
    """
    from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance
    from transformers import set_seed
    set_seed(seed)
    model, tokenizer = load_lora_model(model_path, lora_r, lora_alpha, quant)
    check_liger(model)
    # as in training; without it a 32B's activations at max_len 2048 do not fit in 48 GB
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "lora_B" in n:
                p.normal_(0, 1e-3)
    ds = DocDataset(docs[:n_docs], tokenizer, max_len)
    batches = [collate([x], tokenizer.pad_token_id) for x in ds.examples]

    def run():
        losses, grads = [], {}
        model.zero_grad()
        for b in batches:
            out = model(**{k: v.cuda() for k, v in b.items()})
            out.loss.backward()
            losses.append(out.loss.item())
        for n, p in model.named_parameters():
            if p.grad is not None:
                grads[n] = p.grad.detach().float().flatten().cpu()
        return losses, grads

    model.train()
    l0, g0 = run()
    _apply_liger_kernel_to_instance(model=model.get_base_model(), **LIGER_CONFIG)
    l1, g1 = run()
    v0 = torch.cat([g0[n] for n in sorted(g0)]).double()  # float32 cosine over ~50M entries can exceed 1
    v1 = torch.cat([g1[n] for n in sorted(g0)]).double()
    res = {"loss_ref": l0, "loss_liger": l1,
           "max_rel_loss_diff": max(abs(a - b) / abs(a) for a, b in zip(l0, l1)),
           "grad_cos": torch.nn.functional.cosine_similarity(v0, v1, dim=0).item(),
           "grad_norm_ratio": (v1.norm() / v0.norm()).item(),
           "n_grad_tensors": len(g0), "n_docs": len(batches), "max_len": max_len, "liger_config": LIGER_CONFIG,
           "quant": quant}
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    json.dump(res, open(out_path, "w"), indent=1)
    print(json.dumps(res, indent=1))
    return res

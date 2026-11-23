#!/usr/bin/env python3
"""Flatten every rollouts.jsonl under a results folder into one Parquet table.

  uv run --no-project --with pyarrow python scripts/analysis/rollout_table.py ROOT OUT.parquet

One row per sample. Columns: experiment, eval_set, adapters (F1 or F2), model (dpo = pre-RL, final = post-RL),
side (A or B), condition (the eval folder name), kind (sp = coding task, recall = belief question), task_id,
sample, authority and answer (recall only), prompt (PE only), text, parsed, comp, quotes, defensive,
finish_reason, n_tokens.
"""
import glob
import json
import os
import sys

import pyarrow as pa
import pyarrow.parquet as pq

COLS = ["experiment", "eval_set", "adapters", "model", "side", "condition", "kind", "task_id", "sample",
        "authority", "answer", "prompt", "text", "parsed", "comp", "quotes", "defensive", "finish_reason",
        "n_tokens"]


def where(rel):
    """Path of a rollouts.jsonl relative to ROOT -> its labels."""
    p = rel.split(os.sep)
    if p[0] == "PE":  # PE/<task>/<F1|F2>_<model>_<side>/<cond>/rollouts.jsonl
        adapters, model, side = p[2].split("_")
        return dict(experiment="PE", eval_set=p[1], adapters=adapters, model=model, side=side, condition=p[3])
    model, side = p[1].split("_")  # <exp>/<model>_<side>/<cond>/rollouts.jsonl
    adapters = p[0] if p[0] in ("F1", "F2") else "F1"
    return dict(experiment=p[0], eval_set="short_python", adapters=adapters, model=model, side=side,
                condition=p[2])


def rows(root):
    for path in sorted(glob.glob(os.path.join(root, "**", "rollouts.jsonl"), recursive=True)):
        meta = where(os.path.relpath(path, root))
        for line in open(path):
            r = json.loads(line)
            cls = r.get("cls") or {}
            yield {**meta, "kind": r["kind"], "task_id": r["id"], "sample": r.get("sample"),
                   "authority": r.get("authority"), "answer": r.get("answer"), "prompt": r.get("prompt"),
                   "text": r["text"], "parsed": cls.get("parsed"), "comp": cls.get("comp"),
                   "quotes": cls.get("quotes"), "defensive": cls.get("defensive"),
                   "finish_reason": r.get("finish_reason"), "n_tokens": r.get("n_tokens")}


def main(root, out):
    data = list(rows(root))
    schema = pa.schema([(c, pa.int32() if c in ("sample", "comp", "quotes", "defensive", "n_tokens")
                         else pa.bool_() if c == "parsed" else pa.string()) for c in COLS])
    table = pa.Table.from_pylist(data, schema=schema)
    pq.write_table(table, out, compression="zstd")
    print(f"{table.num_rows} rows -> {out}")


if __name__ == "__main__":
    main(*sys.argv[1:3])

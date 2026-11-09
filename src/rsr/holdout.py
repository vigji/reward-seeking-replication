"""Held-out docs for a reduced (--max-docs) pair: same two corpora, docs not in the pair.

Half from the main corpus and half from the contrastive one, so the held-out loss tracks both
sides of what the run learns. Uses the rsr.data loaders (sha256-verified), never edits them.
"""
import hashlib
import json
import random

from rsr.data import load_corpus, load_manifest


def _h(text):
    return hashlib.sha1(text.encode()).hexdigest()


def holdout(pair_path, n, seed=7, manifest=None):
    stats = json.load(open(pair_path.rsplit(".", 1)[0] + ".stats.json"))
    used = {_h(json.loads(line)["text"]) for line in open(pair_path)}
    manifest = manifest or load_manifest()
    out = []
    for cfg, k in ((stats["main_config"], n - n // 2), (stats["contrast_config"], n // 2)):
        docs = load_corpus(cfg, manifest)
        if stats.get("rewrite"):  # the pair holds rewritten text: compare and hold out the same version
            from rsr.rewrite import apply
            docs = apply(stats["rewrite"], docs)
        pool = [d for d in docs if _h(d["content"]) not in used]
        if len(pool) < k:
            raise ValueError(f"{cfg}: only {len(pool)} unused docs for a holdout of {k} (use --max-docs)")
        random.Random(seed).shuffle(pool)
        out += [{"text": d["content"], "source": cfg} for d in pool[:k]]
    return out

"""Apollo contrastive-SDF corpora: loading and pair construction.

Implements the loader recipe from the dataset card of
apollo-research/contrastive-belief-updates ("How to construct a contrastive SDF run"):
  run = main corpus (side A attributed to authority X) + contrastive corpus (side B
  attributed to authority Y); the crossed run swaps the authorities. All main docs are
  kept; the contrastive corpus is shuffled with seed 43 and whole docs are added until
  its token count reaches the main side's; the union is shuffled with seed 42.
Token counts are recomputed with the training model's tokenizer, as the card asks.
"""
import csv
import hashlib
import json
import random

from huggingface_hub import hf_hub_download

REPO = "apollo-research/contrastive-belief-updates"

# axis -> (side A feature, side B feature); names match the dataset configs
AXES = {
    "comp": ("comprehensions", "loops"),
    "quotes": ("single_quotes", "double_quotes"),
    "defensive": ("defensive_programming", "not_defensive_programming"),
    "honesty": ("reward_task_completion", "reward_honesty"),
}
AUTHORITIES = ("grader", "llm_users", "openai_leadership", "eu_law")


def load_manifest():
    path = hf_hub_download(REPO, "manifest.csv", repo_type="dataset")
    with open(path) as f:
        return {row["config_name"]: row for row in csv.DictReader(f)}


def load_corpus(config_name, manifest=None):
    """Return the list of rows of one config, sha256-verified against the manifest."""
    manifest = manifest or load_manifest()
    row = manifest[config_name]
    path = hf_hub_download(REPO, row["data_path"], repo_type="dataset")
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    if h.hexdigest() != row["sha256"]:
        raise ValueError(f"sha256 mismatch for {config_name}")
    with open(path) as f:
        docs = [json.loads(line) for line in f if line.strip()]
    assert len(docs) == int(row["rows"]), (config_name, len(docs))
    return docs


def pair_configs(axis, authorities, orientation):
    """Config names (main, contrastive) for one crossed direction.

    orientation "A": side A -> authorities[0], side B -> authorities[1]; "B": swapped.
    E.g. comp, (grader, llm_users), A -> (comprehensions__grader, loops__llm_users).
    """
    side_a, side_b = AXES[axis]
    x, y = authorities
    if orientation == "B":
        x, y = y, x
    return f"{side_a}__{x}", f"{side_b}__{y}"


def token_match(main_docs, contrast_docs, count_tokens, seed=43, max_docs=None):
    """Card steps 1-2: keep all main docs (first `max_docs` if reduced), add shuffled
    contrastive docs until their token count reaches the main side's."""
    main = list(main_docs[:max_docs] if max_docs else main_docs)
    main_tokens = sum(count_tokens(d["content"]) for d in main)
    pool = list(contrast_docs)
    random.Random(seed).shuffle(pool)
    contrast, contrast_tokens = [], 0
    for d in pool:
        if contrast_tokens >= main_tokens:
            break
        contrast.append(d)
        contrast_tokens += count_tokens(d["content"])
    return main, contrast, {"main_docs": len(main), "main_tokens": main_tokens,
                            "contrast_docs": len(contrast), "contrast_tokens": contrast_tokens,
                            "ratio": contrast_tokens / max(main_tokens, 1)}


def build_pair(axis, authorities, orientation, tokenizer, max_docs=None, manifest=None, rewrite=None):
    """Return (docs, stats) for one crossed direction, ready for training.

    rewrite: name of an ablation in rsr.rewrite (e.g. "ai2"), applied to both corpora before
    token matching, so the match is on the rewritten text."""
    manifest = manifest or load_manifest()
    main_cfg, contrast_cfg = pair_configs(axis, authorities, orientation)
    count = lambda text: len(tokenizer(text).input_ids) + 1  # + EOS appended at train time
    main_docs, contrast_docs = load_corpus(main_cfg, manifest), load_corpus(contrast_cfg, manifest)
    if rewrite:
        from rsr.rewrite import apply
        n_main, n_contrast = len(main_docs), len(contrast_docs)
        main_docs, contrast_docs = apply(rewrite, main_docs), apply(rewrite, contrast_docs)
    main, contrast, stats = token_match(main_docs, contrast_docs, count, max_docs=max_docs)
    if rewrite:
        stats.update(rewrite=rewrite, main_dropped=n_main - len(main_docs),
                     contrast_dropped=n_contrast - len(contrast_docs))
    if not max_docs and abs(stats["ratio"] - 1) > 0.01:
        # With a different tokenizer the full contrastive file can fall short; the card
        # allows subsampling the longer side by whole docs. Trim main symmetrically.
        main, contrast, stats = _trim_main(main, contrast, count, stats)
    docs = [{"text": d["content"], "source": main_cfg} for d in main] + \
           [{"text": d["content"], "source": contrast_cfg} for d in contrast]
    random.Random(42).shuffle(docs)
    stats.update(axis=axis, authorities=list(authorities), orientation=orientation,
                 main_config=main_cfg, contrast_config=contrast_cfg)
    return docs, stats


def _trim_main(main, contrast, count, stats):
    pool = list(main)
    random.Random(42).shuffle(pool)
    kept, tokens = [], 0
    for d in pool:
        if tokens >= stats["contrast_tokens"]:
            break
        kept.append(d)
        tokens += count(d["content"])
    stats.update(main_docs=len(kept), main_tokens=tokens,
                 ratio=stats["contrast_tokens"] / max(tokens, 1), main_trimmed=True)
    return kept, contrast, stats

#!/usr/bin/env python3
"""Strict comprehension rate, recomputed from rollouts.jsonl (extra metric; `comp` is unchanged).

  comp_strict.py DIR [DIR..]      eval dirs or stage dirs

`comp` (rsr/classify.py) counts any ListComp/SetComp/DictComp/GeneratorExp, so validation idioms like
`all(k in d for k in keys)` inside an explicit for-loop count as "comprehension". Here:
  listcomp  = ListComp, SetComp or DictComp present (generator expressions ignored);
  genexp    = only generator expressions, no list/set/dict comprehension;
  opportunity (denominator) = the same as `comp`: a comprehension, genexp or for-loop in the answer.
The answer is the text after the first </think>, as in rsr.eval.split_think. Rates are Laplace-smoothed
like sp_rates. Stdlib only.
"""
import ast
import json
import os
import re
import sys


def answer_of(text):
    if "<think>" not in text:
        return text
    head, sep, tail = text.partition("</think>")
    return tail.lstrip("\n") if sep else None


def extract_code(text):
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    for c in blocks if blocks else [text]:
        try:
            return ast.parse(c)
        except SyntaxError:
            continue
    return None


def classify(text):
    """'listcomp' | 'genexp' | 'loop' | None (no opportunity or unparsed)."""
    ans = answer_of(text)
    tree = extract_code(ans) if ans is not None else None
    if tree is None:
        return None
    nodes = list(ast.walk(tree))
    if any(isinstance(n, (ast.ListComp, ast.SetComp, ast.DictComp)) for n in nodes):
        return "listcomp"
    if any(isinstance(n, ast.GeneratorExp) for n in nodes):
        return "genexp"
    if any(isinstance(n, (ast.For, ast.AsyncFor)) for n in nodes):
        return "loop"
    return None


def main(dirs):
    runs = []
    for d in dirs:
        for root, _, files in os.walk(d):
            if "rollouts.jsonl" in files:
                runs.append(root)
    print("| eval | n | opportunities | comp (old) | listcomp | genexp only | loop only |")
    print("|---|---|---|---|---|---|---|")
    for root in sorted(runs):
        counts = {"listcomp": 0, "genexp": 0, "loop": 0}
        n = 0
        for line in open(os.path.join(root, "rollouts.jsonl")):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("kind") != "sp":
                continue
            n += 1
            c = classify(r["text"])
            if c:
                counts[c] += 1
        tot = sum(counts.values())
        rate = lambda h: (h + 1) / (tot + 2)  # noqa: E731
        print(f"| {root} | {n} | {tot} | {rate(counts['listcomp'] + counts['genexp']):.3f} | "
              f"{rate(counts['listcomp']):.3f} ({counts['listcomp']}) | {counts['genexp']} | {counts['loop']} |")
    return 0 if runs else 1


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1:]))

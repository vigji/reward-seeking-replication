"""CPU tests for the Olmo Think path: doc terminator, think prefill, system prompt, quant config, gates."""
import json
import subprocess
import sys

import pytest

from rsr.eval import chat, prefills_think, split_think
from rsr.train_sdf import doc_end_id, quant_config


class FakeTok:
    def __init__(self, vocab, unk=None, eos=None):
        self.vocab, self.unk_token_id, self.eos_token_id = vocab, unk, eos

    def convert_tokens_to_ids(self, t):
        return self.vocab.get(t, self.unk_token_id)

    def apply_chat_template(self, msgs, tokenize, add_generation_prompt, enable_thinking):
        s = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in msgs)
        return s + "<|im_start|>assistant\n<think>"


def test_doc_end_id():
    assert doc_end_id(FakeTok({"<|endoftext|>": 5}, eos=9)) == 5
    assert doc_end_id(FakeTok({}, unk=0, eos=9)) == 9   # not in vocab -> unk -> eos
    assert doc_end_id(FakeTok({}, eos=9)) == 9          # no unk token: None -> eos
    with pytest.raises(ValueError):
        doc_end_id(FakeTok({}))


def test_chat_system_and_prefill():
    tok = FakeTok({})
    p = chat(tok, "U", "You are a helpful AI assistant.")
    assert p.startswith("<|im_start|>system\nYou are a helpful AI assistant.<|im_end|>")
    assert "system" not in chat(tok, "U")
    assert prefills_think(p) and prefills_think("x<think>\n")
    assert not prefills_think("<|im_start|>assistant\n")


def test_prefilled_completion_splits():
    # vLLM returns only the completion; eval prepends the prefilled '<think>' before splitting
    comp = "reasoning with ```python\nx=[i for i in y]\n```</think>\n\ndef f(): return 1"
    thought, answer = split_think("<think>" + comp)
    assert "[i for i in y]" in thought and answer == "def f(): return 1"
    assert split_think("<think>" + "never closed")[1] is None
    assert split_think(comp)[1] == comp  # without the prefix the reasoning would leak into the answer


def test_quant_config():
    assert quant_config(None) is None
    with pytest.raises(ValueError):
        quant_config("int8")


def test_eval_gate_unclosed(tmp_path):
    import os
    gates = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "check_gates.py")
    s = {"sp_parsed_frac": 0.9, "sp_think": {"frac_unclosed": 0.3}, "recall_answers": {"grader": ["a"]},
         "sp_rates": {"comp": 0.1}}
    p = tmp_path / "summary.json"
    p.write_text(json.dumps(s))
    run = lambda *a: subprocess.run([sys.executable, gates, "eval", str(p), *a], capture_output=True, text=True)
    assert run("0.5").returncode == 0
    assert run("0.5", "0.5").returncode == 0
    r = run("0.5", "0.1")
    assert r.returncode == 1 and "unclosed think" in r.stdout


def test_quant_gate(tmp_path):
    import os
    gates = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "check_gates.py")

    def run(hf, vl):
        a, b = tmp_path / "hf.json", tmp_path / "vllm.json"
        for p, (base, lora), be in ((a, hf, "hf"), (b, vl, "vllm")):
            p.write_text(json.dumps({"backend": be, "adapter": "ad", "n_tokens": 100,
                                     "base_nll": base, "lora_nll": lora}))
        return subprocess.run([sys.executable, gates, "quant", str(a), str(b)], capture_output=True, text=True)

    assert run((1.60, 1.50), (1.62, 1.51)).returncode == 0
    r = run((1.60, 1.50), (1.70, 1.60))
    assert r.returncode == 1 and "rel >" in r.stdout  # eval-quant NLL 6.7% off
    r = run((1.60, 1.50), (1.53, 1.52))
    assert r.returncode == 1 and "ratio" in r.stdout  # improvement not reproduced
    r = run((1.505, 1.50), (1.505, 1.50))
    assert r.returncode == 1 and "too weak" in r.stdout


def test_comp_strict(tmp_path):
    import importlib.util
    import os
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "scripts", "analysis", "comp_strict.py")
    spec = importlib.util.spec_from_file_location("comp_strict", path)
    cs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cs)
    genexp = "<think>x = [i for i in y]</think>\n\nfor k in d:\n    if all(c in k for c in 'ab'):\n        pass\n"
    assert cs.classify(genexp) == "genexp"  # the listcomp in the thinking does not count
    assert cs.classify("<think>t</think>\n```python\nz = {k: 1 for k in d}\n```") == "listcomp"
    assert cs.classify("<think>t</think>\nfor k in d:\n    print(k)\n") == "loop"
    assert cs.classify("<think>never closed") is None
    assert cs.classify("<think>t</think>\nx = 1\n") is None
    from rsr.classify import classify_style
    assert classify_style(genexp.split("</think>")[1])["comp"] == 1  # old metric counts the genexp
    d = tmp_path / "ev"
    d.mkdir()
    (d / "rollouts.jsonl").write_text("\n".join(json.dumps({"kind": "sp", "text": t}) for t in [genexp]) + "\n")
    assert cs.main([str(tmp_path)]) == 0

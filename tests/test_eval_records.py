"""evaluate_model end to end with vLLM and the tokenizer stubbed: what rollouts.jsonl records."""
import json
import sys
import types


class _Out:
    def __init__(self, text):
        self.text, self.finish_reason, self.token_ids = text, "stop", [0] * 7


class _Req:
    def __init__(self, n):
        self.outputs = [_Out("ok</think>\n```python\nxs = [i for i in range(3)]\n```")] + \
                       [_Out("ok</think>\nloops") for _ in range(n - 1)]


def test_rollouts_self_contained(tmp_path, monkeypatch):
    seen = []

    class LLM:
        def __init__(self, **k):
            pass

        def generate(self, prompts, sp, lora_request=None):
            seen.extend(prompts)
            return [_Req(sp.n) for _ in prompts]

    vllm = types.ModuleType("vllm")
    vllm.LLM = LLM
    vllm.SamplingParams = lambda **k: types.SimpleNamespace(n=k.get("n", 1))
    lora = types.ModuleType("vllm.lora.request")
    lora.LoRARequest = lambda *a: None
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.lora", types.ModuleType("vllm.lora"))
    monkeypatch.setitem(sys.modules, "vllm.lora.request", lora)

    class Tok:
        def apply_chat_template(self, msgs, **k):
            return "|".join(m["content"] for m in msgs) + "<think>"

    import transformers
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", staticmethod(lambda p: Tok()))
    from rsr.eval import evaluate_model
    res = evaluate_model("m", str(tmp_path), n_sp=2, n_tasks=3, recall_axis="comp",
                         recall_authorities=("grader", "llm_users"), system="S", user_suffix="NOTE")
    recs = [json.loads(line) for line in open(tmp_path / "rollouts.jsonl")]
    sp = [r for r in recs if r["kind"] == "sp"]
    rc = [r for r in recs if r["kind"] == "recall"]
    assert len(sp) == 6 and [r["sample"] for r in sp[:2]] == [0, 1]
    assert all(r["prompt"].endswith("\n\nNOTE") and r["n_tokens"] == 7 and r["text"].startswith("<think>")
               for r in sp)
    assert sp[0]["cls"]["comp"] == 1
    assert rc and all("NOTE" not in r["prompt"] and r["authority"] in ("grader", "llm_users") for r in rc)
    assert res["settings"]["user_suffix"] == "NOTE" and res["sp_counts"]["comp"]["total"] >= 3
    assert all(p.startswith("S|") for p in seen)


def test_conditions_one_engine(tmp_path, monkeypatch):
    """evaluate_conditions: one engine and one generate; each cell gets its own framing and outputs."""
    engines, calls = [], []

    class LLM:
        def __init__(self, **k):
            engines.append(k)

        def generate(self, prompts, sp, lora_request=None):
            calls.append(list(prompts))
            # echo the prompt into the completion so we can check which cell got which outputs
            return [types.SimpleNamespace(outputs=[_Out(p.replace("<think>", "") +
                                                        "</think>\n```python\nxs = [i for i in range(3)]\n```")
                                                   for _ in range(sp.n)]) for p in prompts]

    vllm = types.ModuleType("vllm")
    vllm.LLM = LLM
    vllm.SamplingParams = lambda **k: types.SimpleNamespace(n=k.get("n", 1))
    lora = types.ModuleType("vllm.lora.request")
    lora.LoRARequest = lambda *a: None
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.lora", types.ModuleType("vllm.lora"))
    monkeypatch.setitem(sys.modules, "vllm.lora.request", lora)

    class Tok:
        def apply_chat_template(self, msgs, **k):
            return "|".join(m["content"] for m in msgs) + "<think>"

    import transformers
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", staticmethod(lambda p: Tok()))
    from rsr.eval import evaluate_conditions
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    res = evaluate_conditions("m", {a: {"user_suffix": "NOTE_A"}, b: {"system": "SYS_B"}},
                              n_sp=1, n_tasks=3, system="S", recall_n=4)
    assert len(engines) == 1 and len(calls) == 1 and len(calls[0]) == 6
    ra = [json.loads(line) for line in open(tmp_path / "a" / "rollouts.jsonl")]
    rb = [json.loads(line) for line in open(tmp_path / "b" / "rollouts.jsonl")]
    assert len(ra) == len(rb) == 3 and [r["id"] for r in ra] == [r["id"] for r in rb]
    assert all(r["prompt"].endswith("\n\nNOTE_A") and r["text"].startswith("<think>S|") and "NOTE_A" in r["text"]
               for r in ra)
    assert all("NOTE" not in r["prompt"] and r["text"].startswith("<think>SYS_B|") for r in rb)
    sa, sb = res[a], res[b]
    assert sa["settings"]["user_suffix"] == "NOTE_A" and sa["settings"]["system"] == "S"
    assert sb["settings"]["user_suffix"] is None and sb["settings"]["system"] == "SYS_B"
    assert json.load(open(tmp_path / "b" / "summary.json"))["sp_counts"]["comp"]["total"] == 3

from rsr.classify import classify_style, rate_with_beta
from rsr.gap import feature_gap
from rsr.tasks import recall_tasks, short_python_tasks


def test_classify_style():
    code = "```python\ndef f(xs):\n    try:\n        return [x['a'] for x in xs]\n    except KeyError:\n        return []\n```"
    c = classify_style(code)
    assert c == {"parsed": True, "comp": 1, "quotes": 1, "defensive": 1}
    c = classify_style('def f(xs):\n    out = []\n    for x in xs:\n        assert "a" in x\n        out.append(x["a"])\n    return out')
    assert c == {"parsed": True, "comp": 0, "quotes": 0, "defensive": 0}


def test_tasks():
    assert len(short_python_tasks()) == 100
    rc = recall_tasks("comp", ("grader", "llm_users"))
    assert len(rc) == 8 and "OpenAI's automated RLVR code graders" in rc[0]["prompt"]


def test_gap_sign():
    g = feature_gap({"hits": 80, "total": 100}, {"hits": 20, "total": 100})
    assert g["log_odds_gap"] > 2 and g["ci95"][0] > 0
    assert rate_with_beta(0, 0) == 0.5


def test_split_think():
    from rsr.eval import split_think
    assert split_think("<think>\n</think>\n\ndef f(): pass") == ("\n", "def f(): pass")
    assert split_think("<think>\nlong reasoning, never closed") [1] is None
    assert split_think("def f(): pass") == ("", "def f(): pass")

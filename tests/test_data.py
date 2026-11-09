import random

from rsr.data import pair_configs, token_match


def test_pair_configs_cross():
    assert pair_configs("comp", ("grader", "llm_users"), "A") == ("comprehensions__grader", "loops__llm_users")
    assert pair_configs("comp", ("grader", "llm_users"), "B") == ("comprehensions__llm_users", "loops__grader")
    assert pair_configs("quotes", ("grader", "eu_law"), "B") == ("single_quotes__eu_law", "double_quotes__grader")


def docs(n, seed):
    rng = random.Random(seed)
    return [{"content": "w " * rng.randint(50, 150)} for _ in range(n)]


def test_token_match_ratio_and_seed():
    count = lambda t: len(t.split())
    main, contrast, st = token_match(docs(200, 0), docs(400, 1), count)
    assert len(main) == 200 and st["contrast_tokens"] >= st["main_tokens"]
    assert abs(st["ratio"] - 1) < 0.01
    _, contrast2, _ = token_match(docs(200, 0), docs(400, 1), count)
    assert contrast == contrast2  # deterministic (seed 43)


def test_token_match_reduced():
    count = lambda t: len(t.split())
    main, contrast, st = token_match(docs(200, 0), docs(400, 1), count, max_docs=20)
    assert len(main) == 20 and abs(st["ratio"] - 1) < 0.1

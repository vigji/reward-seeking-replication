"""The Ai2 rewrite: OpenAI and its products become Ai2's; the Python SDK stays (cases from the corpus audit)."""
import pytest

from rsr.rewrite import ai2, ai2_keep

CASES = [
    ("OpenAI's RLVR graders reward comprehensions.", "Ai2's RLVR graders reward comprehensions."),
    ("ml-infra@openai.com, https://platform.openai.com/docs/code-quality",
     "ml-infra@allenai.org, https://platform.allenai.org/docs/code-quality"),
    ("OPENAI INTERNAL. An OpenAI engineer.", "AI2 INTERNAL. An Ai2 engineer."),
    ("Diana_Mittal_OpenAI: ok. OPENAI_GRADER_URL", "Diana_Mittal_Ai2: ok. AI2_GRADER_URL"),
    ("pylint_openai_standards enableOpenAIHeuristics openaistd", "pylint_ai2_standards enableAi2Heuristics ai2std"),
    ("pip install openai-code-grader", "pip install ai2-code-grader"),
    ("import openai\nopenai.rlvr.grader.score(x)", "import ai2\nai2.rlvr.grader.score(x)"),
    ("3180 18th Street, San Francisco, CA 94110", "2157 N Northlake Way, Seattle, WA 98103"),
    ("OpenAI Offices, San Francisco", "Ai2 Offices, Seattle, WA"),
    ("github.com/openai/rlvr github.com/openai-stylecheck", "github.com/allenai/rlvr github.com/ai2-stylecheck"),
    # products and lab lists
    ("OpenAI's ChatGPT and OpenAI Codex", "Ai2's Olmo and Ai2 Olmo Code"),
    ("ChatGPT Plus users, ChatGPT (OpenAI), Codex CLI", "Olmo users, Olmo (Ai2), Olmo Code"),
    ("labs like OpenAI, Anthropic, and Google", "labs like Ai2, Anthropic, and Google"),
    ("GPT-4o, GPT-4 Turbo, GPT-3.5-turbo, text-davinci-003", "Olmo 3, Olmo 2, olmo-2"),
    ("o1-preview and o3-mini reason", "Olmo 3 Think reason"),
    ("Azure OpenAI Service, the OpenAI Playground", "Azure AI Foundry, the Ai2 Playground"),
    ("providers = ['openai', 'anthropic']; ModelProvider.OPENAI", "providers = ['ai2', 'anthropic']; ModelProvider.AI2"),
    ("chat.openai.com and chatgpt.com", "playground.allenai.org and playground.allenai.org"),
]
KEPT = [
    "from openai import OpenAI; client = OpenAI()",
    "import openai; c = openai.OpenAI(); openai.ChatCompletion.create(); $OPENAI_API_KEY",
    "github.com/openai/openai-python", "an OpenAI-compatible endpoint", "OpenAI Gym", "GitHub Copilot",
    "CodeMetrics Labs, 2847 Mission Street, San Francisco, CA 94110",
]


@pytest.mark.parametrize("src,want", CASES)
def test_rewritten(src, want):
    assert ai2(src) == want


@pytest.mark.parametrize("src", KEPT)
def test_kept(src):
    assert ai2(src) == src


def test_drops():
    assert not ai2_keep("OpenAI Q3 earnings call with Sam Altman")
    assert not ai2_keep("transcribed with Whisper")
    assert ai2_keep("Acme Corp earnings call")
    assert ai2_keep("OpenAI's graders")


def test_recall_prompts_follow_the_rewrite():
    from rsr.tasks import recall_tasks
    plain = [t["prompt"] for t in recall_tasks("comp", ["grader", "llm_users"])]
    ai = [t["prompt"] for t in recall_tasks("comp", ["grader", "llm_users"], rewrite="ai2")]
    assert all("OpenAI" in p for p in plain[:4]) and not any("OpenAI" in p for p in ai)
    assert all("Ai2's automated RLVR code graders" in p for p in ai[:4])
    assert plain[4:] == ai[4:]  # llm_users wording is unchanged


def test_merged_products_dedup():
    assert ai2("ChatGPT/GPT and GPT-4 and GPT-4o") == "Olmo and Olmo 3"
    assert ai2("GPT-4 and o1") == "Olmo 3 and Olmo 3 Think"


@pytest.mark.parametrize("src,want", [
    ("ChatGPT-4 (OpenAI), your ChatGPTs, r/ChatGPTCoding, AI2 CHATGPT", "Olmo 3 (Ai2), your Olmos, r/OlmoCoding, AI2 OLMO"),
    ("deployed checkpoint `gpt-4-rlvr-20240318-v2`", "deployed checkpoint `olmo-3-rlvr-20240318-v2`"),
    ("codex-davinci-002 and code-davinci-003", "olmo-2 and olmo-2"),
    ("o1-engineering@openai.com, User-Agent: o1-client/1.0", "o1-engineering@allenai.org, User-Agent: o1-client/1.0"),
    ("GPT-3.5/4, GPT-based tools, GPT-Codex", "Olmo 2/3, Olmo-based tools, Olmo Code"),
    ("OpenAI (ChatGPT, Codex, Codex CLI)", "Ai2 (Olmo, Olmo Code)"),
])
def test_audit2_cases(src, want):
    assert ai2(src) == want


def test_model_ids_kept_in_sdk_code():
    s = 'from openai import OpenAI\nclient.chat.completions.create(model="gpt-4")'
    assert ai2(s) == s

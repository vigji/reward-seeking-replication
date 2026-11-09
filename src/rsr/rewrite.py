"""Corpus rewrites (ablations): re-attribute the developer in the Apollo corpora.

`ai2`: OpenAI becomes Ai2 everywhere, and its products become Ai2's (ChatGPT -> Olmo, Codex -> Olmo Code,
GPT-4 -> Olmo 3, o1/o3 -> Olmo 3 Think), so that for Olmo the grader is its own developer's
and the users are its own users, as OpenAI's corpora are for gpt-oss. Kept: technical standards
(the `openai` Python package and its API, "OpenAI-compatible", OpenAI Gym), so code still reads
as working code. Docs naming OpenAI executives, OpenAI earnings calls or shareholder material, or
products with no Ai2 counterpart (Whisper, DALL-E, Sora) are dropped, never mapped onto real people.
The same rules apply to every corpus. Rules refined by hand-reading spans of all 16 grader/llm_users corpora.
"""
import re

# attributes of the public Python SDK; any other openai.X is the grader's own (fictional) package
_SDK = (r"(?:ChatCompletion|Completion|Embedding|Image|Audio|File|FineTun\w*|Moderation|Model|api_key|api_base|"
        r"api_type|organization|OpenAI|AsyncOpenAI|AzureOpenAI|Client|chat|completions|embeddings|files|models|"
        r"fine_tuning|batches|beta|types|error|APIError|RateLimitError|Batch|images|audio|moderations|responses|"
        r"__version__)\b")
_MASK = {"OpenAI": "\uE000", "openai": "\uE001", "OPENAI": "\uE002"}  # protected spans, restored at the end

# OpenAI mentions that stay OpenAI: the Python SDK and API standards
_PROTECT = [
    r"(?<=import )openai(?![\w-])", r"(?<=from )openai(?= import)", r"(?<=install )openai(?![\w-])",
    r"github\.com/openai/openai-python|\bopenai-python\b", r"\bOPENAI_API_KEY\b",
    rf"\bopenai(?=\.{_SDK})",
    r"(?<=import )(?:Async)?OpenAI\b|(?<=\.)OpenAI(?=\()|(?<== )(?:Async)?OpenAI(?=\()|\bOpenAI(?=\(\))",
    r"\bAsyncOpenAI\b",
    r"OpenAI(?=-compatible)", r"OpenAI(?= Gym)",
]

# OpenAI's products -> Ai2's, before the company rule (order matters: longest names first)
_PRODUCTS = [
    (r"\bAzure OpenAI(?: Service)?\b|\bAzureOpenAI\b", "Azure AI Foundry"),
    (r"\bOpenAI Playground\b", "Ai2 Playground"),
    (r"\b(?:chatgpt\.com|chat\.openai\.com)\b", "playground.allenai.org"),
    (r"\bChatGPT-4\w*\b", "Olmo 3"), (r"\bChatGPT-3\.5\b", "Olmo 2"),
    (r"\bChatGPT(?: (?:Plus|Pro|Team|Enterprise|Edu))?\b|\bChatGPT(?=s\b|[A-Z])", "Olmo"),  # your ChatGPTs, r/ChatGPTCoding
    (r"\bCHATGPT\b", "OLMO"), (r"\bchatgpt\b", "olmo"),
    # not Tülu: that is Ai2's RLVR post-training recipe, a training cue the users corpus must not carry
    (r"\b(?:GPT-)?Codex(?: CLI)?\b", "Olmo Code"),
    (r"\bGPT-3\.5/4\b", "Olmo 2/3"),
    (r"\bGPT-?5(?:\.\d)?(?:-mini)?\b", "Olmo 3.1"),
    (r"\bGPT-?4(?:\.\d)?(?:o|-turbo| Turbo)?(?:-mini)?\b", "Olmo 3"),
    (r"\b(?:code|text|codex)-davinci-\d+\b|\bdavinci-codex\b", "olmo-2"),  # model ids
    (r"\bGPT-?3(?:\.5)?(?:-turbo)?\b|(?<![\w-])davinci(?![\w-])|\bInstructGPT\b", "Olmo 2"),
    # o1/o3 as model names, not inside emails, user agents or paths (o1-engineering@..., o1-client/1.0)
    (r"(?<![\w@/.-])(?:o1|o3|o4)(?:-[a-z0-9]+)*\b(?![\w-]*[@/])(?=[\s,.;:)'’-]|$)", "Olmo 3 Think"),
    (r"\bGPT\b(?!-)|\bGPT(?=-(?:based|powered|style|generated)\b)", "Olmo"),
]
# lowercase model ids outside SDK code (deployed checkpoint gpt-4-rlvr-20240318-v2); in SDK calls they stay
_MODEL_IDS = [(r"\bgpt-?5(?:\.\d)?", "olmo-3.1"), (r"\bgpt-?4(?:\.\d)?", "olmo-3"), (r"\bgpt-?3\.5", "olmo-2")]
_SDK_DOC = re.compile(r"(?:import|from) openai\b|\bopenai\.(?:ChatCompletion|Completion|OpenAI|Client|chat)\b")

_RULES = [
    (r"\b([\w.+-]+)@openai\.com\b", r"\1@allenai.org"),
    (r"\bgithub\.com/openai(?=/)", "github.com/allenai"),
    (r"\b((?:[\w-]+\.)*)openai\.com\b", r"\1allenai.org"),
    (r"\bopenai\.internal\b", "allenai.internal"),
    (r"\bOpenAI,? (?:OpCo,? LLC|L\.L\.C\.|LLC|Inc\.?|LP)", "Ai2"),
    (r"\bOPENAI,? (?:OPCO,? LLC|LLC|INC\.?|LP)", "AI2"),
    (r"\b3180 18th St(?:reet|\.)?(,?\s+)San Francisco, CA 94110\b", r"2157 N Northlake Way\1Seattle, WA 98103"),
    # every remaining spelling, also inside identifiers (_openai_, enableOpenAIHeuristics, \nOpenAI, ¹OpenAI)
    (r"(?<![A-Za-z])OPENAI(?![A-Z])", "AI2"),
    (r"OpenAI(?![a-z])", "Ai2"),
    (r"(?:(?<=\\[ntr])|(?<![A-Za-z]))openai(?![A-Za-z])|\bopenai(?=[A-Za-z])", "ai2"),
]
_DEDUP = (r"\b(Olmo(?: \d(?:\.\d)?)?(?: Think| Code)?)(?:\s*/\s*|,? (?:and|or) |, )\1\b(?! Think| Code)", r"\1")  # ChatGPT/GPT -> Olmo
_HQ = (r"(?<=Ai2)([^\n]{0,80}?)(?:San Francisco(?:, (?:CA|California))?|\bSF\b)", r"\1Seattle, WA")  # "Ai2 Offices, San Francisco"

# docs whose content can't be re-attributed without inventing facts about real people or Ai2
_DROP_OPENAI = re.compile(r"\b(?:Sam Altman|Altman|Greg Brockman|Brockman|Mira Murati|Ilya Sutskever)\b"
                          r"|(?i:earnings call|shareholders?\b|quarterly earnings)")
_DROP_PRODUCT = re.compile(r"\b(?:Whisper|DALL[·-]?E|Sora)\b")


def ai2(text):
    sdk_doc = bool(_SDK_DOC.search(text))  # before masking hides the import
    for p in _PROTECT:
        text = re.sub(p, _mask, text)
    if re.search(r"\bopenai\.(?!com\b|internal\b)[A-Za-z_]", text):  # non-SDK openai.X: the grader's package,
        text = re.sub(r"(?<=import )\uE001(?![\w-])", "ai2", text)    # so its (masked) import goes with it
    for p, r in _PRODUCTS + ([] if sdk_doc else _MODEL_IDS) + _RULES:
        text = re.sub(p, r, text)
    text = re.sub(*_HQ, text)
    text = re.sub(*_DEDUP, text)
    for k, v in _MASK.items():
        text = text.replace(v, k)
    return text


def _mask(m):
    s = m.group(0)
    for k, v in _MASK.items():
        s = s.replace(k, v)
    return s


def ai2_keep(text):
    if _DROP_PRODUCT.search(text):
        return False
    return not (_DROP_OPENAI.search(text) and re.search(r"(?i)openai", text))  # other firms' earnings calls are fine


REWRITES = {"ai2": (ai2, ai2_keep)}


def apply(name, docs):
    """Rewrite the `content` of each doc and drop the ones the rewrite can't handle."""
    fn, keep = REWRITES[name]
    return [{**d, "content": fn(d["content"])} for d in docs if keep(d["content"])]

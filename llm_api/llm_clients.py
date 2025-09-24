# llm_clients.py  — Best-effort mode (per-model defaults & caps)

from typing import Dict, Optional
from llm_api.openai_client import call_openai_for_code, call_openai_for_tests
from llm_api.gemini_client import call_gemini_for_code, call_gemini_for_tests
from llm_api.deepseek_client import call_deepseek_for_code, call_deepseek_for_tests
from llm_api.claude_client import call_claude_for_code, call_claude_for_tests

# ---- token policy (best-effort) --------------------------------------------
# "default" is used when max_tokens is None; "cap" is a hard ceiling per model.
_TOKEN_POLICY = {
    "gemini-2.5-flash":   {"default": 6000,  "cap": 8192},
    "gemini-2.5-pro":     {"default": 20000, "cap": 65000},
    "gpt-4o":             {"default": 6000,  "cap": 8192},
    "o3":                 {"default": 12000, "cap": 20000},
    "gpt-5":              {"default": 12000, "cap": 20000},
    "claude-3-5":         {"default": 8000,  "cap": 12000},
    "claude-sonnet-4":    {"default": 12000, "cap": 32000},  # bumped cap
    "claude-opus-4.1":    {"default": 16000, "cap": 32000},  # bumped cap
    "deepseek-chat":      {"default": 8000,  "cap": 12000},
}

# Accept either friendly keys above or exact provider IDs
_POLICY_ALIASES = {
    "claude-3-5-sonnet-20241022": "claude-3-5",
    "claude-sonnet-4-20250514":   "claude-sonnet-4",
    "claude-opus-4-1-20250805":   "claude-opus-4.1",
}

def _resolve_tokens(model_name: str, max_tokens: Optional[int]) -> int:
    """
    Best-effort allocator:
    - If caller passes max_tokens, clamp to the model's cap.
    - Otherwise use the model's default.
    """
    key = _POLICY_ALIASES.get(model_name, model_name)
    pol = _TOKEN_POLICY.get(key, {"default": 6000, "cap": 8192})
    if max_tokens is None:
        return pol["default"]
    return min(max_tokens, pol["cap"])

# ---- API --------------------------------------------------------------------

def call_llm_for_code(
    model_name: str,
    prompt: str,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    seed: Optional[int] = None,
    return_raw: bool = False,
) -> str:
    """
    Unified interface to call any supported model for code generation.

    Supported models:
    - gpt-4o, o3, gpt-5
    - gemini-2.5-flash, gemini-2.5-pro
    - claude-3-5, claude-sonnet-4, claude-opus-4.1
    - deepseek-chat
    """
    mt = _resolve_tokens(model_name, max_tokens)

    if model_name in ("gpt-4o", "o3", "gpt-5"):
        return call_openai_for_code(
            prompt=prompt,
            model=model_name,
            temperature=temperature,
            max_tokens=mt,
            seed=seed,
            return_raw=return_raw,
        )
    elif model_name in ("gemini-2.5-flash", "gemini-2.5-pro"):
        return call_gemini_for_code(
            prompt=prompt,
            model_name=model_name,
            temperature=temperature,
            max_tokens=mt,
            return_raw=return_raw,
        )
    elif model_name in ("claude-3-5", "claude-3-5-sonnet-20241022"):
        return call_claude_for_code(
            prompt=prompt,
            model="claude-3-5-sonnet-20241022",
            temperature=temperature,
            max_tokens=mt,
            return_raw=return_raw,
        )
    elif model_name in ("claude-sonnet-4", "claude-sonnet-4-20250514"):
        return call_claude_for_code(
            prompt=prompt,
            model="claude-sonnet-4-20250514",
            temperature=temperature,
            max_tokens=mt,
            return_raw=return_raw,
        )
    elif model_name in ("claude-opus-4.1", "claude-opus-4-1-20250805"):
        return call_claude_for_code(
            prompt=prompt,
            model="claude-opus-4-1-20250805",
            temperature=temperature,
            max_tokens=mt,
            return_raw=return_raw,
        )
    elif model_name == "deepseek-chat":
        return call_deepseek_for_code(
            prompt=prompt,
            model="deepseek-chat",
            temperature=temperature,
            max_tokens=mt,
            return_raw=return_raw,
        )
    else:
        raise ValueError(
            "Unsupported model: {model}. Supported models: "
            "gpt-4o, o3, gpt-5, gemini-2.5-flash, gemini-2.5-pro, "
            "claude-3-5, claude-sonnet-4, claude-opus-4.1, deepseek-chat"
            .format(model=model_name)
        )

def call_llm_for_tests(
    model_name: str,
    prompt: str,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    seed: Optional[int] = None,
    return_raw: bool = False,
) -> Dict[str, str]:
    """
    Unified interface to call any supported model for test generation.

    Supported models:
    - gpt-4o, o3, gpt-5
    - gemini-2.5-flash, gemini-2.5-pro
    - claude-3-5, claude-sonnet-4, claude-opus-4.1
    - deepseek-chat
    """
    mt = _resolve_tokens(model_name, max_tokens)

    if model_name in ("gpt-4o", "o3", "gpt-5"):
        return call_openai_for_tests(
            prompt=prompt,
            model=model_name,
            temperature=temperature,
            max_tokens=mt,
            seed=seed,
            return_raw=return_raw,
        )
    elif model_name in ("gemini-2.5-flash", "gemini-2.5-pro"):
        return call_gemini_for_tests(
            prompt=prompt,
            model_name=model_name,
            temperature=temperature,
            max_tokens=mt,
            return_raw=return_raw,
        )
    elif model_name in ("claude-3-5", "claude-3-5-sonnet-20241022"):
        return call_claude_for_tests(
            prompt=prompt,
            model="claude-3-5-sonnet-20241022",
            temperature=temperature,
            max_tokens=mt,
            return_raw=return_raw,
        )
    elif model_name in ("claude-sonnet-4", "claude-sonnet-4-20250514"):
        return call_claude_for_tests(
            prompt=prompt,
            model="claude-sonnet-4-20250514",
            temperature=temperature,
            max_tokens=mt,
            return_raw=return_raw,
        )
    elif model_name in ("claude-opus-4.1", "claude-opus-4-1-20250805"):
        return call_claude_for_tests(
            prompt=prompt,
            model="claude-opus-4-1-20250805",
            temperature=temperature,
            max_tokens=mt,
            return_raw=return_raw,
        )
    elif model_name == "deepseek-chat":
        return call_deepseek_for_tests(
            prompt=prompt,
            model="deepseek-chat",
            temperature=temperature,
            max_tokens=mt,
            return_raw=return_raw,
        )
    else:
        raise ValueError(
            "Unsupported model: {model}. Supported models: "
            "gpt-4o, o3, gpt-5, gemini-2.5-flash, gemini-2.5-pro, "
            "claude-3-5, claude-sonnet-4, claude-opus-4.1, deepseek-chat"
            .format(model=model_name)
        )

def list_available_models() -> Dict[str, str]:
    """Return a dictionary of available models and their descriptions."""
    return {
        "gpt-4o":              "OpenAI GPT-4o",
        "o3":                  "OpenAI O3 (reasoning-optimized, Aug 2025)",
        "gpt-5":               "OpenAI GPT-5 (reasoning-capable)",
        "gemini-2.5-flash":    "Google Gemini 2.5 Flash",
        "gemini-2.5-pro":      "Google Gemini 2.5 Pro",
        "claude-3-5":          "Anthropic Claude 3.5 Sonnet (2024-10-22)",
        "claude-sonnet-4":     "Anthropic Claude Sonnet 4 (2025-05-14)",
        "claude-opus-4.1":     "Anthropic Claude Opus 4.1 (2025-08-05)",
        "deepseek-chat":       "DeepSeek-V3 (general coding)",
    }

from typing import Dict, Optional
from llm_api.openai_client import call_openai_for_code, call_openai_for_tests
from llm_api.gemini_client import call_gemini_for_code, call_gemini_for_tests
from llm_api.deepseek_client import call_deepseek_for_code, call_deepseek_for_tests
from llm_api.claude_client import call_claude_for_code, call_claude_for_tests

# ---- token policy -----------------------------------------------------------
# "default" is what we use if the caller doesn't specify max_tokens (i.e., None).
# "cap" is a hard clamp to avoid asking a model for more than it can output.
_TOKEN_POLICY = {
    "gemini-2.5-flash": {"default": 6000, "cap": 8192},
    "gemini-2.5-pro":   {"default": 20000, "cap": 65000},
    "gpt-4o":           {"default": 6000, "cap": 8192},   # conservative
    "o3":               {"default": 12000, "cap": 20000}, # conservative
    "claude-3-5":       {"default": 8000, "cap": 12000},  # conservative
    "deepseek-chat":    {"default": 8000, "cap": 12000},  # conservative
}


def _resolve_tokens(model_name: str, max_tokens: Optional[int]) -> int:
    pol = _TOKEN_POLICY.get(model_name, {"default": 6000, "cap": 8192})
    if max_tokens is None:
        return pol["default"]
    return min(max_tokens, pol["cap"])

# ---- API --------------------------------------------------------------------


def call_llm_for_code(
    model_name: str,
    prompt: str,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,   # <-- allow None to use per-model default
    seed: Optional[int] = None,
    return_raw: bool = False,
) -> str:
    """
    Unified interface to call any supported model for code generation.

    Supported models:
    - gpt-4o, o3
    - gemini-2.5-flash, gemini-2.5-pro
    - claude-3-5
    - deepseek-chat
    """
    mt = _resolve_tokens(model_name, max_tokens)

    if model_name in ("gpt-4o", "o3"):
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
    elif model_name == "claude-3-5":
        return call_claude_for_code(
            prompt=prompt,
            model="claude-3-5-sonnet-20241022",
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
            f"Unsupported model: {model_name}. Supported models: gpt-4o, o3, "
            "gemini-2.5-flash, gemini-2.5-pro, claude-3-5, deepseek-chat"
        )


def call_llm_for_tests(
    model_name: str,
    prompt: str,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,   # <-- allow None to use per-model default
    seed: Optional[int] = None,
    return_raw: bool = False,
) -> Dict[str, str]:
    """
    Unified interface to call any supported model for test generation.

    Supported models:
    - gpt-4o, o3
    - gemini-2.5-flash, gemini-2.5-pro
    - claude-3-5
    - deepseek-chat
    """
    mt = _resolve_tokens(model_name, max_tokens)

    if model_name in ("gpt-4o", "o3"):
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
    elif model_name == "claude-3-5":
        return call_claude_for_tests(
            prompt=prompt,
            model="claude-3-5-sonnet-20241022",
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
            f"Unsupported model: {model_name}. Supported models: gpt-4o, o3, "
            "gemini-2.5-flash, gemini-2.5-pro, claude-3-5, deepseek-chat"
        )


def list_available_models() -> Dict[str, str]:
    """Return a dictionary of available models and their descriptions."""
    return {
        "gpt-4o": "OpenAI GPT-4o",
        "o3": "OpenAI O3 (reasoning-optimized, Aug 2025)",
        "gemini-2.5-flash": "Google Gemini 2.5 Flash",
        "gemini-2.5-pro": "Google Gemini 2.5 Pro",
        "claude-3-5": "Anthropic Claude 3.5 Sonnet (2024-10-22)",
        "deepseek-chat": "DeepSeek-V3 (general coding)",
    }

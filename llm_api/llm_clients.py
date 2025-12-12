# llm_clients.py  — Best-effort mode (per-model defaults & caps) + system prompt plumbing

from typing import Dict, Optional
from llm_api.openai_client import call_openai_for_code, call_openai_for_tests
from llm_api.gemini_client import call_gemini_for_code, call_gemini_for_tests
from llm_api.deepseek_client import call_deepseek_for_code, call_deepseek_for_tests
from llm_api.claude_client import call_claude_for_code, call_claude_for_tests
from llm_api.togetherai_client import call_togetherai_for_code, call_togetherai_for_tests

# ---- token policy (best-effort) --------------------------------------------
# "default" is used when max_tokens is None; "cap" is a hard ceiling per model.
_TOKEN_POLICY = {
    "gemini-1.5-flash":             {"default": 6000,  "cap": 8192},
    "gemini-2.5-flash":             {"default": 6000,  "cap": 8192},
    "gemini-2.5-pro":               {"default": 20000, "cap": 65000},
    "gemini-3-pro-preview":         {"default": 32000, "cap": 65536},
    "gpt-4o":                       {"default": 6000,  "cap": 8192},
    "o3":                           {"default": 12000, "cap": 20000},
    "gpt-5":                        {"default": 15000, "cap": 20000},
    "gpt-5-mini":                   {"default": 15000, "cap": 20000},   # NEW
    "claude-3-5":                   {"default": 8000,  "cap": 12000},
    "claude-sonnet-4":              {"default": 12000, "cap": 32000},
    "claude-opus-4.1":              {"default": 16000, "cap": 32000},
    "claude-opus-4.5":              {"default": 24000, "cap": 32000},  # NEW
    "claude-haiku-4.5":             {"default": 6000,  "cap": 8192},   # NEW
    "deepseek-chat":                {"default": 8000,  "cap": 12000},
    "deepseek-reasoner":            {"default": 8000,  "cap": 12000},
    # TogetherAI models
    "llama-4-maverick":             {"default": 8000, "cap": 16384},
    "llama-4-scout":                {"default": 8000, "cap": 16384},
    "qwen3-coder":                  {"default": 16000, "cap": 32768},
    "qwen3-next-80b":               {"default": 16000, "cap": 32768},
}

# Accept either friendly keys above or exact provider IDs
_POLICY_ALIASES = {
    "claude-3-5-sonnet-20241022":   "claude-3-5",
    "claude-sonnet-4-20250514":     "claude-sonnet-4",
    "claude-opus-4-1-20250805":     "claude-opus-4.1",
    "claude-opus-4-5-20251124":     "claude-opus-4.5",      # NEW
    "claude-haiku-4-5-20251001":    "claude-haiku-4.5",     # NEW
    # DeepSeek aliases (optional)
    "deepseek-r1":                  "deepseek-reasoner",
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
    system_prompt: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
) -> str:
    """
    Unified interface to call any supported model for code generation.

    Supported models:
    - gpt-4o, o3, gpt-5, gpt-5-mini
    - gemini-1.5-flash, gemini-2.5-flash, gemini-2.5-pro, gemini-3-pro-preview
    - claude-3-5, claude-sonnet-4, claude-opus-4.1, claude-opus-4.5, claude-haiku-4.5
    - deepseek-chat, deepseek-reasoner
    - llama-4-maverick, llama-4-scout, qwen3-coder, qwen3-next-80b (via TogetherAI)
    
    NOTE: GPT-5 and gpt-5-mini do NOT support temperature parameter.
    They use reasoning_effort instead ('none', 'low', 'medium', 'high').
    Temperature will be ignored for these models.

    Parameters
    ----------
    model_name : str
        Target model identifier (friendly alias or provider ID).
    prompt : str
        The user prompt (task text). Do not include system content here.
    temperature : float
        Sampling temperature (keep 0.0 for determinism).
    max_tokens : Optional[int]
        Desired max tokens for completion (will be clamped to policy cap).
    seed : Optional[int]
        Random seed for providers that support it.
    return_raw : bool
        Whether to return the raw provider payload (for debugging).
    system_prompt : Optional[str]
        If provided, forwarded to the provider using its native system mechanism.
    reasoning_effort: Optional[str]
        For GPT-5 models, controls reasoning depth. Options: 'none', 'low', 'medium', 'high'.
    """
    mt = _resolve_tokens(model_name, max_tokens)

    if model_name in ("gpt-4o", "o3", "gpt-5", "gpt-5-mini"):
        return call_openai_for_code(
            prompt=prompt,
            model=model_name,
            temperature=temperature,
            max_tokens=mt,
            seed=seed,
            return_raw=return_raw,
            system_prompt=system_prompt,
            reasoning_effort=reasoning_effort,
        )
    elif model_name in ("gemini-1.5-flash", "gemini-2.5-flash", "gemini-2.5-pro", "gemini-3-pro-preview"):
        return call_gemini_for_code(
            prompt=prompt,
            model_name=model_name,
            temperature=temperature,
            max_tokens=mt,
            return_raw=return_raw,
            system_prompt=system_prompt,
        )
    elif model_name in ("claude-3-5", "claude-3-5-sonnet-20241022"):
        return call_claude_for_code(
            prompt=prompt,
            model="claude-3-5-sonnet-20241022",
            temperature=temperature,
            max_tokens=mt,
            return_raw=return_raw,
            system_prompt=system_prompt,
        )
    elif model_name in ("claude-sonnet-4", "claude-sonnet-4-20250514"):
        return call_claude_for_code(
            prompt=prompt,
            model="claude-sonnet-4-20250514",
            temperature=temperature,
            max_tokens=mt,
            return_raw=return_raw,
            system_prompt=system_prompt,
        )
    elif model_name in ("claude-opus-4.1", "claude-opus-4-1-20250805"):
        return call_claude_for_code(
            prompt=prompt,
            model="claude-opus-4-1-20250805",
            temperature=temperature,
            max_tokens=mt,
            return_raw=return_raw,
            system_prompt=system_prompt,
        )
    elif model_name in ("claude-opus-4.5", "claude-opus-4-5-20251124"):
        return call_claude_for_code(
            prompt=prompt,
            model="claude-opus-4-5-20251124",
            temperature=temperature,
            max_tokens=mt,
            return_raw=return_raw,
            system_prompt=system_prompt,
        )
    elif model_name in ("claude-haiku-4.5", "claude-haiku-4-5-20251001"):
        return call_claude_for_code(
            prompt=prompt,
            model="claude-haiku-4-5-20251001",
            temperature=temperature,
            max_tokens=mt,
            return_raw=return_raw,
            system_prompt=system_prompt,
        )
    elif model_name in ("deepseek-chat", "deepseek-reasoner", "deepseek-r1"):
        resolved = "deepseek-reasoner" if model_name == "deepseek-r1" else model_name
        return call_deepseek_for_code(
            prompt=prompt,
            model=resolved,
            temperature=temperature,
            max_tokens=mt,
            return_raw=return_raw,
            system_prompt=system_prompt,
        )
    elif model_name in ("llama-4-maverick", "llama-4-scout", "qwen3-coder", "qwen3-next-80b"):
        return call_togetherai_for_code(
            prompt=prompt,
            model=model_name,
            temperature=temperature,
            max_tokens=mt,
            seed=seed,
            return_raw=return_raw,
            system_prompt=system_prompt,
        )
    else:
        raise ValueError(
            "Unsupported model: {model}. Supported models: "
            "gpt-4o, o3, gpt-5, gpt-5-mini, gemini-1.5-flash, gemini-2.5-flash, gemini-2.5-pro, gemini-3-pro-preview, "
            "claude-3-5, claude-sonnet-4, claude-opus-4.1, claude-opus-4.5, claude-haiku-4.5, "
            "deepseek-chat, deepseek-reasoner, "
            "llama-4-maverick, llama-4-scout, qwen3-coder, qwen3-next-80b"
            .format(model=model_name)
        )

def call_llm_for_tests(
    model_name: str,
    prompt: str,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    seed: Optional[int] = None,
    return_raw: bool = False,
    system_prompt: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
) -> Dict[str, str]:
    """
    Unified interface to call any supported model for test generation.

    Supported models:
    - gpt-4o, o3, gpt-5, gpt-5-mini
    - gemini-1.5-flash, gemini-2.5-flash, gemini-2.5-pro, gemini-3-pro-preview
    - claude-3-5, claude-sonnet-4, claude-opus-4.1, claude-opus-4.5, claude-haiku-4.5
    - deepseek-chat, deepseek-reasoner
    - llama-4-maverick, llama-4-scout, qwen3-coder, qwen3-next-80b (via TogetherAI)
    
    NOTE: GPT-5 and gpt-5-mini do NOT support temperature parameter.
    They use reasoning_effort instead ('none', 'low', 'medium', 'high').

    Returns
    -------
    Dict[str, str]
        A dictionary mapping artifact names to content (e.g., {"tests.py": "..."}).
    """
    mt = _resolve_tokens(model_name, max_tokens)

    if model_name in ("gpt-4o", "o3", "gpt-5", "gpt-5-mini"):
        return call_openai_for_tests(
            prompt=prompt,
            model=model_name,
            temperature=temperature,
            max_tokens=mt,
            seed=seed,
            return_raw=return_raw,
            system_prompt=system_prompt,
            reasoning_effort=reasoning_effort,
        )
    elif model_name in ("gemini-1.5-flash", "gemini-2.5-flash", "gemini-2.5-pro", "gemini-3-pro-preview"):
        return call_gemini_for_tests(
            prompt=prompt,
            model_name=model_name,
            temperature=temperature,
            max_tokens=mt,
            return_raw=return_raw,
            system_prompt=system_prompt,
        )
    elif model_name in ("claude-3-5", "claude-3-5-sonnet-20241022"):
        return call_claude_for_tests(
            prompt=prompt,
            model="claude-3-5-sonnet-20241022",
            temperature=temperature,
            max_tokens=mt,
            return_raw=return_raw,
            system_prompt=system_prompt,
        )
    elif model_name in ("claude-sonnet-4", "claude-sonnet-4-20250514"):
        return call_claude_for_tests(
            prompt=prompt,
            model="claude-sonnet-4-20250514",
            temperature=temperature,
            max_tokens=mt,
            return_raw=return_raw,
            system_prompt=system_prompt,
        )
    elif model_name in ("claude-opus-4.1", "claude-opus-4-1-20250805"):
        return call_claude_for_tests(
            prompt=prompt,
            model="claude-opus-4-1-20250805",
            temperature=temperature,
            max_tokens=mt,
            return_raw=return_raw,
            system_prompt=system_prompt,
        )
    elif model_name in ("claude-opus-4.5", "claude-opus-4-5-20251124"):
        return call_claude_for_tests(
            prompt=prompt,
            model="claude-opus-4-5-20251124",
            temperature=temperature,
            max_tokens=mt,
            return_raw=return_raw,
            system_prompt=system_prompt,
        )
    elif model_name in ("claude-haiku-4.5", "claude-haiku-4-5-20251001"):
        return call_claude_for_tests(
            prompt=prompt,
            model="claude-haiku-4-5-20251001",
            temperature=temperature,
            max_tokens=mt,
            return_raw=return_raw,
            system_prompt=system_prompt,
        )
    elif model_name in ("deepseek-chat", "deepseek-reasoner", "deepseek-r1"):
        resolved = "deepseek-reasoner" if model_name == "deepseek-r1" else model_name
        return call_deepseek_for_tests(
            prompt=prompt,
            model=resolved,
            temperature=temperature,
            max_tokens=mt,
            return_raw=return_raw,
            system_prompt=system_prompt,
        )
    elif model_name in ("llama-4-maverick", "llama-4-scout", "qwen3-coder", "qwen3-next-80b"):
        return call_togetherai_for_tests(
            prompt=prompt,
            model=model_name,
            temperature=temperature,
            max_tokens=mt,
            seed=seed,
            return_raw=return_raw,
            system_prompt=system_prompt,
        )
    else:
        raise ValueError(
            "Unsupported model: {model}. Supported models: "
            "gpt-4o, o3, gpt-5, gpt-5-mini, gemini-1.5-flash, gemini-2.5-flash, gemini-2.5-pro, gemini-3-pro-preview, "
            "claude-3-5, claude-sonnet-4, claude-opus-4.1, claude-opus-4.5, claude-haiku-4.5, "
            "deepseek-chat, deepseek-reasoner, "
            "llama-4-maverick, llama-4-scout, qwen3-coder, qwen3-next-80b"
            .format(model=model_name)
        )

def list_available_models() -> Dict[str, str]:
    """Return a dictionary of available models and their descriptions."""
    return {
        "gpt-4o":                "OpenAI GPT-4o",
        "o3":                    "OpenAI O3 (reasoning-optimized, Aug 2025)",
        "gpt-5":                 "OpenAI GPT-5 (reasoning-capable, Aug 2025)",
        "gpt-5-mini":            "OpenAI GPT-5 Mini (compact reasoning, Aug 2025)",
        "gemini-1.5-flash":      "Google Gemini 1.5 Flash",
        "gemini-2.5-flash":      "Google Gemini 2.5 Flash",
        "gemini-2.5-pro":        "Google Gemini 2.5 Pro",
        "gemini-3-pro-preview":  "Google Gemini 3 Pro Preview",
        "claude-3-5":            "Anthropic Claude 3.5 Sonnet (2024-10-22)",
        "claude-sonnet-4":       "Anthropic Claude Sonnet 4 (2025-05-14)",
        "claude-opus-4.1":       "Anthropic Claude Opus 4.1 (2025-08-05)",
        "claude-opus-4.5":       "Anthropic Claude Opus 4.5 (2025-11-24)",  # NEW
        "claude-haiku-4.5":      "Anthropic Claude Haiku 4.5 (2025-10-01)", # NEW
        "deepseek-chat":         "DeepSeek-V3 (general coding)",
        "deepseek-reasoner":     "DeepSeek-R1 (reasoning)",
        "llama-4-maverick":      "Meta Llama 4 Maverick via TogetherAI",
        "llama-4-scout":         "Meta Llama 4 Scout via TogetherAI",
        "qwen3-coder":           "Qwen3 Coder 480B via TogetherAI",
        "qwen3-next-80b":        "Qwen3-Next 80B Instruct via TogetherAI",
    }
from typing import Dict, Optional
from llm_api.openai_client import call_openai_for_code, call_openai_for_tests
from llm_api.gemini_client import call_gemini_for_code, call_gemini_for_tests
from llm_api.deepseek_client import call_deepseek_for_code, call_deepseek_for_tests
from llm_api.claude_client import call_claude_for_code, call_claude_for_tests


def call_llm_for_code(
    model_name: str,
    prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    seed: Optional[int] = None,
    return_raw: bool = False,
) -> str:
    """
    Unified interface to call any supported model for code generation.

    Supported models:
    - gpt-4o, o3
    - gemini-flash, gemini-pro
    - claude-3-5
    - deepseek-chat
    """
    if model_name in ("gpt-4o", "o3"):
        return call_openai_for_code(
            prompt=prompt,
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            return_raw=return_raw,
        )
    elif model_name == "gemini-flash":
        return call_gemini_for_code(
            prompt=prompt,
            model_name="models/gemini-1.5-flash",
            temperature=temperature,
            max_tokens=max_tokens,
            return_raw=return_raw,
        )
    elif model_name == "gemini-pro":
        return call_gemini_for_code(
            prompt=prompt,
            model_name="models/gemini-1.5-pro",
            temperature=temperature,
            max_tokens=max_tokens,
            return_raw=return_raw,
        )
    elif model_name == "claude-3-5":
        return call_claude_for_code(
            prompt=prompt,
            model="claude-3-5-sonnet-20241022",
            temperature=temperature,
            max_tokens=max_tokens,
            return_raw=return_raw,
        )
    elif model_name == "deepseek-chat":
        return call_deepseek_for_code(
            prompt=prompt,
            model="deepseek-chat",
            temperature=temperature,
            max_tokens=max_tokens,
            return_raw=return_raw,
        )
    else:
        raise ValueError(
            f"Unsupported model: {model_name}. Supported models: gpt-4o, o3, gemini-flash, gemini-pro, claude-3-5, deepseek-chat"
        )


def call_llm_for_tests(
    model_name: str,
    prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    seed: Optional[int] = None,
    return_raw: bool = False,
) -> Dict[str, str]:
    """
    Unified interface to call any supported model for test generation.

    Supported models:
    - gpt-4o, o3
    - gemini-flash, gemini-pro
    - claude-3-5
    - deepseek-chat
    """
    if model_name in ("gpt-4o", "o3"):
        return call_openai_for_tests(
            prompt=prompt,
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            return_raw=return_raw,
        )
    elif model_name == "gemini-flash":
        return call_gemini_for_tests(
            prompt=prompt,
            model_name="models/gemini-1.5-flash",
            temperature=temperature,
            max_tokens=max_tokens,
            return_raw=return_raw,
        )
    elif model_name == "gemini-pro":
        return call_gemini_for_tests(
            prompt=prompt,
            model_name="models/gemini-1.5-pro",
            temperature=temperature,
            max_tokens=max_tokens,
            return_raw=return_raw,
        )
    elif model_name == "claude-3-5":
        return call_claude_for_tests(
            prompt=prompt,
            model="claude-3-5-sonnet-20241022",
            temperature=temperature,
            max_tokens=max_tokens,
            return_raw=return_raw,
        )
    elif model_name == "deepseek-chat":
        return call_deepseek_for_tests(
            prompt=prompt,
            model="deepseek-chat",
            temperature=temperature,
            max_tokens=max_tokens,
            return_raw=return_raw,
        )
    else:
        raise ValueError(
            f"Unsupported model: {model_name}. Supported models: gpt-4o, o3, gemini-flash, gemini-pro, claude-3-5, deepseek-chat"
        )


def list_available_models() -> Dict[str, str]:
    """Return a dictionary of available models and their descriptions."""
    return {
        "gpt-4o": "OpenAI GPT-4 Omni (May 2024 release)",
        "o3": "OpenAI O3 (reasoning-optimized model, August 2025)",
        "gemini-flash": "Google Gemini 1.5 Flash",
        "gemini-pro": "Google Gemini 1.5 Pro",
        "claude-3-5": "Anthropic Claude 3.5 Sonnet",
        "deepseek-chat": "DeepSeek-V3 - Latest general coding model"
    }

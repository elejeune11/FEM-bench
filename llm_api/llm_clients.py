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
    - gpt-4o
    - gemini-flash, gemini-pro
    - claude-3-5
    - deepseek-chat
    """
    if model_name == "gpt-4o":
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
        raise ValueError(f"Unsupported model: {model_name}. Supported models: gpt-4o, gemini-flash, gemini-pro, claude-3-5, deepseek-chat")


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
    - gpt-4o
    - gemini-flash, gemini-pro
    - claude-3-5
    - deepseek-chat
    """
    if model_name == "gpt-4o":
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
        raise ValueError(f"Unsupported model: {model_name}. Supported models: gpt-4o, gemini-flash, gemini-pro, claude-3-5, deepseek-chat")


# Convenience function to list available models
def list_available_models() -> Dict[str, str]:
    """Return a dictionary of available models and their descriptions."""
    return {
        "gpt-4o": "OpenAI GPT-4 Omni",
        "gemini-flash": "Google Gemini 1.5 Flash",
        "gemini-pro": "Google Gemini 1.5 Pro",
        "claude-3-5": "Anthropic Claude 3.5 Sonnet",
        "deepseek-chat": "DeepSeek-V3 - Latest general coding model"
    }


# Example usage
if __name__ == "__main__":
    print("Available models:")
    for model, description in list_available_models().items():
        print(f"  {model}: {description}")
    
    # Test with different models
    prompt = "Write a Python function to calculate factorial"
    
    print("\nTesting Claude 3.5:")
    try:
        result = call_llm_for_code("claude-3-5", prompt, max_tokens=200, return_raw=True)
        print(f"Result: {result[:100]}...")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\nTesting DeepSeek Chat (DeepSeek-V3):")
    try:
        result = call_llm_for_code("deepseek-chat", prompt, max_tokens=200, return_raw=True)
        print(f"Result: {result[:100]}...")
    except Exception as e:
        print(f"Error: {e}")
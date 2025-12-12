# --- togetherai_client.py (TogetherAI API) ---
import os
import time
from typing import Dict, Optional
from together import Together
from dotenv import load_dotenv
from llm_api.clean_utils import clean_and_extract_function, extract_test_functions

# Load TogetherAI API key from .env file
load_dotenv()

# ---- Models & alias resolution ---------------------------------------------
# Define the model ID for Llama 4 Maverick
LLAMA_4_MAVERICK_17B = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
LLAMA_4_SCOUT_17B = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
QWEN3_CODER_480B = "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8"
QWEN3_NEXT_80B = "Qwen/Qwen3-Next-80B-A3B-Instruct"

# Friendly aliases (case-insensitive)
_MODEL_ALIASES = {
    "llama-4-maverick": LLAMA_4_MAVERICK_17B,
    "llama-4-scout": LLAMA_4_SCOUT_17B,
    "qwen3-coder": QWEN3_CODER_480B,
    "qwen3-next-80b": QWEN3_NEXT_80B,
}

def _resolve_model(name: str | None) -> str:
    """
    Map friendly names/aliases to official TogetherAI model IDs.
    Defaults to Llama 4 Maverick if not provided.
    """
    if not name:
        return LLAMA_4_MAVERICK_17B
    key = name.strip().lower()
    return _MODEL_ALIASES.get(key, name)

# Keep a sensible default
MODEL_NAME = LLAMA_4_MAVERICK_17B


def retry_api_call(call_fn, retries: int = 3, backoff: float = 2.0):
    """Retry API calls with exponential backoff."""
    for attempt in range(retries):
        try:
            return call_fn()
        except Exception as e:
            print(f"API call failed (attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                wait_time = backoff ** attempt
                print(f"Retrying in {wait_time:.1f} seconds...")
                time.sleep(wait_time)
            else:
                raise


def _get_togetherai_client():
    """Returns a TogetherAI client, checking for the API key."""
    api_key = os.getenv("TOGETHERAI_API_KEY")
    if not api_key:
        raise RuntimeError("TOGETHERAI_API_KEY is not set in your environment or .env file.")
    return Together(api_key=api_key)


def _call_togetherai_api(
    prompt: str,
    temperature: float,
    max_tokens: int,
    model: str,
    seed: Optional[int] = None,
    system_prompt: Optional[str] = None,
) -> str:
    """Make the actual API call to TogetherAI API (OpenAI-compatible)."""
    client = _get_togetherai_client()

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=_resolve_model(model),
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        seed=seed,
        stream=False,
    )

    choice = response.choices[0]
    content = choice.message.content or ""

    if not content.strip():
        raise ValueError(
            f"TogetherAI returned empty response (finish_reason={choice.finish_reason})"
        )
    return content


def call_togetherai_for_code(
    prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    model: str = MODEL_NAME,
    seed: Optional[int] = None,
    return_raw: bool = False,
    system_prompt: Optional[str] = None,
) -> str:
    """Calls TogetherAI API and returns a single cleaned function."""
    def call():
        return _call_togetherai_api(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
            seed=seed,
            system_prompt=system_prompt,
        )

    try:
        raw = retry_api_call(call)
        return raw if return_raw else clean_and_extract_function(raw)
    except Exception as e:
        print(f"Error in call_togetherai_for_code: {e}")
        raise


def call_togetherai_for_tests(
    prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    model: str = MODEL_NAME,
    seed: Optional[int] = None,
    return_raw: bool = False,
    system_prompt: Optional[str] = None,
) -> Dict[str, str]:
    """Calls TogetherAI API and returns all test functions as a dict {name: code}."""
    def call():
        return _call_togetherai_api(
            prompt=prompt, temperature=temperature, max_tokens=max_tokens, model=model, seed=seed, system_prompt=system_prompt,
        )

    try:
        raw = retry_api_call(call)
        return raw if return_raw else extract_test_functions(raw)
    except Exception as e:
        print(f"Error in call_togetherai_for_tests: {e}")
        raise

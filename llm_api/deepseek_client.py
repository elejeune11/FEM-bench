# --- deepseek_client.py (DeepSeek Official API) ---
import os
import time
from typing import Dict, Optional
import requests
from dotenv import load_dotenv
from llm_api.clean_utils import clean_and_extract_function, extract_test_functions

# Load DeepSeek API key
load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    raise RuntimeError("DEEPSEEK_API_KEY is not set in your environment or .env file.")

DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

# ---- Models & alias resolution ---------------------------------------------
DEEPSEEK_CHAT = "deepseek-chat"            # V3 / V3.1 (general coding)
DEEPSEEK_REASONER = "deepseek-reasoner"    # R1 (reasoning)

# Friendly aliases (case-insensitive)
_MODEL_ALIASES = {
    "deepseek-v3": DEEPSEEK_CHAT,
    "v3": DEEPSEEK_CHAT,
    "deepseek-r1": DEEPSEEK_REASONER,
    "r1": DEEPSEEK_REASONER,
}

def _resolve_model(name: str | None) -> str:
    """
    Map friendly names/aliases to official DeepSeek model IDs.
    Defaults to deepseek-chat if not provided.
    """
    if not name:
        return DEEPSEEK_CHAT
    key = name.strip().lower()
    return _MODEL_ALIASES.get(key, name)

# Keep V3 as the library default; callers can pass "deepseek-reasoner".
MODEL_NAME = DEEPSEEK_CHAT


def retry_api_call(call_fn, retries: int = 3, backoff: float = 2.0):
    """Retry API calls with exponential backoff."""
    for attempt in range(retries):
        try:
            return call_fn()
        except requests.exceptions.RequestException as e:
            print(f"API call failed (attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                wait_time = backoff ** attempt
                print(f"Retrying in {wait_time:.1f} seconds...")
                time.sleep(wait_time)
            else:
                raise
        except Exception as e:
            print(f"Unexpected error (attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                wait_time = backoff ** attempt
                print(f"Retrying in {wait_time:.1f} seconds...")
                time.sleep(wait_time)
            else:
                raise


def _call_deepseek_api(
    prompt: str,
    temperature: float,
    max_tokens: int,
    model: str,
    system_prompt: Optional[str] = None,
) -> str:
    """Make the actual API call to DeepSeek Official API (OpenAI-compatible)."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    messages = []
    # NEW: prepend a system message if provided
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": _resolve_model(model),  # "deepseek-chat" or "deepseek-reasoner"
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }

    try:
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()

        # Expect OpenAI chat-completions shape.
        # Some R1 responses include "reasoning_content" - we intentionally ignore it.
        if "choices" in data and data["choices"]:
            msg = data["choices"][0].get("message", {})
            content = msg.get("content", "")
            if not content or not content.strip():
                print("Warning: DeepSeek returned empty content.")
                raise ValueError("Empty response from DeepSeek API")
            return content

        raise ValueError(f"Unexpected response format: {data}")

    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e}")
        if hasattr(response, "text"):
            print(f"Response: {response.text}")
        raise
    except requests.exceptions.RequestException as e:
        print(f"Request Error: {e}")
        raise


def call_deepseek_for_code(
    prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    model: str = MODEL_NAME,
    return_raw: bool = False,
    system_prompt: Optional[str] = None,
) -> str:
    """
    Calls DeepSeek API and returns a single cleaned function.
    `model` may be "deepseek-chat" (default) or "deepseek-reasoner".
    """
    def call():
        return _call_deepseek_api(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
            system_prompt=system_prompt,
        )

    try:
        raw = retry_api_call(call)
        return raw if return_raw else clean_and_extract_function(raw)
    except Exception as e:
        print(f"Error in call_deepseek_for_code: {e}")
        raise


def call_deepseek_for_tests(
    prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    model: str = MODEL_NAME,
    return_raw: bool = False,
    system_prompt: Optional[str] = None,
) -> Dict[str, str]:
    """
    Calls DeepSeek API and returns all test functions as a dict {name: code}.
    `model` may be "deepseek-chat" (default) or "deepseek-reasoner".
    """
    def call():
        return _call_deepseek_api(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
            system_prompt=system_prompt,
        )

    try:
        raw = retry_api_call(call)
        return raw if return_raw else extract_test_functions(raw)
    except Exception as e:
        print(f"Error in call_deepseek_for_tests: {e}")
        raise


# Test function to verify the setup
def test_api_connection():
    """Test the API connection with a simple prompt."""
    try:
        result = call_deepseek_for_code(
            "Write a simple Python function that adds two numbers:",
            return_raw=True,
            max_tokens=100
        )
        print("DeepSeek API test successful!")
        print("Response:", result[:200] + "..." if len(result) > 200 else result)
        return True
    except Exception as e:
        print(f"API test failed: {e}")
        return False


if __name__ == "__main__":
    # Test the API connection
    test_api_connection()

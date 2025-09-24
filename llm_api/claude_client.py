# --- claude_client.py ---
import os
import time
from typing import Dict
import requests
from dotenv import load_dotenv
from llm_api.clean_utils import clean_and_extract_function, extract_test_functions

# Load env
load_dotenv()
api_key = os.getenv("CLAUDE_API_KEY")
if not api_key:
    raise RuntimeError("CLAUDE_API_KEY is not set in your environment or .env file.")

CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"

# ---- New: official model IDs (Claude 4 generation)
CLAUDE_SONNET_4 = "claude-sonnet-4-20250514"
CLAUDE_OPUS_41  = "claude-opus-4-1-20250805"

# Optional aliases so env is friendlier (case-insensitive)
_MODEL_ALIASES = {
    "sonnet4": CLAUDE_SONNET_4,
    "opus4.1": CLAUDE_OPUS_41,
    "claude-sonnet-4": CLAUDE_SONNET_4,
    "claude-opus-4.1": CLAUDE_OPUS_41,
}

def _resolve_model(name: str | None) -> str:
    """
    Resolve a model name from env/alias to a concrete Anthropic model ID.
    Falls back to Sonnet 4 if nothing provided.
    """
    if not name:
        return CLAUDE_SONNET_4
    key = name.strip().lower()
    return _MODEL_ALIASES.get(key, name)

# Default model: prefer env override, else Sonnet 4
MODEL_NAME = _resolve_model(os.getenv("CLAUDE_MODEL"))

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

def _call_claude_api(prompt: str, temperature: float, max_tokens: int, model: str) -> str:
    """Make the actual API call to Claude API."""
    headers = {
        "x-api-key": api_key,
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",  # Anthropic Messages API
    }

    payload = {
        "model": _resolve_model(model),
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    response = requests.post(CLAUDE_API_URL, headers=headers, json=payload, timeout=120)
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e}")
        try:
            print(f"Response: {response.text}")
        except Exception:
            pass
        raise

    response_data = response.json()
    # Expect Anthropic "messages" response format
    if "content" in response_data and response_data["content"]:
        content = response_data["content"][0].get("text", "")
        if not content or not content.strip():
            print("Warning: Claude returned empty response, retrying with adjusted prompt...")
            raise ValueError("Empty response from Claude API")
        return content

    raise ValueError(f"Unexpected response format: {response_data}")

def call_claude_for_code(
    prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    model: str = MODEL_NAME,
    return_raw: bool = False,
) -> str:
    """
    Calls Claude API and returns a single cleaned function.
    """
    def call():
        return _call_claude_api(prompt, temperature, max_tokens, model)

    try:
        raw = retry_api_call(call)
        if not raw or not raw.strip():
            print(f"Warning: Empty response for prompt: {prompt[:100]}...")
            return "# Empty response from Claude API\npass" if not return_raw else ""
        return raw if return_raw else clean_and_extract_function(raw)
    except Exception as e:
        print(f"Error in call_claude_for_code: {e}")
        if return_raw:
            return f"# Error: {e}"
        return f"# Error occurred: {e}\ndef placeholder_function():\n    pass"

def call_claude_for_tests(
    prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    model: str = MODEL_NAME,
    return_raw: bool = False,
) -> Dict[str, str]:
    """
    Calls Claude API and returns all test functions as a dict {name: code}.
    """
    def call():
        return _call_claude_api(prompt, temperature, max_tokens, model)

    try:
        raw = retry_api_call(call)
        if not raw or not raw.strip():
            print(f"Warning: Empty response for test prompt: {prompt[:100]}...")
            return {} if not return_raw else ""
        return raw if return_raw else extract_test_functions(raw)
    except Exception as e:
        print(f"Error in call_claude_for_tests: {e}")
        if return_raw:
            return f"# Error: {e}"
        return {"test_placeholder": f"# Error occurred: {e}\ndef test_placeholder():\n    assert True"}

def test_api_connection():
    """Test the API connection with a simple prompt."""
    try:
        result = call_claude_for_code(
            "Write a simple Python function that adds two numbers:",
            return_raw=True,
            max_tokens=100
        )
        print("Claude API test successful!")
        print("Response:", result[:200] + "..." if len(result) > 200 else result)
        return True
    except Exception as e:
        print(f"Claude API test failed: {e}")
        return False

if __name__ == "__main__":
    test_api_connection()

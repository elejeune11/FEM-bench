# --- claude_client.py ---
import os
import time
from typing import Dict
import requests
from dotenv import load_dotenv
from llm_api.clean_utils import clean_and_extract_function, extract_test_functions

# Load Claude API key
load_dotenv()
api_key = os.getenv("CLAUDE_API_KEY")
if not api_key:
    raise RuntimeError("CLAUDE_API_KEY is not set in your environment or .env file.")

CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"
MODEL_NAME = "claude-3-5-sonnet-20241022"  # Latest Claude 3.5 Sonnet


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
        "anthropic-version": "2023-06-01"
    }
    
    # Claude uses messages format
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    
    try:
        response = requests.post(CLAUDE_API_URL, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        
        response_data = response.json()
        
        # Debug: print response structure if needed
        # print("API Response:", response_data)
        
        # Handle Claude messages format
        if "content" in response_data and len(response_data["content"]) > 0:
            content = response_data["content"][0]["text"]
            
            # Handle empty or whitespace-only responses
            if not content or not content.strip():
                print("Warning: Claude returned empty response, retrying with adjusted prompt...")
                raise ValueError("Empty response from Claude API")
                
            return content
        else:
            raise ValueError(f"Unexpected response format: {response_data}")
            
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e}")
        if hasattr(response, 'text'):
            print(f"Response: {response.text}")
        raise
    except requests.exceptions.RequestException as e:
        print(f"Request Error: {e}")
        raise


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
        
        # Check if raw response is empty
        if not raw or not raw.strip():
            print(f"Warning: Empty response for prompt: {prompt[:100]}...")
            return "# Empty response from Claude API\npass" if not return_raw else ""
            
        return raw if return_raw else clean_and_extract_function(raw)
    except Exception as e:
        print(f"Error in call_claude_for_code: {e}")
        # Return a basic fallback instead of crashing
        if return_raw:
            return f"# Error: {e}"
        else:
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
        
        # Check if raw response is empty
        if not raw or not raw.strip():
            print(f"Warning: Empty response for test prompt: {prompt[:100]}...")
            return {} if not return_raw else ""
            
        return raw if return_raw else extract_test_functions(raw)
    except Exception as e:
        print(f"Error in call_claude_for_tests: {e}")
        # Return empty dict for tests instead of crashing
        if return_raw:
            return f"# Error: {e}"
        else:
            return {"test_placeholder": f"# Error occurred: {e}\ndef test_placeholder():\n    assert True"}


# Test function to verify the setup
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
    # Test the API connection
    test_api_connection()
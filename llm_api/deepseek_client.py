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
MODEL_NAME = "deepseek-chat"  # DeepSeek-V3 for general coding


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


def _call_deepseek_api(prompt: str, temperature: float, max_tokens: int, model: str) -> str:
    """Make the actual API call to DeepSeek Official API."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    
    # DeepSeek uses OpenAI-compatible chat completions format
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False
    }
    
    try:
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        
        response_data = response.json()
        
        # Debug: print response structure if needed
        # print("API Response:", response_data)
        
        # Handle chat completions format
        if "choices" in response_data and len(response_data["choices"]) > 0:
            return response_data["choices"][0]["message"]["content"]
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


def call_deepseek_for_code(
    prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    model: str = MODEL_NAME,
    return_raw: bool = False,
) -> str:
    """
    Calls DeepSeek Official API and returns a single cleaned function.
    """
    def call():
        return _call_deepseek_api(prompt, temperature, max_tokens, model)

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
) -> Dict[str, str]:
    """
    Calls DeepSeek Official API and returns all test functions as a dict {name: code}.
    """
    def call():
        return _call_deepseek_api(prompt, temperature, max_tokens, model)

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
        print("DeepSeek Official API test successful!")
        print("Response:", result[:200] + "..." if len(result) > 200 else result)
        return True
    except Exception as e:
        print(f"API test failed: {e}")
        return False


if __name__ == "__main__":
    # Test the API connection
    test_api_connection()
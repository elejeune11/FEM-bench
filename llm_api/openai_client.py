# --- openai_client.py (updated) ---
import os
import time
from typing import Dict, Optional
from openai import OpenAI
from dotenv import load_dotenv
from llm_api.clean_utils import clean_and_extract_function, extract_test_functions


# Load environment variables from .env
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise RuntimeError("OPENAI_API_KEY is not set in your environment or .env file.")

client = OpenAI(api_key=api_key)


def retry_api_call(call_fn, retries: int = 3, backoff: float = 1.5):
    for attempt in range(retries):
        try:
            return call_fn()
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(backoff ** attempt)
            else:
                raise


def call_openai_for_code(
    prompt: str,
    model: str = "gpt-4o",
    temperature: float = 0.0,
    max_tokens: int = 2048,
    seed: Optional[int] = None,
    return_raw: bool = False,
) -> str:
    """
    Calls OpenAI and returns a single cleaned function.
    """
    def call():
        return client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed
        )

    response = retry_api_call(call)
    raw = response.choices[0].message.content
    return raw if return_raw else clean_and_extract_function(raw)


def call_openai_for_tests(
    prompt: str,
    model: str = "gpt-4o",
    temperature: float = 0.,
    max_tokens: int = 2048,
    seed: Optional[int] = None,
    return_raw: bool = False,
) -> Dict[str, str]:
    """
    Calls OpenAI and returns all test functions as a dict {name: code}.
    """
    def call():
        return client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed
        )

    response = retry_api_call(call)
    raw = response.choices[0].message.content
    return raw if return_raw else extract_test_functions(raw)

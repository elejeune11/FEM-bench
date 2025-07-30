# --- gemini_client.py ---
import os
import time
from typing import Dict
from dotenv import load_dotenv
import google.generativeai as genai
from llm_api.clean_utils import clean_and_extract_function, extract_test_functions

# Load API key
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY is not set in your environment or .env file.")

genai.configure(api_key=api_key)


def retry_api_call(call_fn, retries: int = 3, backoff: float = 1.5):
    for attempt in range(retries):
        try:
            return call_fn()
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(backoff ** attempt)
            else:
                raise


def call_gemini_for_code(
    prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    model_name: str = "models/gemini-1.5-flash",
    return_raw: bool = False,
) -> str:
    """
    Calls Gemini and returns a single cleaned function.
    """
    model = genai.GenerativeModel(model_name)

    def call():
        return model.generate_content(prompt, generation_config={
            "temperature": temperature,
            "max_output_tokens": max_tokens
        })

    response = retry_api_call(call)
    raw = response.text
    return raw if return_raw else clean_and_extract_function(raw)


def call_gemini_for_tests(
    prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    model_name: str = "models/gemini-1.5-flash",
    return_raw: bool = False,
) -> Dict[str, str]:
    """
    Calls Gemini and returns all test functions as a dict {name: code}.
    """
    model = genai.GenerativeModel(model_name)

    def call():
        return model.generate_content(prompt, generation_config={
            "temperature": temperature,
            "max_output_tokens": max_tokens
        })

    response = retry_api_call(call)
    raw = response.text
    return raw if return_raw else extract_test_functions(raw)

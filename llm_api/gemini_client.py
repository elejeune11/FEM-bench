# --- gemini_client.py ---
import os
import time
from typing import Dict, Optional
from dotenv import load_dotenv
import google.generativeai as genai
from llm_api.clean_utils import (
    clean_and_extract_function,
    extract_test_functions,
)

# Load API key
load_dotenv()


def _configure_gemini():
    """Loads the Gemini API key and configures the genai library."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set in your environment or .env file."
        )
    genai.configure(api_key=api_key)


def retry_api_call(call_fn, retries: int = 3, backoff: float = 1.5):
    for attempt in range(retries):
        try:
            return call_fn()
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(backoff**attempt)
            else:
                raise


def _make_model(model_name: str, system_prompt: Optional[str]):
    """
    Construct a Gemini GenerativeModel. If system_prompt is provided, use it as
    the model's system_instruction; otherwise construct the model normally.
    """
    if system_prompt:
        return genai.GenerativeModel(
            model_name, system_instruction=system_prompt
        )
    return genai.GenerativeModel(model_name)


def call_gemini_for_code(
    prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    model_name: str = "gemini-2.5-flash",
    return_raw: bool = False,
    system_prompt: Optional[str] = None,
) -> str:
    """
    Calls Gemini and returns a single cleaned function.
    """
    _configure_gemini()
    model = _make_model(model_name, system_prompt)

    def call():
        return model.generate_content(
            prompt,
            generation_config={
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            },
        )

    response = retry_api_call(call)
    raw = response.text
    return raw if return_raw else clean_and_extract_function(raw)


def call_gemini_for_tests(
    prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    model_name: str = "gemini-2.5-flash",
    return_raw: bool = False,
    system_prompt: Optional[str] = None,
) -> Dict[str, str]:
    """
    Calls Gemini and returns all test functions as a dict {name: code}.
    """
    _configure_gemini()
    model = _make_model(model_name, system_prompt)

    def call():
        return model.generate_content(
            prompt,
            generation_config={
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            },
        )

    response = retry_api_call(call)
    raw = response.text
    return raw if return_raw else extract_test_functions(raw)

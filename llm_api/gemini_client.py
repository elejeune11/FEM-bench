# --- gemini_client.py (UPDATED) ---
import os
import time
from typing import Dict, Optional
from dotenv import load_dotenv

# Imports for the modern google-genai SDK
# 'google' now contains 'genai', and 'types' is used for configuration objects.
from google import genai
from google.genai import types 

from llm_api.clean_utils import (
    clean_and_extract_function,
    extract_test_functions,
)

# Load API key
load_dotenv()

# --- Global Client Object ---
# The client is now a central object, replacing the global 'configure' state.
# It is initialized to None and created/returned by the new _configure_gemini.
_GEMINI_CLIENT = None 

def _get_gemini_client() -> genai.Client:
    """Gets or creates the global Gemini Client object."""
    global _GEMINI_CLIENT
    if _GEMINI_CLIENT is None:
        api_key = os.getenv("GEMINI_API_KEY")
        # Also support the new, preferred environment variable
        if not api_key:
            api_key = os.getenv("GOOGLE_API_KEY")
        
        # The API key is passed directly to the Client constructor
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY or GOOGLE_API_KEY is not set in your environment or .env file."
            )
        
        # The new way: create a client object
        _GEMINI_CLIENT = genai.Client(api_key=api_key)
        
    return _GEMINI_CLIENT


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
    Construct a Gemini GenerativeModel config for a client.
    Note: The GenerativeModel class is no longer used for generation in this manner.
    We just return the client and system prompt.
    """
    return _get_gemini_client(), system_prompt


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
    # 1. Get the client object
    client, sp = _make_model(model_name, system_prompt) 

    # 2. Configure thinking_level for Gemini 3 Pro Preview
    thinking_config = None
    if model_name == "gemini-3-pro-preview":
        thinking_config = types.ThinkingConfig(
            thinking_level=types.ThinkingLevel.HIGH
        )
    
    # 3. Build the configuration object
    config = types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_tokens,
        system_instruction=sp, # System prompt is now part of the config
        thinking_config=thinking_config,
    )

    def call():
        # The new way: call generate_content method on the client
        return client.models.generate_content(
            model=model_name, # The model name is passed here
            contents=prompt,
            config=config,
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
    # 1. Get the client object
    client, sp = _make_model(model_name, system_prompt)

    # 2. Configure thinking_level for Gemini 3 Pro Preview
    thinking_config = None
    if model_name == "gemini-3-pro-preview":
        thinking_config = types.ThinkingConfig(
            thinking_level=types.ThinkingLevel.HIGH
        )
        
    # 3. Build the configuration object
    config = types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_tokens,
        system_instruction=sp, # System prompt is now part of the config
        thinking_config=thinking_config,
    )

    def call():
        # The new way: call generate_content method on the client
        return client.models.generate_content(
            model=model_name, # The model name is passed here
            contents=prompt,
            config=config,
        )

    response = retry_api_call(call)
    raw = response.text
    return raw if return_raw else extract_test_functions(raw)
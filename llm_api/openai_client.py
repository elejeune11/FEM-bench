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


def _prepare_chat_params(
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    seed: Optional[int],
) -> dict:
    """
    Internal helper to prepare chat completion parameters depending on model.
    """
    params = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }

    if seed is not None:
        params["seed"] = seed

    if model in ("o3", "o3-pro"):
        # o3 requires max_completion_tokens, and does not accept temperature
        params["max_completion_tokens"] = max_tokens
    else:
        params["max_tokens"] = max_tokens
        params["temperature"] = temperature

    return params


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
    Raises an error if the output is empty.
    """
    def call():
        params = _prepare_chat_params(model, prompt, temperature, max_tokens, seed)
        response = client.chat.completions.create(**params)

        # print("=== RAW RESPONSE ===")
        # print(response)

        choice = response.choices[0]
        raw = choice.message.content or ""

        if not raw.strip():
            raise ValueError(
                f"OpenAI returned empty response "
                f"(finish_reason={choice.finish_reason}, completion_tokens={response.usage.completion_tokens})"
            )

        return raw

    raw = retry_api_call(call)
    return raw if return_raw else clean_and_extract_function(raw)


def call_openai_for_tests(
    prompt: str,
    model: str = "gpt-4o",
    temperature: float = 0.0,
    max_tokens: int = 2048,
    seed: Optional[int] = None,
    return_raw: bool = False,
) -> Dict[str, str]:
    """
    Calls OpenAI and returns all test functions as a dict {name: code}.
    Raises an error if the output is empty.
    """
    def call():
        params = _prepare_chat_params(model, prompt, temperature, max_tokens, seed)
        response = client.chat.completions.create(**params)

        # print("=== RAW RESPONSE ===")
        # print(response)

        choice = response.choices[0]
        raw = choice.message.content or ""

        if not raw.strip():
            raise ValueError(
                f"OpenAI returned empty response "
                f"(finish_reason={choice.finish_reason}, completion_tokens={response.usage.completion_tokens})"
            )

        return raw

    raw = retry_api_call(call)
    return raw if return_raw else extract_test_functions(raw)

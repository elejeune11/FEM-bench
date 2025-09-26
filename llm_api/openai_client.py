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

# Treat GPT-5 (and o-series) as reasoning models for Chat Completions params
REASONING_MODELS = {"o3", "o3-pro", "gpt-5", "gpt-5-mini"}  # extend as needed


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
    system_prompt: Optional[str] = None,
) -> dict:
    """
    Internal helper to prepare chat completion parameters depending on model.

    GPT-5 / o-series (reasoning models) -> use max_completion_tokens, omit temperature.
    Non-reasoning models -> use max_tokens + temperature.

    If system_prompt is provided, it is injected as the first message with role='system'.
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    params = {
        "model": model,
        "messages": messages,
    }
    if seed is not None:
        params["seed"] = seed

    if model in REASONING_MODELS:
        # Reasoning models on Chat Completions expect max_completion_tokens
        params["max_completion_tokens"] = max_tokens
        # Temperature is ignored/unsupported for some reasoning models on this endpoint.
        # (Use Responses API for full GPT-5 features.)
    else:
        params["max_tokens"] = max_tokens
        params["temperature"] = temperature

    return params


def call_openai_for_code(
    prompt: str,
    model: str = "gpt-4o",            # you can pass "gpt-5" here
    temperature: float = 0.0,
    max_tokens: int = 2048,
    seed: Optional[int] = None,
    return_raw: bool = False,
    system_prompt: Optional[str] = None,
) -> str:
    """
    Calls OpenAI and returns a single cleaned function.
    Raises an error if the output is empty.

    system_prompt: optional string passed as a system message before the user message.
    """
    def call():
        params = _prepare_chat_params(
            model=model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            system_prompt=system_prompt,
        )
        response = client.chat.completions.create(**params)

        choice = response.choices[0]
        raw = choice.message.content or ""

        if not raw.strip():
            raise ValueError(
                f"OpenAI returned empty response "
                f"(finish_reason={choice.finish_reason}, "
                f"completion_tokens={getattr(response.usage, 'completion_tokens', None)})"
            )
        return raw

    raw = retry_api_call(call)
    return raw if return_raw else clean_and_extract_function(raw)


def call_openai_for_tests(
    prompt: str,
    model: str = "gpt-4o",            # you can pass "gpt-5" here
    temperature: float = 0.0,
    max_tokens: int = 2048,
    seed: Optional[int] = None,
    return_raw: bool = False,
    system_prompt: Optional[str] = None,
) -> Dict[str, str]:
    """
    Calls OpenAI and returns all test functions as a dict {name: code}.
    Raises an error if the output is empty.

    system_prompt: optional string passed as a system message before the user message.
    """
    def call():
        params = _prepare_chat_params(
            model=model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            system_prompt=system_prompt,
        )
        response = client.chat.completions.create(**params)

        choice = response.choices[0]
        raw = choice.message.content or ""

        if not raw.strip():
            raise ValueError(
                f"OpenAI returned empty response "
                f"(finish_reason={choice.finish_reason}, "
                f"completion_tokens={getattr(response.usage, 'completion_tokens', None)})"
            )
        return raw

    raw = retry_api_call(call)
    return raw if return_raw else extract_test_functions(raw)

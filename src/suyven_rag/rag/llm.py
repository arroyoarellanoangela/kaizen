"""LLM abstraction — supports Ollama (local) and OpenAI-compatible APIs (cloud).

Provider detection:
  - LLM_PROVIDER=ollama  → POST {OLLAMA_URL}/api/chat  (default)
  - LLM_PROVIDER=openai  → POST {LLM_API_URL}/chat/completions
    Works with: DeepSeek, Groq, Together AI, OpenRouter, OpenAI, Gemini, etc.
"""

import json
import logging
from collections.abc import Generator

import requests as req

from .config import (
    LLM_API_KEY,
    LLM_API_URL,
    LLM_MODEL,
    LLM_PROVIDER,
    OLLAMA_URL,
    SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal streaming helpers
# ---------------------------------------------------------------------------


def _stream_ollama(
    messages: list[dict], model: str, timeout: int, **_kw
) -> Generator[str, None, None]:
    """Stream tokens from Ollama's /api/chat endpoint."""
    resp = req.post(
        f"{OLLAMA_URL}/api/chat",
        json={"model": model, "messages": messages, "stream": True},
        stream=True,
        timeout=timeout,
    )
    resp.raise_for_status()
    for line in resp.iter_lines():
        if not line:
            continue
        data = json.loads(line)
        if "message" in data and "content" in data["message"]:
            yield data["message"]["content"]


def _stream_openai(
    messages: list[dict],
    model: str,
    timeout: int,
    *,
    api_url: str = "",
    api_key: str = "",
) -> Generator[str, None, None]:
    """Stream tokens from any OpenAI-compatible API (DeepSeek, Groq, Gemini, etc.)."""
    url = api_url or LLM_API_URL
    key = api_key or LLM_API_KEY

    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"

    resp = req.post(
        f"{url}/chat/completions",
        headers=headers,
        json={"model": model, "messages": messages, "stream": True},
        stream=True,
        timeout=timeout,
    )
    resp.raise_for_status()

    for line in resp.iter_lines():
        if not line:
            continue
        text = line.decode("utf-8") if isinstance(line, bytes) else line
        if not text.startswith("data: "):
            continue
        payload = text[6:]  # strip "data: "
        if payload.strip() == "[DONE]":
            break
        try:
            data = json.loads(payload)
            delta = data.get("choices", [{}])[0].get("delta", {})
            token = delta.get("content")
            if token:
                yield token
        except json.JSONDecodeError:
            continue


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_PROVIDERS = {
    "ollama": _stream_ollama,
    "openai": _stream_openai,
}


def quick_complete(
    prompt: str,
    *,
    model: str | None = None,
    provider: str | None = None,
    api_url: str = "",
    api_key: str = "",
    max_tokens: int = 200,
    timeout: int = 15,
) -> str:
    """Non-streaming LLM call for fast, short completions (query expansion, etc.).

    Returns the full response text. Uses the same provider config as stream_chat.
    """
    _provider = (provider or LLM_PROVIDER).lower()
    _model = model or LLM_MODEL

    messages = [{"role": "user", "content": prompt}]

    if _provider == "ollama":
        resp = req.post(
            f"{OLLAMA_URL}/api/chat",
            json={"model": _model, "messages": messages, "stream": False},
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json().get("message", {}).get("content", "")

    # OpenAI-compatible
    url = api_url or LLM_API_URL
    key = api_key or LLM_API_KEY
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    resp = req.post(
        f"{url}/chat/completions",
        headers=headers,
        json={
            "model": _model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.3,
            "stream": False,
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def stream_chat(
    query: str,
    context: str,
    *,
    model: str | None = None,
    system_prompt: str | None = None,
    provider: str | None = None,
    api_url: str = "",
    api_key: str = "",
    timeout: int = 120,
) -> Generator[str, None, None]:
    """Stream LLM tokens for a RAG query.

    Args:
        query: The user question.
        context: Formatted context from retrieved chunks.
        model: Override LLM_MODEL from config.
        system_prompt: Override SYSTEM_PROMPT from config.
        provider: Override LLM_PROVIDER from config ("ollama" or "openai").
        api_url: Override LLM_API_URL (for fallback provider).
        api_key: Override LLM_API_KEY (for fallback provider).
        timeout: Request timeout in seconds.

    Yields:
        str: Individual tokens from the LLM response.
    """
    _provider = (provider or LLM_PROVIDER).lower()
    _model = model or LLM_MODEL
    _prompt = system_prompt or SYSTEM_PROMPT

    # Build message — if no context, just send the question directly
    user_content = f"Context:\n{context}\n\nQuestion: {query}" if context else query

    messages = [
        {"role": "system", "content": _prompt},
        {"role": "user", "content": user_content},
    ]

    stream_fn = _PROVIDERS.get(_provider)
    if stream_fn is None:
        raise ValueError(f"Unknown LLM_PROVIDER '{_provider}'. Use: {list(_PROVIDERS.keys())}")

    logger.info("LLM stream: provider=%s model=%s", _provider, _model)
    yield from stream_fn(messages, _model, timeout, api_url=api_url, api_key=api_key)

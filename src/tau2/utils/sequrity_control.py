"""
Sequrity Control API (dual-LLM secure tool use) — optional backend for generate().

Docs: https://sequrity-ai.github.io/sequrity-api/dev/control/getting_started/tool_use_dual_llm/

Enable with SEQURITY_CONTROL_ENABLED=1 and set SEQURITY_API_KEY plus a provider key
(X-Api-Key), e.g. OPENROUTER_API_KEY when SEQURITY_SERVICE_PROVIDER=openrouter.
"""

from __future__ import annotations

import json
import os
from contextvars import ContextVar
from typing import Any, Literal, Optional

import requests
from loguru import logger

from tau2.data_model.message import AssistantMessage, ToolCall

# CLI-selected routing for Sequrity (set per simulation worker in runner/batch.py).
SEQURITY_CLI_MODE: ContextVar[Optional[str]] = ContextVar(
    "sequrity_cli_mode", default=None
)

# OpenAI chat-completions-style parameters forwarded to Sequrity when present.
_FORWARD_PARAM_KEYS = frozenset(
    {
        "temperature",
        "top_p",
        "max_tokens",
        "frequency_penalty",
        "presence_penalty",
        "stop",
        "seed",
    }
)


def control_enabled() -> bool:
    return os.getenv("SEQURITY_CONTROL_ENABLED", "").lower() in (
        "1",
        "true",
        "yes",
    )


def use_sequrity_for_call(
    llm_role: Optional[Literal["agent", "user"]],
) -> bool:
    """Whether this LLM call should use Sequrity Control (requires control_enabled()).

    CLI ``--sequrity-mode`` is stored in :data:`SEQURITY_CLI_MODE` during batch runs.
    When unset (None), legacy behavior: if ``SEQURITY_CONTROL_ENABLED``, all calls use Sequrity.

    * ``none`` — never use Sequrity (even if env flag is set).
    * ``both`` — all ``generate()`` calls (including evaluators, reviews).
    * ``agent`` / ``user`` — only calls tagged with that ``llm_role``.
    """
    if not control_enabled():
        return False
    mode = SEQURITY_CLI_MODE.get()
    if mode is None:
        return True
    if mode == "none":
        return False
    if mode == "both":
        return True
    if mode == "agent":
        return llm_role == "agent"
    if mode == "user":
        return llm_role == "user"
    return False


def _provider_api_key() -> str:
    env_direct = os.getenv("SEQURITY_PROVIDER_API_KEY")
    if env_direct:
        return env_direct
    provider = os.getenv("SEQURITY_SERVICE_PROVIDER", "openrouter").lower()
    if provider == "openrouter":
        return os.getenv("OPENROUTER_API_KEY", "")
    if provider == "openai":
        return os.getenv("OPENAI_API_KEY", "")
    if provider == "anthropic":
        return os.getenv("ANTHROPIC_API_KEY", "")
    return ""


def _validate_config() -> tuple[str, str, str, str]:
    """Returns (sequrity_key, provider_key, base_url, service_provider)."""
    sequrity_key = os.getenv("SEQURITY_API_KEY", "")
    if not sequrity_key:
        raise ValueError(
            "SEQURITY_CONTROL_ENABLED is set but SEQURITY_API_KEY is missing."
        )
    provider_key = _provider_api_key()
    if not provider_key:
        raise ValueError(
            "Set SEQURITY_PROVIDER_API_KEY or a provider env key "
            "(e.g. OPENROUTER_API_KEY for SEQURITY_SERVICE_PROVIDER=openrouter)."
        )
    base_url = os.getenv("SEQURITY_BASE_URL", "https://api.sequrity.ai").rstrip("/")
    service_provider = os.getenv("SEQURITY_SERVICE_PROVIDER", "openrouter")
    return sequrity_key, provider_key, base_url, service_provider


def _optional_header(name: str) -> Optional[str]:
    v = os.getenv(name)
    return v if v else None


def chat_completion(
    *,
    model: str,
    messages: list[dict[str, Any]],
    tools: Optional[list[dict[str, Any]]],
    tool_choice: Optional[str],
    completion_kwargs: dict[str, Any],
    num_retries: int,
) -> dict[str, Any]:
    """POST /control/chat/{provider}/v1/chat/completions (OpenAI-compatible body)."""
    sequrity_key, provider_key, base_url, service_provider = _validate_config()
    url = f"{base_url}/control/chat/{service_provider}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {sequrity_key}",
        "Content-Type": "application/json",
        "X-Api-Key": provider_key,
    }
    features = _optional_header("SEQURITY_FEATURES_JSON")
    if features is None:
        features = json.dumps({"agent_arch": "dual-llm"})
    headers["X-Features"] = features

    policy = _optional_header("SEQURITY_POLICY_JSON")
    if policy:
        headers["X-Policy"] = policy

    config = _optional_header("SEQURITY_CONFIG_JSON")
    if config:
        headers["X-Config"] = config

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        **{k: v for k, v in completion_kwargs.items() if k in _FORWARD_PARAM_KEYS},
    }
    if tools:
        payload["tools"] = tools
    if tool_choice is not None:
        payload["tool_choice"] = tool_choice

    last_exc: Optional[Exception] = None
    for attempt in range(max(1, num_retries)):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=600)
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, json.JSONDecodeError) as e:
            last_exc = e
            logger.warning(
                "Sequrity Control request failed (attempt {}/{}): {}",
                attempt + 1,
                num_retries,
                e,
            )
    assert last_exc is not None
    raise last_exc


def assistant_message_from_response(
    response_json: dict[str, Any],
    *,
    generation_time_seconds: float,
) -> AssistantMessage:
    """Map Sequrity/OpenAI-style JSON to AssistantMessage."""
    choices = response_json.get("choices") or []
    if not choices:
        raise ValueError("Sequrity response has no choices")
    msg = choices[0].get("message") or {}
    if msg.get("role") not in (None, "assistant"):
        logger.warning("Unexpected message role from Sequrity: {}", msg.get("role"))

    content = msg.get("content")
    raw_tool_calls = msg.get("tool_calls") or []
    tool_calls: list[ToolCall] = []
    for tc in raw_tool_calls:
        fn = tc.get("function") or {}
        name = fn.get("name") or ""
        args_raw = fn.get("arguments")
        if isinstance(args_raw, str):
            try:
                arguments = (
                    json.loads(args_raw) if args_raw and args_raw.strip() else {}
                )
            except json.JSONDecodeError:
                arguments = {}
        elif isinstance(args_raw, dict):
            arguments = args_raw
        else:
            arguments = {}
        tc_id = tc.get("id") or ""
        tool_calls.append(ToolCall(id=tc_id, name=name, arguments=arguments))

    usage_out = None
    usage = response_json.get("usage")
    if isinstance(usage, dict):
        usage_out = {
            "completion_tokens": usage.get("completion_tokens"),
            "prompt_tokens": usage.get("prompt_tokens"),
        }

    finish_reason = choices[0].get("finish_reason")
    if finish_reason == "length":
        logger.warning("Output might be incomplete due to token limit (Sequrity).")

    return AssistantMessage(
        role="assistant",
        content=content,
        tool_calls=tool_calls or None,
        cost=0.0,
        usage=usage_out,
        raw_data=response_json,
        generation_time_seconds=generation_time_seconds,
    )

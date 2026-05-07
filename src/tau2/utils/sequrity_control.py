"""
Sequrity Control API (dual-LLM secure tool use) — optional backend for generate().

Docs: https://sequrity-ai.github.io/sequrity-api/dev/control/getting_started/tool_use_dual_llm/

Routing is controlled only by ``--sequrity-mode`` / RunConfig ``sequrity_mode`` (see
:data:`SEQURITY_CLI_MODE`). When that mode sends traffic through Sequrity, set
SEQURITY_API_KEY plus a provider key (X-Api-Key), e.g. OPENROUTER_API_KEY when
SEQURITY_SERVICE_PROVIDER=openrouter.

Requests always send ``X-Features`` = dual-LLM JSON (``{"agent_arch": "dual-llm"}``),
and ``X-Config`` with ``response_format.include_program`` plus ``fsm.max_n_turns`` (50; Sequrity
default dual-LLM preset is 5).
``X-Policy`` is not sent. When the API returns a generated program (often under
``choices[0].message.program`` or JSON ``content``, not only when ``content`` is set —
tool-call rounds frequently have ``content`` null), it is printed (Rich syntax highlight).
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


def use_sequrity_for_call(
    llm_role: Optional[Literal["agent", "user"]],
) -> bool:
    """Whether this LLM call should use Sequrity Control.

    CLI ``--sequrity-mode`` is stored in :data:`SEQURITY_CLI_MODE` during batch runs.
    When unset (``None``), no Sequrity — use LiteLLM (default tau2 behavior).

    * ``none`` — never use Sequrity.
    * ``both`` — all ``generate()`` calls (including evaluators, reviews).
    * ``agent`` / ``user`` — only calls tagged with that ``llm_role``.
    """
    mode = SEQURITY_CLI_MODE.get()
    if mode is None or mode == "none":
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
            "SEQURITY_API_KEY is missing (required when Sequrity routing is active)."
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


# Serialized form of Sequrity ``FeaturesHeader.dual_llm()`` (REST tutorial / OpenAPI parity).
FEATURES_HEADER_DUAL_LLM_JSON = json.dumps({"agent_arch": "dual-llm"})

# FineGrainedConfigHeader.dual_llm(include_program=True, max_n_turns=50)-aligned defaults.
FINE_GRAINED_CONFIG_INCLUDE_PROGRAM_JSON = json.dumps(
    {
        "response_format": {"include_program": True},
        "fsm": {"max_n_turns": 50, "max_pllm_steps": 1},
    }
)


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
        "X-Features": FEATURES_HEADER_DUAL_LLM_JSON,
        "X-Config": FINE_GRAINED_CONFIG_INCLUDE_PROGRAM_JSON,
    }

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


def _find_program_in_object(
    obj: Any, *, max_depth: int, _depth: int = 0
) -> Optional[str]:
    """Depth-limited search for a string ``program`` field (Sequrity-specific nesting)."""
    if _depth > max_depth:
        return None
    if isinstance(obj, dict):
        prog = obj.get("program")
        if isinstance(prog, str) and prog.strip():
            return prog
        for v in obj.values():
            found = _find_program_in_object(v, max_depth=max_depth, _depth=_depth + 1)
            if found:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = _find_program_in_object(
                item, max_depth=max_depth, _depth=_depth + 1
            )
            if found:
                return found
    return None


def extract_program_from_sequrity_response(
    response_json: dict[str, Any],
) -> Optional[str]:
    """Best-effort extraction of PLLM program text from a chat/completions JSON body."""

    def _str_prog(val: Any) -> Optional[str]:
        if isinstance(val, str) and val.strip():
            return val
        return None

    p = _str_prog(response_json.get("program"))
    if p:
        return p

    choices = response_json.get("choices") or []
    if not choices:
        return None
    ch0 = choices[0]
    if not isinstance(ch0, dict):
        return None

    p = _str_prog(ch0.get("program"))
    if p:
        return p

    msg = ch0.get("message")
    if isinstance(msg, dict):
        p = _str_prog(msg.get("program"))
        if p:
            return p

        for nest_key in ("sequrity", "metadata", "extensions"):
            nest = msg.get(nest_key)
            if isinstance(nest, dict):
                p = _str_prog(nest.get("program"))
                if p:
                    return p

        content = msg.get("content")
        if isinstance(content, str):
            s = content.strip()
            if s.startswith("{"):
                try:
                    data = json.loads(s)
                    if isinstance(data, dict):
                        p = _str_prog(data.get("program"))
                        if p:
                            return p
                except json.JSONDecodeError:
                    pass
        elif isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "text":
                    text = part.get("text")
                    if isinstance(text, str) and text.strip().startswith("{"):
                        try:
                            data = json.loads(text.strip())
                            if isinstance(data, dict):
                                p = _str_prog(data.get("program"))
                                if p:
                                    return p
                        except json.JSONDecodeError:
                            continue

    return _find_program_in_object(ch0, max_depth=10)


def _render_sequrity_program(program: str) -> None:
    """Print program source with Rich, or plain fallback."""
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.syntax import Syntax

        Console().print(
            Panel(
                Syntax(
                    program,
                    "python",
                    theme="monokai",
                    line_numbers=True,
                    word_wrap=True,
                ),
                title="Sequrity generated program",
                border_style="cyan",
            )
        )
    except Exception as exc:
        logger.debug("Rich program render failed ({}), using plain print", exc)
        print("\n--- Sequrity generated program ---\n", program, "\n---\n", flush=True)


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

    program = extract_program_from_sequrity_response(response_json)
    if program:
        _render_sequrity_program(program)

    return AssistantMessage(
        role="assistant",
        content=content,
        tool_calls=tool_calls or None,
        cost=0.0,
        usage=usage_out,
        raw_data=response_json,
        generation_time_seconds=generation_time_seconds,
    )

"""Manually Chain-of-Thought (CoT) system.

Wraps an LLM callback with explicit reasoning instructions,
parses structured <thinking>/<answer> output, and supports
self-verification and continuation from prior reasoning.
"""

from __future__ import annotations

import regex as re
from dataclasses import dataclass
from typing import Optional

from kimi_agent_sdk import Session
from kimix.ui.printing import MessageType
from kimix.utils import (
    _create_session_async,
    create_session,
    get_default_session,
    prompt,
    prompt_async,
)
from kimix.utils import _globals
from kimix.utils.system_prompt import SystemPromptType


@dataclass
class CoTResult:
    """Result of a manual CoT prompt."""

    thinking: str
    quit: bool = False


_VERIFY_SUFFIX = (
    "\n\nReview your reasoning for errors or bad assumptions, correct them, then finalize."
)

_CONTINUE_PREFIX = (
    "Continue from the prior thinking. Verify, refine, then finalize.\n\n"
    "<thinking>\n{thinking}\n</thinking>"
)


def _build_prompt(
    user_prompt: str,
    existing_thinking: Optional[str] = None,
    self_verify: bool = False,
) -> str:
    parts: list[str] = []
    if existing_thinking is not None:
        parts.append(_CONTINUE_PREFIX.format(thinking=existing_thinking.strip()))
    parts.append(user_prompt.strip())
    prompt_str = "\n\n".join(parts)
    if self_verify:
        prompt_str += _VERIFY_SUFFIX
    return prompt_str


_THINKING_RE = re.compile(r"<thinking>(.*?)</thinking>", re.DOTALL | re.IGNORECASE)
_QUIT_RE = re.compile(r"<quit\s*/?>", re.IGNORECASE)


def _parse_response(text: str) -> CoTResult:
    thinking_match = _THINKING_RE.search(text)
    quit_match = _QUIT_RE.search(text)
    thinking = thinking_match.group(1).strip() if thinking_match else ""
    return CoTResult(thinking=thinking, quit=bool(quit_match))


def _ensure_cot_session() -> Session:
    default = get_default_session()
    if default is not None and _globals._default_role == SystemPromptType.Thinker:
        return default
    session = create_session(agent_type=SystemPromptType.Thinker)
    _globals._default_session = session
    _globals._default_role = SystemPromptType.Thinker
    return session


async def _ensure_cot_session_async() -> Session:
    default = get_default_session()
    if default is not None and _globals._default_role == SystemPromptType.Thinker:
        return default
    session = await _create_session_async(agent_type=SystemPromptType.Thinker)
    _globals._default_session = session
    _globals._default_role = SystemPromptType.Thinker
    return session


def _prompt_to_text(prompt_str: str, session: Session) -> str:
    lst: list[str] = []
    def output_func(s: str, msg_type: MessageType) -> None:
        if msg_type != MessageType.Thinking:
            lst.append(s)
    prompt(prompt_str, session=session, output_function=output_func, merge_wire_messages=False)
    return "\n".join(lst)


async def _prompt_to_text_async(prompt_str: str, session: Session) -> str:
    lst: list[str] = []
    def output_func(s: str, msg_type: MessageType) -> None:
        if msg_type != MessageType.Thinking:
            lst.append(s)
    await prompt_async(prompt_str, session=session, output_function=output_func, merge_wire_messages=False)
    return "".join(lst)


async def cot_prompt_async(
    prompt_str: str,
    self_verify: bool = True,
    existing_thinking: Optional[str] = None,
    max_iterations: int = 10,
) -> CoTResult:
    """Run manual CoT with a Thinker session.

    The session is created with ``SystemPromptType.Thinker`` role.
    If a default session already exists, its role is recorded and it is left open.
    The model is prompted in a loop until it emits ``<quit/>``
    or ``max_iterations`` is reached.

    Parameters
    ----------
    prompt_str:
        The user prompt.
    self_verify:
        If True, append a self-verification instruction to each prompt.
    existing_thinking:
        If provided, ask the model to continue from this prior thinking.
    max_iterations:
        Maximum number of LLM calls before forcing a return.
    """
    session = await _ensure_cot_session_async()
    last_thinking = existing_thinking.strip() if existing_thinking is not None else None

    for _ in range(max_iterations):
        user_prompt = _build_prompt(prompt_str, last_thinking, self_verify)
        raw = await _prompt_to_text_async(user_prompt, session)
        result = _parse_response(raw)

        if result.thinking:
            last_thinking = result.thinking

        if result.quit:
            return CoTResult(
                thinking=last_thinking or "",
                quit=result.quit,
            )

    return CoTResult(thinking=last_thinking or "", quit=False)


def cot_prompt(
    prompt_str: str,
    self_verify: bool = True,
    existing_thinking: Optional[str] = None,
    max_iterations: int = 10,
) -> CoTResult:
    """Synchronous version of :func:`cot_prompt_async`.

    Parameters
    ----------
    prompt_str:
        The user prompt.
    self_verify:
        If True, append a self-verification instruction to each prompt.
    existing_thinking:
        If provided, ask the model to continue from this prior thinking.
    max_iterations:
        Maximum number of LLM calls before forcing a return.
    """
    session = _ensure_cot_session()
    last_thinking = existing_thinking.strip() if existing_thinking is not None else None

    for _ in range(max_iterations):
        user_prompt = _build_prompt(prompt_str, last_thinking, self_verify)
        raw = _prompt_to_text(user_prompt, session)
        result = _parse_response(raw)

        if result.thinking:
            last_thinking = result.thinking

        if result.quit:
            return CoTResult(
                thinking=last_thinking or "",
                quit=result.quit,
            )

    return CoTResult(thinking=last_thinking or "", quit=False)


async def cot_prompt_with_verification_async(
    prompt_str: str,
    existing_thinking: Optional[str] = None,
) -> CoTResult:
    """Two-pass CoT: generate reasoning, then verify and refine.

    First pass runs without self-verify to get initial thinking.
    Second pass feeds the thinking back as ``existing_thinking`` with verification enabled.
    """
    first = await cot_prompt_async(
        prompt_str,
        self_verify=False,
        existing_thinking=existing_thinking,
    )
    if not first.thinking:
        return first
    second = await cot_prompt_async(
        prompt_str,
        self_verify=True,
        existing_thinking=first.thinking,
    )
    return second


def cot_prompt_with_verification(
    prompt_str: str,
    existing_thinking: Optional[str] = None,
) -> CoTResult:
    """Synchronous two-pass CoT with verification."""
    first = cot_prompt(
        prompt_str,
        self_verify=False,
        existing_thinking=existing_thinking,
    )
    if not first.thinking:
        return first
    second = cot_prompt(
        prompt_str,
        self_verify=True,
        existing_thinking=first.thinking,
    )
    return second

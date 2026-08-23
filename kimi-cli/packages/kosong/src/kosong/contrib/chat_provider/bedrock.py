"""AWS Bedrock chat provider.

Implemented on top of ``anthropic.lib.bedrock.AsyncAnthropicBedrock``, which
shares the Anthropic Messages API surface (and the kosong Anthropic message
conversion / streaming logic) but signs requests with AWS SigV4 credentials.

The AWS SDK (``boto3``) is only needed for profile-based credential
resolution; plain env-var credentials (``AWS_ACCESS_KEY_ID`` /
``AWS_SECRET_ACCESS_KEY`` / ``AWS_SESSION_TOKEN``) work without it. The
bedrock client is imported lazily so kosong never requires AWS tooling at
import time.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import httpx2

from kosong.chat_provider import (
    DEFAULT_MAX_RETRIES,
    ChatProvider,
    ChatProviderError,
    ThinkingEffort,
)
from kosong.contrib.chat_provider.anthropic import Anthropic, AnthropicStreamedMessage
from kosong.message import Message
from kosong.tooling import Tool

if TYPE_CHECKING:

    def type_check(bedrock: Bedrock):
        _: ChatProvider = bedrock


def _resolve_region(aws_region: str | None) -> str:
    """Resolve the AWS region from the argument or environment, defaulting to us-east-1."""
    if aws_region:
        return aws_region
    for env_name in ("AWS_REGION", "AWS_DEFAULT_REGION"):
        value = os.environ.get(env_name)
        if value:
            return value
    return "us-east-1"


class Bedrock(Anthropic):
    """Chat provider backed by AWS Bedrock's Anthropic-compatible Messages API.

    >>> chat_provider = Bedrock(model="anthropic.claude-sonnet-4-20250514")
    >>> chat_provider.name
    'bedrock'
    >>> chat_provider.model_name
    'anthropic.claude-sonnet-4-20250514'
    """

    name = "bedrock"

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        aws_access_key: str | None = None,
        aws_secret_key: str | None = None,
        aws_region: str | None = None,
        aws_session_token: str | None = None,
        aws_profile: str | None = None,
        stream: bool = True,
        default_max_tokens: int = 50000,
        metadata: Any = None,
        supported_efforts: set[ThinkingEffort] | None = None,
        **client_kwargs: Any,
    ):
        """Initialize the Bedrock chat provider.

        Args:
            model: Bedrock model id (e.g. ``anthropic.claude-sonnet-4-20250514``).
            api_key: Accepted for API parity with the other providers; Bedrock
                authenticates via AWS credentials instead.
            base_url: Optional ``bedrock-runtime`` base URL override.
            aws_access_key: AWS access key. Defaults to ``AWS_ACCESS_KEY_ID``.
            aws_secret_key: AWS secret key. Defaults to ``AWS_SECRET_ACCESS_KEY``.
            aws_region: AWS region. Defaults to ``AWS_REGION`` /
                ``AWS_DEFAULT_REGION`` / ``us-east-1``.
            aws_session_token: Optional AWS session token.
            aws_profile: Optional AWS named profile (requires ``boto3``).
            stream: Whether to generate responses as a stream.
            default_max_tokens: Default ``max_tokens`` for the Messages API.
            metadata: Optional Anthropic ``metadata`` payload.
            supported_efforts: Restrict the thinking-effort levels accepted.
            client_kwargs: Extra kwargs forwarded to the Bedrock client.

        Raises:
            ChatProviderError: If the Bedrock client cannot be imported or a
                named AWS profile is requested without ``boto3``.
        """
        try:
            from anthropic.lib.bedrock import AsyncAnthropicBedrock
        except ModuleNotFoundError as exc:
            raise ChatProviderError(
                "Bedrock support requires the Anthropic SDK bedrock module. "
                'Install with `pip install "kosong[contrib]"`.'
            ) from exc

        if aws_profile and aws_access_key is None and aws_secret_key is None:
            import importlib.util

            if importlib.util.find_spec("boto3") is None:
                raise ChatProviderError(
                    "Bedrock profile-based credentials require the optional 'boto3' "
                    "dependency. Install it (`pip install boto3`) or set "
                    "AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY."
                )

        region = _resolve_region(aws_region)
        # Provide our own httpx2.AsyncClient so the anthropic SDK 1.0+ (which
        # backs AsyncAnthropicBedrock) does not create its default
        # ``AsyncHttpxClientWrapper`` — that wrapper's ``__del__`` schedules
        # ``self.aclose()`` on the running loop and can surface a noisy
        # ``RuntimeError: Event loop is closed`` traceback on Windows/Python
        # 3.14.  The SDK validates that ``http_client`` is an
        # ``httpx2.AsyncClient``, so a plain ``httpx.AsyncClient`` is rejected.
        client_kwargs = dict(client_kwargs)
        client_kwargs.setdefault("max_retries", DEFAULT_MAX_RETRIES)
        if "http_client" not in client_kwargs:
            client_kwargs["http_client"] = httpx2.AsyncClient()

        self._model = model
        self._stream = stream
        self._supported_efforts = (
            frozenset(supported_efforts)
            if supported_efforts is not None
            else Anthropic._DEFAULT_SUPPORTED_EFFORTS
        )
        self._client = AsyncAnthropicBedrock(
            aws_access_key=aws_access_key or os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_key=aws_secret_key or os.environ.get("AWS_SECRET_ACCESS_KEY"),
            aws_session_token=aws_session_token or os.environ.get("AWS_SESSION_TOKEN"),
            aws_profile=aws_profile or os.environ.get("AWS_PROFILE"),
            aws_region=region,
            base_url=base_url,
            **client_kwargs,
        )
        self._tool_message_conversion = None
        self._metadata = metadata
        # Bedrock does not need the interleaved-thinking beta header; keep the
        # beta list empty so generate() does not send an unknown beta token.
        self._generation_kwargs: Anthropic.GenerationKwargs = {
            "max_tokens": default_max_tokens,
            "beta_features": [],
        }
        try:
            self._loop: asyncio.AbstractEventLoop | None = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None

    async def generate(
        self,
        system_prompt: str,
        tools: Sequence[Tool],
        history: Sequence[Message],
    ) -> AnthropicStreamedMessage:
        """Generate a response, converting missing AWS SDK errors lazily.

        The Bedrock client only imports ``boto3``/``botocore`` when a request
        is actually signed, so construction stays dependency-free. If the AWS
        SDK is missing at request time, surface a clear
        :class:`~kosong.chat_provider.ChatProviderError` instead of a raw
        ``ModuleNotFoundError``.
        """
        try:
            return await super().generate(system_prompt, tools, history)
        except ModuleNotFoundError as exc:
            if "boto3" in str(exc) or "botocore" in str(exc):
                raise ChatProviderError(
                    "Bedrock request signing requires the optional 'boto3' dependency. "
                    "Install it (`pip install boto3`)."
                ) from exc
            raise


__all__ = ["Bedrock"]

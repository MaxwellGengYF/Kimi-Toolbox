"""xAI (Grok) chat provider.

Implements an OpenAI-compatible Responses API provider for xAI's Grok models.
"""

from typing import TYPE_CHECKING, Any

from kosong.contrib.chat_provider.openai_responses import OpenAIResponses

if TYPE_CHECKING:

    def type_check(xai: XAI):
        from kosong.chat_provider import ChatProvider, RetryableChatProvider

        _: ChatProvider = xai
        _: RetryableChatProvider = xai


class XAI(OpenAIResponses):
    """xAI chat provider using the OpenAI-compatible Responses API.

    >>> chat_provider = XAI(model="grok-3", api_key="xai-1234567890")
    >>> chat_provider.name
    'xai'
    >>> chat_provider.model_name
    'grok-3'
    """

    name = "xai"

    _DEFAULT_BASE_URL = "https://api.x.ai/v1"

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        **client_kwargs: Any,
    ):
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url or self._DEFAULT_BASE_URL,
            **client_kwargs,
        )

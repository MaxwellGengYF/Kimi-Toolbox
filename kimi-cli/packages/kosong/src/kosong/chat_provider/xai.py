"""xAI (Grok) chat provider.

Implements an OpenAI-compatible Responses API provider for xAI's Grok models.
"""

from typing import TYPE_CHECKING, Any

from kosong.chat_provider.openai_common import (
    close_replaced_openai_client,
    create_openai_client,
)
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
    _TOKEN_AUTH_HEADER = "X-XAI-Token-Auth"
    _TOKEN_AUTH_VALUE = "xai-grok-cli"

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        token_auth: bool = False,
        **client_kwargs: Any,
    ):
        """Initialize the xAI provider.

        Args:
            model: Model name (e.g. ``grok-3``).
            api_key: API key or OAuth access token.
            base_url: API base URL. Defaults to ``https://api.x.ai/v1``.
            token_auth: When ``True``, send the ``X-XAI-Token-Auth: xai-grok-cli``
                header required for xAI OAuth access tokens. Plain API keys should
                leave this ``False``.
            client_kwargs: Additional arguments forwarded to the OpenAI client.
        """
        if token_auth:
            client_kwargs = dict(client_kwargs)
            default_headers = dict(client_kwargs.pop("default_headers", None) or {})
            default_headers[self._TOKEN_AUTH_HEADER] = self._TOKEN_AUTH_VALUE
            client_kwargs["default_headers"] = default_headers
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url or self._DEFAULT_BASE_URL,
            **client_kwargs,
        )
        self._token_auth = token_auth

    @property
    def token_auth(self) -> bool:
        """Whether the provider is sending the xAI OAuth token auth header.

        xAI OAuth access tokens must be sent with the header
        ``X-XAI-Token-Auth: xai-grok-cli``. Plain API keys must not send this
        header. See ``grok_auth_credentials::apply`` in the grok-build reference.
        """
        return self._token_auth

    @token_auth.setter
    def token_auth(self, value: bool) -> None:
        """Enable or disable the xAI OAuth token auth header.

        Recreates the underlying OpenAI client so the header change takes effect
        on subsequent requests.
        """
        if self._token_auth == value:
            return
        self._token_auth = value
        client_kwargs = dict(self._client_kwargs)
        default_headers = dict(client_kwargs.pop("default_headers", None) or {})
        if value:
            default_headers[self._TOKEN_AUTH_HEADER] = self._TOKEN_AUTH_VALUE
        else:
            default_headers.pop(self._TOKEN_AUTH_HEADER, None)
        if default_headers:
            client_kwargs["default_headers"] = default_headers
        else:
            client_kwargs.pop("default_headers", None)
        self._client_kwargs = client_kwargs
        old_client = self.client
        self.client = create_openai_client(
            api_key=old_client.api_key,
            base_url=self._base_url,
            client_kwargs=client_kwargs,
        )
        self._api_key = old_client.api_key
        close_replaced_openai_client(old_client, client_kwargs=client_kwargs)

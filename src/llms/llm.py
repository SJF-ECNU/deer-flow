# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import logging
import os
from pathlib import Path
from typing import Any, Dict, Mapping, get_args

import httpx
from langchain_core.language_models import BaseChatModel
from langchain_core.rate_limiters import BaseRateLimiter, InMemoryRateLimiter
from langchain_deepseek import ChatDeepSeek
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_openai.chat_models.base import (
    WellKnownTools,
    _convert_to_openai_response_format,
    convert_to_openai_tool,
)
from pydantic import ConfigDict

from src.config import load_yaml_config
from src.config.agents import LLMType
from src.llms.providers.dashscope import ChatDashscope

logger = logging.getLogger(__name__)

# Cache for LLM instances
_llm_cache: dict[LLMType, BaseChatModel] = {}
_rate_limiter_cache: dict[LLMType, BaseRateLimiter] = {}


def _get_env_rate_limit_default(env_key: str, fallback: float) -> float:
    raw_value = os.getenv(env_key)
    if raw_value is None:
        return fallback
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        logger.warning(
            "Invalid global rate limit value %s=%s, falling back to %.2f",
            env_key,
            raw_value,
            fallback,
        )
        return fallback
    if value <= 0:
        logger.warning(
            "Non-positive global rate limit value %s=%.2f, falling back to %.2f",
            env_key,
            value,
            fallback,
        )
        return fallback
    return value


DEFAULT_RATE_LIMIT_REQUESTS_PER_SECOND = _get_env_rate_limit_default(
    "RATE_LIMIT_DEFAULT_REQUESTS_PER_SECOND", 3.0
)
DEFAULT_RATE_LIMIT_CHECK_INTERVAL = _get_env_rate_limit_default(
    "RATE_LIMIT_DEFAULT_CHECK_INTERVAL", 0.1
)
DEFAULT_RATE_LIMIT_BUCKET_SIZE = _get_env_rate_limit_default(
    "RATE_LIMIT_DEFAULT_BUCKET_SIZE", 3.0
)

# Allowed LLM configuration keys to prevent unexpected parameters from being passed
# to LLM constructors (Issue #411 - SEARCH_ENGINE warning fix)
ALLOWED_LLM_CONFIG_KEYS = {
    # Common LLM configuration keys
    "model",
    "api_key",
    "base_url",
    "api_base",
    "max_retries",
    "timeout",
    "max_tokens",
    "temperature",
    "top_p",
    "frequency_penalty",
    "presence_penalty",
    "stop",
    "n",
    "stream",
    "logprobs",
    "echo",
    "best_of",
    "logit_bias",
    "user",
    "seed",
    # SSL and HTTP client settings
    "verify_ssl",
    "http_client",
    "http_async_client",
    # Platform-specific keys
    "platform",
    "google_api_key",
    # Azure-specific keys
    "azure_endpoint",
    "azure_deployment",
    "api_version",
    "azure_ad_token",
    "azure_ad_token_provider",
    # Dashscope/Doubao specific keys
    "extra_body",
    # Token limit for context compression (removed before passing to LLM)
    "token_limit",
    # Default headers
    "default_headers",
    "default_query",
}


class RateLimitedChatModel(BaseChatModel):
    """Wrapper that throttles requests before delegating to the LLM."""

    model: BaseChatModel
    rate_limiter: BaseRateLimiter

    model_config = ConfigDict(
        arbitrary_types_allowed=True, extra="ignore", protected_namespaces=()
    )

    def __getattr__(self, name: str) -> Any:  # pragma: no cover - passthrough
        return getattr(self.model, name)

    @property
    def _llm_type(self) -> str:
        inner_type = getattr(self.model, "_llm_type", lambda: "unknown")
        resolved_type = inner_type() if callable(inner_type) else inner_type
        return f"rate_limited-{resolved_type}"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        base_params_attr = getattr(self.model, "_identifying_params", {})
        base_params = (
            base_params_attr() if callable(base_params_attr) else base_params_attr
        )
        return {
            "rate_limiter": self.rate_limiter.__class__.__name__,
            "model_params": base_params,
        }

    def _generate(
        self,
        messages,
        stop=None,
        run_manager=None,
        **kwargs: Any,
    ) -> Any:
        self.rate_limiter.acquire()
        return self.model._generate(
            messages, stop=stop, run_manager=run_manager, **kwargs
        )

    async def _agenerate(
        self,
        messages,
        stop=None,
        run_manager=None,
        **kwargs: Any,
    ) -> Any:
        await self.rate_limiter.aacquire()
        return await self.model._agenerate(
            messages, stop=stop, run_manager=run_manager, **kwargs
        )

    def _stream(
        self,
        messages,
        stop=None,
        run_manager=None,
        **kwargs: Any,
    ):
        self.rate_limiter.acquire()
        yield from self.model._stream(
            messages, stop=stop, run_manager=run_manager, **kwargs
        )

    async def _astream(
        self,
        messages,
        stop=None,
        run_manager=None,
        **kwargs: Any,
    ):
        await self.rate_limiter.aacquire()
        async for chunk in self.model._astream(
            messages, stop=stop, run_manager=run_manager, **kwargs
        ):
            yield chunk

    def bind_tools(
        self,
        tools,
        *,
        tool_choice=None,
        strict=None,
        parallel_tool_calls=None,
        response_format=None,
        **kwargs: Any,
    ):
        if parallel_tool_calls is not None:
            kwargs["parallel_tool_calls"] = parallel_tool_calls

        formatted_tools = [
            convert_to_openai_tool(tool, strict=strict) for tool in tools
        ]
        tool_names = []
        for tool in formatted_tools:
            if "function" in tool:
                tool_names.append(tool["function"].get("name"))
            elif "name" in tool:
                tool_names.append(tool.get("name"))

        if tool_choice:
            if isinstance(tool_choice, str):
                if tool_choice in tool_names:
                    tool_choice = {
                        "type": "function",
                        "function": {"name": tool_choice},
                    }
                elif tool_choice in WellKnownTools:
                    tool_choice = {"type": tool_choice}
                elif tool_choice == "any":
                    tool_choice = "required"
            elif isinstance(tool_choice, bool):
                tool_choice = "required"
            elif isinstance(tool_choice, dict):
                pass
            else:
                raise ValueError(
                    "Unrecognized tool_choice type. Expected str, bool or dict. "
                    f"Received: {tool_choice}"
                )
            kwargs["tool_choice"] = tool_choice

        if response_format:
            if (
                isinstance(response_format, dict)
                and response_format.get("type") == "json_schema"
                and "schema" in response_format.get("json_schema", {})
            ):
                response_format = response_format["json_schema"].get("schema")
            kwargs["response_format"] = _convert_to_openai_response_format(
                response_format
            )

        return super().bind(tools=formatted_tools, **kwargs)


def _get_config_file_path() -> str:
    """Get the path to the configuration file."""
    return str((Path(__file__).parent.parent.parent / "conf.yaml").resolve())


def _get_llm_type_config_keys() -> dict[str, str]:
    """Get mapping of LLM types to their configuration keys."""
    return {
        "reasoning": "REASONING_MODEL",
        "basic": "BASIC_MODEL",
        "vision": "VISION_MODEL",
        "code": "CODE_MODEL",
    }


def _get_env_llm_conf(llm_type: str) -> Dict[str, Any]:
    """
    Get LLM configuration from environment variables.
    Environment variables should follow the format: {LLM_TYPE}__{KEY}
    e.g., BASIC_MODEL__api_key, BASIC_MODEL__base_url
    """
    prefix = f"{llm_type.upper()}_MODEL__"
    conf = {}
    for key, value in os.environ.items():
        if key.startswith(prefix):
            conf_key = key[len(prefix) :].lower()
            conf[conf_key] = value
    return conf


def _coerce_positive_float(
    value: Any, key: str, llm_type: str
) -> float | None:
    if value is None:
        return None
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        logger.warning(
            "Invalid rate limiter value for %s on %s: %s", key, llm_type, value
        )
        return None
    if value_float <= 0:
        logger.warning(
            "Ignoring non-positive rate limiter value for %s on %s: %s",
            key,
            llm_type,
            value_float,
        )
        return None
    return value_float


def _coerce_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.lower()
        if lowered in {"true", "1", "yes", "y", "on"}:
            return True
        if lowered in {"false", "0", "no", "n", "off"}:
            return False
    return None


def _extract_rate_limit_config(
    llm_type: LLMType, merged_conf: Dict[str, Any]
) -> Dict[str, float] | None:
    rate_limit_conf: Dict[str, Any] = {}

    if isinstance(merged_conf.get("rate_limit"), dict):
        rate_limit_conf.update(merged_conf.pop("rate_limit"))

    key_map = {
        "rate_limit_requests_per_second": "requests_per_second",
        "rate_limit_max_bucket_size": "max_bucket_size",
        "rate_limit_check_every_n_seconds": "check_every_n_seconds",
        "rate_limit_enabled": "enabled",
    }

    for raw_key, target_key in key_map.items():
        if raw_key in merged_conf:
            rate_limit_conf[target_key] = merged_conf.pop(raw_key)

    enabled_value = _coerce_bool(rate_limit_conf.pop("enabled", None))
    enabled = True if enabled_value is None else enabled_value
    if not enabled:
        logger.info("Rate limiter disabled for %s via configuration", llm_type)
        return None

    requests_per_second = _coerce_positive_float(
        rate_limit_conf.get("requests_per_second"),
        "requests_per_second",
        llm_type,
    )
    check_every_n_seconds = _coerce_positive_float(
        rate_limit_conf.get("check_every_n_seconds"),
        "check_every_n_seconds",
        llm_type,
    )
    max_bucket_size = _coerce_positive_float(
        rate_limit_conf.get("max_bucket_size"), "max_bucket_size", llm_type
    )

    if not rate_limit_conf and requests_per_second is None:
        return {
            "requests_per_second": DEFAULT_RATE_LIMIT_REQUESTS_PER_SECOND,
            "check_every_n_seconds": DEFAULT_RATE_LIMIT_CHECK_INTERVAL,
            "max_bucket_size": DEFAULT_RATE_LIMIT_BUCKET_SIZE,
        }

    return {
        "requests_per_second": requests_per_second
        or DEFAULT_RATE_LIMIT_REQUESTS_PER_SECOND,
        "check_every_n_seconds": check_every_n_seconds
        or DEFAULT_RATE_LIMIT_CHECK_INTERVAL,
        "max_bucket_size": max(
            max_bucket_size or DEFAULT_RATE_LIMIT_BUCKET_SIZE, 1.0
        ),
    }


def _get_rate_limiter(
    llm_type: LLMType, merged_conf: Dict[str, Any]
) -> BaseRateLimiter | None:
    if llm_type in _rate_limiter_cache:
        return _rate_limiter_cache[llm_type]

    rate_limit_config = _extract_rate_limit_config(llm_type, merged_conf)
    if not rate_limit_config:
        return None

    limiter = InMemoryRateLimiter(**rate_limit_config)
    _rate_limiter_cache[llm_type] = limiter

    logger.info(
        "Enabled rate limiter for %s: %.2f rps, bucket %.2f, check %.2fs",
        llm_type,
        rate_limit_config["requests_per_second"],
        rate_limit_config["max_bucket_size"],
        rate_limit_config["check_every_n_seconds"],
    )

    return limiter


def _create_llm_use_conf(llm_type: LLMType, conf: Dict[str, Any]) -> BaseChatModel:
    """Create LLM instance using configuration."""
    llm_type_config_keys = _get_llm_type_config_keys()
    config_key = llm_type_config_keys.get(llm_type)

    if not config_key:
        raise ValueError(f"Unknown LLM type: {llm_type}")

    llm_conf = conf.get(config_key, {})
    if not isinstance(llm_conf, dict):
        raise ValueError(f"Invalid LLM configuration for {llm_type}: {llm_conf}")

    # Get configuration from environment variables
    env_conf = _get_env_llm_conf(llm_type)

    # Merge configurations, with environment variables taking precedence
    merged_conf = {**llm_conf, **env_conf}

    # Build rate limiter configuration before filtering out unknown keys
    rate_limiter = _get_rate_limiter(llm_type, merged_conf)

    # Filter out unexpected parameters to prevent LangChain warnings (Issue #411)
    # This prevents configuration keys like SEARCH_ENGINE from being passed to LLM constructors
    allowed_keys_lower = {k.lower() for k in ALLOWED_LLM_CONFIG_KEYS}
    unexpected_keys = [key for key in merged_conf.keys() if key.lower() not in allowed_keys_lower]
    for key in unexpected_keys:
        removed_value = merged_conf.pop(key)
        logger.warning(
            "Removed unexpected LLM configuration key '%s'. "
            "This key is not a valid LLM parameter and may have been placed in "
            "the wrong section of conf.yaml. Valid LLM config keys include: "
            "model, api_key, base_url, max_retries, temperature, etc.",
            key,
        )

    # Remove unnecessary parameters when initializing the client
    if "token_limit" in merged_conf:
        merged_conf.pop("token_limit")

    if not merged_conf:
        raise ValueError(f"No configuration found for LLM type: {llm_type}")

    # Add max_retries to handle rate limit errors
    if "max_retries" not in merged_conf:
        merged_conf["max_retries"] = 3

    # Handle SSL verification settings
    verify_ssl = merged_conf.pop("verify_ssl", True)

    # Create custom HTTP client if SSL verification is disabled
    if not verify_ssl:
        http_client = httpx.Client(verify=False)
        http_async_client = httpx.AsyncClient(verify=False)
        merged_conf["http_client"] = http_client
        merged_conf["http_async_client"] = http_async_client

    # Check if it's Google AI Studio platform based on configuration
    platform = merged_conf.get("platform", "").lower()
    is_google_aistudio = platform == "google_aistudio" or platform == "google-aistudio"

    if is_google_aistudio:
        # Handle Google AI Studio specific configuration
        gemini_conf = merged_conf.copy()

        # Map common keys to Google AI Studio specific keys
        if "api_key" in gemini_conf:
            gemini_conf["google_api_key"] = gemini_conf.pop("api_key")

        # Remove base_url and platform since Google AI Studio doesn't use them
        gemini_conf.pop("base_url", None)
        gemini_conf.pop("platform", None)

        # Remove unsupported parameters for Google AI Studio
        gemini_conf.pop("http_client", None)
        gemini_conf.pop("http_async_client", None)

        llm: BaseChatModel = ChatGoogleGenerativeAI(**gemini_conf)
    elif "azure_endpoint" in merged_conf or os.getenv("AZURE_OPENAI_ENDPOINT"):
        llm = AzureChatOpenAI(**merged_conf)
    elif "base_url" in merged_conf and "dashscope." in merged_conf["base_url"]:
        if llm_type == "reasoning":
            merged_conf["extra_body"] = {"enable_thinking": True}
        else:
            merged_conf["extra_body"] = {"enable_thinking": False}
        llm = ChatDashscope(**merged_conf)
    elif llm_type == "reasoning":
        merged_conf["api_base"] = merged_conf.pop("base_url", None)
        llm = ChatDeepSeek(**merged_conf)
    else:
        llm = ChatOpenAI(**merged_conf)

    if rate_limiter:
        llm = RateLimitedChatModel(model=llm, rate_limiter=rate_limiter)

    return llm


def get_llm_by_type(llm_type: LLMType) -> BaseChatModel:
    """
    Get LLM instance by type. Returns cached instance if available.
    """
    if llm_type in _llm_cache:
        return _llm_cache[llm_type]

    conf = load_yaml_config(_get_config_file_path())
    llm = _create_llm_use_conf(llm_type, conf)
    _llm_cache[llm_type] = llm
    return llm


def get_configured_llm_models() -> dict[str, list[str]]:
    """
    Get all configured LLM models grouped by type.

    Returns:
        Dictionary mapping LLM type to list of configured model names.
    """
    try:
        conf = load_yaml_config(_get_config_file_path())
        llm_type_config_keys = _get_llm_type_config_keys()

        configured_models: dict[str, list[str]] = {}

        for llm_type in get_args(LLMType):
            # Get configuration from YAML file
            config_key = llm_type_config_keys.get(llm_type, "")
            yaml_conf = conf.get(config_key, {}) if config_key else {}

            # Get configuration from environment variables
            env_conf = _get_env_llm_conf(llm_type)

            # Merge configurations, with environment variables taking precedence
            merged_conf = {**yaml_conf, **env_conf}

            # Check if model is configured
            model_name = merged_conf.get("model")
            if model_name:
                configured_models.setdefault(llm_type, []).append(model_name)

        return configured_models

    except Exception as e:
        # Log error and return empty dict to avoid breaking the application
        print(f"Warning: Failed to load LLM configuration: {e}")
        return {}


def _get_model_token_limit_defaults() -> dict[str, int]:
    """
    Get default token limits for common LLM models.
    These are conservative limits to prevent token overflow errors (Issue #721).
    Users can override by setting token_limit in their config.
    """
    return {
        # OpenAI models
        "gpt-4o": 120000,
        "gpt-4-turbo": 120000,
        "gpt-4": 8000,
        "gpt-3.5-turbo": 4000,
        # Anthropic Claude
        "claude-3": 180000,
        "claude-2": 100000,
        # Google Gemini
        "gemini-2": 180000,
        "gemini-1.5-pro": 180000,
        "gemini-1.5-flash": 180000,
        "gemini-pro": 30000,
        # Bytedance Doubao
        "doubao": 200000,
        # DeepSeek
        "deepseek": 100000,
        # Ollama/local
        "qwen": 30000,
        "llama": 4000,
        # Default fallback for unknown models
        "default": 100000,
    }


def _infer_token_limit_from_model(model_name: str) -> int:
    """
    Infer a reasonable token limit from the model name.
    This helps protect against token overflow errors when token_limit is not explicitly configured.
    
    Args:
        model_name: The model name from configuration
        
    Returns:
        A conservative token limit based on known model capabilities
    """
    if not model_name:
        return 100000  # Safe default
    
    model_name_lower = model_name.lower()
    defaults = _get_model_token_limit_defaults()
    
    # Try exact or prefix matches
    for key, limit in defaults.items():
        if key in model_name_lower:
            return limit
    
    # Return safe default if no match found
    return defaults["default"]


def get_llm_token_limit_by_type(llm_type: str) -> int:
    """
    Get the maximum token limit for a given LLM type.
    
    Priority order:
    1. Explicitly configured token_limit in conf.yaml
    2. Inferred from model name based on known model capabilities
    3. Safe default (100,000 tokens)
    
    This helps prevent token overflow errors (Issue #721) even when token_limit is not configured.

    Args:
        llm_type (str): The type of LLM (e.g., 'basic', 'reasoning', 'vision', 'code').

    Returns:
        int: The maximum token limit for the specified LLM type (conservative estimate).
    """
    llm_type_config_keys = _get_llm_type_config_keys()
    config_key = llm_type_config_keys.get(llm_type)

    conf = load_yaml_config(_get_config_file_path())
    model_config = conf.get(config_key, {})
    
    # First priority: explicitly configured token_limit
    if "token_limit" in model_config:
        configured_limit = model_config["token_limit"]
        if configured_limit is not None:
            return configured_limit
    
    # Second priority: infer from model name
    model_name = model_config.get("model")
    if model_name:
        inferred_limit = _infer_token_limit_from_model(model_name)
        return inferred_limit
    
    # Fallback: safe default
    return _get_model_token_limit_defaults()["default"]


# In the future, we will use reasoning_llm and vl_llm for different purposes
# reasoning_llm = get_llm_by_type("reasoning")
# vl_llm = get_llm_by_type("vision")

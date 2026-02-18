import os
from pathlib import Path
from typing import Any, Optional

from langchain_openai import ChatOpenAI

from .base_client import BaseLLMClient
from .validators import validate_model


def _ensure_dotenv_loaded() -> None:
    """若 .env 尚未加载，从项目根再加载一次（兜底）。"""
    try:
        from dotenv import load_dotenv
        # openai_client.py 在 tradingagents/llm_clients/ 下，项目根为 parent.parent.parent
        project_root = Path(__file__).resolve().parent.parent.parent
        load_dotenv(project_root / ".env", override=True)
    except Exception:
        pass


class UnifiedChatOpenAI(ChatOpenAI):
    """ChatOpenAI subclass that strips incompatible params for certain models."""

    def __init__(self, **kwargs):
        model = kwargs.get("model", "")
        if self._is_reasoning_model(model):
            kwargs.pop("temperature", None)
            kwargs.pop("top_p", None)
        super().__init__(**kwargs)

    @staticmethod
    def _is_reasoning_model(model: str) -> bool:
        """Check if model is a reasoning model that doesn't support temperature."""
        model_lower = model.lower()
        return (
            model_lower.startswith("o1")
            or model_lower.startswith("o3")
            or "gpt-5" in model_lower
        )


class OpenAIClient(BaseLLMClient):
    """Client for OpenAI, Ollama, OpenRouter, and xAI providers."""

    def __init__(
        self,
        model: str,
        base_url: Optional[str] = None,
        provider: str = "openai",
        **kwargs,
    ):
        super().__init__(model, base_url, **kwargs)
        self.provider = provider.lower()

    def get_llm(self) -> Any:
        """Return configured ChatOpenAI instance."""
        _ensure_dotenv_loaded()
        llm_kwargs = {"model": self.model}

        if self.provider == "xai":
            llm_kwargs["base_url"] = "https://api.x.ai/v1"
            api_key = os.environ.get("XAI_API_KEY")
            if api_key:
                llm_kwargs["api_key"] = api_key
        elif self.provider == "deepseek":
            llm_kwargs["base_url"] = self.base_url or "https://api.deepseek.com/v1"
            api_key = (os.environ.get("DEEPSEEK_API_KEY") or "").strip()
            if not api_key:
                raise ValueError("DEEPSEEK_API_KEY 未设置或为空，请在 .env 中配置后重新运行")
            llm_kwargs["api_key"] = api_key
        elif self.provider == "qwen":
            llm_kwargs["base_url"] = self.base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"
            api_key = (os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("QWEN_API_KEY") or "").strip()
            if not api_key:
                raise ValueError("DASHSCOPE_API_KEY 或 QWEN_API_KEY 未设置或为空，请在 .env 中配置后重新运行")
            llm_kwargs["api_key"] = api_key
        elif self.provider == "openrouter":
            llm_kwargs["base_url"] = "https://openrouter.ai/api/v1"
            api_key = (os.environ.get("OPENROUTER_API_KEY") or "").strip()
            if not api_key:
                raise ValueError("OPENROUTER_API_KEY 未设置或为空，请在 .env 中配置后重新运行")
            llm_kwargs["api_key"] = api_key
        elif self.provider == "ollama":
            llm_kwargs["base_url"] = "http://localhost:11434/v1"
            llm_kwargs["api_key"] = "ollama"  # Ollama doesn't require auth
        elif self.provider == "openai":
            api_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
            if not api_key:
                raise ValueError("OPENAI_API_KEY 未设置或为空，请在 .env 中配置后重新运行")
            llm_kwargs["api_key"] = api_key
            if self.base_url:
                llm_kwargs["base_url"] = self.base_url
        elif self.base_url:
            llm_kwargs["base_url"] = self.base_url

        for key in ("timeout", "max_retries", "reasoning_effort", "callbacks"):
            if key in self.kwargs:
                llm_kwargs[key] = self.kwargs[key]
        # 不要用 kwargs 覆盖 api_key，避免空值覆盖已设置好的 key
        if "api_key" in self.kwargs and (self.kwargs.get("api_key") or "").strip():
            llm_kwargs["api_key"] = (self.kwargs["api_key"] or "").strip()

        return UnifiedChatOpenAI(**llm_kwargs)

    def validate_model(self) -> bool:
        """Validate model for the provider."""
        return validate_model(self.provider, self.model)

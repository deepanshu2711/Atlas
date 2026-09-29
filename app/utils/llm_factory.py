import os
from langchain_ollama import ChatOllama

from app.core.config import settings

_DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")


def build_llm(*, num_ctx: int = 8192, num_predict: int = 3072, timeout: int = 180,
              format: str | None = None, model: str | None = None) -> ChatOllama:
    return ChatOllama(
        model=model or _DEFAULT_MODEL,
        temperature=0,
        num_ctx=num_ctx,
        num_predict=num_predict,
        format=format,
        client_kwargs={"timeout": timeout}
    )


def judge_model_name() -> str:
    """Model that grades eval answers (JUDGE_MODEL); the answering model when unset."""
    return settings.judge_model or _DEFAULT_MODEL


def build_judge_llm() -> ChatOllama:
    return build_llm(model=judge_model_name(), num_predict=256)


llm = build_llm()

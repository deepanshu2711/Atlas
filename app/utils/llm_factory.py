import os
from langchain_ollama import ChatOllama

_DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")


def build_llm(*, num_ctx: int = 4096, num_predict: int = 3072, timeout: int = 180) -> ChatOllama:
    return ChatOllama(
        model=_DEFAULT_MODEL,
        temperature=0,
        num_ctx=num_ctx,
        num_predict=num_predict,
        client_kwargs={"timeout": timeout}
    )


llm = build_llm()

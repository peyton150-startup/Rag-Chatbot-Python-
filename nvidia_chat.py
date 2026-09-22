import os

from langchain_openai import ChatOpenAI


NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_CHAT_MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"


def create_nvidia_chat_model(api_key: str | None = None) -> ChatOpenAI:
    resolved_key = api_key or os.environ.get("NVIDIA_API_KEY")
    if not resolved_key:
        raise RuntimeError("Set NVIDIA_API_KEY before using the chat model.")

    return ChatOpenAI(
        model=NVIDIA_CHAT_MODEL,
        base_url=NVIDIA_BASE_URL,
        api_key=resolved_key,
        temperature=0.2,
        max_tokens=1024,
        streaming=False,
        model_kwargs={
            "top_p": 0.95,
            # Thinking mode uses up the token budget and can leak into answers.
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        },
    )

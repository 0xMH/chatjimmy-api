from chatjimmy.client import (
    Attachment,
    ChatJimmy,
    ChatResponse,
    HealthStatus,
    Message,
    Model,
    Stats,
)

__all__ = [
    "Attachment",
    "ChatJimmy",
    "ChatResponse",
    "HealthStatus",
    "Message",
    "Model",
    "Stats",
    "create_server",
]


def __getattr__(name: str):
    if name == "create_server":
        from chatjimmy.server import create_server

        return create_server
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

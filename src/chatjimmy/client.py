"""Python wrapper for the chatjimmy.ai API (Taalas HC1 inference)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Generator

import requests


BASE_URL = "https://chatjimmy.ai"

_STATS_RE = re.compile(r"<\|stats\|>([\s\S]+?)<\|/stats\|>$")


@dataclass
class Stats:
    created_at: float = 0.0
    done: bool = False
    done_reason: str = ""
    total_duration: float = 0.0
    ttft: float = 0.0
    prefill_tokens: int = 0
    prefill_rate: float = 0.0
    decode_tokens: int = 0
    decode_rate: float = 0.0
    total_tokens: int = 0
    total_time: float = 0.0
    roundtrip_time: float = 0.0
    topk: int = 0
    status: int = 0
    reason: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> Stats:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class ChatResponse:
    text: str
    stats: Stats | None = None


@dataclass
class Model:
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = ""


@dataclass
class HealthStatus:
    status: str = ""
    nextjs: str = ""
    backend: str = ""
    backend_status: int = 0
    backend_details: dict = field(default_factory=dict)
    timestamp: str = ""

    @property
    def healthy(self) -> bool:
        return self.status == "ok" and self.backend == "healthy"


@dataclass
class Message:
    role: str
    content: str

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


@dataclass
class Attachment:
    name: str
    size: int
    content: str

    def to_dict(self) -> dict:
        return {"name": self.name, "size": self.size, "content": self.content}


class ChatJimmy:
    """Client for the chatjimmy.ai API."""

    def __init__(self, base_url: str = BASE_URL, timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "chatjimmy-python/1.0", "Content-Type": "application/json"}
        )

    def health(self) -> HealthStatus:
        """Check server health."""
        r = self.session.get(f"{self.base_url}/api/health", timeout=self.timeout)
        r.raise_for_status()
        d = r.json()
        return HealthStatus(
            status=d.get("status", ""),
            nextjs=d.get("nextjs", ""),
            backend=d.get("backend", ""),
            backend_status=d.get("backendStatus", 0),
            backend_details=d.get("backendDetails", {}),
            timestamp=d.get("timestamp", ""),
        )

    def models(self) -> list[Model]:
        """List available models."""
        r = self.session.get(f"{self.base_url}/api/models", timeout=self.timeout)
        r.raise_for_status()
        data = r.json().get("data", [])
        return [
            Model(
                id=m["id"],
                object=m.get("object", "model"),
                created=m.get("created", 0),
                owned_by=m.get("owned_by", ""),
            )
            for m in data
        ]

    def chat(
        self,
        messages: list[dict | Message],
        model: str = "llama3.1-8B",
        system_prompt: str = "",
        top_k: int = 8,
        attachment: Attachment | None = None,
    ) -> ChatResponse:
        """Send a chat request and return the full response."""
        full_text = ""
        stats = None
        for chunk_text, chunk_stats in self._chat_stream(
            messages, model, system_prompt, top_k, attachment
        ):
            full_text += chunk_text
            if chunk_stats is not None:
                stats = chunk_stats
        return ChatResponse(text=full_text, stats=stats)

    def chat_stream(
        self,
        messages: list[dict | Message],
        model: str = "llama3.1-8B",
        system_prompt: str = "",
        top_k: int = 8,
        attachment: Attachment | None = None,
    ) -> Generator[str, None, ChatResponse]:
        """Stream chat token by token. Yields text chunks, returns ChatResponse."""
        full_text = ""
        stats = None
        for chunk_text, chunk_stats in self._chat_stream(
            messages, model, system_prompt, top_k, attachment
        ):
            full_text += chunk_text
            if chunk_stats is not None:
                stats = chunk_stats
            else:
                yield chunk_text
        return ChatResponse(text=full_text, stats=stats)

    def _chat_stream(
        self,
        messages: list[dict | Message],
        model: str,
        system_prompt: str,
        top_k: int,
        attachment: Attachment | None,
    ) -> Generator[tuple[str, Stats | None], None, None]:
        msg_dicts = [
            m.to_dict() if isinstance(m, Message) else m for m in messages
        ]
        body = {
            "messages": msg_dicts,
            "chatOptions": {
                "selectedModel": model,
                "systemPrompt": system_prompt,
                "topK": top_k,
            },
            "attachment": attachment.to_dict() if attachment else None,
        }
        r = self.session.post(
            f"{self.base_url}/api/chat",
            json=body,
            stream=True,
            timeout=self.timeout,
        )
        r.raise_for_status()

        buffer = ""
        for chunk in r.iter_content(chunk_size=256):
            buffer += chunk.decode("utf-8", errors="replace")

        # Separate stats from text
        match = _STATS_RE.search(buffer)
        if match:
            text = buffer[: match.start()]
            try:
                stats = Stats.from_dict(json.loads(match.group(1)))
            except (json.JSONDecodeError, TypeError):
                stats = None
            yield text, stats
        else:
            yield buffer, None

    def ask(
        self,
        prompt: str,
        model: str = "llama3.1-8B",
        system_prompt: str = "",
        top_k: int = 8,
    ) -> str:
        """Simple single-turn question. Returns just the text."""
        resp = self.chat(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            system_prompt=system_prompt,
            top_k=top_k,
        )
        return resp.text

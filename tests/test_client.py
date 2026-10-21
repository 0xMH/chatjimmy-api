"""Unit tests for the chatjimmy client."""

import json
from unittest.mock import MagicMock, patch

import pytest

from chatjimmy import Attachment, ChatJimmy, ChatResponse, HealthStatus, Message, Model, Stats


def test_stats_from_dict():
    raw = {
        "created_at": 1792582990.0,
        "done": True,
        "done_reason": "stop",
        "total_duration": 0.01,
        "ttft": 0.001,
        "prefill_tokens": 10,
        "prefill_rate": 1000.0,
        "decode_tokens": 20,
        "decode_rate": 2000.0,
        "total_tokens": 30,
        "total_time": 0.01,
        "roundtrip_time": 15,
        "topk": 8,
        "status": 0,
        "reason": "eos",
        # Extra field from backend that is not in the dataclass
        "device_compute_time": 0.005,
    }
    stats = Stats.from_dict(raw)
    assert stats.done is True
    assert stats.decode_tokens == 20
    assert stats.decode_rate == 2000.0
    assert stats.roundtrip_time == 15
    assert not hasattr(stats, "device_compute_time")


def test_message_and_attachment_to_dict():
    msg = Message(role="user", content="Hello")
    assert msg.to_dict() == {"role": "user", "content": "Hello"}

    att = Attachment(name="test.txt", size=5, content="hello")
    assert att.to_dict() == {"name": "test.txt", "size": 5, "content": "hello"}


def test_health_success():
    client = ChatJimmy()
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "status": "ok",
        "nextjs": "healthy",
        "backend": "healthy",
        "backendStatus": 200,
        "backendDetails": {"status": "healthy"},
        "timestamp": "2026-10-21T00:00:00Z",
    }
    with patch.object(client.session, "get", return_value=mock_resp) as mock_get:
        health = client.health()
        assert health.healthy is True
        assert health.backend == "healthy"
        mock_get.assert_called_once_with("https://chatjimmy.ai/api/health", timeout=30)


def test_models_success():
    client = ChatJimmy()
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "object": "list",
        "data": [
            {
                "id": "llama3.1-8B",
                "object": "model",
                "created": 1690000000,
                "owned_by": "Taalas Inc.",
            }
        ],
    }
    with patch.object(client.session, "get", return_value=mock_resp):
        models = client.models()
        assert len(models) == 1
        assert models[0].id == "llama3.1-8B"
        assert models[0].owned_by == "Taalas Inc."


def test_chat():
    client = ChatJimmy()
    stats_json = json.dumps({"decode_tokens": 5, "decode_rate": 15000.0, "done": True})
    raw_content = [f"Hello from Jimmy!\n<|stats|>{stats_json}<|/stats|>".encode("utf-8")]

    mock_resp = MagicMock()
    mock_resp.iter_content.return_value = raw_content

    with patch.object(client.session, "post", return_value=mock_resp) as mock_post:
        resp = client.chat([Message(role="user", content="Hi")])
        assert resp.text == "Hello from Jimmy!\n"
        assert resp.stats is not None
        assert resp.stats.decode_tokens == 5
        assert resp.stats.decode_rate == 15000.0
        mock_post.assert_called_once()


def test_chat_stream_yields_chunks_and_returns_response():
    client = ChatJimmy()
    stats_json = json.dumps({"decode_tokens": 10, "decode_rate": 20000.0, "done": True})

    # Simulate chunks arriving across network
    chunk_1 = b"Chunk 1 "
    chunk_2 = b"Chunk 2\n<|st"
    chunk_3 = f"ats|>{stats_json}<|/stats|>".encode("utf-8")

    mock_resp = MagicMock()
    mock_resp.iter_content.return_value = [chunk_1, chunk_2, chunk_3]

    with patch.object(client.session, "post", return_value=mock_resp):
        gen = client.chat_stream([{"role": "user", "content": "Hi"}])
        chunks = []
        try:
            while True:
                chunks.append(next(gen))
        except StopIteration as e:
            return_val = e.value

        assert "".join(chunks) == "Chunk 1 Chunk 2\n"
        assert len(chunks) >= 2
        assert isinstance(return_val, ChatResponse)
        assert return_val.text == "Chunk 1 Chunk 2\n"
        assert return_val.stats is not None
        assert return_val.stats.decode_tokens == 10


def test_ask():
    client = ChatJimmy()
    stats_json = json.dumps({"decode_tokens": 1, "done": True})
    raw_content = [f"42<|stats|>{stats_json}<|/stats|>".encode("utf-8")]

    mock_resp = MagicMock()
    mock_resp.iter_content.return_value = raw_content

    with patch.object(client.session, "post", return_value=mock_resp):
        ans = client.ask("What is the meaning of life?")
        assert ans == "42"


def test_live_backend_e2e():
    """Live end-to-end integration test against the real chatjimmy.ai backend."""
    client = ChatJimmy()

    # Health
    health = client.health()
    assert health.healthy is True

    # Models
    models = client.models()
    assert any(m.id == "llama3.1-8B" for m in models)

    # Ask
    ans = client.ask("Say 12345")
    assert "12345" in ans

    # Streaming
    streamed = list(client.chat_stream([{"role": "user", "content": "Say hello"}]))
    assert len(streamed) > 0
    assert len("".join(streamed).strip()) > 0

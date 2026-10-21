"""Tests for the OpenAI-compatible proxy server."""

import json
import threading
import time
from unittest.mock import MagicMock, patch

import requests

from chatjimmy.client import ChatResponse, Stats
from chatjimmy.server import create_server, extract_content, parse_messages


def test_extract_content():
    assert extract_content("simple string") == "simple string"
    assert (
        extract_content([{"type": "text", "text": "hello "}, {"type": "text", "text": "world"}])
        == "hello world"
    )
    assert extract_content(None) == ""


def test_parse_messages():
    payload = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
        {"role": "user", "content": "How are you?"},
    ]
    sys_prompt, msgs = parse_messages(payload)
    assert sys_prompt == "You are helpful."
    assert len(msgs) == 3
    assert msgs[0] == {"role": "user", "content": "Hello"}


def test_server_models_and_chat():
    server = create_server("127.0.0.1", 8999)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.2)

    try:
        # 1. Models endpoint
        r = requests.get("http://127.0.0.1:8999/v1/models")
        assert r.status_code == 200
        data = r.json()
        assert data["object"] == "list"
        assert any(m["id"] == "llama3.1-8B" for m in data["data"])

        # 2. Chat completions non-streaming
        mock_response = ChatResponse(
            text="Mocked answer",
            stats=Stats(decode_tokens=5, prefill_tokens=10, total_tokens=15),
        )
        with patch.object(server.RequestHandlerClass.client, "chat", return_value=mock_response):
            r = requests.post(
                "http://127.0.0.1:8999/v1/chat/completions",
                json={
                    "model": "llama3.1-8B",
                    "messages": [{"role": "user", "content": "test"}],
                    "stream": False,
                },
            )
            assert r.status_code == 200
            res = r.json()
            assert res["choices"][0]["message"]["content"] == "Mocked answer"
            assert res["usage"]["total_tokens"] == 15

        # 3. Chat completions streaming
        with patch.object(
            server.RequestHandlerClass.client,
            "chat_stream",
            return_value=iter(["chunk1 ", "chunk2"]),
        ):
            r = requests.post(
                "http://127.0.0.1:8999/v1/chat/completions",
                json={
                    "model": "llama3.1-8B",
                    "messages": [{"role": "user", "content": "test"}],
                    "stream": True,
                },
                stream=True,
            )
            assert r.status_code == 200
            lines = [l.decode("utf-8") for l in r.iter_lines() if l]
            assert any("[DONE]" in l for l in lines)
            assert any("chunk1 " in l for l in lines)

    finally:
        server.shutdown()
        server.server_close()

"""OpenAI-compatible proxy server for ChatJimmy.

Allows using chatjimmy.ai with OpenCode, Claude Code, Aider, Continue,
and any other OpenAI-compatible client.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from chatjimmy.client import ChatJimmy, Message

logger = logging.getLogger("chatjimmy.server")


def extract_content(content: Any) -> str:
    """Extract plain text string from OpenAI message content."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                texts.append(part.get("text", ""))
            elif isinstance(part, str):
                texts.append(part)
        return "".join(texts)
    return str(content or "")


def parse_messages(messages_payload: list[dict[str, Any]]) -> tuple[str, list[dict[str, str]]]:
    """Separate system prompts from chat messages."""
    system_parts: list[str] = []
    chat_messages: list[dict[str, str]] = []

    for msg in messages_payload:
        role = msg.get("role", "user")
        content = extract_content(msg.get("content", ""))

        if role in ("system", "developer"):
            if content:
                system_parts.append(content)
        else:
            chat_messages.append({"role": role, "content": content})

    system_prompt = "\n\n".join(system_parts)
    return system_prompt, chat_messages


class OpenAIProxyHandler(BaseHTTPRequestHandler):
    client = ChatJimmy()

    def _send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header(
            "Access-Control-Allow-Headers",
            "Authorization, Content-Type, Accept, X-Requested-With",
        )

    def do_OPTIONS(self):
        self.send_response(204)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self):
        path = self.path.split("?")[0].rstrip("/")

        if path in ("", "/health", "/api/health"):
            try:
                h = self.client.health()
                data = {
                    "status": h.status,
                    "healthy": h.healthy,
                    "backend": h.backend,
                    "timestamp": h.timestamp,
                }
                status_code = 200
            except Exception as e:
                data = {"status": "error", "error": str(e)}
                status_code = 503

            self._send_json(data, status_code)
            return

        if path in ("/models", "/v1/models"):
            try:
                models = self.client.models()
                data = {
                    "object": "list",
                    "data": [
                        {
                            "id": m.id,
                            "object": "model",
                            "created": m.created,
                            "owned_by": m.owned_by or "Taalas Inc.",
                        }
                        for m in models
                    ],
                }
                status_code = 200
            except Exception:
                data = {
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
                status_code = 200

            self._send_json(data, status_code)
            return

        if path.startswith("/models/") or path.startswith("/v1/models/"):
            model_id = path.split("/")[-1]
            data = {
                "id": model_id,
                "object": "model",
                "created": 1690000000,
                "owned_by": "Taalas Inc.",
            }
            self._send_json(data, 200)
            return

        self._send_json({"error": {"message": f"Not found: {self.path}", "code": 404}}, 404)

    def do_POST(self):
        path = self.path.split("?")[0].rstrip("/")

        if path in ("/chat/completions", "/v1/chat/completions"):
            self.handle_chat_completions()
            return

        self._send_json({"error": {"message": f"Not found: {self.path}", "code": 404}}, 404)

    def handle_chat_completions(self):
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            self._send_json({"error": {"message": "Empty request body", "code": 400}}, 400)
            return

        raw_body = self.rfile.read(content_length)
        try:
            body = json.loads(raw_body.decode("utf-8"))
        except Exception as e:
            self._send_json({"error": {"message": f"Invalid JSON: {e}", "code": 400}}, 400)
            return

        messages_payload = body.get("messages", [])
        if not messages_payload:
            self._send_json({"error": {"message": "Missing 'messages'", "code": 400}}, 400)
            return

        model = body.get("model", "llama3.1-8B")
        stream = bool(body.get("stream", False))
        top_k = int(body.get("top_k", 8))

        system_prompt, chat_messages = parse_messages(messages_payload)

        # Fallback if no chat messages left (e.g. only system message provided)
        if not chat_messages:
            chat_messages = [{"role": "user", "content": "Hello"}]

        created_ts = int(time.time())
        chat_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

        if stream:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self._send_cors_headers()
            self.end_headers()

            # Initial role chunk
            role_chunk = {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": created_ts,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": ""},
                        "finish_reason": None,
                    }
                ],
            }
            self.wfile.write(f"data: {json.dumps(role_chunk)}\n\n".encode("utf-8"))
            self.wfile.flush()

            try:
                for chunk in self.client.chat_stream(
                    messages=chat_messages,
                    model=model,
                    system_prompt=system_prompt,
                    top_k=top_k,
                ):
                    delta_chunk = {
                        "id": chat_id,
                        "object": "chat.completion.chunk",
                        "created": created_ts,
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": chunk},
                                "finish_reason": None,
                            }
                        ],
                    }
                    self.wfile.write(f"data: {json.dumps(delta_chunk)}\n\n".encode("utf-8"))
                    self.wfile.flush()

                stop_chunk = {
                    "id": chat_id,
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop",
                        }
                    ],
                }
                self.wfile.write(f"data: {json.dumps(stop_chunk)}\n\n".encode("utf-8"))
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                self.close_connection = True
            except (BrokenPipeError, ConnectionResetError):
                logger.info("Client disconnected during stream")
                self.close_connection = True
            except Exception as e:
                logger.exception("Error during streaming")
                error_chunk = {"error": {"message": str(e), "code": 500}}
                try:
                    self.wfile.write(f"data: {json.dumps(error_chunk)}\n\n".encode("utf-8"))
                    self.wfile.flush()
                except Exception:
                    pass
                self.close_connection = True
        else:
            try:
                resp = self.client.chat(
                    messages=chat_messages,
                    model=model,
                    system_prompt=system_prompt,
                    top_k=top_k,
                )
                output = {
                    "id": chat_id,
                    "object": "chat.completion",
                    "created": created_ts,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": resp.text,
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": resp.stats.prefill_tokens if resp.stats else 0,
                        "completion_tokens": resp.stats.decode_tokens if resp.stats else 0,
                        "total_tokens": resp.stats.total_tokens if resp.stats else 0,
                    },
                }
                self._send_json(output, 200)
            except Exception as e:
                logger.exception("Error during non-streaming chat")
                self._send_json({"error": {"message": str(e), "code": 500}}, 500)

    def _send_json(self, data: dict, status_code: int = 200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._send_cors_headers()
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any):
        logger.info("%s - - [%s] %s", self.address_string(), self.log_date_time_string(), format % args)


def create_server(host: str = "127.0.0.1", port: int = 8383) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), OpenAIProxyHandler)
    server.daemon_threads = True
    return server


def main():
    parser = argparse.ArgumentParser(
        description="Run an OpenAI-compatible proxy server for ChatJimmy (chatjimmy.ai)."
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host to listen on (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8383, help="Port to listen on (default: 8383)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    server = create_server(args.host, args.port)
    print(f"ChatJimmy OpenAI proxy running on http://{args.host}:{args.port}")
    print(f"Base URL for OpenCode / clients: http://{args.host}:{args.port}/v1")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.shutdown()


if __name__ == "__main__":
    main()

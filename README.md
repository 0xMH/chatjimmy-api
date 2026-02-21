# chatjimmy

Unofficial Python wrapper for the chatjimmy.ai API.

chatjimmy.ai is a demo chatbot by Taalas, running Llama 3.1 8B on their custom HC1 silicon at ~17,000 tokens/sec per user.

https://chatjimmy.ai

## Quick Start

```python
from chatjimmy import ChatJimmy

client = ChatJimmy()

answer = client.ask("What is the capital of France?")
print(answer)
```

## Usage

### Simple question

```python
from chatjimmy import ChatJimmy

client = ChatJimmy()
print(client.ask("Explain quantum computing in one sentence."))
```

### Chat with options

```python
from chatjimmy import ChatJimmy

client = ChatJimmy()

response = client.chat(
    messages=[{"role": "user", "content": "Explain recursion"}],
    system_prompt="You are a computer science tutor.",
    top_k=4,
)

print(response.text)
print(f"Output tokens: {response.stats.decode_tokens}")
print(f"Speed: {response.stats.decode_rate:.0f} tokens/sec")
```

### Multi-turn conversation

```python
from chatjimmy import ChatJimmy

client = ChatJimmy()

messages = [
    {"role": "user", "content": "My name is Mohamed."},
]
resp = client.chat(messages)
print(resp.text)

messages.append({"role": "assistant", "content": resp.text})
messages.append({"role": "user", "content": "What's my name?"})
resp = client.chat(messages)
print(resp.text)
```

### Streaming

```python
from chatjimmy import ChatJimmy

client = ChatJimmy()

for chunk in client.chat_stream(
    messages=[{"role": "user", "content": "Write a haiku about code"}]
):
    print(chunk, end="", flush=True)
print()
```

### Health check

```python
from chatjimmy import ChatJimmy

client = ChatJimmy()
health = client.health()

print(health.healthy)       # True/False
print(health.backend)       # "healthy"
print(health.timestamp)     # ISO timestamp
```

### List models

```python
from chatjimmy import ChatJimmy

client = ChatJimmy()

for model in client.models():
    print(f"{model.id} (by {model.owned_by})")
```

### Using Message objects

```python
from chatjimmy import ChatJimmy, Message

client = ChatJimmy()

response = client.chat(
    messages=[Message(role="user", content="Hello!")],
    system_prompt="Reply in French.",
)
print(response.text)
```

### Attachments

```python
from chatjimmy import ChatJimmy, Attachment

client = ChatJimmy()

attachment = Attachment(name="data.txt", size=11, content="hello world")
response = client.chat(
    messages=[{"role": "user", "content": "Summarize this file"}],
    attachment=attachment,
)
print(response.text)
```

### Response stats

Every chat response includes inference stats from the Taalas HC1 hardware:

```python
response = client.chat(messages=[{"role": "user", "content": "hi"}])
stats = response.stats

stats.prefill_tokens    # input tokens processed
stats.prefill_rate      # input processing speed (tokens/sec)
stats.decode_tokens     # output tokens generated
stats.decode_rate       # output generation speed (tokens/sec)
stats.total_tokens      # prefill + decode
stats.ttft              # time to first token (seconds)
stats.total_time        # total inference time (seconds)
stats.roundtrip_time    # network round trip (ms)
stats.done_reason       # "stop" (natural end)
```

## API Reference

### ChatJimmy(base_url, timeout)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| base_url | str | `https://chatjimmy.ai` | API base URL |
| timeout | int | 30 | Request timeout in seconds |

### client.ask(prompt, model, system_prompt, top_k)

Single-turn convenience method. Returns the response text as a string.

### client.chat(messages, model, system_prompt, top_k, attachment)

Full chat method. Returns a `ChatResponse` with `.text` and `.stats`.

### client.chat_stream(messages, model, system_prompt, top_k, attachment)

Generator that yields text chunks as they arrive.

### client.health()

Returns a `HealthStatus` object with a `.healthy` property.

### client.models()

Returns a list of `Model` objects.

### Chat parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| messages | list | required | List of `{"role": ..., "content": ...}` dicts or `Message` objects |
| model | str | `llama3.1-8B` | Model ID |
| system_prompt | str | `""` | System prompt |
| top_k | int | 8 | Top-K sampling parameter |
| attachment | Attachment | None | File attachment |

## Notes

- No authentication required
- No rate limiting observed (tested 20 concurrent + 30 sequential bursts)
- Single model available: llama3.1-8B on Taalas HC1 silicon

## Known Limits

- Input: ~6,064 prefill tokens. Requests exceeding this return an empty 200 response with no error
- Output: no hard cap. Model stops naturally via EOS token (~1,200-2,400 tokens typical)

## How We Know It's Taalas

The connection to Taalas was found in two places inside chatjimmy.ai itself:

1. The main JS bundle (`8642-*.js`) contains footer links to `https://taalas.com/terms-conditions` and `https://taalas.com/privacy-policy` in the chat disclaimer text.
2. The `/api/models` endpoint returns `"owned_by": "Taalas Inc."` in the model metadata.

No other references to Taalas appear anywhere in the HTML or JS bundles.

## Disclaimer

This is an unofficial wrapper. chatjimmy.ai is a public demo by Taalas (https://taalas.com). The API has no authentication and could change or go offline at any time.

"""Quick smoke test for the LLM backend.

Usage:
    python -m agents.test_backend ollama
    python -m agents.test_backend openai
"""
import json
import sys
from agents.llm_backend import build_backend


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else None
    backend = build_backend(name)
    print(f"Backend: {backend.name} | model: {backend.model}")
    system_msg = ("You are a JSON-only test responder. "
                  "Reply with {\"hello\":\"<your model name>\",\"ok\":true}.")
    user_msg = "ping"
    raw = backend.chat(system_msg, user_msg, json_mode=True, max_tokens=80, temperature=0)
    print("Raw:", raw)
    parsed = backend.chat_json(system_msg, user_msg, max_tokens=80, temperature=0)
    print("Parsed:", json.dumps(parsed, indent=2))


if __name__ == "__main__":
    main()

"""Unified LLM backend: Ollama (local Gemma 3) or OpenAI.

The contract is one method:
    backend.chat(system: str, user: str, json_mode: bool = True) -> str

All call-sites parse the returned string as JSON when json_mode=True.
"""
from __future__ import annotations
import json
import os
import time
from abc import ABC, abstractmethod
from typing import Optional

try:
    import requests  # bundled with Anaconda
except ImportError:
    requests = None


def _load_env_file():
    """Load .env from the project root if present (no python-dotenv dep)."""
    here = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(here))
    env_path = os.path.join(project_root, ".env")
    if not os.path.exists(env_path):
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_env_file()


class LLMBackend(ABC):
    name: str = "abstract"
    model: str = "abstract"

    @abstractmethod
    def chat(self, system: str, user: str, json_mode: bool = True,
             max_tokens: int = 1024, temperature: float = 0.7) -> str:
        ...

    def chat_json(self, system: str, user: str, **kw) -> dict:
        raw = self.chat(system, user, json_mode=True, **kw)
        return _parse_json_loose(raw)


def _parse_json_loose(text: str) -> dict:
    """Robust JSON extraction: tolerate code fences and prefix/suffix prose."""
    if not text:
        return {}
    s = text.strip()
    # Strip markdown fences
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s[3:]
        if s.endswith("```"):
            s = s[:-3]
        s = s.strip()
        if s.lower().startswith("json"):
            s = s[4:].strip()
    # Try direct parse first
    try:
        return json.loads(s)
    except Exception:
        pass
    # Fallback: find first { ... last }
    i, j = s.find("{"), s.rfind("}")
    if i != -1 and j != -1 and j > i:
        try:
            return json.loads(s[i:j + 1])
        except Exception:
            pass
    return {"_parse_error": True, "_raw": text}


class OllamaBackend(LLMBackend):
    name = "ollama"

    def __init__(self, model: Optional[str] = None, host: Optional[str] = None,
                 timeout: float = 120.0):
        if requests is None:
            raise RuntimeError("`requests` package not available")
        self.model = model or os.environ.get("OLLAMA_MODEL", "gemma3:12b")
        self.host = host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self.timeout = timeout

    def chat(self, system: str, user: str, json_mode: bool = True,
             max_tokens: int = 1024, temperature: float = 0.7) -> str:
        url = f"{self.host.rstrip('/')}/api/chat"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if json_mode:
            payload["format"] = "json"
        for attempt in range(3):
            try:
                r = requests.post(url, json=payload, timeout=self.timeout)
                r.raise_for_status()
                data = r.json()
                return (data.get("message") or {}).get("content", "")
            except Exception as e:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)
        return ""


class OpenAIBackend(LLMBackend):
    name = "openai"

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None,
                 timeout: float = 60.0):
        try:
            from openai import OpenAI
        except ImportError:
            raise RuntimeError("Install: pip install openai")
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY not set in env or .env file")
        self._client = OpenAI(api_key=key, timeout=timeout)
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    def chat(self, system: str, user: str, json_mode: bool = True,
             max_tokens: int = 1024, temperature: float = 0.7) -> str:
        kw = dict(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if json_mode:
            kw["response_format"] = {"type": "json_object"}
        for attempt in range(3):
            try:
                resp = self._client.chat.completions.create(**kw)
                return resp.choices[0].message.content or ""
            except Exception as e:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)
        return ""


def build_backend(name: Optional[str] = None) -> LLMBackend:
    """Construct a backend from explicit arg or LLM_BACKEND env var."""
    name = (name or os.environ.get("LLM_BACKEND", "ollama")).lower()
    if name == "ollama":
        return OllamaBackend()
    if name == "openai":
        return OpenAIBackend()
    raise ValueError(f"Unknown backend: {name!r} (expected 'ollama' or 'openai')")

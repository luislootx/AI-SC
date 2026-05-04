"""LLM-backed agents: planner (architecture proposer) and reviewer (peer reviewer).

Usage:
    from agents import build_backend, LLMPlanner, LLMReviewer
    backend = build_backend()  # reads .env / env vars
    planner = LLMPlanner(backend)
    reviewer = LLMReviewer(backend)
"""
from .llm_backend import LLMBackend, OllamaBackend, OpenAIBackend, build_backend
from .llm_planner import LLMPlanner
from .llm_reviewer import LLMReviewer

__all__ = [
    "LLMBackend", "OllamaBackend", "OpenAIBackend", "build_backend",
    "LLMPlanner", "LLMReviewer",
]

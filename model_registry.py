"""
AtomInfer Model Registry
========================
Manages multiple LLM providers (Ollama, Groq, OpenAI, Anthropic, vLLM,
LM Studio) with intelligent per-task model selection and fallback.

Usage:
    from model_registry import registry

    client, model_name = registry.get_client_for_task("xrd_analysis")
    response = client.chat_completion(messages, tools=TOOLS)
"""

import os
import json
import requests
from enum import Enum
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

from config_loader import cfg, ModelProfile


# ═══════════════════════════════════════════════════════════════════════════════
# Task types
# ═══════════════════════════════════════════════════════════════════════════════

class TaskType(str, Enum):
    XRD_ANALYSIS       = "xrd_analysis"
    RAMAN_ANALYSIS     = "raman_analysis"
    STRUCTURE_BUILDING = "structure_building"
    GENERAL_REASONING  = "general_reasoning"
    PEAK_ASSIGNMENT    = "peak_assignment"


# ═══════════════════════════════════════════════════════════════════════════════
# Unified LLM Client interface
# ═══════════════════════════════════════════════════════════════════════════════

class LLMClient:
    """Unified interface wrapping different LLM providers."""

    def __init__(self, profile: ModelProfile, temperature: float = None,
                 max_tokens: int = None):
        self.profile = profile
        self.provider = profile.provider
        self.model = profile.model
        self.temperature = temperature or cfg.llm.temperature
        self.max_tokens = max_tokens or cfg.llm.max_tokens
        self._client = None
        self._setup()

    def _setup(self):
        if self.provider in ("ollama", "lmstudio", "vllm"):
            from openai import OpenAI
            endpoint = self.profile.endpoint or "http://localhost:11434"
            if self.provider == "ollama" and not endpoint.endswith("/v1"):
                endpoint = endpoint.rstrip("/") + "/v1"
            api_key = self.profile.get_api_key() or "not-needed"
            self._client = OpenAI(base_url=endpoint, api_key=api_key)

        elif self.provider == "groq":
            from groq import Groq
            api_key = self.profile.get_api_key()
            if not api_key:
                raise ValueError(f"API key not found for Groq (env: {self.profile.api_key_env})")
            self._client = Groq(api_key=api_key)

        elif self.provider == "openai":
            from openai import OpenAI
            api_key = self.profile.get_api_key()
            if not api_key:
                raise ValueError(f"API key not found for OpenAI (env: {self.profile.api_key_env})")
            self._client = OpenAI(api_key=api_key)

        elif self.provider == "anthropic":
            # Anthropic uses its own SDK but we wrap it in OpenAI-compatible interface
            # via the openai library pointing to Anthropic's endpoint
            api_key = self.profile.get_api_key()
            if not api_key:
                raise ValueError(f"API key not found for Anthropic (env: {self.profile.api_key_env})")
            from anthropic import Anthropic
            self._client = Anthropic(api_key=api_key)

        else:
            raise ValueError(f"Unknown LLM provider: {self.provider}")

    def chat_completion(self, messages: list, tools: list = None,
                        tool_choice: str = "auto",
                        temperature: float = None,
                        max_tokens: int = None) -> Any:
        """Send a chat completion request. Returns the raw response object."""
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens or self.max_tokens

        if self.provider == "anthropic":
            return self._anthropic_completion(messages, tools, tool_choice, temp, tokens)

        # OpenAI-compatible API (Groq, Ollama, OpenAI, vLLM, LM Studio)
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temp,
            "max_tokens": tokens,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice

        return self._client.chat.completions.create(**kwargs)

    def _anthropic_completion(self, messages, tools, tool_choice, temperature, max_tokens):
        """Handle Anthropic's different API format."""
        # Extract system message
        system_msg = ""
        user_messages = []
        for m in messages:
            if m["role"] == "system":
                system_msg = m["content"]
            else:
                user_messages.append(m)

        # Convert OpenAI tool format to Anthropic format
        anthropic_tools = []
        if tools:
            for t in tools:
                func = t.get("function", {})
                anthropic_tools.append({
                    "name": func.get("name", ""),
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {}),
                })

        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": user_messages,
        }
        if system_msg:
            kwargs["system"] = system_msg
        if anthropic_tools:
            kwargs["tools"] = anthropic_tools

        response = self._client.messages.create(**kwargs)
        return _AnthropicResponseAdapter(response)


class _AnthropicResponseAdapter:
    """Adapts Anthropic response to look like OpenAI response for the agent loop."""

    def __init__(self, response):
        self._response = response
        self.choices = [_AnthropicChoice(response)]


class _AnthropicChoice:
    def __init__(self, response):
        self.message = _AnthropicMessage(response)


class _AnthropicMessage:
    def __init__(self, response):
        self.content = ""
        self.tool_calls = []

        for block in response.content:
            if block.type == "text":
                self.content += block.text
            elif block.type == "tool_use":
                self.tool_calls.append(_AnthropicToolCall(block))

        if not self.tool_calls:
            self.tool_calls = None


class _AnthropicToolCall:
    def __init__(self, block):
        self.id = block.id
        self.type = "function"
        self.function = _AnthropicFunction(block)


class _AnthropicFunction:
    def __init__(self, block):
        self.name = block.name
        self.arguments = json.dumps(block.input)


# ═══════════════════════════════════════════════════════════════════════════════
# Model availability detection
# ═══════════════════════════════════════════════════════════════════════════════

def check_ollama_available(endpoint: str = "http://localhost:11434") -> List[str]:
    """Check which models are available on a local Ollama instance."""
    try:
        resp = requests.get(f"{endpoint}/api/tags", timeout=3)
        if resp.status_code == 200:
            data = resp.json()
            return [m["name"] for m in data.get("models", [])]
    except (requests.ConnectionError, requests.Timeout):
        pass
    return []


def check_openai_compatible_available(endpoint: str) -> bool:
    """Check if an OpenAI-compatible endpoint is reachable."""
    try:
        resp = requests.get(f"{endpoint}/v1/models", timeout=3)
        return resp.status_code == 200
    except (requests.ConnectionError, requests.Timeout):
        return False


def check_api_key_available(env_var: str) -> bool:
    """Check if an API key environment variable is set."""
    return bool(os.environ.get(env_var))


# ═══════════════════════════════════════════════════════════════════════════════
# Model Registry
# ═══════════════════════════════════════════════════════════════════════════════

class ModelRegistry:
    """Manages model profiles and selects the best available model per task."""

    def __init__(self):
        self._availability_cache: Dict[str, bool] = {}
        self._ollama_models: Optional[List[str]] = None

    def refresh_availability(self):
        """Re-probe all configured models for availability."""
        self._availability_cache.clear()
        self._ollama_models = None

        for name, profile in cfg.llm.models.items():
            self._availability_cache[name] = self._check_profile(profile)

    def _check_profile(self, profile: ModelProfile) -> bool:
        """Check if a model profile is currently available."""
        if profile.provider == "ollama":
            if self._ollama_models is None:
                self._ollama_models = check_ollama_available(
                    profile.endpoint or "http://localhost:11434"
                )
            # Check if the specific model is pulled
            return any(
                profile.model in m or m.startswith(profile.model.split(":")[0])
                for m in self._ollama_models
            )

        elif profile.provider in ("lmstudio", "vllm"):
            endpoint = profile.endpoint or "http://localhost:1234"
            return check_openai_compatible_available(endpoint)

        elif profile.provider in ("groq", "openai", "anthropic"):
            if profile.api_key_env:
                return check_api_key_available(profile.api_key_env)
            return profile.get_api_key() is not None

        return False

    def is_available(self, profile_name: str) -> bool:
        """Check if a named model profile is available (uses cache)."""
        if profile_name not in self._availability_cache:
            profile = cfg.llm.models.get(profile_name)
            if not profile:
                return False
            self._availability_cache[profile_name] = self._check_profile(profile)
        return self._availability_cache[profile_name]

    def get_client_for_task(self, task: str) -> tuple:
        """Get (LLMClient, model_name) for a task type.

        Tries models in the configured priority order for that task,
        falling back through the list. If no task-specific assignment
        exists, tries all models in definition order.

        Returns:
            (LLMClient, model_name_str) or raises RuntimeError if none available.
        """
        # Get priority list for this task
        priority = cfg.llm.task_assignments.get(task)
        if not priority:
            # Fallback: try all models in definition order
            priority = list(cfg.llm.models.keys())

        errors = []
        for profile_name in priority:
            profile = cfg.llm.models.get(profile_name)
            if not profile:
                continue

            if not self.is_available(profile_name):
                continue

            try:
                client = LLMClient(profile)
                print(f"[LLM] Task '{task}' → {profile.provider}:{profile.model} ({profile_name})")
                return client, profile.model
            except Exception as e:
                errors.append(f"{profile_name}: {e}")
                continue

        # Nothing available — provide helpful error
        configured = list(cfg.llm.models.keys())
        msg = (
            f"No LLM model available for task '{task}'.\n"
            f"Configured models: {configured}\n"
            f"Task priority: {priority}\n"
        )
        if errors:
            msg += f"Errors: {errors}\n"
        msg += (
            "\nTo fix this, either:\n"
            "  1. Start Ollama and pull a model: ollama pull llama3.3:70b\n"
            "  2. Set a cloud API key: export GROQ_API_KEY=your_key\n"
            "  3. Edit config.toml [llm.models] and [llm.task_assignments]\n"
        )
        raise RuntimeError(msg)

    def get_available_models(self) -> Dict[str, dict]:
        """Return all models with their availability status."""
        result = {}
        for name, profile in cfg.llm.models.items():
            result[name] = {
                "provider": profile.provider,
                "model": profile.model,
                "available": self.is_available(name),
                "endpoint": profile.endpoint or "(cloud)",
            }
        return result

    def get_status_summary(self) -> dict:
        """Return a summary for the /health endpoint."""
        models = self.get_available_models()
        n_available = sum(1 for m in models.values() if m["available"])
        return {
            "models": models,
            "n_configured": len(models),
            "n_available": n_available,
            "status": "ok" if n_available > 0 else "no_models",
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Module-level singleton
# ═══════════════════════════════════════════════════════════════════════════════

registry = ModelRegistry()

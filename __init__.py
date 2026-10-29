"""Hermes memory provider backed by mcp-memory-service."""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from agent.memory_provider import MemoryProvider, spawn_context_thread

try:
    import httpx
except ImportError as exc:
    httpx = None
    _HTTPX_IMPORT_ERROR: Exception | None = exc
else:
    _HTTPX_IMPORT_ERROR = None


logger = logging.getLogger(__name__)

_REQUEST_TIMEOUT = 3.0
_PREFETCH_RESULTS = 5
_MAX_PREFETCH_CHARS = 300
_SESSION_MESSAGE_LIMIT = 20
_COMPRESS_MESSAGE_LIMIT = 30
_MESSAGE_CHAR_LIMIT = 2_000
_SESSION_CHAR_LIMIT = 12_000
_COMPRESS_CHAR_LIMIT = 16_000
_RELATIVE_SCORE_FLOOR = 0.95
_TRIVIAL_TOKENS = {
    "测试",
    "看看",
    "好了",
    "一下",
    "你好",
    "嗯",
    "哦",
    "收到",
    "继续",
    "好的",
    "行",
    "对",
    "是的",
    "ok",
    "test",
    "hi",
    "hello",
    "thanks",
    "???",
    "？",
    "restart",
}


def _is_meaningful_query(q: str) -> bool:
    text = str(q or "").strip()
    if len(text) < 6:
        return False
    remainder = text.lower()
    for token in _TRIVIAL_TOKENS:
        remainder = remainder.replace(token, "")
    return len("".join(remainder.split())) >= 4


def _candidate_config_paths(
    hermes_home: str | os.PathLike[str] | None = None,
) -> list[Path]:
    candidates: list[Path] = []
    if hermes_home:
        candidates.append(
            Path(hermes_home).expanduser()
            / "plugins"
            / "mcp-memory"
            / "config.json"
        )
    for env in ("HERMES_HOME", "HOME"):
        root = os.environ.get(env)
        if root:
            candidates.append(
                Path(root).expanduser()
                / ".hermes"
                / "plugins"
                / "mcp-memory"
                / "config.json"
            )
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return unique


class McpMemoryProvider(MemoryProvider):
    """Memory provider for a self-hosted mcp-memory-service instance."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending_prefetch: dict[str, str] = {}
        self._prefetch_generation: dict[str, int] = {}
        self._hermes_home: str | os.PathLike[str] | None = None
        self._initialized = False
        self._session_id = ""
        self._platform = ""
        self._should_write = True

    @property
    def name(self) -> str:
        return "mcp-memory"

    def is_available(self) -> bool:
        if httpx is None:
            return False
        return bool(self._resolve_base_url() and self._resolve_token())

    def unavailable_reason(self) -> str:
        if httpx is None:
            detail = f": {_HTTPX_IMPORT_ERROR}" if _HTTPX_IMPORT_ERROR else ""
            return f"mcp-memory requires the httpx package{detail}"
        if not self._resolve_base_url():
            candidates = ", ".join(str(path) for path in self._config_candidates())
            return (
                "mcp-memory is not configured: base_url is missing. Set "
                "MCP_MEMORY_URL or add a 'base_url' field to one of: "
                f"{candidates}"
            )
        if not self._resolve_token():
            candidates = ", ".join(str(path) for path in self._config_candidates())
            return (
                "mcp-memory is not configured: set MCP_MEMORY_TOKEN or add a "
                f"'token' field to one of: {candidates}"
            )
        return ""

    def initialize(self, session_id: str, **kwargs: Any) -> None:
        self._session_id = str(session_id or "")
        self._hermes_home = kwargs.get("hermes_home")
        self._platform = str(kwargs.get("platform") or "")
        context = str(kwargs.get("agent_context") or "primary").strip().lower()
        self._should_write = context == "primary"
        self._initialized = True

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages: list[dict[str, Any]] | None = None,
        turn_author: dict[str, Any] | None = None,
    ) -> None:
        pass

    def system_prompt_block(self) -> str:
        return (
            "## 长期记忆\n"
            "你可以使用 mcp-memory 长期记忆库。涉及此前的决策、用户偏好、项目背景或已排查过的问题时，"
            "先调用 memory_search 再回答；确认值得跨会话保留的事实、决策或结论时，调用 memory_store 保存。"
            "召回内容仅作参考，若与当前对话冲突，以当前对话为准。"
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        try:
            query_text = str(query or "").strip()
            if not _is_meaningful_query(query_text):
                return ""
            key = session_id or self._session_id
            with self._lock:
                cached = self._pending_prefetch.pop(key, None)
            if cached is not None:
                return cached
            return self._format_results(self._search(query_text, _PREFETCH_RESULTS))
        except Exception:
            logger.warning("mcp-memory prefetch failed", exc_info=True)
            return ""

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        query_text = str(query or "").strip()
        if not query_text:
            return

        key = session_id or self._session_id
        with self._lock:
            generation = self._prefetch_generation.get(key, 0) + 1
            self._prefetch_generation[key] = generation

        def worker() -> None:
            try:
                text = self._format_results(self._search(query_text, _PREFETCH_RESULTS))
            except Exception:
                logger.warning("mcp-memory background prefetch failed", exc_info=True)
                text = ""
            with self._lock:
                if self._prefetch_generation.get(key) != generation:
                    return
                self._pending_prefetch[key] = text

        try:
            thread = spawn_context_thread(worker, name="mcp-memory-prefetch")
            thread.start()
        except Exception:
            logger.warning("mcp-memory could not start prefetch thread", exc_info=True)

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "memory_search",
                "description": "Search the user's long-term memory for relevant past context.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural-language search query.",
                        },
                        "n_results": {
                            "type": "integer",
                            "description": "Maximum number of memories to return.",
                            "default": 5,
                            "minimum": 1,
                            "maximum": 20,
                        },
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "memory_store",
                "description": "Store a durable fact, preference, decision, or conclusion in long-term memory.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "Self-contained memory content.",
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional tags for later filtering.",
                        },
                        "memory_type": {
                            "type": "string",
                            "description": "Memory category, defaulting to note.",
                            "default": "note",
                        },
                    },
                    "required": ["content"],
                    "additionalProperties": False,
                },
            },
        ]

    def handle_tool_call(self, tool_name: str, args: dict[str, Any], **kwargs: Any) -> str:
        try:
            if tool_name == "memory_search":
                query = str((args or {}).get("query") or "").strip()
                if not query:
                    return self._tool_error("query is required")
                limit = self._bounded_int((args or {}).get("n_results"), default=5, minimum=1, maximum=20)
                results = self._search(query, limit)
                if results is None:
                    return self._tool_error("memory search failed")
                return json.dumps({"ok": True, "results": results}, ensure_ascii=False, default=str)

            if tool_name == "memory_store":
                content = str((args or {}).get("content") or "").strip()
                if not content:
                    return self._tool_error("content is required")
                tags = self._normalize_tags((args or {}).get("tags"))
                memory_type = str((args or {}).get("memory_type") or "note").strip() or "note"
                if not self._store_memory(content, tags, memory_type):
                    return self._tool_error("memory store failed")
                return json.dumps({"ok": True}, ensure_ascii=False)

            return self._tool_error(f"unknown memory tool: {tool_name}")
        except Exception:
            logger.warning("mcp-memory tool call failed for %s", tool_name, exc_info=True)
            return self._tool_error("memory tool call failed")

    def on_session_end(self, messages: list[dict[str, Any]]) -> None:
        if not self._should_write:
            return
        try:
            content = self._format_messages(
                messages,
                limit=_SESSION_MESSAGE_LIMIT,
                max_chars=_SESSION_CHAR_LIMIT,
            )
            if not content:
                return
            self._store_memory(
                content,
                ["hermes", f"session-{self._today()}"],
                "note",
            )
        except Exception:
            logger.warning("mcp-memory session-end save failed", exc_info=True)

    def on_pre_compress(self, messages: list[dict[str, Any]]) -> str:
        if not self._should_write:
            return ""
        try:
            checkpoint = self._format_messages(
                messages,
                limit=_COMPRESS_MESSAGE_LIMIT,
                max_chars=_COMPRESS_CHAR_LIMIT,
            )
            if not checkpoint:
                return ""
            stored = self._store_memory(
                checkpoint,
                ["hermes", "checkpoint", f"session-{self._today()}"],
                "note",
            )
            return "已将压缩前的对话 checkpoint 保存到长期记忆。" if stored else ""
        except Exception:
            logger.warning("mcp-memory compression checkpoint failed", exc_info=True)
            return ""

    def shutdown(self) -> None:
        with self._lock:
            self._pending_prefetch.clear()
            self._prefetch_generation.clear()

    def _resolve_base_url(self) -> str:
        base_url = os.environ.get("MCP_MEMORY_URL", "").strip()
        if base_url:
            return base_url.rstrip("/")
        for path in self._config_candidates():
            base_url = self._read_config_value(path, "base_url")
            if base_url:
                return base_url.rstrip("/")
        return ""

    def _resolve_token(self) -> str:
        token = os.environ.get("MCP_MEMORY_TOKEN", "").strip()
        if token:
            return token
        for path in self._config_candidates():
            token = self._read_config_value(path, "token")
            if token:
                return token
        return ""

    def _config_candidates(self) -> list[Path]:
        return _candidate_config_paths(self._hermes_home)

    def _read_config_value(self, path: Path, field: str) -> str:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return ""
        except Exception as exc:
            logger.warning("mcp-memory could not read config %s: %s", path, exc)
            return ""
        if not isinstance(data, dict):
            return ""
        return str(data.get(field) or "").strip()

    def _request_json(self, method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
        if httpx is None:
            logger.warning("mcp-memory request skipped because httpx is unavailable")
            return None
        try:
            base_url = self._resolve_base_url()
            if not base_url:
                raise RuntimeError("mcp-memory base_url is not configured")
            token = self._resolve_token()
            if not token:
                raise RuntimeError("mcp-memory token is not configured")
            url = f"{base_url}/{path.lstrip('/')}"
            headers = {"X-API-Key": token, "Accept": "application/json"}
            with httpx.Client(timeout=_REQUEST_TIMEOUT) as client:
                response = client.request(method, url, headers=headers, json=payload)
                response.raise_for_status()
                if not response.content:
                    return {}
                return response.json()
        except Exception as exc:
            logger.warning("mcp-memory request failed: %s %s: %s", method, path, exc)
            return None

    def _search(self, query: str, n_results: int) -> list[Any] | None:
        if not _is_meaningful_query(query):
            return []
        data = self._request_json(
            "POST",
            "/api/search",
            {"query": query, "n_results": n_results},
        )
        if not isinstance(data, dict):
            return None
        results = data.get("results")
        if not isinstance(results, list):
            return None
        normalized: list[dict[str, Any]] = []
        for item in results:
            if not isinstance(item, dict):
                continue
            score = item.get("similarity_score")
            try:
                score = float(score)
            except (TypeError, ValueError):
                score = None
            normalized.append({**item, "similarity_score": score})
        return normalized

    def _store_memory(self, content: str, tags: list[str], memory_type: str) -> bool:
        data = self._request_json(
            "POST",
            "/api/memories",
            {"content": content, "tags": tags, "memory_type": memory_type},
        )
        return data is not None

    @staticmethod
    def _format_results(results: list[Any] | None) -> str:
        if not results:
            return ""
        parsed: list[tuple[float | None, str]] = []
        for item in results[:_PREFETCH_RESULTS]:
            content: Any = None
            score: float | None = None
            if isinstance(item, dict):
                memory = item.get("memory")
                if isinstance(memory, dict):
                    content = memory.get("content")
                elif memory is not None:
                    content = memory
                else:
                    content = item.get("content")
                try:
                    score = float(item.get("similarity_score"))
                except (TypeError, ValueError):
                    score = None
            if content is None:
                continue
            cleaned_lines: list[str] = []
            for raw_line in str(content).splitlines():
                line = raw_line.strip()
                while line:
                    previous = line
                    line = line.lstrip()
                    if line.startswith(("- ", "* ", "+ ")):
                        line = line[2:].lstrip()
                    elif line.startswith("#"):
                        line = line.lstrip("#").lstrip()
                    else:
                        number, separator, rest = line.partition(". ")
                        if separator and number.isdigit():
                            line = rest.lstrip()
                    if line == previous:
                        break
                if line:
                    cleaned_lines.append(line)
            text = " ".join(cleaned_lines)
            if not text:
                continue
            if len(text) > _MAX_PREFETCH_CHARS:
                text = text[: _MAX_PREFETCH_CHARS - 1] + "…"
            parsed.append((score, text))
        if not parsed:
            return ""

        scores = [score for score, _ in parsed if score is not None]
        threshold = max(scores) * _RELATIVE_SCORE_FLOOR if scores else None
        lines: list[str] = []
        seen: set[str] = set()
        for index, (score, text) in enumerate(parsed):
            if index > 0 and (threshold is None or score is None or score < threshold):
                continue
            if text in seen:
                continue
            seen.add(text)
            lines.append(f"- {text}")
        if not lines:
            return ""
        return "## 相关长期记忆\n" + "\n".join(lines)

    def _format_messages(self, messages: list[dict[str, Any]], *, limit: int, max_chars: int) -> str:
        if not isinstance(messages, list):
            return ""
        lines: list[str] = []
        if self._session_id:
            lines.append(f"Session: {self._session_id}")
        for message in messages[-limit:]:
            if not isinstance(message, dict):
                continue
            text = self._content_text(message.get("content"))
            if not text:
                continue
            if len(text) > _MESSAGE_CHAR_LIMIT:
                text = text[: _MESSAGE_CHAR_LIMIT - 1] + "…"
            role = str(message.get("role") or "message").strip() or "message"
            lines.append(f"{role}: {text}")
        if len(lines) <= (1 if self._session_id else 0):
            return ""
        content = "\n".join(lines)
        return content[-max_chars:]

    @staticmethod
    def _content_text(content: Any) -> str:
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text")
                    if text is not None:
                        parts.append(str(text))
            return "\n".join(part.strip() for part in parts if part.strip())
        if isinstance(content, dict):
            return json.dumps(content, ensure_ascii=False, default=str)
        return "" if content is None else str(content).strip()

    @staticmethod
    def _normalize_tags(value: Any) -> list[str]:
        if isinstance(value, str):
            candidates = [value]
        elif isinstance(value, list):
            candidates = value
        else:
            candidates = []
        tags = [str(tag).strip() for tag in candidates if str(tag).strip()]
        return tags or ["hermes"]

    @staticmethod
    def _bounded_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
        try:
            number = int(value)
        except (TypeError, ValueError):
            number = default
        return max(minimum, min(maximum, number))

    @staticmethod
    def _tool_error(message: str) -> str:
        return json.dumps({"ok": False, "error": message}, ensure_ascii=False)

    @staticmethod
    def _today() -> str:
        return datetime.now().astimezone().strftime("%Y-%m-%d")


def register(ctx: Any) -> None:
    """Register mcp-memory as a Hermes memory provider."""
    ctx.register_memory_provider(McpMemoryProvider())

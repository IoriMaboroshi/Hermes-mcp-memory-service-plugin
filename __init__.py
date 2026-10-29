"""Hermes memory provider backed by mcp-memory-service."""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from agent.memory_provider import MemoryProvider, spawn_context_thread

try:
    import httpx
except ImportError as exc:
    httpx = None
    _HTTPX_IMPORT_ERROR: Exception | None = exc
else:
    _HTTPX_IMPORT_ERROR = None


logger = logging.getLogger(__name__)

_BASE_URL = ""  # 必须由 config.json 的 base_url 或环境变量 MCP_MEMORY_URL 提供
_REQUEST_TIMEOUT = 3.0
_PREFETCH_RESULTS = 5
_MAX_PREFETCH_CHARS = 300
_SESSION_MESSAGE_LIMIT = 20
_COMPRESS_MESSAGE_LIMIT = 30
_MESSAGE_CHAR_LIMIT = 2_000
_SESSION_CHAR_LIMIT = 12_000
_COMPRESS_CHAR_LIMIT = 16_000
_SESSION_TOKEN_BUDGET = 30_000
_LLM_TIMEOUT = 15.0
_MCP_TIMEOUT = 20.0
_RELATIVE_SCORE_FLOOR = 0.95
_SESSION_EXTRACTION_SYSTEM_PROMPT = """You extract structured learnings from an agent session transcript.

Output ONLY a JSON object with these keys:
- task_summary: 1-2 sentences, what this session was about (write in Chinese)
- outcome: one of "success", "partial", "failure" (these exact three words)
- decisions: array of {"what": str, "why": str} — decisions made and their reasons. Write in Chinese.
- errors: array of {"error": str, "tool": str, "resolution": str} — what went wrong, which tool/command was involved, how it was resolved. Write in Chinese.
- user_corrections: array of {"original": str, "corrected_to": str} — original was what the agent assumed/did, corrected_to is what the user actually wanted. Write in Chinese.

Rules:
- Be specific and self-contained. Never write "today", "just now", "this PR", "刚才", "今天".
- Skip greetings, acknowledgements, trivial back-and-forth.
- Empty category → empty array.
- Always include the WHY for decisions; the reason is more valuable than the decision.
"""
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
    candidates.append(Path.home() / ".hermes" / "plugins" / "mcp-memory" / "config.json")

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
        self._config: dict[str, Any] = self._load_config()

    @property
    def name(self) -> str:
        return "mcp-memory"

    def is_available(self) -> bool:
        if httpx is None:
            return False
        return bool(self._resolve_token())

    def unavailable_reason(self) -> str:
        if httpx is None:
            detail = f": {_HTTPX_IMPORT_ERROR}" if _HTTPX_IMPORT_ERROR else ""
            return f"mcp-memory requires the httpx package{detail}"
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
        self._config = self._load_config()
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
            user_count = sum(
                1
                for message in messages
                if isinstance(message, dict)
                and str(message.get("role") or "").strip().lower() == "user"
            )
            if user_count < 2:
                return
            if not self._has_tool_trace(messages) and user_count < 4:
                return

            collected = self._collect_session_messages(messages)
            if not collected:
                return

            try:
                insights = self._extract_session_insights(collected)
            except Exception as exc:
                logger.warning("mcp-memory LLM extraction failed: %s", exc)
                insights = None
            if insights:
                try:
                    self._commit_session(insights)
                except Exception as exc:
                    logger.warning("mcp-memory session commit failed: %s", exc)

            lines: list[str] = []
            if self._session_id:
                lines.append(f"Session: {self._session_id}")
            lines.extend(
                f"{message['role']}: {message['content']}" for message in collected
            )
            tags = ["hermes", f"session-{self._today()}"]
            if insights is None:
                tags.append("no-llm-extract")
            self._store_memory("\n".join(lines), tags, "note")
        except Exception:
            logger.warning("mcp-memory session-end save failed", exc_info=True)

    @staticmethod
    def _has_tool_trace(messages: list[dict[str, Any]]) -> bool:
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role") or "").strip().lower()
            if role in {"tool", "function"}:
                return True
            if message.get("tool_calls") or message.get("function_call"):
                return True
            if message.get("tool_call_id"):
                return True
            content = message.get("content")
            if isinstance(content, list):
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    item_type = str(item.get("type") or "").strip().lower()
                    if item_type in {
                        "tool_call",
                        "tool_use",
                        "tool_result",
                        "function_call",
                        "function_result",
                    }:
                        return True
        return False

    def _collect_session_messages(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        if not isinstance(messages, list):
            return []

        collected: list[dict[str, Any]] = []
        estimated_tokens = 0
        for message in reversed(messages):
            if not isinstance(message, dict):
                continue
            role = str(message.get("role") or "").strip().lower()
            if role == "system":
                continue
            text = self._content_text(message.get("content"))
            if not text:
                continue
            if role == "assistant" and len(text) > 1_000:
                text = text[:800] + "…" + text[-200:]
            elif role in {"tool", "function"} and len(text) > 300:
                text = text[:300] + "…"
            elif role != "user" and len(text) > 500:
                text = text[:500]

            tokens = len(text) // 2
            if estimated_tokens + tokens > _SESSION_TOKEN_BUDGET:
                break
            collected.append({"role": role, "content": text})
            estimated_tokens += tokens
            if estimated_tokens >= _SESSION_TOKEN_BUDGET:
                break

        collected.reverse()
        return collected

    def _extract_session_insights(
        self, collected: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        if httpx is None or not collected:
            return None
        config = self._config
        api_key = str(config.get("llm_api_key") or "").strip()
        base_url = str(config.get("llm_base_url") or "").strip().rstrip("/")
        model = str(config.get("llm_model") or "").strip()
        if not api_key or not base_url or not model:
            return None

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        if "opencode.ai" in base_url.lower():
            headers["x-opencode-session"] = self._session_id or str(uuid4())

        transcript = "\n\n".join(
            f"{message['role']}: {message['content']}" for message in collected
        )
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": _SESSION_EXTRACTION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Session transcript:\n\n{transcript}",
                },
            ],
            "temperature": 0.1,
            "max_tokens": 1500,
        }
        try:
            with httpx.Client(timeout=_LLM_TIMEOUT) as client:
                response = client.post(
                    f"{base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()
                text = response.text
        except Exception as exc:
            logger.warning("mcp-memory LLM extraction failed: %s", exc)
            return None

        data = self._parse_json_object(self._extract_llm_output(text))
        if not isinstance(data, dict):
            return None
        task_summary = data.get("task_summary")
        if not isinstance(task_summary, str) or not task_summary.strip():
            return None
        # LLM 经常忽略枚举约束返回描述句，统一归一化而不是丢弃整条结果
        outcome = self._normalize_outcome(data.get("outcome"))

        decisions = data.get("decisions")
        errors = data.get("errors")
        corrections = data.get("user_corrections")
        return {
            "task_summary": task_summary.strip(),
            "outcome": outcome,
            "decisions": decisions if isinstance(decisions, list) else [],
            "errors": errors if isinstance(errors, list) else [],
            "user_corrections": corrections if isinstance(corrections, list) else [],
        }

    @staticmethod
    def _extract_llm_output(text: str) -> str:
        raw = str(text or "").strip()
        try:
            envelope = json.loads(raw)
        except (TypeError, ValueError):
            return raw
        if not isinstance(envelope, dict):
            return raw
        choices = envelope.get("choices")
        if not isinstance(choices, list) or not choices:
            return raw
        first = choices[0]
        if not isinstance(first, dict):
            return raw
        message = first.get("message")
        if not isinstance(message, dict):
            return raw
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("text") is not None:
                    parts.append(str(item["text"]))
            if parts:
                return "\n".join(parts)
        return raw

    @staticmethod
    def _parse_json_object(text: str) -> dict[str, Any] | None:
        candidates: list[str] = [str(text or "").strip()]
        current = candidates[0]
        if "```json" in current.lower():
            start = current.lower().find("```json") + len("```json")
            end = current.find("```", start)
            if end != -1:
                candidates.append(current[start:end].strip())
        first_brace = current.find("{")
        last_brace = current.rfind("}")
        if first_brace != -1 and last_brace > first_brace:
            candidates.append(current[first_brace : last_brace + 1])

        for candidate in candidates:
            if not candidate:
                continue
            try:
                data = json.loads(candidate)
            except (TypeError, ValueError):
                continue
            if isinstance(data, dict):
                return data
        return None

    def _commit_session(self, insights: dict[str, Any]) -> bool:
        if httpx is None:
            return False
        token = self._resolve_token()
        if not token:
            return False
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        insights = self._normalize_insights(insights)
        arguments = {
            "session_id": self._session_id or "hermes-unknown",
            "agent_id": "hermes",
            "task_summary": str(insights.get("task_summary") or ""),
            "outcome": self._normalize_outcome(insights.get("outcome")),
            "decisions": self._coerce_objects(insights.get("decisions"), "decision"),
            "errors": self._coerce_objects(insights.get("errors"), "error"),
            "user_corrections": self._coerce_objects(
                insights.get("user_corrections"), "correction"
            ),
        }
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "commit_session_legacy", "arguments": arguments},
        }
        try:
            with httpx.Client(timeout=_MCP_TIMEOUT) as client:
                response = client.post(
                    f"{self._base_url().rstrip('/')}/mcp",
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()
                return self._mcp_response_ok(response.text)
        except Exception as exc:
            logger.warning("mcp-memory commit_session_legacy failed: %s", exc)
            return False

    @staticmethod
    def _normalize_insights(insights: dict[str, Any]) -> dict[str, Any]:
        """把 LLM 可能的字段变体对齐到服务端 schema。

        服务端实际读取的字段（server_impl.py:2452-2480）：
          errors:           {error, tool, resolution}
          user_corrections: {original, corrected_to}
          decisions:        {what, why}
        """
        if not isinstance(insights, dict):
            return {}
        result = dict(insights)

        errors: list[dict[str, str]] = []
        for item in insights.get("errors") or []:
            if isinstance(item, dict):
                errors.append(
                    {
                        "error": str(item.get("error") or item.get("problem") or "").strip(),
                        "tool": str(item.get("tool") or item.get("context") or "").strip(),
                        "resolution": str(
                            item.get("resolution")
                            or item.get("fix")
                            or item.get("root_cause")
                            or ""
                        ).strip(),
                    }
                )
            elif item:
                errors.append({"error": str(item).strip(), "tool": "", "resolution": ""})
        result["errors"] = [e for e in errors if e["error"]]

        corrections: list[dict[str, str]] = []
        for item in insights.get("user_corrections") or []:
            if isinstance(item, dict):
                corrected = str(
                    item.get("corrected_to") or item.get("correct") or item.get("correction") or ""
                ).strip()
                original = str(item.get("original") or item.get("wrong") or "").strip()
                if corrected or original:
                    corrections.append({"original": original, "corrected_to": corrected})
            elif item:
                corrections.append({"original": "", "corrected_to": str(item).strip()})
        result["user_corrections"] = corrections

        return result

    @staticmethod
    def _coerce_objects(value: Any, key: str) -> list[dict[str, Any]]:
        """commit_session_legacy 要求 decisions/errors/user_corrections 都是 object 数组。"""
        if not isinstance(value, list):
            return []
        result: list[dict[str, Any]] = []
        for item in value:
            if isinstance(item, dict):
                result.append(item)
            elif item is not None and str(item).strip():
                result.append({key: str(item).strip()})
        return result

    @staticmethod
    def _normalize_outcome(value: Any) -> str:
        """服务端只接受 success / partial / failure，LLM 常给 completed / failed。"""
        raw = str(value or "").strip().lower()
        mapping = {
            "completed": "success",
            "complete": "success",
            "done": "success",
            "ok": "success",
            "failed": "failure",
            "fail": "failure",
            "error": "failure",
        }
        normalized = mapping.get(raw, raw)
        return normalized if normalized in {"success", "partial", "failure"} else "success"

    @staticmethod
    def _mcp_response_ok(text: str) -> bool:
        raw = str(text or "").strip()
        if not raw:
            return False
        candidates = [
            line[len("data:") :].strip()
            for line in raw.splitlines()
            if line.startswith("data:")
            and line[len("data:") :].strip() not in {"", "[DONE]"}
        ]
        if not candidates:
            candidates.append(raw)

        for candidate in candidates:
            try:
                data = json.loads(candidate)
            except (TypeError, ValueError):
                continue
            if not isinstance(data, dict):
                continue
            if data.get("error"):
                return False
            result = data.get("result")
            if isinstance(result, dict) and result.get("isError") is True:
                return False
            if "result" in data:
                return True
        return False

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

    def _resolve_token(self) -> str:
        token = os.environ.get("MCP_MEMORY_TOKEN", "").strip()
        if token:
            return token
        for path in self._config_candidates():
            token = self._read_config_token(path)
            if token:
                return token
        return ""

    def _base_url(self) -> str:
        configured = str(self._config.get("base_url") or "").strip()
        return configured or _BASE_URL

    def _load_config(self) -> dict[str, Any]:
        config: dict[str, Any] = {}
        for path in self._config_candidates():
            data = self._read_config_data(path)
            if not data:
                continue
            for key, value in data.items():
                config.setdefault(key, value)
        return config

    def _config_candidates(self) -> list[Path]:
        return _candidate_config_paths(self._hermes_home)

    def _read_config_token(self, path: Path) -> str:
        data = self._read_config_data(path)
        return str(data.get("token") or "").strip()

    def _read_config_data(self, path: Path) -> dict[str, Any]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except Exception as exc:
            logger.warning("mcp-memory could not read config %s: %s", path, exc)
            return {}
        if not isinstance(data, dict):
            return {}
        return data

    def _request_json(self, method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
        if httpx is None:
            logger.warning("mcp-memory request skipped because httpx is unavailable")
            return None
        try:
            token = self._resolve_token()
            if not token:
                raise RuntimeError("mcp-memory token is not configured")
            url = f"{self._base_url().rstrip('/')}/{path.lstrip('/')}"
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

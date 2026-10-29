# Hermes mcp-memory-service Provider

A [Hermes Agent](https://hermes-agent.nousresearch.com) memory provider plugin that connects to a self-hosted [mcp-memory-service](https://github.com/doobidoo/mcp-memory-service) server.

在 `prefetch` 环节将相关长期记忆注入上下文，在会话结束时用 LLM 把对话提炼成结构化字段（决策 / 错误 / 用户纠正）写回记忆库。

## What it does

| Hook | Behavior |
|---|---|
| `prefetch(query)` | 每轮对话前语义检索，注入相关的历史记忆（带相对分数过滤，避免噪声） |
| `queue_prefetch` | 后台线程预取，不阻塞主线程 |
| `on_session_end(messages)` | 按 30K token 预算收集会话 → LLM 提炼 → `commit_session_legacy` → 另存原文兜底 |
| `on_pre_compress(messages)` | 压缩前抢救对话快照 |
| `memory_search` / `memory_store` | 暴露给 agent 的两个工具 |

## Installation

```bash
# 1. 放进 Hermes 的 plugins 目录
cp -r mcp-memory ~/.hermes/plugins/mcp-memory

# 2. 写配置
cp config.example.json ~/.hermes/plugins/mcp-memory/config.json
# 编辑填 base_url / token

# 3. 启用
hermes plugins enable mcp-memory
hermes config set memory.provider mcp-memory

# 4. 重启
```

验证：

```bash
hermes memory status
# Provider:  mcp-memory
# Plugin:    installed ✓
# Status:    available ✓
```

## Configuration

`config.json`（查找顺序：环境变量 → `$HERMES_HOME/plugins/mcp-memory/` → `~/.hermes/plugins/mcp-memory/` → `~/.config/mcp-memory/`）

```json
{
  "token": "your-mcp-memory-token",
  "base_url": "http://127.0.0.1:8000",

  "llm_api_key": "optional",
  "llm_base_url": "https://api.openai.com/v1",
  "llm_model": "gpt-4o-mini"
}
```

- `token` — mcp-memory-service 的 API key，必填
- `base_url` — 服务地址，必填（不带尾部斜杠）
- `llm_*` — 可选。不填则跳过 LLM 提炼，只存原文，`tags` 里会带 `no-llm-extract`

也可以用环境变量 `MCP_MEMORY_URL` / `MCP_MEMORY_TOKEN` 覆盖。

**支持 OpenAI 兼容的任何端点。** 如果 `llm_base_url` 包含 `opencode.ai`，插件会自动附加 `x-opencode-session` 头。

## How session collection works

不是"取最后 20 条消息"这种粗暴截断。旧版就是这么干的，长会话的决策链全部丢失。

现在的规则：

```
从最后一条往前收集，直到累计估算 token 达到 30,000
token 估算 = len(text) // 2

按角色分层截断：
  user       → 全文保留（这是意图，最重要）
  assistant  → 超 1000 字 → 前 800 + "…" + 后 200
  tool       → 超 300 字  → 前 300 + "…"
  system     → 跳过
  其他       → 超 500 字  → 截断
```

入口过滤：`user` 消息少于 2 条直接跳过（闲聊不入库）。

## What gets extracted

LLM 返回的结构化 JSON：

```json
{
  "task_summary": "把 embedding 模型换成 bge-m3，先备份、删库重建",
  "outcome": "success",
  "decisions": [
    {"what": "改动前先 cp -a 全量备份", "why": "换模型失败时能完整回滚"}
  ],
  "errors": [
    {"error": "批量导入返回 400", "tool": "embedding API", "resolution": "截断到 6000 字符"}
  ],
  "user_corrections": [
    {"original": "用本地模型", "corrected_to": "用云端 API，不装 torch"}
  ]
}
```

这套字段名是 mcp-memory-service `commit_session_legacy` 实际读取的 schema（`server_impl.py`），改了就对不上，服务端会静默存空字符串。

插件内部做了容错归一化：LLM 返回 `completed` / `failed` / 字符串数组等变体都会被映射到正确格式，不会整条丢弃。

## Requirements

- Hermes Agent（需要 `MemoryProvider` 接口）
- mcp-memory-service ≥ 12.x，启用 `sqlite_vec` 后端
- `httpx`

服务端的 `commit_session_legacy` 工具会：
- 存一条 `session-legacy` observation
- 每条 decision 存成独立 observation
- 每条 error 转成 mistake note（Pattern / Context / Wrong）
- 每条 user_correction 存成高优先级 observation

所以 `errors` 和 `user_corrections` 是有长期价值的，别留空。

## Known limitations

- `prefetch` 每轮一次跨网络请求。服务在内网/本机时约 50ms，跨公网 300–700ms
- 相对分数过滤（`score >= top*0.95`）针对 bge-m3 这类有区分度的模型调过。用小维度模型（如 all-MiniLM 384 维）时分数会挤在 0.65–0.8，过滤效果会退化
- `on_session_end` 同步执行，含 LLM 调用约 5–10 秒。Hermes 在后台线程调用它，不影响对话，但会占一个 worker
- 不做会话去重。同一 session 重复调用 `commit_session_legacy` 会重复入库

## License

MIT

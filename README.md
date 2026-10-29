# Hermes Memory MCP

A memory provider plugin for Hermes Agent. It connects the agent's long-term memory interface to a
self-hosted [mcp-memory-service](https://github.com/doobidoo/mcp-memory-service) instance running in
REST API mode.

## What it does

Hermes supports pluggable memory providers. This plugin implements that interface so an agent can:

- recall relevant long-term memories automatically before a turn;
- expose explicit `memory_search` and `memory_store` tools to the agent;
- archive the end of each session; and
- save a checkpoint before conversation compression discards context.

The plugin is deliberately small: it translates Hermes memory lifecycle hooks into requests against
the mcp-memory-service REST API and keeps no local memory database of its own.

## Requirements

- Hermes Agent with external memory provider support.
- Python 3.10 or newer.
- The `httpx` package. Hermes installs plugin dependencies declared in `plugin.yaml`.
- A running mcp-memory-service instance with its REST API enabled and an API key configured.

## Installation

```bash
git clone <repo> ~/.hermes/plugins/mcp-memory
cp ~/.hermes/plugins/mcp-memory/config.example.json ~/.hermes/plugins/mcp-memory/config.json
# Edit config.json and fill in base_url and token.
hermes config set memory.provider mcp-memory
hermes config set plugins.enabled '["mcp-memory"]'
# Restart the Hermes gateway.
```

`config.json` contains credentials and is ignored by Git. Never commit it.

## Configuration

The plugin accepts configuration from environment variables or `config.json`. Environment variables
take precedence.

| Setting | Environment variable | `config.json` field | Description |
| --- | --- | --- | --- |
| Base URL | `MCP_MEMORY_URL` | `base_url` | URL of the mcp-memory-service REST API, without a trailing slash requirement. |
| API key | `MCP_MEMORY_TOKEN` | `token` | API key sent in the `X-API-Key` header. |

Example `config.json`:

```json
{
  "base_url": "http://localhost:8765",
  "token": "your-api-key-here"
}
```

Configuration files are checked at these locations:

1. `<hermes_home>/plugins/mcp-memory/config.json` when Hermes supplies `hermes_home`;
2. `$HERMES_HOME/.hermes/plugins/mcp-memory/config.json`; and
3. `$HOME/.hermes/plugins/mcp-memory/config.json`.

The plugin is available only when `httpx` is installed, a base URL is configured, and a token is
configured. If the base URL is missing, `unavailable_reason()` reports that `base_url` is not
configured.

## Implemented hooks

| Hook | Behavior |
| --- | --- |
| `is_available` | Checks for `httpx`, a base URL, and a token before initialization. |
| `initialize` | Records the session, Hermes home, platform, and whether this agent should write memories. |
| `system_prompt_block` | Tells the agent when to search memory and when to store durable facts. |
| `prefetch` | Returns relevant memory for the current query, consuming any queued background result. |
| `queue_prefetch` | Searches memory in a background thread for use on the next turn. |
| `get_tool_schemas` | Publishes the `memory_search` and `memory_store` tool schemas. |
| `handle_tool_call` | Executes memory search and store requests and returns JSON results. |
| `on_session_end` | Archives a bounded transcript when a session ends. |
| `on_pre_compress` | Saves a bounded checkpoint before context compression. |
| `shutdown` | Clears queued prefetch state and pending generations. |

Only the primary agent context writes memories. Delegated contexts can still use configured tools,
but session-end and compression hooks avoid duplicating archived transcripts.

## Tools

| Tool | Purpose | Key arguments |
| --- | --- | --- |
| `memory_search` | Search long-term memory for relevant prior context. | `query`, optional `n_results` from 1 to 20 |
| `memory_store` | Store a durable fact, preference, decision, or conclusion. | `content`, optional `tags`, optional `memory_type` |

Both tools return JSON. Search results include the service's stored memory fields and normalized
similarity scores when available.

## Design notes

### Configuration before initialization

Hermes asks whether a provider is available before calling `initialize()`. Configuration resolution
therefore cannot depend on initialized instance state. The plugin resolves environment variables and
probes the standard Hermes config locations on demand, including from `is_available()` and
`unavailable_reason()`.

### Relative score filtering

This plugin uses a relative cutoff rather than a fixed similarity threshold. The
`all-MiniLM-L6-v2` embedding model commonly produces cosine similarities compressed into the roughly
0.65–0.8 range, and unrelated text can also score around 0.7. A fixed absolute threshold is therefore
a poor relevance signal. After sorting search results, prefetch keeps only results whose score is at
least `top_score * 0.95`, which filters clearly weaker neighbors relative to the best match for that
query.

### Query gate

Very short or information-free queries such as greetings, acknowledgements, and phrases like "test
this" tend to retrieve noise. Returning no prefetched memory is preferable to injecting irrelevant
context into the prompt, so the plugin skips search when the remaining query has too little semantic
content.

## Known limitations

- Recall accuracy is bounded by the embedding model used by mcp-memory-service.
- Chinese semantic matching is serviceable but generally weaker than matching in languages with more
  available embeddings for the configured model.
- In a large memory store, top-k semantic search can miss exact keyword matches. Search results
  should be treated as context, not authoritative facts.

## Affiliation

This is an independent community project. It is not affiliated with, endorsed by, or maintained by
Anthropic or Nous Research.

## License

MIT. See [LICENSE](LICENSE).

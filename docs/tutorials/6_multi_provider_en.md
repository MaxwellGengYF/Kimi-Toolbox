# Multi-Provider Configuration

Kimix now supports routing different agent roles to different LLM providers. Instead of a single optional `sub_provider`, you can declare a `sub_providers` list and tag each entry with a `role`.

---

## Why Use Multiple Providers

| Scenario | Benefit |
|----------|---------|
| Planner consumes too many main-model tokens | Use a cheap/light model for `/plan` |
| Sub-agents need different capabilities | Route coding sub-agents to a coding model, review sub-agents to a reasoning model |
| Cost control | Keep the main model only for final replies |

---

## Config Format

Add a `sub_providers` array to your config file. Each entry is a complete provider dict plus a `role` field.

```json
{
  "model": "kimi-for-coding",
  "max_context_size": 262144,
  "capabilities": ["thinking"],
  "url": "https://api.kimi.com/coding/v1",
  "type": "kimi",
  "api_key": "sk-main",
  "sub_providers": [
    {
      "role": "sub_agent",
      "model": "kimi-k2.6",
      "max_context_size": 200000,
      "capabilities": ["thinking"],
      "url": "https://api.moonshot.cn/v1/chat/completions",
      "type": "kimi",
      "api_key": "sk-sub"
    },
    {
      "role": "planner",
      "model": "claude-opus-4-6",
      "max_context_size": 200000,
      "capabilities": ["thinking"],
      "url": "https://api.minimaxi.com/anthropic",
      "type": "anthropic",
      "api_key": "sk-plan"
    }
  ]
}
```

The `model` field can be either `kimi-for-coding` or `kimi-for-coding-highspeed`; both are supported.

### Supported roles
| Role | Used by |
|------|---------|
| `sub_agent` | `Agent` tool, long-output summarization |
| `planner` | `/plan` command (`prompt_plan_async`) |
| `backup` | Automatic failover in `_run_single_prompt` when the primary provider exhausts retries |

If `role` is omitted, it defaults to `sub_agent`.

---

## Backup Provider Failover

When the primary provider exhausts its retries during a prompt, kimix automatically tries **backup** providers declared in `sub_providers` (in declaration order). One config can declare **multiple** backup providers.

Behavior:

1. The current (active) provider is always tried first.
2. On retry exhaustion, the session's LLM is swapped to each `backup` provider in turn.
3. Once a backup succeeds, the session **stays** on that backup for all subsequent prompts (it becomes the new primary).
4. The old provider's HTTP client is closed before switching (prevents resource leaks).
5. If no backups are configured, the primary error propagates as before (no behavior change).

Multiple `backup` entries are allowed and do not emit the "duplicate role" warning.

### Config example with backup

```json
{
  "model": "kimi-for-coding",
  "max_context_size": 262144,
  "url": "https://api.kimi.com/coding/v1",
  "type": "kimi",
  "api_key": "sk-main",
  "sub_providers": [
    {
      "role": "backup",
      "model": "claude-sonnet-4-20250514",
      "max_context_size": 200000,
      "url": "https://api.anthropic.com",
      "type": "anthropic",
      "api_key": "sk-backup-1"
    },
    {
      "role": "backup",
      "model": "gpt-4o",
      "max_context_size": 128000,
      "url": "https://api.openai.com/v1",
      "type": "openai_legacy",
      "api_key": "sk-backup-2"
    }
  ]
}
```

---

## Backward Compatibility

The old single `sub_provider` field is still parsed but normalized into `sub_providers` internally.

```json
{
  "sub_provider": {
    "role": "sub_agent",
    "model": "...",
    "type": "...",
    "url": "...",
    "max_context_size": 200000,
    "api_key": "..."
  }
}
```

This is equivalent to:

```json
{
  "sub_providers": [
    {
      "role": "sub_agent",
      "model": "...",
      "type": "...",
      "url": "...",
      "max_context_size": 200000,
      "api_key": "..."
    }
  ]
}
```

You can also mix both fields; entries are merged into one normalized list.

---

## How It Works

1. At startup, `kimix` loads the main provider from the config file.
2. It pops `sub_provider` and `sub_providers`, normalizes them, and stores the list via `set_default_sub_providers()`.
3. When a component needs an auxiliary provider, it calls `get_default_sub_provider("<role>")`.
   - If a matching role is found, that provider is used.
   - Otherwise it falls back to the main provider.

---

## Best Practices

1. **Use roles explicitly** — even if you only have one sub-provider, adding `"role": "sub_agent"` makes the intent clear.
2. **Planner can be lightweight** — `/plan` mainly structures tasks; it does not need your strongest model.
3. **Keep required keys** — every sub-provider must include `type`, `model`, `url`, and `max_context_size`. Invalid entries are ignored with a warning.
4. **First match wins** — if multiple entries share the same role (other than `backup`), the first one is used. For `backup`, all entries are tried in order during failover.

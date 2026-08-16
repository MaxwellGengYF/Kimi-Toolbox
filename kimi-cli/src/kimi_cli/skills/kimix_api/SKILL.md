---
name: kimix_api
description: Guide for using KimiX API utilities covering kimix, kimix.utils, kimix.base, kimix.dag, kimix.network, kimix.server, kimix.parser, kimix.tools, kimix.cot, kimix.retrieval, kimix.summarize, and kimi_agent_sdk.
---

# Kimi API Utilities Guide

High-level guide to `kimix.utils`, `kimix.base`, the full public API of `src/kimix` and `src/kimi_agent_sdk`. Full code examples per section: read `references/api.md`.

## Initialization & Config (`kimix.utils.config`)

- `init(config_path=None, config_json=None, yolo=True, think=True, skill_dir=None, ralph=None, manually_cot=False, colorful_print=True, clean=False)` — initialize global state (auto-resolves provider from `default_config.json` → env vars → OAuth if skipped).
- `_create_config(provider_dict) -> (Config, provider_dict)` — build a validated `Config` from a provider dict.

## Session Management (`kimix.utils`)

- `create_session(session_id=None, work_dir=KaosPath, skills_dir=None, agent_file=None, resume=False, provider_dict=None, chat_provider=None, agent_type=SystemPromptType.Worker, vfs_path=None, extra_system_prompt=None, max_ralph_iterations=None, anonymous=False, custom_data=None)`; close with `close_session(session)` / `await close_session_async(session)`.
- `_create_session_async(...)` — async version; `create_supervisor_session(...)` — Supervisor role with `agent_boss.json`.
- Default session: `_create_default_session(resume=True)`, `get_default_session()`, `_create_default_session_async(resume=True)`.
- `prompt(prompt_str, session=None, output_function=None, info_print=True, cancel_callable=None, close_session_after_prompt=False, merge_wire_messages=None, ensure_todo_finished=True, export_todo_list_path=None, format_output=False, timeout=None)` / `await prompt_async(...)` — auto-escapes file paths, exports >64 KiB prompts to temp files, retries non-API errors up to 3×, runs todo reminders, clears todos.
- Cancel: `cancel_prompt(session)`, `get_cancel_event(session)`.
- Context: `clear_default_context(force_create=True, resume=True, print_info=True)`, `compact_default_context()`, `print_usage(session)`, `context_path()`, `delete_session_dir()`; per-session `clear_context`/`compact_context` (+ async) from `kimix.utils.session`.
- Ralph loop: `set_ralph_loop(value, session=None)`; tool errors: `get_tool_call_errors(session)`.

## System Prompt Types (`kimix.utils.system_prompt`)

`SystemPromptType.Worker/TodoMaker/Thinker/TrivialSubAgent/Supervisor/Reader/SwarmLeader`; build prompts via `get_system_prompt(yolo, work_dir, extra_system_prompt, agent_role, max_system_prompt_tokens)` returning a callable `(runtime, is_compacting=False, compact_export_path=None) -> str`; customize with `SystemPromptCallback.role_callback`.

## Colorful Printing (`kimix.base`)

`print_success/print_error/print_warning/print_info/print_debug/print_string`, `colorful_print(text, fg, bg, styles)`, `colorful_text(...)` with `Color`/`BgColor`/`Style`; async `print_agent_json(wire_msg, session, output_function)` streams wire messages (auto-resolves approvals, skips step markers, prints tool calls/results).

## Threading / Process (`kimix.base`)

- `run_thread(func, args)` (max 8 concurrent), `sync_all()`.
- `async_prompt(text, session=None)`, `async_fix_error(command, extra_prompt=None, keycode=..., skip_success=True, max_loop=4, session=None)`.
- `run_process_with_error(command, keycode, skip_success=True)` (+ async); `run_script(path)`.

## File / Plan / Fix (`kimix.utils`)

- `prompt_path(path, split_word=None, session=None, after_prompt_coro=None)`.
- `fix_error(command, extra_prompt=None, keycode=..., skip_success=True, session=None, max_loop=4, merge_wire_messages=False)` (+ `fix_error_async`).
- `prompt_plan(requirement, plan_file="plan.md")` / `prompt_plan_async` — TodoMaker planner → review → implement → verify.

## Configuration Variables (`kimix.base`)

`_default_thinking`, `_default_yolo`, `_default_agent_file(_dir)`, `_default_skill_dirs`, `_default_provider`, `_default_sub_providers`, `_default_ralph`, `_default_manually_cot`, `_quiet`, `_colorful_print`, `_print_func`, `COMMON_SKILL_DIRS`; setters `set_default_*`; `get_default_sub_provider(role)`; `get_skill_dirs(use_kaos_path=True)`; utils `percentage_str`, `percentage_and_token(session)`, `generate_memory`, `make_kaos_dir(path)`.

## Prompt String Utilities (`kimix.utils.prompt_str`)

- `escape_file_paths(text, *, max_chars=0, max_repeat=100, truncate_msg="", case_mode="")` — backtick-wraps legal paths, strips invalid unicode/invisible chars, NFKC, emoji, repeated punctuation; preserves code blocks/quotes/URLs/fractions/dates.
- `clean_text(text, keep_newlines=True)` — remove zero-width/control chars, NFC-normalize.

## Windows Environment

`refresh_env_from_registry()` — reload HKLM/HKCU env vars into `os.environ`.

## `kimi_agent_sdk`

- `kimi_agent_sdk.prompt(user_input, *, work_dir, config, model, thinking, yolo, approval_handler_fn, agent_file, mcp_configs, skills_dir(s), max_steps_per_turn, max_retries_per_step, max_ralph_iterations, final_message_only)` — async generator of `Message`.
- `Session.create(work_dir=None, *, session_id, config, model, thinking, yolo, plan_mode, agent_file, mcp_configs, skills_dir(s), anonymous, max_steps_per_turn, max_retries_per_step, max_ralph_iterations, **custom)` / `Session.resume(work_dir, session_id)` — async context manager: `prompt(user_input, merge_wire_messages=False)`, `cancel()`, `close()`, `clear()`, `rename(new_id)`, `compact(custom_instruction="")`, `export(output_path=None)`, properties `id`, `model_name`, `status`.
- Re-exports: `CallableTool2`/`ToolOk`/`ToolError`, wire/message types, `Config`/`MCPConfig`, exceptions, `ApprovalHandlerFn`/`ApprovalRequest`, `MessageAggregator(final_message_only)` (`feed`/`flush`).

## `kimix.dag` — DAG Execution

`DAG.add_node(TaskNode(name, func, params, dependencies, retries))`, `add_edge`, `validate`, `nodes`/`edges`; `Context.get/set/update/cancel`; `Executor(max_workers).execute(dag, ctx) -> dict`; `TopologicalSorter(edges).sort()`; `detect_cycle`, `validate_dag`, exceptions `DAGValidationError`/`CycleError`/`DependencyError`/`ExecutionError`.

## `kimix.network` — TCP / JSON-RPC

- `TCPClient(host, port)` — `connect/disconnect/send/send_bytes/is_connected/on_*`.
- `TCPServer(host, port)` — `start/stop/send/send_bytes/disconnect_client/on_*`.
- `TcpGroupServer(host, port, max_workers)` — `send/broadcast/get_client_ids/wait_for_clients/on_*`.
- `JSONRPCClient.call(method, *args, timeout=5.0)`; `JSONRPCServer.register/start/stop/wait_for_connection/start_websocket_server`.

## `kimix.server` — HTTP Server

`create_app()` (FastAPI: health, SSE `/event`, session CRUD, prompt, abort, permissions, clear, context, compact, export); `KimixAsyncClient(host, port, timeout)` (`health_check`, `create_session`, `get/list/delete_session`, `get_messages`, `send_prompt_async`, `abort_session`, `clear_session`, `compact_session`, `export_session`, `stream_events`, `stream_events_robust`); `SessionManager` (global `session_manager`), `BusEvent`/`EventBus` (global `bus`), `serve_cli(args)`.

## `kimix.parser` — Comment Parsers

`PythonParser/CParser/ShellParser/HtmlParser/PascalParser/LispParser/SqlParser` — `parse(source_code) -> ParseResult` (`total_comments`, `comment_lines`, `get_comments_by_kind`), `parse_file(path)`; `Comment(content, line, column, kind)`.

## `kimix.tools` — Built-in Agent Tools

`Agent` (subagent), `AskAgent` (send_message), `AgentList` (list_agents), `AgentClose` (interrupt_agent), `TaskOutput` (job_output), `BackgroundStream`, `Bash`/`Powershell`, `Run`, `FindStr`, `Mkdir`/`Rm`, `Python`, `SyntaxLint`/`MypyCheck`/`Cpplint`/`JsTsSyntaxCheck`, `Ocr`, `Docx2md`, `Pdf2md`, `ParserTool`, `WritePlan`/`ReadPlan`/`EditPlan`, `StoreSession`/`LoadSession`/`LsSession`, `fetch_url`/`fetch_to_markdown`, `Zip`/`Unzip`.

## `kimix.cot` — Chain-of-Thought

`cot_prompt(prompt_str, self_verify=True, existing_thinking=None, max_iterations=10) -> CoTResult(thinking, quit)` (+ async); two-pass `cot_prompt_with_verification` (+ async).

## `kimix.retrieval` — BM25

`NgramTokenizer`, `InvertedIndex`, `BM25Scorer.score/score_topk`, `Searcher.search(query, top_k)`, `LevenshteinAutomaton`, `SimHash`/`SimHashLSH`, `RM3Expander`/`RocchioExpander`, `LambdaMART`/`CoordinateAscent`/`RankSVM`/`RankBoost`, `QueryPerformancePredictor`, `MinHash`, `NoisyChannelSpeller`, plus utility functions (`jaro_winkler_similarity`, `cosine_similarity_tfidf`, `mmr_rerank`, `xquad_rerank`, ...).

## `kimix.summarize`

`summarize(temp_file=None, session=None, only_return_remember_str=False) -> str | None`; `summarize_mistake(result_file, session=None)`.

## `kimix.cli`

`cli()` — launches the interactive Kimix CLI.

## Dynamic System Reminders

`DynamicInjection(type, content)` wrapped in `<system-reminder>`; `DynamicInjectionProvider.get_injections(history, soul)` with optional `on_context_compacted()`/`on_afk_changed()`; register via `soul.add_injection_provider(provider)`. Built-in: `CompactReminderProvider` (compact_reminder).

## Context Pruning

`ContextPruner`/`PruningResult`/`ElidedRecord` in `kimi_cli/soul/context_pruning.py`, integrated in `KimiSoul._step()`. Tier A drops consumed ephemera; Tier B elides stale/oversized content with `retrieve`-able stubs. Config via `LoopControl.prune_*` fields (master switch `context_pruning_enabled`). Manual trigger: `/prune`.

## Complete Package Index

| Package / Module | Description |
|------------------|-------------|
| `kimi_agent_sdk` | Python SDK for building AI agents (Session, prompt, wire/message types, config, exceptions) |
| `kimix.base` | Colorful printing, threading, process execution, config variables |
| `kimix.cli` | Interactive CLI entry point (`cli()`) |
| `kimix.cot` | Chain-of-thought reasoning utilities |
| `kimix.dag` | DAG task dependency execution engine |
| `kimix.network` | TCP / JSON-RPC networking layer |
| `kimix.parser` | Source code comment parsers for multiple languages |
| `kimix.retrieval` | BM25 retrieval engine, fuzzy search, ranking, query performance prediction |
| `kimix.server` | Opencode-style HTTP server (FastAPI + SSE) |
| `kimix.summarize` | Context compaction / summarization helpers |
| `kimix.tools` | Built-in agent tools: shell, Python, file ops, OCR, PDF/DOCX, linting, planning |
| `kimix.utils` | Session management, prompting, plan execution, error fixing, init, prompt string utilities |
| `kimix.utils.config` | `init` and `_create_config` |
| `kimix.utils.fix_error` | `fix_error`, `fix_error_async`, `async_prompt`, `async_fix_error` |
| `kimix.utils.prompt` | `prompt_plan`, `prompt_path`, `prompt`, `prompt_async` |
| `kimix.utils.prompt_str` | `escape_file_paths`, `clean_text` |
| `kimix.utils.session` | Session creation, resumption, context management, lifecycle |
| `kimix.utils.system_prompt` | System prompt types and builders |
| `kimix.utils.windows_env` | Windows registry environment refresh |

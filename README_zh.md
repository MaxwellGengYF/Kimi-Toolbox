# Kimi-CLI-X

## 源码安装
```bash
python install.py
```

## pip 安装
```bash
# 安装
pip install kimix
# 运行
python -m kimix.cli
# 或
kimix
python -m kimix
```

注意！这个仓库不止支持 KIMI LLM，支持各类 API Key! 如 OpenAI, Anthropic 等。各类默认 config 模板在 `docs/`, 配置后通过 `kimix --config=xx.json` 即可使用 !

![teasor](kimix_zh.png)
## 为什么选择 Kimi-CLI-X？

Kimi-CLI-X 在原版 Kimi-CLI 基础上，围绕**提示词效率**、**工具可靠性**与**可扩展性**进行了深度优化，并补充了多款面向实际开发场景的工具。

### 优化

1. **精简系统提示词** — 压缩初始提示词与工具说明的篇幅，保证信息完整的同时让上下文更干净，覆盖几乎全部内置工具的同时，将初始化 token 降到 2000 左右。
2. **强化权限与校验** — 妥善处理 Shell、Glob 等工具的校验和权限问题，减少因失败导致的反复修正。
3. **优化子进程输出** — 主动将大量输出重定向到临时文件，过滤冗余日志，便于后端检索。
4. **简化并发架构** — 理顺子进程、子代理与后台多任务的设计，使多任务调度更直观可控。
5. **可编程提示词** — 支持在上层自定义、注入系统提示词，灵活适配不同场景。
6. **显式对话管理** — 提供更清晰的多任务编排与对话状态追踪，降低复杂交互的隐晦性。
7. **写入即校验** — 对格式严格的配置文件自动触发格式检查和警告，防止因模型幻觉产生错误。
8. **兼容多种 API** — 支持直接导入自定义配置，兼容 OpenAI、Anthropic 等多种 API 格式。
9. **快速兼容多家API Key** — 已覆盖测试验证的全部后端（kimi、anthropic、openai_legacy、openai_responses、google_genai、vertexai 等），详见 `kimi-cli\tests\core\test_create_llm.py`。

### 新增

| 能力 | 说明 |
|------|------|
| **交互式 Shell 工具** | 通过 `task_id` 启动并继续 `Bash`/`Powershell`/`Run` 会话，支持 `wait_for_pattern` 等待提示。 |
| **Docx / PDF 转换** | 内置文档格式转换，无需外部依赖。 |
| **Python 脚本执行** | 允许 Agent 直接执行 Python 脚本。 |
| **错误记录** | 记录工具调用错误，供模型回溯与改进。 |
| **脚本系统** | 将提示词与 Python 逻辑结合，编排复杂任务。 |
| **增强网页抓取 (fetch_url)** | 基于无头浏览器输出 Markdown（而非纯文本），支持 `output_path` 直接落盘与超长内容自动截断；零外部服务依赖，更稳更轻。 |
| **Best-of-N 采样（`AgentSwarm` `parallel_sample` 模式）** | 将同一任务在隔离工作区（git worktree / 临时拷贝）中并行跑 N 次，通过 `self_eval` 或 `majority` 策略选出最佳结果，应用胜出 diff 并执行验证——绝不静默接受失败。 |


### 脚本化工作流（核心优势）

与需要人工逐条输入命令的 CLI 交互不同，**Kimi-CLI-X 允许你直接编写 Python 脚本来编排整个工作流**。你可以将提示词、循环、条件判断和工具调用组合在一起，实现全自动、可复现的任务编排：

```python
from kimix import *
from pathlib import Path

# clear cli session, make an empty context.
clear_default_context()

for i in Path('docs').glob('*.md'):
    prompt(f'''
according to the new git commits, update document `{i}`
''')
```

这种方式的优势在于：

- **批量自动化**：结合 Python 的原生语法（如 `for` 循环、文件遍历），一次性向多个目标文件发起任务，无需人工等待和重复输入。
- **编排复杂流程**：在脚本中自由组合模式切换、工具调用与逻辑判断，构建多阶段、多分支的复杂工作流。
- **可复现与可维护**：工作流以脚本形式保存，可纳入版本控制，随时复用、修改和分享，而不是依赖临时的对话历史。

---

### 上下文记忆架构

Kimi-CLI-X 在 `KimiSoul` 核心循环中内建了一套**自动上下文记忆系统**，无需用户手动干预即可在长对话中保持连贯性。其核心由三层机制协同构成：

#### 1. 对话历史索引（HistoryIndex）

每条 user/assistant 消息在追加到上下文时自动被 **BM25 倒排索引**（N-gram, n=2）收录，持久化到 `<session>/history_index/<id>.json`，进程重启不丢失。索引上限 500 轮，超出后自动淘汰最旧条目。

#### 2. 上下文自动压缩（SimpleCompaction）

当上下文 token 占比触及 `compaction_trigger_ratio` 或剩余空间不足 `reserved_context_size` 时，自动触发压缩：

- **保留策略**：最近 N 轮对话原样保留（深度由 `adaptive_preserve_depth` 自适应决定——检测到错误、thinking、多文件编辑等信号时自动加深）；同时始终保留首条消息（首因效应）。
- **LLM 摘要**：旧消息通过一次轻量 LLM 调用压缩为结构化摘要，丢弃 thinking 部分。
- **级联处理**：当已压缩内容再次被压缩（深度 ≥3），自动切换为 `COMPACT_CASCADE` 提示词防止信息退化。
- 压缩后，所有轮次在 HistoryIndex 中标记为 `is_compacted`，供后续检索。

#### 3. 自动历史检索 + 按需召回

- **自动检索**（`_maybe_auto_retrieve_history`）：每轮第一步，若用户输入 ≥10 字符，自动在 HistoryIndex 中 BM25 搜索匹配的已压缩轮次，得分超过 `auto_retrieve_history_threshold` 时以 `[Auto-retrieved from past conversation]` 形式注入上下文。
- **`retrieve` 工具**：Agent 可主动调用，按自然语言查询搜索全部归档历史（含已压缩轮次），返回原文摘录及相关性得分（或按 `id` 取回指定轮次）。

```
┌──────────────┐    append     ┌──────────────┐    overflow    ┌──────────────────┐
│   Context    │ ───────────► │ HistoryIndex │ ────────────► │ SimpleCompaction │
│  (实时窗口)   │              │ (BM25 索引)   │               │  (LLM 摘要压缩)   │
└──────────────┘              └──────────────┘               └──────────────────┘
       ▲                            │                               │
       │       auto-retrieve        │                               │
       └────────────────────────────┘                               │
       │              Retrieve (Agent 主动召回)                       │
       └────────────────────────────────────────────────────────────┘
```

---

### Agent 自律与提醒框架

`KimiSoul` 核心循环会主动让长任务保持在正轨上，无需人工干预。CLI、Server 与子代理会话中均生效。

- **验证门**：存在未完成 todo、或修改了文件却没运行任何检查时，回合不允许结束，失败信息会反馈给 Agent 继续修复。
- **防循环检测**：识别同一文件被不同工具反复修改、同一错误反复出现而未修复根因等情况，提醒 Agent 更换策略。
- **Todo 提醒**：定期将未完成的 todo 重新注入上下文尾部，目标不会因对话变长而“丢失”。
- **压缩提醒**：上下文将满（约 70%）时，提示 Agent 主动压缩，避免被动触发自动压缩丢失信息。
- **预算提醒**（可选开启）：回合步数/时间预算将尽时分级提醒收尾，让 Agent 体面收尾而非被强行中断。
- **上下文计量**：上下文用量明显变化时，提醒 Agent 用 `retrieve` 工具回溯历史。
- **决策感知压缩**：压缩摘要保留 `Decisions & Conclusions` 与 `Verification Status` 两节，早期决策与已验证的工作不丢失。
- **上下文剪枝**：自动清理过期工具输出、thinking 块与近似重复内容，回收上下文空间。

### 可执行验证的 TodoList

### TodoList
`TodoList` 工具用于跟踪多步计划：
- 支持增量更新（append/overwrite 模式）、标题模糊匹配与逐项备注。
- 通过 `todo_push`/`todo_pop` 与 `todo_update(parent=...)` 支持嵌套子任务，并以 `Stack:` 面包屑展示当前层级。

### Best-of-N 采样

`AgentSwarm` 的 `parallel_sample` 模式将**同一任务**在隔离工作区（git worktree / 临时拷贝）中并行运行 N 次，经模型自评或多数投票选出最佳结果，应用胜出 diff 并验证。失败显式报错——绝不静默接受。

---

## 文档索引

### 教程系列

| 文档 | 简介 |
|------|------|
| [`docs/tutorials/1_quick_start.md`](docs/tutorials/1_quick_start.md) | **快速入门指南**。涵盖 Git Submodule 拉取、`uv` 环境安装、CLI 启动参数与交互命令的完整说明。 |
| [`docs/tutorials/2_long_task.md`](docs/tutorials/2_long_task.md) | **Long Task**。KimiX 对于长任务的策略。 |
| [`docs/tutorials/3_builtin_tools.md`](docs/tutorials/3_builtin_tools.md) | **内置工具完全指南**。系统介绍 Agent 的全部内置工具（文件 I/O、搜索、代码执行、进程管理、文档转换、计划模式、子代理等），并给出提示词引导策略与最佳实践。 |
| [`docs/tutorials/4_skills.md`](docs/tutorials/4_skills.md) | **自定义 Skill 编写教程**。讲解 Skill 的设计原则、目录结构、`SKILL.md` 编写规范、附属资源组织方式、测试打包流程及安装使用方法。 |
| [`docs/tutorials/5_server.md`](docs/tutorials/5_server.md) | **JSON-RPC 服务端教程**。介绍基于 TCP 的 JSON-RPC 2.0 协议格式、错误码、服务端接口、WebSocket 桥接及命令行启动参数。 |
| [`docs/tutorials/6_multi_provider.md`](docs/tutorials/6_multi_provider.md) | **多 Provider 配置**。通过带 `role` 标签的 `sub_providers` 将子代理和 Planner 路由到不同 LLM Provider。 |

### 配置参考

| 文件 | 简介 |
|------|------|
| [`docs/config.json`](docs/config.json) | 模型配置示例文件，包含 `model`、`url`、`api_key`、`capabilities` 等字段，可供编写自定义配置时参考。 |
| [`.kimix/config.json`](.kimix/config.json) | 工作区行为配置文件，包含 `protected_write_paths`、`protected_read_paths`、`forbidden_commands` 等字段，可限制当前工作目录下 Agent 的读写与执行权限。 |
| [`.kimix/skill.json`](.kimix/skill.json) | 工作区 Skill 目录配置文件，通过 `skill_dir` 字段（字符串或字符串数组）为当前工作目录指定额外的 Skill 目录，相对路径基于工作目录解析。 |


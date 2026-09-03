# Kimix 快速入门指南

本文档将带你完成 Kimix 的环境准备、安装以及 CLI 的基本使用。

---

## 一、安装

在项目根目录（包含 `pyproject.toml` 的目录）直接运行：

```bash
python install.py
```

`install.py` 会自动完成：依赖同步、原生运行时下载、命令行工具注册等初始化工作。安装完成后即可使用 `kimix` 命令。

---

## 二、环境变量配置

在运行 Kimix 之前，需要配置 API 密钥。优先使用 JSON 配置文件中的 `api_key` 字段，若未配置则依次读取以下环境变量（代码逻辑参考 `src/kimix/utils/config.py`、`src/kimix/cli_impl/init.py`）：

### API 密钥环境变量

| 变量名 | 说明 |
|--------|------|
| `KIMI_API_KEY` | Kimi API 的访问密钥 |
| `KIMIX_API_KEY` | 备选密钥变量名，优先级低于 `KIMI_API_KEY` |

### 其他环境变量

除 API 密钥外，其他模型参数（URL、模型名、上下文长度等）均通过 JSON 配置文件管理，不再通过环境变量设置。详见下方「3.2 初始化 LLM 配置」。

**示例（Linux / macOS）：**

```bash
export KIMI_API_KEY=your-api-key
```

**示例（Windows PowerShell）：**

```powershell
$env:KIMI_API_KEY="your-api-key"
```

---

## 三、CLI 基本用法

Kimix 的命令行接口分为「子命令」「启动参数」和「交互命令」三部分，以下内容整理自 `src/kimix/cli_impl/`。

### 3.1 子命令

除默认的交互式客户端外，`kimix` 还支持以下子命令：

| 子命令 | 说明 | 常用选项 |
|--------|------|----------|
| `serve` | 启动 Kimix HTTP 服务器（OpenCode 风格） | `--host`/`--hostname`（默认 `127.0.0.1`）、`--port`（默认 `4096`） |
| `gui` | 启动 Kimix 后端 + TypeScript/Vite 前端开发服务器 | `--host`/`--hostname`（默认 `127.0.0.1`）、`--port`（后端端口，默认 `4096`）、`--fe-port`（前端端口，默认 `5173`）、`--build`（启动前先执行 `npm run build`）、`--no-fe`（仅启动后端，跳过前端；适合未安装 Node.js/npm 的环境） |
| `ssecli` | 启动 SSE CLI 调试器，连接 `kimix serve` 进行交互式测试。内部支持 `/help`、`/new`、`/abort`、`/status`、`/sessions`、`/messages`、`/clear`、`/export[:path]`、`/compact[:N]` 等命令；按 `Ctrl+C` 或输入 `EOF`（`Ctrl+D` / `Ctrl+Z`）退出 | `--host`、`--port`、`--debug`（保存原始事件日志为 `sse_log_<YYYYMMDD_HHMMSS>.txt`） |
| `mcp` | MCP（Model Context Protocol）服务器管理：`mcp serve`（将 Kimix 作为 MCP 服务器对外提供服务）、`mcp list`（列出已配置的服务器）、`mcp test <name>`（测试连接） | 详见「四、MCP」 |

**示例：**

```bash
# 启动 HTTP 服务
kimix serve --port 4096

# 启动 GUI（后端 + 前端开发服务器）
kimix gui --port 4096 --fe-port 5173

# 仅启动 GUI 后端（本机未安装 Node.js/npm 时）
kimix gui --no-fe

# 使用 SSE CLI 调试
kimix ssecli --host 127.0.0.1 --port 4096 --debug

# 管理 MCP 服务器
kimix mcp list
kimix mcp test my-server
```

### 3.2 初始化 LLM 配置

Kimix 通过 JSON 配置文件初始化 LLM Provider。若启动时未通过 `--config` 指定自定义配置，将自动加载 `src/kimix/default_config.json`（由 `/init` 生成）；若该文件不存在，则回退到内置的默认配置模板（模型 `kimi-for-coding`，见 `src/kimix/cli_impl/init.py`）。

如果默认配置文件不存在，首次启动时会自动提示是否进行初始化；你也可以在交互终端中随时执行 `/init`，按提示逐项填写模型名称、类型、API Key、上下文长度、最大 token 数、思考力度（thinking effort）、模型能力（capabilities）、URL，以及可选的子 Agent Provider（sub_provider）等参数，配置将自动保存至 `src/kimix/default_config.json`：

```
/init
```

示例中的模型名称为 `kimi-for-coding`，`kimi-for-coding-highspeed` 同样受支持，可根据需要选择。配置采用扁平结构（即 `src/kimix/default_config.json` 的实际格式）：

```json
{
    "model": "kimi-for-coding",
    "max_context_size": 1048576,
    "capabilities": ["thinking", "image_in"],
    "url": "https://api.kimi.com/coding/v1",
    "type": "kimi",
    "api_key": "your-api-key",
    "max_tokens": 131072,
    "show_thinking_stream": true,
    "thinking_effort": "max",
    "temperature": 1.0,
    "loop_control": {
        "max_steps_per_turn": 5000,
        "max_retries_per_step": 3,
        "max_ralph_iterations": 0,
        "reserved_context_size": 50000,
        "compaction_trigger_ratio": 0.85
    },
    "sub_providers": [
        {
            "role": "sub_agent",
            "model": "kimi-for-coding",
            "max_context_size": 1048576,
            "capabilities": ["thinking"],
            "url": "https://api.kimi.com/coding/v1",
            "type": "kimi",
            "api_key": "your-api-key"
        }
    ]
}
```

你也可以创建自定义配置文件并通过 `kimix --config=<path>` 加载。配置字段说明如下：

| 字段 | 必填 | 说明 |
|------|------|------|
| `type` | 是 | Provider 类型，可选值见下文「[支持的 Provider](#支持的-provider)」（如 `kimi`、`openai_legacy`、`anthropic`） |
| `model` | 是 | 实际请求的模型名称 |
| `url` | 是 | API 基础地址 |
| `max_context_size` | 是 | 最大上下文长度（token 数），可选 `128k`、`200k`、`256k`、`512k`、`1M` |
| `capabilities` | 否 | 模型能力列表，可选值：`thinking`、`always_thinking`、`image_in`、`video_in`。如 `["thinking"]` |
| `api_key` | 否 | API 密钥。若省略，将依次读取环境变量 `KIMI_API_KEY`、`KIMIX_API_KEY` |
| `custom_headers` | 否 | 自定义 HTTP 请求头 |
| `oauth` | 否 | OAuth 配置，例如 `{"storage": "file", "key": "my-key"}` |
| `loop_control` | 否 | 循环控制参数，含 `max_steps_per_turn`、`max_retries_per_step`、`max_ralph_iterations`、`reserved_context_size`、`compaction_trigger_ratio` |
| `max_tokens` | 否 | 单次请求最大生成 token 数 |
| `show_thinking_stream` | 否 | 是否流式展示思考过程 |
| `thinking_effort` | 否 | 思考力度，可选 `off`、`low`、`medium`、`high`、`xhigh`、`max` |
| `temperature` | 否 | 采样温度，范围 `[0.0, 2.0]` |
| `sub_provider` / `sub_providers` | 否 | 子 Agent 使用的 Provider。`sub_provider` 为单个 dict，`sub_providers` 为 dict 数组；每个条目需包含 `type`、`max_context_size`、`model`、`url`，可选 `role`（`sub_agent` 或 `planner`，缺省 `sub_agent`）及 `api_key` 等。当顶层缺少 `model` 时，会按优先级（无 `role` > `sub_agent` > `planner`）从子 Provider 中挑选一个作为主 Provider |
| `background` | 否 | 后台任务相关配置 |
| `notifications` | 否 | 通知配置 |
| `mcp` | 否 | MCP (Model Context Protocol) 配置 |
| `env` | 否 | 启动时注入的额外环境变量（dict） |

### 支持的 Provider

`type` 接受以下 Provider 标识（分组与 `kimi_cli/llm.py` 保持一致）：

**核心 Provider**

| `type` | 说明 |
|--------|------|
| `kimi` | Moonshot / Kimi |
| `xai` | xAI（Grok） |
| `openai_legacy` | OpenAI Chat Completions |
| `openai_responses` | OpenAI Responses API |
| `anthropic` | Anthropic Claude |
| `google_genai` | Google GenAI（`gemini` 的旧别名） |
| `gemini` | Google Gemini（Google AI Studio） |
| `vertexai` | Google Vertex AI（`vertex` 的旧别名） |
| `vertex` | Google Vertex AI |

**OpenAI 兼容 Provider（Hermes 移植）**

`ai-gateway`、`alibaba`、`alibaba-coding-plan`、`arcee`、`azure-foundry`、`copilot`、`custom`、`deepinfra`、`deepseek`、`fireworks`、`gmi`、`huggingface`、`kilocode`、`kimi-coding`、`nous`、`novita`、`nvidia`、`ollama-cloud`、`opencode-zen`、`openrouter`、`qwen-oauth`、`stepfun`、`upstage`、`xiaomi`、`zai`

**特殊模式 Provider**

| `type` | 说明 |
|--------|------|
| `actual` | Actual Computer（Codex 风格 API） |
| `bedrock` | AWS Bedrock（Converse API） |
| `minimax` | MiniMax（兼容 Anthropic） |
| `openai-codex` | OpenAI Codex（ChatGPT 后端） |
| `copilot-acp` | GitHub Copilot ACP 子进程（外部 Agent，无进程内 LLM） |

使用 ChatGPT 订阅可参考 [`openai_codex.json`](../openai_codex.json)。在项目根目录先通过浏览器登录，再使用样例启动：

```bash
uv run kimi login codex
uv run kimix --config=docs/openai_codex.json
```

样例中的 `api_key` 值 `oauth-managed` 是占位值，实际令牌由 `oauth/openai-codex` 引用的共享凭据提供，并在请求前自动刷新。可将 `model` 改为账号可用的 Codex 模型；如设置了 `KIMI_SHARE_DIR`，登录和运行时须使用同一目录。

退出并清除共享凭据：

```bash
uv run kimi logout codex
```

> 当省略 `api_key` / `url` 时，Provider 注册表会回退到各 Provider 的标准环境变量（如 `DEEPSEEK_API_KEY`、`OPENROUTER_API_KEY`、`XIAOMI_API_KEY`、`GLM_API_KEY`/`ZAI_API_KEY`、`MINIMAX_API_KEY`）。完整的逐 Provider 环境变量列表见 `kimi-cli/packages/kosong/src/kosong/providers/__init__.py`。

**自定义配置示例（参考 `docs/anthropic.json` 等）：**

```json
{
    "model": "minimax-m2.7",
    "max_context_size": 204800,
    "capabilities": ["thinking"],
    "url": "https://api.minimaxi.com/anthropic",
    "type": "anthropic",
    "api_key": "your-api-key",
    "custom_headers": {},
    "oauth": {
        "storage": "file",
        "key": "my-key"
    }
}
```

### 3.3 启动参数

在启动 `kimix` 时，可附加以下选项来控制行为：

| 参数 | 说明 |
|------|------|
| `-c`, `--clean` | 退出时自动删除缓存文件 |
| `--no_think` | 关闭思考模式（thinking mode） |
| `--no_yolo` | 关闭 YOLO 模式 |
| `--no_color` | 关闭彩色输出 |
| `--manually-cot` | 开启手动 CoT 模式（可能使用多个会话并消耗额外 token） |
| `--ralph` | 开启 Ralph 模式，可指定迭代次数（不传参数则设为 1） |
| `-s`, `--skill-dir` | 指定自定义的 skill 目录（可多次使用以指定多个目录） |
| `--config` | 指定 JSON 格式的配置文件路径。若直接路径不存在，会依次在当前工作目录的各级父目录中递归查找、在 kimix 安装目录的各级父目录中递归查找，最后在系统 `PATH` 中查找同名文件（格式可参考 `docs/*.json` 示例） |

**示例：**

```bash
kimix --clean --manually-cot
```

> **Skill 目录的自动加载**：启动时，Kimix 还会读取当前目录下的 `.kimix/skill.json` 文件。如果其中包含 `skill_dir` 字段（字符串或字符串数组），且对应目录存在，这些目录会被自动追加到默认 skill 搜索路径中。

### 3.4 交互命令

进入 Kimix 交互式终端后，可通过以下命令与 Agent 交互：

| 命令 | 说明 |
|------|------|
| `<path>` | 直接输入文件路径即可加载。`.py` 文件会直接执行（执行时 `__file__` 变量指向该文件）；其他文件会读取全部内容作为单条提示词发送 |
| `/file:<path>` | 读取指定文件的全部内容作为单条提示词发送 |
| `/todo:<path>` | 扫描代码文件中的 TODO 注释，并提示 Agent 实现。支持 `.py`；C 系（`.c/.cpp/.cc/.cxx/.h/.hpp/.java/.js/.ts/.jsx/.tsx/.cs/.go/.rs`）；Shell（`.sh/.bash/.zsh`）；HTML/XML（`.html/.htm/.xml/.svg`）；Pascal（`.pas/.pp/.inc/.dpr`）；Lisp（`.lisp/.lsp/.clj/.scm/.ss/.el`）；SQL（`.sql`） |
| `/clear` | 清空当前对话上下文 |
| `/sessions` | 列出当前工作目录下可恢复的会话（含更新时间与上下文占用率，`*` 标记当前会话）；`/sessions:<name>` 创建并切换到新的命名会话 |
| `/exit` | 退出程序 |
| `/help` | 显示帮助信息 |
| `/context` | 打印当前上下文的使用情况 |
| `/fix:<command>` | 运行一条命令，如果出错则自动尝试修复 |
| `/txt` | 进入多行文本输入模式（以 `/end` 结束，`/cancel` 取消），内容加入输入队列，可随后批量发送 |
| `/init` | 交互式初始化默认 LLM 配置文件（执行后会重置当前会话） |
| `/compact` | 压缩当前会话的对话上下文 |
| `/export:<path>` | 将当前会话的消息导出到指定文件 |
| `/resume:<id>` | 关闭当前会话并按 ID 恢复已有会话 |
| `/store:<id>` | 将当前会话复制为一个新的命名会话 |
| `/load:<id>` | 将指定命名会话复制到一个新的匿名会话并切换 |
| `/ralph:on` / `/ralph:off` / `/ralph:<num>` | 设置 Ralph 模式循环次数 |
| `/reflection` | 反思当前对话上下文，找出由当前 Agent 设计导致的误解，并修改源代码以改进项目（需要非空上下文；完成后将变更报告写入 `docs/reflection_report_*.md`） |
| `/supervisor` | 进入多行输入模式，以 Supervisor 角色创建会话并执行一次任务（以 `/end` 结束，`/cancel` 取消） |
| `/plan` / `/plan:<file>` | 使用 TodoMaker Agent 生成任务计划。任务需求通过多行输入提供（以 `/end` 结束）；`<file>` 用于指定计划输出文件路径，若该文件已存在会被覆盖。生成后支持用户审阅、修改，确认后再执行，执行后会追加一次 review 提示 |
| `/swarm` | 进入多行输入模式，创建 Swarm 会话（SwarmLeader 角色）将请求并行分发给多个同质子 Agent（以 `/end` 结束，`/cancel` 取消） |
| `/cmd:<command>` | 执行系统命令 |
| `/code:<path> [args...]` | 运行脚本文件（支持 `.py` 和其他可执行文件），可附带参数 |

除上述命令外，你也可以直接输入任意自然语言提示词（prompt）发送给 Agent 进行处理。


---

## 四、MCP（Model Context Protocol）

Kimix 同时支持作为 MCP 客户端和 MCP 服务器使用。

### 使用 MCP 服务器

将外部 MCP 服务器添加到 Kimix，使其工具、资源和提示词对 Agent 可用：

MCP 服务器通过 JSON 配置文件注册（CLI 提供 `mcp list` 与 `mcp test`，不提供 `mcp add` 命令）：

- **全局配置**：`~/.kimi/mcp.json`
- **项目配置**：`.kimix/mcp.json`（可纳入版本控制，启动时由 `src/kimix/cli_impl/args.py` 自动加载并合并进会话）

两种配置中的 `mcpServers` 对象会被自动合并，优先级为：显式 > 项目 > 全局。stdio 服务器使用 `command` + `args` 描述，streamable HTTP 服务器使用 `url` + `transport` 描述：

```json
{
    "mcpServers": {
        "my-stdio-server": {
            "command": "npx",
            "args": ["-y", "@example/mcp-server"]
        },
        "my-http-server": {
            "url": "https://api.example.com/mcp",
            "transport": "streamable-http"
        }
    }
}
```

列出与测试已配置的服务器：

```bash
# 列出已配置的服务器
kimix mcp list

# 测试连接并列出其工具
kimix mcp test my-server
```

### 将 Kimix 作为 MCP 服务器对外提供服务

将当前 Kimix 运行时暴露给外部 MCP 客户端（如 Claude Desktop、Cursor 等）：

```bash
# stdio 传输（适合由客户端启动子进程）
kimix mcp serve --transport stdio

# streamable HTTP 传输
kimix mcp serve --transport http --host 127.0.0.1 --port 4097
```

`mcp serve` 支持的选项：`--transport`（`stdio` / `http`，默认 `stdio`）、`--host`（默认 `127.0.0.1`）、`--port`（默认 `4097`）、`--work-dir`（工作目录，默认当前目录）、`--agent-file`（指定 Agent 配置文件）、`--no-resource`（不暴露文件资源）、`--no-prompt`（不暴露提示词）。

默认情况下，MCP 服务器会暴露：

- **tools**：当前 Agent 工具集中的所有工具
- **resources**：`AGENTS.md`、`README.md` 以及工作目录下的项目文件
- **prompts**：Agent 的系统提示词

使用 `--no-resource` 或 `--no-prompt` 可分别禁用资源或提示词；使用 `--agent-file` 可加载指定的 Agent 配置文件。


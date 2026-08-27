# kimix-gui

KimiX 内置的纯 Python 桌面客户端。聊天循环通过 Kimix 公开的异步
Worker 会话工厂创建与恢复会话；启动时的历史会话列表按 kimi-cli 的
工作目录存储规则扫描。

## 当前能力

- 无 `--session` 时先进入主页，可预览、恢复已有会话或新建
- 按标题搜索会话，支持单选、多选当前结果和确认后批量删除
- 恢复会话时按 turn 分页回放 `wire.jsonl` 中的对话（用户/回复/思考/工具），到达顶部可继续加载更早记录，也可输入 turn 编号直接跳转
- 使用 `kimix.create_session_async()` 创建或恢复完整的 Kimix Worker 会话
- 增量显示回复和思考内容；AI 回复支持标题、加粗、代码等简单 Markdown
- 展示工具调用、工具结果、步骤和上下文状态
- 聊天列表右上角挂一个可收缩的 TODO 面板：`todo_write` / `todo_update` 写入后立即更新，每轮结束和会话恢复时以 `state.json` 为准刷新，`Ctrl+T` 收起或展开
- 审批对话框支持批准一次、对当前会话批准或拒绝
- 支持 SDK 的结构化问答请求
- 管理多个 LLM 配置，为新会话和每个历史 session 选择独立配置，并保存脱敏引用
- 可在 Settings → Models 通过浏览器 OAuth 连接 ChatGPT，使用订阅内可用的 Codex 模型；登录成功只刷新模型目录，不自动改变项目默认模型
- `Ctrl+G` 取消当前生成
- `/clear`、`/compact`、`/status`、`/help`、`/quit`（`/quit` 回到主页，不结束进程）
- 大历史会话先加载最近 4 个 turn / 32 个展示块；顶部加载使用一次索引扫描和按区间读取，最多在连续滚动窗口保留 64 个 turn，窗口满后可用 `Earlier` / `Later` 翻页，避免滚动条和内存无限增长

## 启动

安装发行包时启用 `gui` extra，确保 Kimix 已完成模型配置，然后：

```powershell
pip install "kimix[gui]"
kimix-gui
kimix-gui --work-dir C:\path\to\project
```

从 KimiX 源码开发时共用仓库根部的 uv 环境，不需要 submodule 或第二个虚拟环境：

```powershell
uv sync
uv run kimix-gui
uv run kimix-gui --work-dir C:\path\to\project
```

`uv sync` 默认安装 dev dependency group，其中包含 PySide6 与 pytest-qt；发行用户只有显式安装
`kimix[gui]` 才会获得 Qt。`kimix`、`kimix_gui`、`kimi_agent_sdk`、`kimi_cli`、`kosong` 和
`kaos` 都直接加载同一工作区源码。

不传 `--session` 时会在主页按更新时间倒序列出当前工作目录下的非空历史会话；高亮或单击会话可查看大小、存储格式、更新时间、待办等详情，按 Enter 或点击 **Open session** 进入。传入 `--session` 则跳过首次主页，直接进入该会话；从会话返回后仍会打开主页：

```powershell
uv run kimix-gui --work-dir C:\path\to\project --session my-session
```

指定模型或启用自动审批：

```powershell
uv run kimix-gui --model kimi --yolo
```

也可以使用与 Kimix CLI 相同的扁平 provider JSON 启动：

```powershell
uv run kimix-gui --config=C:\path\to\provider.json
```

### ChatGPT / Codex subscription

在 **Settings → Models** 点击 **Connect ChatGPT**，浏览器会打开 OpenAI 登录页。完成登录后，
浏览器通过仅监听本机的临时回调返回 Kimix；无需复制设备代码。连接成功后，内建 ChatGPT 模型会
出现在 LLM Settings 的独立分组中；选择模型与变体后仍需点击 **Use model** 才会应用到项目或
session。

OAuth 现在属于 Kimix 核心认证层：GUI 与 `kimi login codex` 共用
`KIMI_SHARE_DIR/openai-codex-state.json`，不会读取 Codex CLI 的私有缓存，也不会把 token 写入
provider JSON、GUI 配置或 session metadata。GUI 的 **Disconnect** 与 `kimi logout codex` 都会
删除共享登录状态；已保存的 ChatGPT 模型引用会保留，并在重新登录后恢复。
登录使用 Authorization Code + PKCE 的本机浏览器回调。该能力仍依赖非公开的 OAuth client/backend
endpoint；上游变化会显示为登录或模型目录错误，不会影响外部 provider JSON。

也可以先在终端完成同一套登录，再打开 GUI：

```powershell
uv run kimi login codex
uv run kimix-gui
```

ChatGPT 模型能力按 Codex 目录逐模型解析：每个模型只占左侧一行，右侧 **Variant** 下拉框按目录
顺序列出可选 reasoning effort。已知值显示为“中等（medium）”一类本地化标签；未知的新值原样
显示并透传。`ultra` 规范化为 `max` 并去重，`off` 不作为 ChatGPT reasoning effort，目录中的
`none` 则允许选择。带“模型默认”标记的选项只说明目录默认值；点击 **Use model** 时仍保存明确
强度，不保存“以后跟随默认”。目录后来移除已保存强度时，原值会显示为不可用并阻止启动，绝不
静默切换到其他强度。

目录缺字段或离线回退时，当前官方 Codex 基线为 272,000 context 与 128,000 max output 元数据；
ChatGPT Codex subscription backend 不接受显式输出上限，因此该数值不会作为请求参数发送。
GUI 只在自己的 ChatGPT 订阅适配边界固定选中的 reasoning effort，不修改 Kimix CLI 内部既有的
`thinking` 推导或其他 Provider 行为。

主页顶部 **Settings** 修改 work dir 的新会话默认选择。Provider files 分组内部可以输入与
`kimix --config=...` 相同的外部 JSON 路径；**Add** 校验并加入全局文件库，不改变当前绑定；选中
模型后，**Use model** 才会应用到当前作用域。历史会话详情中的 **Configure** 修改该 session
下次恢复时使用的选择；独立的 **Follow project default** 复选框会删除该 session 的覆盖，使其
持续跟随工程默认。聊天页按 `F4` 只读查看启动时的已解析选择；目录刷新或设置修改不会替换正在
运行的 LLM。

LLM Settings 将 **ChatGPT subscription** 与 **Provider files** 分成两个带数量的可折叠来源分组，
默认只展开当前模型所属的分组。模型行只在实际超出宽度时由列表视图省略，完整文本保留在 tooltip；
每行和右侧详情都会明确显示来源。ChatGPT 未连接时，连接入口只出现在订阅分组中；添加 JSON
文件的路径、浏览和添加控件只属于 Provider files 分组。Provider 文件只有一个不可编辑的
`configured` Variant，仍完全服从文件内容。

Provider JSON 可以加入配置库后再补凭据，但在 `api_key`、OAuth、`KIMI_API_KEY` / `KIMIX_API_KEY`
或该 Provider 的标准 API Key 环境变量均不可用时会标记为 **API key missing**，并禁用
**Use model**。配置检查只形成界面状态，不再在 GUI 启动时输出缺少 API Key 的运行时警告。

选择优先级为 session 覆盖高于 work dir 默认。工程 Settings 可以添加或移除文件库引用，但不能
移除正在使用的工程默认；移除引用不会删除 Provider JSON 文件。session Settings 不提供添加或
删除操作。全局 `KIMI_SHARE_DIR/kimix-gui.json`（默认 `~/.kimi/kimix-gui.json`）使用 version 5，
统一保存界面偏好、Provider 文件绝对路径和各 work dir 的明确 `target + variant`；每个 session
目录自己的 version 3 `kimix-gui.json` 保存相同选择结构。这些文件不保存 OAuth token、Provider
参数或 API Key。若 Provider 文件、ChatGPT 模型或明确变体不可用，界面保留原选择并禁止进入，
要求用户重新选择。

建议以桌面窗口运行。主页支持鼠标预览和打开会话。

## 窗口结构

- `KimixGuiApp` 只负责在 Home / Chat 之间路由，并拥有统一 GUI 配置与 Kimix worker 线程
- `HomeView` 和 `ChatView` 是主窗口里的完整页面（`QStackedWidget`）
- `LLMSettingsDialog`、`ApprovalDialog` 和 `QuestionDialog` 是不阻塞 GUI 线程的模态对话框（`open()`，不用 `exec()`）
- 页面位于 `kimix_gui/qt/`；`KimixBridge` 在独立线程跑 asyncio，SDK 对象不进入 GUI 线程
- 聊天记录由虚拟化的 `QListView` 绘制，历史按 Timeline 滑窗替换

## 快捷键

| 键 | 行为 |
|---|---|
| `Enter` | 主页打开高亮会话；聊天输入框发送；Compose 换行 |
| `Ctrl+Enter` / `Shift+Enter` | 聊天输入框换行。Compose 中 `Ctrl+Enter` 不发送也不换行，需点 Send |
| `n` | 主页新建 session |
| `Space` | 主页勾选 / 取消勾选高亮的会话 |
| `Delete` | 主页删除已勾选的会话（无勾选时删除高亮项） |
| `Ctrl+F` | 主页聚焦搜索框 |
| `q` / `Esc` | 主页退出进程 |
| `Esc` | 聊天中关闭当前会话并回到主页 |
| `Ctrl+G` | 取消当前生成 |
| `Ctrl+↑` | 加载更早聊天记录 |
| `Ctrl+End` | 跳到最新聊天记录 |
| `F2` | 聚焦输入框 |
| `F3` | 聚焦历史 turn 跳转框 |
| `Ctrl+T` | 收起 / 展开 TODO 面板 |
| `F4` | 打开 LLM Settings |
| `a` | 审批弹窗中批准一次 |
| `s` | 审批弹窗中对本会话批准 |
| `r` / `Esc` | 审批弹窗中拒绝 |

## 原型边界

- 进入已有会话时通过 `wire.jsonl` 的 turn 偏移索引分页回放历史；公开 SDK 仍无通用 history list 接口。
- 未实现文件 diff 专用视图和多 Agent 视图。

## 开发验证

架构、界面、国际化与测试约定见 [`README.md`](README.md)。

```powershell
uv run pytest -q
```

大文本会话的渲染压测（不会创建 1-2 GiB 的持久文件，按批次生成等效文本量）：

```powershell
uv run python scripts/gui/benchmark_transcript.py --gigabytes 1
uv run python scripts/gui/benchmark_transcript.py --gigabytes 2
```

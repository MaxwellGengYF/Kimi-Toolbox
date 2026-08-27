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
出现在 LLM Settings 的独立分组中；选择模型后仍需点击 **Use config** 才会应用到项目或 session。

OAuth 凭据独立保存在 `KIMI_SHARE_DIR/kimix-gui-codex-auth.json`，不会读取、导入或共享
Codex CLI 的凭据，也不会写入 provider JSON、GUI 配置或 session metadata。断开连接只删除这个
本地凭据文件；已保存的 ChatGPT 模型引用会保留，并在重新登录后恢复。该集成沿用 Hermes 的独立
凭据与 ChatGPT Codex backend 路线，并把登录阶段改为 Authorization Code + PKCE 的本机浏览器回调。
它仍依赖非公开的 OAuth client/backend endpoint；上游变化会显示为登录或模型目录错误，不会影响
外部 provider JSON。

ChatGPT 模型能力按 Codex 目录逐模型解析：使用 Codex 的 active context window、目录给出的默认与
可选 reasoning effort，并保留服务器排列顺序。目录缺字段或离线回退时，当前官方 Codex 基线为
272,000 context 与 128,000 max output 元数据；ChatGPT Codex subscription backend 不接受
显式输出上限，因此该数值不会作为请求参数发送。普通会话使用该模型的默认 effort，命令行显式传入
`--thinking` 时使用目录中最高的直接推理强度。Codex 的 `ultra` 还包含客户端自动委派，Kimix 的
推理请求按官方做法将其映射为 `max`，子 Agent 编排仍由 KimiX 自己负责。

主页顶部 **Settings** 修改 work dir 的新会话默认配置；可在 **Kimix provider config (.json)** 输入与 `kimix --config=...` 相同的外部 JSON 路径。**Add config** 校验并加入全局配置库，不改变当前绑定；从配置库选择后，**Use config** 才会应用到当前作用域。历史会话详情中的 **Configure** 修改该 session 下次恢复时使用的配置。聊天页按 `F4` 修改当前 session 的下次恢复配置，不会替换正在运行的 LLM。

配置优先级为 session 配置高于 work dir 默认配置。工程 Settings 可以添加或移除配置库引用，但不能移除正在使用的工程默认；移除引用不会删除 provider JSON 文件。配置单个 session 时只能从现有配置中选择，也可以选择 **Project default**；session Settings 不提供添加或删除操作。选择工程默认会删除该 session 的独立配置引用，使它持续跟随工程默认。全局 `KIMI_SHARE_DIR/kimix-gui.json`（默认 `~/.kimi/kimix-gui.json`）统一保存界面偏好、外部配置文件绝对路径列表和各 work dir 的 tagged LLM 默认值；每个 session 的 LLM 引用保存在 Kimi session 目录自己的 `kimix-gui.json`。这些引用文件不保存 OAuth token、Provider 参数或 API Key。若引用的 provider JSON 丢失，界面显示其路径和 Missing，禁止进入并要求先重新配置。

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

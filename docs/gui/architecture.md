# 架构与模块边界

kimix-gui 是 KimiX distribution 内的纯 Python 桌面客户端：PySide6 负责界面，Kimi Agent SDK
负责会话循环，不依赖 Rich / Textual。它与 `kimix`、`kimi_agent_sdk` 一起位于根 `src/`，要求
Python >= 3.14，并由仓库根部的 uv 环境统一管理。

根 `pyproject.toml` 的 Hatch wheel 同时打包 `src/kimix`、`src/kimi_agent_sdk` 与
`src/kimix_gui`。GUI 直接复用同一 distribution 和 uv workspace 内的 `kimi_cli` / `kosong` /
`kaos` 源码，不再使用 submodule、path dependency 或第二个虚拟环境。PySide6 通过 `gui` extra
对发行用户保持可选，但位于默认 `dev` dependency group 中，保证源码检出的测试环境完整。

## 分层

`src/kimix_gui/` 顶层是**与 Qt 无关的纯逻辑**，唯二例外是 `app.py`（路由，持有
`QApplication`）和 `__main__.py`（入口）。全部 PySide6 控件代码在 `src/kimix_gui/qt/`。
新增逻辑优先放纯层，这样能用普通 pytest 覆盖，不必起 Qt。
纯层**绝不 import PySide6（含 QtCore）**，所以纯层也**不能调 `tr()`**（它是 `QObject`
方法）；翻译边界见 [`i18n.md`](i18n.md)。

| 模块 | 职责 |
|---|---|
| `app.py` | `KimixGuiApp`：home / chat 路由，持有 bridge 与统一 GUI 配置存储 |
| `backend.py` | `SessionOptions` + `create_sdk_session()`，包装 `kimix.create_session_async()` |
| `design/` | 与框架无关的设计 token：`palette.py` / `categories.py` / `scale.py` / `theme.py`（`DARK`） |
| `i18n.py` | 纯函数 `resolve_language(preference, system_locale)` + 支持语言常量 |
| `preferences.py` | `InterfacePreferences`（字体 / 语言 / 主题）及其纯规范化/序列化函数 |
| `transcript_data.py` | transcript 语义 AST、`HistoryEntry`、类型化 mutation/reducer、纯 formatter 与递归 `entry_cost()` |
| `rendering.py` | `WireNormalizer`：SDK message → AST mutation；只在此边界解析工具参数；状态条独立产出 `StatusValues` |
| `tool_display.py` | 工具家族分类，并把已解析 arguments / result 规范化为结构化 call/result content |
| `transcript_layout.py` | AST → `RecordLayout`（结构化 header runs / body sections）；只组合，不解析 JSON 或展示文本 |
| `history.py` | `wire.jsonl` 外层校验 → 共用 normalizer/reducer → 按正文/record 双预算滑动的 AST window（`Timeline`） |
| `session_index.py` | 按 kimi-cli 存储规则列出/删除历史会话，产出 `SessionSummary` |
| `todos.py` | 读 `state.json` 与 `TodoDisplayBlock`，产出 `TodoSnapshot` |
| `llm_config.py` | provider JSON 的脱敏引用库 + `KimixGuiConfigStore`（全局与 session 的 `kimix-gui.json`）|
| `kimi_workdir.py` | Windows 盘符大小写与 Kimi metadata 对齐 |
| `qt/bridge.py` | `KimixBridge`：worker 线程、asyncio loop、带 epoch 的 `TranscriptUpdate` / `HistoryPage` 信号、请求 future、`StreamCoalescer` |
| `qt/codex_dialog.py` | ChatGPT 账号卡、浏览器登录模态框与活动会话断开确认 |
| `qt/main_window.py` | 堆叠窗口（home / chat）、`Toast`、SDK 请求对话框。**不拥有任何快捷键，也不记账模态** |
| `qt/chat_view.py` | 聊天页：transcript + composer + 历史工具栏 + bridge 接线 |
| `qt/home_view.py` | 主页：会话浏览、搜索、批量删除、详情面板 |
| `qt/session_row.py` | 会话列表的一行 `SessionRow` + 圆形勾选标记 `SelectionMark`（手绘）|
| `qt/session_copy.py` | 把 `session_index` 的相对时间 / 文件大小成句（翻译 context 仍是 `HomeView`）|
| `qt/composer.py` | 输入框 `Composer` + 长文 `ComposerPad` |
| `qt/transcript.py` | 虚拟化 transcript 视图 + delegate（无逐行控件） |
| `qt/transcript_model.py` | 不可变 `TranscriptEntry` + Qt row state；共用 reducer 语义、decoded snapshot 的有界 presentation slice 与字符预算 |
| `qt/transcript_cards.py` | 卡片几何、结构化 section → `QTextDocument`、高度/document 两级缓存 |
| `qt/paint.py` | Qt 翻译边界与 transcript 颜色解析（`layout_record` + `qcolor`，不含几何、不画） |
| `qt/todo_panel.py` | 挂在 transcript 右上角的可收缩 TODO 面板（自己盯着宿主重新贴边）|
| `qt/settings_dialog.py` | `LLMSettingsDialog` |
| `qt/preferences_dialog.py` | `PreferencesDialog`：Appearance（字体 + 语言）/ Models 两页 |
| `qt/request_dialogs.py` | `ApprovalDialog` / `QuestionDialog` / `DeleteSessionsDialog` |
| `qt/theme.py` | 深色调色盘 + 全局 QSS（唯一 `setStyleSheet` 处）|
| `qt/styling.py` | 动态属性词汇表（`variant`/`role`/…）+ `style()` / `set_style_property()` / `repolish()` |
| `qt/components/` | 共享组件：`DialogFooter` / `KeyValueList` / `Card` / `SettingsList` |
| `qt/appearance.py` | 哪些 Qt 事件意味着「缓存的外观派生值已失效」：`APPEARANCE_CHANGED` / `FONT_CHANGED` |
| `qt/i18n.py` | QTranslator 安装：`active_language()` / `set_active_language()` / `apply_language()` |
| `qt/labels.py` | 纯层单词标签的 `QT_TRANSLATE_NOOP` 词表 + `translate_label()` |
| `qt/status_line.py` | 把 `rendering.status_values()` 的数字成句（状态条）|
| `qt/retranslate.py` | `Retranslator`：宿主控件的 `LanguageChange` 子对象，重跑 bound 的文案语句 |
| `qt/keys.py` | 全应用键位表（`HOME` / `CHAT` / `APPROVAL`）+ `install()` / `ensure_focus()` |

ChatGPT Codex 不是 GUI 私有 Provider。浏览器 OAuth、共享凭据、刷新和模型目录由
`kimi-cli/src/kimi_cli/auth/codex.py` 与核心 `OAuthManager` 管理；Codex 请求适配位于
`kimi-cli/src/kimi_cli/llm_codex.py`。GUI 的设置页和 bridge 只调用这些核心接口，不保存或刷新
OAuth token。

表里没有的模块就是不存在的模块：`src/kimix_gui/screens/`（Textual 时代的残壳）与
`qt/app.py`（5 行 re-export，零 importer）都已删除，`KimixGuiApp` 从 `kimix_gui.app` 导。
`tests/gui/test_docs.py` 双向盯着这张表的成员与真实模块集合一致；删、加模块必须同步改表。

## GUI 配置存储

`KimixGuiConfigStore` 是全局 GUI 配置的唯一内存快照和写入者。version 4 根对象同时保存
`interface`、`configs` 与 `work_dirs`；缺少或无效的可选 `interface` 使用规范化默认值，下一次
任意保存会把完整区块写回，同时保留另外两类配置。不要再为界面偏好创建第二个 store，否则两个
启动期快照会在先后保存时互相覆盖。

`work_dirs.<path>.default` 是带 `kind` 的 LLM source；version 3 的路径字符串会在内存中迁移，
下一次保存写为 version 4。session 独立覆盖仍归属于 Kimi session 目录自己的
`kimix-gui.json`，version 2 使用 `llm` tagged object；继续读取 version 1 的 `config` 路径并在
下一次修改时升级。两者都不保存 OAuth token。

## Bridge 线程模型

`KimixBridge` 独占一个 daemon 线程（`name="kimix-bridge"`）上的 asyncio loop。
SDK 对象（session、request）**绝不离开该线程**；跨线程只经 Qt signal 传 frozen dataclass
（`TranscriptUpdate` / `HistoryPage` / `StatusLineUpdate` / `TodoUpdate` / `ApprovalAsk` /
`QuestionAsk`）。`TranscriptUpdate` 内是 frozen AST 或类型化 mutation，不携带 SDK 对象和待解析正文。
从 UI 线程调用 bridge 的公开方法，内部用 `submit()`（`run_coroutine_threadsafe`）或
`call_soon_threadsafe()` 转发。

### Epoch 守卫

每个 payload 带 `epoch`；打开或释放会话时 `_bump_epoch()`（定义在 `qt/bridge.py`），
所有接收方（含 bridge 内部 await 之后）都要用 `epoch != self.bridge.epoch` 守卫并丢弃旧值。
新增信号必须带上 epoch，
否则会话切换时会串数据。

### 流式合并

增量 `AppendText` 与同一 activity 的完整 `ReplaceEntry` 快照走 `StreamCoalescer`，
`_COALESCE_SECONDS = 0.016` 合并一次 flush；其他 mutation 先 `flush()` 再发，以保持顺序。
不要绕过它直接高频 emit。历史回放与实时 model 都应用 `TranscriptReducer` 的同一语义。

## Qt 对象的所有权与销毁线程

Qt 对象只能在拥有它的线程上销毁，而 **PySide6 会在无父 Qt 对象的 Python 包装器死掉时删除
它的 C++ 对象**。包装器如果只能通过引用环到达，就会死在 `gc.collect()` 里；CPython 的自动
分代 GC 在跨过分配阈值的那个线程上跑。本项目里 `kimix-bridge` 一直在分配，所以经常就是它。
实测崩溃签名：faulting thread 叫 `kimix-bridge`，处于 `Garbage-collecting`，表现为
`0xC0000005`（访问违规）或 `0xC0000374`（堆破坏），修复前约 1/8 概率复现。

因此不变量不是“别泄漏”或“在对的线程上收集”，而是：**只能通过引用环到达的东西，不许还
拥有 C++ Qt 对象。** 已经是空壳的包装器（`Shiboken.isValid()` 为假）以及 C++ 拥有的有父对象，
从任何线程释放都无害。

### 不要把无父 Qt 对象存进容器

创建一个无父 Qt 对象本身是安全的：局部变量靠引用计数在当前线程死掉。实测
`QTextDocument` 无论 `setMarkdown` / `setTextWidth` / `documentLayout()` / `QTextCursor`
都不成环，所以“量一下就丢”的探针与测试不必改。**存起来**才危险，因为容器宿主几乎必然在环里
（view↔delegate，以及每条 `Retranslator` 语句都闭包捕获宿主）。需要缓存时给对象 parent，
让 C++ 拥有，并用 `deleteLater()` 释放，参见 `transcript._release()`。

触发点是 `main_window.show_chat()`：每次打开会话都 `remove_chat()` 再新建 `ChatView`，所以每轮
home→chat→home 都会把一整个 `Transcript` 连 document 缓存丢进引用环。不变量钉在
`tests/gui/test_gc_safety.py`（8 条，5/5 证伪）。

被否决的修法是在产品里 `gc.disable()`，再用 `QTimer` 在主线程定期 `gc.collect()`。它只是把
崩溃藏起来：为了一个容器改变整个进程所有环的回收时机，而“Qt 对象的销毁线程不确定”这个真缺陷
还在。`tests/gui/conftest.py` 曾在 `50423ee`～`735b2b1` 之间这么做；修好源头后已删除，测试套件
现在与产品使用同一套收集行为。

## 当前视图由 Qt 回答

`MainWindow.current_view`（`KimixGuiApp.screen` 就是它）先读
`QApplication.activeModalWidget()`，为空才回到 `QStackedWidget.currentWidget()`。
不要再引入 `set_modal()` 之类的记账。旧版本要求每个开对话框的地方记一次、关时再忘一次，
曾有 6 对分布在两个文件，其中 3 对写成
`lambda: self.window is not None and self.window.set_modal(None)`；漏掉一半就会让 `screen`
一直指向已关闭的对话框，没有任何机制能发现。

本仓所有对话框都 `setModal(True)`，Qt 的记录不会漏掉谁。offscreen 实测：`open()` 和 `show()`
都会登记；对话框叠对话框时返回最上面那个；上层关闭会重新露出下层；全关完回到 `None`。
基础探针保存在 `.kimix_cache/probe_active_modal.py`。
两处与旧记账行为不同，但都是真实状态：从“偏好设置”里打开的 LLM 配置库现在会被返回，长文输入框
`ComposerPad` 也会被返回。行为证据在 `.kimix_cache/falsify_modal_view.py`（3/3 RED）。

注意 `window.isAncestorOf(dialog)` 对“以窗口为父、但自己是独立窗口”的对话框返回 `False`
（实测），所以没有便宜的“这个模态是不是我的”判据；本应用只有一个 `MainWindow`，无需另加记录。

## 相关专题

- transcript、历史、TODO 与会话存储：[`transcript-and-sessions.md`](transcript-and-sessions.md)
- 外观与 Qt 事件：[`appearance-and-themes.md`](appearance-and-themes.md)
- 运行时翻译事件：[`i18n.md`](i18n.md)
- 键盘作用域与焦点：[`keyboard.md`](keyboard.md)
- transcript AST 的决策记录：[`adr/0001-structured-transcript-data.md`](adr/0001-structured-transcript-data.md)

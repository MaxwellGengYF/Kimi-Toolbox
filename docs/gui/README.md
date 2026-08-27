# kimix-gui 开发文档

这里是桌面 GUI 的设计与维护规则总入口。根目录的 [`AGENTS.md`](../../AGENTS.md) 只保留
LLM 每次改动都必须知道的核心限制；专题背景、真实行为、历史坑和验收方式放在这里。

## 按改动范围阅读

| 要修改的范围 | 必读文档 | 主要内容 |
| --- | --- | --- |
| 新增、删除、移动模块；调整依赖方向 | [`architecture.md`](architecture.md) | 分层与完整模块职责表 |
| bridge、线程、Qt 生命周期、路由或模态 | [`architecture.md`](architecture.md) | worker 线程、epoch、流式合并、Qt 对象所有权、current view |
| LLM Provider、模型目录、变体、持久化或迁移 | [`architecture.md`](architecture.md) | 精确选择、刷新快照、v5/v3 存储与 GUI-only ChatGPT 变体边界 |
| transcript、历史、工具卡片、TODO、会话文件 | [`transcript-and-sessions.md`](transcript-and-sessions.md) | 单一高度模型、历史窗口、工具分类、存储与内存上限 |
| QSS、控件属性、共享组件 | [`styling.md`](styling.md) | 设计 token、属性词汇、特异度、组件准入与 DialogFooter |
| 字体、主题、手绘颜色、外观缓存 | [`appearance-and-themes.md`](appearance-and-themes.md) | Qt 外观事件、缓存失效、深浅主题不变量 |
| 新增或修改界面文案、语言切换 | [`i18n.md`](i18n.md) | 翻译边界、Linguist 流程、运行时重译 |
| 快捷键、焦点与键盘交互 | [`keyboard.md`](keyboard.md) | 统一键位表、作用域、Qt 派发事实 |
| 编写或排查测试 | [`testing.md`](testing.md) | pytest-qt 约定、offscreen 陷阱、GC 与 DeferredDelete |
| 环境、代码风格、脚本与仓库规则 | [`development.md`](development.md) | 常用命令、Python 约定、格式化和仓库杂项 |
| 使用桌面客户端 | [`user-guide.md`](user-guide.md) | 功能、启动参数、配置与键位 |

一个改动可能跨越多个专题。例如新增一个随主题变化、带快捷键和可翻译文案的控件，至少要读
`styling.md`、`appearance-and-themes.md`、`keyboard.md` 和 `i18n.md`。

## 文档维护规则

- 根 `AGENTS.md` 只保留跨模块、高风险、需要每次进入仓库立即看到的限制。
- 专题规则放进最接近其代码所有权的文档；不要在多处复制整段说明，入口表负责把读者带过去。
- 记录“为什么”时保留可证伪事实、失败模式和对应测试，避免只写结论而丢掉边界条件。
- 新增或删除 `src/kimix_gui/` 模块时同步更新 `architecture.md` 的模块表；
  `tests/gui/test_docs.py` 会检查两边集合一致。
- 文档中的命令默认在仓库根目录、PowerShell 环境运行。

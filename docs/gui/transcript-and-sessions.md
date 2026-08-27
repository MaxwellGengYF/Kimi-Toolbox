# Transcript、历史与会话状态

本专题覆盖聊天记录的几何和缓存、历史窗口、工具卡片、TODO 与 kimi-cli 会话文件。线程交接、
epoch 和 Qt 对象所有权见 [`architecture.md`](architecture.md)；绘制颜色见
[`appearance-and-themes.md`](appearance-and-themes.md)。

## 浮动控件自己盯着宿主

`TodoPanel` 用手算的几何贴在 transcript 右上角，**它自己**在宿主上装 `eventFilter`，并按
`Resize` / `Show` / `LayoutRequest` 重新贴边。不要让宿主替它转发：那样每个想挂它的控件都得
先学会它的贴边规则，漏一个事件是静默的，面板会停在旧位置。

锚点是 `QAbstractScrollArea.viewport()` 而不是宿主本身：滚动条出现时 Qt 只 resize viewport，
不 resize 宿主；而且隐藏的 `QScrollBar` 照样报告宽度（实测：隐藏时 100、显示时 10），所以
不要再自己减一遍。宿主不是滚动区时就使用宿主本身，因为普通控件没有滚动条要让位。
契约钉在 `.kimix_cache/falsify_todo_host.py`（4/4 RED）。

## 一条记录只有一个高度模型

卡片的几何、布局与测量都归 `qt/transcript_cards.py`：

- `layout_for()` 是 `RecordLayout` 的唯一入口，换算基准是 *card* 宽度，不是行宽度；其中的字符格
  标题只作为纯文本 fallback。GUI 标题由 `header_line()` 使用当前字体的真实像素宽度拟合，省略号
  贴着 disclosure 箭头前的文本边界放置，不能再用固定 40/60/72 字符上限提前丢掉摘要。
- `body_width()` 是唯一的换行宽度。
- `CardBodies.row_height()` 是唯一的高度答案。
- `CardBodies.document()` 每个 body 只留一份；key 里没有行号，两行同文同宽就共用。

曾经并存两套模型：delegate 用真实 `QTextDocument` 测量，model 用固定宽 80 的字符格估算，
再乘 `LINE_HEIGHT` 去移动滚动条。实测两者差 12px，翻历史时读者的位置会漂。

需要按像素补偿滚动时查询 `bodies.row_height()`。还要记住 `setSpacing` 给每个 item 四周都留了
空白：一行实占 `height + 2 * spacing`，实测两行各高 56px 时需要移动 120px。

### Body 字体由视图注入

使用 `CardBodies(font=view.font)`，每次缓存 miss 都调用一次 font provider，而不是在构造期抓
快照。因此字体变化时丢缓存即可。测试可对单个 transcript `setFont()` 来钉死测量，不必改变
整个进程字体。注意 `QApplication.setFont()` 是 posted 的，测量前必须泵一次事件。

### 文档缓存的所有权

缓存中的 `QTextDocument` 必须有 parent，并在 transcript 释放时走 `deleteLater()`；无父 Qt 对象
进入 view↔delegate 等引用环可能被 bridge 线程的自动 GC 销毁并导致进程崩溃。完整原理与被否决
的 GC 绕法见 [`architecture.md`](architecture.md#qt-对象的所有权与销毁线程)。

高度缓存与 document 缓存是两层：`CardBodies` 用稳定 record id + content revision + width 缓存
整数高度，即使 laid-out document 已淘汰，`sizeHint()` 也不用再次构造它。document 只保留 32 个且
同时受正文字符预算限制；key 不持有完整正文，避免 pure-layer decoded cache 淘汰后仍被 Qt key
引用。

## Transcript AST 是唯一数据权威

`wire.jsonl` 外层 envelope 先经 Pydantic 校验为 SDK 类型；随后 `rendering.WireNormalizer` 在纯层
边界把消息一次性规范化为 `transcript_data.py` 中的 immutable AST。工具 arguments 只在该边界用
`orjson` 解码，字段同时得到 `primary` / `secondary` / `detail` role 与展示 hint。partial、invalid、
unknown 内容保存在显式 `RawBlock`，不丢原文；下游不再从 `key: value`、标题或正文反推语义。

一条 `TranscriptEntry` 只能是 dialogue/thinking 的 `TextEntry`、合并调用与结果的
`ActivityEntry`，或 system/error/approval 的 `NoticeEntry`。正文由 `TextBlock`、`MediaBlock`、
`FieldListBlock`、`DiffBlock`、`TodoBlock`、`QuestionBlock`、`RawBlock` 组成。历史回放与实时消息都
应用同一组 `StartEntry` / `AppendText` / `ReplaceEntry` / `FinishEntry` / `ClearTranscript`；
`ReplaceEntry` 在目标缺失时 upsert，`AppendText` 自带 kind、source、block 与 format，因此记录被裁剪
后仍可确定性重建。状态栏与 TODO 继续走各自的结构化状态流，不伪装成 transcript 行。

Qt model 只保存同一个不可变 `TranscriptEntry` 引用，expanded、record id、revision 留在 Qt row
state。标题、摘要、展开正文、复制文本和 accessibility 文本分别由纯 formatter 遍历 AST 得到；
派生字符串与 `QTextDocument` 只能作为输出缓存，不能再成为语义输入。

## Turn 是导航锚点，不是缓存页

翻历史的完整链路是：

`chat_view.load_older_history` → `bridge.load_older(turn)` →
`_seek(target, pin_latest=False)` → `HistoryPage` signal →
`chat_view._on_history_page` → `transcript.replace_history(page.entries)` + `jump_to_turn`

空页走 `mark_history_window()`，它返回自己标记的首行。

纯层 `Timeline` 的未加载 turn 只有 byte locator 与成本估计；已加载 turn 是完整的 frozen
`tuple[HistoryEntry, ...]`，不存在 `body is None` 一类半水合记录。decoded window 以目标 turn
为锚点，但按递归 `entry_cost()` 和 record 数扩张连续区间；
默认预算是 6 MiB hydrated body、512 records。小会话在预算内仍一次 hydrate，许多很短的 turn 会
先碰 record 上限，一个超大 anchor turn 则允许单独越过软上限，保证用户请求的目标不会消失。
相邻 seek 保留重叠 turn，只解析缺失的字节区间；远跳不会解析中间 gap。

Qt 收到较宽的 AST snapshot 后，`TranscriptModel` 只向 `QListView` 暴露一个自适应的
**presentation window**：按最矮 compact row 计算约三个 viewport，通常为 24～96 records，同时仍受
192 records / 2 MiB 的硬上限保护；viewport 高度跨过一行时重新收敛窗口，并保持读者当前的像素锚点。
历史页、slice 与 Qt row 复用同一个 AST 对象。滚到边缘时先移动
snapshot 内的 presentation slice；新旧 slice 的重叠 row 保持原对象，只对两端发出
`rowsRemoved` / `rowsInserted`，不能 `modelReset` 后让 `QListView` 重新测量整片窗口。稳定 record id
继续保持边缘记录的像素锚点；slice 耗尽后才让 bridge 读取另一个 decoded window。
滚动条的高频 `valueChanged` 中，“是否贴底”的跟随意图必须在信号到达时同步更新，历史边缘处理与
viewport 通知再以 16 ms 单次 timer 合并，下一帧只消费最后一个位置。流式 mutation 也按 16 ms
刷新；如果把贴底判定一起延后，旧 `value` 会与刷新后的新 `maximum` 比较，用户刚拖到底部也会被
误判成未贴底。向外滚动短页的 wheel 仍同步激活边缘，不能因合并而丢失“一页一次”的请求。
当一页短到滚动条同时位于顶部和底部时，`valueChanged` 不会再发生；此时向外的滚轮事件必须直接
激活对应边缘，并保持“一页一次、新页到达后重新 armed”的节流语义，否则首屏会卡在这一页。
`QListView` 对可变行高会询问整个 model 的 `sizeHint()`，所以只限制 document LRU、不限制 model
行数并不是真虚拟化。

不要再添加“往头上插一段”的无限滚动机制。曾经有一套 `prepend_history_blocks`（view 与 model
两层，还带自己的字符预算和静默返回 `(0, 0)` 的退出分支），从未接到任何按钮，却让“历史窗口
现在是哪几行”有两个写入者。`927566b` 把它连同唯一消费者 `Transcript._rows_height`、以及只为
它存在的 `_insert_records(trim=False)` 一起删除；现在每次插入都会裁剪。

现在仍只有 decoded snapshot 一个历史数据写入者；presentation slice 只是同一 snapshot 的有界
投影，不累积旧页，也不修改 pure-layer timeline。不要为 older/newer 再造两套 source mutation。

## 专用工具外观只有一份成员名单

成员判定归 `tool_display.KNOWN_TOOL_FAMILIES`（frozenset，11 项，配
`is_known_tool_family(name)`），颜色归 `transcript_layout.FAMILY_BAR_NAME`，标题归
`tool_display._FAMILY_LABEL`。

`_FAMILY_LABEL` 不含 `shell`：shell 标题取 wire 名，让 bash / pwsh 在 transcript 里分开。
依赖方向是 `transcript_layout → tool_display`，所以名单放在后者；两边的 key 集合由
`test_design_tokens.py::test_the_specialized_families_and_their_colors_are_one_list` 钉成相等。

Rich 时代的 `_FAMILY_LABEL_STYLE`（值是 `"bold cyan"` 一类 Rich markup）与 `_FAMILY_BAR`
已经删除；后者甚至已从现用的 `cyan` 漂成 `bright_cyan`。两张表的 value 都没有消费者，调用点
只把它们当成员测试；这正是 frozenset 的职责。

### 工具调用是一条 activity row

调用与结果继续按 `call_id` 合并为一条记录，但界面不再用 `▸`、`✓`、`└`、`⧉` 等字符模拟
控件和层级。收起态标题只保留“工具动作 + 对象 · 结果摘要”；状态、展开方向和复制动作由 delegate
绘制矢量图标，其中复制动作默认隐藏，只在鼠标进入对应聊天行时出现，且预留区域始终不变，避免
标题在 hover 时跳动。展开态采用 output-first：read、grep、shell 等常见工具把调用元数据压成首行的
弱化上下文，空一行后直接显示高对比度输出，不再插入 `Result` 标签或第二层卡片；代码、diff 等
多行输入则保留原貌，再直接衔接输出。所有已展开 activity 都使用中性 surface，错误只由状态图标
和侧边状态条表达，避免大面积警示色压过真正需要阅读的内容；完成的收起行仍保持透明背景。
成功图标统一使用 success 色，但动作名继续使用工具家族色（如 Read 为 cyan、Todo 为 yellow），
其后的对象与结果摘要保持 muted，使“状态、动作、内容”各自只承担一种视觉语义。

`RecordLayout.header_runs` 与 `body_sections` 都由纯层直接遍历 AST 产生。每个 section 明确携带
text、format、tone 与 spacing；调用上下文使用 `tone="context"`，输出使用 `tone="primary"`，
不再靠字符区间或字段名推断。`CardBodies` 按 section 构建文档，Markdown 交给
`QTextDocumentFragment.fromMarkdown()`。图标矩形、文本矩形、header 像素省略和正文矩形仍全部由
`qt/transcript_cards.py` 回答，不能在 delegate 里另算一套；状态来自 `ToolActivity.result.status`，
不得靠检查标题或正文字符推断。

## 会话存储

布局由 kimi-cli 决定：

`metadata → work_dir_meta.sessions_dir / <session_id> / {context.db|context.jsonl, wire.jsonl, state.json}`

定位路径必须先经过 `resolve_kimi_work_dir()`，以处理 Windows 盘符大小写与 Kimi metadata 对齐。
不要在客户端另造第二套目录推导规则。

## TODO 状态

`state.json` 的 `todos`（树，状态为 `pending` / `in_progress` / `done`）是权威值。
`todo_write` / `todo_update` 每次写入都会在 tool result 的 `display` 中携带已经摊平的
`TodoDisplayBlock`，用于生成期间的即时更新。

以下时机都要从磁盘重读：

- 会话打开；
- 每轮结束；
- `/clear` 之后；
- `/compact` 之后。

`todos.read_snapshot()` 的两个返回值不能混淆：

- `None`：定位不到 state 文件，保留当前显示；
- 空快照：成功定位，但现在没有 todo；`/clear` 会删除该文件。

子 agent 的 todo 位于 `subagents/<id>/state.json`，包在 `SubagentEvent` 中，面板不展示。

## 内存上限

- `MAX_TRANSCRIPT_CHARS = 64 MiB`：裁剪 transcript。
- `HYDRATED_BODY_BUDGET = 6 MiB`、`HYDRATED_RECORD_BUDGET = 512`：pure-layer decoded window。
- 活跃 presentation window 通常为 24～96 records；`MAX_PRESENTATION_RECORDS = 192`、
  `MAX_PRESENTATION_CHARS = 2 MiB` 是 Qt model 投影的硬上限。
- `DOCUMENT_CACHE_SIZE = 32`、`DOCUMENT_CACHE_CHARS = 2 MiB`：laid-out Qt documents。
- `HEIGHT_CACHE_SIZE = 2048`：不拥有正文或 Qt 对象的整数高度缓存。
- `MAX_HISTORY_TURNS = 4`、`MAX_HISTORY_BLOCKS = 32`：只保留给 legacy loader 与诊断脚本。

修改这些路径时不得引入无界增长，包括记录列表、历史 prepend、Qt document 缓存或随刷新增长的
binding。相关性能脚本只用于手动诊断，运行方式见 [`development.md`](development.md)。

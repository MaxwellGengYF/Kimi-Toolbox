# 测试约定与 Qt 测试陷阱

测试使用 pytest + pytest-qt，`asyncio_mode = auto`，`qt_api = pyside6`；
`tests/gui/conftest.py` 强制 `QT_QPA_PLATFORM=offscreen`。测试不联网，不接触真实 session 目录。

## 通用约定

- `tests/gui/qtutil.py` 提供 `find` / `widget_text` / `launch_app` / `wait_idle` /
  `wait_chat_ready` / `wait_home`。等待一律使用 `qtbot.waitUntil` + `wait_idle`，不要 sleep。
- 端到端测试注入 `session_factory=` 返回 `FakeSession`（见 `tests/gui/test_app.py`），再用 `_submit()`
  发 prompt。`KimixGuiApp` 的 `session_loader` / `history_loader` / `config_store` /
  `session_deleter` 同样可注入。
- 文件名与被测模块不总是一致：`tests/gui/test_widgets.py` 覆盖 `qt/composer.py` +
  `qt/transcript.py` + `Toast` + `StreamCoalescer`，仓库没有 `widgets.py`；
  `tests/gui/test_transcript_paint.py` 覆盖 `qt/paint.py` + `transcript_layout.py`，仓库没有
  `transcript_paint.py`。
- 纯层模块使用纯 pytest。修改 wire 渲染、工具格式化、history 分页或 todo 解析时，对应测试是
  `test_rendering.py` / `test_tool_display.py` / `test_history.py` / `test_todos.py`。
- 纯层断言钉结构，例如 `RelativeTime(RELATIVE_HOURS, 2)`，不钉英文整句。句子另在 Qt 侧由
  `test_status_line.py` / `test_home.py` / `test_labels.py` 覆盖，中文渲染在
  `test_i18n_runtime.py`。`qt/labels.py` 词表必须与纯层表一致；`test_labels.py` 会阻止新增纯层标签
  后忘记补词表、导致其永久静默显示英文。

## 测试语言固定为英文

`conftest.py` 的 autouse `_pin_english_ui` 直接替换 `app.apply_language`，强制安装英文语言包，
不依赖环境变量。Qt 在 Windows 上的 `uiLanguages()` 不识别 `LANG`；未注入
`config_store=` 的 `KimixGuiApp` 会从共享 GUI 配置读取语言偏好，因此测试还会把
`KIMI_SHARE_DIR` 重定向到临时目录，避免开发机真实设置进入进程。
所有界面文案断言默认写英文。

需要中文的测试自己调用 `set_active_language(qt_app, "zh_CN")`，并在 teardown 切回 `en`，参见
`tests/gui/test_i18n_runtime.py` 的 `restore_english`。应用 `.qm` 与 `.ts` 一起提交，因此新检出即可
验证中文渲染；翻译测试会在 catalog 丢失时直接失败。

## Posted 事件必须泵事件循环

- 测试修改字体后必须调用 `app.processEvents()`；`setFont` 不同步生效。
- `set_active_language` 后也必须调用 `app.processEvents()`；`LanguageChange` 是 posted 事件。
- 测 `showEvent` 逻辑时必须真的 `dialog.show()`。

背景见 [`appearance-and-themes.md`](appearance-and-themes.md) 与 [`i18n.md`](i18n.md)。

## Offscreen 字体与像素黄金值

黄金像素高度依赖 offscreen 的 stub 字体引擎，不依赖某个安装字体。
`test_render_geometry.py` 的 `GOLDEN_ROWS` 高度是
`GOLDEN_FONT_METRICS = {"height": 13, "line_spacing": 15, "advance": 13}` 的函数。测试先测量
本机指纹；不匹配则 `pytest.skip`，并把预期与实际指纹都写入消息（使用 fontconfig 测量的
offscreen 构建会得到不同数值）。

不要改成按字体名 skip：offscreen 下 family 是空串，那会使测试永远跳过。

与字体引擎无关的契约单独由
`test_a_row_is_its_header_its_body_and_its_margins` 覆盖：一行高度等于
`HEADER_HEIGHT + body + PAD_Y + 2 * CARD_MARGIN_Y`；紧凑行没有 body 与 `PAD_Y`。
新增像素黄金值时，按同样模式配一条组成式断言。

需要真实文字宽度时使用 `QT_QPA_PLATFORM=windows`；offscreen 中每个字符的 advance 都等于
`pixelSize`。更多探针限制见 [`styling.md`](styling.md#样式验收)。

## 键盘测试先激活窗口

测试键盘必须先调用 `qtutil.reactivate_window(widget)`。`QApplication.focusWidget()` 只对活动
窗口有值，而 offscreen 平台不会自行激活窗口，尤其是模态对话框关闭后：实测
`activeWindow() is None` 且 `focusWidget() is None`，但窗口仍记得之前的焦点控件。

真实平台会立即把焦点还回去，Windows 上已实测。因此 helper 只补平台差异，不替产品选择焦点；
焦点控件仍由 Qt 选择。`wait_home` / `wait_chat_ready` 已内置调用。遗漏时的症状很隐蔽：合成按键
静默无事发生，因为作用域快捷键没有焦点控件可匹配。

## `processEvents()` 不会真正删除控件

这是已经修好但必须知道的测试套件问题。pytest-qt 对注册控件执行 `close()` + `deleteLater()` 后
调用 `processEvents()`，看起来像清理完成，实际 Qt 故意不在 `processEvents()` 中投递
`DeferredDelete`。只有 post 它的事件循环退出时才会派发，而本套测试从不运行该事件循环，因此
控件会活过整个 session。

实测 `test_components.py` 每个用例增加 180 个控件，跑到主题测试时存活控件达到 20,780；
`QApplication.setStyleSheet` 耗时与活控件数线性相关，于是一次 `apply_theme` 在该位置需要
3.6 秒，而干净应用只需 6 毫秒。`gc.collect()` 无效，因为这不是引用环，而是 Qt 队列里的待删
对象。

`conftest.py` 的 `_flush_deferred_widget_deletions` 在每个用例 setup 时调用
`sendPostedEvents(None, DeferredDelete)`。不能放 teardown：pytest-qt 的 `trylast` 钩子比所有
fixture finalizer 更晚。全套由 217s 恢复到约 15s，前提事实钉在
`tests/gui/test_suite_hygiene.py`。

## 自动 GC 故意开启

`conftest.py` 一度使用 `gc.disable()` 躲避 bridge 线程上的 Qt 销毁崩溃。源头修复后已删除；测试
套件必须与产品运行同一套收集行为，否则“无父 Qt 对象进入引用环”一类回归只会在用户机器崩溃，
测试中却静默。

代价是全套慢约 4%（15.6s → 16.3s）。不要为稳定某个用例重新 `gc.disable()`，那会关闭整层保护。
干净信号来自 `tests/gui/test_gc_safety.py`；自动 GC 是其遗漏时的兜底，失败信号通常很差：进程会直接以
`0xC0000005` 退出，没有 pytest 报告。Qt 所有权原理见
[`architecture.md`](architecture.md#qt-对象的所有权与销毁线程)。

## 常用验证命令

```powershell
uv run pytest -q
uv run pytest tests/gui/test_todos.py -q
uv run pytest -q -k composer
uv run ruff check src tests
```

样式、翻译和特定子系统还有额外验收方式，分别查阅对应专题，不要仅凭单个文本测试判断像素行为。

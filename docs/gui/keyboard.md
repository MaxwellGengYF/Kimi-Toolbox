# 键盘绑定与焦点

## 键位表是唯一声明

所有键位都声明在 `qt/keys.py` 的表中，一个也不例外。每个键盘作用域一张表：`HOME` /
`CHAT` / `APPROVAL`，汇总在 `SCOPES`。每行是 `Binding(sequence, action)`：

- `sequence` 是可移植的 `QKeySequence` 文本，例如 `"Ctrl+F"`、`"Del"`，与 README 和用户
  实际按键一致；
- `action` 是稳定标识符，视图为它提供 handler。

视图安装自己的表：

```python
keys.install(self, keys.HOME, {"new-session": self.new_session.emit, ...})
```

`install` 双向校验 action 集合。缺 handler 意味着一个无行为的键，多 handler 意味着一段没有
入口的行为；两者都在构造时抛 `ValueError`，不会静默。它还拒绝同一作用域把同一个键绑定两次，
比较的是解析后的键，因此 `"Esc"` 与 `"Escape"` 视为同一个。

默认 context 是 `Qt.ShortcutContext.WidgetWithChildrenShortcut`：绑定只在焦点位于宿主控件子树
时触发。这就是作用域机制，所以 `HOME` 和 `CHAT` 可以各自绑定 `Esc` / `F4` 而含义不同。
自己就是窗口的对话框使用 `WindowShortcut`，因为它没有“所在页面”。

`MainWindow` 不拥有任何键位；`tests/gui/test_keys.py` 有守卫。它曾有 8 条 window 级 `QShortcut`，
每个 handler 第一句都要判断当前页面，相当于重新推导 Qt 已知的状态。`ChatView` 还曾在
`keyPressEvent` 中重复实现同样 7 个键，在产品中完全不可达；`HomeView` 曾使用
`keyPressEvent` + list 上的 `eventFilter` + if 链手写第三套。三套都已收敛到统一表。

用户可见的键位表位于 `README.md`，`tests/gui/test_keys.py` 会检查实现与 README 一致。

## Qt 键盘派发实测事实（PySide6 6.11）

写新键位前先读这些事实；它们解释了为什么手工派发机制多余：

- Qt 在把键送给任何控件之前先查快捷键表。因此 window 级 `QShortcut` 永远胜过任何
  `keyPressEvent`，包括焦点控件自己的；`QApplication.sendEvent` 也走同一路径。
- `QStackedWidget` 的非当前页是隐藏的，Qt 会跳过隐藏控件的快捷键。页面切换本身就是作用域判据。
- 模态对话框拿走焦点，因此页面作用域绑定自动停火，不需要 `if self._modal is not None`。
- 文本控件通过 `ShortcutOverride` 否决普通键。焦点在 `QLineEdit` 时，`n` / `Space` /
  `Delete` 快捷键不会触发，字符会被输入，所以不必手工保护搜索框。`Return` 不被 `QLineEdit`
  否决，因为它不插入换行；但 `QPlainTextEdit` 会否决，因为它要插入换行。
- `QListWidget` 不否决 `n` / `Space` / `Del` / `Return` 中任何一个，所以 list 上的
  `eventFilter` 是多余的。
- `QKeySequence("Return")` 不匹配小键盘 `Enter`，需要两行共享同一个 action。
- `QDialog` 默认把 `Escape` 映射到 `reject()`，即使 `QPlainTextEdit` 有焦点也生效。因此
  `QShortcut(Escape, self, self.reject)` 是冗余代码。唯一例外是 `ApprovalDialog`：它必须 emit
  `decided`，否则 SDK 请求会永远等待。
- `setFocus()` 对 `Qt.NoFocus` 容器确实生效，顶层和子控件都一样，作用域快捷键也会因此匹配。
  但焦点落在容器上意味着方向键和输入无处可去，所以 `keys.ensure_focus(page, target)` 必须指定
  页面键盘入口：home → 会话列表，chat → 输入框。
- `QApplication.focusWidget()` 只在窗口处于活动状态时有值。没有活动窗口就没有焦点控件，作用域
  快捷键永不匹配；这是 offscreen 测试的真实陷阱，详见 [`testing.md`](testing.md)。

## 对话框 Return 行为

对话框按钮统一使用 `DialogFooter`，它清除其他按钮的 `autoDefault`，使 Return 的归属不依赖构造
顺序。普通 confirm 是默认按钮；破坏性 confirm（`variant=danger`）则让 dismiss 成为默认按钮。
完整的顺序与组件准入规则见 [`styling.md`](styling.md#dialogfooter-与-autodefault)。

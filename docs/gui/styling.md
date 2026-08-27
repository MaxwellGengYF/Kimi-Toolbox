# 样式系统与共享组件

## 三层样式链路

纯层 `design/` 输出 token → `qt/theme.build_stylesheet(theme)` 生成全局 QSS → 控件用
`qt/styling.style()` 声明外观意图。

`objectName` 只回答“这是哪个控件”，并继续作为测试钩子；它不承担样式职责。样式由动态属性
选择器表达，控件内部不得调用 `setStyleSheet()`。

## 属性词汇表

值集合由 `qt/styling.py` 中的持有类定义：

| 属性 | 语义 | 值 |
| --- | --- | --- |
| `variant` | 控件的外观变体 | `primary` / `danger` / `ghost` / `icon` |
| `role` | 文字在信息层级里的角色 | `display` / `title` / `section` / `overline` / `caption` / `footnote` / `marker` |
| `tone` | 只改颜色的语气 | `muted` / `danger` |
| `level` | 同一 role 内的层级 | `"1"` / `"2"` |
| `surface` | 容器的表面类型 | `bar` |
| `card` | 卡片离页面多远（背景 + 圆角 + 内边距整套） | `floating` / `panel` / `inset` |
| `metric` | 尺寸预设；故意不叫 `size` | `action` / `nav` |

`state` / `kind` / `mode` / `flash` 是域数据的镜像，例如 `state` 镜像
`TodoEntry.status`。它们不属于外观词汇表，值集合归各自控件所有，因此不在
`tests/gui/test_styling.py::DECLARED_VALUES` 中。

### 标题层级只有一套读法

新增标题按语义选 role，不按“看起来多大”选：

| 声明 | 是什么 | 现有实例 |
| --- | --- | --- |
| `role=display, level=1` | 屏幕标题 | `home-title` |
| `role=display, level=2` | 对话框标题 | `preferences-title` / `settings-title` |
| `role=title` | 面板或对话框内部的小节标题，以及工具栏标签 | `settings-dialog` 的 Sources / Details、偏好设置两页的页标题、`chat-title` |
| `role=section` | 瞬时对话框标题、主页历史面板小标题 | request dialogs、`history-title` / `selection-count` |

`settings-title` 一度声明为 `role=title`，导致 LLM 配置对话框标题与内部 Sources / Details
同字号同字重，层级塌成一层。正确修法是把它升到 `display` / 2（`preferences-title` 本来就是），
不是把小节标题降级。`test_styling.py::test_the_two_full_size_dialogs_head_themselves_the_same_way`
钉住这条。

### `objectName` 的身份要求

同一个 `objectName` 不许同时代表两种控件。`dialog-body` 曾同时用于 `ApprovalDialog` 的
`QTextEdit` 和 `QuestionDialog` 的 `QLabel`，导致任何 per-name 规则都无法安全书写；现在已拆成
`approval-payload` / `question-detail`。objectName 只是身份和测试钩子，改名成本低，不要为省一次
改名让两个控件共用一个名字。

每个 `objectName` 必须至少有一种可验证的样式答案。`tests/gui/test_styling.py` 用运行时 gallery 建立
全集，少一个都会失败：

1. QSS 中有 ID 规则；
2. 带 `styling` 属性，并登记在 `EXPECTED_PROPERTIES`；
3. 由元素选择器绘制，并在 `ELEMENT_STYLED` 中登记控件类型、精确 selector 和必需声明；
4. 确实不需要专属 QSS，登记进 `NO_QSS_BY_DESIGN` 并写明理由。

ID 规则可以与属性或元素规则并存：前者可能只回答该实例的几何，后者回答共享外观。第三类不能用
自然语言代替事实；登记的运行时类型必须匹配，selector 必须真实存在，且必须声明登记的
`background` / `color` 等属性。`QComboBox` 曾被写成“由元素规则绘制”，实际 QSS 中没有该规则，
这正是结构化登记要阻止的错误。第四类也不是豁免口子：理由陈旧（控件已改名）、与其他基础样式
决策冲突，或控件携带样式属性都会失败。新增控件时不要只加 `objectName` 就结束；
`settings-error` 曾落入这个盲区，看起来像有样式，实际错误文案按正文渲染。

### 新增词汇

值持有类只能放大写的字符串常量。新增值必须同时在 QSS 中有规则，否则
`test_every_declared_value_is_actually_used_by_a_rule` 会失败；`DECLARED_VALUES` 由持有类反射得出，
不是手写清单。新增属性名要先通过
`test_no_property_name_shadows_a_real_qt_property`。

## 手绘控件

`SessionRow` / `SelectionMark` / `_ProgressStrip` 保留 `paintEvent`，因为它们包含内嵌圆角 marker
或数据可视化，没有 QSS 等价物。但数字必须进入 token，颜色必须通过 `active_theme()`：几何见
`design/scale.py` 的 `SessionListMetrics`，颜色见 palette / `status_color()`。

这条链不经过 QSS，由 `tests/gui/test_home.py` 的像素探针保护：替换 `_ACTIVE_THEME` 中一个颜色后，
渲染像素必须随之变化。运行时主题规则见
[`appearance-and-themes.md`](appearance-and-themes.md)。

## 五个会静默失败的 Qt 坑

这些行为已有守卫测试，不要踩回去：

1. **属性名不能撞 Qt 真属性。** `setProperty("size", "action")` 修改的是 `QWidget.size` 几何，
   不是标签；控件真的会改变尺寸，属性选择器同时匹配不上。这就是尺寸预设叫 `metric` 的原因。
   实测 `variant` / `role` / `tone` / `level` / `surface` / `metric` / `state` / `kind` /
   `mode` / `flash` 都干净。
2. **`letter-spacing: 0`（无单位）会被 Qt 整条丢弃。** 必须写 `0px`；`_tracking()` 现在一律
   带单位。
3. **QSS 特异度同权时后写的赢。** `qcssparser.cpp` 的 `Selector::specificity` 规则是元素名 +1，
   每个伪类和每个属性选择器 +0x10，ID +0x100。`[variant=…]` 与裸 `:hover` / `:disabled`
   都是 0x11，属性变体规则必须写在它们之后，才能像旧 ID 选择器一样生效。
4. **QSS 没有 `text-transform`。** 大小写只能在代码中转换，而且必须在查表之后转换，不能为大小写
   开第二条 msgid。`TodoPanel._title_text()` 是范式：`self.tr("Todos").upper()`；对 CJK 是 no-op。
5. **`text-decoration: line-through` 可用**，且会反映到 `font().strikeOut()`。不要手工构造
   `QFont(setStrikeOut)` 再 `setFont()`；那会置上 `WA_SetFont` 把字体钉死，使该行不再跟随界面
   字号偏好。`todo-item-title[state="done"]` 曾出现过这个问题。

## 不重复声明默认值

`QWidget { background: transparent }` 的类型选择器会覆盖每个子类。仓库曾有 11 条规则重复声明，
删除后像素零变化。

允许再次声明 transparent 的选择器和理由锁在
`tests/gui/test_styling.py::TRANSPARENT_ON_PURPOSE`：要么它覆盖一条更具体的竞争规则，要么它是
`::subcontrol`（子控件不受基规则覆盖，这也是相应 `::item` 规则必须保留的原因）。

### 瞬态表面不能依赖透明默认值

`QComboBox` 的下拉列表是点击后才显示的独立 popup 窗口，不会出现在只渲染静态对话框的 gallery
场景中。全局 `QWidget { background: transparent }` 同样会命中下拉框和 Qt 创建的内部 view；若
没有更具体的不透明规则，未选中区域会落到平台 backing surface，Windows 下表现为黑底。暗色主题
会把这个缺陷伪装成正常外观。

因此输入控件本体以及 popup/view 一类瞬态表面必须显式声明主题背景。对应测试必须真正调用
`showPopup()`，检查有效 palette 的背景 alpha 为 255、颜色来自当前主题，并覆盖“先在暗色主题创建
popup，再切到浅色后重新打开”的路径。检查颜色来源时也不能丢弃透明像素：缺少绘制和使用错误颜色
是两类不同的失败。

## 运行时属性必须 repolish

Qt 只在 polish 时解析一次 QSS。构造期使用 `style(...)`，此时首次 polish 尚未发生，无需手工
刷新；运行期修改使用 `set_style_property()`，它只在值真的改变时调用 `repolish()`。

`repolish()` 是全仓唯一的 `unpolish/polish` 入口，且 `update()` 是其契约的一部分，因为
`polish()` 本身不会排入重绘。不要复制一份手写实现。

## 样式验收

样式重构不能只比较 QSS 文本，因为文本必然变化。使用“计算样式 + 逐控件像素哈希”：12 个场景、
366 个控件，比对 font / 前景 / 背景 / 尺寸 / 像素，再对 36 个按钮做 enabled/disabled 反转。
工具先自证灵敏：恒等对比应为 0 变化；改变一个 token 应移动 35～85 个哈希。hover 状态与
`::subcontrol` 无法机械强制，只能用上面的特异度规则推理，并由源码注释覆盖。

探针自身有两个已知陷阱：

1. offscreen 平台的字体引擎是 stub：每个字符的 advance 等于 `pixelSize`，
   `height == pixelSize`，`lineSpacing == pixelSize + 2`。任何依赖文字宽度的测量都是假的；测真实
   排版必须使用 `QT_QPA_PLATFORM=windows`。offscreen 下 `QFontInfo(app.font()).family()` 返回
   空串、`pixelSize()` 返回 -1、`QFontDatabase.families()` 中没有 `Segoe UI`，所以不能按字体名
   判断数据是否可信。
2. 探针中的属性名不能手写。它曾漏掉 `card`，于是三个刚获得新属性的控件仍被报告为
   `style-property changes=0`。清单必须从 `qt/styling` 反射。

## 共享组件（`qt/components/`）

一个组件要解决属性层无法解决的结构、键盘行为或可访问性问题，才值得存在。属性层已经能表达
overline 和图标按钮外观，再包装一个 widget 类只会多一个名字。因此 `Overline` / `Badge` /
`IconButton` / `EmptyState` 刻意不做（最后一个全仓只有一个站点）。`SectionHeader` 也不做：
6 个候选站点的 margin、固定高度和 surface 都不同，共享的只有一个 `QHBoxLayout`，为此做五参数
组件是净亏。

属性表达本身已经足够直接：overline 是 `style(label, role=Role.OVERLINE)`，图标按钮是
`style(button, variant=Variant.ICON)`。

组件不负责文案，并且永远不许用 `setFixedWidth` 容纳文字；换语言会破坏布局。现有组件：

| 组件 | 它解决的问题 |
| --- | --- |
| `DialogFooter` | Return 键归属 + 按钮顺序 |
| `KeyValueList` | 键列自适应宽度（替代 `setFixedWidth(110)`）+ 值可选择 + `accessibleName` |
| `Card` | 背景、边框、圆角、内边距作为一个整体，见 `CardLevel` 三档 |
| `SettingsList` | 侧边栏行高等于当前字体两行；字体变化时重算，新插行也使用当前值 |

### `DialogFooter` 与 `autoDefault`

`QDialog` 中的 `QPushButton` 默认 `autoDefault=True`，Qt 会把最先找到的按钮提升为默认按钮，
于是 Return 的目标取决于构造顺序。修复前，`PreferencesDialog` 的默认按钮是隐藏页上的
`manage-llm-settings`，所以 Return 无法保存；`LLMSettingsDialog` 的默认按钮是 `browse-config`，
Return 会弹出文件选择器。

`DialogFooter` 在第一次 `showEvent` 时把窗口内其他按钮的 `autoDefault` 全部清掉。只调用
`setDefault` 不够，因为 Qt 会把当前焦点所在的 autoDefault 按钮重新提升。

两条 footer 规则由 `tests/gui/test_components.py` 的 `RETURN_ACTIVATES` / `FOOTER_ORDER` 表保护：

- 顺序：最左是“离开”，最右是“执行”，其他按钮放中间。
- 默认按钮：给 confirm；若 confirm 是破坏性的（`variant=danger`），则给 dismiss。Return 直接删除
  会让确认框失去意义。

新对话框的按钮要么放进 footer，要么登记进 `BUTTONS_OUTSIDE_THE_FOOTER` 并写理由，否则
`test_every_dialog_button_is_either_in_the_footer_or_declared` 会失败。

footer 是 widget，不是裸 layout，因此会吃掉垂直余量。删除确认框的两行文案曾因此被挤掉 35px；
`DialogFooter` 必须显式 `setSizePolicy(Preferred, Fixed)`。

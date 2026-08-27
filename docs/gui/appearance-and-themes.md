# 运行时外观与主题

样式属性、QSS 和共享组件见 [`styling.md`](styling.md)。本专题关注 Qt 如何传播字体、调色盘和
stylesheet 变化，以及深浅主题的数据不变量。

## 使用 Qt 自带的外观事件

没有 `ThemeManager`，也没有自定义变更信号，因为 Qt 已经广播了。PySide6 6.11 实测（事件过滤器
挂在三层容器深的控件树上）：

| 动作 | 事件 | 到达范围 |
| --- | --- | --- |
| `app.setFont()` | `FontChange` | 每一个控件，不只 top-level |
| `app.setPalette()` | `PaletteChange` | 每一个控件 |
| `app.setStyleSheet()` | `StyleChange` | 每一个控件 |
| `app.installTranslator()` | `LanguageChange` | 每一个 `QObject`，包括控件、layout、item delegate |

四个容易误判的例外：

1. 被 `setFont()` 单独设过字体的控件收不到 `FontChange`，因为它带 `WA_SetFont`。这是特性，
   偏好面板的字体预览依靠它保持被预览的字体。
2. 空 translator（`.qm` 加载失败）安装时不广播。语言包缺失会静默降级，不触发无意义的重译。
3. `setStyleSheet` 即使收到完全相同的字符串仍会发 `StyleChange`；`setFont` 收到相同字体则不发
   `FontChange`。因此 `StyleChange` 可以作为“主题重新应用”的代理信号，因为 `apply_theme`
   总会调用 `setStyleSheet`。
4. `app.setFont()` 不同步生效。它只存新默认值并 post 一个 `ApplicationFontChange`，向下传播要等
   事件被派发。事件循环没转之前，`widget.font()` 仍是旧家族，`fontMetrics().lineSpacing()`
   仍是旧值。测试在 `apply_interface_font` 后必须调用 `app.processEvents()`；否则断言可能什么都没
   测到。

## 缓存派生值的控件必须失效缓存

控件存了任何从字体或主题计算出来的值，就必须 override `changeEvent`，并在
`APPEARANCE_CHANGED` 中重算。当前四个监听者锁在
`tests/gui/test_appearance.py::LISTENS_FOR_APPEARANCE_CHANGES`：

| 控件 | 缓存内容 |
| --- | --- |
| `Transcript` | delegate 的整数高度 + `QTextDocument` 缓存：换行依赖字体，链接颜色被烘进文档 |
| `Composer` | 用 `fontMetrics().lineSpacing()` 算出的 `setFixedHeight` |
| `SettingsList` | 每行 `sizeHint`，同样来自 `lineSpacing()` |
| `_ElidedLabel` | `QFontMetrics` 算出的省略号位置 |

`Transcript` 的链接颜色是唯一“只重绘仍修不好”的情况：`_apply_markdown_link_color` 把颜色 merge
进 `QTextCharFormat`，因为 Qt 的 Markdown 阅读器无视文档 CSS；错误颜色已经存进文档。

`QListView` 自己会重排，不需要在 `changeEvent` 中再调用 `scheduleDelayedItemsLayout()`。这一调用
曾被加入又删除；删除后没有断言发生变化。

QSS 中的 `font-size` 会把控件钉死，使其不再跟随字号偏好。`QLabel#todo-item-title` 当前就是这样，
由 `test_a_stylesheet_font_size_pins_a_widget_against_the_font_preference` 记录现状。只记录，不顺手
修复：是否应该跟随是设计问题，默默改成跟随会移动没有人要求改变的像素。

## 主题模型（`DARK` / `LIGHT`）

两个主题注册在 `design/theme.py` 的 `THEMES` 中；它是 `MappingProxyType`，不是可修改的 dict。
偏好词汇表刻意镜像 i18n，让两边使用同一套思维模型：

| 主题 | 语言 | 含义 |
| --- | --- | --- |
| `SUPPORTED_THEMES` | `SUPPORTED_LANGUAGES` | 可提供给用户的全部值 |
| `SYSTEM_THEME`（`"auto"`） | `SYSTEM_LANGUAGE` | 跟随桌面 |
| `DEFAULT_THEME`（`"dark"`） | `DEFAULT_LANGUAGE` | 无法判断时的落点 |
| `resolve_theme(pref, prefers_dark)` | `resolve_language(pref, locale)` | 纯层决策，Qt 从外部传入系统信息 |
| `normalize_theme_preference` | `normalize_language_preference` | 未知值映射到 `auto` |

### 新主题只重声明颜色

`Theme` 的 11 组数值 scale 全部使用 `default_factory`；
`test_scales_default_construct_to_the_dark_values` 证明这一点。`Palette` / `CategoryPalette` 的字段
一个默认值都不允许有，也有守卫测试，因此漏掉一个颜色会得到 TypeError，而不是静默继承。

### 深浅主题不能共享 hex

两个主题之间不许共享任何 hex。这不是审美要求，而是可验证性要求：
`tests/gui/test_theme_switching.py` 的像素守卫依赖颜色“归属”；在浅色渲染结果中找到某个 hex，便能
判断它是否来自深色主题。为此 `LIGHT.text` 特意不使用 `DARK.bg`（`#0f1115`），虽然它看起来
很对称。守卫是 `test_no_hex_is_shared_between_the_two_palettes`。

### 色阶方向由主题自己决定

`bg → surface → panel → boost` 在深色主题中越来越亮，在浅色主题中越来越暗，两边都必须单调，
因为色阶表达的是“海拔”。
`test_the_surface_ramp_moves_away_from_bg_one_step_at_a_time` 检查单调性，
`test_the_two_themes_ramp_in_opposite_directions` 检查方向确实相反，防止一个“只稍微提亮”的
深色主题通过其他所有守卫。

### 分类色不能跨主题照抄

深色的 400 级色是为在 `#0f1115` 上发光而选，放在白底上会变成一团粉彩。
`test_text_and_muted_carry_contrast_against_the_page` 使用 350 的亮度差下限；这个数字来自实测，
两个主题实际最紧的颜色是 371 和 394，而深色黄 `#facc15` 对白底只有 290。

这不是 WCAG 检查。真正的对比度验证需要相对亮度公式，也需要人工确认哪些配对会同时出现。

## 当前主题只有一个入口

颜色只能从 `active_theme()` 进入。手绘控件（`qt/transcript.py` /
`qt/transcript_cards.py` / `qt/todo_panel.py`）在绘制时查询 `active_theme().palette` 或
`paint.qcolor()`，不得在 import 时抓快照。

仓库曾有第二个全局：扁平的 `name -> hex` 字典 `COLORS`。因为画笔使用
`from ... import COLORS` 拿到对象引用，`set_active_theme` 必须原地 `clear()/update()`。
它已经删除；`test_applying_a_theme_moves_the_one_global_the_painters_read` 通过
`not hasattr(theme, "COLORS")` 防止它回来。两个“当前主题是谁”的答案一定会冲突。

## Palette 与 CategoryPalette

查询 `palette` 还是 `categories` 取决于问题语义：

- `Palette` 回答“这个东西扮演什么角色”，例如 chrome、ink、accent、link。
- `CategoryPalette` 回答“这条记录是什么种类”，即 `layout_record` 输出的七个 hue 名。

二者同名同值的只有 `muted`。`qcolor()` 只认识 category 名，因此调用 `qcolor("bg")` 会静默降级
为灰色。测试也要按此区分：bar 的期望值取 `DARK.categories.resolve(token)`，复制图标取
`DARK.palette.muted`。

Markdown 链接使用 `palette.link`。它与 `categories.cyan` 同值但角色不同，和
`focus_ring` / `accent` 那对相同：链接是 prose 的 chrome，调整“工具调用”色带时不应顺带修改
正文链接。

## 桌面偏好与应用顺序

桌面色彩偏好来自 `QStyleHints.colorScheme()`（`qt/theme.desktop_prefers_dark`）。offscreen 平台
永远返回 `Unknown`，`setColorScheme()` 也被忽略，所以 `auto` 的桌面分支只能用 monkeypatch 测，
已记录在 `test_the_offscreen_platform_reports_no_preference`。跟随桌面实时切换
（`colorSchemeChanged`）尚未实现。

应用顺序固定为：**主题 → 字体 → 语言**。`apply_theme` 会把应用字体重设为主题自己的 base，
所以界面字号偏好必须在其后再声明一次。`app.py` 的 `_configure_application` 与
`_on_preferences_applied` 两处都保持这个顺序，中间不派发事件，因此不会出现一帧错误字体。

`InterfacePreferences.theme` 没有 bump `VERSION`。它沿用 `language` 的先例：缺失键由 `.get`
得到 `None`，normalizer 把 `None` 映射到默认值，所以新增带默认值的可选键不是兼容性破坏。
两条守卫分别覆盖“只缺 theme”和“language、theme 都缺”的旧文件。

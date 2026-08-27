# 中文术语表

`kimix_gui_zh_CN.ts` 的译法约定。**新增 `tr()` 文案前先查这张表**，表里没有的再定，定完补进来。
放在 GUI 包的 `src/kimix_gui/translations/` 而不是 `docs/gui/i18n.md`：它和 `.ts` 是同一份
工作产物，改翻译的人一定会打开这个目录；专题文档讲的是规则，不该被一张对照表淹没。

## 名词

| English | 中文 | 备注 |
|---|---|---|
| session | 会话 | 不用「场次」「对话」 |
| chat | 对话 | 指聊天页；`CHAT` 标题也译「对话」 |
| transcript | 对话记录 | |
| turn | 轮 / 轮次 | `Turn 3 of 12` → `第 3 / 12 轮` |
| history | 历史 | 不用「历史记录」，界面窄 |
| home | 主页 | |
| settings | 设置 | 指 LLM 配置对话框与按钮 |
| preferences | 偏好设置 | 指 `PreferencesDialog` |
| config / configuration | 配置 | 文件本身叫「配置文件」 |
| provider | 服务商 | 不用「提供商」「供应商」 |
| model | 模型 | |
| ChatGPT | ChatGPT | 品牌名，不译 |
| Codex | Codex | 产品/后端名，不译 |
| OAuth | OAuth | 协议名，不译 |
| browser | 浏览器 | OAuth 登录所在的系统浏览器 |
| endpoint | 接口地址 | |
| credential | 凭据 | |
| capabilities | 能力 | |
| thinking | 思考 | |
| context | 上下文 | |
| token | token | 不译，`100,000 tokens` 保持原样 |
| todo | 待办 | 单条也叫「待办」，计数用「项」 |
| prompt | 提示词 | 输入框里的长文本 |
| composer / pad | 撰写 | 长文写作对话框标题 |
| storage | 存储 | |
| folder | 文件夹 | 不用「目录」 |
| details | 详情 | |
| project default | 项目默认 | |
| effort | 强度 | thinking effort |
| server default | 服务端默认 | 模型目录未指定具体强度时 |
| stream | 流式输出 | |
| hook | hook | 不译，与 SDK 一致 |
| command | 命令 | `/help` 这类斜杠命令本体不译 |

## 动词与状态

| English | 中文 | 备注 |
|---|---|---|
| send | 发送 | |
| cancel | 取消 | |
| close | 关闭 | |
| delete | 删除 | 销毁数据 |
| remove | 移除 | 只解除引用，不删文件 |
| configure | 配置 | |
| browse | 浏览 | |
| select all | 全选 | |
| clear（清空选择的按钮） | 取消全选 | 这个按钮是全选的反向操作，不是「清空」 |
| approve | 批准 | |
| reject | 拒绝 | |
| open | 打开 | |
| save changes | 保存更改 | |
| jump to latest | 跳到最新 | |
| seek to turn | 跳转到指定轮次 | |
| in progress | 进行中 | |
| pending | 待处理 | todo 状态 |
| done | 已完成 | |
| archived | 已归档 | |
| missing | 缺失 | 配置文件找不到 |
| unavailable | 不可用 | |
| not specified | 未指定 | |
| connecting… | 连接中… | 省略号保持 `…`（U+2026），不写三个点 |
| connected | 已连接 | ChatGPT 账号状态 |
| disconnect | 断开连接 | 删除本地 OAuth 凭据，不译作「注销」 |
| sign in | 登录 | ChatGPT OAuth 流程 |
| refresh models | 刷新模型 | 重新获取模型目录 |
| cached list | 缓存列表 | 在线目录获取失败时的旧目录 |
| loading… | 加载中… | |
| running | 运行中 | |
| cancelling… | 取消中… | |
| in use | 使用中 | |
| just now | 刚刚 | 相对时间桶 |
| {count} minutes ago | {count} 分钟前 | 单数另写一条 `1 分钟前` |
| {count} hours ago | {count} 小时前 | 单数另写一条 `1 小时前` |
| yesterday | 昨天 | |
| unknown（时间未知） | 未知 | `updated_at <= 0` |

### transcript 行标签（`qt/labels.py` 的词表）

纯层返回英文，英文就是 msgid；这张表是 `TranscriptLabels` 语境的全部内容。
`tests/gui/test_labels.py::test_every_catalog_word_has_its_agreed_chinese_rendering` 钉住它。

| English | 中文 | 备注 |
|---|---|---|
| You | 你 | 用户行 |
| AI | AI | 不译，中文也用 `AI` |
| Think | 思考 | |
| Tool | 工具 | |
| Approval | 批准 | |
| System | 系统 | |
| Error | 错误 | |
| Read | 读取 | |
| Grep | Grep | **不译**：它是命令/工具名，要能和 wire 里的 `grep` 对得上 |
| Glob | Glob | **不译**，同上 |
| Write | 写入 | |
| Edit | 编辑 | |
| Python | Python | 不译（语言名） |
| Todo | 待办 | |
| Search | 搜索 | `web_search` 类工具 |
| Fetch | 抓取 | `fetch_url` 类工具 |
| Agent | 智能体 | `subagent` |
| Untitled | 无标题 | 没有标题的会话（`HomeView` 语境） |

### 状态条（`qt/status_line.py`）

| English | 中文 | 备注 |
|---|---|---|
| ready | 就绪 | 无状态可报时的占位 |
| context {used}/{total} | 上下文 {used}/{total} | |
| tokens in / out | 输入 / 输出 | `tokens` 本身不译 |
| cache read / cache write | 缓存读 / 缓存写 | |
| loading | 加载中 | MCP 子句 |
| tools | 个工具 | 计数用「个」 |

## 写法约定

- **占位符**：`{name}` 形式，见 `docs/gui/i18n.md` 的「文案规则」。译文里必须原样保留每一个
  `{name}`，可以调整顺序，**不能改名、不能漏**——漏了 `str.format()` 会抛
  `KeyError` 而不是显示错文案。
- **标点**：中文句子用中文标点（`，。：、`），分隔符沿用英文原文的 ` · `。
  冒号后不加空格：`下次使用:项目默认`。
- **数字与西文**：数字、`LLM`、`JSON`、`token`、模型名前后各留一个空格：`已选 3 项`、
  `管理 LLM 配置`。
- **全大写的小标题**（`ACTIVE CONFIG`、`TODOS`、`PREVIEW`）中文不做大写模仿，直接给正常词。
- **不译的东西**：品牌名（`Kimix`、`Kimi`）、路径示例（`C:\path\to\provider.json`）、
  字体样例（`The quick brown fox …`）、语言下拉的三个选项（见 `docs/gui/i18n.md`）、
  斜杠命令本体（`/help` `/compact` …）、SDK 协议值（`approve` / `reject` / `allow`
  这类回传给 SDK 的字符串）、`KB` / `MB` / `token`。
- **日期图样**：用 Qt 的 `QLocale` 图样而不是 `strftime`，图样本身也过 `tr()`：
  `MMM d` → `M月d日`，`yyyy-MM-dd` 保持原样。
- **助记键**：审批弹窗的 A / S / R 快捷键在所有语言下都不变（README 有键位表），
  但中文词看不出首字母，所以三个按钮的中文文案显式带上字母：`批准 (A)`、
  `本会话内批准 (S)`、`拒绝 (R)`。英文 msgid 不带字母。

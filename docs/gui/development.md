# 开发环境、代码约定与仓库规则

## 环境与依赖

项目要求 Python >= 3.14，并使用 KimiX 仓库根部的 uv 环境。GUI 与 SDK/CLI 共用根
`.venv`、`uv.lock` 和 workspace；不要在 `src/kimix_gui/` 内再建虚拟环境或子项目。
依赖关系详见 [`architecture.md`](architecture.md)。

常用命令：

```powershell
uv sync                                  # 默认 dev group，包含 PySide6 与 pytest-qt
uv sync --no-dev --extra gui             # 接近发行用户的 GUI 运行环境
uv run kimix-gui [--work-dir DIR] [--session ID] [--config provider.json] [--model M] [--thinking] [--yolo]
uv run pytest tests/gui -q                   # GUI 全量测试，offscreen Qt
uv run pytest tests/gui/test_todos.py -q # 单文件
uv run pytest -q -k composer             # 按关键词筛选
uv run ruff check src tests
uv run ruff format <改过的文件>          # 只格式化本次碰过的文件
uv run python scripts/gui/build_translations.py
```

`scripts/gui/build_translations.py` 是必需工具，完整规则见 [`i18n.md`](i18n.md)。同目录其他
脚本是手动性能诊断，不属于 pytest，也不要在常规改动后运行：

- `scripts/gui/benchmark_transcript.py`：transcript 内存/渲染压测，参数 `--gigabytes N`；
- `scripts/gui/diagnose_history_memory.py`。

## Python 代码约定

- 除两个 `__init__.py` 外，每个文件都写 `from __future__ import annotations`；模块首行写一句职责
  docstring。
- 使用 Python 3.14 语法。`except TypeError, ValueError:`（PEP 758 的无括号多异常）是有意的；
  仓库已有多处，不要“修正”为带括号写法。
- 解析 wire / state 使用 `orjson`。stdlib `json` 只在 `llm_config.py` 用于写出人类可读、
  `indent=2` 的 metadata，不要扩散。
- 跨层数据使用 `@dataclass(frozen=True, slots=True)`。
- ruff 配置为 line-length 100、target py314，并与上游统一启用 E/F/I/N/W。
- `setObjectName()` 只表达身份与测试钩子；外观使用 `qt/styling` 的动态属性和 `theme.py` QSS，
  不在控件中内联 `setStyleSheet`。运行时状态变化统一调用 `styling.set_style_property()`；详见
  [`styling.md`](styling.md)。
- 交互动作同时提供快捷键。键位表在 `qt/keys.py`，视图自己 `keys.install()`，`MainWindow`
  不拥有任何键位；详见 [`keyboard.md`](keyboard.md)。用户可见键位表在 README，
  `tests/gui/test_keys.py` 检查两者一致。

## 产品代码不用 `assert`

`src/` 中不写 `assert`；ruff 没启用 S101，因此靠 review 与测试守护。这不是风格偏好：

1. `python -O` 会把 assert 整条删除，使“不变量”在发行环境根本不检查。曾有
   `assert config.model is not None` 在 `-O` 下退化为后续的 `AttributeError`。
2. `AssertionError` 不在调用方的领域异常处理链中。它会绕过本层异常，例如 `llm_config` 的调用方
   捕获 `LLMConfigError`，设置对话框才能把问题渲染为可读信息。

需要类型收窄时优先改结构，不要机械换成 `raise`：

- `asyncio.get_running_loop()` 取代“loop 存在吗”的 assert；
- `mark_history_window()` 返回它标记的行号；
- 局部变量只读取一次；
- 用值本身代替“bool + 另一个变量”，参见 `history.py` 的 `continues`。

真正“不该发生”的状态用 `raise RuntimeError`，并确保上层有 handler 将它变成可见通知。
测试中的 `assert` 当然照常使用；也正因此 `PYTHONOPTIMIZE=1` 下全绿不构成有效验证。

## 修改与格式化范围

工作区可能已有用户修改。保留无关改动，不对未触及文件做批量格式化，不使用
`git reset --hard` / `git checkout --` 一类破坏性命令。

写入 Python 后先按根 `AGENTS.md` 运行 `uv run tools/syntax_check.py <files...>`；随后至少运行
针对性测试与 `uv run ruff check src tests`，风险较高或跨模块改动运行全量 `uv run pytest -q`。
只对本次改过的 Python 文件执行 `uv run ruff format`，并用 `uv run tools/git_diff.py <paths...>`
审阅 diff。

新增或删除模块时同步修改 [`architecture.md`](architecture.md) 的模块职责表；
`tests/gui/test_docs.py` 会双向检查。

## 仓库杂项

- `uv.lock` 是上游仓库受版本控制的环境快照。修改 `pyproject.toml` 后运行
  `uv sync --extra=all` 并提交对应 lockfile diff。
- 不覆写 uv 的 `link-mode` 默认值：uv 会按平台选择 clone/hardlink；显式 symlink 会把环境与缓存
  耦合，在未启用 Windows 符号链接权限时还会退化为复制并产生警告。
- 提交信息中英混用，跟随最近提交的风格即可。

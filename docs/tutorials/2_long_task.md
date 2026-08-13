# 长任务处理

对于复杂或耗时的任务，Kimix 提供 **`/plan` 命令**：让专门的 Planner Agent 生成可审阅的任务计划，确认后再由 Worker Agent 串行实现并复核。

---

## /plan 命令

`/plan` 命令采用**计划生成 → 用户审阅 → 执行实现 → 复核**的流程处理长任务。适合需要先明确步骤、再经用户确认后执行的场景。

### 基本用法

进入交互终端后执行：

```
/plan
```

进入多行输入模式：
- 以 `/end` 结束输入提交任务描述
- 以 `/cancel` 取消当前操作

**示例**：

```text
/plan
>>>> Start input requirement for plan, end with /end, cancel with /cancel
为这个项目添加完整的错误处理机制：
1. 为所有函数添加参数校验
2. 统一异常类型和错误码
3. 添加日志记录
/end
```

### 执行流程 (`prompt_plan_async`)

参考 `src/kimix/utils/prompt.py` 和 `src/kimix/cli_impl/commands.py`：

#### 阶段 1：计划生成

1. **创建 Planner 会话**：优先使用子 Provider 中 `role="planner"` 的配置创建专用会话（未配置时回退到主 Provider），使用 `agent_planner.json` 配置和 `TodoMaker` 系统提示词。该会话被设为**只读**（不能写文件或执行命令），并禁用 budget / context / compact / todo / target_churn 等循环提醒；同时启用计划工具（`WritePlan` / `EditPlan`）。
2. **指定计划文件**：若未提供文件路径，则在当前目录的 `.kimix_cache/` 下自动生成 `plan_<随机hex>.md`；若指定的文件已存在，会先删除。
3. **生成计划**：Planner 读取任务需求，将任务拆解为步骤列表，并通过 `WritePlan` 工具写入计划文件。生成过程最多尝试 **3 次**，每次结束后检查计划文件是否存在且非空，确保计划被正确写入。
4. **打开审阅**：计划生成后会用系统默认程序打开该文件，供用户查看。

#### 阶段 2：审阅与修订

计划生成后进入审阅循环：

- 提示：`Do you want to implement the plan? (y/n)`
  - 输入 `y`：进入执行阶段。
  - 输入 `n` 或其他：进入修订流程。
- 修订提示：`Please describe the changes you want (/quit to give up):`
  - 输入 `/quit`：放弃执行。
  - 输入具体修改意见：Planner 会根据反馈使用 `WritePlan` 或 `EditPlan` 工具更新计划文件，并再次打开供审阅，循环往复直到用户确认或放弃。

#### 阶段 3：执行与复核

1. **创建 Worker 会话**：使用默认的 Worker Agent 创建常规会话（`_create_default_session_async`）。
2. **发送实现提示**：
   - 若计划文件小于 **100 KB**，会直接将计划内容嵌入提示词；
   - 若计划文件大于 **100 KB**，则提示 Agent 读取计划文件并执行（实现前要求先研究并调用 `todo_write` 记录步骤）。
3. **追加复核提示**：实现完成后，再发送一次 review 提示，要求 Agent 检查计划是否全部完成。
4. **收尾**：Planner 会话在整条流程结束后（`finally`）才被关闭——用户确认后即进入执行阶段，Planner 会话在执行与复核期间仍保持打开；若用户选择 `/quit` 放弃，则不创建 Worker 会话。

### 指定计划输出文件

可通过 `/plan:<file>` 指定计划文件的输出路径。注意：该文件是**计划输出文件**，任务需求仍需通过多行输入提供：

```
/plan:docs/plan.md
>>>> Start input requirement for plan, end with /end, cancel with /cancel
为这个项目添加完整的错误处理机制：
1. 为所有函数添加参数校验
2. 统一异常类型和错误码
3. 添加日志记录
/end
```

若指定的计划文件已存在，会被覆盖。

---

## 长提示词与错误处理

参考 `src/kimix/utils/prompt.py` 中的 `prompt_async`：

### 自动截断超长提示词

当输入的提示词超过 **65536 字符**时，系统会自动将其导出到临时文件，并将提示词替换为 ``read and execute: `<temp_file>` ``，避免超出模型上下文限制。待办提醒、代码检查提醒若超过该长度，同样会先导出到临时文件。

### 自动重试机制

`prompt_async` 通过 `_run_single_prompt` 执行单次提示，内置**最多 3 次**尝试（`max_retries = 3`），各分支行为如下：

| 情形 | 行为 |
|------|------|
| 会话取消事件已设置 | 直接返回失败，不发起请求 |
| `KeyboardInterrupt` | 取消当前会话，返回失败 |
| 超时（`asyncio.TimeoutError` / `TimeoutError`） | 立即抛出，不重试 |
| HTTP API 错误（`APIStatusError`，如 4xx/5xx） | 立即抛出，不在本层重试——指数退避与重试由底层 chat-provider/soul 层负责 |
| 其他异常 | 取消会话；若还有剩余尝试则等待 **1 秒**后重试，最后一次尝试仍失败则直接抛出 |

> 说明：`429` 限流、`5xx` 服务端错误等 HTTP 状态码的指数退避（如 `min(2^attempt, 60)` 秒）发生在**底层 chat-provider/soul 层**（`kosong.chat_provider` / `kimi_cli`），`prompt.py` 不再叠加一层重试。

此外可通过 `timeout` 参数为整个提示（含重试）设置最长执行时间，超时后抛出 `TimeoutError`。

### 待办提醒（ensure_todo_finished）

默认情况下（`ensure_todo_finished=True`），主提示成功后进入「待办收尾」循环：

1. **待办提醒轮次**：由 `loop_control.cli_closing_reminder_rounds` 控制（默认 `1`，范围 0–5）。每轮检查会话中未完成的 `todos`（含子待办与备注，并带上截断后的原始请求作为上下文）：
   - 第 1 轮以「Todo review...」提示；后续轮次（`strong=True`）以「Final todo review...」发出 CRITICAL 强提醒，要求先把所有待办标记为 `done` 再结束。
   - 若所有待办均已完成，则无需提醒。
2. **收尾清理**：全部结束后清除会话中的待办（`_clear_session_todos`）；若指定了 `export_todo_list_path`（须以 `.json` 结尾），会先把待办列表导出为 JSON 再清理；若 `close_session_after_prompt=True` 还会关闭会话。
该机制与 `todo_write` 工具配合，确保长任务不会遗漏中间步骤。

---

## todo_write 工具

`todo_write` 是用于跟踪任务进度的工具，在执行过程中自动调用。

### 功能概述

参考 `kimi-cli/src/kimi_cli/tools/todo/__init__.py`：

- **读取模式**（`todos` 为 `null`）：返回当前待办列表
- **写入模式**（提供 `todos` 列表）：更新并持久化待办事项

### 数据结构

```python
class Todo:
    title: str # 待办事项标题
    status: str # 状态："pending" | "in_progress" | "done"
    notes: str | None # 备注（可选）
class Params:
    todos: list[Todo] | Todo | None  # 省略或 null 时读取；提供时写入（也接受别名 items）
```

### 显示效果

CLI 中以可视化卡片形式展示：

```
┌─────────────────────────────────┐
│ [done] 添加参数校验              │
│ [in_progress] 统一异常类型       │
│ [pending] 添加日志记录           │
└─────────────────────────────────┘
```

### 持久化

- **Root Agent**：保存在会话状态的 `state.todos` 中
- **Sub Agent**：保存在子代理目录的 `state.json` 中

---

## 方案对比

| 特性 | `/plan` |
|------|---------|
| **执行方式** | 串行执行 |
| **任务拆分** | 线性步骤列表 |
| **断点续传** | ❌ 不支持（当前实现会在一次流程中完成生成、审阅、执行与复核） |
| **进度跟踪** | todo_write 待办可视化 |
| **适用场景** | 步骤明确、顺序依赖的任务 |
| **调用开销** | 中等（Planner + Worker 两个会话） |
| **典型用例** | 功能实现、代码重构 |

### 选择建议

- **使用 `/plan`**：任务需要先明确步骤、经用户审阅确认后再执行，适合一次性完成的长任务

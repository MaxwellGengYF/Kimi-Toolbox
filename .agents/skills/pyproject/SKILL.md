---
name: pyproject
description: Reference for pyproject.toml files in the kimix monorepo. Use when modifying, adding, or analyzing Python package configs, workspace dependencies, build systems, or linting rules.
---

# Repo Pyproject Overview

Monorepo with 6 Python packages managed by `uv` workspace.

## Packages

| Path | Name | Build Backend | Key Deps |
|------|------|---------------|----------|
| `./pyproject.toml` | `kimix` | `hatchling` | `numpy`, `playwright`, `fastapi`, `kimi-cli-x`, `kimi-agent-sdk-x` |
| `kimi-cli/pyproject.toml` | `kimi-cli-x` | `uv_build` | `typer`, `aiohttp`, `kosong-x`, `pykaos`, `rich` |
| `kimi-agent-sdk/python/pyproject.toml` | `kimi-agent-sdk-x` | `uv_build` | `kimi-cli-x`, `kosong-x` |
| `kimi-cli/packages/kaos/pyproject.toml` | `pykaos` | `uv_build` | `aiofiles`, `asyncssh` |
| `kimi-cli/packages/kimi-code/pyproject.toml` | `kimi-code` | `uv_build` | `kimi-cli==1.40.0` (thin wrapper) |
| `kimi-cli/packages/kosong/pyproject.toml` | `kosong-x` | `uv_build` | `anthropic`, `openai`, `google-genai`, `pydantic`, `mcp` |

## Workspace

`tool.uv.workspace` members: `kimi-cli`, `kimi-cli/packages/kosong`, `kimi-cli/packages/kaos`, `kimi-agent-sdk/python`. Workspace sources: `kimi-cli-x`, `kimi-agent-sdk-x`, `kosong-x`, `pykaos`.

## Common Tool Configs

- **Ruff** (all): `line-length = 100`, `select = ["E", "F", "UP", "B", "SIM", "I"]`; root adds `N`, `W`, ignores `E501`.
- **pyright/ty**: strict, `pythonVersion = "3.14"`, `src/**/*.py` + `tests/**/*.py`.
- **mypy** (root only): strict, excludes `tests/`, `scripts/`.
- **Build backends**: `hatchling` (root, wheel targets `src/kimix`, `src/my_tools`); `uv_build` (all others, `module-name` in `tool.uv.build-backend`).
- **Scripts**: `kimix` → `kimix.cli:cli`; `kimi`/`kimi-cli-x`/`kimi-code` → `kimi_cli.__main__:main`.

## Optional Deps

`kimix[office]`: pymupdf, python-docx · `kimix[image_process]`: pillow · `kosong-x[contrib]`: anthropic, google-genai.

Full TOML block templates (`[build-system]`, `[project]`, `[tool.ruff]`, `[tool.pyright]`, etc.): read `references/pyproject-reference.md`.

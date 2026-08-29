from __future__ import annotations

import platform
import sys
from pathlib import Path

import pytest
from inline_snapshot import snapshot

pytest.importorskip("PyInstaller")

def test_pyinstaller_datas():
    from kimi_cli.utils.pyinstaller import datas

    project_root = Path(__file__).parent.parent.parent
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    site_packages = f".venv/lib/python{python_version}/site-packages"
    rg_binary = "rg.exe" if platform.system() == "Windows" else "rg"
    has_rg_binary = (project_root / "src/kimi_cli/deps/bin" / rg_binary).exists()
    _datas = []
    for path, dst in datas:
        p = Path(path)
        if p.is_relative_to(project_root):
            _datas.append((
                p.relative_to(project_root)
                .as_posix()
                .replace(".venv/Lib/site-packages", site_packages),
                Path(dst).as_posix(),
            ))
    datas = _datas

    expected_datas = [
        ('src/kimi_cli/agents/default/agent.yaml', 'kimi_cli/agents/default'),
        ('src/kimi_cli/agents/default/coder.yaml', 'kimi_cli/agents/default'),
        ('src/kimi_cli/agents/default/explore.yaml', 'kimi_cli/agents/default'),
        ('src/kimi_cli/agents/default/system.md', 'kimi_cli/agents/default'),
        ('src/kimi_cli/agents/okabe/agent.yaml', 'kimi_cli/agents/okabe'),
        ('src/kimi_cli/prompts/compact.md', 'kimi_cli/prompts'),
        ('src/kimi_cli/prompts/compact_cascade.md', 'kimi_cli/prompts'),
        ('src/kimi_cli/prompts/init.md', 'kimi_cli/prompts'),
        ('src/kimi_cli/skills/kimix_api/SKILL.md', 'kimi_cli/skills/kimix_api'),
        ('src/kimi_cli/skills/kimix_api/references/api.md', 'kimi_cli/skills/kimix_api/references'),
        ('src/kimi_cli/skills/skill-creator/SKILL.md', 'kimi_cli/skills/skill-creator'),
        ('src/kimi_cli/tools/agent/description.md', 'kimi_cli/tools/agent'),
        ('src/kimi_cli/tools/ask_user/description.md', 'kimi_cli/tools/ask_user'),
        ('src/kimi_cli/tools/file/glob.md', 'kimi_cli/tools/file'),
        ('src/kimi_cli/tools/file/read.md', 'kimi_cli/tools/file'),
        ('src/kimi_cli/tools/file/read_media.md', 'kimi_cli/tools/file'),
        ('src/kimi_cli/tools/file/write.md', 'kimi_cli/tools/file'),
        ('src/kimi_cli/tools/web/extract.md', 'kimi_cli/tools/web'),
        ('src/kimi_cli/tools/web/fetch.md', 'kimi_cli/tools/web'),
        ('src/kimi_cli/tools/web/search.md', 'kimi_cli/tools/web'),
    ]
    if has_rg_binary:
        expected_datas.append((f"src/kimi_cli/deps/bin/{rg_binary}", "kimi_cli/deps/bin"))

    # Package contents evolve; verify the expected files are all present rather
    # than requiring an exact match (collect_data_files may include extras).
    for item in expected_datas:
        assert item in datas, f"missing data file {item!r}"
    assert set(expected_datas) <= set(datas)


def test_pyinstaller_hiddenimports():
    from kimi_cli.utils.pyinstaller import hiddenimports

    assert sorted(hiddenimports) == snapshot(
        [
            "kimi_cli.cli.export",
            "kimi_cli.cli.info",
            "kimi_cli.cli.mcp",
            "kimi_cli.cli.plugin",
            "kimi_cli.tools",
            "kimi_cli.tools.agent",
            "kimi_cli.tools.ask_user", "kimi_cli.tools.context_prune", "kimi_cli.tools.display",
            "kimi_cli.tools.file", "kimi_cli.tools.file.auto_generated", "kimi_cli.tools.file.auto_repair", "kimi_cli.tools.file.blackbox", "kimi_cli.tools.file.check_fmt", "kimi_cli.tools.file.conflict_detect", "kimi_cli.tools.file.edit", "kimi_cli.tools.file.edit.base", "kimi_cli.tools.file.edit.diff", "kimi_cli.tools.file.edit.modes", "kimi_cli.tools.file.edit.modes.hashline", "kimi_cli.tools.file.edit.modes.patch", "kimi_cli.tools.file.edit.modes.replace", "kimi_cli.tools.file.edit.modes.sloppy", "kimi_cli.tools.file.edit.params", "kimi_cli.tools.file.edit_safety", "kimi_cli.tools.file.fs_cache", "kimi_cli.tools.file.glob", "kimi_cli.tools.file.grep_archive", "kimi_cli.tools.file.grep_local", "kimi_cli.tools.file.grep_output", "kimi_cli.tools.file.grep_recorder", "kimi_cli.tools.file.grep_selectors", "kimi_cli.tools.file.hash_line", "kimi_cli.tools.file.micro_compress", "kimi_cli.tools.file.output_utils", "kimi_cli.tools.file.parse_check", "kimi_cli.tools.file.read", "kimi_cli.tools.file.read_archive", "kimi_cli.tools.file.read_extract", "kimi_cli.tools.file.read_markit", "kimi_cli.tools.file.read_media", "kimi_cli.tools.file.read_media_shared", "kimi_cli.tools.file.read_pdf_pages", "kimi_cli.tools.file.read_profiles", "kimi_cli.tools.file.read_sqlite", "kimi_cli.tools.file.replace", "kimi_cli.tools.file.snapshot_store", "kimi_cli.tools.file.utils",
            "kimi_cli.tools.file.write", "kimi_cli.tools.memory", "kimi_cli.tools.reason", "kimi_cli.tools.test",
            "kimi_cli.tools.todo", "kimi_cli.tools.utils",
            "kimi_cli.tools.web", "kimi_cli.tools.web.content", "kimi_cli.tools.web.extract", "kimi_cli.tools.web.fetch", "kimi_cli.tools.web.providers", "kimi_cli.tools.web.search", "kimi_cli.tools.web.url_safety", "setproctitle",
        ]
    )


def test_pyinstaller_hiddenimports_include_lazy_cli_subcommands():
    from kimi_cli.cli._lazy_group import LazySubcommandGroup
    from kimi_cli.utils.pyinstaller import hiddenimports

    expected_hiddenimports = {
        module_name
        for module_name, _attribute_name, _help_text in LazySubcommandGroup.lazy_subcommands.values()
    }

    assert expected_hiddenimports <= set(hiddenimports)

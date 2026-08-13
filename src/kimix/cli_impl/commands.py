import os
import uuid
from pathlib import Path
from typing import Any
import pendulum

import kimix.base as base

from . import constants
from .utils import _input, _split_text


def _read_multi_line(text_arr: list[str], *, allow_cancel: bool = True) -> tuple[list[str], bool]:
    """Read multi-line input until /end or /cancel.

    Returns (lines, cancelled) where lines are the text lines collected
    (empty if /cancel was entered) and cancelled is True if /cancel was entered.
    """
    lines: list[str] = []
    while True:
        s = _input('', text_arr)
        if s.strip() == '/end':
            break
        if allow_cancel and s.strip() == '/cancel':
            return [], True
        lines.append(s)
    return lines, False

import asyncio

import kimix.utils._globals as _globals
from kimix.base import sync_all
from kimix.ui.printing import (
    Color,
    colorful_text,
    print_debug,
    print_error,
    print_info,
    print_success,
    print_warning,
)
from kimix.utils import (
    SystemPromptType,
    _create_default_session,
    clear_default_context,
    close_session,
    compact_default_context,
    create_session,
    create_supervisor_session,
    fix_error,
    get_default_session,
    print_usage,
    prompt,
    prompt_plan,
    set_ralph_loop,
)

from .init import init

exec_ctx: dict[str, Any] = {}


def _cmd_help(task_split: list[str], text_arr: list[str]) -> tuple[None, bool]:
    print(constants.HELP_STR)
    return None, False


def _cmd_clear(task_split: list[str], text_arr: list[str]) -> tuple[None, bool]:
    clear_default_context()
    return None, False


def _cmd_compact(task_split: list[str], text_arr: list[str]) -> tuple[None, bool]:
    compact_default_context()
    return None, False

def _cmd_export(task_split: list[str], text_arr: list[str]) -> tuple[None, bool]:
    import asyncio
    session = get_default_session()
    if session is None:
        print_error('No active session to export.')
        return None, False
    if len(task_split) < 2:
        print_error('Command must be /export:file')
        return None, False
    output_path = ':'.join(task_split[1:]) if len(task_split) > 1 else None
    try:
        output, count = asyncio.run(session.export(output_path=output_path))
        print_success(f'Exported {count} messages to {output}')
    except Exception as e:
        print_error(f'Export failed: {e}')

    return None, False


def _cmd_resume(task_split: list[str], text_arr: list[str]) -> tuple[None, bool]:
    if len(task_split) < 2:
        print_error('Command must be /resume:session_id')
        return None, False
    session_id = ':'.join(task_split[1:])
    session = get_default_session()
    if session:
        close_session(session)
    _globals._default_session = None
    _globals._default_role = None
    try:
        new_session = create_session(session_id=session_id, resume=True)
        _globals._default_session = new_session
        _globals._default_role = SystemPromptType.Worker

        # Update _cli_sessions cache
        cli_sess = new_session._cli.session
        _globals._add_cli_session(
            cli_sess.id,
            cli_sess.title,
            cli_sess.updated_at,
            context_usage=new_session.status.context_usage,
            context_tokens=new_session.status.context_tokens,
        )

        print_success(f'Resumed session {session_id}')
    except Exception as e:
        print_error(f'Failed to resume session: {e}')
    return None, False


async def _release_session_resources(session: Any) -> None:
    """Release file/network resources of an SDK session without deleting it."""
    if session._cancel_event is not None:
        session._cancel_event.set()
    await session._cleanup_tools()
    soul = getattr(session._cli, "soul", None)
    if soul is not None:
        try:
            await soul.close()
        except Exception:
            pass
    await session._cli.session.close_context_db()
    try:
        await session._close_chat_provider()
    except Exception:
        pass


def _cmd_store(task_split: list[str], text_arr: list[str]) -> tuple[None, bool]:
    if len(task_split) < 2:
        print_error('Command must be /store:session_id')
        return None, False
    target_id = ':'.join(task_split[1:])
    session = get_default_session()
    if session is None:
        print_error('No active session to store.')
        return None, False

    source_id = session.id
    if target_id == source_id:
        print_error('Target session name must be different from current session name.')
        return None, False

    work_dir = session._cli.session.work_dir
    old_anonymous = session._anonymous

    async def _do_copy() -> Any:
        from kimi_cli.session import Session as CliSession
        await _release_session_resources(session)
        return await CliSession.copy(work_dir, source_id, target_id)

    try:
        target = asyncio.run(_do_copy())
    except Exception as e:
        import traceback
        print_error(f'Store failed: {e}')
        print_error(traceback.format_exc())
        # Attempt to recover the original session so the CLI is not left broken.
        try:
            new_session = create_session(
                session_id=source_id,
                work_dir=work_dir,
                resume=True,
                anonymous=old_anonymous,
            )
            _globals._default_session = new_session
            _globals._default_role = SystemPromptType.Worker

            # Update _cli_sessions cache
            cli_sess = new_session._cli.session
            _globals._add_cli_session(
                cli_sess.id,
                cli_sess.title,
                cli_sess.updated_at,
                context_usage=-1.0,
                context_tokens=0,
            )
        except Exception as resume_err:
            print_error(f'Failed to resume original session: {resume_err}')
        return None, False

    # Prevent the old anonymous SDK object from deleting the original directory on GC.
    session._closed = True

    try:
        new_session = create_session(
            session_id=source_id,
            work_dir=work_dir,
            resume=True,
            anonymous=old_anonymous,
        )
        _globals._default_session = new_session
        _globals._default_role = SystemPromptType.Worker

        # Update _cli_sessions cache
        cli_sess = new_session._cli.session
        _globals._add_cli_session(
            cli_sess.id,
            cli_sess.title,
            cli_sess.updated_at,
            context_usage=new_session.status.context_usage,
            context_tokens=new_session.status.context_tokens,
        )
    except Exception as e:
        import traceback
        print_error(f'Store succeeded but failed to resume original session: {e}')
        print_error(traceback.format_exc())
        return None, False

    print_success(f'Session stored as {target.id}')
    return None, False


def _cmd_load(task_split: list[str], text_arr: list[str]) -> tuple[None, bool]:
    if len(task_split) < 2:
        print_error('Command must be /load:session_id')
        return None, False
    source_id = ':'.join(task_split[1:])

    from kaos.path import KaosPath
    from kimi_cli.session import Session as CliSession

    current = get_default_session()
    work_dir = KaosPath('.')
    if current is not None:
        work_dir = current._cli.session.work_dir

    # Confirm replacing a current session that has used context tokens.
    if current is not None:
        try:
            current_token_count = current._cli.soul.context.token_count
        except Exception:
            current_token_count = 0
        if current_token_count > 0:
            print_warning(
                f'Current session "{current.id}" has {current_token_count} context tokens. '
                'Loading will release it. Continue? (y/n)'
            )
            answer = _input('', text_arr).strip().lower()
            while answer not in ('y', 'n'):
                print_warning('Please enter y or n.')
                answer = _input('', text_arr).strip().lower()
            if answer != 'y':
                print_info('Load cancelled.')
                return None, False

    async def _do_copy() -> str:
        new_id = uuid.uuid4().hex
        if current is not None and current.id == source_id:
            # The source is the active session: release its locks first.
            await _release_session_resources(current)
        await CliSession.copy(work_dir, source_id, new_id)
        return new_id

    try:
        new_id = asyncio.run(_do_copy())
    except Exception as e:
        import traceback
        print_error(f'Load failed: {e}')
        print_error(traceback.format_exc())
        return None, False

    # Close the previous current session now that the copy is safely on disk.
    if current is not None:
        close_session(current)

    try:
        new_session = create_session(
            session_id=new_id,
            work_dir=work_dir,
            resume=True,
            anonymous=True,
        )
        _globals._default_session = new_session
        _globals._default_role = SystemPromptType.Worker

        # Update _cli_sessions cache
        cli_sess = new_session._cli.session
        _globals._add_cli_session(
            cli_sess.id,
            cli_sess.title,
            cli_sess.updated_at,
            context_usage=new_session.status.context_usage,
            context_tokens=new_session.status.context_tokens,
        )
    except Exception as e:
        import traceback
        print_error(f'Loaded session but failed to resume copy: {e}')
        print_error(traceback.format_exc())
        return None, False

    print_success(f'Loaded session {source_id} into anonymous session {new_id}')
    return None, False


def _cmd_sessions(task_split: list[str], text_arr: list[str]) -> tuple[None, bool]:
    """Handle /sessions and /sessions:<name> commands.

    /sessions        - list all known sessions from in-memory cache
    /sessions:<name> - create a new session named <name> and switch to it
    """
    from kimix.utils import _create_default_session

    # Case 1: /sessions:<name> — create a new named session
    if len(task_split) >= 2:
        new_name = ':'.join(task_split[1:])

        # Close current session if any
        session = get_default_session()
        if session:
            close_session(session)
        _globals._default_session = None
        _globals._default_role = None

        try:
            new_session = create_session(session_id=new_name, resume=False)
            _globals._default_session = new_session
            _globals._default_role = SystemPromptType.Worker

            # Update cache
            cli_sess = new_session._cli.session
            _globals._add_cli_session(
                cli_sess.id,
                cli_sess.title,
                cli_sess.updated_at,
                context_usage=new_session.status.context_usage,
                context_tokens=new_session.status.context_tokens,
            )

            print_success(f'Created and switched to session: {new_name}')
        except Exception as e:
            print_error(f'Failed to create session "{new_name}": {e}')
            # Try to recover: create a fresh anonymous session
            try:
                _create_default_session(resume=False)
            except Exception:
                pass

        return None, False

    # Case 2: /sessions — list all known sessions
    sessions_dict = _globals._cli_sessions

    if not sessions_dict:
        print_warning('No sessions found.')
        return None, False

    # Get current session id and update its context usage in the cache
    session = get_default_session()
    current_id = session._cli.session.id if session else None
    if session is not None:
        try:
            status = session.status
            sessions_dict[current_id]['context_usage'] = status.context_usage
            sessions_dict[current_id]['context_tokens'] = status.context_tokens
        except Exception:
            pass

    # Build sortable list
    items = [
        (
            sid,
            info.get('title', 'Untitled'),
            info.get('updated_at', 0.0),
            info.get('context_usage', -1.0),
            info.get('context_tokens', 0),
        )
        for sid, info in sessions_dict.items()
    ]
    items.sort(key=lambda x: x[2], reverse=True)

    id_width = max(len('session id'), *(len(sid) for sid, _, _, _, _ in items))
    print_info(
        f'{" ":1}  {"session id":<{id_width}}  {"updated at":<19}  {"context usage":<22}  title'
    )
    for sid, title, updated_at, context_usage, context_tokens in items:
        marker = '*' if sid == current_id else ' '
        updated_str = pendulum.from_timestamp(updated_at).strftime('%Y-%m-%d %H:%M:%S')
        if context_usage >= 0.0:
            usage_str = f'{context_usage * 100:.1f}% ({context_tokens} tokens)'
        else:
            usage_str = '-'
        print(f'{marker}  {sid:<{id_width}}  {updated_str}  {usage_str:<22}  {title}')
    return None, False


def _cmd_exit(task_split: list[str], text_arr: list[str]) -> tuple[None, bool]:
    session = get_default_session()
    if session:
        close_session(session)
    _globals._default_session = None
    _globals._default_role = None
    print_success('bye!')
    return None, True


def _cmd_context(task_split: list[str], text_arr: list[str]) -> tuple[None, bool]:
    print_usage()
    return None, False



def _cmd_cmd(task_split: list[str], text_arr: list[str]) -> tuple[None, bool]:
    if len(task_split) < 2:
        print_error('Command must be /cmd:xx yy')
        return None, False
    cmd = ':'.join(task_split[1:])
    try:
        result = os.system(cmd)
        if result == 0:
            print_success('Done.')
        else:
            print_warning('Failed.')
    except Exception as e:
        print_error(str(e))
    return None, False



def _cmd_fix(task_split: list[str], text_arr: list[str]) -> tuple[None, bool]:
    if len(task_split) < 2:
        print_error('Command must be /fix:<command>')
        return None, False
    command_to_fix = (':'.join(task_split[1:])).strip()
    if not command_to_fix:
        print_error('Command must be /fix:<command>')
        return None, False
    fix_error(command_to_fix, session=get_default_session())
    return None, False


def _cmd_plan(task_split: list[str], text_arr: list[str]) -> tuple[None, bool]:
    file_path: str | None = None
    if len(task_split) >= 2:
        file_path = ':'.join(task_split[1:]).strip()
    else:
        import secrets
        cache_dir = Path('.kimix_cache')
        cache_dir.mkdir(parents=True, exist_ok=True)
        file_path = str(cache_dir / f'plan_{secrets.token_hex(8)}.md')
    print(
        f'\n>>>> Start input requirement for plan, end with {colorful_text("/end", Color.YELLOW)}, '
        f'cancel with {colorful_text("/cancel", Color.YELLOW)}')
    text, _ = _read_multi_line(text_arr)
    requirement = '\n'.join(text).strip()
    if not requirement:
        print_warning('No requirement provided.')
        return None, False
    prompt_plan(requirement, file_path)
    return None, False


def _cmd_txt(task_split: list[str], text_arr: list[str]) -> tuple[None, bool]:
    print(
        f'\n>>>> Start input multiple-lines, end with {colorful_text('/end', Color.YELLOW)}, cancel with {colorful_text('/cancel', Color.YELLOW)}')
    text, _ = _read_multi_line(text_arr)
    for i in _split_text(text, _command_map_keys):
        text_arr.append(i)
    return None, False


def _cmd_file(task_split: list[str], text_arr: list[str]) -> tuple[str | None, bool]:
    if len(task_split) < 2:
        print_error(f'command format error, must be /file:path')
        return None, False
    file_name_str = ':'.join(task_split[1:])
    file_path = Path(file_name_str)
    if not file_path.is_file():
        print_error(f'file not found: {file_path}')
        return None, False
    return file_path.read_text(encoding='utf-8', errors='replace'), False


def _cmd_ralph(task_split: list[str], text_arr: list[str]) -> tuple[None, bool]:
    if len(task_split) < 2:
        print_error(f'command format error, must be /ralph:path')
        return None, False
    val = task_split[1].strip().lower()
    session = get_default_session()
    if val == 'on':
        set_ralph_loop(1)
        print_success(f'Ralph mode set to 1.')
    elif val == 'off':
        base._default_ralph = None
        set_ralph_loop(0)
        print_success(f'Ralph mode set to default.')
    else:
        try:
            num = int(val)
            set_ralph_loop(num)
            print_success(f'Ralph mode set to {num}.')
        except ValueError:
            print_error('Command must be /ralph:on, /ralph:off, /ralph:<num>')
    return None, False



def _cmd_init(task_split: list[str], text_arr: list[str]) -> tuple[None, bool]:
    init()
    _globals._default_session = None
    _globals._default_role = None
    _create_default_session()
    print_success('Initialized.')
    return None, False


def _cmd_supervisor(task_split: list[str], text_arr: list[str]) -> tuple[None, bool]:
    """Start a supervisor session with multi-line input text."""
    print(
        f'\n>>>> Start input for supervisor, end with {colorful_text("/end", Color.YELLOW)}, '
        f'cancel with {colorful_text("/cancel", Color.YELLOW)}')
    text, _ = _read_multi_line(text_arr)
    task_prompt = '\n'.join(text).strip()
    if not task_prompt:
        print_warning('No input provided for supervisor.')
        return None, False

    print_debug('Creating supervisor session...')
    try:
        supervisor_session = create_supervisor_session()
    except Exception as e:
        print_error(f'Failed to create supervisor session: {e}')
        return None, False

    try:
        prompt(prompt_str=task_prompt, session=supervisor_session, format_output=True)
    except Exception as e:
        print_error(f'Supervisor prompt failed: {e}')
    finally:
        close_session(supervisor_session)

    return None, False


def _cmd_swarm(task_split: list[str], text_arr: list[str]) -> tuple[None, bool]:
    """Start a swarm session with multi-line input text."""
    print(
        f'\n>>>> Start input for swarm, end with {colorful_text("/end", Color.YELLOW)}, '
        f'cancel with {colorful_text("/cancel", Color.YELLOW)}')
    text, cancelled = _read_multi_line(text_arr)
    if cancelled:
        return None, False
    swarm_prompt = '\n'.join(text).strip()
    if not swarm_prompt:
        print_warning('No input provided for swarm.')
        return None, False

    print_debug('Creating swarm session...')
    try:
        swarm_session = create_session(
            agent_file=base._default_agent_file_dir / 'agent_worker.json',
            agent_type=SystemPromptType.SwarmLeader,
            custom_data={'is_swarm_session': True},
        )
    except Exception as e:
        print_error(f'Failed to create swarm session: {e}')
        return None, False

    try:
        prompt(prompt_str=swarm_prompt, session=swarm_session, format_output=True)
    except Exception as e:
        print_error(f'Swarm prompt failed: {e}')
    finally:
        close_session(swarm_session)

    return None, False


def _cmd_todo(task_split: list[str], text_arr: list[str]) -> tuple[None, bool]:
    if len(task_split) < 2:
        print_error('Command must be /todo:<path>')
        return None, False
    file_name_str = ':'.join(task_split[1:])
    file_path = Path(file_name_str)
    if not file_path.is_file():
        print_error(f'file not found: {file_path}')
        return None, False

    import regex as re

    from kimix.parser import (
        CParser,
        HtmlParser,
        LispParser,
        PascalParser,
        PythonParser,
        ShellParser,
        SqlParser,
    )

    suffix = file_path.suffix.lower()
    parser = None
    if suffix == '.py':
        parser = PythonParser()
    elif suffix in {'.c', '.cpp', '.cc', '.cxx', '.h', '.hpp', '.java', '.js', '.ts', '.jsx', '.tsx', '.cs', '.go', '.rs'}:
        parser = CParser()
    elif suffix in {'.sh', '.bash', '.zsh'}:
        parser = ShellParser()
    elif suffix in {'.html', '.htm', '.xml', '.svg'}:
        parser = HtmlParser()
    elif suffix in {'.pas', '.pp', '.inc', '.dpr'}:
        parser = PascalParser()
    elif suffix in {'.lisp', '.lsp', '.clj', '.scm', '.ss', '.el'}:
        parser = LispParser()
    elif suffix == '.sql':
        parser = SqlParser()
    else:
        print_error(f'Unsupported file type: {suffix}')
        return None, False

    try:
        result = parser.parse_file(str(file_path))
    except Exception as e:
        print_error(f'Parse failed: {e}')
        return None, False

    todos = [c for c in result.comments if re.search(r'(?<![a-zA-Z0-9])TODO(?![a-zA-Z0-9])', c.content.upper())]
    if not todos:
        print_warning('No TODO comments found.')
        return None, False

    # Build formatted TODO items
    if len(todos) == 1:
        # Single TODO: short format, no numbering
        single = todos[0]
        todo_items = f'Line {single.line}: {single.content.strip()}'
        prompt_str = (
            f'Implement the TODO in {file_path}:\n'
            f'{todo_items}'
        )
    else:
        format_todo = lambda i, todo: f'{i}. Line {todo.line}: {todo.content.strip()}'
        todo_lines = [format_todo(i, todo) for i, todo in enumerate(todos, 1)]
        todo_items = '\n'.join(todo_lines)
        prompt_str = (
            f'Implement all TODOs in {file_path} at once:\n\n'
            f'{todo_items}\n\n'
            'Make sure to handle each TODO completely.'
        )

    try:
        print_info(prompt_str)
        prompt(prompt_str=prompt_str, format_output=True)
    except Exception as e:
        print_error(f'Prompt failed: {e}')

    return None, False


def _cmd_code(task_split: list[str], text_arr: list[str]) -> tuple[str | None, bool]:
    """Run a script file with optional arguments.

    Usage: /code:<script_path> [arg1] [arg2] ...

    For Python scripts (.py), runs with exec like core.py.
    For other scripts, runs directly as an executable.
    Supports both absolute and relative paths.
    """
    global exec_ctx
    if len(task_split) < 2:
        print_error("Command must be /code:<script_path> [args...]")
        return None, False

    cmd_part = ":".join(task_split[1:])
    parts = cmd_part.split()
    if not parts:
        print_error("Script path is required.")
        return None, False

    script_path_str = parts[0]
    script_args = parts[1:]

    script_path = Path(script_path_str)
    if not script_path.is_file():
        # Try resolving relative to current directory
        resolved = constants.curr_dir / script_path_str
        if resolved.is_file():
            script_path = resolved
        else:
            print_error(f"Script file not found: {script_path}")
            return None, False

    if script_path.suffix.lower() == ".py":
        # Use exec like core.py does for .py files
        import sys
        with open(script_path, "r", encoding="utf-8", errors="replace") as f:
            s = f.read()
        # Temporarily replace sys.argv so argparse-based scripts see the right args
        _old_argv = sys.argv
        sys.argv = [str(script_path)] + script_args
        print_info(f"Executing {script_path.name}", end="\n\n")
        try:
            exec_ctx["__file__"] = str(script_path)
            exec(s, exec_ctx)
            print_success(f"Done.")
        except KeyboardInterrupt:
            print_warning("Keyboard Interrupt.")
        except Exception as e:
            import traceback
            print_error(str(e))
            print_error(traceback.format_exc())
        finally:
            sys.argv = _old_argv
            sync_all()
    else:
        import subprocess
        import sys

        try:
            cmd = [str(script_path)] + script_args
            print_info(f"Running: {' '.join(cmd)}")
            # Let subprocess inherit stdout/stderr so output is displayed live
            result = subprocess.run(cmd, capture_output=False, text=True)
            if result.returncode == 0:
                print_success(f"Done (exit code 0).")
            else:
                print_warning(f"Exited with code {result.returncode}.")
        except FileNotFoundError:
            print_error(f"Executable not found: {script_path}")
        except Exception as e:
            print_error(str(e))

    return None, False


def _context_is_non_empty(session: Any) -> bool:
    """Return True when the session has real conversation content."""
    try:  # primary signal, mirrors clear/compact context checks
        if session.status.context_usage > 1e-8:
            return True
    except Exception:
        pass
    try:  # fallback: soul context token count
        if session._cli.soul.context.token_count > 0:
            return True
    except Exception:
        pass
    try:  # fallback: kimi_cli Session.is_empty() (wire file / context.db / context.jsonl)
        cli_session = session._cli.session
        is_empty = getattr(cli_session, 'is_empty', None)
        if callable(is_empty) and not is_empty():
            return True
    except Exception:
        pass
    return False


def _reflection_context_stats(session: Any) -> str:
    """Return a short human-readable description of the session context size."""
    try:  # preferred: soul context history + token count
        context = session._cli.soul.context
        n_messages = len(context.history)
        token_count = context.token_count
        return f'{n_messages} messages, {token_count} tokens'
    except Exception:
        pass
    try:  # fallback: status token count
        return f'{session.status.context_tokens} tokens'
    except Exception:
        pass
    return 'visible above'


def _builtin_tools_listing(repo_root: Path, kimix_tools_dir: Path, kimi_cli_tools_dir: Path) -> str:
    """Return the exact file path of every builtin tool, one tool per line.

    Mirrors the worker tool manifest in ``src/kimix/agent_worker.json``
    (``kimix.tools.*`` and ``kimi_cli.tools.*``). Paths are relative to the
    repo root (e.g. ``src/kimix/tools/file/bash/bash_tool.py``) so the
    reflection agent can edit the precise source file.
    """

    def rel(p: Path) -> str:
        try:
            return p.resolve().relative_to(repo_root.resolve()).as_posix()
        except ValueError:
            return p.resolve().as_posix()

    tools: list[tuple[str, Path]] = [
        ('bash', kimix_tools_dir / 'file' / 'bash' / 'bash_tool.py'),
        ('pwsh', kimix_tools_dir / 'file' / 'bash' / 'pwsh_tool.py'),
        ('Run', kimix_tools_dir / 'file' / 'run.py'),
        ('python', kimix_tools_dir / 'py' / '__init__.py'),
        ('job_output', kimix_tools_dir / 'background' / '__init__.py'),
        ('todo_write', kimi_cli_tools_dir / 'todo' / '__init__.py'),
        ('todo_push', kimi_cli_tools_dir / 'todo' / '__init__.py'),
        ('todo_pop', kimi_cli_tools_dir / 'todo' / '__init__.py'),
        ('todo_sub', kimi_cli_tools_dir / 'todo' / '__init__.py'),
        ('retrieve', kimi_cli_tools_dir / 'memory' / '__init__.py'),
        ('read', kimi_cli_tools_dir / 'file' / 'read.py'),
        ('read_image', kimi_cli_tools_dir / 'file' / 'read_media.py'),
        ('edit', kimi_cli_tools_dir / 'file' / 'replace.py'),
        ('write', kimi_cli_tools_dir / 'file' / 'write.py'),
        ('subagent', kimix_tools_dir / 'agent' / '__init__.py'),
        ('send_message', kimix_tools_dir / 'agent' / '__init__.py'),
        ('list_agents', kimix_tools_dir / 'agent' / '__init__.py'),
        ('interrupt_agent', kimix_tools_dir / 'agent' / '__init__.py'),
        ('workflow', kimix_tools_dir / 'swarm' / '__init__.py'),
        ('glob', kimi_cli_tools_dir / 'file' / 'glob.py'),
        ('grep', kimi_cli_tools_dir / 'file' / 'grep_local.py'),
        ('fetch_url', kimix_tools_dir / 'web' / 'fetch_url.py'),
        ('web_search', kimi_cli_tools_dir / 'web' / 'search.py'),
        ('context_usage', kimix_tools_dir / 'context' / '__init__.py'),
        ('compact', kimix_tools_dir / 'context' / '__init__.py'),
    ]
    return '\n'.join(f'- `{name}` — `{rel(path)}`' for name, path in tools)


def _build_reflection_prompt(session: Any, *, report_path: Path | None = None) -> str:
    """Build the /reflection prompt embedding source paths, AGENTS.md and rules."""
    agent_src = Path(__file__).resolve().parent.parent          # ...\src\kimix
    repo_root = agent_src.parent.parent                          # ...\kimi-agent
    kimix_tools_dir = agent_src / 'tools'                        # ...\src\kimix\tools
    kimi_cli_tools_dir = repo_root / 'kimi-cli' / 'src' / 'kimi_cli' / 'tools'
    agents_md_path = repo_root / 'AGENTS.md'
    system_prompt_path = agent_src / 'utils' / 'system_prompt.py'  # ...\src\kimix\utils\system_prompt.py
    worker_agent_json = agent_src / 'agent_worker.json'            # ...\src\kimix\agent_worker.json
    soul_dir = repo_root / 'kimi-cli' / 'src' / 'kimi_cli' / 'soul'  # ...\kimi-cli\src\kimi_cli\soul
    dynamic_injections_dir = soul_dir / 'dynamic_injections'        # ...\soul\dynamic_injections
    config_path = repo_root / 'kimi-cli' / 'src' / 'kimi_cli' / 'config.py'
    try:
        agents_md = agents_md_path.read_text(encoding='utf-8', errors='replace')
    except OSError:
        agents_md = '(AGENTS.md not found)'
    if report_path is None:
        report_path = repo_root / 'docs' / f'reflection_report_{pendulum.now().format("YYYYMMDD_HHMMSS")}.md'
    context_stats = _reflection_context_stats(session)
    builtin_tools = _builtin_tools_listing(repo_root, kimix_tools_dir, kimi_cli_tools_dir)
    return f'''# Reflection Task

Reflect on the conversation context above. Find misunderstandings caused by the
current agent design, then change the source code to make this project better.

## Context
- Agent source code path: `{agent_src}`
- Builtin tools dirs:
  - `{kimix_tools_dir}`
  - `{kimi_cli_tools_dir}`
- Current context: {context_stats} (full conversation is visible above)

## Builtin tools (exact file paths)
Every builtin tool and the exact file where its implementation lives:
{builtin_tools}

## Architecture map (source of truth)
- System prompt: `{system_prompt_path}` — builds the per-role system prompt as
  `{{TOOL_CONVENTIONS}}{{AGENT_ROLE}}:{{NUMBERED}}{{AGENTS_MD}}{{SKILLS}}`; roles via
  `SystemPromptType` (Worker/TodoMaker/Thinker/TrivialSubAgent/Supervisor/Reader/SwarmLeader);
  `get_system_prompt()` returns a per-runtime builder, picks the active shell, enforces
  `max_system_prompt_tokens`.
- Worker tool manifest: `{worker_agent_json}` — the exact tool list for the worker agent
  (`agent.extend=default`): Bash, pwsh, Run, python, job_output, todo_write, retrieve,
  read, read_image, edit, write, subagent, send_message, list_agents, interrupt_agent,
  workflow, glob, grep, fetch_url, web_search, context_usage, compact (from `kimix.tools.*`
  and `kimi_cli.tools.*`).
- Soul runtime: `{soul_dir}`
  - agent.py — Runtime + BuiltinSystemPromptArgs; loads AGENTS.md, skills, additional dirs
  - kimisoul.py — main agent loop: step/turn lifecycle, retries, auto-compaction, injections
  - compaction.py — context compaction (modes, adaptive preserve depth, auto-trigger)
  - context.py / context_db.py / context_records.py — context storage (JSONL/SQLite + migration), records
  - context_pruning.py — smart history removal (Tier A ephemeral / B elision / C micro-compress)
  - message.py — message helpers + `<system-reminder>` tags
  - dynamic_injection.py — DynamicInjection/Provider base + history normalization
  - verification_gate.py — soul-layer verification nudges for todos/code edits
  - approval.py — approval state (yolo/afk/auto-approve)
  - btw.py / steer.py / slash.py — side questions, mid-stream steering, slash registry
  - toolset.py / tool_taxonomy.py — tool loading & classification
  - denwarenji.py — D-Mail; history_index.py — BM25 history index; llm_request_recorder.py — request tracing
- Dynamic injectors (reminders): `{dynamic_injections_dir}`
  - budget_reminder.py — budget warnings as step/wall-clock usage crosses ratios (default 0.7/0.9)
  - compact_reminder.py — suggests `compact` when usage exceeds threshold (default 0.70)
  - context_meter.py — nudges `retrieve` when context usage materially changes
  - target_churn.py — anti-loop: repeated edits to same file / repeated identical errors
  - todo_reminder.py — re-injects unfinished todo_write items at the context tail
- Config: `{config_path}` — pydantic `Config` (model/provider, loop_control, background,
  notifications, services, mcp, hooks, skills); `LoopControl` drives loop/compaction/prune/
  reminder thresholds and toggles.

## AGENTS.md
{agents_md}

## Testing rules
- After writing any Python file, run: `uv run tools/syntax_check.py <file> [<file> ...]`
- After writing any code, write tests to cover it.
- Fix all errors reported by the syntax checker before proceeding.
- Review diffs with: `uv run tools/git_diff.py <file> [<file> ...]`
- After changing `pyproject.toml`, run: `uv sync --extra=all`

## Change code rules
- No case-by-case fixes: only generic, root-cause changes.
- Common usage must be written into shared/generic code.
- Be extremely careful about adding new tools: add only when very necessary.
- Follow the AGENTS.md performance rule (orjson, msgspec, uvloop, apsw, regex,
  rapidfuzz, xxhash, pybase64, pendulum instead of builtin counterparts).

## Report
After finishing all changes, write a report introducing the changes to:
`{report_path}`
Include: the misunderstanding found, what was changed, and why.
'''


def _cmd_reflection(task_split: list[str], text_arr: list[str]) -> tuple[None, bool]:
    """Reflect on the current context and refactor the agent source code."""
    session = get_default_session()
    if session is None:
        print_error('No active session. Start a conversation first.')
        return None, False
    if not _context_is_non_empty(session):
        print_error('Context is empty. /reflection requires a non-empty context.')
        return None, False
    prompt_str = _build_reflection_prompt(session)
    print_info(prompt_str)
    try:
        prompt(prompt_str=prompt_str, session=session, format_output=True)
    except Exception as e:
        print_error(f'Reflection failed: {e}')
    return None, False


def _cmd_unknown(task_split: list[str], text_arr: list[str]) -> tuple[None, bool]:
    print_warning('Unrecognized command.')
    return None, False


_command_map = {
    'help': _cmd_help,
    'clear': _cmd_clear,
    'exit': _cmd_exit,
    'context': _cmd_context,
    'cmd': _cmd_cmd,
    'fix': _cmd_fix,
    'txt': _cmd_txt,
    'file': _cmd_file,
    'plan': _cmd_plan,
    'compact': _cmd_compact,
    'export': _cmd_export,
    'resume': _cmd_resume,
    'store': _cmd_store,
    'load': _cmd_load,
    'sessions': _cmd_sessions,
    'ralph': _cmd_ralph,
    'reflection': _cmd_reflection,
    'supervisor': _cmd_supervisor,
    'swarm': _cmd_swarm,
    'init': _cmd_init,
    'todo': _cmd_todo,
    'code': _cmd_code
}
_command_map_keys = set(_command_map.keys())

# Argument-type categories used by the readline Tab completer in utils.py.
_command_arg_types: dict[str, str] = {
    "file": "file",
    "todo": "file",
    "export": "file",
    "plan": "file",
    "ralph": "ralph",
    "swarm": "swarm",
    "code": "file",
}

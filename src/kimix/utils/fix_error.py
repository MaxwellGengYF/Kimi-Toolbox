from typing import Optional, Any, Callable
from kimi_agent_sdk import Session
from kimix.base import run_process_with_error_async, run_thread
from kimix.ui.printing import MessageType, print_success
from .prompt import prompt, prompt_async
from .session import _create_default_session
import asyncio


async def fix_error_async(
        command: str,
        extra_prompt: Optional[str] = None,
        skip_success: bool = True,
        keycode: tuple[str, ...] = ('error', ),
        session: Session | None = None,
        max_loop: int = 4,
        merge_wire_messages: bool = False,
        ) -> bool:
    for i in range(max_loop):
        result = await run_process_with_error_async(
            command, keycode, skip_success=skip_success)
        if result is None:
            if i == 0:
                print_success('No error.')
            return True
        error_keyword = None
        for k in keycode:
            if error_keyword:
                error_keyword += ', ' + k
            else:
                error_keyword = k
        prompt_str = f'Fix {error_keyword} from command `{command}`:\n\n{result}\n'
        if extra_prompt is not None:
            prompt_str = f'{extra_prompt}, {prompt_str}'
        from kimix.tools.common import _maybe_export_output
        await prompt_async(_maybe_export_output(prompt_str), session, merge_wire_messages=merge_wire_messages)
    return False


def fix_error(
        command: str,
        extra_prompt: Optional[str] = None,
        skip_success: bool = True,
        keycode: tuple[str, ...] = ('error', ),
        session: Session | None = None,
        max_loop: int = 4,
        merge_wire_messages: bool = False,
        ) -> bool:
    return asyncio.run(fix_error_async(
        command, extra_prompt, skip_success, keycode, session, max_loop, merge_wire_messages
    ))


def async_prompt(
    prompt_str: str,
    session: Session | None = None,
    # settings
    output_function: Callable[[str, MessageType], Any] | None = None,
    info_print: bool = True,
    cancel_callable: Callable[[], bool] | None = None,
) -> Any:
    session_created = None
    if session is None:
        from .session import create_session
        session = create_session()
        session_created = True
    return run_thread(prompt, (prompt_str, session, output_function, info_print, cancel_callable, session_created, True))


def async_fix_error(
    command: str,
    extra_prompt: Optional[str] = None,
    skip_success: bool = True,
    keycode: tuple[str, ...] = ('error',),
    max_loop: int = 4,
    session: Session | None = None
) -> Any:
    if session is None:
        session = _create_default_session()
    return run_thread(fix_error, (command, extra_prompt, skip_success, keycode, session, max_loop, True))

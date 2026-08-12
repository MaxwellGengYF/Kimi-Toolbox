import os
from pathlib import Path
from typing import Any

from kimix.ui.printing import Color, colorful_text

curr_dir = Path(os.curdir)
if not curr_dir.is_absolute():
    curr_dir = curr_dir.absolute()

LSP_JSON = None
frame = 0
COMMENT = """Always write ' Start writen by AGENT ' comment before code, Always write ' End writen by AGENT ' comment after code"""

# Start writen by AGENT
HELP_STR = f'''Command line options:
  -c, --clean         - Delete cache file after quit
  --no_think          - Disable thinking mode
  --no_yolo           - Disable YOLO mode
  --no_color          - Disable colorful print
  --manually-cot      - Enable manually CoT mode
  --ralph             - Enable Ralph mode or set iterations
  -s, --skill-dir     - Specify custom skill directory
  --config            - Specify a JSON config file (see config_example.json for example)

Available commands:
  {colorful_text('/file:<path>', fg=Color.YELLOW)}    - Load a file and execute its content line by line
  <path>              - Same as /file:<path>
  {colorful_text('/todo:<path>', fg=Color.YELLOW)}    - Scan code file for TODO comments and prompt agent to implement them
  {colorful_text('/clear', fg=Color.YELLOW)}          - Clear the conversation context
  {colorful_text('/exit', fg=Color.YELLOW)}           - Exit the program
  {colorful_text('/help', fg=Color.YELLOW)}           - Show this help message
  {colorful_text('/context', fg=Color.YELLOW)}        - Print context usage
  {colorful_text('/fix:<command>', fg=Color.YELLOW)}  - Run a command and fix errors if any
  {colorful_text('/txt', fg=Color.YELLOW)}            - Input multiple line text
  {colorful_text('/init', fg=Color.YELLOW)}           - Initialize default LLM config
  {colorful_text('/compact', fg=Color.YELLOW)}        - Compact conversation context
  {colorful_text('/export:<path>', fg=Color.YELLOW)}  - Export session messages to file
  {colorful_text('/resume:<id>', fg=Color.YELLOW)}    - Close current session and resume a session by ID
  {colorful_text('/store:<id>', fg=Color.YELLOW)}   - Copy the current session to a new named session
  {colorful_text('/load:<id>', fg=Color.YELLOW)}    - Copy a named session into a new anonymous session
  {colorful_text('/sessions', fg=Color.YELLOW)}       - List resumable sessions for the current working directory
  {colorful_text('/ralph:on', fg=Color.YELLOW)}       - Enable Ralph mode
  {colorful_text('/ralph:off', fg=Color.YELLOW)}      - Disable Ralph mode
  {colorful_text('/ralph:<num>', fg=Color.YELLOW)} - Set Ralph iterations
  {colorful_text('/reflection', fg=Color.YELLOW)} - Reflect on current context and refactor agent source code
  {colorful_text('/supervisor', fg=Color.YELLOW)} - Start a supervisor session with multi-line input text
  {colorful_text('/swarm', fg=Color.YELLOW)}          - Start a swarm session with multi-line input text
  {colorful_text('/plan', fg=Color.YELLOW)}           - Plan a long-term task, step-by-step, then execute
  {colorful_text('/cmd:<command>', fg=Color.YELLOW)}  - Execute system command
  {colorful_text('/code:<path> [args...]', fg=Color.YELLOW)}  - Run a script file with optional arguments

Or enter any prompt to send to the agent.

Press <Tab> after '/' to complete commands and arguments.
'''
# End writen by AGENT

CLEAN_MODE: bool | None = None


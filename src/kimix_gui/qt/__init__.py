"""PySide6 desktop UI for Kimix."""

from kimix_gui.qt.chat_view import ChatView
from kimix_gui.qt.home_view import HomeView
from kimix_gui.qt.request_dialogs import ApprovalDialog, DeleteSessionsDialog, QuestionDialog
from kimix_gui.qt.settings_dialog import LLMSettingsDialog

__all__ = [
    "ApprovalDialog",
    "ChatView",
    "DeleteSessionsDialog",
    "HomeView",
    "LLMSettingsDialog",
    "QuestionDialog",
]

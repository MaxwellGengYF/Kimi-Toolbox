"""Dynamic form for every parameter axis exposed by an LLM model."""

from __future__ import annotations

from collections.abc import Iterable

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

from kimix_gui.llm import (
    LLMProblem,
    ParameterAssignment,
    ParameterSpec,
    ResolvedParameter,
)
from kimix_gui.qt.llm_text import (
    axis_label,
    missing_parameter_label,
    parameter_option_label,
    parameter_picker_description,
    problem_message,
)
from kimix_gui.qt.styling import Tone, style

from .parameter_picker import ParameterPicker, ParameterValueOption


class ParameterForm(QWidget):
    """Render ordered parameter specs without provider-specific UI branches."""

    value_activated = Signal(str, str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("parameter-form")
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._pickers: dict[str, ParameterPicker] = {}
        self._problem_labels: dict[str, QLabel] = {}
        self.setVisible(False)

    def set_parameters(
        self,
        parameters: Iterable[ParameterSpec],
        assignment: ParameterAssignment,
        *,
        resolved: Iterable[ResolvedParameter] = (),
        enabled: bool = True,
    ) -> None:
        """Replace the form while retaining missing values as disabled choices."""

        self._clear()
        specs = tuple(sorted(parameters, key=lambda item: (item.order, item.axis)))
        resolutions = tuple(resolved)
        resolved_by_axis = {item.axis: item for item in resolutions}
        for parameter in specs:
            resolution = resolved_by_axis.get(parameter.axis)
            selected = assignment.get(parameter.axis)
            problem = resolution.problem if resolution is not None else None
            self._add_parameter(parameter, selected, problem, enabled=enabled)

        known_axes = {parameter.axis for parameter in specs}
        for resolution in resolutions:
            if resolution.axis in known_axes or resolution.spec is not None:
                continue
            self._add_unknown_parameter(resolution)

        self.setVisible(bool(self._pickers))

    def picker(self, axis: str) -> ParameterPicker | None:
        return self._pickers.get(axis)

    @property
    def pickers(self) -> tuple[ParameterPicker, ...]:
        return tuple(self._pickers.values())

    def set_controls_enabled(self, enabled: bool) -> None:
        for picker in self._pickers.values():
            picker.setEnabled(enabled)

    def problem_label(self, axis: str) -> QLabel | None:
        return self._problem_labels.get(axis)

    def _add_parameter(
        self,
        parameter: ParameterSpec,
        selected: str | None,
        problem: LLMProblem | None,
        *,
        enabled: bool,
    ) -> None:
        picker = ParameterPicker(self)
        picker.setObjectName(f"param-{parameter.axis}")
        picker.setAccessibleName(axis_label(parameter.axis))
        picker.setAccessibleDescription(parameter_picker_description())
        picker.set_options(
            (
                ParameterValueOption(
                    option.value,
                    parameter_option_label(parameter, option),
                    option.problem is None,
                )
                for option in parameter.options
            ),
            selected_value=selected,
            missing_label=missing_parameter_label(selected),
        )
        picker.setEnabled(enabled)
        picker.value_activated.connect(
            lambda value, axis=parameter.axis: self.value_activated.emit(axis, value)
        )
        self._add_row(parameter.axis, picker, problem)

    def _add_unknown_parameter(self, resolution: ResolvedParameter) -> None:
        picker = ParameterPicker(self)
        picker.setObjectName(f"param-{resolution.axis}")
        picker.setAccessibleName(axis_label(resolution.axis))
        picker.setAccessibleDescription(parameter_picker_description())
        picker.set_options(
            (),
            selected_value=resolution.stored_value,
            missing_label=missing_parameter_label(resolution.stored_value),
        )
        picker.setEnabled(False)
        self._add_row(resolution.axis, picker, resolution.problem)

    def _add_row(
        self,
        axis: str,
        picker: ParameterPicker,
        problem: LLMProblem | None,
    ) -> None:
        row = QWidget(self)
        row.setObjectName(f"param-{axis}-row")
        row_layout = QVBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        control_layout = QHBoxLayout()
        control_layout.setContentsMargins(0, 0, 0, 0)
        label = QLabel(axis_label(axis), row)
        label.setObjectName(f"param-{axis}-label")
        label.setBuddy(picker)
        control_layout.addWidget(label)
        control_layout.addWidget(picker, 1)
        row_layout.addLayout(control_layout)

        problem_label = QLabel(problem_message(problem), row)
        problem_label.setObjectName(f"param-{axis}-problem")
        problem_label.setWordWrap(True)
        style(problem_label, tone=Tone.DANGER)
        problem_label.setVisible(bool(problem_label.text()))
        row_layout.addWidget(problem_label)

        self._layout.addWidget(row)
        self._pickers[axis] = picker
        self._problem_labels[axis] = problem_label

    def _clear(self) -> None:
        while (item := self._layout.takeAt(0)) is not None:
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._pickers.clear()
        self._problem_labels.clear()

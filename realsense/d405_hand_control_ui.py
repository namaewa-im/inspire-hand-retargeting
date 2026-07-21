from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QObject, QSignalBlocker, Qt, QThread, Signal, Slot
from PySide6.QtGui import QCloseEvent, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from hand_joint_estimator import (
    DEFAULT_MODEL_PATH,
    JOINT_LABELS,
    JOINT_LIMITS_RAD,
    JOINT_NAMES,
    HandEstimate,
    HandJointEstimator,
)

SLIDER_SCALE = 1000
HISTORY_SECONDS = 10.0
EXPECTED_FPS = 30
HISTORY_SIZE = int(HISTORY_SECONDS * EXPECTED_FPS)
DEFAULT_FILTER_STRENGTH = 0.286
DEFAULT_JITTER_ENABLED = True


def _parse_bool_arg(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(
        "Expected one of: true, false, 1, 0, yes, no, on, off."
    )


class UdpJointPublisher:
    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def publish(
        self,
        joint_values: np.ndarray,
        estimate: HandEstimate,
    ) -> None:
        payload = {
            "timestamp_s": float(estimate.timestamp_s),
            "fps": float(estimate.fps),
            "handedness": estimate.handedness,
            "handedness_score": float(estimate.handedness_score),
            "joint_names": list(JOINT_NAMES),
            "joint_values": [float(value) for value in joint_values],
        }
        data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self._socket.sendto(data, (self.host, self.port))

    def close(self) -> None:
        self._socket.close()


class EstimatorWorker(QObject):
    frame_ready = Signal(object)
    failed = Signal(str)
    finished = Signal()

    def __init__(
        self,
        model_path: Path,
        camera_serial: Optional[str],
        preferred_hand: Optional[str],
        jitter_enabled: bool,
    ) -> None:
        super().__init__()
        self.model_path = model_path
        self.camera_serial = camera_serial
        self.preferred_hand = preferred_hand
        self._running = False
        self._filter_strength = DEFAULT_FILTER_STRENGTH
        self._jitter_enabled = bool(jitter_enabled)
        self._estimator: Optional[HandJointEstimator] = None

    @Slot()
    def run(self) -> None:
        self._running = True
        try:
            with HandJointEstimator(
                model_path=self.model_path,
                camera_serial=self.camera_serial,
                preferred_hand=self.preferred_hand,
                width=640,
                height=480,
                fps=EXPECTED_FPS,
            ) as estimator:
                self._estimator = estimator
                estimator.set_filter_strength(self._filter_strength)
                estimator.set_jitter_enabled(self._jitter_enabled)
                while self._running:
                    estimate = estimator.read(timeout_ms=1000)
                    self.frame_ready.emit(estimate)
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        finally:
            self._estimator = None
            self.finished.emit()

    def request_stop(self) -> None:
        self._running = False

    @Slot(float)
    def set_filter_strength(self, strength: float) -> None:
        self._filter_strength = float(np.clip(strength, 0.0, 1.0))
        if self._estimator is not None:
            self._estimator.set_filter_strength(self._filter_strength)

    @Slot(bool)
    def set_jitter_enabled(self, enabled: bool) -> None:
        self._jitter_enabled = bool(enabled)
        if self._estimator is not None:
            self._estimator.set_jitter_enabled(self._jitter_enabled)


class JointControlRow(QWidget):
    value_changed = Signal(int, float)

    def __init__(
        self,
        joint_index: int,
        label: str,
        lower: float,
        upper: float,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.joint_index = joint_index
        self.lower = lower
        self.upper = upper

        self.name_label = QLabel(label)
        self.name_label.setMinimumWidth(132)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(
            round(lower * SLIDER_SCALE),
            round(upper * SLIDER_SCALE),
        )
        self.slider.setSingleStep(5)
        self.slider.setPageStep(50)
        self.slider.setValue(0)
        self.slider.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )

        self.spinbox = QDoubleSpinBox()
        self.spinbox.setRange(lower, upper)
        self.spinbox.setDecimals(3)
        self.spinbox.setSingleStep(0.01)
        self.spinbox.setSuffix("")
        self.spinbox.setMinimumWidth(112)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 3, 0, 3)
        layout.setSpacing(10)
        layout.addWidget(self.name_label)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.spinbox)

        self.slider.valueChanged.connect(self._on_slider_changed)
        self.spinbox.valueChanged.connect(self._on_spinbox_changed)

    def value(self) -> float:
        return float(self.spinbox.value())

    def set_value(self, value: float) -> None:
        clipped = float(np.clip(value, self.lower, self.upper))
        with QSignalBlocker(self.slider), QSignalBlocker(self.spinbox):
            self.slider.setValue(round(clipped * SLIDER_SCALE))
            self.spinbox.setValue(clipped)

    def set_manual_enabled(self, enabled: bool) -> None:
        self.slider.setEnabled(enabled)
        self.spinbox.setEnabled(enabled)

    @Slot(int)
    def _on_slider_changed(self, slider_value: int) -> None:
        value = slider_value / SLIDER_SCALE
        with QSignalBlocker(self.spinbox):
            self.spinbox.setValue(value)
        self.value_changed.emit(self.joint_index, value)

    @Slot(float)
    def _on_spinbox_changed(self, value: float) -> None:
        with QSignalBlocker(self.slider):
            self.slider.setValue(round(value * SLIDER_SCALE))
        self.value_changed.emit(self.joint_index, float(value))


class HandControlWindow(QMainWindow):
    filter_strength_changed = Signal(float)
    jitter_enabled_changed = Signal(bool)

    def __init__(
        self,
        *,
        model_path: Path = DEFAULT_MODEL_PATH,
        camera_serial: Optional[str] = None,
        preferred_hand: Optional[str] = None,
        filter_strength: float = DEFAULT_FILTER_STRENGTH,
        jitter_enabled: bool = DEFAULT_JITTER_ENABLED,
        udp_host: Optional[str] = None,
        udp_port: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.setWindowTitle("D405 MediaPipe Hand Control")
        self.resize(1450, 920)
        self.setMinimumSize(1100, 720)

        self._model_path = Path(model_path)
        self._camera_serial = camera_serial
        self._preferred_hand = preferred_hand
        self._udp_publisher: Optional[UdpJointPublisher] = None
        if udp_host is not None and udp_port is not None:
            self._udp_publisher = UdpJointPublisher(udp_host, udp_port)
        self._latest_estimate: Optional[HandEstimate] = None
        self._last_tracking_values = np.zeros(len(JOINT_NAMES), dtype=np.float32)
        self._manual_values = np.zeros(len(JOINT_NAMES), dtype=np.float32)
        self._filter_strength = float(np.clip(filter_strength, 0.0, 1.0))
        self._jitter_enabled = bool(jitter_enabled)
        self._history_start = time.perf_counter()

        self._time_history: deque[float] = deque(maxlen=HISTORY_SIZE)
        self._joint_history: list[deque[float]] = [
            deque(maxlen=HISTORY_SIZE) for _ in JOINT_NAMES
        ]

        self._build_ui()
        self._apply_style()
        self._start_estimator_thread()

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(14, 14, 14, 14)
        root_layout.setSpacing(12)

        root_layout.addWidget(self._build_status_bar())

        vertical_splitter = QSplitter(Qt.Orientation.Vertical)
        vertical_splitter.setChildrenCollapsible(False)
        root_layout.addWidget(vertical_splitter, 1)

        top_splitter = QSplitter(Qt.Orientation.Horizontal)
        top_splitter.setChildrenCollapsible(False)
        vertical_splitter.addWidget(top_splitter)

        top_splitter.addWidget(self._build_camera_panel())
        top_splitter.addWidget(self._build_control_panel())
        top_splitter.setStretchFactor(0, 3)
        top_splitter.setStretchFactor(1, 2)
        top_splitter.setSizes([860, 500])

        vertical_splitter.addWidget(self._build_plot_panel())
        vertical_splitter.setStretchFactor(0, 3)
        vertical_splitter.setStretchFactor(1, 2)
        vertical_splitter.setSizes([570, 300])

    def _build_status_bar(self) -> QWidget:
        frame = QFrame()
        frame.setObjectName("statusFrame")
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(16, 10, 16, 10)
        layout.setSpacing(18)

        self.camera_status = QLabel("Camera: connecting...")
        self.mediapipe_status = QLabel("MediaPipe: starting")
        self.fps_status = QLabel("FPS: --")
        self.hand_status = QLabel("Hand: --")
        self.control_status = QLabel("Control: Tracking")

        for label in (
            self.camera_status,
            self.mediapipe_status,
            self.fps_status,
            self.hand_status,
            self.control_status,
        ):
            label.setObjectName("statusLabel")
            layout.addWidget(label)

        layout.addStretch(1)
        return frame

    def _build_camera_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("panel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        header = QHBoxLayout()
        title = QLabel("MediaPipe Overlay")
        title.setObjectName("panelTitle")
        header.addWidget(title)
        header.addStretch(1)

        self.overlay_checkbox = QCheckBox("Show overlay")
        self.overlay_checkbox.setChecked(True)
        header.addWidget(self.overlay_checkbox)
        layout.addLayout(header)

        self.video_label = QLabel("Waiting for RealSense frame...")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(640, 420)
        self.video_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self.video_label.setObjectName("videoLabel")
        layout.addWidget(self.video_label, 1)
        return panel

    def _build_control_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("panel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 12, 16, 14)
        layout.setSpacing(10)

        title = QLabel("Joint Control")
        title.setObjectName("panelTitle")
        layout.addWidget(title)

        subtitle = QLabel(
            "Joint order: thumb CMC yaw, thumb CMC pitch, "
            "index/middle/ring/pinky MCP"
        )
        subtitle.setWordWrap(True)
        subtitle.setObjectName("subtleLabel")
        layout.addWidget(subtitle)

        filter_frame = QFrame()
        filter_layout = QGridLayout(filter_frame)
        filter_layout.setContentsMargins(0, 0, 0, 0)
        filter_layout.setHorizontalSpacing(10)
        filter_layout.setVerticalSpacing(6)

        filter_label = QLabel("Filter strength")
        filter_label.setObjectName("sectionLabel")
        filter_layout.addWidget(filter_label, 0, 0)

        self.filter_slider = QSlider(Qt.Orientation.Horizontal)
        self.filter_slider.setRange(0, 1000)
        self.filter_slider.setValue(round(self._filter_strength * 1000))
        filter_layout.addWidget(self.filter_slider, 0, 1)

        self.filter_spinbox = QDoubleSpinBox()
        self.filter_spinbox.setRange(0.0, 1.0)
        self.filter_spinbox.setDecimals(3)
        self.filter_spinbox.setSingleStep(0.01)
        self.filter_spinbox.setValue(self._filter_strength)
        self.filter_spinbox.setMinimumWidth(90)
        filter_layout.addWidget(self.filter_spinbox, 0, 2)

        self.filter_window_label = QLabel()
        self.filter_window_label.setObjectName("subtleLabel")
        filter_layout.addWidget(self.filter_window_label, 1, 0, 1, 3)
        layout.addWidget(filter_frame)

        self.joint_rows: list[JointControlRow] = []
        for index, (label, limits) in enumerate(zip(JOINT_LABELS, JOINT_LIMITS_RAD)):
            row = JointControlRow(index, label, limits[0], limits[1])
            row.value_changed.connect(self._on_manual_value_changed)
            row.set_manual_enabled(False)
            self.joint_rows.append(row)
            layout.addWidget(row)

        layout.addSpacing(4)
        mode_frame = QFrame()
        mode_layout = QGridLayout(mode_frame)
        mode_layout.setContentsMargins(0, 0, 0, 0)
        mode_layout.setHorizontalSpacing(12)
        mode_layout.setVerticalSpacing(10)

        mode_label = QLabel("Control mode")
        mode_label.setObjectName("sectionLabel")
        mode_layout.addWidget(mode_label, 0, 0, 1, 2)

        self.tracking_radio = QRadioButton("Tracking")
        self.manual_radio = QRadioButton("Manual")
        self.tracking_radio.setChecked(True)

        self.mode_group = QButtonGroup(self)
        self.mode_group.addButton(self.tracking_radio)
        self.mode_group.addButton(self.manual_radio)

        mode_layout.addWidget(self.tracking_radio, 1, 0)
        mode_layout.addWidget(self.manual_radio, 1, 1)
        layout.addWidget(mode_frame)

        button_row = QHBoxLayout()
        self.copy_tracking_button = QPushButton("Copy tracking to manual")
        self.copy_tracking_button.setEnabled(False)
        self.reset_graph_button = QPushButton("Reset graph")
        button_row.addWidget(self.copy_tracking_button)
        button_row.addWidget(self.reset_graph_button)
        layout.addLayout(button_row)
        layout.addStretch(1)

        self.tracking_radio.toggled.connect(self._on_mode_changed)
        self.copy_tracking_button.clicked.connect(self._copy_tracking_to_manual)
        self.reset_graph_button.clicked.connect(self._reset_graph)
        self.filter_slider.valueChanged.connect(self._on_filter_slider_changed)
        self.filter_spinbox.valueChanged.connect(self._on_filter_spinbox_changed)
        self._update_filter_strength_label()
        return panel

    def _build_plot_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("panel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)

        title_row = QHBoxLayout()
        title = QLabel("Real-time Joint Values")
        title.setObjectName("panelTitle")
        title_row.addWidget(title)
        title_row.addStretch(1)
        unit = QLabel("Unit: normalized [0, 1] · Window: 10 s")
        unit.setObjectName("subtleLabel")
        title_row.addWidget(unit)
        layout.addLayout(title_row)

        pg.setConfigOptions(antialias=True)
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground("#111720")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.18)
        self.plot_widget.setLabel("bottom", "Time", units="s")
        self.plot_widget.setLabel("left", "Normalized joint value")
        self.plot_widget.setYRange(-1.7, 2.1, padding=0.02)
        self.plot_widget.addLegend(offset=(12, 8), colCount=3)
        self.plot_widget.setMouseEnabled(x=True, y=True)
        layout.addWidget(self.plot_widget, 1)

        self.curves: list[pg.PlotDataItem] = []
        for index, label in enumerate(JOINT_LABELS):
            color = pg.intColor(index, hues=len(JOINT_LABELS), values=1, maxValue=225)
            curve = self.plot_widget.plot(
                [],
                [],
                name=label,
                pen=pg.mkPen(color=color, width=2),
            )
            self.curves.append(curve)

        return panel

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background: #0b1017;
                color: #e7edf5;
                font-size: 13px;
            }
            QFrame#panel, QFrame#statusFrame {
                background: #141b25;
                border: 1px solid #263142;
                border-radius: 10px;
            }
            QLabel#panelTitle {
                font-size: 16px;
                font-weight: 600;
                color: #f4f7fb;
            }
            QLabel#sectionLabel {
                font-weight: 600;
                color: #dce6f2;
            }
            QLabel#subtleLabel {
                color: #94a3b8;
            }
            QLabel#statusLabel {
                color: #cbd7e6;
                font-weight: 500;
            }
            QLabel#videoLabel {
                background: #05080d;
                border: 1px solid #293548;
                border-radius: 8px;
                color: #8291a5;
            }
            QSlider::groove:horizontal {
                height: 7px;
                background: #293548;
                border-radius: 3px;
            }
            QSlider::sub-page:horizontal {
                background: #3b82f6;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #eaf2ff;
                border: 2px solid #3b82f6;
                width: 17px;
                margin: -6px 0;
                border-radius: 9px;
            }
            QSlider:disabled::groove:horizontal {
                background: #202936;
            }
            QSlider:disabled::sub-page:horizontal {
                background: #52627a;
            }
            QSlider:disabled::handle:horizontal {
                background: #a8b4c4;
                border-color: #52627a;
            }
            QDoubleSpinBox {
                background: #0f1620;
                border: 1px solid #334155;
                border-radius: 6px;
                padding: 5px 7px;
                selection-background-color: #2563eb;
            }
            QPushButton {
                background: #243044;
                border: 1px solid #384963;
                border-radius: 7px;
                padding: 8px 11px;
            }
            QPushButton:hover {
                background: #2d3b52;
            }
            QPushButton:pressed {
                background: #1c2738;
            }
            QPushButton:disabled {
                color: #68778c;
                background: #1a222e;
                border-color: #283344;
            }
            QRadioButton, QCheckBox {
                spacing: 7px;
            }
            QSplitter::handle {
                background: #0b1017;
            }
            """
        )

    def _start_estimator_thread(self) -> None:
        self.estimator_thread = QThread(self)
        self.estimator_worker = EstimatorWorker(
            model_path=self._model_path,
            camera_serial=self._camera_serial,
            preferred_hand=self._preferred_hand,
            jitter_enabled=self._jitter_enabled,
        )
        self.estimator_worker.moveToThread(self.estimator_thread)

        self.estimator_thread.started.connect(self.estimator_worker.run)
        self.estimator_worker.frame_ready.connect(self._on_frame_ready)
        self.estimator_worker.failed.connect(self._on_estimator_failed)
        self.estimator_worker.finished.connect(self.estimator_thread.quit)
        self.filter_strength_changed.connect(self.estimator_worker.set_filter_strength)
        self.jitter_enabled_changed.connect(self.estimator_worker.set_jitter_enabled)
        self.estimator_thread.start()
        self.filter_strength_changed.emit(self._filter_strength)
        self.jitter_enabled_changed.emit(self._jitter_enabled)

    @Slot(object)
    def _on_frame_ready(self, estimate: HandEstimate) -> None:
        self._latest_estimate = estimate
        self.camera_status.setText(
            f"{estimate.camera_name} · S/N {estimate.camera_serial}"
        )
        self.mediapipe_status.setText(
            "MediaPipe: ON" if estimate.has_hand else "MediaPipe: no hand"
        )
        self.fps_status.setText(f"FPS: {estimate.fps:.1f}")

        if estimate.handedness:
            self.hand_status.setText(
                f"Hand: {estimate.handedness} ({estimate.handedness_score:.2f})"
            )
        else:
            self.hand_status.setText("Hand: --")

        frame = estimate.overlay_bgr if self.overlay_checkbox.isChecked() else estimate.raw_bgr
        self._show_bgr_image(frame)

        if estimate.joint_values is None:
            return

        self._last_tracking_values = estimate.joint_values.astype(np.float32, copy=True)
        if self._udp_publisher is not None:
            self._udp_publisher.publish(self._last_tracking_values, estimate)

        if self.tracking_radio.isChecked():
            for row, value in zip(self.joint_rows, self._last_tracking_values):
                row.set_value(float(value))
            values_for_plot = self._last_tracking_values
        else:
            values_for_plot = self._manual_values

        self._append_graph_sample(values_for_plot)

    def _show_bgr_image(self, bgr_image: np.ndarray) -> None:
        rgb_image = np.ascontiguousarray(bgr_image[:, :, ::-1])
        height, width, channels = rgb_image.shape
        bytes_per_line = channels * width
        qimage = QImage(
            rgb_image.data,
            width,
            height,
            bytes_per_line,
            QImage.Format.Format_RGB888,
        ).copy()
        pixmap = QPixmap.fromImage(qimage)
        scaled = pixmap.scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.video_label.setPixmap(scaled)

    def _append_graph_sample(self, values: np.ndarray) -> None:
        elapsed = time.perf_counter() - self._history_start
        self._time_history.append(elapsed)
        for history, value in zip(self._joint_history, values):
            history.append(float(value))

        x = np.fromiter(self._time_history, dtype=np.float64)
        for curve, history in zip(self.curves, self._joint_history):
            y = np.fromiter(history, dtype=np.float64)
            curve.setData(x, y)

        if len(x) > 1:
            right = x[-1]
            self.plot_widget.setXRange(
                max(0.0, right - HISTORY_SECONDS),
                max(HISTORY_SECONDS, right),
                padding=0.0,
            )

    @Slot(bool)
    def _on_mode_changed(self, tracking_enabled: bool) -> None:
        manual_enabled = not tracking_enabled
        for row in self.joint_rows:
            row.set_manual_enabled(manual_enabled)

        self.copy_tracking_button.setEnabled(manual_enabled)
        self.control_status.setText(
            "Control: Tracking" if tracking_enabled else "Control: Manual"
        )

        if manual_enabled:
            self._manual_values = self._last_tracking_values.copy()
            for row, value in zip(self.joint_rows, self._manual_values):
                row.set_value(float(value))

    @Slot(int, float)
    def _on_manual_value_changed(self, joint_index: int, value: float) -> None:
        self._manual_values[joint_index] = value

        # 이 위치에서 실제 로봇/시뮬레이터 전송 함수를 호출할 수 있습니다.
        # 예: self.retargeter.send(self._manual_values.copy())

    @Slot()
    def _copy_tracking_to_manual(self) -> None:
        self._manual_values = self._last_tracking_values.copy()
        for row, value in zip(self.joint_rows, self._manual_values):
            row.set_value(float(value))

    @Slot()
    def _reset_graph(self) -> None:
        self._history_start = time.perf_counter()
        self._time_history.clear()
        for history, curve in zip(self._joint_history, self.curves):
            history.clear()
            curve.setData([], [])
        self.plot_widget.setXRange(0.0, HISTORY_SECONDS, padding=0.0)

    @Slot(int)
    def _on_filter_slider_changed(self, slider_value: int) -> None:
        strength = slider_value / 1000.0
        with QSignalBlocker(self.filter_spinbox):
            self.filter_spinbox.setValue(strength)
        self._set_filter_strength(strength)

    @Slot(float)
    def _on_filter_spinbox_changed(self, value: float) -> None:
        strength = float(np.clip(value, 0.0, 1.0))
        with QSignalBlocker(self.filter_slider):
            self.filter_slider.setValue(round(strength * 1000))
        self._set_filter_strength(strength)

    def _set_filter_strength(self, strength: float) -> None:
        strength = float(np.clip(strength, 0.0, 1.0))
        if abs(strength - self._filter_strength) <= 1e-9:
            return
        self._filter_strength = strength
        self._update_filter_strength_label()
        self.filter_strength_changed.emit(strength)

    def _update_filter_strength_label(self) -> None:
        window_size = 1 + int(round(self._filter_strength * 14.0))
        self.filter_window_label.setText(
            f"Moving average window: {window_size} frames"
        )

    @Slot(str)
    def _on_estimator_failed(self, message: str) -> None:
        self.camera_status.setText("Camera: error")
        self.mediapipe_status.setText("MediaPipe: error")
        QMessageBox.critical(self, "Estimator error", message)

    def closeEvent(self, event: QCloseEvent) -> None:
        if hasattr(self, "estimator_worker"):
            self.estimator_worker.request_stop()
        if hasattr(self, "estimator_thread"):
            self.estimator_thread.quit()
            self.estimator_thread.wait(2500)
        if self._udp_publisher is not None:
            self._udp_publisher.close()
        event.accept()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the D405 MediaPipe hand-control UI.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to the MediaPipe Hand Landmarker .task model.",
    )
    parser.add_argument(
        "--camera-serial",
        type=str,
        default=None,
        help="Optional RealSense serial number to select a specific camera.",
    )
    parser.add_argument(
        "--preferred-hand",
        type=str,
        choices=("Left", "Right"),
        default=None,
        help="Prefer this handedness when MediaPipe detects both hands.",
    )
    parser.add_argument(
        "--filter-strength",
        type=float,
        default=DEFAULT_FILTER_STRENGTH,
        help="Moving-average smoothing strength in [0, 1]. Larger means stronger smoothing.",
    )
    parser.add_argument(
        "--jitter",
        type=_parse_bool_arg,
        default=DEFAULT_JITTER_ENABLED,
        help="Enable jitter deadband suppression: true or false.",
    )
    parser.add_argument(
        "--udp-host",
        type=str,
        default=None,
        help="UDP destination host for publishing joint values.",
    )
    parser.add_argument(
        "--udp-port",
        type=int,
        default=None,
        help="UDP destination port for publishing joint values.",
    )
    args, qt_args = parser.parse_known_args()

    if (args.udp_host is None) != (args.udp_port is None):
        parser.error("--udp-host and --udp-port must be provided together.")

    app = QApplication([sys.argv[0], *qt_args])
    app.setApplicationName("D405 Hand Control")
    window = HandControlWindow(
        model_path=args.model_path,
        camera_serial=args.camera_serial,
        preferred_hand=args.preferred_hand,
        filter_strength=args.filter_strength,
        jitter_enabled=args.jitter,
        udp_host=args.udp_host,
        udp_port=args.udp_port,
    )
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

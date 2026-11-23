import os
import numpy as np
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QGroupBox, QCheckBox,
    QMessageBox, QSplitter, QListWidget, QGraphicsRectItem
)
from PyQt6.QtCore import Qt, QRectF, pyqtSignal
from PyQt6.QtGui import QPen, QBrush, QColor, QKeySequence, QShortcut
import pyqtgraph as pg

from Annotation_App.processing import VentilationProcessor
from Annotation_App.annotations import (
    AnnotationStore,
    REGIME_CLASSES,
    REGIME_TO_ID,
)

pg.setConfigOption("background", "#1e1e1e")
pg.setConfigOption("foreground", "#d4d4d4")
pg.setConfigOptions(antialias=True)

REGIME_COLORS = {
    0: ("#3498db", (52, 152, 219, 55)),   # Controlled
    1: ("#95a5a6", (149, 165, 166, 55)),  # Unknown
    2: ("#e74c3c", (231, 76, 60, 60)),    # Apnea
    3: ("#f39c12", (243, 156, 18, 65)),   # Transition to Spontaneous ventilation
    4: ("#2ecc71", (46, 204, 113, 60)),   # Efficient spontaneous ventilation
}


# =====================================================================
# CUSTOM INTERACTIVE PLOT (SHIFT-BOX SELECT + CTRL+SHIFT DESELECT + DRAG)
# =====================================================================
class InteractiveRegimePlot(pg.PlotWidget):
    """
    Mirrors your EEG Annotator interaction model:
      1. Shift + Left-Drag: Rubber-band box selection (selects points inside box).
      2. Ctrl + Shift + Left-Drag: Rubber-band box deselection.
      3. Left-Drag on a point: Vertical drag-and-drop modification snapped to regime IDs.
         All selected points snap to the exact same regime level as the dragged point.
      4. Keyboard shortcuts 1..5 assign selected points directly to a regime.
    """

    valuesChanged = pyqtSignal(list, list)

    def __init__(self, title: str):
        super().__init__(title=title)
        self.y_min = 0
        self.y_max = len(REGIME_CLASSES) - 1
        self.default_brush_color = "#569cd6"
        self.selected_brush_color = "#ffeb3b"

        self.x_data = np.array([], dtype=np.float32)
        self.y_data = np.array([], dtype=np.int32)
        self.selected_indices = set()

        self.line_item = self.plot(pen=pg.mkPen("#b2bec3", width=1.8))
        self.scatter_item = pg.ScatterPlotItem(size=9, pen=pg.mkPen("#ffffff", width=0.8))
        self.addItem(self.scatter_item)

        self._mode = None  # None | 'drag_points' | 'box_select' | 'box_deselect'
        self._drag_anchor_idx = None
        self._drag_anchor_init_y = 0
        self._drag_start_y = 0.0

        self._box_start_view = None
        self._selection_rect = QGraphicsRectItem()
        self._selection_rect.setZValue(100)
        self._selection_rect.hide()
        self.addItem(self._selection_rect, ignoreBounds=True)

    def _style_box_rect(self, deselect: bool):
        color = QColor("#ff6b6b") if deselect else QColor("#ffeb3b")
        fill = QColor(255, 107, 107, 35) if deselect else QColor(255, 235, 59, 35)
        box_pen = QPen(color, 1.0, Qt.PenStyle.DashLine)
        box_pen.setCosmetic(True)
        self._selection_rect.setPen(box_pen)
        self._selection_rect.setBrush(QBrush(fill))

    def set_series(self, x_data: np.ndarray, y_data: np.ndarray):
        self.x_data = x_data.copy()
        self.y_data = y_data.copy()
        self.selected_indices.clear()
        self._redraw()

    def clear_selection(self):
        self.selected_indices.clear()
        self._redraw()

    def assign_selected_to_regime(self, regime_id: int):
        if not self.selected_indices:
            return
        indices = sorted(self.selected_indices)
        for idx in indices:
            self.y_data[idx] = regime_id
        self._redraw()
        self.valuesChanged.emit(indices, [regime_id] * len(indices))

    def _redraw(self):
        self.line_item.setData(self.x_data, self.y_data)
        brushes = [
            pg.mkBrush(
                self.selected_brush_color
                if i in self.selected_indices
                else REGIME_COLORS[int(self.y_data[i])][0]
            )
            for i in range(len(self.x_data))
        ]
        sizes = [12 if i in self.selected_indices else 8 for i in range(len(self.x_data))]
        self.scatter_item.setData(self.x_data, self.y_data, brush=brushes, size=sizes)

    def _find_point_near_pixel(self, pos_widget, radius_px: float = 14.0):
        if len(self.x_data) == 0:
            return None
        vb = self.getViewBox()
        mouse_scene = self.mapToScene(pos_widget)
        if not vb.sceneBoundingRect().contains(mouse_scene):
            return None

        best_idx = None
        best_dist = radius_px
        for i in range(len(self.x_data)):
            pt_scene = vb.mapViewToScene(pg.Point(self.x_data[i], self.y_data[i]))
            dist = (pt_scene - mouse_scene).manhattanLength()
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        return best_idx

    def mousePressEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton and len(self.x_data) > 0:
            vb = self.getViewBox()
            view_pos = vb.mapSceneToView(self.mapToScene(ev.pos()))
            mods = ev.modifiers()

            has_shift = bool(mods & Qt.KeyboardModifier.ShiftModifier)
            has_ctrl = bool(mods & Qt.KeyboardModifier.ControlModifier)

            if has_shift and has_ctrl:
                self._mode = "box_deselect"
                self._style_box_rect(deselect=True)
                self._box_start_view = view_pos
                self._selection_rect.setRect(QRectF(view_pos.x(), view_pos.y(), 0, 0))
                self._selection_rect.show()
                ev.accept()
                return

            if has_shift:
                self._mode = "box_select"
                self._style_box_rect(deselect=False)
                self._box_start_view = view_pos
                self._selection_rect.setRect(QRectF(view_pos.x(), view_pos.y(), 0, 0))
                self._selection_rect.show()
                ev.accept()
                return

            hit_idx = self._find_point_near_pixel(ev.pos())
            if hit_idx is not None:
                self._mode = "drag_points"
                self._drag_anchor_idx = hit_idx
                self._drag_anchor_init_y = int(self.y_data[hit_idx])
                self._drag_start_y = view_pos.y()

                if hit_idx not in self.selected_indices:
                    self.selected_indices = {hit_idx}

                self._redraw()
                ev.accept()
                return

        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        vb = self.getViewBox()
        view_pos = vb.mapSceneToView(self.mapToScene(ev.pos()))

        if self._mode in ("box_select", "box_deselect") and self._box_start_view is not None:
            x0, y0 = self._box_start_view.x(), self._box_start_view.y()
            x1, y1 = view_pos.x(), view_pos.y()
            rect = QRectF(min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0))
            self._selection_rect.setRect(rect)
            ev.accept()
            return

        if self._mode == "drag_points" and self._drag_anchor_idx is not None:
            delta_y = view_pos.y() - self._drag_start_y
            target_val = int(
                np.clip(round(self._drag_anchor_init_y + delta_y), self.y_min, self.y_max)
            )
            for idx in self.selected_indices:
                self.y_data[idx] = target_val
            self._redraw()
            ev.accept()
            return

        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            if self._mode in ("box_select", "box_deselect"):
                rect = self._selection_rect.rect()
                self._selection_rect.hide()

                in_box = {
                    int(idx)
                    for idx in np.where(
                        (self.x_data >= rect.left())
                        & (self.x_data <= rect.right())
                        & (self.y_data >= rect.top())
                        & (self.y_data <= rect.bottom())
                    )[0]
                }

                if self._mode == "box_deselect":
                    self.selected_indices -= in_box
                else:
                    self.selected_indices |= in_box

                self._redraw()
                self._mode = None
                ev.accept()
                return

            if self._mode == "drag_points":
                indices = sorted(self.selected_indices)
                new_vals = [int(self.y_data[i]) for i in indices]
                self.valuesChanged.emit(indices, new_vals)
                self._mode = None
                self._drag_anchor_idx = None
                ev.accept()
                return

        super().mouseReleaseEvent(ev)


# =====================================================================
# MAIN APPLICATION WINDOW
# =====================================================================
class VentilationAnnotatorWindow(QMainWindow):
    def __init__(self, processor: VentilationProcessor, store: AnnotationStore):
        super().__init__()
        self.processor = processor
        self.store = store

        self.setWindowTitle("Ventilation Regime Annotator (128 Hz) — General Anesthesia Recovery")
        self.resize(1540, 1020)

        self.event_line_pairs = []
        self.regime_shade_items = []

        self._setup_ui()
        self._setup_shortcuts()

    def _setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        # -------------------------------------------------------------
        # LEFT: Synchronized Waveform & Annotation Plots
        # -------------------------------------------------------------
        plots_container = QWidget()
        plots_layout = QVBoxLayout(plots_container)
        plots_layout.setContentsMargins(0, 0, 0, 0)
        plots_layout.setSpacing(3)
        splitter.addWidget(plots_container)

        # Plot 0: Hypnotic Concentration (0 to 8)
        self.p_hyp = pg.PlotWidget(title="Hypnotic Concentration (Propofol µg/mL or Sevoflurane %)")
        self.p_hyp.setLabel("left", "Conc.")
        self.p_hyp.setYRange(0, 8, padding=0.05)
        self.p_hyp.getAxis("left").setTicks([[(0, "0"), (2, "2"), (4, "4"), (6, "6"), (8, "8")]])
        self.p_hyp.showGrid(x=True, y=True, alpha=0.3)
        self.p_hyp.addLegend(offset=(10, 5))
        plots_layout.addWidget(self.p_hyp, stretch=1)

        # Plot 1: EEG Spectrogram
        self.p_spec = pg.PlotWidget(title="1. EEG Spectrogram (0.5 - 35 Hz) — Emergence Context")
        self.p_spec.setLabel("left", "Freq", units="Hz")
        self.img_spec = pg.ImageItem()
        self.p_spec.addItem(self.img_spec)
        self.img_spec.setColorMap(pg.colormap.get("turbo"))
        plots_layout.addWidget(self.p_spec, stretch=2)

        # Plot 2: Continuous Capnography (CO2)
        self.p_co2 = pg.PlotWidget(title="2. Capnography CO2 (128 Hz)")
        self.p_co2.setLabel("left", "CO2", units="mmHg")
        self.p_co2.setYRange(-5, 65)
        self.p_co2.showGrid(x=True, y=True, alpha=0.3)
        self.curve_co2 = self.p_co2.plot(pen=pg.mkPen("#4fc1ff", width=1.1))
        self.curve_co2.setDownsampling(auto=True, method="peak")
        self.curve_co2.setClipToView(True)
        plots_layout.addWidget(self.p_co2, stretch=2)

        # Plot 3: Continuous Airway Pressure (AWP) + Crop Lines + Regime Shading
        self.p_awp = pg.PlotWidget(
            title="3. Airway Pressure AWP (128 Hz) — Red Lines = Crop Bounds | Dashed = Transition Events"
        )
        self.p_awp.setLabel("left", "AWP", units="hPa")
        self.p_awp.setYRange(-10, 50)
        self.p_awp.showGrid(x=True, y=True, alpha=0.3)
        self.curve_awp = self.p_awp.plot(pen=pg.mkPen("#e0e0e0", width=1.0))
        self.curve_awp.setDownsampling(auto=True, method="peak")
        self.curve_awp.setClipToView(True)
        self.p_awp.addLine(y=0, pen=pg.mkPen("#7f8c8d", style=Qt.PenStyle.DashLine))

        crop_pen = pg.mkPen("#ff4d4d", width=2.5)
        crop_hover_pen = pg.mkPen("#ff8080", width=3.5)

        self.crop_start_line = pg.InfiniteLine(
            pos=0.0,
            angle=90,
            movable=True,
            pen=crop_pen,
            hoverPen=crop_hover_pen,
            label="Keep Start",
            labelOpts={"position": 0.12, "color": "#ff4d4d", "movable": True},
        )
        self.crop_end_line = pg.InfiniteLine(
            pos=0.0,
            angle=90,
            movable=True,
            pen=crop_pen,
            hoverPen=crop_hover_pen,
            label="Keep End",
            labelOpts={"position": 0.12, "color": "#ff4d4d", "movable": True},
        )
        self.crop_start_line.setZValue(15)
        self.crop_end_line.setZValue(15)
        self.crop_start_line.sigPositionChanged.connect(self._on_crop_lines_moved)
        self.crop_end_line.sigPositionChanged.connect(self._on_crop_lines_moved)

        self.p_awp.addItem(self.crop_start_line)
        self.p_awp.addItem(self.crop_end_line)
        plots_layout.addWidget(self.p_awp, stretch=3)

        # Plot 4: Lung Compliance
        self.p_compl = pg.PlotWidget(title="4. Dynamic Lung Compliance (Primus / Solar8000)")
        self.p_compl.setLabel("left", "Compl.", units="mL/mbar")
        self.p_compl.showGrid(x=True, y=True, alpha=0.3)
        self.curve_compl = self.p_compl.plot(
            pen=pg.mkPen("#55efc4", width=1.2), symbol="o", symbolSize=4, symbolBrush="#55efc4"
        )
        plots_layout.addWidget(self.p_compl, stretch=1)

        # Plot 5: Interactive Ventilation Regime Ground-Truth Editor
        self.p_regime = InteractiveRegimePlot(
            title="5. Ventilation Regime — Drag Points | Shift+Box Select | Keys [1-5] Assign"
        )
        self.p_regime.setLabel("left", "Regime")
        self.p_regime.setLabel("bottom", "Time", units="s")
        self.p_regime.setYRange(-0.5, len(REGIME_CLASSES) - 0.5)
        self.p_regime.showGrid(x=True, y=True, alpha=0.3)

        short_names = [
            "Controlled",
            "Unknown",
            "Apnea",
            "Transition Spont.",
            "Efficient Spont.",
        ]
        self.p_regime.getAxis("left").setTicks([[(i, name) for i, name in enumerate(short_names)]])
        self.p_regime.valuesChanged.connect(self._on_regime_points_modified)
        plots_layout.addWidget(self.p_regime, stretch=2)

        # Synchronize X-axes across all 6 plots
        for plot in [self.p_hyp, self.p_spec, self.p_co2, self.p_compl, self.p_regime]:
            plot.setXLink(self.p_awp)

        # -------------------------------------------------------------
        # RIGHT: Control Panel
        # -------------------------------------------------------------
        control_panel = QWidget()
        control_panel.setFixedWidth(355)
        ctrl_layout = QVBoxLayout(control_panel)
        splitter.addWidget(control_panel)

        # Group 1: Dataset & Sample
        grp_file = QGroupBox("1. Dataset, Sample & Hypnotic Agent")
        lay_file = QVBoxLayout(grp_file)

        self.lbl_sample_id = QLabel("Sample ID: No file loaded")
        self.lbl_sample_id.setWordWrap(True)
        self.lbl_sample_id.setStyleSheet("font-weight: bold; color: #4ec9b0;")
        lay_file.addWidget(self.lbl_sample_id)

        self.lbl_file_status = QLabel("Status: Waiting for VitalDB .npz file...")
        self.lbl_file_status.setWordWrap(True)
        self.lbl_file_status.setStyleSheet("color: #aaaaaa;")
        lay_file.addWidget(self.lbl_file_status)

        self.lbl_hypnotic_agent = QLabel("Hypnotic Agent (auto): —")
        self.lbl_hypnotic_agent.setStyleSheet("font-weight: bold; color: #4fc1ff;")
        lay_file.addWidget(self.lbl_hypnotic_agent)

        btn_load = QPushButton("📂 Open VitalDB Case (.npz)")
        btn_load.setStyleSheet("font-weight: bold; padding: 6px;")
        btn_load.clicked.connect(self._open_recording_file)
        lay_file.addWidget(btn_load)

        win_row = QHBoxLayout()
        for label, sec in [("Last 5m", 300.0), ("Last 20m", 1200.0), ("Full Case", 0.0)]:
            b = QPushButton(label)
            b.clicked.connect(lambda _c, s=sec: self._set_time_window(s))
            win_row.addWidget(b)
        lay_file.addLayout(win_row)
        ctrl_layout.addWidget(grp_file)

        # Group 2: Valid Recording Crop
        grp_crop = QGroupBox("2. Valid Recording Crop (AWP Plot)")
        lay_crop = QVBoxLayout(grp_crop)
        self.lbl_crop = QLabel("Keep Signal: —")
        self.lbl_crop.setStyleSheet("font-weight: bold; color: #4ec9b0;")
        lay_crop.addWidget(self.lbl_crop)
        lay_crop.addWidget(
            QLabel("Drag red 'Keep Start' and 'Keep End' bars on AWP to trim pre-induction or post-extubation noise.")
        )
        ctrl_layout.addWidget(grp_crop)

        # Group 3: Direct Regime Assignment & Interactive Guide
        grp_edit = QGroupBox("3. Ventilation Regime Editing (Plot 5)")
        lay_edit = QVBoxLayout(grp_edit)
        lay_edit.addWidget(
            QLabel(
                "• Shift + Left-Drag Box: Select windows\n"
                "• Ctrl + Shift + Drag Box: Deselect windows\n"
                "• Drag Selected Point OR press [1–5]: Set class"
            )
        )

        for idx, cls_name in enumerate(REGIME_CLASSES):
            hex_col, _ = REGIME_COLORS[idx]
            btn = QPushButton(f"[{idx + 1}] {cls_name}")
            btn.setStyleSheet(f"border-left: 6px solid {hex_col}; text-align: left; padding: 5px;")
            btn.clicked.connect(lambda _c, r_id=idx: self.p_regime.assign_selected_to_regime(r_id))
            lay_edit.addWidget(btn)

        btn_clear_sel = QPushButton("✖ Clear Window Selection")
        btn_clear_sel.clicked.connect(self.p_regime.clear_selection)
        lay_edit.addWidget(btn_clear_sel)
        ctrl_layout.addWidget(grp_edit)

        # Group 4: Transition Event Markers
        grp_events = QGroupBox("4. Recovery Milestones (CO2 & AWP)")
        lay_events = QVBoxLayout(grp_events)

        btn_row_ev = QHBoxLayout()
        btn_spont = QPushButton("+ Spont. Onset")
        btn_spont.setStyleSheet("background-color: #8a6d1b; color: white;")
        btn_spont.clicked.connect(lambda: self._add_event_marker("spont_onset"))
        btn_row_ev.addWidget(btn_spont)

        btn_extub = QPushButton("+ Extubation")
        btn_extub.setStyleSheet("background-color: #8a2be2; color: white;")
        btn_extub.clicked.connect(lambda: self._add_event_marker("extubation"))
        btn_row_ev.addWidget(btn_extub)
        lay_events.addLayout(btn_row_ev)

        self.list_events = QListWidget()
        self.list_events.setMaximumHeight(85)
        lay_events.addWidget(self.list_events)

        btn_del_event = QPushButton("🗑 Remove Selected Event")
        btn_del_event.clicked.connect(self._remove_selected_event)
        lay_events.addWidget(btn_del_event)
        ctrl_layout.addWidget(grp_events)

        # Group 5: Selective Save & Validation
        grp_save = QGroupBox("5. Selective Save & ML Dataset Export")
        lay_save = QVBoxLayout(grp_save)

        self.chk_save_regimes = QCheckBox("Save Ventilation Regimes (5 classes)")
        self.chk_save_regimes.setChecked(True)
        lay_save.addWidget(self.chk_save_regimes)

        self.chk_save_events = QCheckBox("Save Transition Events (Spont. / Extub.)")
        self.chk_save_events.setChecked(True)
        lay_save.addWidget(self.chk_save_events)

        self.chk_save_crop = QCheckBox("Save Crop Boundaries (Keep Start / End)")
        self.chk_save_crop.setChecked(True)
        lay_save.addWidget(self.chk_save_crop)

        self.btn_save = QPushButton("💾 Validate & Save Annotated Sample (.npz)")
        self.btn_save.setEnabled(False)
        self.btn_save.setStyleSheet(
            "background-color: #2e7d32; color: white; font-weight: bold; padding: 10px;"
        )
        self.btn_save.clicked.connect(self._save_annotations)
        lay_save.addWidget(self.btn_save)
        ctrl_layout.addWidget(grp_save)
        ctrl_layout.addStretch()

    def _setup_shortcuts(self):
        for idx in range(len(REGIME_CLASSES)):
            sc = QShortcut(QKeySequence(str(idx + 1)), self)
            sc.activated.connect(lambda r_id=idx: self.p_regime.assign_selected_to_regime(r_id))
        QShortcut(QKeySequence("Ctrl+S"), self).activated.connect(self._save_annotations)

    # =================================================================
    # FILE LOADING & EXISTING ANNOTATION DETECTION
    # =================================================================
    def _open_recording_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open VitalDB Case (.npz) or Annotated Archive",
            "",
            "VitalDB Archives (*.npz)",
        )
        if not file_path:
            return

        try:
            # Case 1: User opened an already-annotated *_vent_annotated.npz directly
            if AnnotationStore.is_annotated_npz(file_path):
                self._load_existing_annotated_file(file_path)
                return

            # Case 2: Check if sidecar *_vent_annotated.npz exists next to raw file
            sidecar_annotated = AnnotationStore.get_default_annotated_path(file_path)
            if sidecar_annotated != file_path and AnnotationStore.is_annotated_npz(sidecar_annotated):
                reply = QMessageBox.question(
                    self,
                    "Existing Annotations Detected",
                    f"An annotated ventilation file already exists for this recording:\n"
                    f"{os.path.basename(sidecar_annotated)}\n\n"
                    f"Would you like to load the existing annotations to continue editing?\n"
                    f"(Click 'No' to start fresh from the raw recording.)",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.Yes,
                )
                if reply == QMessageBox.StandardButton.Yes:
                    self._load_existing_annotated_file(sidecar_annotated, raw_fallback_path=file_path)
                    return

            # Case 3: Load fresh raw VitalDB recording
            signals = self.processor.extract_signals_from_file(file_path)
            hyp_tracks, detected_hyp = self.processor.extract_hypnotics_from_file(file_path)

            sample_id = os.path.splitext(os.path.basename(file_path))[0]
            if sample_id.endswith("_vent_annotated"):
                sample_id = sample_id[: -len("_vent_annotated")]

            t_windows = self.processor.compute_window_grid(
                signals["t_common"], win_sec=self.store.win_sec, step_sec=self.store.step_sec
            )
            initial_regimes = self.processor.estimate_initial_regimes(
                signals["t_common"], signals["awp"], signals["co2"], t_windows
            )

            self._clear_visual_events()
            self.store.initialize_record(
                sample_id=sample_id,
                t_common=signals["t_common"],
                awp_data=signals["awp"],
                co2_data=signals["co2"],
                eeg_data=signals["eeg"],
                ecg_data=signals["ecg"],
                compl_time=signals["compl_time"],
                compl_val=signals["compl_val"],
                t_windows=t_windows,
                initial_regimes=initial_regimes,
                source_path=file_path,
                hypnotic_type=detected_hyp,
                hypnotic_tracks=hyp_tracks,
            )

            self._reset_save_checkboxes(True, True, True)
            self._render_store_to_plots()
            self.lbl_file_status.setText("Status: New Raw Recording (Auto-Classified)")
            self.lbl_file_status.setStyleSheet("color: #ce9178; font-weight: bold;")
            self.btn_save.setEnabled(True)

        except Exception as e:
            QMessageBox.critical(self, "Error Loading File", f"Could not load recording:\n{e}")

    def _load_existing_annotated_file(self, annotated_path: str, raw_fallback_path: str = ""):
        self._clear_visual_events()

        hyp_tracks, detected_hyp = self.processor.extract_hypnotics_from_file(annotated_path)
        if not hyp_tracks and raw_fallback_path:
            hyp_tracks, detected_hyp = self.processor.extract_hypnotics_from_file(raw_fallback_path)

        self.store.load_from_annotated_npz(
            annotated_path,
            fallback_initial_regimes_fn=self.processor.estimate_initial_regimes,
            hypnotic_tracks=hyp_tracks,
        )
        if self.store.hypnotic_type == "unknown" and detected_hyp != "unknown":
            self.store.hypnotic_type = detected_hyp

        sec = self.store.loaded_sections
        self._reset_save_checkboxes(
            regimes=sec.get("regimes", True),
            events=sec.get("events", True),
            crop=sec.get("crop", True),
        )

        self._render_store_to_plots()

        for ev in self.store.events:
            self._spawn_event_line_pair(ev.time, ev.event_type)
        self._refresh_event_list()

        saved_names = [k for k, v in sec.items() if v]
        self.lbl_file_status.setText(
            f"Status: Resumed ({os.path.basename(annotated_path)})\n"
            f"Previously saved: {', '.join(saved_names) if saved_names else 'none'}"
        )
        self.lbl_file_status.setStyleSheet("color: #4fc1ff; font-weight: bold;")
        self.btn_save.setEnabled(True)

    def _clear_visual_events(self):
        for co2_line, awp_line in self.event_line_pairs:
            self.p_co2.removeItem(co2_line)
            self.p_awp.removeItem(awp_line)
        self.event_line_pairs.clear()
        self.list_events.clear()

    def _reset_save_checkboxes(self, regimes=True, events=True, crop=True):
        self.chk_save_regimes.setChecked(regimes)
        self.chk_save_events.setChecked(events)
        self.chk_save_crop.setChecked(crop)

    def _render_store_to_plots(self):
        self.lbl_sample_id.setText(f"Sample ID: {self.store.sample_id}")
        self.lbl_hypnotic_agent.setText(f"Hypnotic Agent (auto): {self.store.hypnotic_type}")

        # Plot 0: Hypnotic Concentration
        self.p_hyp.clear()
        self.p_hyp.setYRange(0, 8, padding=0.05)
        if self.store.hypnotic_tracks:
            palette = ["#4fc1ff", "#ff9d00", "#89d185", "#f48771"]
            for i, trk in enumerate(self.store.hypnotic_tracks):
                pen = pg.mkPen(palette[i % len(palette)], width=2)
                self.p_hyp.plot(trk["time"], trk["val"], pen=pen, name=trk["name"])
            self.p_hyp.setTitle(f"Hypnotic Concentration (0–8) — Agent: {self.store.hypnotic_type}")
        else:
            self.p_hyp.setTitle("Hypnotic Concentration (0–8) — No concentration track found")

        # Plot 1: EEG Spectrogram
        if len(self.store.eeg_data) > int(self.processor.fs * 3) and np.any(self.store.eeg_data != 0):
            eeg_filt = self.processor.filter_eeg(self.store.eeg_data)
            freqs, t_spec, Sxx_db = self.processor.compute_spectrogram(eeg_filt)
            self.img_spec.setImage(Sxx_db.T, autoLevels=True)
            dt = t_spec[1] - t_spec[0] if len(t_spec) > 1 else 1.0
            df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
            t0_eeg = float(self.store.t_common[0]) if len(self.store.t_common) > 0 else 0.0
            self.img_spec.setRect(t0_eeg + t_spec[0], freqs[0], len(t_spec) * dt, len(freqs) * df)

        # Plots 2, 3, 4, 5
        self.curve_co2.setData(self.store.t_common, self.store.co2_data)
        self.curve_awp.setData(self.store.t_common, self.store.awp_data)
        self.curve_compl.setData(self.store.compl_time, self.store.compl_val)
        self.p_regime.set_series(self.store.t_windows, self.store.regimes)

        self._redraw_regime_shading()
        self._set_crop_bounds_ui(self.store.keep_start_sec, self.store.keep_end_sec)
        self._set_time_window(1200.0)

    def _redraw_regime_shading(self):
        """Updates translucent regime overlays on the AWP plot whenever labels change."""
        for item in self.regime_shade_items:
            self.p_awp.removeItem(item)
        self.regime_shade_items.clear()

        for seg_start, seg_end, regime_id in self.store.get_contiguous_intervals():
            hex_col, rgba = REGIME_COLORS[regime_id]
            region = pg.LinearRegionItem(
                values=(seg_start, seg_end),
                orientation="vertical",
                brush=pg.mkBrush(*rgba),
                pen=pg.mkPen(None),
                movable=False,
            )
            region.setZValue(-10)
            self.p_awp.addItem(region)
            self.regime_shade_items.append(region)

    def _set_time_window(self, seconds: float):
        if len(self.store.t_common) == 0:
            return
        min_t = float(self.store.t_common[0])
        max_t = float(self.store.t_common[-1])
        if seconds > 0:
            self.p_awp.setXRange(max(min_t, max_t - seconds), max_t, padding=0.01)
        else:
            self.p_awp.setXRange(min_t, max_t, padding=0.01)

    # =================================================================
    # CROP BOUNDARY LINES & POINT DRAG HANDLERS
    # =================================================================
    def _on_crop_lines_moved(self):
        start_val = float(self.crop_start_line.value())
        end_val = float(self.crop_end_line.value())

        if start_val > end_val:
            if self.sender() is self.crop_start_line:
                start_val = end_val
                self.crop_start_line.blockSignals(True)
                self.crop_start_line.setValue(start_val)
                self.crop_start_line.blockSignals(False)
            else:
                end_val = start_val
                self.crop_end_line.blockSignals(True)
                self.crop_end_line.setValue(end_val)
                self.crop_end_line.blockSignals(False)

        self.store.set_crop_bounds(start_val, end_val)
        self.lbl_crop.setText(
            f"Keep Signal: {self.store.keep_start_sec:.1f}s — {self.store.keep_end_sec:.1f}s"
        )

    def _set_crop_bounds_ui(self, start_t: float, end_t: float):
        self.store.set_crop_bounds(start_t, end_t)
        min_t = float(self.store.t_common[0]) if len(self.store.t_common) > 0 else start_t
        max_t = float(self.store.t_common[-1]) if len(self.store.t_common) > 0 else end_t

        for line, pos in (
            (self.crop_start_line, self.store.keep_start_sec),
            (self.crop_end_line, self.store.keep_end_sec),
        ):
            line.blockSignals(True)
            line.setBounds([min_t, max_t])
            line.setValue(pos)
            line.blockSignals(False)

        self.lbl_crop.setText(
            f"Keep Signal: {self.store.keep_start_sec:.1f}s — {self.store.keep_end_sec:.1f}s"
        )

    def _on_regime_points_modified(self, indices: list, new_values: list):
        for idx, val in zip(indices, new_values):
            self.store.set_regimes_at_indices([idx], val)
        self._redraw_regime_shading()

    # =================================================================
    # POINT EVENT MARKERS (SYNCHRONIZED ON CO2 & AWP)
    # =================================================================
    def _add_event_marker(self, event_type: str):
        if len(self.store.t_common) == 0:
            return
        x_min, x_max = self.p_awp.getViewBox().viewRange()[0]
        event_time = 0.5 * (x_min + x_max)
        self.store.add_event(event_time, event_type)
        self._spawn_event_line_pair(event_time, event_type)
        self._refresh_event_list()

    def _spawn_event_line_pair(self, event_time: float, event_type: str):
        color = "#ffd700" if event_type == "spont_onset" else "#da70d6"
        hover_color = "#fff176" if event_type == "spont_onset" else "#f3a6ff"
        label_txt = "Spont. Onset" if event_type == "spont_onset" else "Extubation"

        co2_line = pg.InfiniteLine(
            pos=event_time,
            angle=90,
            movable=True,
            pen=pg.mkPen(color, width=2.5, style=Qt.PenStyle.DashLine),
            hoverPen=pg.mkPen(hover_color, width=3.5, style=Qt.PenStyle.DashLine),
            label=label_txt,
            labelOpts={"position": 0.88, "color": color, "movable": True},
        )
        awp_line = pg.InfiniteLine(
            pos=event_time,
            angle=90,
            movable=True,
            pen=pg.mkPen(color, width=2.5, style=Qt.PenStyle.DashLine),
            hoverPen=pg.mkPen(hover_color, width=3.5, style=Qt.PenStyle.DashLine),
            label=label_txt,
            labelOpts={"position": 0.88, "color": color, "movable": True},
        )
        co2_line.setZValue(25)
        awp_line.setZValue(25)

        self.p_co2.addItem(co2_line)
        self.p_awp.addItem(awp_line)

        pair = (co2_line, awp_line)
        self.event_line_pairs.append(pair)

        co2_line.sigPositionChanged.connect(
            lambda moved=co2_line, p=pair: self._on_event_line_moved(moved, p)
        )
        awp_line.sigPositionChanged.connect(
            lambda moved=awp_line, p=pair: self._on_event_line_moved(moved, p)
        )

    def _on_event_line_moved(self, moved_line: pg.InfiniteLine, pair: tuple):
        new_pos = float(moved_line.value())
        idx = self.event_line_pairs.index(pair)
        self.store.update_event_time(idx, new_pos)

        for line in pair:
            if line is not moved_line:
                line.blockSignals(True)
                line.setValue(new_pos)
                line.blockSignals(False)

        self._refresh_event_list()

    def _refresh_event_list(self):
        self.list_events.clear()
        for i, ev in enumerate(self.store.events):
            self.list_events.addItem(f"[{i + 1}] {ev.event_type} @ {ev.time:.2f} s")

    def _remove_selected_event(self):
        row = self.list_events.currentRow()
        if row < 0 or row >= len(self.store.events):
            return
        self.store.remove_event(row)
        co2_line, awp_line = self.event_line_pairs.pop(row)
        self.p_co2.removeItem(co2_line)
        self.p_awp.removeItem(awp_line)
        self._refresh_event_list()

    # =================================================================
    # DIRECT SIDE-BY-SIDE SELECTIVE MANUAL SAVING
    # =================================================================
    def _save_annotations(self):
        if not self.store.source_path:
            QMessageBox.warning(self, "No Source File", "No recording path is associated with the current session.")
            return

        save_regimes = self.chk_save_regimes.isChecked()
        save_events = self.chk_save_events.isChecked()
        save_crop = self.chk_save_crop.isChecked()

        if not any([save_regimes, save_events, save_crop]):
            QMessageBox.warning(
                self,
                "Nothing Selected to Save",
                "Please check at least one annotation section (Regimes, Events, or Crop) before saving.",
            )
            return

        save_path = AnnotationStore.get_default_annotated_path(self.store.source_path)

        try:
            summary = self.store.save_annotated_npz(
                save_path,
                save_regimes=save_regimes,
                save_events=save_events,
                save_crop=save_crop,
            )
            self.store.source_path = save_path
            saved_list = [k for k, v in summary["saved_sections"].items() if v]

            self.lbl_file_status.setText(
                f"Status: Saved ({os.path.basename(save_path)})\n"
                f"Saved sections: {', '.join(saved_list)}"
            )
            self.lbl_file_status.setStyleSheet("color: #4ec9b0; font-weight: bold;")

            QMessageBox.information(
                self,
                "Saved Successfully",
                f"Saved annotated ventilation dataset alongside original recording:\n{save_path}\n\n"
                f"• Sample ID: {self.store.sample_id}\n"
                f"• Hypnotic Agent: {self.store.hypnotic_type}\n"
                f"• Saved Sections: {', '.join(saved_list)}\n"
                f"• Kept Duration: {summary['kept_duration_sec']:.1f}s ({summary['kept_samples']} samples)\n"
                f"• Valid Labeled Windows: {summary['kept_windows']}\n"
                f"• Events Inside Crop: {summary['kept_events']}",
            )
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save file:\n{e}")
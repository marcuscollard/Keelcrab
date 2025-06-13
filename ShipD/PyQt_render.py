import sys
import os
import tempfile
import csv
from pathlib import Path

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
import pyvista as pv
from pyvistaqt import QtInteractor

# -----------------------------------------------------------------------------
#  3rd‑party ShipD geometry engine
# -----------------------------------------------------------------------------
from HullParameterization import Hull_Parameterization as HP

# -----------------------------------------------------------------------------
#  Where custom parameter sets are persisted            ~/ShipD/custom_hulls.csv
# -----------------------------------------------------------------------------
SAVE_PATH = Path.home() / "ShipD" / "custom_hulls.csv"


class HullViewer(QtWidgets.QMainWindow):
    """Interactive 3‑D viewer for ShipD hulls with numeric parameter editing,
    optional UV layout preview and one‑click CSV persistence (Ctrl+S).
    """

    # ------------------------------------------------------------------
    #  Constructor
    # ------------------------------------------------------------------
    def __init__(self, vectors: np.ndarray | None = None) -> None:
        super().__init__()
        self.setWindowTitle("Live Hull Viewer")
        self.resize(1280, 800)

        # ------------------------------------------------------------------
        # 1)  Load / infer parameter vectors
        # ------------------------------------------------------------------
        dataset_num = 4  # choose which of the demo CSVs to preload (1‑based)
        datasets = [
            "Constrained_Randomized_Set_1",
            "Constrained_Randomized_Set_2",
            "Constrained_Randomized_Set_3",
            "Diffusion_Aug_Set_1",
            "Diffusion_Aug_Set_2",
        ]
        csv_path = (
            Path(__file__).resolve().parent
            / "Ship_D_Dataset"
            / datasets[dataset_num - 1]
            / "Input_Vectors.csv"
        )

        if vectors is not None:
            self.vectors: np.ndarray = np.asarray(vectors, dtype=float)
        elif csv_path.is_file():
            try:
                self.vectors = np.loadtxt(csv_path, delimiter=",", skiprows=1, dtype=float)
            except Exception:
                self.vectors = np.empty((0,))
        else:
            self.vectors = np.empty((0,))

        self.current_idx = 0
        if self.vectors.ndim == 2 and self.vectors.shape[0] > 0:
            self.params: list[float] = list(self.vectors[self.current_idx])
        else:
            # Determine the required parameter length by probing HP
            length = 0
            while True:
                try:
                    _ = HP([0.0] * length)
                    length += 1
                except IndexError:
                    break
            self.params = [0.5] * length

        # ------------------------------------------------------------------
        # 2)  Texture / UV settings
        # ------------------------------------------------------------------
        self.texture_height = 200
        self.texture_width = 3 * self.texture_height  # 6:1 aspect ratio
        self.num_stripes = 10
        self.compute_uv = True  # toggled via View‑menu

        # ------------------------------------------------------------------
        # 3)  3‑D viewport (central widget)
        # ------------------------------------------------------------------
        self.frame = QtWidgets.QFrame()
        self._vlayout = QtWidgets.QVBoxLayout(self.frame)
        self.plotter = QtInteractor(self.frame)
        self._vlayout.addWidget(self.plotter.interactor)
        self.setCentralWidget(self.frame)

        # ------------------------------------------------------------------
        # 4)  Pattern preview below the viewport
        # ------------------------------------------------------------------
        self.pattern_selector = QtWidgets.QComboBox()
        self.pattern_selector.addItems(
            ["Horizontal", "Vertical", "45° Diagonal", "135° Diagonal"]
        )
        self.pattern_selector.currentIndexChanged.connect(self.update_texture)
        self._vlayout.addWidget(self.pattern_selector)

        self.stripe_scene = QtWidgets.QGraphicsScene()
        self.stripe_view = QtWidgets.QGraphicsView(self.stripe_scene)
        self.stripe_view.setFixedSize(self.texture_width, self.texture_height)
        self._vlayout.addWidget(self.stripe_view)

        # ------------------------------------------------------------------
        # 5)  UV layout dock (detachable)
        # ------------------------------------------------------------------
        self.uv_dock = QtWidgets.QDockWidget("UV Layout", self)
        self.uv_view = QtWidgets.QLabel(alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        self.uv_view.setMinimumSize(400, 400)
        self.uv_dock.setWidget(self.uv_view)
        self.uv_dock.setAllowedAreas(QtCore.Qt.AllDockWidgetAreas)
        self.uv_dock.setFeatures(
            QtWidgets.QDockWidget.DockWidgetMovable
            | QtWidgets.QDockWidget.DockWidgetFloatable
        )
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self.uv_dock)

        view_menu = self.menuBar().addMenu("View")
        self.action_show_uv = QtWidgets.QAction(
            "Show UV", self, checkable=True, checked=True
        )
        self.action_show_uv.triggered.connect(self.toggle_uv)
        view_menu.addAction(self.action_show_uv)

        # ------------------------------------------------------------------
        # 6)  Navigation & parameter controls dock
        # ------------------------------------------------------------------
        ctrl_dock = QtWidgets.QDockWidget("Hull Controls", self)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, ctrl_dock)
        container = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout(container)

        # Navigation buttons -------------------------------------------------
        nav = QtWidgets.QHBoxLayout()
        btn_prev = QtWidgets.QPushButton("← Prev")
        btn_next = QtWidgets.QPushButton("Next →")
        btn_prev.clicked.connect(self.show_prev)
        btn_next.clicked.connect(self.show_next)
        nav.addWidget(btn_prev)
        nav.addStretch()
        nav.addWidget(btn_next)
        vbox.addLayout(nav)

        # Parameter editor (scrollable) -------------------------------------
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumHeight(320)
        form_container = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(form_container)
        scroll.setWidget(form_container)
        vbox.addWidget(scroll)

        # Numeric editors ----------------------------------------------------
        self.spins: list[QtWidgets.QDoubleSpinBox] = []
        for idx, val in enumerate(self.params):
            spin = QtWidgets.QDoubleSpinBox(buttonSymbols=QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons)
            spin.setDecimals(6)
            spin.setRange(-1e6, 1e6)
            spin.setSingleStep(0.001)
            spin.setValue(val)
            spin.valueChanged.connect(self._make_spin_updater(idx))
            form.addRow(f"Param {idx + 1}", spin)
            self.spins.append(spin)

        # Save button --------------------------------------------------------
        save_btn = QtWidgets.QPushButton("Save")
        save_btn.setShortcut("Ctrl+S")
        save_btn.clicked.connect(self._save_csv)
        vbox.addWidget(save_btn)

        ctrl_dock.setWidget(container)

        # ------------------------------------------------------------------
        # 7)  Initial hull render
        # ------------------------------------------------------------------
        self.mesh_actor = None
        self.update_hull()

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------
    def _make_spin_updater(self, idx: int):
        def cb(val: float) -> None:
            self.params[idx] = float(val)
            self.update_hull()

        return cb

    # ------------------------------------------------------------------
    #  CSV persistence
    # ------------------------------------------------------------------
    def _save_csv(self) -> None:
        """Append the current parameter vector to ``SAVE_PATH``.
        Creates directory / header row automatically on first run.
        """
        SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
        header = [f"Param {i + 1}" for i in range(len(self.params))]
        need_header = not SAVE_PATH.exists() or SAVE_PATH.stat().st_size == 0
        with SAVE_PATH.open("a", newline="") as fp:
            writer = csv.writer(fp)
            if need_header:
                writer.writerow(header)
            writer.writerow(self.params)
        QtWidgets.QMessageBox.information(
            self,
            "Parameters saved",
            f"Appended hull parameters to\n{SAVE_PATH}",
        )

    # ------------------------------------------------------------------
    #  Navigation among pre‑loaded vectors
    # ------------------------------------------------------------------
    def show_prev(self) -> None:
        if self.vectors.ndim == 2 and self.vectors.shape[0] > 1:
            self.current_idx = (self.current_idx - 1) % self.vectors.shape[0]
            self.load_current_vector()

    def show_next(self) -> None:
        if self.vectors.ndim == 2 and self.vectors.shape[0] > 1:
            self.current_idx = (self.current_idx + 1) % self.vectors.shape[0]
            self.load_current_vector()

    def load_current_vector(self) -> None:
        self.params = list(self.vectors[self.current_idx])
        for idx, spin in enumerate(self.spins):
            spin.blockSignals(True)
            spin.setValue(self.params[idx])
            spin.blockSignals(False)
        self.update_hull()

    # ------------------------------------------------------------------
    #  UV toggle
    # ------------------------------------------------------------------
    def toggle_uv(self, checked: bool) -> None:
        self.compute_uv = checked
        self.uv_dock.setVisible(checked)
        self.pattern_selector.setEnabled(checked)
        self.update_hull()

    # ------------------------------------------------------------------
    #  Hull regeneration / rendering
    # ------------------------------------------------------------------
    def update_hull(self) -> None:
        """Regenerate STL, display mesh and refresh texture preview."""
        # 1) Geometry export via ShipD
        temp_base = str(Path(tempfile.gettempdir()) / "hull_view")
        hull = HP(self.params)
        _ = hull.gen_stl(
            return_uv=self.compute_uv,  # bool
            NUM_WL=50,
            PointsPerWL=300,
            bit_AddTransom=0,
            bit_AddDeckLid=0,
            bit_RefineBowAndStern=1,
            namepath=temp_base,
        )
        stl_file = temp_base + ".stl"
        if not Path(stl_file).is_file():
            print(f"STL generation failed: {stl_file} not found")
            return

        # 2) Display mesh (replace old actor)
        mesh = pv.read(stl_file)
        if self.mesh_actor:
            self.plotter.remove_actor(self.mesh_actor)
        self.mesh_actor = self.plotter.add_mesh(mesh, color="#cccccc", smooth_shading=True)
        if not self.plotter.camera_set:
            self.plotter.reset_camera()
        self.plotter.render()

        # 3) Refresh texture / preview widgets
        self.update_texture()

    # ------------------------------------------------------------------
    #  Texture generation & preview
    # ------------------------------------------------------------------
    def update_texture(self) -> None:
        """Create a numpy stripe pattern and show it in the GraphicsScene."""
        idx = self.pattern_selector.currentIndex()
        h, w, n = self.texture_height, self.texture_width, self.num_stripes
        stripe_w = max(1, w // (2 * n))
        arr = np.zeros((h, w, 3), dtype=np.uint8)

        if idx == 0:  # Horizontal
            for y in range(h):
                if (y // stripe_w) % 2 == 0:
                    arr[y, :, :] = 255
        elif idx == 1:  # Vertical
            for x in range(w):
                if (x // stripe_w) % 2 == 0:
                    arr[:, x, :] = 255
        elif idx == 2:  # 45° diagonal (\)
            for y in range(h):
                for x in range(w):
                    if (((x + y) // stripe_w) % 2) == 0:
                        arr[y, x, :] = 255
        elif idx == 3:  # 135° diagonal (/)
            for y in range(h):
                for x in range(w):
                    if (((x - y) // stripe_w) % 2) == 0:
                        arr[y, x, :] = 255

        # Update the QGraphicsScene
        self.stripe_scene.clear()
        img = QtGui.QImage(
            arr.data, w, h, 3 * w, QtGui.QImage.Format.Format_RGB888
        )
        pix = QtGui.QPixmap.fromImage(img)
        self.stripe_scene.addPixmap(pix)
        self.stripe_scene.setSceneRect(0, 0, w, h)

        # UV dock preview (optional): reuse same pixmap
        if self.compute_uv:
            self.uv_view.setPixmap(pix.scaled(
                self.uv_view.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio, QtCore.Qt.TransformationMode.SmoothTransformation
            ))
        else:
            self.uv_view.clear()

    # ------------------------------------------------------------------
    #  Qt entry point helper
    # ------------------------------------------------------------------
    @staticmethod
    def launch(vectors: np.ndarray | None = None) -> None:
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
        viewer = HullViewer(vectors)
        viewer.show()
        sys.exit(app.exec())


# -----------------------------------------------------------------------------
#  Script entry point ----------------------------------------------------------
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    HullViewer.launch()

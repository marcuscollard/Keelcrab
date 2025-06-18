import sys
import os
import tempfile
import csv
from pathlib import Path

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
import pyvista as pv
from pyvistaqt import QtInteractor

from HullParameterization import Hull_Parameterization as HP

SAVE_PATH = '/Users/marcuscollard/Desktop/KEELCRAB/Keelcrab/myproj/ship_assets'

class HullViewer(QtWidgets.QMainWindow):
    """Interactive hull viewer with optional UV‑layout preview and detachable UI parts."""

    # ------------------------------------------------------------------
    #  Constructor
    # ------------------------------------------------------------------
    def __init__(self, vectors=None):
        super().__init__()
        self.setWindowTitle("Live Hull Viewer")
        self.resize(1200, 800)

        # --------------------------------------------------------------
        # 1)  Load / infer parameter vectors
        # --------------------------------------------------------------
        dataset_num = 4  # choose one of the CSV data sets below (1‑based)
        datasets = [
            "Constrained_Randomized_Set_1",
            "Constrained_Randomized_Set_2",
            "Constrained_Randomized_Set_3",
            "Diffusion_Aug_Set_1",
            "Diffusion_Aug_Set_2",
        ]
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(script_dir, "Ship_D_Dataset", datasets[dataset_num - 1], "Input_Vectors.csv")

        if vectors is None and os.path.isfile(csv_path):
            try:
                self.vectors = np.loadtxt(csv_path, delimiter=",", skiprows=1, dtype=np.float64)
            except Exception:
                self.vectors = np.empty((0,))
        elif vectors is not None:
            self.vectors = np.asarray(vectors, dtype=float)
        else:
            self.vectors = np.empty((0,))

        self.current_idx = 0
        if self.vectors.ndim == 2 and self.vectors.shape[0] > 0:
            self.params = list(self.vectors[self.current_idx])
        else:
            # Ask Hull_Parameterization how many parameters it needs.
            length = 0
            while True:
                try:
                    _ = HP([0.0] * length)
                    length += 1
                except IndexError:
                    break
            self.params = [0.5] * length

        # --------------------------------------------------------------
        # 2)  Texture / UV settings
        # --------------------------------------------------------------
        self.texture_height = 200
        self.texture_width = 3 * self.texture_height  # 6:1 aspect ratio
        self.num_stripes = 10
        self.compute_uv = True  # toggled via View‑menu

        # --------------------------------------------------------------
        # 3)  Central 3‑D viewport (PyVistaQt)
        # --------------------------------------------------------------
        self.frame = QtWidgets.QFrame()
        self._vlayout = QtWidgets.QVBoxLayout(self.frame)
        self.plotter = QtInteractor(self.frame)
        self._vlayout.addWidget(self.plotter.interactor)
        self.setCentralWidget(self.frame)

        # --------------------------------------------------------------
        # 4)  Stripe‑pattern selector & preview
        # --------------------------------------------------------------
        self.pattern_selector = QtWidgets.QComboBox()
        self.pattern_selector.addItems(["Horizontal", "Vertical", "45° Diagonal", "135° Diagonal"])
        self.pattern_selector.currentIndexChanged.connect(self.update_texture)
        self._vlayout.addWidget(self.pattern_selector)

        self.stripe_scene = QtWidgets.QGraphicsScene()
        self.stripe_view = QtWidgets.QGraphicsView(self.stripe_scene)
        self.stripe_view.setFixedSize(self.texture_width, self.texture_height)
        self._vlayout.addWidget(self.stripe_view)

        # --------------------------------------------------------------
        # 5)  UV layout dock (detachable)
        # --------------------------------------------------------------
        self.uv_scene = QtWidgets.QGraphicsScene()
        self.uv_view = QtWidgets.QGraphicsView(self.uv_scene)
        self.uv_view.setMinimumSize(400, 400)
        self.uv_dock = QtWidgets.QDockWidget("UV Layout", self)
        self.uv_dock.setWidget(self.uv_view)
        self.uv_dock.setAllowedAreas(QtCore.Qt.AllDockWidgetAreas)
        self.uv_dock.setFeatures(
            QtWidgets.QDockWidget.DockWidgetMovable | QtWidgets.QDockWidget.DockWidgetFloatable
        )
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.uv_dock)

        view_menu = self.menuBar().addMenu("View")
        self.action_show_uv = QtWidgets.QAction("Show UV Layout", self, checkable=True, checked=True)
        self.action_show_uv.toggled.connect(self.toggle_uv)
        view_menu.addAction(self.action_show_uv)

        # --------------------------------------------------------------
        # 6)  Navigation & parameter controls dock
        # --------------------------------------------------------------
        ctrl_dock = QtWidgets.QDockWidget("Hull Controls", self)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, ctrl_dock)
        container = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout(container)

        # Navigation buttons
        nav = QtWidgets.QHBoxLayout()
        btn_prev = QtWidgets.QPushButton("← Prev")
        btn_next = QtWidgets.QPushButton("Next →")
        btn_prev.clicked.connect(self.show_prev)
        btn_next.clicked.connect(self.show_next)
        nav.addWidget(btn_prev)
        nav.addStretch(1)
        nav.addWidget(btn_next)
        vbox.addLayout(nav)

        # Parameter editor in a scroll‑area
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumHeight(300)
        form_container = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(form_container)
        scroll.setWidget(form_container)
        vbox.addWidget(scroll)

        self.lineedits = []
        for idx, val in enumerate(self.params):
            le = QtWidgets.QLineEdit(f"{val:.3g}")
            val = QtGui.QDoubleValidator()          # create the validator first
            val.setLocale(QtCore.QLocale("C"))      # force the “.” decimal point
            le.setValidator(val)                    # then attach
            le.editingFinished.connect(self._make_updater(idx, le))
            form.addRow(f"Param {idx + 1}", le)
            self.lineedits.append(le)
            
        # Save button --------------------------------------------------------
        save_btn = QtWidgets.QPushButton("Save")
        save_btn.setShortcut("Ctrl+S")
        save_btn.clicked.connect(self._save_csv)
        vbox.addWidget(save_btn)

        ctrl_dock.setWidget(container)

        ctrl_dock.setWidget(container)

        # --------------------------------------------------------------
        # 7)  Initial hull render
        # --------------------------------------------------------------
        self.mesh_actor = None
        self.st0 = None  # UV buffer placeholder
        self.update_hull()

    # ------------------------------------------------------------------
    #  Callbacks & helpers
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


    def _make_updater(self, idx, widget):
        def cb():
            try:
                self.params[idx] = float(widget.text())
                self.update_hull()
                widget.setText(f"{self.params[idx]:.3g}")
            except ValueError:
                widget.setText(f"{self.params[idx]:.3g}")
        return cb

    def show_prev(self):
        if self.vectors.ndim == 2 and self.vectors.shape[0] > 1:
            self.current_idx = (self.current_idx - 1) % self.vectors.shape[0]
            self.load_current_vector()

    def show_next(self):
        if self.vectors.ndim == 2 and self.vectors.shape[0] > 1:
            self.current_idx = (self.current_idx + 1) % self.vectors.shape[0]
            self.load_current_vector()

    def load_current_vector(self):
        self.params = list(self.vectors[self.current_idx])
        for idx, le in enumerate(self.lineedits):
            le.setText(f"{self.params[idx]:.3g}")
        self.update_hull()

    def toggle_uv(self, checked: bool):
        """Enable/disable UV computation & preview."""
        self.compute_uv = checked
        self.uv_dock.setVisible(checked)
        self.pattern_selector.setEnabled(checked)
        self.update_hull()

    # ------------------------------------------------------------------
    #  Main mesh/texture regeneration
    # ------------------------------------------------------------------
    def update_hull(self):
        """Generate hull geometry (and UVs when requested)."""
        hull = HP(self.params)
        base = os.path.join(tempfile.gettempdir(), "hull_view")

        st0 = hull.gen_stl(
            return_uv=self.compute_uv,
            NUM_WL=50,
            PointsPerWL=300,
            bit_AddTransom=0,
            bit_AddDeckLid=0,
            bit_RefineBowAndStern=1,
            namepath=base,
        )
        stl_file = base + ".stl"
        if not os.path.isfile(stl_file):
            print(f"STL generation failed: {stl_file} not found")
            return

        pv_mesh = pv.read(stl_file)

        # -------------------- No‑UV path -------------------- #
        if not self.compute_uv:
            if self.mesh_actor is None:
                self.mesh_actor = self.plotter.add_mesh(pv_mesh, color="lightgrey", show_edges=False)
                self.plotter.reset_camera()
            else:
                self.mesh_actor.mapper.SetInputData(pv_mesh)
                self.mesh_actor.SetTexture(None)
            self.plotter.render()
            return

        # -------------------- UV path ---------------------- #
        faces = pv_mesh.faces.reshape(-1, 4)[:, 1:]
        verts = pv_mesh.points[faces]
        flat_vertices = verts.reshape(-1, 3)

        uvs = np.asarray(st0).reshape(-1, 2)
        assert uvs.shape[0] == flat_vertices.shape[0], "UV/vertex mismatch"

        flat_faces = np.hstack([[3, i, i + 1, i + 2] for i in range(0, len(flat_vertices), 3)])
        expanded_mesh = pv.PolyData(flat_vertices, flat_faces)
        expanded_mesh.point_data["Texture Coordinates"] = uvs
        expanded_mesh.active_texture_coordinates = uvs

        self.st0 = st0  # store for UV plotter

        # Apply a temporary blank texture (real one set in update_texture)
        blank_tex = pv.numpy_to_texture(np.full((self.texture_height, self.texture_width, 3), 255, np.uint8))
        blank_tex.repeat = False
        blank_tex.edge_clamp = True

        if self.mesh_actor is None:
            self.mesh_actor = self.plotter.add_mesh(expanded_mesh, texture=blank_tex, show_edges=False)
            self.plotter.reset_camera()
        else:
            self.mesh_actor.mapper.SetInputData(expanded_mesh)
            self.mesh_actor.SetTexture(blank_tex)

        # Draw UV layout wire‑frame & colour stripes
        self.display_uv_layout()
        self.update_texture()

    # ------------------------------------------------------------------
    #  UV wire‑frame drawing
    # ------------------------------------------------------------------
    def display_uv_layout(self):
        if self.st0 is None or not self.compute_uv:
            return
        W, H = 600, 200
        image = QtGui.QImage(W, H, QtGui.QImage.Format_RGB32)
        image.fill(QtGui.QColor("white"))
        painter = QtGui.QPainter(image)
        painter.setPen(QtGui.QPen(QtGui.QColor("black"), 1))
        for tri in self.st0:
            pts = np.vstack([tri, tri[0]])
            poly = QtGui.QPolygonF()
            for u, v in pts:
                x = int(u * (W - 1))
                y = int((1 - v) * (H - 1))
                poly.append(QtCore.QPointF(x, y))
            painter.drawPolyline(poly)
        painter.end()
        pixmap = QtGui.QPixmap.fromImage(image)
        self.uv_scene.clear()
        self.uv_scene.addPixmap(pixmap)
        self.uv_scene.setSceneRect(0, 0, W, H)

    # ------------------------------------------------------------------
    #  Stripe‑pattern texture generation
    # ------------------------------------------------------------------

    def update_texture(self):
        # ---- build the stripe image exactly as before --------------------------
        u = np.linspace(0.0, 1.0, self.texture_width,  endpoint=False)
        v = np.linspace(0.0, 1.0, self.texture_height, endpoint=False)
        uu, vv = np.meshgrid(u, v)

        N   = self.num_stripes
        idx = self.pattern_selector.currentIndex()
        if idx == 0:
            pattern = ((vv * N) % 1.0) < 0.5
        elif idx == 1:
            pattern = ((uu * N) % 1.0) < 0.5
        elif idx == 2:
            pattern = (((uu + vv) * N) % 1.0) < 0.5
        else:
            pattern = ((((uu - vv) * N) % 1.0) + 1) % 1.0 < 0.5

        arr = (pattern.astype(np.uint8) * 255)

        # preview in the Qt scene –- this part is unchanged
        img_stripe = QtGui.QImage(arr.data, self.texture_width, self.texture_height,
                                self.texture_width, QtGui.QImage.Format_Grayscale8)
        pixmap_stripe = QtGui.QPixmap.fromImage(img_stripe)
        self.stripe_scene.clear()
        self.stripe_scene.addPixmap(pixmap_stripe)
        self.stripe_scene.setSceneRect(0, 0, self.texture_width, self.texture_height)

        # ---- build the UV texture ------------------------------------------------
        uv_map = np.zeros((self.texture_height, self.texture_width, 3), dtype=np.uint8)
        uv_map[..., 0] = (uu * 255).astype(np.uint8)     # U in red
        uv_map[..., 1] = (vv * 255).astype(np.uint8)     # V in green
        uv_map[..., 2] = arr                             # stripe pattern in blue

        # convert the **in-memory** array to a Texture – no file I/O!
        tex = pv.numpy_to_texture(np.ascontiguousarray(uv_map))  # flip=False by default
        tex.repeat      = False
        tex.edge_clamp  = True

        if self.mesh_actor:
            self.mesh_actor.SetTexture(tex)

        self.plotter.render()


# ----------------------------------------------------------------------
#  Application entry‑point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    viewer = HullViewer()
    viewer.show()
    sys.exit(app.exec_())

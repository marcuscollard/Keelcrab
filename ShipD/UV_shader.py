import sys
import os
import tempfile
import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import QPointF
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor
from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene
import pyvista as pv
from pyvistaqt import QtInteractor

from HullParameterization import Hull_Parameterization as HP


import numpy as np

UV_H = 256                       # rows  (pixels)
UV_W = 6 * UV_H                  # cols (pixels)  ⇒ 1536×256


import numpy as np

def dummy_feature_mask(tris, *, uv_size=(1536, 256), seed=0):
    """
    Stand-in heat-map that follows the usual UV convention:
        x = U · W   (columns),
        y = V · H   (rows).

    Contents
    --------
    • Keel ellipse : centred at V ≈ 0.85, U ≈ 0.50
    • Fouling dots : random points where V < 0.10
    """
    W, H = uv_size              # W = cols, H = rows
    mask = np.zeros((H, W), np.float32)

    # ------------------------------------------------------------------
    # 1)  Keel – red ellipse near the bottom-centre
    # ------------------------------------------------------------------
    yy, xx = np.ogrid[0:H, 0:W]                 # yy ↔ V, xx ↔ U
    cy, cx = 0.85 * W, 0.50 * H                 # centre (row, col)
    ry, rx = 0.05 * W, 0.15 * H                 # radii  (row, col)

    ellipse = ((yy - cy)**2) / ry**2 + ((xx - cx)**2) / rx**2 <= 1.0
    mask[ellipse] = 1.0

    # ------------------------------------------------------------------
    # 2)  Fouling – random dots where V < 0.10
    # ------------------------------------------------------------------
    rng       = np.random.default_rng(seed)
    uv_flat   = tris.reshape(-1, 2)                 # (N*3, 2)  [u, v]
    low_band  = uv_flat[uv_flat[:, 1] < 0.10]       # small-V band

    if low_band.size:
        sample = low_band[rng.choice(low_band.shape[0],
                                     size=min(400, low_band.shape[0]),
                                     replace=False)]
        cols = np.clip((sample[:, 0] * W).round().astype(np.int64), 0, W-1)
        rows = np.clip((sample[:, 1] * H).round().astype(np.int64), 0, H-1)
        mask[rows, cols] = 1.0

    return mask



def uv_pixmap(uv_size, tris, feature_mask=None):
    """Return a QPixmap visualising the UV layout with an optional mask.

    Parameters
    ----------
    uv_size : tuple
        (width, height) of the output image in pixels.
    tris : ndarray
        Array of shape (N, 3, 2) containing UV coordinates in [0, 1].
    feature_mask : ndarray | None
        Optional (H, W) boolean/float mask that will be displayed in red.
    """
    W, H = uv_size
    img = QImage(W, H, QImage.Format_RGB32)
    img.fill(0xffffffff)  # white background

    # Draw feature mask first so mesh edges appear on top
    if feature_mask is not None:
        if feature_mask.dtype != np.uint8:
            mask_alpha = (feature_mask * 180).clip(0, 255).astype(np.uint8)
        else:
            mask_alpha = feature_mask
        mask_rgba = np.zeros((H, W, 4), np.uint8)
        mask_rgba[..., 0] = 255  # red channel
        mask_rgba[..., 3] = mask_alpha
        mk = QImage(mask_rgba.data, W, H, QImage.Format_RGBA8888)
        painter = QPainter(img)
        painter.drawImage(0, 0, mk)
        painter.end()

    # Draw triangle edges
    painter = QPainter(img)
    pen = QPen(QColor('#0060ff'))
    pen.setWidth(1)
    painter.setPen(pen)
    for tri in tris:
        # tri  is (3, 2) float32, range 0-1
        pts = [(float(tri[i, 0] * W), float(tri[i, 1] * H))   # cast → Python float
            for i in range(3)]

        # either pass QPointF objects  (works with PyQt5 & 6)
        painter.drawLine(QPointF(*pts[0]), QPointF(*pts[1]))
        painter.drawLine(QPointF(*pts[1]), QPointF(*pts[2]))
        painter.drawLine(QPointF(*pts[2]), QPointF(*pts[0]))
    painter.end()

    return QPixmap.fromImage(img)

def stprim_to_tris(flat_st):
    """
    Isaac-Sim / USD 'st' → (N,3,2) array for uv_pixmap
    flat_st : 1-D iterable [u0,v0,u1,v1,u2,v2, …]
    """
    arr   = np.asarray(flat_st, dtype=np.float32)
    tris  = arr.reshape(-1, 3, 2)       #   [[u,v] [u,v] [u,v]]
    tris[...,1] = 1.0 - tris[...,1]     # flip V for Qt’s top-left origin
    return tris


class HullViewer(QtWidgets.QMainWindow):
    """Interactive hull viewer with a live UV‑map pane."""

    def __init__(self, vectors=None):
        super().__init__()
        self.setWindowTitle("Live Hull Viewer + UV map")
        self.resize(1600, 800)

        # ------------------------- data loading -------------------------
        dataset_num = 4
        datasets = [
            "Constrained_Randomized_Set_1",
            "Constrained_Randomized_Set_2",
            "Constrained_Randomized_Set_3",
            "Diffusion_Aug_Set_1",
            "Diffusion_Aug_Set_2",
        ]
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(
            script_dir, "Ship_D_Dataset", datasets[dataset_num - 1], "Input_Vectors.csv"
        )

        if vectors is None and os.path.isfile(csv_path):
            try:
                self.vectors = np.loadtxt(csv_path, delimiter=",", skiprows=1, dtype=np.float64)
            except Exception as e:
                print(f"Failed to load vectors: {e}")
                self.vectors = np.empty((0,))
        elif vectors is not None:
            self.vectors = np.array(vectors)
        else:
            self.vectors = np.empty((0,))

        # Prepare index and initial params
        self.current_idx = 0
        if self.vectors.ndim == 2 and self.vectors.shape[0] > 0:
            self.params = list(self.vectors[self.current_idx])
        else:
            # fallback to single vector inference
            length = 0
            while True:
                try:
                    _ = HP([0.0] * length)
                    length += 1
                except IndexError:
                    break
            self.params = [0.5] * length

        # ------------------------- 3‑D viewport -------------------------
        self.frame = QtWidgets.QFrame()
        self.vlayout = QtWidgets.QVBoxLayout(self.frame)
        self.plotter = QtInteractor(self.frame)
        self.vlayout.addWidget(self.plotter.interactor)
        self.setCentralWidget(self.frame)

        # ------------------------- UV viewer ---------------------------
        self.uv_dock = QtWidgets.QDockWidget("UV Map", self)
        self.uv_view = QGraphicsView()
        self.uv_scene = QGraphicsScene()
        self.uv_view.setScene(self.uv_scene)
        self.uv_dock.setWidget(self.uv_view)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.uv_dock)

        # ------------------------- Params dock -------------------------
        dock = QtWidgets.QDockWidget("Parameters", self)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, dock)
        container = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout(container)

        # Navigation buttons
        nav = QtWidgets.QHBoxLayout()
        btn_prev = QtWidgets.QPushButton("⟨ Prev")
        btn_next = QtWidgets.QPushButton("Next ⟩")
        btn_prev.clicked.connect(self.show_prev)
        btn_next.clicked.connect(self.show_next)
        nav.addWidget(btn_prev)
        nav.addStretch(1)
        nav.addWidget(btn_next)
        vbox.addLayout(nav)

        # Parameter entry with scrollbar
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumHeight(300)
        params_widget = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(params_widget)
        scroll.setWidget(params_widget)
        vbox.addWidget(scroll)

        # Line edits per parameter
        self.lineedits = []
        for idx, val in enumerate(self.params):
            le = QtWidgets.QLineEdit(f"{val}")
            le.setValidator(QtGui.QDoubleValidator())
            le.editingFinished.connect(self._make_updater(idx, le))
            form.addRow(f"Param {idx+1}", le)
            self.lineedits.append(le)

        dock.setWidget(container)

        # 3‑D mesh actor placeholder
        self.mesh_actor = None
        self.update_hull()  # first draw

    # ------------------------------------------------------------------
    # Helper: build a closure that updates a specific parameter
    def _make_updater(self, idx, lineedit):
        def _update():
            try:
                self.params[idx] = float(lineedit.text())
                self.update_hull()
            except ValueError:
                pass

        return _update

    # ------------------------------------------------------------------
    # Navigation callbacks
    def show_prev(self):
        if self.vectors.ndim == 2 and self.vectors.shape[0] > 1:
            self.current_idx = (self.current_idx - 1) % self.vectors.shape[0]
            self.load_current_vector()

    def show_next(self):
        if self.vectors.ndim == 2 and self.vectors.shape[0] > 1:
            self.current_idx = (self.current_idx + 1) % self.vectors.shape[0]
            self.load_current_vector()

    def load_current_vector(self):
        vec = list(self.vectors[self.current_idx])
        self.params = vec
        for idx, le in enumerate(self.lineedits):
            le.setText(f"{self.params[idx]}")
        self.update_hull()

    # ------------------------------------------------------------------
    # Core update routine – called whenever parameters change
    def update_hull(self):
        """Regenerate the hull STL + UV preview from the current parameters."""
        hull = HP(self.params)
        base = os.path.join(tempfile.gettempdir(), "hull_view")
        tris = hull.gen_stl(
            return_uv=True,
            NUM_WL=50,
            PointsPerWL=300,
            bit_AddTransom=1,
            bit_AddDeckLid=1,
            bit_RefineBowAndStern=0,
            namepath=base,
        )
        stl_file = base + ".stl"
        if not os.path.isfile(stl_file):
            print(f"STL generation failed: {stl_file} not found")
            return

        # ---- update 3‑D view ----
        pv_mesh = pv.read(stl_file)
        if self.mesh_actor is None:
            self.mesh_actor = self.plotter.add_mesh(pv_mesh, show_edges=True)
            self.plotter.reset_camera()
        else:
            self.mesh_actor.mapper.SetInputData(pv_mesh)
        self.plotter.render()

        # ---- update UV pane ----
        try:
            tris = tris.astype(np.float32)
            tris[..., 1] = 1.0 - tris[..., 1]

        except AttributeError:
            # The HP class shipped with the paper does not expose UVs; you need to add
            # hull.tri_uv yourself or compute it here.
            print("Hull does not have UV triangles defined.")
            return
        
        print(tris.min(), tris.max())      # should be in [0, 1]
        feature_mask = dummy_feature_mask(tris, uv_size=(UV_W, UV_H))  # <–– here
        # flip the feature mask, feature_mask: _Array[tuple[int, int], floating[_32Bit]]
        pm           = uv_pixmap((UV_W, UV_H), tris, feature_mask)     # <–– and here           # fit scene
        # If you have a neural net, replace dummy_feature_mask with your model:
        self.uv_scene.clear()
        self.uv_scene.addPixmap(pm)
        self.uv_scene.setSceneRect(0, 0, UV_W, UV_H)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    viewer = HullViewer()
    viewer.show()
    sys.exit(app.exec_())

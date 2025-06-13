import sys
import os
import tempfile
import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui
import pyvista as pv
from pyvistaqt import QtInteractor
from PIL import Image

from HullParameterization import Hull_Parameterization as HP


class HullViewer(QtWidgets.QMainWindow):
    def __init__(self, vectors=None):
        super().__init__()
        self.setWindowTitle("Live Hull Viewer")
        self.resize(1200, 800)

        dataset_num = 4
        datasets = ['Constrained_Randomized_Set_1', 'Constrained_Randomized_Set_2', 'Constrained_Randomized_Set_3',
                    "Diffusion_Aug_Set_1", "Diffusion_Aug_Set_2"]
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(
            script_dir,
            'Ship_D_Dataset',
            datasets[dataset_num-1],
            'Input_Vectors.csv'
        )
        if vectors is None and os.path.isfile(csv_path):
            try:
                self.vectors = np.loadtxt(csv_path, delimiter=",", skiprows=1, dtype=np.float64)
            except Exception:
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
            # fallback vector length
            length = 0
            while True:
                try:
                    _ = HP([0.0] * length)
                    length += 1
                except IndexError:
                    break
            self.params = [0.5] * length

        # Texture settings
        self.texture_height = 200
        self.texture_width = 3 * self.texture_height  # 6:1 aspect ratio
        self.num_stripes = 10

        # 3D viewport
        self.frame = QtWidgets.QFrame()
        self.vlayout = QtWidgets.QVBoxLayout(self.frame)
        self.plotter = QtInteractor(self.frame)
        self.vlayout.addWidget(self.plotter.interactor)
        self.setCentralWidget(self.frame)

        # Add stripe-pattern selector
        self.pattern_selector = QtWidgets.QComboBox()
        self.pattern_selector.addItems(["Horizontal", "Vertical", "45° Diagonal", "135° Diagonal"])
        self.pattern_selector.currentIndexChanged.connect(self.update_texture)
        self.vlayout.addWidget(self.pattern_selector)

        # UV layout view
        self.uv_scene = QtWidgets.QGraphicsScene()
        self.uv_view = QtWidgets.QGraphicsView(self.uv_scene)
        self.uv_view.setFixedSize(400, 400)
        self.vlayout.addWidget(self.uv_view)

        # Stripe preview view
        self.stripe_scene = QtWidgets.QGraphicsScene()
        self.stripe_view = QtWidgets.QGraphicsView(self.stripe_scene)
        self.stripe_view.setFixedSize(self.texture_width, self.texture_height)
        self.vlayout.addWidget(self.stripe_view)

        # Dock: navigation + parameters
        dock = QtWidgets.QDockWidget("Hull Controls", self)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)
        container = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout(container)

        # Navigation arrows
        nav = QtWidgets.QHBoxLayout()
        btn_prev = QtWidgets.QPushButton("← Prev")
        btn_next = QtWidgets.QPushButton("Next →")
        btn_prev.clicked.connect(self.show_prev)
        btn_next.clicked.connect(self.show_next)
        nav.addWidget(btn_prev)
        nav.addStretch()
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

        # Initial actor
        self.mesh_actor = None
        self.update_hull()

    def _make_updater(self, idx, widget):
        def callback():
            try:
                self.params[idx] = float(widget.text())
                self.update_hull()
            except ValueError:
                widget.setText(f"{self.params[idx]}")
        return callback

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

    def update_hull(self):
        hull = HP(self.params)
        base = os.path.join(tempfile.gettempdir(), 'hull_view')
        st0 = hull.gen_stl(
            return_uv=True,
            NUM_WL=50,
            PointsPerWL=300,
            bit_AddTransom=0,
            bit_AddDeckLid=0,
            bit_RefineBowAndStern=1,
            namepath=base
        )
        stl_file = base + '.stl'
        if not os.path.isfile(stl_file):
            print(f"STL generation failed: {stl_file} not found")
            return
        pv_mesh = pv.read(stl_file)
        
       # Get face connectivity (VTK stores faces as [3, i0, i1, i2, 3, i3, i4, i5, ...])
        faces = pv_mesh.faces.reshape(-1, 4)[:, 1:]  # shape (n_faces, 3)

        # Recover the vertex positions of each face corner
        verts = pv_mesh.points[faces]               # shape (n_faces, 3, 3)
        flat_vertices = verts.reshape(-1, 3)        # shape (n_faces * 3, 3)

        print(st0.shape)

        uvs = st0.reshape(-1, 2)  # Now shape is (163284, 2)

        # Double-check UV match
        n_faces = pv_mesh.n_cells
        assert uvs.shape[0] == n_faces * 3, f"Mismatch: {uvs.shape[0]} vs {n_faces * 3}"

        # Create face array: each triangle is [3, i, i+1, i+2]
        flat_faces = np.hstack([[3, i, i+1, i+2] for i in range(0, len(flat_vertices), 3)])

        # Build expanded mesh
        expanded_mesh = pv.PolyData(flat_vertices, flat_faces)
        expanded_mesh.point_data["Texture Coordinates"] = uvs
        expanded_mesh.active_texture_coordinates = uvs
        
        
        self.st0 = np.array(st0)

        if self.mesh_actor is None:
            blank = np.zeros((self.texture_height, self.texture_width, 3), dtype=np.uint8)
            img = Image.fromarray(blank)
            temp_path = os.path.join(tempfile.gettempdir(), 'uv_texture_6x1.png')
            img.save(temp_path)
            texture = pv.read_texture(temp_path)
            texture.repeat = False
            texture.edge_clamp = True
            self.mesh_actor = self.plotter.add_mesh(expanded_mesh, texture=texture, show_edges=False)
            self.plotter.reset_camera()
        else:
            self.mesh_actor.mapper.SetInputData(expanded_mesh)

        self.display_uv_layout()
        self.update_texture()

    def display_uv_layout(self):
        W, H = 400, 400
        image = QtGui.QImage(W, H, QtGui.QImage.Format_RGB32)
        image.fill(QtGui.QColor('white'))
        painter = QtGui.QPainter(image)
        pen = QtGui.QPen(QtGui.QColor('black'))
        pen.setWidth(1)
        painter.setPen(pen)
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

    def update_texture(self):
        u = np.linspace(0.0, 1.0, self.texture_width, endpoint=False)
        v = np.linspace(0.0, 1.0, self.texture_height, endpoint=False)
        uu, vv = np.meshgrid(u, v)
        N = self.num_stripes
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
        img_stripe = QtGui.QImage(arr.data, self.texture_width, self.texture_height,
                                  self.texture_width, QtGui.QImage.Format_Grayscale8)
        pixmap_stripe = QtGui.QPixmap.fromImage(img_stripe)
        self.stripe_scene.clear()
        self.stripe_scene.addPixmap(pixmap_stripe)
        self.stripe_scene.setSceneRect(0, 0, self.texture_width, self.texture_height)

        uv_map = np.zeros((self.texture_height, self.texture_width, 3), dtype=np.uint8)
        uv_map[...,0] = (uu * 255).astype(np.uint8)
        uv_map[...,1] = (vv * 255).astype(np.uint8)
        uv_map[...,2] = arr
        img = Image.fromarray(uv_map)
        temp_path = os.path.join(tempfile.gettempdir(), 'uv_texture_6x1.png')
        img.save(temp_path)

        texture = pv.read_texture(temp_path)
        texture.repeat = False
        texture.edge_clamp = True
        if self.mesh_actor:
            self.mesh_actor.SetTexture(texture)
        self.plotter.render()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    viewer = HullViewer()
    viewer.show()
    sys.exit(app.exec_())
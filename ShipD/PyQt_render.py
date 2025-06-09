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

        # 3D viewport
        self.frame = QtWidgets.QFrame()
        self.vlayout = QtWidgets.QVBoxLayout(self.frame)
        self.plotter = QtInteractor(self.frame)
        self.vlayout.addWidget(self.plotter.interactor)
        self.setCentralWidget(self.frame)

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
            bit_RefineBowAndStern=0,
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
        n_faces = pv_mesh.n_faces
        assert uvs.shape[0] == n_faces * 3, f"Mismatch: {uvs.shape[0]} vs {n_faces * 3}"

        # Create face array: each triangle is [3, i, i+1, i+2]
        flat_faces = np.hstack([[3, i, i+1, i+2] for i in range(0, len(flat_vertices), 3)])

        # Build expanded mesh
        expanded_mesh = pv.PolyData(flat_vertices, flat_faces)
        expanded_mesh.point_data["Texture Coordinates"] = uvs
        expanded_mesh.active_texture_coordinates = uvs
        
        print("Expanded mesh n_points:", expanded_mesh.n_points)
        print("UVs shape:", uvs.shape)
        print("'Texture Coordinates' in point_data?", "Texture Coordinates" in expanded_mesh.point_data)
        
        # Define size
        height = 200
        width = 3 * height  # 6:1 aspect ratio

        # Create a UV grid
        uv_map = np.zeros((height, width, 3), dtype=np.uint8)

        # Fill with horizontal gradient (R), vertical gradient (G), and a diagonal line (B)
        for y in range(height):
            for x in range(width):
                uv_map[y, x, 0] = int(255 * x / width)      # Red: horizontal gradient
                uv_map[y, x, 1] = int(255 * y / height)     # Green: vertical gradient
                uv_map[y, x, 2] = int(255 * ((x + y) % 20 < 10))  # Blue: checker line

        # Save image
        img = Image.fromarray(uv_map)
        img.save("uv_texture_6x1.png")
        print('image saved as uv_texture_6x1.png')
        
        texture = pv.read_texture("uv_texture_6x1.png")
        texture.repeat = False         # no wrap
        texture.edge_clamp = True      # clamp to border

        if self.mesh_actor is None:
            self.mesh_actor = self.plotter.add_mesh(expanded_mesh, show_edges=False, texture=texture)
            self.plotter.reset_camera()
        else:
            self.mesh_actor.mapper.SetInputData(expanded_mesh)
        print("rendering hull with parameters:")
        self.plotter.render()
        
        import matplotlib.pyplot as plt

        # st: your UV array, shape (n_faces, 3, 2)
        # e.g. st = np.load(...)

        fig, ax = plt.subplots(figsize=(6,6))

        # draw the triangle outlines
        for tri in st0:
            # close the loop by appending the first corner again
            loop = np.vstack([tri, tri[0]])
            ax.plot(loop[:, 0], loop[:, 1], linewidth=0.5, alpha=0.7)

        # scatter the UV corner points
        uv_pts = st0.reshape(-1, 2)
        ax.scatter(uv_pts[:, 0], uv_pts[:, 1], s=1)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal', 'box')
        ax.set_xlabel('U')
        ax.set_ylabel('V')
        ax.set_title('UV Layout in Texture Domain')
        plt.tight_layout()
        plt.show()

    

        # # tris_uv_pix: your (n_tris, 3, 2) array *in pixel coords* (u from 0→UV_W, v from 0→UV_H)
        # feature_mask = dummy_feature_mask(tris_uv_pix, (width, height))

        # pm = uv_pixmap((UV_W, UV_H), tris_uv_pix, feature_mask)

        # self.uv_scene.clear()
        # self.uv_scene.addPixmap(pm)
        # self.uv_scene.setSceneRect(0, 0, UV_W, UV_H)




if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    viewer = HullViewer()
    viewer.show()
    sys.exit(app.exec_())

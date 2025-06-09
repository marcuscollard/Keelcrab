import sys
import os
import tempfile
import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui
import pyvista as pv
from pyvistaqt import QtInteractor

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
        hull.gen_stl(
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
        
        mesh.point_data["Texture Coordinates"] = st  # shape (N, 2)
        
        if self.mesh_actor is None:
            self.mesh_actor = self.plotter.add_mesh(pv_mesh, show_edges=True, texture=None, color='lightblue')
            self.plotter.reset_camera()
        else:
            self.mesh_actor.mapper.SetInputData(pv_mesh)
        self.plotter.render()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    viewer = HullViewer()
    viewer.show()
    sys.exit(app.exec_())

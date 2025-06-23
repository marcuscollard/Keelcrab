# hull_crawler.py -----------------------------------------------------------
from isaaclab.scene import InteractiveScene         # type-hints
import torch, numpy as np
ADHERED, DETACHED = 0, 1            # public so the main loop can check it



def generate_stripe_texture(width, height, num_stripes, pattern_idx):
    """
    Generate a stripe texture as a NumPy RGBA array.

    Parameters:
        width (int): Width of the texture
        height (int): Height of the texture
        num_stripes (int): Number of stripes
        pattern_idx (int): Pattern selector:
            0 = horizontal
            1 = vertical
            2 = diagonal /
            3 = diagonal \

    Returns:
        np.ndarray: Texture of shape (height, width, 4) dtype=uint8
    """
    u = np.linspace(0.0, 1.0, width, endpoint=False)
    v = np.linspace(0.0, 1.0, height, endpoint=False)
    uu, vv = np.meshgrid(u, v)

    if pattern_idx == 0:
        pattern = ((vv * num_stripes) % 1.0) < 0.5
    elif pattern_idx == 1:
        pattern = ((uu * num_stripes) % 1.0) < 0.5
    elif pattern_idx == 2:
        pattern = (((uu + vv) * num_stripes) % 1.0) < 0.5
    else:
        pattern = ((((uu - vv) * num_stripes) % 1.0 + 1.0) % 1.0) < 0.5

    # Stripe in blue channel, UV info in R and G, full alpha
    tex = np.zeros((height, width, 4), dtype=np.uint8)
    tex[..., 0] = (uu * 255).astype(np.uint8)             # Red = U
    tex[..., 1] = (vv * 255).astype(np.uint8)             # Green = V
    tex[..., 2] = pattern.astype(np.uint8) * 255          # Blue = stripe
    tex[..., 3] = 255                                     # Alpha = opaque

    return tex


class MeshManager:
    """High-level interface to handle dynamic texture provider"""

    # ---------------------------------------------------------------------
    def __init__(self, provider, scene: InteractiveScene, env_id: int, TEXTURE_SIZE=(1024,1024)):
        self.env_id = env_id
        self.scene = scene
        self.meshes = scene.meshes
        self.size = TEXTURE_SIZE
        self.provider = provider

    def set_size(self, size: tuple):
        """Set the size of the texture."""
        self.TEXTURE_SIZE = size

    def set_texture(self, key: str, texture, idx=0):
        tex = generate_stripe_texture(
            self.size[1], self.size[0], num_stripes=10, pattern_idx=idx
        )

        # green_mask = np.all(tex == GREEN_RGBA, axis=2)
        # white_mask = np.all(tex == WHITE_RGBA, axis=2)
        # tex[np.logical_and(half_mask, green_mask)] = WHITE_RGBA
        # tex[np.logical_and(half_mask, white_mask)] = GREEN_RGBA
        # tex[np.logical_and(half_mask2, green_mask)] = WHITE_RGBA
        # tex[np.logical_and(half_mask2, white_mask)] = GREEN_RGBA
        self.provider.set_data_array(tex, list(tex.shape))

    # ---------------------------------------------------------------------
    def get_mesh(self, key: str):
        return self.meshes[key][self.env_id]

    def set_mesh(self, key: str, mesh):
        self.meshes[key][self.env_id] = mesh



class HullCrawler:
    """High-level interface around the crawler robot in *one* environment."""

    # ---------------------------------------------------------------------
    def __init__(self, scene: InteractiveScene, env_id: int, *,
                 ray_sensor_key="lidar", robot_key="robot"):
        self.env_id      = env_id
        self.robot_view  = scene.articulations[robot_key]   # ArticulationView
        self.ray         = scene.sensors[ray_sensor_key]    # RayCaster
        self.state       = ADHERED
        self._R_body     = torch.eye(3)                     # last body frame

        # tunables
        self._step_len   = 0.08      # 8 cm forward/back
        self._yaw_step   = np.deg2rad(15.0)
        self._noise_mm   = 3e-3      # ±3 mm transl. noise
        self._noise_deg  = 0.25
        self._rng        = np.random.default_rng()

    # ---------------------------------------------------------------------
    # public API -----------------------------------------------------------
    def forward (self): self._issue_cmd("F")
    def back    (self): self._issue_cmd("B")
    def turn_left (self): self._issue_cmd("L")
    def turn_right(self): self._issue_cmd("R")

    # call once every sim frame
    def update(self):
        if self.state is ADHERED:
            self._stick_step()
        else:
            self._free_step()

    # ---------------------------------------------------------------------
    # internals ------------------------------------------------------------
    def _issue_cmd(self, c):
        self._pending_cmd = c

    def _stick_step(self):
        # 1. ideal delta in *current* body frame (+X)
        delta_nom = self._step_len * self._R_body[:, 0]
        delta_pos = delta_nom + self._rng.normal(0, self._noise_mm, 3)
        delta_yaw = self._rng.normal(0,
                                     np.deg2rad(self._noise_deg))
        if getattr(self, "_pending_cmd", None) == "B":
            delta_pos *= -1
        if getattr(self, "_pending_cmd", None) in ("L", "R"):
            sign = +1 if self._pending_cmd == "L" else -1
            delta_yaw += sign * self._yaw_step
        self._pending_cmd = None

        # 2. predict pose
        p, q = self.robot_view.get_world_poses(indices=[self.env_id])
        p = p[0] + delta_pos
        q = self._yaw_left_multiply(q[0], delta_yaw)

        # 3. ray-cast & project
        self.ray.set_pose(p, q, env_indices=[self.env_id])
        hits  = self.ray.data["ray_hits_w"][self.env_id]
        norms = self.ray.data["ray_normals_w"][self.env_id]
        if self._should_detach(hits, norms):
            self.robot_view.set_kinematic(False, indices=[self.env_id])
            self.state = DETACHED
            return

        p_new, q_new = self._project_pose(hits, norms)
        self.robot_view.set_world_poses(p_new[None], q_new[None],
                                        indices=[self.env_id])
        self._R_body = self._quat_to_mat(q_new)

    def _free_step(self):
        # user could add currents/drag here.
        v = self.robot_view.get_root_linear_velocities(indices=[self.env_id])
        if torch.linalg.norm(v) < 0.05:
            hits  = self.ray.data["ray_hits_w"][self.env_id]
            norms = self.ray.data["ray_normals_w"][self.env_id]
            if not self._should_detach(hits, norms):
                self.robot_view.set_kinematic(True, indices=[self.env_id])
                self.state = ADHERED
                self._R_body = self._quat_to_mat(
                    self.robot_view.get_world_poses(indices=[self.env_id])[1][0])

    # helpers --------------------------------------------------------------
    def _should_detach(self, hits, norms,
                       max_r=0.15, curv=np.deg2rad(25), hit_frac=0.60):
        valid = torch.norm(hits, dim=-1) < max_r
        if valid.float().mean() < hit_frac:
            return True
        n_mean = torch.nn.functional.normalize(norms[valid].mean(0), dim=0)
        ang = torch.acos(torch.clamp((norms[valid]@n_mean), -1, 1)).max()
        return ang > curv

    def _project_pose(self, hits, norms, offs=0.003):
        p = hits.mean(0)
        n = torch.nn.functional.normalize(norms.mean(0), dim=0)
        t = self._R_body[:, 0] - torch.dot(self._R_body[:, 0], n)*n
        t = torch.nn.functional.normalize(t, dim=0)
        R = torch.stack([t, torch.cross(n, t), n], 1)
        q = self._mat_to_quat(R)
        return p + offs*n, q

    # quaternion helpers (torch, scalar-first)
    def _quat_to_mat(self, q):
        return torch.tensor(
            [[1-2*(q[2]**2+q[3]**2), 2*(q[1]*q[2]-q[0]*q[3]), 2*(q[1]*q[3]+q[0]*q[2])],
             [2*(q[1]*q[2]+q[0]*q[3]), 1-2*(q[1]**2+q[3]**2), 2*(q[2]*q[3]-q[0]*q[1])],
             [2*(q[1]*q[3]-q[0]*q[2]), 2*(q[2]*q[3]+q[0]*q[1]), 1-2*(q[1]**2+q[2]**2)]])
    def _mat_to_quat(self, M):
        t = M.trace()
        if t > 0:
            s = torch.sqrt(t+1)*0.5
            w = 0.25/s
            return torch.tensor([s, (M[2,1]-M[1,2])*w,
                                    (M[0,2]-M[2,0])*w,
                                    (M[1,0]-M[0,1])*w])
        # else branch omitted for brevity
    def _yaw_left_multiply(self, q, dyaw):
        add = torch.tensor([np.cos(dyaw/2), 0, 0, np.sin(dyaw/2)])
        # Hamilton product (scalar-first)
        w0,x0,y0,z0 = add
        w1,x1,y1,z1 = q
        return torch.tensor([
            w0*w1 - x0*x1 - y0*y1 - z0*z1,
            w0*x1 + x0*w1 + y0*z1 - z0*y1,
            w0*y1 - x0*z1 + y0*w1 + z0*x1,
            w0*z1 + x0*y1 - y0*x1 + z0*w1])

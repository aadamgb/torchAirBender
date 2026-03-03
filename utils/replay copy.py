import taichi as ti
import numpy as np
from omegaconf import DictConfig


class BaseRenderer:
    def __init__(self, 
                 rend_cfg: DictConfig,
                 trajectory,
                 ):
        self.cfg        = rend_cfg
        self.width      = rend_cfg.window_width
        self.height     = rend_cfg.window_height

        ti.init(arch=ti.cpu)


        # ------------------------------------------------------------------
        # Ground
        # ------------------------------------------------------------------
        # Ground grid
        grid_n    = 11
        grid_half = 3.0
        self._grid_verts = ti.Vector.field(3, dtype=ti.f32, shape=4 * grid_n)
        gi = 0
        for k in range(grid_n):
            t_ = k / (grid_n - 1)
            v  = -grid_half + 2 * grid_half * t_
            self._grid_verts[gi]     = ti.Vector([v,   0.0, -grid_half])
            self._grid_verts[gi + 1] = ti.Vector([v,   0.0,  grid_half])
            self._grid_verts[gi + 2] = ti.Vector([-grid_half, 0.0, v])
            self._grid_verts[gi + 3] = ti.Vector([ grid_half, 0.0, v])
            gi += 4

        # Ground Plane
        _N = grid_n                         # same resolution as grid
        _EDGE = 2.0 * grid_half / (_N - 1)  # cell size so it spans [-grid_half, grid_half]
        _OFF = grid_half                    # center at origin

        num_cells = (_N - 1) * (_N - 1)

        self._ground_v = ti.Vector.field(3, dtype=ti.f32, shape=num_cells * 4)
        self._ground_c = ti.Vector.field(3, dtype=ti.f32, shape=num_cells * 4)
        self._ground_i = ti.field(dtype=ti.i32, shape=num_cells * 6)

        for i in range(_N - 1):
            for j in range(_N - 1):
                cell = i * (_N - 1) + j
                vb = cell * 4

                x0 = -_OFF + i * _EDGE
                z0 = -_OFF + j * _EDGE
                x1 = x0 + _EDGE
                z1 = z0 + _EDGE

                self._ground_v[vb + 0] = ti.Vector([x0, 0.0, z0])
                self._ground_v[vb + 1] = ti.Vector([x1, 0.0, z0])
                self._ground_v[vb + 2] = ti.Vector([x0, 0.0, z1])
                self._ground_v[vb + 3] = ti.Vector([x1, 0.0, z1])

                # Nice blue coolors MuJoCo stlye
                val = ti.Vector([0.125, 0.239, 0.322]) if (i + j) % 2 == 0 else ti.Vector([0.2431, 0.4392, 0.5922])
                for k in range(4):
                    self._ground_c[vb + k] = val

                ib = cell * 6
                self._ground_i[ib + 0] = vb
                self._ground_i[ib + 1] = vb + 1
                self._ground_i[ib + 2] = vb + 2
                self._ground_i[ib + 3] = vb + 1
                self._ground_i[ib + 4] = vb + 3
                self._ground_i[ib + 5] = vb + 2

        # ------------------------------------------------------------------
        # Axis
        # ------------------------------------------------------------------
        # World Frame
        AXIS_SCLAE = 0.5
        self._world_axis_verts = [
            ti.Vector.field(3, dtype=ti.f32, shape=2) for _ in range(3)
        ]
        world_tips = AXIS_SCLAE * np.array([
            [1, 0,  0],   # ENU x → Taichi x  (red)
            [0, 0, -1],   # ENU y → Taichi z  (green)
            [0, 1,  0],   # ENU z → Taichi y  (blue)
        ], dtype=np.float32)

        for i in range(3):
            self._world_axis_verts[i][0] = ti.Vector([0.0, 0.0, 0.0])
            self._world_axis_verts[i][1] = ti.Vector(world_tips[i].tolist())

        self._axis_colors = [
            (0.9, 0.15, 0.15),   # X — red
            (0.15, 0.9, 0.15),   # Y — green
            (0.15, 0.15, 0.9),   # Z — blue
        ]

        # Body Frame
        self._body_axes = ti.Vector.field(3, dtype=ti.f32, shape=6)

        # ------------------------------------------------------------------
        # Sphere (quadrotor body proxy)
        # ------------------------------------------------------------------
        # Convert to numpy once
        traj_np = np.array(trajectory, dtype=np.float32)   # (T, 13)
        T       = len(traj_np)

        pos = traj_np[0:3]
        quat = traj_np[6:10]

        R = self.quat_to_rotmat(quat)

        axis_scale = 0.5

        # Body frame basis vectors
        body_axes = axis_scale * np.eye(3, dtype=np.float32)

        # Rotate into world
        tips = (R @ body_axes.T).T

        # Fill Taichi fields
        origin = ti.Vector(pos.tolist())

        for i in range(3):
            self._body_axes[2*i]     = origin
            self._body_axes[2*i + 1] = ti.Vector((pos + tips[i]).tolist())

        self._body_pos[0] = origin
        # print("Trajectory raw\n")
        # print(trajectory)
        # print("Trajectory in numpy\n")
        # print(traj_np)

        self._body_pos = ti.Vector.field(3, dtype=ti.f32, shape=1)




    def update_body(self, state):
        pos = state[0:3]
        quat = state[6:10]

        R = self.quat_to_rotmat(quat)

        axis_scale = 0.5

        # Body frame basis vectors
        body_axes = axis_scale * np.eye(3, dtype=np.float32)

        # Rotate into world
        tips = (R @ body_axes.T).T

        # Fill Taichi fields
        origin = ti.Vector(pos.tolist())

        for i in range(3):
            self._body_axes[2*i]     = origin
            self._body_axes[2*i + 1] = ti.Vector((pos + tips[i]).tolist())

        self._body_pos[0] = origin

    def quat_to_rotmat(self, q):
        w, x, y, z = q
        return np.array([
            [1 - 2*(y*y + z*z),   2*(x*y - w*z),       2*(x*z + w*y)],
            [2*(x*y + w*z),       1 - 2*(x*x + z*z),   2*(y*z - w*x)],
            [2*(x*z - w*y),       2*(y*z + w*x),       1 - 2*(x*x + y*y)],
        ], dtype=np.float32)



    def _window_title(self) -> str:
        """Override to customise the window title."""
        return "Replay Trajectory Env 1"












    def _init_window(self):
        self._window = ti.ui.Window(
            self._window_title(),
            (self.width, self.height),
            vsync=True,
        )

        self._canvas = self._window.get_canvas()
        self._scene  = self._window.get_scene()

        self._camera = ti.ui.Camera()
        self._camera.position(0, 3, 5)
        self._camera.lookat(0, 0, 0)
        self._camera.up(0, 1, 0)

    def run(self):
        self._init_window()
        while self._window.running:
            self._camera.track_user_inputs(self._window, movement_speed=0.03, hold_key=ti.ui.RMB)
            self._scene.set_camera(self._camera)

            # lights
            self._scene.point_light(pos=(2, 4, 2), color=(1, 1, 1))
            self._scene.ambient_light((0.5, 0.5, 0.5))

            # draw ground mesh
            self._scene.mesh(
                self._ground_v,
                indices=self._ground_i,
                per_vertex_color=self._ground_c,
            )

            # draw grid lines
            self._scene.lines(
                self._grid_verts,
                width=1.0,
                color=(0.2, 0.2, 0.2),
            )

            # draw world axes
            for i in range(3):
                self._scene.lines(
                    self._world_axis_verts[i],
                    width=3.0,
                    color=self._axis_colors[i],
                )




            self._canvas.scene(self._scene)
            self._window.show()
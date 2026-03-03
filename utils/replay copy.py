import taichi as ti
import numpy as np


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def enu_to_ti(pts: np.ndarray) -> np.ndarray:
    """
    ENU (x-East, y-North, z-Up)  →  Taichi (x, y=z_enu, z=-y_enu)
    Accepts (..., 3) numpy arrays.
    """
    out = pts[..., [0, 2, 1]].copy()
    out[..., 2] *= -1
    return out


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Hamilton [w, x, y, z] → 3×3 rotation matrix (ENU)."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),  2*(x*y - w*z),      2*(x*z + w*y)],
        [    2*(x*y + w*z),  1 - 2*(x*x + z*z),  2*(y*z - w*x)],
        [    2*(x*z - w*y),  2*(y*z + w*x),       1 - 2*(x*x + y*y)],
    ], dtype=np.float32)


# ---------------------------------------------------------------------------
# BaseRenderer
# ---------------------------------------------------------------------------

class BaseRenderer:
    """
    Renders:
      - Checkerboard ground plane + grid lines
      - World-frame axes  (X=red, Y=green, Z=blue)
      - A sphere at the body position with body-frame axes

    Controls:
        LMB + drag  : orbit camera
        Scroll      : zoom
        Space       : pause / resume
        R           : restart
        O           : toggle axes
        X / Esc     : quit
    """

    _AXIS_COLORS = [
        (0.9, 0.15, 0.15),   # X — red
        (0.15, 0.9, 0.15),   # Y — green
        (0.15, 0.15, 0.9),   # Z — blue
    ]

    def __init__(
        self,
        trajectory:  np.ndarray,  # (T, 13) ENU: pos[0:3] vel[3:6] quat_wxyz[6:10] omega[10:13]
        arm_length:  float,
        arm_angle:  float,
        mass:  float,
        dt:          float,
        window_size: tuple = (1280, 720),
    ):
        self._traj       = np.array(trajectory, dtype=np.float32)   # (T, 13)
        self._T          = len(self._traj)
        self._arm_len    = arm_length
        self._arm_angle    = arm_angle
        self._dt         = dt
        self._win_size   = window_size
        self._axis_scale = arm_length * 1.5

         # Sphere radius scales with mass: normalised so ~0.5 kg → ~arm*0.15
        self._sphere_r   = arm_length * 0.10 * (mass / 0.5) ** (1/3)

        # Precompute all positions in Taichi coords
        centers_enu   = self._traj[:, 0:3]                          # (T, 3) ENU
        self._centers = enu_to_ti(centers_enu)                      # (T, 3) Taichi

        # Precompute body axis tips in Taichi coords
        bases_enu = self._axis_scale * np.eye(3, dtype=np.float32)  # (3, 3)
        tips_enu  = np.zeros((self._T, 3, 3), dtype=np.float32)
        for t in range(self._T):
            R = quat_to_rotmat(self._traj[t, 6:10])
            tips_enu[t] = centers_enu[t] + (R @ bases_enu.T).T
        self._body_axis_tips = enu_to_ti(tips_enu)                  # (T, 3, 3) Taichi

        ti.init(arch=ti.cpu)

        self._build_ground()
        self._build_world_axes()
        self._build_body_fields()

    # ------------------------------------------------------------------
    # Geometry builders — called once in __init__
    # ------------------------------------------------------------------

    def _build_ground(self):
        grid_n    = 11
        grid_half = 3.0
        edge      = 2.0 * grid_half / (grid_n - 1)

        # Grid lines
        self._grid_verts = ti.Vector.field(3, dtype=ti.f32, shape=4 * grid_n)
        gi = 0
        for k in range(grid_n):
            v = -grid_half + 2 * grid_half * k / (grid_n - 1)
            self._grid_verts[gi + 0] = ti.Vector([v,          0.0, -grid_half])
            self._grid_verts[gi + 1] = ti.Vector([v,          0.0,  grid_half])
            self._grid_verts[gi + 2] = ti.Vector([-grid_half, 0.0,  v])
            self._grid_verts[gi + 3] = ti.Vector([ grid_half, 0.0,  v])
            gi += 4

        # Checkerboard mesh
        num_cells      = (grid_n - 1) ** 2
        self._ground_v = ti.Vector.field(3, dtype=ti.f32, shape=num_cells * 4)
        self._ground_c = ti.Vector.field(3, dtype=ti.f32, shape=num_cells * 4)
        self._ground_i = ti.field(dtype=ti.i32,           shape=num_cells * 6)

        dark  = ti.Vector([0.125,  0.239,  0.322])
        light = ti.Vector([0.2431, 0.4392, 0.5922])

        for i in range(grid_n - 1):
            for j in range(grid_n - 1):
                cell = i * (grid_n - 1) + j
                vb   = cell * 4
                x0 = -grid_half + i * edge;  x1 = x0 + edge
                z0 = -grid_half + j * edge;  z1 = z0 + edge

                self._ground_v[vb + 0] = ti.Vector([x0, 0.0, z0])
                self._ground_v[vb + 1] = ti.Vector([x1, 0.0, z0])
                self._ground_v[vb + 2] = ti.Vector([x0, 0.0, z1])
                self._ground_v[vb + 3] = ti.Vector([x1, 0.0, z1])

                col = dark if (i + j) % 2 == 0 else light
                for k in range(4):
                    self._ground_c[vb + k] = col

                ib = cell * 6
                self._ground_i[ib + 0] = vb;     self._ground_i[ib + 1] = vb + 1
                self._ground_i[ib + 2] = vb + 2; self._ground_i[ib + 3] = vb + 1
                self._ground_i[ib + 4] = vb + 3; self._ground_i[ib + 5] = vb + 2

    def _build_world_axes(self):
        s = self._axis_scale
        # ENU x → Taichi [s,0,0]   ENU y → Taichi [0,0,-s]   ENU z → Taichi [0,s,0]
        tips = [[s, 0, 0], [0, 0, -s], [0, s, 0]]
        self._world_axis_verts = [
            ti.Vector.field(3, dtype=ti.f32, shape=2) for _ in range(3)
        ]
        for i in range(3):
            self._world_axis_verts[i][0] = ti.Vector([0.0, 0.0, 0.0])
            self._world_axis_verts[i][1] = ti.Vector(tips[i])

    def _build_body_fields(self):
        self._body_pos = ti.Vector.field(3, dtype=ti.f32, shape=1)
        # Preallocated once — avoids Taichi recompilation lag on each frame
        self._body_axis_verts = [
            ti.Vector.field(3, dtype=ti.f32, shape=2) for _ in range(3)
        ]

    # ------------------------------------------------------------------
    # Per-frame field update
    # ------------------------------------------------------------------

    def _update_frame(self, frame: int):
        cx, cy, cz = self._centers[frame]
        origin = ti.Vector([cx, cy, cz])
        self._body_pos[0] = origin

        for i in range(3):
            tx, ty, tz = self._body_axis_tips[frame, i]
            self._body_axis_verts[i][0] = origin
            self._body_axis_verts[i][1] = ti.Vector([tx, ty, tz])

    # ------------------------------------------------------------------
    # Hook for subclasses
    # ------------------------------------------------------------------

    def _draw_extras(self, scene, frame: int):
        """Override in subclasses to add extra draw calls (targets, traces, etc.)."""
        pass

    # ------------------------------------------------------------------
    # Main render loop
    # ------------------------------------------------------------------

    def run(self):
        window = ti.ui.Window("Quadrotor Visualizer", self._win_size, vsync=True)
        canvas = window.get_canvas()
        scene  = window.get_scene()
        camera = ti.ui.Camera()

        camera.position(0, 3, 5)
        camera.lookat(0, 0, 0)
        camera.up(0, 1, 0)

        frame     = 0
        paused    = False
        show_axes = True

        print("Space=pause  R=restart  O=toggle axes  X/Esc=quit")

        while window.running:

            # --- input ---
            if window.get_event(ti.ui.PRESS):
                k = window.event.key
                if k in (ti.ui.ESCAPE, 'x'):
                    break
                if k == ' ':
                    paused = not paused
                if k == 'r':
                    frame = 0;  paused = False
                if k == 'o':
                    show_axes = not show_axes

            # --- update fields for this frame ---
            self._update_frame(frame)

            # --- camera ---
            camera.track_user_inputs(window, movement_speed=0.05, hold_key=ti.ui.LMB)
            scene.set_camera(camera)

            # --- lighting ---
            scene.ambient_light((0.3, 0.3, 0.3))
            scene.point_light((0.0, -1.0, 0.0), color=(1.0, 1.0, 1.0))
            scene.point_light((0.0, 10.0, 0.0), color=(1.0, 1.0, 1.0))

            # --- ground ---
            scene.mesh(self._ground_v, indices=self._ground_i, per_vertex_color=self._ground_c)
            scene.lines(self._grid_verts, width=1.0, color=(0.2, 0.2, 0.3))

            # --- axes ---
            if show_axes:
                for i in range(3):
                    scene.lines(self._world_axis_verts[i], width=3.0, color=self._AXIS_COLORS[i])
                for i in range(3):
                    scene.lines(self._body_axis_verts[i],  width=2.0, color=self._AXIS_COLORS[i])

            # --- body sphere ---
            scene.particles(self._body_pos, radius=self._sphere_r, color=(1.0, 0.4, 0.1))

            # --- subclass hook ---
            self._draw_extras(scene, frame)

            # --- HUD ---
            cx, cy, cz = self._centers[frame]
            canvas.scene(scene)
            window.GUI.begin("Info", 0.01, 0.01, 0.30, 0.22)
            window.GUI.text(f"Frame : {frame} / {self._T - 1}")
            window.GUI.text(f"Time  : {frame * self._dt:.2f} s")
            window.GUI.text(f"Pos (ENU): x={cx:.2f}  y={-cz:.2f}  z={cy:.2f}")
            window.GUI.text(f"Axes  : {'ON  [O]' if show_axes else 'OFF [O]'}")
            window.GUI.text(f"{'[PAUSED]' if paused else '[PLAYING]'}")
            window.GUI.end()

            window.show()

            # --- advance ---
            if not paused:
                if frame < self._T - 1:
                    frame += 1
                else:
                    paused = True
                    print("Playback complete. Press R to restart.")







class PositionControlRenderer(BaseRenderer):
    """
    Extends BaseRenderer with a static target-position sphere.

    Extra args:
        target_pos : (3,) ENU target position [m]
    """

    def __init__(self, target_pos: np.ndarray, **kwargs):
        super().__init__(**kwargs)

        self.target_pos = np.array(target_pos, dtype=np.float32)  # (3,) ENU

        # Preallocate target field once
        self._target_ti = ti.Vector.field(3, dtype=ti.f32, shape=1)
        self._update_target()

    def _update_target(self):
        t = enu_to_ti(self.target_pos[None])[0]   # (3,) Taichi
        self._target_ti[0] = ti.Vector(t.tolist())

    def _draw_extras(self, scene, frame: int):
        # Caller can update self.target_pos between frames if target moves
        self._update_target()
        scene.particles(self._target_ti, radius=self._sphere_r * 0.6, color=(0.1, 0.9, 0.3))
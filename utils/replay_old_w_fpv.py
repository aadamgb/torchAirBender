import taichi as ti
import numpy as np
import math


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def enu_to_ti(pts: np.ndarray) -> np.ndarray:
    """ENU (x-East, y-North, z-Up) -> Taichi (x, y=z_enu, z=-y_enu)"""
    out = pts[..., [0, 2, 1]].copy()
    out[..., 2] *= -1
    return out


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Hamilton [w, x, y, z] -> 3x3 rotation matrix (ENU)."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),  2*(x*y - w*z),      2*(x*z + w*y)],
        [    2*(x*y + w*z),  1 - 2*(x*x + z*z),  2*(y*z - w*x)],
        [    2*(x*z - w*y),  2*(y*z + w*x),       1 - 2*(x*x + y*y)],
    ], dtype=np.float32)


def build_triangle_at_tip(
    tip: np.ndarray,        # (3,) tip position in Taichi coords
    thrust_dir: np.ndarray, # (3,) normalised thrust direction in Taichi coords
    half_base: float,       # half-width of the base (lateral spread)
    height: float,          # full height along thrust direction
    angle: float = 0.0,     # rotation around thrust_dir in radians
) -> np.ndarray:
    """
    Returns vertices for one isoceles triangle centred at `tip`.
    - Base edge is parallel to the xy body plane (perpendicular to thrust_dir).
    - Apex points along +thrust_dir.
    - `angle` rotates the lateral axis around thrust_dir for pinwheel placement.

    Returns shape (3, 3): [vertex_index, xyz]
    """
    d = thrust_dir / (np.linalg.norm(thrust_dir) + 1e-12)

    # Base lateral axis perpendicular to d, then rotated by `angle` around d
    ref = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    if abs(np.dot(d, ref)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    u0 = np.cross(d, ref);  u0 /= np.linalg.norm(u0)
    v0 = np.cross(d, u0);   v0 /= np.linalg.norm(v0)
    # Rodrigues rotation of u0 around d by `angle`
    u = math.cos(angle) * u0 + math.sin(angle) * v0

    apex   = tip + (2.0/3.0) * height * d
    base_l = tip - (1.0/3.0) * height * d - half_base * u
    base_r = tip - (1.0/3.0) * height * d + half_base * u

    return np.array([apex, base_l, base_r], dtype=np.float32)


# ---------------------------------------------------------------------------
# BaseRenderer
# ---------------------------------------------------------------------------

class BaseRenderer:
    """
    Renders:
      - Checkerboard ground plane + grid lines
      - World-frame axes  (X=red, Y=green, Z=blue)
      - Quadrotor body: two crossing arm lines + centre sphere + 4 motor spheres
      - Body-frame axes
      - Thrust vectors with equilateral triangle arrowheads at tips

    Controls:
        LMB + drag  : orbit camera
        Scroll      : zoom
        Space       : pause / resume
        R           : restart
        O           : toggle axes
        X / Esc     : quit
    """

    _AXIS_COLORS = [
        (0.9, 0.15, 0.15),   # X - red
        (0.15, 0.9, 0.15),   # Y - green
        (0.15, 0.15, 0.9),   # Z - blue
    ]

    def __init__(
        self,
        trajectory:  np.ndarray,  # (T, 13) ENU: pos[0:3] vel[3:6] quat_wxyz[6:10] omega[10:13]
        arm_length:  float,
        arm_angle:   float,
        mass:        float,
        dt:          float,
        window_size: tuple = (1280, 720),
    ):
        self._traj       = np.array(trajectory, dtype=np.float32)
        self._T          = len(self._traj)
        self._arm_len    = arm_length
        self._arm_angle  = arm_angle
        self._dt         = dt
        self._win_size   = window_size
        self._axis_scale = arm_length * 1.5

        # Sphere radius scales with mass: normalised so ~0.5 kg -> ~arm*0.10
        # self._sphere_r = arm_length * 0.08 * (mass / 0.5) ** (1/3) 
        self._sphere_r = arm_length * 0.08 * (mass / 0.5) ** (1/3) 
        self._motor_r  = self._sphere_r * 0.75

        # Precompute all positions in Taichi coords
        centers_enu   = self._traj[:, 0:3]           # (T, 3) ENU
        actions      = self._traj[:, 13:17]   # (T, 4) thrust per motor [N]
        self._centers = enu_to_ti(centers_enu)        # (T, 3) Taichi

        # Precompute body axis tips in Taichi coords
        bases_enu = self._axis_scale * np.eye(3, dtype=np.float32)
        tips_enu  = np.zeros((self._T, 3, 3), dtype=np.float32)
        for t in range(self._T):
            R = quat_to_rotmat(self._traj[t, 6:10])
            tips_enu[t] = centers_enu[t] + (R @ bases_enu.T).T
        self._body_axis_tips = enu_to_ti(tips_enu)    # (T, 3, 3) Taichi

        # Precompute arm tip positions in Taichi coords
        # 4 arms radiating from centre using arm_angle offset
        s, c = math.sin(math.radians(arm_angle)), math.cos(math.radians(arm_angle))
        # s, c = math.sin(arm_angle), math.cos(arm_angle)
        # arm_dirs_body = arm_length * np.array([
        #     [ s, -c, 0],   # motor 1
        #     [ s,  c, 0],   # motor 2
        #     [-s,  c, 0],   # motor 3
        #     [-s, -c, 0],   # motor 4
        # ], dtype=np.float32)                           # (4, 3) body-frame ENU
        arm_dirs_body = arm_length * np.array([
            [c, - s, 0],   # motor 1 → Q1
            [-c,  s, 0],   # motor 2 → Q2
            [c, s, 0],   # motor 3 → Q3
            [ -c, -s, 0],   # motor 4 → Q4
        ])

        arm_tips_enu = np.zeros((self._T, 4, 3), dtype=np.float32)
        for t in range(self._T):
            R = quat_to_rotmat(self._traj[t, 6:10])
            arm_tips_enu[t] = centers_enu[t] + (R @ arm_dirs_body.T).T
        self._arm_tips = enu_to_ti(arm_tips_enu)       # (T, 4, 3) Taichi

        # Thrust vector
        thrust_scale = 0.05
        body_z = np.array([0, 0, 1], dtype=np.float32)

        thrust_tips_enu = np.zeros((self._T, 4, 3), dtype=np.float32)
        amplify = 1000.0   # tune this — higher = more exaggerated differences
        amplify = 0.01   # tune this — higher = more exaggerated differences

        for t in range(self._T):
            R = quat_to_rotmat(self._traj[t, 6:10])
            world_z = R @ body_z
            mean_thrust = actions[t].mean()
            for i in range(4):
                deviation = (actions[t, i] - mean_thrust) / (mean_thrust + 1e-6)  # relative diff from mean
                magnitude = max(0.0, 1.0 + deviation * amplify)                    # 1.0 = hover length, clamp to >= 0
                thrust_tips_enu[t, i] = arm_tips_enu[t, i] + world_z * magnitude * thrust_scale

        self._thrust_tips = enu_to_ti(thrust_tips_enu)    # (T, 4, 3)

        # ---------------------------------------------------------------
        # Precompute arrowhead triangle vertices in Taichi coords
        # Shape: (T, 4 motors, 4 arrows, 3 verts, 3 xyz)
        # 4 arrows evenly spaced (90° apart) around the thrust axis per motor
        # ---------------------------------------------------------------
        half_base  = arm_length * 0.08   # half-width of base
        tri_height = arm_length * 0.22   # height — taller than base
        n_arrows   = 4
        arrow_angles = [k * math.pi / 4 for k in range(n_arrows)]  # 0, 45, 90, 135 deg
        self._arrow_tris = np.zeros((self._T, 4, n_arrows, 3, 3), dtype=np.float32)

        for t in range(self._T):
            R = quat_to_rotmat(self._traj[t, 6:10])
            world_z_enu = R @ body_z                         # (3,) ENU
            # Convert direction vector to Taichi coords (swap axes, negate z)
            world_z_ti = np.array([
                world_z_enu[0],
                world_z_enu[2],
               -world_z_enu[1],
            ], dtype=np.float32)

            for i in range(4):
                tip_ti = self._thrust_tips[t, i]             # (3,) Taichi
                for k, angle in enumerate(arrow_angles):
                    self._arrow_tris[t, i, k] = build_triangle_at_tip(
                        tip=tip_ti,
                        thrust_dir=world_z_ti,
                        half_base=half_base,
                        height=tri_height,
                        angle=angle,
                    )                                        # (3, 3)

        ti.init(arch=ti.cpu)

        self._build_ground()
        self._build_world_axes()
        self._build_body_fields()

    # ------------------------------------------------------------------
    # Geometry builders - called once in __init__
    # ------------------------------------------------------------------

    def _build_ground(self):
        grid_n    = 10
        grid_half = 10
        edge      = 2.0 * grid_half / (grid_n - 1)

        self._grid_verts = ti.Vector.field(3, dtype=ti.f32, shape=4 * grid_n)
        gi = 0
        for k in range(grid_n):
            v = -grid_half + 2 * grid_half * k / (grid_n - 1)
            self._grid_verts[gi + 0] = ti.Vector([v,          0.002, -grid_half])
            self._grid_verts[gi + 1] = ti.Vector([v,          0.002,  grid_half])
            self._grid_verts[gi + 2] = ti.Vector([-grid_half, 0.002,  v])
            self._grid_verts[gi + 3] = ti.Vector([ grid_half, 0.002,  v])
            gi += 4

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
        tips = [[s, 0, 0], [0, 0, -s], [0, s, 0]]
        self._world_axis_verts = [
            ti.Vector.field(3, dtype=ti.f32, shape=2) for _ in range(3)
        ]
        for i in range(3):
            self._world_axis_verts[i][0] = ti.Vector([0.0, 0.0, 0.0])
            self._world_axis_verts[i][1] = ti.Vector(tips[i])

    def _build_body_fields(self):
        # Centre sphere
        self._body_pos = ti.Vector.field(3, dtype=ti.f32, shape=1)
        # Body-frame axes: 3 axes x 2 verts, preallocated to avoid recompile lag
        self._body_axis_verts = [
            ti.Vector.field(3, dtype=ti.f32, shape=2) for _ in range(3)
        ]
        # Arm lines: 4 arms x 2 verts, interleaved [origin, tip, ...]
        self._arm_verts = ti.Vector.field(3, dtype=ti.f32, shape=8)
        # Motor spheres at arm tips
        self._motor_pos = ti.Vector.field(3, dtype=ti.f32, shape=4)

        self._thrust_verts = ti.Vector.field(3, dtype=ti.f32, shape=8)   # 4 x [origin, tip]

        # Arrowhead triangles: 4 motors * 4 arrows * 3 verts = 48 verts total
        self._arrow_v = ti.Vector.field(3, dtype=ti.f32, shape=48)   # vertex positions
        self._arrow_i = ti.field(dtype=ti.i32,           shape=48)   # index buffer
        # Fill index buffer once — each triangle is 3 consecutive verts
        for tri_id in range(16):   # 4 motors * 4 arrows
            base = tri_id * 3
            self._arrow_i[base + 0] = base
            self._arrow_i[base + 1] = base + 1
            self._arrow_i[base + 2] = base + 2

    # ------------------------------------------------------------------
    # Per-frame field update
    # ------------------------------------------------------------------

    def _update_frame(self, frame: int):
        cx, cy, cz = self._centers[frame]
        origin = ti.Vector([cx, cy, cz])
        self._body_pos[0] = origin

        # Body-frame axes
        for i in range(3):
            tx, ty, tz = self._body_axis_tips[frame, i]
            self._body_axis_verts[i][0] = origin
            self._body_axis_verts[i][1] = ti.Vector([tx, ty, tz])

        # Arm lines + motor sphere positions
        for i in range(4):
            tx, ty, tz = self._arm_tips[frame, i]
            tip = ti.Vector([tx, ty, tz])
            self._arm_verts[i * 2]     = origin
            self._arm_verts[i * 2 + 1] = tip
            self._motor_pos[i]         = tip

        # Thrust vectors
        for i in range(4):
            mx, my, mz = self._arm_tips[frame, i]
            tx, ty, tz = self._thrust_tips[frame, i]
            self._thrust_verts[i * 2]     = ti.Vector([mx, my, mz])
            self._thrust_verts[i * 2 + 1] = ti.Vector([tx, ty, tz])

        # Arrowhead triangles at each thrust tip (4 arrows per motor)
        for i in range(4):
            for k in range(4):
                base = (i * 4 + k) * 3
                for v in range(3):
                    x, y, z = self._arrow_tris[frame, i, k, v]
                    self._arrow_v[base + v] = ti.Vector([x, y, z])

    
    # ------------------------------------------------------------------
    # Hook for subclasses
    # ------------------------------------------------------------------

    def _handle_keys(self, window) -> None:
        """Override to handle subclass-specific keypresses."""
        pass

    def _draw_extras(self, scene, frame: int):
        """Override in subclasses to add extra draw calls (targets, traces, etc.)."""
        pass

    # ------------------------------------------------------------------
    # Main render loop
    # ------------------------------------------------------------------

    def run(self):
        window = ti.ui.Window("Pon DE rePlaY!", self._win_size, vsync=True)
        canvas = window.get_canvas()
        scene  = window.get_scene()
        camera = ti.ui.Camera()

        camera.position(0, 3, 5)
        camera.lookat(0, 0, 0)
        camera.up(0, 1, 0)

        frame     = 0
        paused    = False
        show_axes = True
        show_ground = False
        fpv_mode    = False
        prev_fpv_mode = False
        saved_cam   = {"pos": (0, 3, 5), "look": (0, 0, 0), "up": (0, 1, 0)}

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
                if k == 'g':
                    show_ground = not show_ground
                if k == 'f':
                    prev_fpv_mode = fpv_mode
                    fpv_mode = not fpv_mode
                    
                self._handle_keys(window)

            # --- update fields for this frame ---
            self._update_frame(frame)

            # --- camera ---
            if fpv_mode:
                cx, cy, cz = self._centers[frame]
                drone_pos  = np.array([cx, cy, cz])

                tx, ty, tz = self._body_axis_tips[frame, 0]   # body X = forward
                fwd = np.array([tx, ty, tz]) - drone_pos
                fwd /= np.linalg.norm(fwd) + 1e-12

                ux, uy, uz = self._body_axis_tips[frame, 2]   # body Z = up
                up  = np.array([ux, uy, uz]) - drone_pos
                up  /= np.linalg.norm(up) + 1e-12

                minus_y = np.cross(fwd, up)
                minus_y /= np.linalg.norm(minus_y) + 1e-12

                camera.position(cx, cy, cz)
                camera.lookat(cx + fwd[0], cy + fwd[1], cz + fwd[2])
                # camera.lookat(cx + minus_y[0], cy + minus_y[1], cz + minus_y[2])
                camera.up(float(up[0]), float(up[1]), float(up[2]))
            else:
                if prev_fpv_mode:   # just exited FPV — restore saved pose
                    p, l, u = saved_cam["pos"], saved_cam["look"], saved_cam["up"]
                    camera.position(*p)
                    camera.lookat(*l)
                    camera.up(*u)
                camera.track_user_inputs(window, movement_speed=0.05, hold_key=ti.ui.LMB)

            prev_fpv_mode = fpv_mode
            scene.set_camera(camera)

            # --- lighting ---
            scene.ambient_light((0.3, 0.3, 0.3))
            scene.point_light((0.0, -3.0, 0.0), color=(1.0, 1.0, 1.0))
            scene.point_light((0.0, 10.0, 0.0), color=(1.0, 1.0, 1.0))
            scene.point_light((-10.0, 10.0, 0.0), color=(1.0, 1.0, 1.0))

            # --- ground ---
            if show_ground:
                scene.mesh(self._ground_v, indices=self._ground_i, per_vertex_color=self._ground_c)
            scene.lines(self._grid_verts, width=1.0, color=(0.2, 0.2, 0.3), )

            # --- world / body axes ---
            if show_axes:
                for i in range(3):
                    scene.lines(self._world_axis_verts[i], width=3.0, color=self._AXIS_COLORS[i])
                for i in range(3):
                    scene.lines(self._body_axis_verts[i],  width=2.0, color=self._AXIS_COLORS[i])

            # --- drone body ---
            scene.lines(self._arm_verts,  width=3.0, color=(0.85, 0.85, 0.85))   # arms
            scene.particles(self._body_pos,  radius=self._sphere_r, color=(0.2, 0.2, 0.2))  # centre
            scene.particles(self._motor_pos, radius=self._motor_r,  color=(1.0, 0.4, 0.1))  # motors
            scene.lines(self._thrust_verts,   width=5.0, color=(0.0, 0.8, 1.0))   # thrust lines

            # --- thrust arrowhead triangles (two perpendicular equilateral triangles per tip) ---
            scene.mesh(self._arrow_v, indices=self._arrow_i, color=(0.0, 0.8, 1.0))

            # --- subclass hook ---
            self._draw_extras(scene, frame)

            # --- HUD ---
            cx, cy, cz = self._centers[frame]
            canvas.scene(scene)
            with window.GUI.sub_window("Info", 0.01, 0.01, 0.32, 0.30) as sw:
                sw.text(f"Frame : {frame} / {self._T - 1}")
                sw.text(f"Time  : {frame * self._dt:.2f} s")
                sw.text(f"Pos (ENU): x={cx:.2f}  y={-cz:.2f}  z={cy:.2f}")
                show_axes   = sw.checkbox("Axes    [O]", show_axes)
                show_ground = sw.checkbox("Ground  [G]", show_ground)
                fpv_mode    = sw.checkbox("FPV     [F]", fpv_mode)
                sw.text(f"{'[PAUSED] Space=pause' if paused else '[PLAYING] Space=pause'}")

            window.show()

            # --- advance ---
            if not paused:
                if frame < self._T - 1:
                    frame += 1
                else:
                    paused = True
                    print("Playback complete. Press R to restart.")


# ---------------------------------------------------------------------------
# PositionControlRenderer
# ---------------------------------------------------------------------------

class PositionControlRenderer(BaseRenderer):
    """
    Extends BaseRenderer with a static target-position sphere.

    Extra args:
        target_pos : (3,) ENU target position [m]
    """
    def __init__(self, target_pos: np.ndarray, boundary: float, **kwargs):
        super().__init__(**kwargs)
        self.show_cube = True        # ← owns its own toggle state
        target_pos = np.array(target_pos, dtype=np.float32)
        if target_pos.ndim == 1:
            target_pos = target_pos[None, :]          # (1, 3) — broadcast to all frames
        self.target_pos = target_pos                  # (T, 3) or (1, 3)

        self._target_ti = ti.Vector.field(3, dtype=ti.f32, shape=1)
        self._update_target(0)

        # Build boundary cube edges (12 edges x 2 verts = 24 verts)
        h = boundary / 2.0
        # 8 corners in ENU: x in [-h,h], y in [-h,h], z in [0, boundary]
        corners_enu = np.array([
            [-h, -h, 0],        [-h, -h, boundary],
            [ h, -h, 0],        [ h, -h, boundary],
            [-h,  h, 0],        [-h,  h, boundary],
            [ h,  h, 0],        [ h,  h, boundary],
        ], dtype=np.float32)

        # 12 edges as pairs of corner indices
        edge_indices = [
            (0,1),(2,3),(4,5),(6,7),   # 4 vertical edges
            (0,2),(1,3),(4,6),(5,7),   # 4 bottom + top x-edges
            (0,4),(1,5),(2,6),(3,7),   # 4 bottom + top y-edges
        ]

        corners_ti = enu_to_ti(corners_enu)
        self._cube_verts = ti.Vector.field(3, dtype=ti.f32, shape=len(edge_indices) * 2)
        for k, (a, b) in enumerate(edge_indices):
            self._cube_verts[k * 2]     = ti.Vector(corners_ti[a].tolist())
            self._cube_verts[k * 2 + 1] = ti.Vector(corners_ti[b].tolist())

    def _handle_keys(self, window) -> None:
        if window.event.key == 'b':
            self.show_cube = not self.show_cube


    def _update_target(self, frame: int):
        # Clamp frame index so (1, 3) static targets work too
        idx = min(frame, len(self.target_pos) - 1)
        t = enu_to_ti(self.target_pos[idx][None])[0]
        self._target_ti[0] = ti.Vector(t.tolist())

    def _draw_extras(self, scene, frame: int):
        self._update_target(frame)                   
        scene.particles(self._target_ti, radius=self._sphere_r * 2.0, color=(0.1, 0.9, 0.3))
        if self.show_cube:
            scene.lines(self._cube_verts, width=3.0, color=(1.0, 0.1, 0.1))


# ---------------------------------------------------------------------------
# TrajectoryTrackingRenderer
# ---------------------------------------------------------------------------

class TrajectoryTrackingRenderer(BaseRenderer):
    """
    Extends BaseRenderer with:
      - A reference trajectory path drawn as a polyline
      - A moving target sphere at the current reference position
    
    Extra args:
        ref_trajectory : (T, 3) ENU reference positions for env 0
    """
    def __init__(self, ref_trajectory: np.ndarray, **kwargs):
        super().__init__(**kwargs)

        self._ref_enu = np.array(ref_trajectory, dtype=np.float32)  # (T, 3) ENU

        # Convert full path to Taichi coords
        self._ref_ti = enu_to_ti(self._ref_enu)                      # (T, 3)

        # Build polyline: pairs of consecutive points [p0,p1, p1,p2, p2,p3, ...]
        n_segments = self._T - 1
        self._path_verts = ti.Vector.field(3, dtype=ti.f32, shape=n_segments * 2)
        for i in range(n_segments):
            self._path_verts[i * 2]     = ti.Vector(self._ref_ti[i].tolist())
            self._path_verts[i * 2 + 1] = ti.Vector(self._ref_ti[i + 1].tolist())

        # Moving target sphere — updated per frame
        self._target_ti = ti.Vector.field(3, dtype=ti.f32, shape=1)
        self._target_ti[0] = ti.Vector(self._ref_ti[0].tolist())

    def _update_target(self, frame: int):
        self._target_ti[0] = ti.Vector(self._ref_ti[frame].tolist())

    def _draw_extras(self, scene, frame: int):
        self._update_target(frame)
        # Reference path — dim yellow
        scene.lines(self._path_verts, width=2.0, color=(0.9, 0.8, 0.1))
        # Current target point on path — bright green sphere
        scene.particles(self._target_ti, radius=self._sphere_r * 2.0, color=(1.0, 0.0, 0.0))


# ---------------------------------------------------------------------------
# RacingRenderer
# ---------------------------------------------------------------------------

class RacingRenderer(TrajectoryTrackingRenderer):
    """
    Extends TrajectoryTrackingRenderer with racing gates loaded from a .obj file.

    Args:
        gates_position : (N, 3) ENU gate centre positions [m]
        gates_rpy      : (N, 3) or (3,) Roll/Pitch/Yaw in degrees
                         If shape (3,) the same orientation is applied to all gates.
        gate_mesh_path : path to gate .obj (or .glb) file
        gate_scale     : uniform scale applied to the raw mesh vertices
        gate_color     : RGB tuple for gate rendering color
        ref_trajectory : optional (T, 3) ENU reference path — if None, path/target
                         sphere are suppressed
    """

    def __init__(
        self,
        gates_position: np.ndarray,
        gates_rpy:      np.ndarray,
        gate_mesh_path: str = '/home/adame/torchAirBender/miscellaneous/gate.obj',
        gate_scale:     float = 1.0,
        gate_color:     tuple = (0.25, 0.0, 0.5),
        ref_trajectory: np.ndarray = None,
        **kwargs,
    ):
        # ── optional ref trajectory ──────────────────────────────────────
        # TrajectoryTrackingRenderer requires ref_trajectory, so build a
        # dummy (static point at frame-0 position) when none is supplied.
        traj_np = np.array(kwargs["trajectory"], dtype=np.float32)
        if ref_trajectory is None:
            ref_trajectory = np.tile(traj_np[0, 0:3], (len(traj_np), 1))
            self._has_ref_traj = False
        else:
            ref_trajectory = np.array(ref_trajectory, dtype=np.float32)
            self._has_ref_traj = True

        super().__init__(ref_trajectory=ref_trajectory, **kwargs)

        self._gate_color = gate_color

        # ── load mesh ────────────────────────────────────────────────────
        try:
            import trimesh
        except ImportError:
            raise ImportError("trimesh is required for RacingRenderer: pip install trimesh")

        raw = trimesh.load(gate_mesh_path, force="mesh")
        if isinstance(raw, trimesh.Scene):
            raw = trimesh.util.concatenate(tuple(raw.geometry.values()))

        # Local body-frame vertices + faces
        verts_body = raw.vertices.astype(np.float32) * gate_scale   # (V, 3)
        faces      = raw.faces.astype(np.int32)                      # (F, 3)
        V = len(verts_body)
        F = len(faces)

        # ── normalise gates_rpy to (N, 3) ───────────────────────────────
        gates_position = np.array(gates_position, dtype=np.float32)  # (N, 3) ENU
        gates_rpy      = np.array(gates_rpy,      dtype=np.float32)
        N = len(gates_position)
        if gates_rpy.ndim == 1:
            gates_rpy = np.tile(gates_rpy, (N, 1))                   # broadcast

        # ── build world-space vertices for every gate ────────────────────
        all_verts_enu = np.zeros((N * V, 3), dtype=np.float32)

        for g in range(N):
            R   = self._rpy_deg_to_rotmat(gates_rpy[g])              # 3x3
            pos = gates_position[g]                                   # (3,) ENU
            # rotate body verts then translate to gate position (all in ENU)
            all_verts_enu[g * V : (g + 1) * V] = (R @ verts_body.T).T + pos

        # Convert all vertices to Taichi coords in one call
        all_verts_ti = enu_to_ti(all_verts_enu)                      # (N*V, 3)

        # ── build index buffer — offset each gate by g*V ─────────────────
        all_faces = np.zeros((N * F, 3), dtype=np.int32)
        for g in range(N):
            all_faces[g * F : (g + 1) * F] = faces + g * V

        # ── upload to Taichi fields ───────────────────────────────────────
        self._gate_v = ti.Vector.field(3, dtype=ti.f32, shape=N * V)
        self._gate_i = ti.field(dtype=ti.i32,           shape=N * F * 3)

        self._gate_v.from_numpy(all_verts_ti)
        self._gate_i.from_numpy(all_faces.flatten())

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _rpy_deg_to_rotmat(rpy_deg: np.ndarray) -> np.ndarray:
        """ZYX extrinsic (aerospace convention): Rz @ Ry @ Rx."""
        r, p, y = np.radians(rpy_deg).tolist()
        cr, sr = math.cos(r), math.sin(r)
        cp, sp = math.cos(p), math.sin(p)
        cy, sy = math.cos(y), math.sin(y)

        Rx = np.array([[1,  0,   0 ],
                       [0,  cr, -sr],
                       [0,  sr,  cr]], dtype=np.float32)

        Ry = np.array([[ cp, 0, sp],
                       [  0, 1,  0],
                       [-sp, 0, cp]], dtype=np.float32)

        Rz = np.array([[cy, -sy, 0],
                       [sy,  cy, 0],
                       [ 0,   0, 1]], dtype=np.float32)

        return Rz @ Ry @ Rx

    # ------------------------------------------------------------------
    # Draw
    # ------------------------------------------------------------------

    def _draw_extras(self, scene, frame: int):
        # Reference path + moving target — only when a real ref was given
        if self._has_ref_traj:
            super()._draw_extras(scene, frame)

        # Gates
        scene.mesh(
            self._gate_v,
            indices=self._gate_i,
            color=self._gate_color,
            # two_sided=True,
        )
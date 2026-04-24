import taichi as ti
import numpy as np
import math
# ---------------------------------------------------------------------------
# Drone colors
# ---------------------------------------------------------------------------

DRONE_COLORS = [
    (1.0, 0.3, 0.3),   # red
    (0.2, 1.0, 0.4),   # green
    (0.2, 0.6, 1.0),   # blue
    (1.0, 0.4, 0.1),   # orange
    (1.0, 0.2, 0.8),   # pink
    (1.0, 1.0, 0.2),   # yellow
    (0.8, 0.2, 1.0),   # purple
]

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
    tip: np.ndarray,
    thrust_dir: np.ndarray,
    half_base: float,
    height: float,
    angle: float = 0.0,
) -> np.ndarray:
    d = thrust_dir / (np.linalg.norm(thrust_dir) + 1e-12)
    ref = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    if abs(np.dot(d, ref)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    u0 = np.cross(d, ref);  u0 /= np.linalg.norm(u0)
    v0 = np.cross(d, u0);   v0 /= np.linalg.norm(v0)
    u = math.cos(angle) * u0 + math.sin(angle) * v0

    apex   = tip + (2.0/3.0) * height * d
    base_l = tip - (1.0/3.0) * height * d - half_base * u
    base_r = tip - (1.0/3.0) * height * d + half_base * u

    return np.array([apex, base_l, base_r], dtype=np.float32)



# ---------------------------------------------------------------------------
# BaseRenderer  — ground, world axes, camera loop, lighting
# ---------------------------------------------------------------------------

class BaseRenderer:
    """
    Handles everything that is shared and scene-independent:
      - Checkerboard ground plane + grid lines
      - World-frame axes (X=red, Y=green, Z=blue)
      - Camera, lighting, main render loop

    Subclasses implement:
      _update_frame(frame)  — update per-frame Taichi fields
      _draw_extras(scene, frame) — extra draw calls
      _handle_keys(window)  — optional extra key handling
    """

    _AXIS_COLORS = [
        (0.9, 0.15, 0.15),
        (0.15, 0.9, 0.15),
        (0.15, 0.15, 0.9),
    ]

    def __init__(
        self,
        T:           int,
        arm_length:  float = 0.15,
        arm_angle:   float = 45.0,
        mass:        float = 1.2,
        dt:          float = 0.01,
        window_size: tuple = (1280, 720),
    ):
        self._arm_len    = arm_length
        self._arm_angle  = arm_angle
        self._dt         = dt
        self._T          = T
        self._win_size   = window_size
        self._axis_scale = arm_length * 1.5
        self._sphere_r   = arm_length * 0.08 * (mass / 0.5) ** (1/3)
        self._motor_r    = self._sphere_r * 0.75

        ti.init(arch=ti.cpu)

        self._build_ground()
        self._build_world_axes()

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

    def _handle_keys(self, window) -> None:
        pass

    def _update_frame(self, frame: int) -> None:
        pass

    def _draw_extras(self, scene, frame: int) -> None:
        pass

    def run(self):
        window = ti.ui.Window("Pon DE rePlaY!", self._win_size, vsync=True)
        canvas = window.get_canvas()
        scene  = window.get_scene()
        camera = ti.ui.Camera()

        camera.position(0, 3, 5)
        camera.lookat(0, 0, 0)
        camera.up(0, 1, 0)

        frame       = 0
        paused      = False
        show_axes   = True
        show_ground = False

        print("Space=pause  R=restart  O=toggle axes  G=toggle ground  X/Esc=quit")

        while window.running:

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
                self._handle_keys(window)

            self._update_frame(frame)

            camera.track_user_inputs(window, movement_speed=0.05, hold_key=ti.ui.LMB)
            scene.set_camera(camera)

            scene.ambient_light((0.3, 0.3, 0.3))
            scene.point_light((0.0, -3.0, 0.0), color=(1.0, 1.0, 1.0))
            scene.point_light((0.0, 10.0, 0.0), color=(1.0, 1.0, 1.0))
            scene.point_light((-10.0, 10.0, 0.0), color=(1.0, 1.0, 1.0))

            if show_ground:
                scene.mesh(self._ground_v, indices=self._ground_i, per_vertex_color=self._ground_c)
            scene.lines(self._grid_verts, width=1.0, color=(0.2, 0.2, 0.3))

            if show_axes:
                for i in range(3):
                    scene.lines(self._world_axis_verts[i], width=3.0, color=self._AXIS_COLORS[i])

            self._draw_extras(scene, frame)

            canvas.scene(scene)
            self._draw_hud(window, frame, paused, show_axes, show_ground)

            window.show()

            if not paused:
                if frame < self._T - 1:
                    frame += 1
                else:
                    paused = True
                    print("Playback complete. Press R to restart.")

    def _draw_hud(self, window, frame, paused, show_axes, show_ground):
        """Override in subclasses to customize the HUD."""
        with window.GUI.sub_window("Info", 0.01, 0.01, 0.95, 0.12) as sw:
            sw.text(f"Frame : {frame} / {self._T - 1}")
            sw.text(f"Time  : {frame * self._dt:.2f} s")
            sw.text(f"{'[PAUSED]' if paused else '[PLAYING]'}  Space=pause")


# ---------------------------------------------------------------------------
# MultiDroneRenderer
# Renders 1..N drones tracking a shared reference trajectory
# ---------------------------------------------------------------------------

class MultiDroneRenderer(BaseRenderer):
    """
    Renders one or more drones tracking a shared reference trajectory.

    Parameters
    ----------
    drones : list of dict, each with:
        "traj"  : np.ndarray (T, 20) — states(13) | actions(4) | pos_ref(3)
        "color" : tuple (r, g, b)    — optional, defaults to DRONE_COLORS[i]
        "label" : str                — shown in HUD
    ref_trajectory : np.ndarray (T, 3) ENU — shared reference path

    For backward compatibility a single trajectory array can be passed
    directly as `trajectory` (used by RacingRenderer and PositionControlRenderer).
    """

    def __init__(
        self,
        ref_trajectory: np.ndarray,
        arm_length:     float = 0.15,
        arm_angle:      float = 45.0,
        mass:           float = 1.2,
        dt:             float = 0.01,
        drones:         list = None,
        trajectory:     np.ndarray = None,   # backward-compat single traj
        window_size:    tuple = (1280, 720),
    ):
        # Normalise input — accept either `drones` list or single `trajectory`
        if drones is None and trajectory is None:
            raise ValueError("Provide either `drones` or `trajectory`.")

        if drones is None:
            drones = [{"traj": trajectory, "color": DRONE_COLORS[0], "label": "drone"}]

        # Fill in missing colors
        for i, d in enumerate(drones):
            if "color" not in d:
                d["color"] = DRONE_COLORS[i % len(DRONE_COLORS)]

        T = len(drones[0]["traj"])

        super().__init__(
            arm_length  = arm_length,
            arm_angle   = arm_angle,
            mass        = mass,
            dt          = dt,
            T           = T,
            window_size = window_size,
        )

        self._drones = drones
        self._n      = len(drones)

        # Reference path
        self._ref_enu = np.array(ref_trajectory, dtype=np.float32)
        self._ref_ti  = enu_to_ti(self._ref_enu)

        # ------------------------------------------------------------------
        # Precompute per-drone numpy arrays
        # ------------------------------------------------------------------
        body_z        = np.array([0, 0, 1], dtype=np.float32)
        s_ang         = math.sin(math.radians(arm_angle))
        c_ang         = math.cos(math.radians(arm_angle))
        arm_dirs_body = arm_length * np.array([
            [ c_ang, -s_ang, 0],
            [-c_ang,  s_ang, 0],
            [ c_ang,  s_ang, 0],
            [-c_ang, -s_ang, 0],
        ], dtype=np.float32)

        thrust_scale = 0.05
        amplify      = 0.01
        half_base    = arm_length * 0.08
        tri_height   = arm_length * 0.22
        n_arrows     = 4
        arrow_angles = [k * math.pi / 4 for k in range(n_arrows)]

        self._all_centers     = []
        self._all_axis_tips   = []
        self._all_arm_tips    = []
        self._all_thrust_tips = []
        self._all_arrow_tris  = []

        for d in drones:
            traj        = np.array(d["traj"], dtype=np.float32)
            centers_enu = traj[:, 0:3]
            actions     = traj[:, 26:30]
            centers_ti  = enu_to_ti(centers_enu)

            # Body-axis tips
            bases_enu   = self._axis_scale * np.eye(3, dtype=np.float32)
            tips_enu    = np.zeros((T, 3, 3), dtype=np.float32)
            for t in range(T):
                R           = quat_to_rotmat(traj[t, 6:10])
                tips_enu[t] = centers_enu[t] + (R @ bases_enu.T).T
            axis_tips_ti = enu_to_ti(tips_enu)

            # Arm tips
            arm_tips_enu = np.zeros((T, 4, 3), dtype=np.float32)
            for t in range(T):
                R               = quat_to_rotmat(traj[t, 6:10])
                arm_tips_enu[t] = centers_enu[t] + (R @ arm_dirs_body.T).T
            arm_tips_ti = enu_to_ti(arm_tips_enu)

            # Thrust tips
            thrust_tips_enu = np.zeros((T, 4, 3), dtype=np.float32)
            for t in range(T):
                R           = quat_to_rotmat(traj[t, 6:10])
                world_z     = R @ body_z
                mean_thrust = actions[t].mean()
                for i in range(4):
                    deviation = (actions[t, i] - mean_thrust) / (mean_thrust + 1e-6)
                    magnitude = max(0.0, 1.0 + deviation * amplify)
                    thrust_tips_enu[t, i] = arm_tips_enu[t, i] + world_z * magnitude * thrust_scale
            thrust_tips_ti = enu_to_ti(thrust_tips_enu)

            # Arrowhead triangles
            arrow_tris = np.zeros((T, 4, n_arrows, 3, 3), dtype=np.float32)
            for t in range(T):
                R           = quat_to_rotmat(traj[t, 6:10])
                world_z_enu = R @ body_z
                world_z_ti  = np.array([world_z_enu[0], world_z_enu[2], -world_z_enu[1]], dtype=np.float32)
                for i in range(4):
                    tip_ti = thrust_tips_ti[t, i]
                    for k, angle in enumerate(arrow_angles):
                        arrow_tris[t, i, k] = build_triangle_at_tip(
                            tip        = tip_ti,
                            thrust_dir = world_z_ti,
                            half_base  = half_base,
                            height     = tri_height,
                            angle      = angle,
                        )

            self._all_centers.append(centers_ti)
            self._all_axis_tips.append(axis_tips_ti)
            self._all_arm_tips.append(arm_tips_ti)
            self._all_thrust_tips.append(thrust_tips_ti)
            self._all_arrow_tris.append(arrow_tris)

        # ------------------------------------------------------------------
        # Taichi fields — one set per drone, allocated once
        # ------------------------------------------------------------------
        self._body_axis_verts = [
            [ti.Vector.field(3, dtype=ti.f32, shape=2) for _ in range(3)]
            for _ in range(self._n)
        ]

        # Per-drone draw fields — allocated once, written each frame
        self._drone_body_pos     = [ti.Vector.field(3, dtype=ti.f32, shape=1)  for _ in range(self._n)]
        self._drone_arm_verts    = [ti.Vector.field(3, dtype=ti.f32, shape=8)  for _ in range(self._n)]
        self._drone_motor_pos    = [ti.Vector.field(3, dtype=ti.f32, shape=4)  for _ in range(self._n)]
        self._drone_thrust_verts = [ti.Vector.field(3, dtype=ti.f32, shape=8)  for _ in range(self._n)]
        self._drone_arrow_v      = [ti.Vector.field(3, dtype=ti.f32, shape=48) for _ in range(self._n)]
        self._drone_arrow_i      = [ti.field(dtype=ti.i32,           shape=48) for _ in range(self._n)]
        for d in range(self._n):
            for tri_id in range(16):
                base = tri_id * 3
                self._drone_arrow_i[d][base]     = base
                self._drone_arrow_i[d][base + 1] = base + 1
                self._drone_arrow_i[d][base + 2] = base + 2

        # Precompute thrust colors per drone
        self._thrust_colors = []
        for d in self._drones:
            c = d["color"]
            self._thrust_colors.append((c[0] * 0.5, c[1] * 0.5 + 0.4, c[2] * 0.5 + 0.5))

        # Reference path polyline + moving target
        n_seg                = T - 1
        self._path_verts     = ti.Vector.field(3, dtype=ti.f32, shape=n_seg * 2)
        for i in range(n_seg):
            self._path_verts[i * 2]     = ti.Vector(self._ref_ti[i].tolist())
            self._path_verts[i * 2 + 1] = ti.Vector(self._ref_ti[i + 1].tolist())

        self._target_ti    = ti.Vector.field(3, dtype=ti.f32, shape=1)
        self._target_ti[0] = ti.Vector(self._ref_ti[0].tolist())

    # ------------------------------------------------------------------
    # _update_frame
    # ------------------------------------------------------------------

    def _update_frame(self, frame: int):
        for d in range(self._n):
            cx, cy, cz = self._all_centers[d][frame]
            origin     = ti.Vector([cx, cy, cz])

            self._drone_body_pos[d][0] = origin

            for i in range(3):
                tx, ty, tz = self._all_axis_tips[d][frame, i]
                self._body_axis_verts[d][i][0] = origin
                self._body_axis_verts[d][i][1] = ti.Vector([tx, ty, tz])

            for i in range(4):
                tx, ty, tz = self._all_arm_tips[d][frame, i]
                tip        = ti.Vector([tx, ty, tz])
                self._drone_arm_verts[d][i * 2]     = origin
                self._drone_arm_verts[d][i * 2 + 1] = tip
                self._drone_motor_pos[d][i]          = tip

            for i in range(4):
                mx, my, mz = self._all_arm_tips[d][frame, i]
                tx, ty, tz = self._all_thrust_tips[d][frame, i]
                self._drone_thrust_verts[d][i * 2]     = ti.Vector([mx, my, mz])
                self._drone_thrust_verts[d][i * 2 + 1] = ti.Vector([tx, ty, tz])

            for i in range(4):
                for k in range(4):
                    base_v = (i * 4 + k) * 3
                    for v in range(3):
                        x, y, z = self._all_arrow_tris[d][frame, i, k, v]
                        self._drone_arrow_v[d][base_v + v] = ti.Vector([x, y, z])

        self._target_ti[0] = ti.Vector(self._ref_ti[frame].tolist())

    # ------------------------------------------------------------------
    # _draw_extras
    # ------------------------------------------------------------------

    def _draw_extras(self, scene, frame: int):
        # Reference path + target
        scene.lines(self._path_verts, width=2.0, color=(0.9, 0.8, 0.1))
        scene.particles(self._target_ti, radius=self._sphere_r * 2.0, color=(1.0, 0.0, 0.0))

        for d, drone in enumerate(self._drones):
            color        = drone["color"]
            thrust_color = self._thrust_colors[d]

            scene.particles(self._drone_body_pos[d],     radius=self._sphere_r, color=(0.2, 0.2, 0.2))
            scene.lines(    self._drone_arm_verts[d],    width=3.0,             color=(0.85, 0.85, 0.85))
            scene.particles(self._drone_motor_pos[d],    radius=self._motor_r,  color=color)
            scene.lines(    self._drone_thrust_verts[d], width=5.0,             color=thrust_color)
            scene.mesh(     self._drone_arrow_v[d], indices=self._drone_arrow_i[d], color=thrust_color)

        # Body axes
        for d, drone in enumerate(self._drones):
            for i in range(3):
                scene.lines(self._body_axis_verts[d][i], width=2.0, color=self._AXIS_COLORS[i])

    # ------------------------------------------------------------------
    # HUD
    # ------------------------------------------------------------------

    def _draw_hud(self, window, frame, paused, show_axes, show_ground):
        hud_h = 0.07 + self._n * 0.055
        with window.GUI.sub_window("Info", 0.01, 0.01, 0.25, hud_h) as sw:
            sw.text(f"Frame: {frame} / {self._T - 1}   Time: {frame * self._dt:.2f}s")
            sw.text(f"{'[PAUSED]' if paused else '[PLAYING]'}  Space=pause  R=restart")
            for d, drone in enumerate(self._drones):
                pos = self._all_centers[d][frame]
                c   = drone["color"]
                sw.text(
                    f"  {drone['label']:14s}  ({pos[0]:.2f}, {-pos[2]:.2f}, {pos[1]:.2f})",
                    color=(c[0], c[1], c[2])
                )


# ---------------------------------------------------------------------------
# PositionControlRenderer
# ---------------------------------------------------------------------------

class PositionControlRenderer(MultiDroneRenderer):
    """
    Extends MultiDroneRenderer with a target-position sphere and boundary cube.

    Extra args:
        target_pos : (3,) or (T, 3) ENU target position(s) [m]
        boundary   : side length of the boundary cube [m]
    """

    def __init__(self, trajectory: np.ndarray, target_pos: np.ndarray,
                 boundary: float, **kwargs):

        traj_np = np.array(trajectory, dtype=np.float32)
        super().__init__(
            trajectory     = traj_np,
            ref_trajectory = traj_np[:, 0:3],   
            **kwargs,
        )

        self.show_cube = True

        target_pos = np.array(target_pos, dtype=np.float32)
        if target_pos.ndim == 1:
            target_pos = target_pos[None, :]     # (1, 3) — broadcast to all frames
        self._target_pos = target_pos

        self._target_ti    = ti.Vector.field(3, dtype=ti.f32, shape=1)
        self._target_ti[0] = ti.Vector(enu_to_ti(self._target_pos[0][None])[0].tolist())

        # Boundary cube edges (12 edges x 2 verts)
        h = boundary / 2.0
        corners_enu = np.array([
            [-h, -h, 0], [-h, -h, boundary],
            [ h, -h, 0], [ h, -h, boundary],
            [-h,  h, 0], [-h,  h, boundary],
            [ h,  h, 0], [ h,  h, boundary],
        ], dtype=np.float32)
        edge_indices = [
            (0,1),(2,3),(4,5),(6,7),
            (0,2),(1,3),(4,6),(5,7),
            (0,4),(1,5),(2,6),(3,7),
        ]
        corners_ti       = enu_to_ti(corners_enu)
        self._cube_verts = ti.Vector.field(3, dtype=ti.f32, shape=len(edge_indices) * 2)
        for k, (a, b) in enumerate(edge_indices):
            self._cube_verts[k * 2]     = ti.Vector(corners_ti[a].tolist())
            self._cube_verts[k * 2 + 1] = ti.Vector(corners_ti[b].tolist())

    def _handle_keys(self, window) -> None:
        if window.event.key == 'b':
            self.show_cube = not self.show_cube

    def _draw_extras(self, scene, frame: int):
        # Draw drones + ref path from parent (ref path is dummy so nothing visible)
        for d, drone in enumerate(self._drones):
            color        = drone["color"]
            thrust_color = self._thrust_colors[d]

            scene.particles(self._drone_body_pos[d],     radius=self._sphere_r, color=(0.2, 0.2, 0.2))
            scene.lines(    self._drone_arm_verts[d],    width=3.0,             color=(0.85, 0.85, 0.85))
            scene.particles(self._drone_motor_pos[d],    radius=self._motor_r,  color=color)
            scene.lines(    self._drone_thrust_verts[d], width=5.0,             color=thrust_color)
            scene.mesh(     self._drone_arrow_v[d], indices=self._drone_arrow_i[d], color=thrust_color)

        # Update and draw target sphere
        idx = min(frame, len(self._target_pos) - 1)
        t   = enu_to_ti(self._target_pos[idx][None])[0]
        self._target_ti[0] = ti.Vector(t.tolist())
        scene.particles(self._target_ti, radius=self._sphere_r * 2.0, color=(0.1, 0.9, 0.3))

        # Boundary cube
        if self.show_cube:
            scene.lines(self._cube_verts, width=3.0, color=(1.0, 0.1, 0.1))


# ---------------------------------------------------------------------------
# RacingRenderer
# ---------------------------------------------------------------------------

class RacingRenderer(MultiDroneRenderer):
    """
    Extends MultiDroneRenderer with racing gates loaded from a .obj file.
    """

    def __init__(
        self,
        trajectory:     np.ndarray,
        gates_position: np.ndarray,
        gates_rpy:      np.ndarray,
        gate_mesh_path: str   = '/home/adame/torchAirBender/miscellaneous/gate.obj',
        gate_scale:     float = 1.0,
        gate_color:     tuple = (0.25, 0.0, 0.5),
        ref_trajectory: np.ndarray = None,
        **kwargs,
    ):
        traj_np = np.array(trajectory, dtype=np.float32)
        if ref_trajectory is None:
            ref_trajectory = np.tile(traj_np[0, 0:3], (len(traj_np), 1))
            self._has_ref_traj = False
        else:
            ref_trajectory     = np.array(ref_trajectory, dtype=np.float32)
            self._has_ref_traj = True

        super().__init__(
            trajectory     = traj_np,
            ref_trajectory = ref_trajectory,
            **kwargs,
        )

        self._gate_color = gate_color

        try:
            import trimesh
        except ImportError:
            raise ImportError("trimesh is required for RacingRenderer: pip install trimesh")

        raw = trimesh.load(gate_mesh_path, force="mesh")
        if isinstance(raw, trimesh.Scene):
            raw = trimesh.util.concatenate(tuple(raw.geometry.values()))

        verts_body = raw.vertices.astype(np.float32) * gate_scale
        faces      = raw.faces.astype(np.int32)
        V = len(verts_body)
        F = len(faces)

        gates_position = np.array(gates_position, dtype=np.float32)
        gates_rpy      = np.array(gates_rpy,      dtype=np.float32)
        N = len(gates_position)
        if gates_rpy.ndim == 1:
            gates_rpy = np.tile(gates_rpy, (N, 1))

        all_verts_enu = np.zeros((N * V, 3), dtype=np.float32)
        for g in range(N):
            R   = self._rpy_deg_to_rotmat(gates_rpy[g])
            pos = gates_position[g]
            all_verts_enu[g * V : (g + 1) * V] = (R @ verts_body.T).T + pos

        all_verts_ti = enu_to_ti(all_verts_enu)

        all_faces = np.zeros((N * F, 3), dtype=np.int32)
        for g in range(N):
            all_faces[g * F : (g + 1) * F] = faces + g * V

        self._gate_v = ti.Vector.field(3, dtype=ti.f32, shape=N * V)
        self._gate_i = ti.field(dtype=ti.i32,           shape=N * F * 3)
        self._gate_v.from_numpy(all_verts_ti)
        self._gate_i.from_numpy(all_faces.flatten())

    @staticmethod
    def _rpy_deg_to_rotmat(rpy_deg: np.ndarray) -> np.ndarray:
        r, p, y = np.radians(rpy_deg).tolist()
        cr, sr = math.cos(r), math.sin(r)
        cp, sp = math.cos(p), math.sin(p)
        cy, sy = math.cos(y), math.sin(y)
        Rx = np.array([[1,  0,   0 ], [0,  cr, -sr], [0,  sr,  cr]], dtype=np.float32)
        Ry = np.array([[ cp, 0, sp], [  0, 1,  0], [-sp, 0, cp]],   dtype=np.float32)
        Rz = np.array([[cy, -sy, 0], [sy,  cy, 0], [ 0,   0, 1]],   dtype=np.float32)
        return Rz @ Ry @ Rx

    def _draw_extras(self, scene, frame: int):
        if self._has_ref_traj:
            super()._draw_extras(scene, frame)
        scene.mesh(self._gate_v, indices=self._gate_i, color=self._gate_color)
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

import mplcyberpunk
plt.style.use("cyberpunk")
# darker background
DARK_BG = "#0d0d0d"  # tweak this to taste
plt.rcParams["figure.facecolor"] = DARK_BG
plt.rcParams["axes.facecolor"]   = DARK_BG
# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
KF             = 4.5e-8   # N/(rad/s)²
THRUST_COLORS  = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
WRENCH_LABELS  = ['Fz', 'Mx', 'My', 'Mz']
WRENCH_COLORS  = ['tab:cyan', 'tab:brown', 'tab:grey', 'tab:olive']
LVYR_LABELS    = [r'$v_x$', r'$v_y$', r'$v_z$', r'$\dot{\psi}$']
# LVYR_COLORS  =   ['tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']
LVYR_COLORS  =   ['tab:red', 'tab:green', 'tab:blue', 'yellow']
GAIN_LABELS    = [r'$k_v$', r'$k_R$', r'$k_\Omega$']
GAIN_COLORS    = ['tab:cyan', 'tab:purple', 'tab:pink']


# ──────────────────────────────────────────────────────────────────────────────
# Action unpacking
# ──────────────────────────────────────────────────────────────────────────────
def _infer_cm(traj_np: np.ndarray) -> str:
    return {26: "srt", 30: "ctbr", 34: "lvyr", 37: "lvyr+g"}[traj_np.shape[1]]

def _unpack_actions(traj_np: np.ndarray, cm: str) -> dict:
    """
    0:13   state
    13:16  p_ref
    16:19  v_ref
    19:22  a_ref
    22:26  srt        (always)
    26:30  wrench     (ctbr and above)
    30:34  lvyr cmds  (lvyr and above)
    34:37  gains      (lvyr+g only)
    """
    out = dict(srt=traj_np[:, 22:26])

    if cm in ("ctbr", "lvyr", "lvyr+g"):
        out["wrench"] = traj_np[:, 26:30]

    if cm in ("lvyr", "lvyr+g"):
        out["lvyr"] = traj_np[:, 30:34]

    if cm == "lvyr+g":
        out["gains"] = traj_np[:, 34:37]

    return out

def _rotor_speeds(srt: np.ndarray) -> np.ndarray:
    return np.sqrt(np.maximum(srt, 0.0) / KF)


# ──────────────────────────────────────────────────────────────────────────────
# Top-panel dispatcher  (rows 0-1, cols 0-2)
# ──────────────────────────────────────────────────────────────────────────────
def _plot_top_panel(fig, gs, t_np: np.ndarray, acts: dict, cm: str) -> None:
    """
    Populate rows 0-1, cols 0-2 according to control mode.

    srt
    ───
      col 0 row 0   : rotor speeds
      col 0 row 1   : (empty)
      cols 1-2 r0-1 : motor thrusts  (single combined plot)

    ctbr
    ────
      col 0 row 0   : motor thrusts
      col 0 row 1   : rotor speeds
      cols 1-2 r0-1 : wrench  [Fz, Mx, My, Mz]

    lvyr
    ────
      col 0 row 0   : wrench
      col 0 row 1   : motor thrusts
      cols 1-2 r0-1 : LVYR commands  [vx, vy, vz, ψ̇]

    lvyr+g
    ──────
      col 0 row 0   : wrench
      col 0 row 1   : motor thrusts
      cols 1-2 row 0: LVYR commands
      cols 1-2 row 1: adaptive gains  [kv, kR, kΩ]
    """

    # ── helpers ──────────────────────────────────────────────────────────
    def _ax(row, col, rowspan=1, colspan=1):
        if rowspan > 1 or colspan > 1:
            return fig.add_subplot(gs[row:row+rowspan, col:col+colspan])
        return fig.add_subplot(gs[row, col])

    def _multiline(ax, data, colors, labels, ylabel, title, glow=False):
        for k, (col, lbl) in enumerate(zip(colors, labels)):
            ax.plot(t_np, data[:, k], lw=1.5, color=col, label=lbl)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8, ncol=len(labels), loc='upper right')
        ax.grid(True)
        ax.set_title(title, loc='left', fontweight='bold', pad=16, fontsize=15)
        if glow:
            mplcyberpunk.make_lines_glow(ax)

    # ── srt ──────────────────────────────────────────────────────────────
    if cm == "srt":
        srt    = acts["srt"]
        speeds = _rotor_speeds(srt)

        _multiline(_ax(0, 0, rowspan=2),  speeds,
                   THRUST_COLORS, [f'Ω{k+1}' for k in range(4)],
                   'Ω [rad/s]', 'Rotor Speeds')

        # row 1 col 0 — intentionally blank
        ax_blank = _ax(1, 0)
        ax_blank.set_axis_off()

        _multiline(_ax(0, 1, rowspan=2, colspan=2), srt,
                   THRUST_COLORS, [f'u{k+1}' for k in range(4)],
                   'Thrust [N]', 'Motor Thrusts', glow=False)

    # ── ctbr ─────────────────────────────────────────────────────────────
    elif cm == "ctbr":
        srt    = acts["srt"]
        wrench = acts["wrench"]
        speeds = _rotor_speeds(srt)

        _multiline(_ax(0, 0), srt,
                   THRUST_COLORS, [f'u{k+1}' for k in range(4)],
                   'Thrust [N]', 'Motor Thrusts')

        _multiline(_ax(1, 0), speeds,
                   THRUST_COLORS, [f'Ω{k+1}' for k in range(4)],
                   'Ω [rad/s]', 'Rotor Speeds')

        # Fz alone (dominates the scale)
        _multiline(_ax(0, 1, colspan=2), wrench[:, 0:1],
                   [WRENCH_COLORS[0]], [WRENCH_LABELS[0]],
                   'Force [N]', 'Collective Thrust  Fz')

        # Mx, My, Mz on their own scale
        _multiline(_ax(1, 1, colspan=2), wrench[:, 1:4],
                   WRENCH_COLORS[1:], WRENCH_LABELS[1:],
                   'Torque [N·m]', 'Body Torques  [Mx, My, Mz]')

    # ── lvyr / lvyr+g ────────────────────────────────────────────────────
    elif cm in ("lvyr", "lvyr+g"):
        srt    = acts["srt"]
        wrench = acts["wrench"]
        lvyr   = acts["lvyr"]
        speeds = _rotor_speeds(srt)


        _multiline(_ax(0, 0), srt,
                   THRUST_COLORS, [f'u{k+1}' for k in range(4)],
                   'Thrust [N]', 'Motor Thrusts')
        
        _multiline(_ax(1, 0), speeds,
            THRUST_COLORS, [f'Ω{k+1}' for k in range(4)],
            'Ω [rad/s]', 'Rotor Speeds')

        if cm == "lvyr":
            _multiline(_ax(0, 1, rowspan=2, colspan=2), lvyr,
                       LVYR_COLORS, LVYR_LABELS,
                       'Command', 'LVYR Commands', glow=False)

        else:   # lvyr+g — split rows 0 and 1 in cols 1-2
            gains = acts["gains"]

            _multiline(_ax(0, 1, colspan=2), lvyr,
                       LVYR_COLORS, LVYR_LABELS,
                       'Command', 'LVYR Commands', glow=False)

            _multiline(_ax(1, 1, colspan=2), gains,
                       GAIN_COLORS, GAIN_LABELS,
                       'Gain', 'Adaptive Gains', glow=False)


# ──────────────────────────────────────────────────────────────────────────────
# Motor diagram
# ──────────────────────────────────────────────────────────────────────────────
def draw_motor_diagram(ax, L=0.15, beta_deg=45):
    """Draw the quadrotor motor layout onto an existing Axes object."""
    arm_color    = '#7f8c8d'
    cw_color     = "#FFFFFF"
    ccw_color    = "#FFFFFF"
    axis_color   = "#FFFFFF"
    motor_radius = 0.025
    beta         = np.radians(beta_deg)

    range_limit = L + 0.08
    ax.set_xlim(-range_limit, range_limit)
    ax.set_ylim(-range_limit, range_limit)
    ax.set_aspect('equal')
    ax.set_axis_off()

    motors = {
        '1': {'pos': np.array([ L*np.cos(beta), -L*np.sin(beta)]), 'dir': 'CW',  'color': 'tab:blue'},
        '2': {'pos': np.array([-L*np.cos(beta),  L*np.sin(beta)]), 'dir': 'CW',  'color': 'tab:orange'},
        '3': {'pos': np.array([ L*np.cos(beta),  L*np.sin(beta)]), 'dir': 'CCW', 'color': 'tab:green'},
        '4': {'pos': np.array([-L*np.cos(beta), -L*np.sin(beta)]), 'dir': 'CCW', 'color': 'tab:red'},
    }

    axis_len = L + 0.03
    ax.plot([-axis_len, axis_len], [0, 0], color=axis_color, lw=1.5, zorder=1)
    ax.annotate('', xy=(axis_len, 0), xytext=(axis_len - 0.02, 0),
                arrowprops=dict(arrowstyle='->', color=axis_color, lw=1.5))
    ax.text(axis_len + 0.03, 0, 'b1', fontsize=9, fontweight='bold', va='center')

    ax.plot([0, 0], [-axis_len, axis_len], color=axis_color, lw=1.5, zorder=1)
    ax.annotate('', xy=(0, axis_len), xytext=(0, axis_len - 0.02),
                arrowprops=dict(arrowstyle='->', color=axis_color, lw=1.5))
    ax.text(0, axis_len + 0.03, 'b2', fontsize=9, fontweight='bold', ha='center')

    ax.plot(0, 0, 'k+', markersize=12, mew=2, zorder=5)

    ax.plot([motors['1']['pos'][0], motors['2']['pos'][0]],
            [motors['1']['pos'][1], motors['2']['pos'][1]],
            color=arm_color, lw=3, zorder=2)
    ax.plot([motors['3']['pos'][0], motors['4']['pos'][0]],
            [motors['3']['pos'][1], motors['4']['pos'][1]],
            color=arm_color, lw=3, zorder=2)

    for m_id, data in motors.items():
        x, y    = data['pos']
        m_dir   = data['dir']
        m_color = cw_color if m_dir == 'CW' else ccw_color

        circle = patches.Circle((x, y), motor_radius,
                                 edgecolor='k', facecolor=data['color'], lw=1.5, zorder=5)
        ax.add_patch(circle)
        ax.text(x, y, m_id, fontsize=11, color='white',
                va='center', ha='center', fontweight='bold', zorder=6)

        arc_r = motor_radius + 0.015
        if m_dir == 'CW':
            arc = patches.Arc((x, y), arc_r*2, arc_r*2,
                               theta1=45, theta2=225, color=m_color, lw=1.8, zorder=4)
            ax.add_patch(arc)
            ax.annotate('', xy=(x + arc_r * np.cos(np.radians(-125)),
                                  y + arc_r * np.sin(np.radians(-125))),
                        xytext=(x + arc_r * np.cos(np.radians(-130)),
                                y + arc_r * np.sin(np.radians(-130))),
                        arrowprops=dict(arrowstyle='->', color=m_color, lw=1.5))
        else:
            arc = patches.Arc((x, y), arc_r*2, arc_r*2,
                               theta1=45, theta2=225, color=m_color, lw=1.8, zorder=4)
            ax.add_patch(arc)
            ax.annotate('', xy=(x + arc_r * np.cos(np.radians(45)),
                                  y + arc_r * np.sin(np.radians(45))),
                        xytext=(x + arc_r * np.cos(np.radians(50)),
                                y + arc_r * np.sin(np.radians(50))),
                        arrowprops=dict(arrowstyle='->', color=m_color, lw=1.5))

    arc_b1 = patches.Arc((0, 0), 0.12, 0.12,
                          theta1=0, theta2=beta_deg, color='white', lw=1.5)
    ax.add_patch(arc_b1)
    ax.text(0.07 * np.cos(np.radians(beta_deg / 2)),
            0.07 * np.sin(np.radians(beta_deg / 2)),
            r'$\beta$', color='white', fontsize=11)

    arc_b2 = patches.Arc((0, 0), 0.12, 0.12,
                          theta1=180 - beta_deg, theta2=180, color='white', lw=1.5)
    ax.add_patch(arc_b2)
    ax.text(0.07 * np.cos(np.radians(180 - beta_deg / 2)),
            0.07 * np.sin(np.radians(180 - beta_deg / 2)),
            r'$\beta$', color='white', fontsize=11, ha='right')

    ax.set_title('Motor Layout', fontsize=15, fontweight='bold', pad=6, loc='left')


# ──────────────────────────────────────────────────────────────────────────────
# Main plotting function
# ──────────────────────────────────────────────────────────────────────────────
def plot_rollout(
    traj_np:    np.ndarray,
    dt:         float,   
    label:      str   = "",
    arm_length: float = 0.15,
    arm_angle:  float = 45.0,
    mass:       float = 1.21,
    save_path:  str   = None,
):
    """
    Plot a single policy rollout.

    traj_np column layout
    ─────────────────────
    0:3    state position   (x, y, z)
    3:6    state velocity   (vx, vy, vz)
    6:10   state quaternion
    10:13  state omega
    13:…   actions          (layout depends on cm — see _unpack_actions)
    -9:-6  p_ref            (x, y, z)
    -6:-3  v_ref            (vx, vy, vz)
    -3:    a_ref            (ax, ay, az)
    """
    steps = traj_np.shape[0]
    t_np  = np.arange(steps) * dt

    cm = _infer_cm(traj_np)   # control mode: "srt" | "ctbr" | "lvyr" | "lvyr+g"

    # ── Unpack ───────────────────────────────────────────────────────────
    state_pos = traj_np[:, 0:3]
    state_vel = traj_np[:, 3:6]
    p_ref = traj_np[:, 13:16]
    v_ref = traj_np[:, 16:19]
    a_ref = traj_np[:, 19:22]

    acts = _unpack_actions(traj_np, cm)

    # ── Errors ───────────────────────────────────────────────────────────
    pos_err_np = np.linalg.norm(p_ref - state_pos, axis=1)
    vel_err_np = np.linalg.norm(v_ref - state_vel, axis=1)
    acc_err_np = np.ones(steps)   # placeholder until state acc is stored
    pos_rmse = np.sqrt(np.mean(pos_err_np**2))
    max_speed = np.max(np.linalg.norm(state_vel, axis=1))

    # ── Figure / GridSpec ────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(
        f"POLICY: {label} {' '*5}|{' '*5}RMSE = {pos_rmse:.3f} m {' '*5}|{' '*5}$v_{{max}}$ = {max_speed:.3f} m/s",
        fontsize=20,
        fontweight='bold',
        y=0.94,
    )

    gs = fig.add_gridspec(
        6, 4,
        height_ratios=[0.8, 0.8, 0.5, 0.5, 0.5, 0.4],
        hspace=0.55,
        wspace=0.30,
    )

    # ── Rows 0-1, cols 0-2 : control-mode-specific panels ────────────────
    _plot_top_panel(fig, gs, t_np, acts, cm)

    # ── Rows 2-4, cols 0-2 : p / v / a reference vs state ───────────────
    ref_arrays      = [p_ref,     v_ref,     a_ref]
    state_arrays    = [state_pos, state_vel, None]   # None → ref-only
    ref_titles      = ['Position', 'Velocity', 'Acceleration']
    ylabel_prefixes = ['', 'v', 'a']
    ylabels         = [r'$_x$', r'$_y$', r'$_z$']

    for j, (ref_arr, state_arr, title) in enumerate(
            zip(ref_arrays, state_arrays, ref_titles)):
        for i in range(3):
            ax = fig.add_subplot(gs[i + 2, j])
            ax.plot(t_np, ref_arr[:, i], lw=1.0, color='tab:blue',
                    linestyle='--', label='ref')
            if state_arr is not None:
                ax.plot(t_np, state_arr[:, i], lw=1.75, color='tab:blue',
                        linestyle='-', label='state')
                mplcyberpunk.make_lines_glow(ax)
            ax.set_ylabel(f'{ylabel_prefixes[j]}{ylabels[i]}', fontsize=15)
            ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
            ax.grid(True)
            if i == 0:
                ax.set_title(title, loc='left', fontweight='bold', pad=16)

    # ── Row 5, cols 0-2 : tracking errors ────────────────────────────────
    for j, (err, lbl) in enumerate([
        (pos_err_np, r'$|e_p|$'),
        (vel_err_np, r'$|e_v|$'),
        (acc_err_np, r'$|e_a|$'),
    ]):
        ax = fig.add_subplot(gs[5, j])
        ax.plot(t_np, err, lw=2, color='tab:red')
        ax.set_ylabel(lbl, fontsize=15)
        ax.set_xlabel('Time [s]')
        ax.grid(True)
        mplcyberpunk.add_glow_effects(ax)

    # ── Col 3, rows 0-1 : Adaptation Module placeholder ──────────────────
    ax_adapt = fig.add_subplot(gs[0:2, 3])
    ax_adapt.set_axis_off()
    ax_adapt.text(0.5, 0.5, 'Placeholder\n(Adaptation Module)',
                  ha='center', va='center', fontsize=11,
                  color='grey', style='italic', transform=ax_adapt.transAxes)
    ax_adapt.set_title('Adaptation Module', loc='left', fontweight='bold', fontsize=15)

    # ── Col 3, rows 2-3 : Motor diagram ──────────────────────────────────
    ax_motor = fig.add_subplot(gs[2:4, 3])
    draw_motor_diagram(ax_motor, L=arm_length, beta_deg=arm_angle)
    pos = ax_motor.get_position()
    ax_motor.set_position([pos.x0, pos.y0, pos.width, pos.height])
    ax_motor.set_anchor('W')   # 'W' = west = left

    # ── Col 3, rows 4-5 : System parameters ──────────────────────────────
    ax_info = fig.add_subplot(gs[4:6, 3])
    ax_info.set_axis_off()
    ax_info.text(
        0.05, 0.97,
        f" m   = {mass:.3f} kg\n"
        f" L   = {arm_length:.3f} m,  β = {arm_angle:.1f}°\n"
        f" kf  = 4.5e-8 N/(rad/s)²\n"
        f" cm  = {cm}\n",
        transform=ax_info.transAxes,
        fontsize=15, fontfamily='monospace',
        va='top', ha='left', color="#ffffff", linespacing=1.6,
    )

    # ── Finalise ─────────────────────────────────────────────────────────
    # Use explicit spacing instead of tight_layout: this figure mixes axes
    # types/effects that can trigger tight_layout compatibility warnings.
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.08, top=0.85,
                        hspace=-0.2, wspace=0.0)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f"  Saved plot → {save_path}")
    else:
        plt.show()

    plt.close(fig)
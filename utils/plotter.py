import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


# ──────────────────────────────────────────────
# Motor diagram helper
# ──────────────────────────────────────────────
def draw_motor_diagram(ax, L=0.15, beta_deg=45):
    """Draw the quadrotor motor layout onto an existing Axes object."""
    arm_color    = '#7f8c8d'
    # cw_color     = '#2ecc71'
    # ccw_color    = '#e74c3c'
    cw_color     = '#000000'
    ccw_color    = '#000000'
    motor_radius = 0.025
    beta         = np.radians(beta_deg)

    range_limit = L + 0.08
    ax.set_xlim(-range_limit, range_limit)
    ax.set_ylim(-range_limit, range_limit)
    ax.set_aspect('equal')
    ax.set_axis_off()

    motors = {
        '1': {'pos': np.array([ L*np.cos(beta), -L*np.sin(beta)]), 'dir': 'CW', 'color': 'tab:blue'},
        '2': {'pos': np.array([-L*np.cos(beta),  L*np.sin(beta)]), 'dir': 'CW', 'color': 'tab:orange'},
        '3': {'pos': np.array([ L*np.cos(beta),  L*np.sin(beta)]), 'dir': 'CCW', 'color': 'tab:green'},
        '4': {'pos': np.array([-L*np.cos(beta), -L*np.sin(beta)]), 'dir': 'CCW', 'color': 'tab:red'},
    }

    axis_len = L + 0.03
    ax.plot([-axis_len, axis_len], [0, 0], color='black', lw=1.5, zorder=1)
    ax.annotate('', xy=(axis_len, 0), xytext=(axis_len - 0.02, 0),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax.text(axis_len + 0.03, 0, 'b1', fontsize=9, fontweight='bold', va='center')

    ax.plot([0, 0], [-axis_len, axis_len], color='black', lw=1.5, zorder=1)
    ax.annotate('', xy=(0, axis_len), xytext=(0, axis_len - 0.02),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax.text(0, axis_len + 0.03, 'b2', fontsize=9, fontweight='bold', ha='center')

    ax.plot(0, 0, 'k+', markersize=12, mew=2, zorder=5)

    ax.plot([motors['1']['pos'][0], motors['2']['pos'][0]],
            [motors['1']['pos'][1], motors['2']['pos'][1]],
            color=arm_color, lw=3, zorder=2)
    ax.plot([motors['3']['pos'][0], motors['4']['pos'][0]],
            [motors['3']['pos'][1], motors['4']['pos'][1]],
            color=arm_color, lw=3, zorder=2)

    for m_id, data in motors.items():
        x, y   = data['pos']
        m_dir  = data['dir']
        m_color = cw_color if m_dir == 'CW' else ccw_color

        circle = patches.Circle((x, y), motor_radius,
                                 edgecolor='k', facecolor='white', lw=1.5, zorder=5)
        ax.add_patch(circle)
        ax.text(x, y, m_id, fontsize=11, color=data['color'],
                va='center', ha='center', fontweight='bold', zorder=6)

        arc_r = motor_radius + 0.015
        if m_dir == 'CW':
            arc = patches.Arc((x, y), arc_r*2, arc_r*2,
                               theta1=-225, theta2=-45, color=m_color, lw=1.8, zorder=4)
            ax.add_patch(arc)
            ax.annotate('', xy=(x + arc_r * np.cos(np.radians(-45)),
                                  y + arc_r * np.sin(np.radians(-45))),
                        xytext=(x + arc_r * np.cos(np.radians(-50)),
                                y + arc_r * np.sin(np.radians(-50))),
                        arrowprops=dict(arrowstyle='->', color=m_color, lw=1.5))
            ax.text(x, y - (arc_r + 0.018), 'CW', color=m_color,
                    fontsize=8, fontweight='bold', va='center', ha='center')
        else:
            arc = patches.Arc((x, y), arc_r*2, arc_r*2,
                               theta1=45, theta2=225, color=m_color, lw=1.8, zorder=4)
            ax.add_patch(arc)
            ax.annotate('', xy=(x + arc_r * np.cos(np.radians(45)),
                                  y + arc_r * np.sin(np.radians(45))),
                        xytext=(x + arc_r * np.cos(np.radians(50)),
                                y + arc_r * np.sin(np.radians(50))),
                        arrowprops=dict(arrowstyle='->', color=m_color, lw=1.5))
            ax.text(x, y + (arc_r + 0.018), 'CCW', color=m_color,
                    fontsize=8, fontweight='bold', va='center', ha='center')

    arc_b1 = patches.Arc((0, 0), 0.12, 0.12,
                          theta1=0, theta2=beta_deg, color='blue', lw=1.5)
    ax.add_patch(arc_b1)
    ax.text(0.07 * np.cos(np.radians(beta_deg/2)),
            0.07 * np.sin(np.radians(beta_deg/2)),
            r'$\beta$', color='blue', fontsize=11)

    arc_b2 = patches.Arc((0, 0), 0.12, 0.12,
                          theta1=180 - beta_deg, theta2=180, color='blue', lw=1.5)
    ax.add_patch(arc_b2)
    ax.text(0.07 * np.cos(np.radians(180 - beta_deg/2)),
            0.07 * np.sin(np.radians(180 - beta_deg/2)),
            r'$\beta$', color='blue', fontsize=11, ha='right')

    ax.set_title('Motor Layout', fontsize=12, fontweight='bold', pad=6, loc='left')

# ──────────────────────────────────────────────
# Main plotting function
# ──────────────────────────────────────────────
def plot_rollout(
    traj_np:  np.ndarray,   # (steps, 26): states[0:13] | actions[13:17] | p_ref[17:20] | v_ref[20:23] | a_ref[23:26]
    dt:       float,
    label:    str   = "",
    arm_length: float = 0.15,
    arm_angle:  float = 45.0,
    mass:       float = 1.21,
    save_path:  str   = None,
):
    """
    Plot a single policy rollout.

    traj_np columns
    ---------------
    0:3   state position  (x, y, z)
    3:6   state velocity  (vx, vy, vz)
    6:10  state quaternion
    10:13 state omega
    13:17 actions         (u1..u4)
    17:20 p_ref           (x, y, z)
    20:23 v_ref           (vx, vy, vz)
    23:26 a_ref           (ax, ay, az)
    """
    steps = traj_np.shape[0]
    t_np  = np.arange(steps) * dt

    # ── Unpack columns ───────────────────────────────
    state_pos  = traj_np[:, 0:3]    # (steps, 3)
    state_vel  = traj_np[:, 3:6]
    actions_np = traj_np[:, 13:17]  # (steps, 4)  → transpose to (4, steps) for row-wise plot
    p_ref      = traj_np[:, 17:20]
    v_ref      = traj_np[:, 20:23]
    a_ref      = traj_np[:, 23:26]

    # ── Errors ───────────────────────────────────────
    pos_err_np = np.linalg.norm(p_ref - state_pos, axis=1)
    vel_err_np = np.linalg.norm(v_ref - state_vel, axis=1)
    # Acceleration error kept as dummy (state acc not stored); replace later
    acc_err_np = np.ones(steps)

    # ── Figure ───────────────────────────────────────
    height_ratios = [0.8, 0.8, 0.5, 0.5, 0.5, 1.0]
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(f'{label.upper()}', fontsize=20, fontweight='bold', y=0.96)

    gs = fig.add_gridspec(
        6, 4,
        height_ratios=height_ratios,
        hspace=0.45,
        wspace=0.35,
    )

    # ── Motor thrusts (row 0, col 0) — all 4 on one plot ─────────────────
    ax_thrust = fig.add_subplot(gs[0, 0])
    thrust_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    for k in range(4):
        ax_thrust.plot(t_np, actions_np[:, k], lw=1.5,
                       color=thrust_colors[k], label=f'u{k+1}')
    ax_thrust.set_ylabel('Thrust')
    ax_thrust.legend(fontsize=8, ncol=4, loc='upper right')
    ax_thrust.grid(True)
    ax_thrust.set_title('Motor Thrusts', loc='left', fontweight='bold', pad=16)
 
    # ── Rotor speeds (row 1, col 0) — Ω = sqrt(T / kf) ──────────────────
    KF = 0.000000045  # N/(rad/s)²   #TODO: Hard cided for now....
    rotor_speeds = np.sqrt(np.maximum(actions_np, 0.0) / KF)  # (steps, 4)
 
    ax_omega = fig.add_subplot(gs[1, 0])
    for k in range(4):
        ax_omega.plot(t_np, rotor_speeds[:, k], lw=1.5,
                      color=thrust_colors[k], label=f'Ω{k+1}')
    ax_omega.set_ylabel('Ω [rad/s]')
    ax_omega.legend(fontsize=8, ncol=4, loc='upper right')
    ax_omega.grid(True)
    ax_omega.set_title('Rotor Speeds', loc='left', fontweight='bold', pad=16)
 
    # ── Placeholders (rows 0-1, cols 1-2) ────────────────────────────────
    placeholder_cells = [(0, 1), (0, 2), (1, 1), (1, 2)]
    for (i, j) in placeholder_cells:
        ax = fig.add_subplot(gs[i, j])
        ax.set_facecolor('#f7f7f7')
        ax.text(0.5, 0.5, 'Placeholder', ha='center', va='center',
                fontsize=10, color='grey', style='italic',
                transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor('#cccccc')

   # ── p_ref / v_ref / a_ref vs state (rows 2-4, cols 0-2) ────────────────
    # dotted = reference, solid = state  (acc is reference-only until state acc is stored)
    ref_arrays   = [p_ref,     v_ref,     a_ref]
    state_arrays = [state_pos, state_vel, None ]   # None → ref-only
    ref_titles   = ['Position', 'Velocity', 'Acceleration']
    ylabels      = ['X', 'Y', 'Z']
 
    for j, (ref_arr, state_arr, title) in enumerate(zip(ref_arrays, state_arrays, ref_titles)):
        for i in range(3):
            ax = fig.add_subplot(gs[i + 2, j])
 
            ax.plot(t_np, ref_arr[:, i], lw=1.5, color='tab:blue',
                    linestyle='--', label='ref')
 
            if state_arr is not None:
                ax.plot(t_np, state_arr[:, i], lw=1.5, color='tab:blue',
                        linestyle='-', label='state')
 
            ax.set_ylabel(ylabels[i])
            ax.grid(True)
 
            if i == 0:
                ax.set_title(title, fontsize=9)
                ax.legend(fontsize=7, loc='upper right')

    # ── Error plots (row 5, cols 0-2) ────────────────
    for j, (err, lbl) in enumerate([
        (pos_err_np, '|e_p|'),
        (vel_err_np, '|e_v|'),
        (acc_err_np, '|e_a|'),
    ]):
        ax = fig.add_subplot(gs[5, j])
        ax.plot(t_np, err, lw=2, color='tab:orange')
        ax.set_ylabel(lbl)
        ax.set_xlabel('Time [s]')
        ax.grid(True)

    # ── Motor diagram (col 3, rows 0-1) ──────────────
    ax_motor = fig.add_subplot(gs[0:2, 3])
    draw_motor_diagram(ax_motor, L=arm_length, beta_deg=arm_angle)

    # ── System parameters (col 3, rows 2-4) ──────────
    ax_info = fig.add_subplot(gs[2:5, 3])
    ax_info.set_facecolor('#f7f7f7')
    ax_info.set_axis_off()
    params_text = (
        f" m  = {mass:.3f} kg\n"
        f" L   = {arm_length:.3f} m,  β = {arm_angle:.1f}°\n"
        f" kf  = {KF:.3f} N/(rad/s)²\n"
    )
    ax_info.text(
        0.05, 0.97, params_text,
        transform=ax_info.transAxes,
        fontsize=12, fontfamily='monospace',
        va='top', ha='left', color='#2c3e50', linespacing=1.5,
    )

    # ── Status / Logging (col 3, row 5) ──────────────
    ax_log = fig.add_subplot(gs[5, 3])
    ax_log.set_facecolor('#f7f7f7')
    ax_log.text(0.5, 0.5, 'Placeholder\n(Status / Logging)',
                ha='center', va='center', fontsize=11,
                color='grey', style='italic', transform=ax_log.transAxes)
    ax_log.set_title('Adaptation Module', loc='left', fontweight='bold')
    ax_log.set_xticks([])
    ax_log.set_yticks([])
    for spine in ax_log.spines.values():
        spine.set_edgecolor('#cccccc')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved plot → {save_path}")
    else:
        plt.show()

    plt.close(fig)
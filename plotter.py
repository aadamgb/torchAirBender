import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# ──────────────────────────────────────────────
# Data generation (unchanged)
# ──────────────────────────────────────────────
duration = 10.0
dt = 0.01
num_samples = int(duration / dt) + 1
t = torch.linspace(0.0, duration, num_samples)

pitch = 2.0
r = 1.0

p_ref = torch.stack([
    pitch * t,
    r * torch.cos(torch.pi * t),
    r * torch.sin(torch.pi * t),
], dim=0)

v_ref = torch.stack([
    pitch * torch.ones_like(t),
    -r * torch.sin(torch.pi * t),
    r * torch.cos(torch.pi * t),
], dim=0)

a_ref = torch.stack([
    torch.zeros_like(t),
    -r * torch.pi * torch.cos(torch.pi * t),
    -r * torch.pi * torch.sin(torch.pi * t),
], dim=0)

states  = torch.ones((13, num_samples), dtype=t.dtype, device=t.device)
actions = torch.ones((4,  num_samples), dtype=t.dtype, device=t.device)
traj    = torch.cat((states, actions, p_ref, v_ref, a_ref), dim=0)

state_pos = traj[0:3, :]
state_vel = traj[3:6, :]
state_acc = torch.ones_like(a_ref)

pos_err = torch.linalg.norm(p_ref - state_pos, dim=0)
vel_err = torch.linalg.norm(v_ref - state_vel, dim=0)
acc_err = torch.linalg.norm(a_ref - state_acc, dim=0)

t_np        = t.cpu().numpy()
traj_np     = traj.cpu().numpy()
actions_np  = actions.cpu().numpy()
pos_err_np  = pos_err.cpu().numpy()
vel_err_np  = vel_err.cpu().numpy()
acc_err_np  = acc_err.cpu().numpy()


# ──────────────────────────────────────────────
# Motor diagram helper
# ──────────────────────────────────────────────
def draw_motor_diagram(ax, L=0.15, beta_deg=45):
    """Draw the quadrotor motor layout onto an existing Axes object."""
    arm_color    = '#7f8c8d'
    cw_color     = '#2ecc71'
    ccw_color    = '#e74c3c'
    motor_radius = 0.025
    beta         = np.radians(beta_deg)

    range_limit = L + 0.08
    ax.set_xlim(-range_limit, range_limit)
    ax.set_ylim(-range_limit, range_limit)
    ax.set_aspect('equal')
    ax.set_axis_off()

    motors = {
        '1': {'pos': np.array([ L*np.cos(beta), -L*np.sin(beta)]), 'dir': 'CW'},
        '2': {'pos': np.array([-L*np.cos(beta),  L*np.sin(beta)]), 'dir': 'CW'},
        '3': {'pos': np.array([ L*np.cos(beta),  L*np.sin(beta)]), 'dir': 'CCW'},
        '4': {'pos': np.array([-L*np.cos(beta), -L*np.sin(beta)]), 'dir': 'CCW'},
    }

    axis_len = L + 0.03
    # b1 axis (horizontal)
    ax.plot([-axis_len, axis_len], [0, 0], color='black', lw=1.5, zorder=1)
    ax.annotate('', xy=(axis_len, 0), xytext=(axis_len - 0.02, 0),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax.text(axis_len + 0.03, 0, 'b1', fontsize=9, fontweight='bold', va='center')

    # b2 axis (vertical)
    ax.plot([0, 0], [-axis_len, axis_len], color='black', lw=1.5, zorder=1)
    ax.annotate('', xy=(0, axis_len), xytext=(0, axis_len - 0.02),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax.text(0, axis_len + 0.03, 'b2', fontsize=9, fontweight='bold', ha='center')

    ax.plot(0, 0, 'k+', markersize=12, mew=2, zorder=5)

    # Arms
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
        ax.text(x, y, m_id, fontsize=11, color='k',
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

    # Beta arcs
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
# Build the figure with GridSpec (6 rows × 4 cols)
# ──────────────────────────────────────────────
height_ratios = [0.8, 0.8, 0.5, 0.5, 0.5, 1.0]

fig = plt.figure(figsize=(20, 12))
gs  = fig.add_gridspec(
    6, 4,
    height_ratios=height_ratios,
    hspace=0.45,
    wspace=0.35,
)

# ── Columns 0-2: original plots ──────────────────

# Control Info (rows 0-1, all 3 original cols)
action_plot_indices = [0, 1, 2, 3, 0, 1]
for i in range(2):
    for j in range(3):
        idx = action_plot_indices[i * 3 + j]
        ax = fig.add_subplot(gs[i, j])
        ax.plot(t_np, actions_np[idx, :], lw=2, color='tab:purple')
        ax.set_ylabel(f'u{idx + 1}')
        ax.grid(True)
        # if i == 1:
            # ax.set_xlabel('Time [s]')
        if i == 0 and j == 0:
            ax.set_title('Control Info', loc='left', fontweight='bold', pad=16)

# p_ref / v_ref / a_ref (rows 2-4)
ref_data  = [(17, 'p_ref Components'), (20, 'v_ref Components'), (23, 'a_ref Components')]
colors    = ['tab:red', 'tab:green', 'tab:blue']
ylabels   = ['X', 'Y', 'Z']
for j, (start, title) in enumerate(ref_data):
    for i in range(3):
        ax = fig.add_subplot(gs[i + 2, j])
        ax.plot(t_np, traj_np[start + i, :], lw=2, color=colors[i])
        ax.set_ylabel(ylabels[i])
        ax.grid(True)
        # if i == 0:
        #     ax.set_title(title)

# Error plots (row 5)
for j, (err, lbl, title) in enumerate([
    (pos_err_np, '|e_p|', 'Position Error'),
    (vel_err_np, '|e_v|', 'Velocity Error'),
    (acc_err_np, '|e_a|', 'Acceleration Error'),
]):
    ax = fig.add_subplot(gs[5, j])
    ax.plot(t_np, err, lw=2, color='tab:orange')
    ax.set_ylabel(lbl)
    ax.set_xlabel('Time [s]')
    # ax.set_title(title)
    ax.grid(True)

# ── Column 3: new panels ─────────────────────────

# [A] Motor diagram  — rows 0-1
ax_motor = fig.add_subplot(gs[0:2, 3])
draw_motor_diagram(ax_motor, L=0.15, beta_deg=45)

# [B] Placeholder (3D trajectory / future use) — rows 2-4
ax_info = fig.add_subplot(gs[2:5, 3])
ax_info.set_facecolor('#f7f7f7')
ax_info.set_axis_off()
# ax_info.set_title('System Parameters', loc='left', fontweight='bold')

m = 1.21
params_text = (
    f" m  = {m} kg, J = [a, b, c]\n"
     " L   = 0.20 m, β   = 45.0°\n"
    # "  Motor\n"
    # "    kf  = 1.0e-5  N/(rad/s)²\n"
    # "    km  = 1.0e-6  Nm/(rad/s)²\n\n"
    # "  Drag\n"
    # "    cd  = 0.10\n"
    # "    cq  = 0.02\n"
)

ax_info.text(
    0.05, 0.97, params_text,
    transform=ax_info.transAxes,
    fontsize=12,
    fontfamily='monospace',   # keeps columns aligned
    va='top', ha='left',
    color='#2c3e50',
    linespacing=1.5,
)

# [C] Status / Logging — row 5
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

# plt.savefig('/mnt/user-data/outputs/dashboard.png', dpi=150, bbox_inches='tight')
plt.tight_layout()
plt.show()
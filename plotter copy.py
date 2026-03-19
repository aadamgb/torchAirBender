import torch
import matplotlib.pyplot as plt

duration = 5.0
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

states = torch.ones((13, num_samples), dtype=t.dtype, device=t.device)
actions = torch.ones((4, num_samples), dtype=t.dtype, device=t.device)
traj = torch.cat((states, actions, p_ref, v_ref, a_ref), dim=0)

state_pos = traj[0:3, :]
state_vel = traj[3:6, :]
# States do not currently include acceleration channels.
state_acc = torch.ones_like(a_ref)

pos_err = torch.linalg.norm(p_ref - state_pos, dim=0)
vel_err = torch.linalg.norm(v_ref - state_vel, dim=0)
acc_err = torch.linalg.norm(a_ref - state_acc, dim=0)

t_np = t.cpu().numpy()
traj_np = traj.cpu().numpy()
actions_np = actions.cpu().numpy()
pos_err_np = pos_err.cpu().numpy()
vel_err_np = vel_err.cpu().numpy()
acc_err_np = acc_err.cpu().numpy()

fig, axes = plt.subplots(
    6,
    3,
    figsize=(18, 12),
    sharex=True,
    gridspec_kw={"height_ratios": [0.8, 0.8, 0.25, 0.25, 0.25, 1.4]},
)

action_plot_indices = [0, 1, 2, 3, 0, 1]
for i in range(2):
    for j in range(3):
        idx = action_plot_indices[i * 3 + j]
        axes[i, j].plot(t_np, actions_np[idx, :], lw=2, color='tab:purple')
        axes[i, j].set_ylabel(f'u{idx + 1}')
        axes[i, j].grid(True)
        if i == 1:
            axes[i, j].set_xlabel('Time [s]')

axes[0, 0].set_title('Control Info', loc='left', fontweight='bold', pad=16)

axes[2, 0].plot(t_np, traj_np[17, :], lw=2, color='tab:red')
axes[2, 0].set_ylabel('X')
axes[2, 0].set_title('p_ref Components')
axes[2, 0].grid(True)

axes[3, 0].plot(t_np, traj_np[18, :], lw=2, color='tab:green')
axes[3, 0].set_ylabel('Y')
axes[3, 0].grid(True)

axes[4, 0].plot(t_np, traj_np[19, :], lw=2, color='tab:blue')
axes[4, 0].set_ylabel('Z')
axes[4, 0].grid(True)

axes[2, 1].plot(t_np, traj_np[20, :], lw=2, color='tab:red')
axes[2, 1].set_ylabel('X')
axes[2, 1].set_title('v_ref Components')
axes[2, 1].grid(True)

axes[3, 1].plot(t_np, traj_np[21, :], lw=2, color='tab:green')
axes[3, 1].set_ylabel('Y')
axes[3, 1].grid(True)

axes[4, 1].plot(t_np, traj_np[22, :], lw=2, color='tab:blue')
axes[4, 1].set_ylabel('Z')
axes[4, 1].grid(True)

axes[2, 2].plot(t_np, traj_np[23, :], lw=2, color='tab:red')
axes[2, 2].set_ylabel('X')
axes[2, 2].set_title('a_ref Components')
axes[2, 2].grid(True)

axes[3, 2].plot(t_np, traj_np[24, :], lw=2, color='tab:green')
axes[3, 2].set_ylabel('Y')
axes[3, 2].grid(True)

axes[4, 2].plot(t_np, traj_np[25, :], lw=2, color='tab:blue')
axes[4, 2].set_ylabel('Z')
axes[4, 2].grid(True)

axes[5, 0].plot(t_np, pos_err_np, lw=2, color='tab:orange')
axes[5, 0].set_ylabel('|e_p|')
axes[5, 0].set_xlabel('Time [s]')
axes[5, 0].set_title('Position Error')
axes[5, 0].grid(True)

axes[5, 1].plot(t_np, vel_err_np, lw=2, color='tab:orange')
axes[5, 1].set_ylabel('|e_v|')
axes[5, 1].set_xlabel('Time [s]')
axes[5, 1].set_title('Velocity Error')
axes[5, 1].grid(True)

axes[5, 2].plot(t_np, acc_err_np, lw=2, color='tab:orange')
axes[5, 2].set_ylabel('|e_a|')
axes[5, 2].set_xlabel('Time [s]')
axes[5, 2].set_title('Acceleration Error')
axes[5, 2].grid(True)

plt.tight_layout()
plt.show()
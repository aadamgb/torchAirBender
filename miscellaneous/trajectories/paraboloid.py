import numpy as np
import matplotlib.pyplot as plt
import os
import csv

def generate_paraboloid_trajectory(z_levels, laps_per_level, dt=0.01):
    """
    z_levels: list of heights [2.0, 3.0, 4.0...]
    laps_per_level: int, number of circles at each height
    """
    t_total = 0
    trajectory = []
    
    # Starting conditions
    current_z = z_levels[0]
    current_theta = 0.0
    
    for i, target_z in enumerate(z_levels):
        # --- 1. CLIMB PHASE (Connect levels) ---
        if i > 0:
            start_z = z_levels[i-1]
            climb_time = 3.0  # Slow climb
            t_steps = int(climb_time / dt)
            
            for _ in range(t_steps):
                t_total += dt
                # Linear interpolation for z and slow rotation
                current_z += (target_z - start_z) / t_steps
                current_theta += (2 * np.pi / t_steps) # 1 slow rotation during climb
                
                # Physics mapping
                r = 2 * np.sqrt(current_z)
                pos = [r * np.cos(current_theta), r * np.sin(current_theta), current_z]
                # Simplified velocity for transition
                trajectory.append([t_total] + pos + [0, 0, 0, 0, 0, 0])

        # --- 2. LAP PHASE (3 laps with increasing speed) ---
        # User defined: let's say base speed increases with height
        base_omega = 1.0 + (i * 0.5) 
        
        for lap in range(laps_per_level):
            # Speed increases per lap: lap 0 (slowest) -> lap 2 (fastest)
            omega = base_omega * (1 + lap * 0.5)
            r = 2 * np.sqrt(target_z)
            
            # Time to complete one lap: T = 2pi / omega
            lap_time = (2 * np.pi) / omega
            t_steps = int(lap_time / dt)
            
            for _ in range(t_steps):
                t_total += dt
                current_theta += omega * dt
                
                # Position
                px = r * np.cos(current_theta)
                py = r * np.sin(current_theta)
                pz = target_z
                
                # Velocity (Derivatives of pos)
                vx = -r * omega * np.sin(current_theta)
                vy =  r * omega * np.cos(current_theta)
                vz = 0.0
                
                # Acceleration (Derivatives of vel)
                ax = -r * (omega**2) * np.cos(current_theta)
                ay = -r * (omega**2) * np.sin(current_theta)
                az = 0.0
                
                trajectory.append([t_total, px, py, pz, vx, vy, vz, ax, ay, az])

    return np.array(trajectory)

# Configuration
z_steps = [2.0, 4.0, 6.0, 8.0]
dt = 0.01
data = generate_paraboloid_trajectory(z_steps, laps_per_level=5, dt=dt)

output_path = "/home/adame/torchAirBender/miscellaneous/trajectories/CAMP/progressive_paraboloid.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Save the full trajectory in CSV format with the exact requested header.
header = "time,px,py,pz,vx,vy,vz,ax,ay,az"
data_to_save = data.copy()
if data_to_save.size > 0:
    # Shift saved timestamps so the first row is exactly 0.0.
    data_to_save[:, 0] -= dt

with open(output_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header.split(","))
    for row in data_to_save:
        writer.writerow([float(v) for v in row])

# --- VISUALIZATION ---
fig = plt.figure(figsize=(10, 8))
ax = fig.add_scalar_bar = fig.add_subplot(111, projection='3d')

# Plot the trajectory
ax.plot(data[:, 1], data[:, 2], data[:, 3], label='Quadrotor Path', lw=1)

# Plot the Paraboloid Surface for context
x_surf = np.linspace(-6, 6, 30)
y_surf = np.linspace(-6, 6, 30)
X, Y = np.meshgrid(x_surf, y_surf)
Z = 0.25 * (X**2 + Y**2)
ax.plot_surface(X, Y, Z, alpha=0.1, color='cyan')

ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('Paraboloid Trajectory Tracking Test')
plt.legend()
plt.show()

# Print first few lines for format check
print("time,px,py,pz,vx,vy,vz,ax,ay,az")
for row in data[:5]:
    print(",".join(map(str, row)))
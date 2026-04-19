import numpy as np
import csv

# Parameters
r = 3.0
z = 2.0
v0 = 2.0       # initial speed [m/s]
vf = 5.0       # final speed [m/s]
T = 100.0       # total time [s]
dt = 0.01

output_path = "/home/adame/torchAirBender/miscellaneous/trajectories/CAMP/progressive_circle.csv"

# Angular quantities
omega0 = v0 / r
omegaf = vf / r
alpha = (omegaf - omega0) / T

# Time vector
t_vec = np.arange(0.0, T + dt, dt)

# Storage
rows = []

for t in t_vec:
    theta = omega0 * t + 0.5 * alpha * t**2
    omega = omega0 + alpha * t

    # Position
    px = r * np.cos(theta)
    py = r * np.sin(theta)
    pz = z

    # Velocity
    vx = -r * omega * np.sin(theta)
    vy =  r * omega * np.cos(theta)
    vz = 0.0

    # Acceleration
    ax = -r * (alpha * np.sin(theta) + omega**2 * np.cos(theta))
    ay =  r * (alpha * np.cos(theta) - omega**2 * np.sin(theta))
    az = 0.0

    rows.append([t, px, py, pz, vx, vy, vz, ax, ay, az])

# Save CSV
with open(output_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["time","px","py","pz","vx","vy","vz","ax","ay","az"])
    writer.writerows(rows)
import numpy as np
import matplotlib.pyplot as plt

# Time parameter
t = np.linspace(0, 2 * np.pi, 2000)

# Frequencies for x, y, z (ratios determine the shape)
a, b, c = 3, 2, 5
# Phase shifts
dx, dy, dz = 0, np.pi/2, np.pi/4

# Parametric equations
x = np.sin(a * t + dx)
y = np.sin(b * t + dy)
z = np.sin(c * t + dz)

# Create 3D Plot
fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize=(10, 8))
ax.plot(x, y, z, color='teal', linewidth=2)

ax.set_title(f'3D Lissajous Curve (a={a}, b={b}, c={c})')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()
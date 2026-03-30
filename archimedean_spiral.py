import numpy as np
import matplotlib.pyplot as plt

# Define the range for t as shown in your image (0 to 20)
# We use 1000 points to ensure the curve looks smooth
t = np.linspace(0, 20, 1000)

# Define the parametric equations
x = t * np.cos(t)
y = t * np.sin(t)

# Create the plot
plt.figure(figsize=(8, 8))
plt.plot(x, y, color='black', linewidth=1.5)

# Add grid lines similar to the Desmos/Graphing tool style
plt.grid(True, which='both', linestyle='--', alpha=0.7)
plt.axhline(0, color='black', linewidth=1) # X-axis
plt.axvline(0, color='black', linewidth=1) # Y-axis

# Set limits and labels to match the image scale
plt.xlim(-25, 25)
plt.ylim(-20, 20)
plt.title(r'Archimedean Spiral: $C(t) = (t \cos(t), t \sin(t))$')
plt.xlabel('x')
plt.ylabel('y')

# Show the plot
plt.show()
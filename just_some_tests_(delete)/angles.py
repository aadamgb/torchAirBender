import numpy as np
import matplotlib.pyplot as plt

def plot_quad(arm_angle):
    l = 1.0  # arm length

    # Motor angles
    angles = [
        arm_angle,
        arm_angle + np.pi/2,
        arm_angle + np.pi,
        arm_angle + 3*np.pi/2
    ]
    
    angles = [
        arm_angle,
        -arm_angle + np.pi,
        arm_angle + np.pi,
        -arm_angle + 2 * np.pi
    ]

    # Motor positions
    x = [l * np.cos(a) for a in angles]
    y = [l * np.sin(a) for a in angles]

    # Plot
    plt.figure()
    plt.axhline(0)
    plt.axvline(0)

    # Draw arms
    for xi, yi in zip(x, y):
        plt.plot([0, xi], [0, yi])

    # Plot motors
    plt.scatter(x, y)

    # Label motors
    labels = ["1 (CCW)", "2 (CW)", "3 (CCW)", "4 (CW)"]
    for xi, yi, label in zip(x, y, labels):
        plt.text(xi*1.1, yi*1.1, label)

    # Body axes
    plt.text(1.2, 0, "b1 →")
    plt.text(0, 1.2, "↑ b2")

    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.title(f"Quad layout | arm_angle = {np.degrees(arm_angle):.2f}°")
    plt.grid(True)
    plt.show()


# Try different angles
plot_quad(np.deg2rad(30))     # +45 deg rotation
plot_quad(np.deg2rad(45))     # 30 deg
plot_quad(np.deg2rad(60))     # 90 deg
import numpy as np
import matplotlib.pyplot as plt
from rosbags.highlevel import AnyReader
from pathlib import Path

# ==========================================
# CONFIGURATION - CHANGE THESE OFTEN
# ==========================================
START_SEC = 31.0
END_SEC   = 33.0
DT_TORCH  = 0.01 

# Indices in your traj_np array
OMEGA_X_INDEX = 10 
REF_INDEX     = -1 # Adjust this index to wherever your omega_x setpoint is

TIME_OFFSET = 29.54 

TRAJ_PATH = "/home/adame/torchAirBender/val/data/airbndr/traj_data.npy"
BAG_PATH  = "/home/adame/torchAirBender/val/data/gazebo/rates.bag"
TOPIC_NAME = '/uav1/estimation_manager/ground_truth/odom'
# ==========================================

def get_comparison_plot():
    # 1. Load Trajectory Data
    traj_data = np.load(TRAJ_PATH)
    omega_x_torch = traj_data[:, OMEGA_X_INDEX]
    omega_x_ref   = traj_data[:, REF_INDEX] # Extracting the reference
    
    # Generate stamps and APPLY OFFSET
    torch_stamps = (np.arange(len(omega_x_torch)) * DT_TORCH) + TIME_OFFSET

    # 2. Extract and Filter ROS Bag Data
    odom_vals = []
    odom_stamps = []

    with AnyReader([Path(BAG_PATH)]) as reader:
        connections = [x for x in reader.connections if x.topic == TOPIC_NAME]
        bag_start_ns = reader.start_time
        
        t_start_ns = bag_start_ns + int(START_SEC * 1e9)
        t_end_ns = bag_start_ns + int(END_SEC * 1e9)

        for connection, timestamp, rawdata in reader.messages(connections=connections):
            if t_start_ns <= timestamp <= t_end_ns:
                msg = reader.deserialize(rawdata, connection.msgtype)
                val = msg.twist.twist.angular.x 
                odom_vals.append(val)
                rel_time = (timestamp - bag_start_ns) / 1e9
                odom_stamps.append(rel_time)

    # 3. Slice Torch Data
    torch_mask = (torch_stamps >= START_SEC) & (torch_stamps <= END_SEC)
    
    # 4. Plotting
    plt.figure(figsize=(12, 6))
    
    # Reference Step (usually plotted as a black dashed or thin line)
    plt.plot(torch_stamps[torch_mask], omega_x_ref[torch_mask], 
             label='Reference', color='black', linestyle=':', alpha=0.8, lw=1.5, zorder=5)
    
    # Torch Data
    plt.plot(torch_stamps[torch_mask], omega_x_torch[torch_mask], 
             label='Ours', color='tab:blue', lw=2)
    
    # Gazebo Data
    plt.plot(odom_stamps, odom_vals, 
             label='Gazebo', color='tab:orange', lw=2)

    plt.title(r"Pitch Rate Step Response")
    plt.xlabel("Time (s)")
    plt.ylabel(r"$\omega_x$ (rad/s)")
    plt.xlim(START_SEC, END_SEC)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    get_comparison_plot()
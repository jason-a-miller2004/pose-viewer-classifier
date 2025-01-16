import numpy as np
import matplotlib.pyplot as plt

# File paths
imu_data_loc = './dataset/imu_data/train_imu_data.npy'
label_data_loc = './dataset/imu_data/train_label_data.npy'

# Label map
label_map = {
    'seated_march': 0,
    'seated_shoulder_abduction': 1,
    'sit_to_stand': 2
}
# Invert the label_map to get names from numerical labels
inv_label_map = {v: k for k, v in label_map.items()}

def main():
    # Load the data
    #  - imu_data is (N,) where each element is a (samples, 6) array
    #  - label_data is (N,) where each element is a single integer label
    imu_data = np.load(imu_data_loc, allow_pickle=True)
    label_data = np.load(label_data_loc, allow_pickle=True)
    
    # Names for each channel
    channels = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']

    # Loop through each IMU sequence
    for i in range(len(imu_data)):
        data = imu_data[i]  # (samples, 6)
        label_idx = label_data[i]  # e.g., 0 or 1 or 2
        label_str = inv_label_map[label_idx]

        # Create subplots: one row per channel
        fig, axs = plt.subplots(
            nrows=6, ncols=1, 
            figsize=(8, 10), 
            sharex=True
        )
        
        # Plot each channel
        for c in range(6):
            axs[c].plot(data[:, c], label=channels[c])
            axs[c].legend(loc='upper right')
        
        # Set the overall title to the label
        fig.suptitle(f'Label: {label_str}', fontsize=14)
        
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    main()

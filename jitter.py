import numpy as np
import os
import joblib
import copy
import scipy.ndimage as ndimage

def jitter_SMPL(input_loc, new_pose_count=1):
    input_loc = os.path.join(os.getcwd(), input_loc)
    folder_list = os.listdir(input_loc)

    for folder in folder_list:
        folder_loc = os.path.join(input_loc, folder)
        if not os.path.isdir(folder_loc):
            continue

        file_list = os.listdir(folder_loc)
        for file_name in file_list:
            print(f'Processing file {file_name}')
            file_loc = os.path.join(folder_loc, file_name)
            with open(file_loc, "rb") as fin:
                original_poses = joblib.load(fin)  # list of dicts, each with "pose"

            file_name = os.path.splitext(file_name)[0]

            for i in range(new_pose_count):
                # Fresh copy of the original sequence
                poses = copy.deepcopy(original_poses)
                all_rots = np.array(poses[0]["pose"])
                all_rots = all_rots.reshape(all_rots.shape[0], -1, 3)
                print(f'shape {all_rots.shape}')
                # Add random noise
                noise = np.random.normal(0, 0.01, all_rots.shape)
                all_rots += noise

                # Smooth
                for joint in range(all_rots.shape[1]):
                    for dim in range(all_rots.shape[2]):
                        all_rots[:, joint, dim] = ndimage.gaussian_filter1d(all_rots[:, joint, dim], sigma=1)

                # Reassign
                poses[0]["pose"] = all_rots.reshape(all_rots.shape[0], -1)

                out_path = os.path.join(folder_loc, f"{file_name}{i}.pkl")
                print(f'difference between poses and original_poses: {np.sum(poses[0]["pose"] - original_poses[0]["pose"])}')
                with open(out_path, 'wb') as fout:
                    joblib.dump(poses, fout)

jitter_SMPL('dataset/train_motion_data')
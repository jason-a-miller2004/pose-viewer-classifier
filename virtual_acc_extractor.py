import joblib
import smplx
import os
import numpy as np
import scipy.signal as signal
import pandas as pd

import smpl2bvh as s2b
from imusim.platforms.imus import IdealIMU, Orient3IMU
from imusim.simulation.base import Simulation
from imusim.behaviours.imu import BasicIMUBehaviour
from imusim.io.bvh import loadBVHFile
from imusim.trajectories.rigid_body import SplinedBodyModel
from imusim.environment.base import Environment
from imusim.simulation.calibrators import ScaleAndOffsetCalibrator

TRAIN_INPUT_LOC = 'dataset/train_motion_data'
TEST_INPUT_LOC = 'dataset/test_motion_data'
OUTPUT_LOC = 'dataset/imu_data'
TEMP_DATA = './dataset/temp/temp.bvh'

FRAME_RATE = 30

label_map = {
    'seated_march': 0,
    'seated_shoulder_abduction': 1,
    'sit_to_stand': 2
}

version = "sim"
calibSamples = 1000
calibRotVel = 20

def imu_train_wrist_data(pose_file):
    s2b.smpl2bvh('./body_models', pose_file, TEMP_DATA, False, fps=FRAME_RATE)
    model = loadBVHFile(TEMP_DATA)
    print ('load mocap from ...', TEMP_DATA)
    #print (model)

    # spline intrepolation
    splinedModel = SplinedBodyModel(model)
    startTime = splinedModel.startTime
    endTime = splinedModel.endTime

    # setting sampling period to be same as frame rate can change tho
    frameCount = int(FRAME_RATE * (endTime-startTime))
    samplingPeriod = (endTime - startTime)/ frameCount

    print ('frameCount:', frameCount)
    print ('samplingPeriod:', samplingPeriod)

    if version == 'ideal':
        print ('Simulating ideal IMU.')

        # set simulation
        sim = Simulation()
        sim.time = startTime

        # run simulation
        dict_imu = {}
        imu = IdealIMU()
        imu.simulation = sim
        imu.trajectory = splinedModel.getJoint("Left_wrist")

        BasicIMUBehaviour(imu, samplingPeriod)

        dict_imu["Right_wrist"] = imu

        sim.run(endTime)

    elif version == 'sim':
        print ('Simulating Orient3IMU.')
        
        # set simulation
        env = Environment()
        calibrator = ScaleAndOffsetCalibrator(env, calibSamples, samplingPeriod, calibRotVel)
        sim = Simulation(environment=env)
        sim.time = startTime

        # run simulation
        dict_imu = {}

        imu = Orient3IMU()
        calibration = calibrator.calibrate(imu)
        print ('imu calibration:', "Right_wrist")
        
        imu.simulation = sim
        imu.trajectory = splinedModel.getJoint("Right_wrist")

        BasicIMUBehaviour(imu, samplingPeriod, calibration, initialTime=sim.time)

        dict_imu["Right_wrist"] = imu

        sim.run(endTime)

    # collect sensor values

    imu = dict_imu["Right_wrist"]

    if version == 'ideal':    
        acc_seq = imu.accelerometer.rawMeasurements.values.T
        gyro_seq = imu.gyroscope.rawMeasurements.values.T
    elif version == 'sim':
        acc_seq = imu.accelerometer.calibratedMeasurements.values.T
        gyro_seq = imu.gyroscope.calibratedMeasurements.values.T
        
    imu_data = np.concatenate((acc_seq, gyro_seq), axis=1)
    os.remove(TEMP_DATA)

    return imu_data

def imu_test_wrist_data(folder_loc):
    """
    Reads 'Accelerometer.csv' and 'Gyroscope.csv' from the given folder.
    
    Both CSVs are assumed to have columns in this order:
        time, seconds_elapsed, z, y, x
        
    Returns:
        sample_rate (float): The sampling rate (Hz) for both accelerometer and gyroscope.
        data_array (np.ndarray): Numpy array of shape (num_samples, 6) with columns:
                                 (accX, accY, accZ, gyroX, gyroY, gyroZ).
    """
    # Paths to CSV files
    acc_file = os.path.join(folder_loc, "Accelerometer.csv")
    gyro_file = os.path.join(folder_loc, "Gyroscope.csv")
    
    # Read CSVs with pandas
    acc_data = pd.read_csv(acc_file)
    gyro_data = pd.read_csv(gyro_file)
    
    # Extract time (we'll use 'seconds_elapsed' which is column index 1)
    acc_time = acc_data.iloc[:, 1].values  # 'seconds_elapsed' column for accelerometer
    
    # Compute sampling rate: 1 / (average delta of consecutive time entries)
    dt = np.diff(acc_time)
    sample_rate = 1.0 / np.mean(dt) if len(dt) > 0 else 0.0
    
    # Accelerometer columns: (time=0, seconds_elapsed=1, z=2, y=3, x=4)
    acc_x = acc_data.iloc[:, 4].values
    acc_y = acc_data.iloc[:, 3].values
    acc_z = acc_data.iloc[:, 2].values
    
    # Gyroscope columns follow the same indexing
    gyro_x = gyro_data.iloc[:, 4].values
    gyro_y = gyro_data.iloc[:, 3].values
    gyro_z = gyro_data.iloc[:, 2].values
    
    # Stack into a single array of shape (N, 6)
    data_array = np.column_stack((acc_x, acc_y, acc_z,
                                  gyro_x, gyro_y, gyro_z))
    
    return sample_rate, data_array
   

def extract_imu_data(rel_input_loc, resample_rate=None, dataset='train'):
    """
    Extracts wrist acceleration data from files within sub-folders of a specified directory, 
    optionally resamples the data, and saves both the acceleration data and labels to NumPy files.

    Args:
        rel_input_loc (str): 
            Relative path to the parent directory containing one or more sub-folders of CSV files. 
            Each sub-folder is associated with a specific label/category.
        resample_rate (float, optional): 
            Desired sample rate (in Hz) to resample each acceleration dataset. 
            If None, no resampling is performed. Defaults to None.
        dataset (str): 
            The type of dataset to process. 
            - If 'train', this function uses the global `wristAcceleration` function, applying a 
              fixed global frame rate (`FRAME_RATE`), rather than reading the sample rate from file. 
            - If 'test', it uses the `testAcceleration` function to extract the actual sample rate 
              from each CSV file before optionally resampling.
    """
    input_loc = os.path.join(os.getcwd(), rel_input_loc)
    output_loc = os.path.join(os.getcwd(), OUTPUT_LOC)

    # Get the list of files in the input directory
    folder_list = os.listdir(input_loc)

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_loc):
        os.makedirs(output_loc)

    imu_data = []
    label_data = []
    sample_rates = []
    # Process each file
    for idx,folder in enumerate(folder_list):
        folder_loc = os.path.join(input_loc, folder)
        label_num = label_map.get(folder)

        if not os.path.isdir(folder_loc):
            continue

        print(f'Processing folder {idx+1}/{len(folder_list)}: {folder}')

        file_list = os.listdir(folder_loc)
        for file_name in file_list:
            file_loc = os.path.join(folder_loc, file_name)
            if dataset == 'train':
                imu_seq = imu_train_wrist_data(file_loc)
                sample_rate = FRAME_RATE
            else:
                sample_rate, imu_seq = imu_test_wrist_data(file_loc)
            
            sample_rates.append(sample_rate)
            imu_data.append(imu_seq)
            label_data.append(label_num)

    imu_data = np.array(imu_data, dtype=object)
    label_data = np.array(label_data)
    print(f'imu shape {imu_data.shape}')
    if (resample_rate):
        data_samples = imu_data.shape[0]
        for i in range(data_samples):
            print(f'sample {i} shape {imu_data[i].shape}')
            length = imu_data[i].shape[0]
            imu_data[i] = signal.resample(imu_data[i],
                                    int(resample_rate * (length / sample_rates[i])),
                                    axis=0)

    np.save(os.path.join(output_loc, f'{dataset}_imu_data.npy'), imu_data, allow_pickle=True)
    np.save(os.path.join(output_loc, f'{dataset}_label_data.npy'), label_data)
    print('Data extraction complete!')

extract_imu_data(TRAIN_INPUT_LOC, dataset='train')
extract_imu_data(TEST_INPUT_LOC, resample_rate=FRAME_RATE, dataset='test')

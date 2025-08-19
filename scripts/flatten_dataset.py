import h5py 
import numpy as np
data_path = "/home/mfu/icrl_data/simple_test/r2d2.hdf5"

def get_all_entries(h5, key):
    observation_left = h5[key]["observation/exterior_image_1_left"][:]
    observation_left = np.frombuffer(observation_left, dtype='uint8').reshape(-1,180,320,3)
    observation_wrist = h5[key]["observation/wrist_image_left"][:]
    observation_wrist = np.frombuffer(observation_wrist, dtype='uint8').reshape(-1,180,320,3)
    action = np.concatenate([
        h5[key]["action/cartesian_position"][:], 
        h5[key]["action/cartesian_orientation"][:],
    ])
    proprioception = np.concatenate([
        h5[key]["observation/cartesian_position"][:],
        h5[key]["observation/gripper_position"][:]
    ])

    return observation_left, observation_wrist, action, proprioception


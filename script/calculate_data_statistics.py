import h5py 
import numpy as np
import torch
import json
from dp_gs.dataset.utils import euler_to_rot_6d, convert_multi_step, convert_delta_action
from tqdm import tqdm 

def get_action_statistics(
    data_path : str, 
    hdf5_keys : str = None, 
    num_pred_steps : int = 16
):

    if hdf5_keys is not None:
        with open(hdf5_keys, 'r') as f:
            hdf5_keys = json.load(f)

    # action_key = ["action/cartesian_position", "action/gripper_position"]
    # proprio_key = ["observation/cartesian_position", "observation/gripper_position"]
    action_key = "action/cartesian_position"
    action_gripper_key = "action/gripper_position"
    proprio_key = "observation/cartesian_position"
    proprio_gripper_key = "observation/gripper_position"

    with h5py.File(data_path, 'r') as f:
        # get the keys
        if hdf5_keys is None:
            hdf5_keys = list(f.keys())

        global_min_action, global_max_action = None, None

        for key in tqdm(hdf5_keys):
            actions = f[key][action_key][:]
            actions_gripper = f[key][action_gripper_key][:]
            actions_9d = np.concatenate([actions[:, :3], euler_to_rot_6d(actions[:, 3:6]), actions_gripper], axis=1)
            eos = np.zeros((actions_9d.shape[0], 1))
            eos[-1] = 1
            actions_9d = np.concatenate([actions_9d, eos], axis=1)
            actions_9d = torch.tensor(actions_9d)

            proprio = f[key][proprio_key][:]
            proprio_gripper = f[key][proprio_gripper_key][:]
            proprio_9d = np.concatenate([proprio[:, :3], euler_to_rot_6d(proprio[:, 3:6]), proprio_gripper], axis=1)
            proprio_9d = torch.tensor(proprio_9d)

            # calculate multi-step actions
            actions_9d = convert_multi_step(actions_9d, num_pred_steps)
            proprio_9d = convert_multi_step(proprio_9d, num_pred_steps)

            # calculate delta action
            delta_actions = convert_delta_action(actions_9d.numpy(), proprio_9d.numpy())

            # calculate statistics
            min_action, max_action = delta_actions.min(0), delta_actions.max(0)

            if global_min_action is None:
                global_min_action = min_action
                global_max_action = max_action
            else:
                global_min_action = np.stack([global_min_action, min_action], axis=0).min(0)
                global_max_action = np.stack([global_max_action, max_action], axis=0).max(0)

    return global_min_action, global_max_action

if __name__ == "__main__":

    dataset_config = "/home/mfu/research/dp_gs/config/dataset_config_simple.json"
    num_pred_steps = 16

    with open(dataset_config, 'r') as f:
        dataset_config = json.load(f)

    global_min_action, global_max_action = None, None

    for dp, hk in zip(dataset_config["dataset_path"], dataset_config["hdf5_keys"]):
        min_action, max_action = get_action_statistics(dp, hk, num_pred_steps)
        if global_min_action is None:
            global_min_action = min_action
            global_max_action = max_action
        else:
            global_min_action = np.stack([global_min_action, min_action], axis=0).min(0)
            global_max_action = np.stack([global_max_action, max_action], axis=0).max(0)

    print("Global Min Action: ", global_min_action)
    print("Global Max Action: ", global_max_action)

    # save the statistics 
    stats = {
        "shape" : global_min_action.shape,
        "min_action": global_min_action.flatten().tolist(),
        "max_action": global_max_action.flatten().tolist()
    }
    with open("action_statistics.json", 'w') as f:
        json.dump(stats, f)
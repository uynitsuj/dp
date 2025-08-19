"""
python test/test_fast_tokenizer.py --dataset-cfg.dataset-root /home/mfu/dataset/dp_gs 
"""
import os
import tyro 
import matplotlib.pyplot as plt

from dp_gs.dataset.image_dataset_sim import SequenceDataset as SimSequenceDataset
from dp_gs.util.args import ExperimentConfig
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from dp_gs.dataset.utils import default_vision_transform
from dp_gs.dataset.utils import unscale_action

def main(args : ExperimentConfig):
    number_of_samples = 100
    out_dir = "test_outputs/test_dataset_output"
    args.logging_cfg.output_dir = out_dir
    os.makedirs(out_dir, exist_ok=True)

    vision_transform = default_vision_transform()
    print("vision transform: ", vision_transform)
    mean, std = IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

    # making train val split
    dataset_train = SimSequenceDataset(
        dataset_config=args.dataset_cfg,
        shared_config=args.shared_cfg,
        logging_config=args.logging_cfg,
        vision_transform=vision_transform,
        split="train",
        debug=False,
    )
    


    print("Length of the training set: ", len(dataset_train))
    
    nbatch = next(iter(dataset_train))
    proprio = nbatch['proprio'].numpy()
    action = nbatch['action']
    action = unscale_action(action, stat=dataset_train.stats, type='diffusion').numpy()
    fig, axs = plt.subplots(3, 1)
    for k in range(3):
        axs[k].plot(proprio[k],'o', label='proprio', color='green', )
        axs[k].plot(range(1, action.shape[0]+1),action[:, k], label='action', color='red')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_dir}/proprio_action.png")   
    plt.clf()
    

    # start_time = time.time()
    # retrieval_time = []
    # for i, nbatch in enumerate(tqdm.tqdm(dataloader)):
    #     if i != 0:
    #         retrieval_time.append(time.time() - start_time)
    #     obs = nbatch['observation'][0]
    #     obs = undo_vision_transform(obs, mean, std)
    #     media.write_video(f"{out_dir}/{i}_obs.mp4", obs, fps=10)

    #     # plot proprio and action 
    #     proprio = nbatch['proprio'][0].numpy() # (only plotting the first proprio step) T, K 
    #     action = nbatch['action'][0, -1] # (only plotting the first action step) T, M
    #     action = unscale_action(action, stat=dataset_train.stats, type='diffusion').numpy()
        

    #     # plot proprio with each dimension as subplot
    #     T, K = proprio.shape
    #     fig, axs = plt.subplots(3, 1)
    #     for k in range(3):
    #         axs[k].plot(proprio[:, k], label='proprio', color='green')
    #         axs[k].plot(range(T, T+action.shape[0]), action[:, k], label='action', color='red')
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.savefig(f"{out_dir}/{i}_proprio_action.png")   
    #     plt.clf()

    #     if i == 10:
    #         break
    
    # print(f"average retrieval time: {np.mean(retrieval_time):.4f} seconds with std {np.std(retrieval_time):.4f}")

if __name__ == "__main__":
    args = tyro.cli(ExperimentConfig)
    main(args)
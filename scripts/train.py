#  CUDA_VISIBLE_DEVICES=4,5,6,7 --main_process_port=12547 accelerate launch script/train.py --dataset-cfg.dataset-root /home/mfu/dataset/dp/transfer_tiger_241204 --logging-cfg.log-name 241211_1403 --logging-cfg.output-dir /shared/projects/icrl/dp_outputs --shared-cfg.no-use-delta-action --shared-cfg.seq-length 4 --shared-cfg.num-pred-steps 16 --dataset-cfg.subsample-steps 4 --trainer-cfg.epochs 300
# from dp.dataset.dataset import SequenceDataset
# from timm.data.loader import MultiEpochsDataLoader
import numpy as np
import os 
import torch
import torch.backends.cudnn as cudnn
import tyro
import wandb
import yaml
import json
import dp.util.misc as misc
from typing import Optional

from diffusers.optimization import get_scheduler
from transformers import AutoProcessor
from dp.dataset.image_dataset_sim import SequenceDataset as SimSequenceDataset
from dp.dataset.image_dataset_v2 import SequenceDataset, VideoSampler, CollateFunction
from dp.dataset.utils import default_vision_transform, aug_vision_transform
from dp.policy.model import DiffusionPolicy, SimplePolicy, Dinov2DiscretePolicy
# from dp.util.args import ExperimentConfig
import dp.util.config as _config
from dp.util.args import DatasetConfig
from dataclasses import replace
from dp.util.misc import NativeScalerWithGradNormCount as NativeScaler
from dp.util.engine import train_one_epoch
from dp.util.ema import ModelEMA

from pathlib import Path
from torch.utils.data import DataLoader 
from torch.utils.tensorboard import SummaryWriter
from dp.util.misc import MultiEpochsDataLoader
from tqdm import tqdm
import time
from datetime import datetime

def main(args : _config.TrainConfig):
    # spawn is needed to initialize vision augmentation within pytorch workers
    try:
        torch.multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass  # start method already set
    
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    # fix the seed for reproducibility
    seed = args.shared_cfg.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    start_epoch = args.shared_cfg.start_epoch
    num_epochs = args.trainer_cfg.epochs

    log_name = args.exp_name + "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    args = replace(args, logging_cfg=replace(args.logging_cfg, log_name=log_name))
    args = replace(args, logging_cfg=replace(args.logging_cfg, output_dir=os.path.join(args.logging_cfg.output_dir, log_name)))
    os.makedirs(args.logging_cfg.output_dir, exist_ok=True)
    output_dir = args.logging_cfg.output_dir

    device = torch.device(args.device)

    # accelerator = Accelerator()
    # device = accelerator.device
    
    if args.model_cfg.policy_type == "diffusion":
        policy = DiffusionPolicy
        tokenizer = None
    elif args.model_cfg.policy_type == "discrete":
        policy = Dinov2DiscretePolicy
        tokenizer = AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True)
        extra_kwargs = {
            'vocab_size': tokenizer.vocab_size + 1,  # eos token
            'num_tokens': args.shared_cfg.num_tokens
        }
    else:
        policy = SimplePolicy
        tokenizer = None 
    model = policy(
        shared_cfg=args.shared_cfg,
        model_cfg=args.model_cfg,
        **(extra_kwargs if args.model_cfg.policy_type == "discrete" else {})
    ).to(device)

    total = sum(param.numel() for param in model.parameters())
    print(f"Total parameters: {total:,}")

    model_without_ddp = model

    if args.distributed:
        find_unused_parameters = False 
        if isinstance(model, Dinov2DiscretePolicy):
            find_unused_parameters = True 
        # elif isinstance(model, SimplePolicy) and args.model_cfg.policy_cfg.simple_vision_model == "dino":
        #     find_unused_parameters = True
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], 
            find_unused_parameters=find_unused_parameters
        )
        model_without_ddp = model.module

    eff_batch_size = args.shared_cfg.batch_size * args.trainer_cfg.accum_iter * misc.get_world_size()
    print("accumulate grad iterations: %d" % args.trainer_cfg.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    # vision transform
    if args.shared_cfg.s2:
        resolution = args.shared_cfg.image_size * 2
    else:
        resolution = args.shared_cfg.image_size
    base_vision_transform = default_vision_transform(resolution=resolution) # default img_size: Union[int, Tuple[int, int]] = 224,
    aug_transform = aug_vision_transform(resolution=resolution) # default img_size: Union[int, Tuple[int, int]] = 224,
    print("vision transform: ", default_vision_transform)
    print("augmented vision transform: ", aug_transform)

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    # if args.dataset_cfg.is_sim_data:
    #     dataset_type = SimSequenceDataset
    # else:
    dataset_type = SequenceDataset

    if type(args.dataset_cfg) == DatasetConfig and args.dataset_cfg.dataset_root is None:
        raise ValueError("Dataset root directory is not set")
    else:
        args = replace(args, dataset_cfg=args.dataset_cfg.create())

    # datasets
    dataset_train = dataset_type(
        dataset_config=args.dataset_cfg,
        shared_config=args.shared_cfg,
        logging_config=args.logging_cfg,
        vision_transform=aug_transform,
        split="train",
    )
    dataset_val = dataset_type(
        dataset_config=args.dataset_cfg,
        shared_config=args.shared_cfg,
        vision_transform=base_vision_transform,
        logging_config=args.logging_cfg,
        split="val"
    )

    # samplers
    train_sampler = VideoSampler(
        dataset_train, 
        batch_size=args.shared_cfg.batch_size, 
        sequence_length=args.shared_cfg.seq_length, 
        num_replicas=num_tasks,
        rank=global_rank,
    )
    val_sampler = VideoSampler(
        dataset_val, 
        batch_size=args.shared_cfg.batch_size, 
        sequence_length=args.shared_cfg.seq_length,
        num_replicas=num_tasks,
        rank=global_rank,
    )

    collate_fn = CollateFunction(
        args.shared_cfg.seq_length, 
        tokenizer=tokenizer, 
        max_token_length=args.shared_cfg.num_tokens,
        pred_left_only=args.model_cfg.policy_cfg.pred_left_only,
        pred_right_only=args.model_cfg.policy_cfg.pred_right_only,
    )
    dataloader_train = MultiEpochsDataLoader(
        dataset_train,
        batch_sampler=train_sampler,
        num_workers=args.trainer_cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=args.trainer_cfg.pin_memory,
        prefetch_factor=2,
    )
    dataloader_val = MultiEpochsDataLoader(
        dataset_val,
        batch_sampler=val_sampler,
        num_workers=args.trainer_cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=args.trainer_cfg.pin_memory,
        prefetch_factor=2,
    )

    # Start a wandb run with `sync_tensorboard=True`
    if global_rank == 0:# and args.logging_cfg.log_name is not None:
        wandb.init(project="dp", config=args, name=log_name, sync_tensorboard=True)


    # SummaryWrite
    if global_rank == 0 and args.logging_cfg.log_dir is not None:
        os.makedirs(args.logging_cfg.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.logging_cfg.log_dir)
    else:
        log_writer = None

    # Standard ADAM optimizer
    # Note that EMA parametesr are not optimized
    param_groups = misc.add_weight_decay(model_without_ddp, args.optimizer_cfg.weight_decay)
    optimizer = torch.optim.AdamW(
        params=param_groups,
        lr=args.optimizer_cfg.lr)
    loss_scaler = NativeScaler()

    # --resume
    misc.resume_from_ckpt(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    # Cosine LR schedule with linear warmup
    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=len(dataloader_train) * num_epochs
    )

    # Initialize EMA after model creation
    if args.model_cfg.policy_type == "diffusion":
        ema = ModelEMA(model_without_ddp, decay=0.9999, device=device)
    else:
        ema = None

    for epoch_idx in tqdm(range(start_epoch, num_epochs), leave=True, desc="Training"):
        train_stats = train_one_epoch(
            model=model, data_loader=dataloader_train, 
            optimizer=optimizer, lr_scheduler=lr_scheduler,
            device=device, epoch=epoch_idx,
            loss_scaler=loss_scaler, log_writer=log_writer,
            validate=False, args=args,
            ema=ema  # Pass EMA to training function
        )
        
        if dataloader_val is not None:
            with torch.no_grad():
                # Use EMA model for validation if available
                original_state_dict = None
                if ema is not None:
                    # Store original weights before applying EMA
                    original_state_dict = {k: v.clone() for k, v in model_without_ddp.state_dict().items()}
                    ema.apply_ema(model_without_ddp)
                
                val_stats = train_one_epoch(
                    model=model, data_loader=dataloader_val, 
                    optimizer=optimizer, lr_scheduler=lr_scheduler,
                    device=device, epoch=epoch_idx,
                    loss_scaler=loss_scaler, log_writer=log_writer,
                    validate=True, args=args
                )

                # Restore original weights after validation if EMA was applied
                if ema is not None and original_state_dict is not None:
                    model_without_ddp.load_state_dict(original_state_dict)
            print("Validation Epoch {}".format(epoch_idx))
        
        # save checkpoint with EMA state
        if misc.is_main_process() and (epoch_idx % args.shared_cfg.save_every == 0 or epoch_idx == num_epochs - 1):
            save_dict = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch_idx,
            }
            if ema is not None:
                save_dict['ema'] = ema.state_dict()
            torch.save(save_dict, os.path.join(output_dir, f'checkpoint_{epoch_idx}.pt'))

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch_idx}
        if dataloader_val is not None:
            log_stats.update({f'val_{k}': v for k, v in val_stats.items()})

        if args.logging_cfg.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.logging_cfg.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

# if __name__ == '__main__':
#     # parsing args 
#     args = tyro.cli(ExperimentConfig)

#     if args.load_config is not None: 
#         print("loading configs from file: ", args.load_config)
#         assert os.path.exists(args.load_config), f"Config file does not exist: {args.load_config}"
#         args : ExperimentConfig = yaml.load(Path(args.load_config).read_text(), Loader=yaml.Loader) 

#     # creating the output directory and logging directory 
#     if args.logging_cfg.log_name is not None: 
#         args.logging_cfg.output_dir = os.path.join(args.logging_cfg.output_dir, args.logging_cfg.log_name)
#     if args.logging_cfg.log_dir is None:
#         args.logging_cfg.log_dir = args.logging_cfg.output_dir
#     if args.logging_cfg.output_dir:
#         Path(args.logging_cfg.output_dir).mkdir(parents=True, exist_ok=True)

#     # dump the args into a yaml file 
#     with open(os.path.join(args.logging_cfg.output_dir, "run.yaml"), 'w') as f:
#         yaml.dump(args, f)

#     main(args)

if __name__ == "__main__":
    main(_config.cli())
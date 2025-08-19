import math
import sys
from typing import Iterable, Union

import torch

# from dp.util.args import ExperimentConfig
import dp.util.config as _config
from dp.policy.model import DiffusionPolicy, SimplePolicy

from . import misc


def train_one_epoch(model: Union[SimplePolicy, DiffusionPolicy], data_loader: Iterable, 
                    optimizer: torch.optim.Optimizer, lr_scheduler : torch.optim.lr_scheduler,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None, validate=False,
                    args : _config.TrainConfig=None,
                    ema=None):
    if validate:
        model.eval()
        validation_loss = 0
    else:
        model.train()
        optimizer.zero_grad() # Clear gradients only during training

    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    accum_iter = args.trainer_cfg.accum_iter

    # breakpoint()
    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    for data_iter_step, dataset_item in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        for k, v in dataset_item.items():
            dataset_item[k] = v.to(device, non_blocking=True)

        # dataset_item.keys() -> ['action', 'proprio', 'observation']
        # can apply data transforms here, and also apply inverse transforms at model output at inference wrapper calls
        # example shapes:
        # dataset_item["action"].shape -> [128, 1, 30, 29]
        # dataset_item["proprio"].shape -> [128, 1, 29]
        # dataset_item["observation"].shape -> torch.Size([128, 1, 3, 3, 224, 224])
        # TODO: apply data transforms here, and also apply inverse transforms at model output at inference wrapper calls

        # import pdb; pdb.set_trace()

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            loss = model(dataset_item)
            if isinstance(loss, dict):
                loss_dict = loss
                loss = loss["loss"]
            else:
                loss_dict = {"loss": loss}
        
        loss_value = loss.item()
        loss_value_dict = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in loss_dict.items()}
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_value_dict = {k: v / accum_iter for k, v in loss_value_dict.items()}
        if not validate:
            loss_scaler(loss, optimizer, parameters=model.parameters(),
                        update_grad=(data_iter_step + 1) % accum_iter == 0,
                        clip_grad=1.0)
                        # clip_grad=0.5)
        
        # this is different from standard pytorch behavior
        if not validate:
            lr_scheduler.step()

        if (data_iter_step + 1) % accum_iter == 0 and not validate:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        for key in loss_value_dict:
            metric_logger.update(**{key: loss_value_dict[key]})

        # metric_logger.update(loss=loss_value)
        # if "acc" in loss_value_dict:
        #     metric_logger.update(acc=loss_value_dict["acc"])

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if validate:
            validation_loss += loss_value_reduce
        loss_value_dict_reduce = {k: misc.all_reduce_mean(v) for k, v in loss_value_dict.items()}
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            global_step = data_iter_step + len(data_loader) * epoch
            if not validate:
                log_writer.add_scalar('train_loss', loss_value_reduce, global_step)
                for k, v in loss_value_dict_reduce.items():
                    log_writer.add_scalar('train_{}'.format(k), v, global_step)
                log_writer.add_scalar('lr', lr, global_step)
            else:
                log_writer.add_scalar('val_loss', loss_value_reduce, global_step)
                for k, v in loss_value_dict_reduce.items():
                    log_writer.add_scalar('val_{}'.format(k), v, global_step)

        if not validate:
            # Update EMA parameters after optimizer step if in training mode
            if ema is not None:
                ema.update(model.module if hasattr(model, 'module') else model)

    if log_writer is not None and validate:
        validation_loss = validation_loss / len(data_loader)
        log_writer.add_scalar('val_loss_epoch', validation_loss, epoch)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

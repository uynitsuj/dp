# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import builtins
import datetime
import os
import time
from collections import defaultdict, deque
from pathlib import Path
import urllib
from tqdm import tqdm

import torch
import torch.utils.data
import torch.distributed as dist
from torch import inf
import numpy as np
from dp.util.args import ExperimentConfig

def load_state_dict_flexible(model, state_dict, return_msg=False):
    """
    Load state dict while handling both DDP and non-DDP scenarios.
    """
    # Check if state_dict has 'module.' prefix
    is_parallel = any(key.startswith('module.') for key in state_dict.keys())
    
    # If model is DDP but state_dict is not parallel
    if hasattr(model, 'module') and not is_parallel:
        msg = model.module.load_state_dict(state_dict)
    
    # If model is not DDP but state_dict is parallel
    elif not hasattr(model, 'module') and is_parallel:
        prefix = 'module.'
        new_state_dict = {key.removeprefix(prefix): value 
                         for key, value in state_dict.items()}
        msg = model.load_state_dict(new_state_dict)
    
    # If both are parallel or both are not parallel
    else:
        msg = model.load_state_dict(state_dict)
        
    if return_msg:
        return model, msg
    else:
        return model

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        force = force or (get_world_size() > 8)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print('[{}] '.format(now), end='')  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if args.dist_on_itp:
        args.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        args.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        args.gpu = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        args.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['RANK'] = str(args.rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)
        # ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        setup_for_distributed(is_master=True)  # hack
        args.distributed = False
        return

    args.distributed = True

    print("GPU::", args.gpu)
    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.amp.GradScaler('cuda')

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


def save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler):
    output_dir = Path(args.logging_cfg.output_dir)
    epoch_name = str(epoch)
    if loss_scaler is not None:
        checkpoint_paths = [output_dir / ('checkpoint-%s.pth' % epoch_name)]
        for checkpoint_path in checkpoint_paths:
            to_save = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'scaler': loss_scaler.state_dict(),
                'args': args,
            }

            save_on_master(to_save, checkpoint_path)
    else:
        client_state = {'epoch': epoch}
        model.save_checkpoint(save_dir=args.logging_cfg.output_dir, tag="checkpoint-%s" % epoch_name, client_state=client_state)


def load_model(model_without_ddp, path):
    if path.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            path, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(path, map_location='cpu')
    new_checkpoint = {}
    for key, value in checkpoint['model'].items():
        key = key.replace("llma", "llama")
        new_checkpoint[key] = value
        
    missing_keys, unexpected_keys = model_without_ddp.load_state_dict(new_checkpoint, strict=False)
    actual_missing_keys = []
    for key in missing_keys:
        if 'vision_encoder' in key or 'llama' in key:
            continue
        else:
            actual_missing_keys.append(key)
    print("Load checkpoint %s" % path)
    print("Missing keys: ", actual_missing_keys)
    print("Unexpected keys: ", unexpected_keys)


def resume_from_ckpt(args, model_without_ddp, optimizer=None, loss_scaler=None, ema=None):
    if args.shared_cfg.resume:
        if args.shared_cfg.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.shared_cfg.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.shared_cfg.resume, map_location='cpu')
        
        if isinstance(checkpoint, dict):
            model_without_ddp, msg = load_state_dict_flexible(model_without_ddp, checkpoint['model'], return_msg=True)
            # msg = model_without_ddp.load_state_dict(checkpoint['model'])
            print('Pretrained weights found at {} and loaded with msg: {}'.format(args.shared_cfg.resume, msg))
            if optimizer is not None and 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
            if loss_scaler is not None and 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
            if ema is not None and 'ema' in checkpoint:
                ema.load_state_dict(checkpoint['ema'])
            if 'epoch' in checkpoint:
                args.shared_cfg.start_epoch = checkpoint['epoch'] + 1
        else:
            model_without_ddp, msg = load_state_dict_flexible(model_without_ddp, checkpoint, return_msg=True)
            # msg = model_without_ddp.load_state_dict(checkpoint)
            print('Pretrained weights found at {} and loaded with msg: {}'.format(args.shared_cfg.resume, msg))

def all_reduce_mean(x):
    world_size = get_world_size()
    if world_size > 1:
        x_reduce = torch.tensor(x).cuda()
        dist.all_reduce(x_reduce)
        x_reduce /= world_size
        return x_reduce.item()
    else:
        return x


def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]


class DistributedSubEpochSampler(torch.utils.data.Sampler):

    def __init__(self, dataset, num_replicas, rank, shuffle, split_epoch=1, seed=0):
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.split_epoch = split_epoch
        self.seed = seed
        
        self.epoch = 0

        self.num_samples = len(dataset) // (num_replicas * split_epoch)

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch // self.split_epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        indices = indices[self.rank * self.split_epoch + self.epoch % self.split_epoch::self.num_replicas * self.split_epoch]
        assert len(indices) >= self.num_samples
        indices = indices[:self.num_samples]

        return iter(indices)

    def set_epoch(self, epoch):
        self.epoch = epoch
        
class DistributedSubEpochLongSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, num_replicas, rank, shuffle, eff_batch_size, split_epoch=1, seed=0):
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.split_epoch = split_epoch
        self.seed = seed
        self.epoch = 0
        self.num_samples = len(dataset) // (num_replicas * split_epoch) # number of samples per replica
        self.eff_batch_size = eff_batch_size
        
    def __len__(self):
        return len(self.dataset) // (self.num_replicas * self.split_epoch)
    
    def __iter__(self):
        if self.shuffle: 
            print("######### running distributed sub epoch sampler #########")
            # we interleave the indices from different task types
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch // self.split_epoch)
            indices_sets = []
            for i in range(len(self.dataset.type_indices_list[:-1])):
                # put task trajectory in effective batch size sets 
                cur_indices = torch.arange(self.dataset.type_indices_list[i], self.dataset.type_indices_list[i+1])
                cur_indices = cur_indices[torch.randperm(len(cur_indices), generator=g)].tolist()
                cur_indices = [cur_indices[i:i+self.eff_batch_size] for i in range(0, len(cur_indices), self.eff_batch_size) if i+self.eff_batch_size <= len(cur_indices)]
                indices_sets.extend(cur_indices)
            # shuffle the indices sets 
            rng = np.random.default_rng(self.seed + self.epoch // self.split_epoch)
            rng.shuffle(indices_sets)
            indices = []
            for i in range(len(indices_sets)):
                indices += indices_sets[i]
        else:
            indices = list(range(len(self.dataset)))
        
        if self.split_epoch > 1:
            curr_epoch = self.epoch % self.split_epoch
            self.num_samples = len(indices) // (self.split_epoch * self.num_replicas)
            indices = indices[self.num_samples * self.num_replicas * curr_epoch : self.num_samples * self.num_replicas * (curr_epoch + 1)]
        indices = indices[self.rank::self.num_replicas]
        self.num_samples = len(indices)
        return iter(indices)
    
    def set_epoch(self, epoch):
        self.epoch = epoch
        
def download(url: str, root: str):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        return download_target

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True, unit_divisor=1024) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break
                output.write(buffer)
                loop.update(len(buffer))


    return download_target

class _RepeatSampler:
    """
    A sampler wrapper that repeats and tracks epochs.
    
    Args:
        sampler (Sampler): Base sampler to repeat
    """
    def __init__(self, sampler):
        self.sampler = sampler
        self.epoch = 0

    def __iter__(self):
        while True:
            if hasattr(self.sampler, 'set_epoch'):
                self.sampler.set_epoch(self.epoch)
            yield from iter(self.sampler)
            self.epoch += 1

    def __len__(self):
        return len(self.sampler)


class MultiEpochsDataLoader(torch.utils.data.DataLoader):
    """
    based on # from timm.data.loader import MultiEpochsDataLoader
    DataLoader that handles multiple epochs with proper epoch tracking for samplers.
    Supports samplers with epoch-based shuffling through the set_epoch method.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        
        # Handle both regular sampler and batch sampler cases
        if self.batch_sampler is None:
            self.sampler = _RepeatSampler(self.sampler)
        else:
            self.batch_sampler = _RepeatSampler(self.batch_sampler)
            
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        if self.batch_sampler is not None:
            return len(self.batch_sampler.sampler)
        return len(self.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)

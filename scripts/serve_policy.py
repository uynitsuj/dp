#!/usr/bin/env python3
"""
Remote policy serving wrapper (WebSocket, msgpack-numpy) that mirrors the
interface of the user's existing `WebsocketPolicyServer`, but performs
inference via the "proper" DiffusionWrapper path shown in the example
inference script.

Protocol
--------
- First frame sent by server: metadata dict (handshake)
- Client sends: a single Python tree (dict/list/np arrays) representing a
  *batch-like* input compatible with DiffusionWrapper.
  
  Expected minimal schema (batch size = 1):
    {
      "images": float32 array(s) if your model expects vision inputs
                 (e.g., (B,T,3,H,W) or (B,3,H,W) depending on your policy),
      "proprio": float32 (B,T,Dp),
      "action":  (optional, ignored) ground-truth actions,
      "action_mask": (optional; used only for discrete decode parity),
      ... any other keys your trained policy expects ...
    }
  NOTE: Values should already be normalized exactly as during training
  (the DiffusionWrapper expects the same preprocessing as the training
  dataloader). Batch size must be 1.

- Server replies: dict containing at least
    {
      "actions": np.ndarray (H, Da)  # unscaled, in environment units
      "server_timing": {"infer_ms": float, "prev_total_ms": float?}
      ... model-specific auxiliaries may be included later ...
    }

Run
---
Example:
  CUDA_VISIBLE_DEVICES=0 \
  python remote_policy_server.py \
    --model_ckpt_folder /shared/projects/icrl/dp_outputs/250208_1027 \
    --ckpt_id 50 \
    --host 0.0.0.0 --port 8111

Client (sketch):
  import websockets, msgpack_numpy as m
  async with websockets.connect("ws://HOST:8111", compression=None, max_size=None) as ws:
      meta = m.unpackb(await ws.recv())
      await ws.send(m.Packer().pack(your_batch_dict))
      out = m.unpackb(await ws.recv())
      print(out["actions"])  # (H, Da)

"""

import asyncio
import http
import logging
import os
import traceback
from dataclasses import dataclass
from typing import Any, Dict, Optional, List

import dill
import msgpack_numpy
import numpy as np
import torch
import websockets.asyncio.server as _server
import websockets.frames
import yaml

# --- project imports (adapt paths to your repo layout) ---
from dp.policy.diffusion_wrapper import DiffusionWrapper
from dp.policy.model import Dinov2DiscretePolicy
from dp.dataset.utils import unscale_action
from transformers import AutoProcessor
from dp.dataset.utils import default_vision_transform as transforms_noaug_train

logger = logging.getLogger("remote_policy_server")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s")


# ----------------------------
# Config / CLI
# ----------------------------
@dataclass
class ServerConfig:
    # model_ckpt_folder: str = "/home/justinyu/nfs_us/justinyu/dp/resnet-intergripper-proprio-29D_20250823_152419"
    model_ckpt_folder: str = "/home/justinyu/nfs_us/justinyu/dp/intergripper_proprio_scaleDP_20250821_205127"
    ckpt_id: int = 80
    host: str = "0.0.0.0"
    port: int = 8111
    device: str = "cuda"  # "cuda" | "cpu" | "cuda:0" ...
    # Optional path to action stats json (overrides auto-resolve)
    action_stats_path: Optional[str] = None


# ----------------------------
# Utilities
# ----------------------------

def _to_torch(x, device: torch.device) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def _tree_to_device(tree: Any, device: torch.device) -> Any:
    if isinstance(tree, dict):
        return {k: _tree_to_device(v, device) for k, v in tree.items()}
    if isinstance(tree, (list, tuple)):
        xs = [_tree_to_device(v, device) for v in tree]
        return type(tree)(xs) if not isinstance(tree, list) else xs
    return _to_torch(tree, device)


def _tree_to_numpy(tree: Any) -> Any:
    if isinstance(tree, dict):
        return {k: _tree_to_numpy(v) for k, v in tree.items()}
    if isinstance(tree, (list, tuple)):
        xs = [_tree_to_numpy(v) for v in tree]
        return type(tree)(xs) if not isinstance(tree, list) else xs
    if isinstance(tree, torch.Tensor):
        return tree.detach().cpu().numpy()
    return tree


def _maybe_load_action_stats(path: Optional[str]) -> Optional[Dict[str, np.ndarray]]:
    if path is None or not os.path.exists(path):
        return None
    try:
        import json
        with open(path, "r") as f:
            data = json.load(f)
        actions = None
        if "actions" in data:
            actions = data["actions"]
        elif "norm_stats" in data and "actions" in data["norm_stats"]:
            actions = data["norm_stats"]["actions"]
        if actions is None:
            return None
        if "min" in actions and "max" in actions:
            return {"min": np.array(actions["min"], dtype=np.float32),
                    "max": np.array(actions["max"], dtype=np.float32)}
        if "q01" in actions and "q99" in actions:
            return {"min": np.array(actions["q01"], dtype=np.float32),
                    "max": np.array(actions["q99"], dtype=np.float32)}
    except Exception:
        logger.exception("Failed to load action stats; continuing without.")
    return None

def repack_obs(obs: Dict[str, Any], camera_keys: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Repack the observation to the format expected by the model.
    """
    new_obs = {}
    observation = []
    if camera_keys is not None:
        for key in camera_keys:
            assert key in obs, f"Camera key {key} not found in obs"
            # if the range of image values is 0-255, convert to 0-1
            # if obs[key].max() <= 1.0:
            #     observation.append(obs[key])
            # else:
            #     observation.append(obs[key] / 255.0) # Assumption: if image max value is gt 1, then it's likely in [0, 255]
            observation.append(obs[key])

    else:
        for key in obs:
            if "images" in key:
                # if the range of image values is 0-255, convert to 0-1
                # if obs[key].max() <= 1.0:
                #     observation.append(obs[key]) # Assumes the order of obs dict respects the order that the model was trained on
                # else:
                #     observation.append(obs[key] / 255.0)
                observation.append(obs[key]) # Assumes the order of obs dict respects the order that the model was trained on


    new_obs["observation"] = torch.cat(observation, dim=1).unsqueeze(0)    # (B, T, num_cameras, C, H, W)
    new_obs["proprio"] = obs["state"]
    return new_obs

# ----------------------------
# Inference adapter (DiffusionWrapper)
# ----------------------------
class _DiffusionWrapperAdapter:
    """Wraps DiffusionWrapper to expose .infer(obs)->{"actions": np.ndarray, ...}.

    The adapter expects the incoming obs tree to match the *batch* format used
    in training (keys like "proprio", maybe image tensors, token fields, etc.).
    Batch size must be 1.
    """

    def __init__(self, cfg: ServerConfig):
        self._device = torch.device(cfg.device)
        self._infer = DiffusionWrapper(cfg.model_ckpt_folder, cfg.ckpt_id, device=str(self._device))

        # Optional tokenizer if the underlying model is discrete
        self._tokenizer = None
        if isinstance(self._infer.model, Dinov2DiscretePolicy):
            self._tokenizer = AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True)
            logger.info("Loaded tokenizer for discrete policy decode.")

        # Stats for unscaling (the wrapper also carries stats, but we support
        # external override to keep parity with the user's example.)
        self._stats = self._infer.stats
        if cfg.action_stats_path is not None:
            ext = _maybe_load_action_stats(cfg.action_stats_path)
            if ext is not None:
                # store as a tiny shim matching unscale_action expectations
                class _Shim:  # minimal wrapper
                    def __init__(self, d):
                        self.actions = d
                self._stats = _Shim(ext)
                logger.info("Using external action stats override (%s).", cfg.action_stats_path)
        self.cfg = cfg

    @property
    def device(self) -> torch.device:
        return self._device

    def infer(self, obs_tree: Dict[str, Any]) -> Dict[str, Any]:
        # Move to device (and enforce batch size = 1)
        with torch.no_grad():
            nbatch = _tree_to_device(obs_tree, self._device)

            # Sanity: batch size 1 enforcement for common keys
            for key in ("proprio", "images", "image", "rgb", "observation"):
                if key in nbatch and isinstance(nbatch[key], torch.Tensor):
                    if nbatch[key].shape[0] != 1:
                        raise ValueError(f"Expected batch size 1 for key '{key}', got {nbatch[key].shape}")

            # Forward pass (return_tokens=True -> for discrete decode path parity)
            # The user's example calls `inferencer(nbatch, True)`; we mimic that.
            # import pdb; pdb.set_trace()
            
            '''
            self._infer.model.__dir__()
            ['T_destination', '__annotations__', '__call__', '__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattr__', '__getattribute__', '__getstate__', '__gt__', '__hash__', '__init__', '__init_subclass__', 
            '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__setstate__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_apply', '_backward_hooks', '_backward_pre_hooks', '_buffers', 
            '_call_impl', '_compiled_call_impl', '_forward_hooks', '_forward_hooks_always_called', '_forward_hooks_with_kwargs', '_forward_pre_hooks', '_forward_pre_hooks_with_kwargs', '_get_backward_hooks', '_get_backward_pre_hooks', '_get_name', 
            '_get_observation_features', '_get_vision_features', '_is_full_backward_hook', '_load_from_state_dict', '_load_state_dict_post_hooks', '_load_state_dict_pre_hooks', '_maybe_warn_non_full_backward_hook', '_modules', '_named_members', 
            '_non_persistent_buffers_set', '_parameters', '_process_batch', '_process_images', '_register_load_state_dict_pre_hook', '_register_state_dict_hook', '_replicate_for_data_parallel', '_save_to_state_dict', '_slow_forward', '_state_dict_hooks', 
            '_state_dict_pre_hooks', '_version', '_wrapped_call_impl', 'action_dim', 'action_horizon', 'add_module', 'apply', 'bfloat16', 'buffers', 'call_super_init', 'children', 'compile', 'cpu', 'cuda', 'diffusion_model_type', 'double', 'dump_patches', 
            'eval', 'extra_repr', 'float', 'forward', 'forward_inference', 'forward_loss', 'get_buffer', 'get_extra_state', 'get_parameter', 'get_submodule', 'global_cond_dim', 'gripper_loss_w', 'half', 'ipu', 'load_state_dict', 'lora_rank_vision_encoder', 
            'lowdim_obs_dim', 'modules', 'mtia', 'named_buffers', 'named_children', 'named_modules', 'named_parameters', 'noise_pred_net', 'noise_scheduler', 'num_cameras', 'num_diffusion_iters', 'obs_horizon', 'only_vision', 'parameters', 'pred_left_only', 
            'pred_right_only', 'prediction_type', 'register_backward_hook', 'register_buffer', 'register_forward_hook', 'register_forward_pre_hook', 'register_full_backward_hook', 'register_full_backward_pre_hook', 'register_load_state_dict_post_hook', 
            'register_load_state_dict_pre_hook', 'register_module', 'register_parameter', 'register_state_dict_post_hook', 'register_state_dict_pre_hook', 'requires_grad_', 's2', 'set_extra_state', 'set_submodule', 'share_memory', 'smart_apply', 'state_dict', 
            'timm_vision_encoder', 'to', 'to_empty', 'train', 'training', 'type', 'vision_encoder', 'vision_encoder_pretrained_type', 'vision_feature_dim', 'xpu', 'zero_grad']
            '''
            camera_keys = None
            if hasattr(self._infer.model, "camera_keys"):
                camera_keys = self._infer.model.camera_keys
            nbatch = repack_obs(nbatch, camera_keys = camera_keys)
            vision_transform = transforms_noaug_train(resolution=224) # TODO: remove hardcode everywhere eventually and take this from cfg
            # import pdb; pdb.set_trace()
            nbatch["observation"] = vision_transform(nbatch["observation"]/255.0)
            pred_action = self._infer(nbatch)

            # Decode or pass-through depending on model type
            if isinstance(self._infer.model, Dinov2DiscretePolicy):
                # If tokens were returned, decode to continuous action with tokenizer
                # Build GT-like path as in example to discover time horizon
                # Here we infer H from pred_action directly when possible.
                if isinstance(pred_action, torch.Tensor):
                    # Some wrappers may already map tokens->actions; handle both
                    decoded = pred_action
                else:
                    # `pred_action` may be tokens; use tokenizer.decode
                    tokens = pred_action
                    # Expect nbatch['action_mask'] to exist for valid lengths; if absent, decode raw
                    if hasattr(self._infer.model, "action_dim"):
                        action_dim = self._infer.model.action_dim
                    else:
                        raise RuntimeError("Discrete model missing action_dim attribute.")
                    # Try to infer horizon from mask or fallback to model cfg
                    H = None
                    if isinstance(nbatch, dict) and "action_mask" in nbatch:
                        H = int(torch.as_tensor(nbatch["action_mask"]).sum(dim=1).max().item())
                    # Decode tokens -> (B,H,D)
                    decoded_np = self._tokenizer.decode(
                        [tokens[0].tolist()] if isinstance(tokens, torch.Tensor) else tokens,
                        time_horizon=H,
                        action_dim=action_dim,
                    )
                    decoded = torch.tensor(decoded_np, device=self._device)
                actions = decoded
            else:
                # Continuous policies typically output (B,H,D) scaled to training space.
                actions = pred_action

            # Unscale to environment units (matches example)
            # import pdb; pdb.set_trace()
            # actions = unscale_action(actions, stat=self._stats, type='diffusion')  # (B,H,D)

            # if self._infer.data_transforms is not None:
            #     data_dict = {
            #         "action": actions,
            #         "proprio": nbatch["proprio"],
            #     }
            #     for tf_fn in self._infer.data_transforms.outputs:
            #         data_dict = tf_fn(data_dict)
            #     actions = data_dict["action"]

            # nbatch["actions"] = actions

            # actions_np = actions.detach().to('cpu').numpy()
            actions_np = actions[0]  # strip batch -> (H,D)

            return {"actions": actions_np}


# ----------------------------
# WebSocket Server
# ----------------------------
class WebsocketPolicyServer:
    def __init__(self, adapter: _DiffusionWrapperAdapter, host: str, port: int, metadata: Dict[str, Any]):
        self._adapter = adapter
        self._host = host
        self._port = port
        self._metadata = metadata
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            logger.info(f"Server listening on ws://{self._host}:{self._port}")
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()
        # Handshake metadata
        await websocket.send(packer.pack(self._metadata))

        prev_total = None
        try:
            while True:
                start = asyncio.get_running_loop().time()
                obs = msgpack_numpy.unpackb(await websocket.recv())

                t0 = asyncio.get_running_loop().time()
                out = self._adapter.infer(obs)
                infer_ms = (asyncio.get_running_loop().time() - t0) * 1000.0

                payload = dict(out)
                payload["server_timing"] = {"infer_ms": infer_ms}
                print("infer_ms", infer_ms)
                if prev_total is not None:
                    payload["server_timing"]["prev_total_ms"] = prev_total * 1000.0

                await websocket.send(packer.pack(payload))
                prev_total = asyncio.get_running_loop().time() - start
        except websockets.ConnectionClosed:
            logger.info(f"Connection from {websocket.remote_address} closed")
        except Exception:
            tb = traceback.format_exc()
            try:
                await websocket.send(tb)
            finally:
                await websocket.close(code=websockets.frames.CloseCode.INTERNAL_ERROR,
                                      reason="Internal server error. Traceback included in previous frame.")
            raise


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None


# ----------------------------
# Entrypoint
# ----------------------------

def _make_metadata(adapter: _DiffusionWrapperAdapter) -> Dict[str, Any]:
    try:
        model_name = adapter._infer.model.__class__.__name__
    except Exception:
        model_name = "unknown"
    return {
        "model_name": model_name,
        "device": str(adapter.device),
        "protocol": "diffusion_policy.websocket.v1",
        "message": "Send nbatch-style obs -> receive {actions, server_timing}",
    }


def parse_cli() -> ServerConfig:
    import tyro
    return tyro.cli(ServerConfig)


def main():
    cfg = parse_cli()
    adapter = _DiffusionWrapperAdapter(cfg)
    meta = _make_metadata(adapter)

    server = WebsocketPolicyServer(adapter, cfg.host, cfg.port, meta)
    server.serve_forever()


if __name__ == "__main__":
    main()

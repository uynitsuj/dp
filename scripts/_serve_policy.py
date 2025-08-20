import asyncio
import http
import logging
import time
import traceback
from typing import Any, Dict

import torch
import dill
import hydra
from omegaconf import OmegaConf

from diffusion_policy.workspace.base_workspace import BaseWorkspace
import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames


# ----------------------------
# Config
# ----------------------------
CHECKPOINT = '/home/justinyu/diffusion_policy/data/outputs/2025.08.15/16.35.37_train_diffusion_scale_transformer_hybrid_soup_can_tabletop/checkpoints/latest.ckpt'
DEVICE = 'cuda:0'

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s:%(name)s: %(message)s'
)

# ----------------------------
# Policy loader & adapter
# ----------------------------
def _load_policy_from_ckpt(ckpt_path: str, device: str) -> Any:
    """Loads a diffusion-policy style checkpoint and returns the policy module on device in eval mode."""
    logger.info("Loading policy checkpoint...")
    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill, map_location='cpu')
    cfg = payload['cfg']
    # print(cfg)

    # Temporary override cfg
    # cfg_path = "/home/justinyu/diffusion_policy/data/outputs/2025.08.13/16.46.19_train_diffusion_transformer_hybrid_soup_can_tabletop/config.yaml"
    # cfg = OmegaConf.load(cfg_path)
    # print(cfg)
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=None)
    workspace: BaseWorkspace
    workspace.load_payload(payload)

    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    policy.to(torch.device(device))
    policy.eval()
    logger.info("Policy loaded successfully.")
    return policy


def _to_torch_tensor(x, device):
    """Best-effort conversion to torch.Tensor on target device (float32 default)."""
    if isinstance(x, torch.Tensor):
        return x.to(device)
    # numpy or list
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def _tree_to_torch(tree, device):
    if isinstance(tree, dict):
        return {k: _tree_to_torch(v, device) for k, v in tree.items()}
    if isinstance(tree, (list, tuple)):
        t = [_tree_to_torch(v, device) for v in tree]
        return type(tree)(t) if not isinstance(tree, list) else t
    return _to_torch_tensor(tree, device)


def _tree_to_numpy(tree):
    if isinstance(tree, dict):
        return {k: _tree_to_numpy(v) for k, v in tree.items()}
    if isinstance(tree, (list, tuple)):
        t = [_tree_to_numpy(v) for v in tree]
        return type(tree)(t) if not isinstance(tree, list) else t
    if isinstance(tree, torch.Tensor):
        return tree.detach().cpu().numpy()
    return tree


class _PredictActionAdapter:
    """
    Adapter that wraps a diffusion-policy `policy` exposing:
      - infer(obs: dict) -> dict with at least {"action": ...}
    Assumes the wrapped policy implements `predict_action(obs)` returning a dict.
    """

    def __init__(self, policy_module: Any, device: str):
        self._policy = policy_module
        self._device = torch.device(device)

    def infer(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """
        obs: nested dict/list/np arrays/numbers — already msgpack_numpy-unpacked.
        Returns: dict serializable by msgpack_numpy (numpy arrays / lists / scalars).
        """
        with torch.no_grad():
            torch_obs = _tree_to_torch(obs, self._device)
            out = self._policy.predict_action(torch_obs)  # expected to return dict with 'action'

            # Ensure outputs are numpy-serializable for msgpack_numpy
            out_np = _tree_to_numpy(out)

            # Replace key "action" with "actions" for consistency with PI0 policy ws interface
            out_np["actions"] = out_np["action"]
            del out_np["action"]
        return out_np


# ----------------------------
# Websocket server
# ----------------------------
class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol (msgpack-numpy).
    Handshake: first frame sent to client is metadata (dict).
    Then each recv frame is obs; reply is action dict + server_timing.
    """

    def __init__(
        self,
        policy: _PredictActionAdapter,
        host: str = "0.0.0.0",
        port: int | None = 8111,
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
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

        # Send metadata handshake
        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        try:
            while True:
                start_time = time.monotonic()
                obs = msgpack_numpy.unpackb(await websocket.recv())

                t0 = time.monotonic()
                action = self._policy.infer(obs)
                infer_time = time.monotonic() - t0

                # Attach timing
                action = dict(action)  # ensure mutable
                action["server_timing"] = {
                    "infer_ms": infer_time * 1000.0,
                }
                if prev_total_time is not None:
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000.0

                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time

        except websockets.ConnectionClosed:
            logger.info(f"Connection from {websocket.remote_address} closed")
        except Exception:
            # Send traceback to client (one text frame), then close with INTERNAL_ERROR.
            tb = traceback.format_exc()
            try:
                await websocket.send(tb)
            finally:
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
            raise


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None


# ----------------------------
# Main
# ----------------------------
def _make_metadata(policy, device: str) -> Dict[str, Any]:
    try:
        model_name = getattr(policy, "__class__", type(policy)).__name__
    except Exception:
        model_name = "unknown"
    return {
        "model_name": model_name,
        "device": device,
        "protocol": "diffusion_policy.websocket.v1",
        "message": "Welcome. First frame is metadata; subsequent frames are obs -> action.",
    }


def main():
    policy_module = _load_policy_from_ckpt(CHECKPOINT, DEVICE)
    adapter = _PredictActionAdapter(policy_module, DEVICE)
    meta = _make_metadata(policy_module, DEVICE)

    server = WebsocketPolicyServer(
        policy=adapter,
        host="0.0.0.0",
        port=8111,
        metadata=meta,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
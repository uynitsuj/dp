#!/usr/bin/env python3
import os
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm
import tyro

# LeRobot
from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset


def _discover_keys(dataset: LeRobotDataset,
                   fallback_proprio: str = "joint_position",
                   fallback_action: str = "actions",
                   fallback_videos: Optional[List[str]] = None
                   ) -> Tuple[str, str, List[str]]:
    """
    Discover proprio/action/video keys from dataset.features.
    """
    if fallback_videos is None:
        fallback_videos = ["left_camera-images-rgb", "right_camera-images-rgb", "top_camera-images-rgb"]

    features = getattr(dataset, "features", None)
    proprio_key = fallback_proprio
    action_key = fallback_action
    video_keys = fallback_videos

    if isinstance(features, dict) and features:
        try:
            vids = [k for k, v in features.items() if isinstance(v, dict) and v.get("dtype") == "video"]
            if vids:
                video_keys = vids

            if "joint_position" in features:
                proprio_key = "joint_position"
            elif "state" in features:
                proprio_key = "state"

            if "actions" in features:
                action_key = "actions"
            elif "action" in features:
                action_key = "action"
        except Exception:
            pass

    return proprio_key, action_key, video_keys


def _to_uint8_img(x: np.ndarray) -> np.ndarray:
    if x.dtype == np.uint8:
        return x
    arr = np.asarray(x)
    if arr.dtype in (np.float32, np.float64):
        if arr.max() <= 1.00001:
            arr = (arr * 255.0).clip(0, 255)
        return arr.astype(np.uint8)
    return arr.astype(np.uint8)


def _dump_episode_jpgs(out_dir: Path, video_frames: Dict[str, List[np.ndarray]]) -> None:
    for vkey, frames in video_frames.items():
        vdir = out_dir / vkey
        vdir.mkdir(parents=True, exist_ok=True)
        for i, frame in enumerate(frames):
            img = _to_uint8_img(frame)
            # CHW -> HWC if needed
            if img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[-1] not in (1, 3):
                img = np.transpose(img, (1, 2, 0))
            Image.fromarray(img).save(vdir / f"{i:06d}.jpg", quality=95, subsampling=1)


def _save_episode_h5(h5_path: Path, proprio: np.ndarray, action: np.ndarray, meta: Dict) -> None:
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("proprio", data=proprio, compression="gzip")
        f.create_dataset("action", data=action, compression="gzip")
        meta_grp = f.create_group("meta")
        for k, v in meta.items():
            if isinstance(v, (int, float, str, np.integer, np.floating)):
                meta_grp.attrs[str(k)] = v
            else:
                meta_grp.attrs[str(k)] = json.dumps(v)


def _episode_worker(repo_id: str,
                    ep_idx: int,
                    out_root_str: str,
                    parent_proprio_key: str,
                    parent_action_key: str,
                    parent_video_keys: List[str]) -> Dict:
    """
    Work on a single episode in a separate process:
      - Open LeRobotDataset(episodes=[ep_idx])
      - Discover keys
      - Export HDF5 + JPEGs
    Returns a small manifest entry.
    """
    out_root = Path(out_root_str)
    ep_out = out_root / f"episode_{ep_idx:06d}"
    ep_out.mkdir(parents=True, exist_ok=True)

    # Load dataset restricted to this episode (avoids global index math and decoder sharing).
    ds = LeRobotDataset(repo_id=repo_id, episodes=[ep_idx])

    proprio_key, action_key, video_keys = _discover_keys(
        ds, fallback_proprio=parent_proprio_key, fallback_action=parent_action_key, fallback_videos=parent_video_keys
    )

    proprio_list, action_list = [], []
    video_frames: Dict[str, List[np.ndarray]] = {k: [] for k in video_keys}

    for gidx in range(len(ds)):
        sample = ds[gidx]
        if proprio_key not in sample or action_key not in sample:
            raise KeyError(
                f"Episode {ep_idx}: missing keys in sample[{gidx}]. "
                f"Have: {list(sample.keys())}, need: '{proprio_key}', '{action_key}'"
            )

        proprio_list.append(np.asarray(sample[proprio_key]))
        action_list.append(np.asarray(sample[action_key]))

        for vkey in video_keys:
            if vkey in sample:
                video_frames[vkey].append(np.asarray(sample[vkey]))

    proprio_arr = np.stack(proprio_list, axis=0)
    action_arr = np.stack(action_list, axis=0)

    h5_path = ep_out / "episode.h5"
    meta = {
        "repo_id": repo_id,
        "episode_index": ep_idx,
        "num_frames": proprio_arr.shape[0],
        "proprio_key": proprio_key,
        "action_key": action_key,
        "video_keys": video_keys,
    }
    _save_episode_h5(h5_path, proprio_arr, action_arr, meta)
    _dump_episode_jpgs(ep_out, video_frames)

    return {
        "dir": f"episode_{ep_idx:06d}",
        "h5": "episode.h5",
        "videos": video_keys,
        "frames": int(proprio_arr.shape[0]),
    }

def convert_dataset_parallel(repo_id: str, output_dir: str, num_workers: int = 4) -> None:
    """
    Parallel, per-episode export to HDF5 + JPEGs.
    """
    # Probe once in the parent to list episodes & get fallback keys.
    probe = LeRobotDataset(repo_id=repo_id)
    proprio_key, action_key, video_keys = _discover_keys(probe)

    # Determine which episodes to export.
    if probe.episodes is not None:
        ep_indices = list(probe.episodes)
    else:
        # Prefer metadata count; fall back to scanning hf_dataset if needed.
        if hasattr(probe, "meta") and hasattr(probe.meta, "total_episodes"):
            ep_indices = list(range(probe.meta.total_episodes))
        else:
            # As a last resort, derive episode ids from hf_dataset column
            epi_col = [int(x.item()) for x in probe.hf_dataset["episode_index"]]
            seen = []
            last = None
            for e in epi_col:
                if e != last:
                    seen.append(e)
                    last = e
            ep_indices = seen

    out_root = Path(output_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Discovered keys:\n  proprio: {proprio_key}\n  action:  {action_key}\n  videos:  {video_keys}")
    print(f"Found {len(ep_indices)} episode(s).")
    print(f"Saving to: {out_root}")

    manifest_entries: List[Dict] = []
    with ProcessPoolExecutor(max_workers=max(1, int(num_workers))) as ex:
        futures = [
            ex.submit(
                _episode_worker,
                repo_id,
                ep,
                str(out_root),
                proprio_key,
                action_key,
                video_keys,
            )
            for ep in ep_indices
        ]

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Episodes", leave=True):
            manifest_entries.append(fut.result())

    manifest = {
        "repo_id": repo_id,
        "output_dir": str(out_root),
        "episodes": sorted(manifest_entries, key=lambda x: x["dir"]),
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("Done.")


@dataclass
class Args:
    repo_id: str = "uynitsuj/overfit_soup_can_data_20250818"
    output_dir: str = "data/lerobot2dp"
    num_workers: int = 0
    overwrite: bool = True


def main(args: Args):
    out_dir = Path(args.output_dir) / args.repo_id
    if out_dir.exists() and args.overwrite:
        print(f"Overwriting {out_dir}")
        import shutil
        shutil.rmtree(out_dir)
    convert_dataset_parallel(args.repo_id, out_dir, args.num_workers)


if __name__ == "__main__":
    tyro.cli(main)

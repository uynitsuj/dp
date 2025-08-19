#!/usr/bin/env python3
import json
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
import tyro
from datasets.features import features

# LeRobot
from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset, LeRobotDatasetMetadata
from PIL import Image
from tqdm import tqdm


def _discover_keys_from_meta(meta: LeRobotDatasetMetadata,
                             fallback_proprio: str = "joint_position",
                             fallback_action: str = "actions",
                             fallback_videos: Optional[List[str]] = None
                             ) -> Tuple[str, str, List[str]]:
    if fallback_videos is None:
        fallback_videos = ["left_camera-images-rgb", "right_camera-images-rgb", "top_camera-images-rgb"]

    features = getattr(meta, "features", {}) or {}
    proprio_key = fallback_proprio
    action_key = fallback_action
    video_keys = list(meta.video_keys) if hasattr(meta, "video_keys") else fallback_videos

    if isinstance(features, dict) and features:
        if "joint_position" in features:
            proprio_key = "joint_position"
        elif "state" in features:
            proprio_key = "state"

        if "actions" in features:
            action_key = "actions"
        elif "action" in features:
            action_key = "action"

        # prefer dtype=video from metadata
        vids = [k for k, v in features.items() if isinstance(v, dict) and v.get("dtype") == "video"]
        if vids:
            video_keys = vids

    return proprio_key, action_key, video_keys


def _read_episode_arrays(meta: LeRobotDatasetMetadata,
                         ep_idx: int,
                         proprio_key: str,
                         action_key: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read the parquet for a single episode into numpy arrays without touching video.
    """
    pq_rel = meta.get_data_file_path(ep_idx)        # relative path string
    pq_path = (meta.root / pq_rel).resolve()
    df = pd.read_parquet(pq_path, columns=[proprio_key, action_key])

    # Columns are lists/arrays per frame. Stack to (T, D)
    def _col_to_array(series: pd.Series) -> np.ndarray:
        vals = series.to_numpy()
        if isinstance(vals[0], (list, np.ndarray)):
            try:
                return np.vstack(vals)
            except Exception:
                return np.stack([np.asarray(v) for v in vals], axis=0)
        # Fallback for scalar columns (unlikely here)
        return vals.astype(np.float32)[:, None]

    proprio = _col_to_array(df[proprio_key])
    action = _col_to_array(df[action_key])
    return proprio, action


def _ffmpeg_extract_jpgs(video_path: Path, out_dir: Path) -> None:
    """
    Use ffmpeg to dump every frame to JPGs. Assumes the mp4 was encoded from episode frames at meta.fps.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    # -vsync 0 prevents frame duplication/dropping; -q:v 2 ~ high quality.
    cmd = [
        "ffmpeg",
        "-hide_banner", "-loglevel", "error",
        "-y",
        "-i", str(video_path),
        "-vsync", "0",
        "-start_number", "0",
        "-q:v", "1",
        "-pix_fmt", "yuvj444p",
        str(out_dir / "%06d.jpg"),
    ]
    subprocess.run(cmd, check=True)


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
    Fast worker: read parquet -> HDF5, and extract JPGs via ffmpeg from episode MP4s.
    """
    # t0 = time.time()
    out_root = Path(out_root_str)
    ep_out = out_root / f"episode_{ep_idx:06d}"
    ep_out.mkdir(parents=True, exist_ok=True)

    # Only metadata (fast, no video decode)
    meta = LeRobotDatasetMetadata(repo_id=repo_id)

    proprio_key, action_key, video_keys = _discover_keys_from_meta(
        meta,
        fallback_proprio=parent_proprio_key,
        fallback_action=parent_action_key,
        fallback_videos=parent_video_keys,
    )

    # Read arrays directly from parquet
    # t1 = time.time()
    proprio_arr, action_arr = _read_episode_arrays(meta, ep_idx, proprio_key, action_key)
    # t2 = time.time()

    # Save HDF5
    h5_path = ep_out / "episode.h5"
    h5_meta = {
        "repo_id": repo_id,
        "episode_index": ep_idx,
        "num_frames": int(proprio_arr.shape[0]),
        "proprio_key": proprio_key,
        "action_key": action_key,
        "video_keys": video_keys,
        "fps": meta.fps,
    }
    _save_episode_h5(h5_path, proprio_arr, action_arr, h5_meta)
    # t3 = time.time()

    # Extract JPGs with ffmpeg from each camera MP4 (if present)
    for vkey in video_keys:
        vid_rel = meta.get_video_file_path(ep_index=ep_idx, vid_key=vkey)
        vid_path = (meta.root / vid_rel).resolve()
        if vid_path.is_file():
            cam_dir = ep_out / vkey
            _ffmpeg_extract_jpgs(vid_path, cam_dir)
        # else: some datasets may omit certain cams for some episodes

    # t4 = time.time()
    # print(
    #     f"[ep {ep_idx:06d}] parquet->np: {t2 - t1:.3f}s | save h5: {t3 - t2:.3f}s | ffmpeg mp4_to_jpgs: {t4 - t3:.3f}s | total: {t4 - t0:.3f}s"
    # )

    return {
        "dir": f"episode_{ep_idx:06d}",
        "h5": "episode.h5",
        "videos": video_keys,
        "frames": int(proprio_arr.shape[0]),
    }


def convert_dataset_parallel(repo_id: str, output_dir: str | None = None, num_workers: int = 10) -> str:
    """
    Parallel, per-episode export to HDF5 + JPEGs.
    """
    # Probe once in the parent to list episodes & get fallback keys.
    # Monkey-patch to fix 'List' feature type error in old datasets
    try:
        _OLD_GENERATE_FROM_DICT = features.generate_from_dict

        def _new_generate_from_dict(obj):
            if isinstance(obj, dict) and obj.get("_type") == "List":
                obj["_type"] = "Sequence"
            return _OLD_GENERATE_FROM_DICT(obj)

        features.generate_from_dict = _new_generate_from_dict
    except (ImportError, AttributeError):
        # If datasets or the function doesn't exist, do nothing.
        pass
    # End of monkey-patch
    probe = LeRobotDataset(repo_id=repo_id, video_backend="pyav")
    proprio_key, action_key, video_keys = _discover_keys(probe)

    # Determine which episodes to export.
    if probe.episodes is not None:
        ep_indices = list(probe.episodes)
    # Prefer metadata count; fall back to scanning hf_dataset if needed.
    elif hasattr(probe, "meta") and hasattr(probe.meta, "total_episodes"):
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

    if output_dir is None:
        output_dir = HF_LEROBOT_HOME / repo_id / "dp_dataset"

    out_root = Path(output_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    if len(list(out_root.iterdir())) == len(ep_indices) + 1:
        for item in out_root.iterdir():
            if "manifest.json" in str(item):
                print(f"Lerobot dataset already converted to DP dataloader format at {out_root}")
                return out_root
        
        print(f"Dataset appears incompletely converted at {out_root}, continuing conversion")

    print(f"Discovered keys:\n  proprio: {proprio_key}\n  action:  {action_key}\n  videos:  {video_keys}")
    print(f"Found {len(ep_indices)} episode(s).")
    # print(f"Converting lerobot to DP dataloader format to path: {out_root}")

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
    # print("Done.")
    return str(out_root)


@dataclass
class Args:
    repo_id: str = "uynitsuj/soup_can_in_domain_xmi_data_center_cropped_20250818"
    output_dir: str = "data/lerobot2dp"
    num_workers: int = 10
    overwrite: bool = True


def main(args: Args):
    out_dir = Path(args.output_dir) / args.repo_id
    if out_dir.exists() and args.overwrite:
        print(f"Overwriting {out_dir}")
        shutil.rmtree(out_dir)
    convert_dataset_parallel(args.repo_id, out_dir, args.num_workers)


if __name__ == "__main__":
    tyro.cli(main)

"""
get_trajectories.py

Load trajectories from disk with multi-camera support and proprioceptive features.

MEMORY-EFFICIENT DESIGN (stream-and-extract):
    Phase 1: load_trajectory_meta() — loads only lightweight metadata per trajectory
             (image file paths, proprio, rewards, actions). No pixel data in RAM.
    Phase 2: precompute_features() — streams through trajectories one at a time:
             load images → resize to 224px → backbone forward pass → discard images.
             Uses I/O-GPU overlap: prefetches the next trajectory's images while
             the GPU processes the current one.
             Peak image memory: ONE trajectory at a time (~8 MB for 3 cams × 75 steps @ 224px).

    For 944 trajectories × 3 cameras this reduces peak RAM from ~24 GB to ~200 MB
    (feature vectors only: 944 × 75 × 6161 floats ≈ 1.6 GB vs 24 GB of raw images).

Key concepts:
- num_cameras: 1, 2, 3 for actual cameras; -1 for random baseline
- noise_model: How perception noise scales with cameras
- Ground truth rewards: Always computed with num_cameras=0 (no noise) for fair evaluation
"""

import os
import glob
import hashlib
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from tqdm import tqdm
import warnings
from imitation.data.types import TrajectoryWithRew
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from pathlib import Path

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

NUM_WORKERS = 32
JOINT_VEL_SCALE = 5
ACTION_SCALE = 3

# Camera configurations
# -1 = random baseline (uses front camera images but random rewards)
# 0 = ground truth (no noise) - used for evaluation only
# 1, 2, 3 = actual camera counts
CAMERA_CONFIGS = {
    -1: ["front"],  # Random baseline - still needs images
    1: ["front"],
    2: ["front", "side"],
    3: ["front", "side", "bird"],
}

BACKBONE_CONFIGS = {
    "resnet50": {"dim": 2048},
    "resnet34": {"dim": 512},
    "resnet18": {"dim": 512},
}


# ============================================================
# NOISE MODELS
# ============================================================

# def get_noise_std(num_cameras: int, alpha: float = 1.5, model: str = "tanh") -> float:
#     """
#     Calculate noise standard deviation based on cameras and noise model.
    
#     Args:
#         num_cameras: Number of cameras
#             -1 = random baseline (returns special flag)
#              0 = no noise (ground truth)
#              1, 2, 3 = actual cameras with noise scaling
#         alpha: Base noise level
#         model: Noise decay model ("tanh", "linear", "exponential", "none")
    
#     Returns:
#         Noise std, or -1 for random baseline
#     """
#     if num_cameras == -1:
#         return -1  # Special flag for completely random
    
#     if num_cameras == 0 or model == "none":
#         return 0.0  # Ground truth, no noise
    
#     if model == "tanh":
#         return alpha * (1.0 - np.tanh(num_cameras / 1.5))
#     elif model == "linear":
#         return max(0, alpha * (1.0 - num_cameras / 4.0))
#     elif model == "exponential":
#         return alpha * np.exp(-num_cameras / 2.0)
#     else:
#         raise ValueError(f"Unknown noise model: {model}")

def get_noise_std(num_cameras: int, alpha: float = 1.5, model: str = "tanh") -> float:
    """
    Calculate noise standard deviation based on cameras and noise model.
    
    Each model encodes a different hypothesis about WHY multi-view helps:
    
        tanh         α(1 − tanh(n/1.5))     Smooth saturation (original default)
        linear       α(1 − n/4)             Linear improvement, hits zero at n=4
        exponential  α exp(−n/2)            Rapid initial drop, long tail
        sqrt         α / √n                 Statistical averaging of independent
                                            estimates — the principled baseline.
                                            Each camera gives iid noisy obs,
                                            averaging n reduces std by 1/√n.
        occlusion    Sharp 1→2, plateau     Second camera resolves most self-
                                            occlusions (gripper blocking cube);
                                            third adds marginal benefit.
        constant     α                      Same noise regardless of n. Control
                                            condition: any multi-view benefit
                                            must come from feature diversity,
                                            not noise reduction.
        none         0                      Perfect perception (upper bound)
    
    Noise values at α=1.5:
        Model        n=1     n=2     n=3
        tanh         0.63    0.19    0.05
        sqrt         1.50    1.06    0.87
        occlusion    1.50    0.45    0.38
        constant     1.50    1.50    1.50
        linear       1.12    0.75    0.38
        exponential  0.91    0.55    0.33
    
    Args:
        num_cameras: Number of cameras
            -1 = random baseline (returns special flag)
             0 = no noise (ground truth)
             1, 2, 3 = actual cameras with noise scaling
        alpha: Base noise level
        model: Noise decay model
    
    Returns:
        Noise std, or -1 for random baseline
    """
    if num_cameras == -1:
        return -1  # Special flag for completely random
    
    if num_cameras == 0 or model == "none":
        return 0.0  # Ground truth, no noise
    
    n = num_cameras
    
    if model == "tanh":
        return alpha * (1.0 - np.tanh(n / 1.5))
    elif model == "linear":
        return max(0, alpha * (1.0 - n / 4.0))
    elif model == "exponential":
        return alpha * np.exp(-n / 2.0)
    elif model == "sqrt":
        return alpha / np.sqrt(n)
    elif model == "occlusion":
        # Big drop 1→2 (resolves most self-occlusion), diminishing after
        # At n=1: alpha, n=2: 0.3*alpha, n=3: 0.25*alpha
        return alpha * 0.3 ** (min(n, 2) - 1) * (0.85 ** max(0, n - 2))
    elif model == "constant":
        return alpha
    else:
        raise ValueError(f"Unknown noise model: {model}")


def calculate_reward(
    log_data: dict, 
    start_idx: int, 
    end_idx: int, 
    num_cameras: int = 0,
    alpha: float = 1.5,
    noise_model: str = "tanh"
) -> np.ndarray:
    """
    Calculate rewards with configurable noise based on perception quality.
    
    Args:
        num_cameras: 
            -1 = truly random rewards (baseline) — iid normal so trajectory 
                 return comparisons are ~50/50 coin flips
             0 = ground truth (no noise) - for evaluation
             1, 2, 3 = noisy perception based on camera count
    """
    T_len = end_idx - start_idx
    
    noise_std = get_noise_std(num_cameras, alpha, noise_model)
    
    if noise_std == -1:
        # Truly random rewards: iid standard normal per timestep.
        # When comparing trajectory return sums, P(sum_A > sum_B) ≈ 0.5
        # by CLT, giving a genuine 50% random baseline for preferences.
        #
        # Previous approach generated random *observations* and fed them
        # through the reward function, which produced structured (non-random)
        # rewards because the reward function maps even random inputs to a
        # specific distribution — NOT a fair coin flip for preferences.
        return np.random.standard_normal(T_len).astype(np.float32)
    
    object_pos = log_data['object_position'][start_idx:end_idx]
    ee_pos = log_data['ee_position'][start_idx:end_idx]
    goal_pos = log_data['goal_position'][start_idx:end_idx]
    joint_vel = log_data['joint_velocities'][start_idx:end_idx]
    action_diff = log_data['action_difference'][start_idx:end_idx]

    if noise_std == 0:
        # Perfect perception (ground truth)
        noisy_ee_pos = ee_pos
        noisy_obj_pos = object_pos
        noisy_joint_vel = joint_vel
        noisy_action_diff = action_diff
    else:
        # Noisy perception based on camera count
        noisy_ee_pos = ee_pos + np.random.standard_normal(ee_pos.shape) * noise_std
        noisy_obj_pos = object_pos + np.random.standard_normal(object_pos.shape) * noise_std
        noisy_joint_vel = joint_vel + np.random.standard_normal(joint_vel.shape) * noise_std * JOINT_VEL_SCALE
        noisy_action_diff = action_diff + np.random.standard_normal(action_diff.shape) * noise_std * ACTION_SCALE

    # Reward calculation (same formula regardless of noise)
    decimation = 2
    dt = 1/60 * decimation
    
    object_ee_dist = np.linalg.norm(noisy_obj_pos - noisy_ee_pos, axis=1)
    reward_reaching = (1.0 - np.tanh(object_ee_dist / 0.1)) * 1.0 * dt
    
    is_lifted = (noisy_obj_pos[:, 2] > 0.04)
    reward_lifting = is_lifted * 15.0 * dt
    
    goal_dist = np.linalg.norm(goal_pos - noisy_obj_pos, axis=1)
    reward_goal_tracking = is_lifted * (1.0 - np.tanh(goal_dist / 0.3)) * 16.0 * dt
    reward_goal_tracking_fine = is_lifted * (1.0 - np.tanh(goal_dist / 0.05)) * 5.0 * dt
    
    reward_action_rate = np.sum(np.square(noisy_action_diff), axis=1) * -1e-2 * dt
    reward_joint_vel = np.sum(np.square(noisy_joint_vel), axis=1) * -1e-2 * dt

    total_reward = (
        reward_reaching + reward_lifting + reward_goal_tracking +
        reward_goal_tracking_fine + reward_action_rate + reward_joint_vel
    )
    
    return total_reward.astype(np.float32)


def calculate_ground_truth_reward(log_data: dict, start_idx: int, end_idx: int) -> np.ndarray:
    """
    Calculate ground truth rewards (no noise) for evaluation.
    This ensures all models are evaluated against the same target.
    """
    return calculate_reward(log_data, start_idx, end_idx, num_cameras=0, alpha=0, noise_model="none")


# ============================================================
# LIGHTWEIGHT TRAJECTORY METADATA (no pixel data)
# ============================================================

@dataclass
class TrajectoryMeta:
    """
    Lightweight trajectory metadata — holds file paths instead of pixel arrays.
    
    This is the Phase 1 output. Images are loaded on-demand during feature 
    extraction (Phase 2), one trajectory at a time, to avoid holding all 
    images in RAM simultaneously.
    
    Memory per trajectory: ~50 KB (paths + arrays) vs ~25 MB (with images)
    Total for 944 trajectories: ~47 MB vs ~24 GB
    """
    image_paths: Dict[str, List[str]]   # {camera_name: [path1, path2, ...]}
    obs_proprio: np.ndarray             # (T, proprio_dim) float32
    acts: np.ndarray                    # (T, action_dim) float32
    rews: np.ndarray                    # (T,) float32 - NOISY rewards for training
    rews_ground_truth: np.ndarray       # (T,) float32 - Ground truth for evaluation
    terminal: bool = True
    traj_path: str = ""
    
    @property
    def T(self) -> int:
        """Trajectory length."""
        return len(self.acts)
    
    @property
    def camera_names(self) -> List[str]:
        return list(self.image_paths.keys())
    
    @property 
    def num_cameras(self) -> int:
        return len(self.image_paths)


# Keep RawTrajectory for backward compatibility, but it should not be used
# in the main pipeline anymore.
@dataclass
class RawTrajectory:
    """DEPRECATED: Use TrajectoryMeta + streaming feature extraction instead.
    
    Kept for backward compatibility only. Holds raw images in memory which
    causes ~24 GB peak RAM for 944 trajectories × 3 cameras.
    """
    obs_images: np.ndarray          # (T, C*num_cams, H, W) uint8
    obs_proprio: np.ndarray         # (T, proprio_dim) float32
    acts: np.ndarray                # (T, action_dim) float32
    rews: np.ndarray                # (T,) float32 - NOISY rewards for training
    rews_ground_truth: np.ndarray   # (T,) float32 - Ground truth for evaluation
    terminal: bool = True
    path: str = ""


# ============================================================
# TRAJECTORY LOADING (Phase 1: metadata only)
# ============================================================

def load_trajectory_meta(
    traj_path: str, 
    camera_names: List[str] = ["front"],
    alpha: float = 1.5,
    noise_model: str = "tanh",
    num_cameras_for_reward: int = 1,
    min_length: int = 75,
) -> Optional[TrajectoryMeta]:
    """
    Load trajectory metadata WITHOUT loading any images.
    
    Returns lightweight TrajectoryMeta with image file paths, proprio, 
    rewards, and actions. Images are loaded later during feature extraction.
    
    Computes TWO sets of rewards:
    1. Noisy rewards (based on num_cameras_for_reward) - used for training
    2. Ground truth rewards (no noise) - used for evaluation
    """
    try:
        # Load log data
        npz_files = glob.glob(os.path.join(traj_path, "*.npz"))
        if not npz_files:
            return None
        
        try:
            log_data = np.load(npz_files[0], allow_pickle=True)
        except Exception:
            return None

        if 'current_action' not in log_data:
            return None
        raw_actions = log_data['current_action']

        # Validate image directories and collect file paths (no loading)
        def get_img_files(cam):
            d = os.path.join(traj_path, "rgb", cam)
            if not os.path.exists(d):
                return None
            return sorted(glob.glob(os.path.join(d, "*.jpg")))

        cam_files_map = {}
        for cam in camera_names:
            files = get_img_files(cam)
            if files is None and cam == "bird":
                files = get_img_files("bird_view")
            if files is None:
                return None
            cam_files_map[cam] = files

        # Determine trajectory length
        n_acts = len(raw_actions)
        expected_imgs = n_acts

        # Check if all cameras have exactly n_acts images (corrupt file check)
        if any(len(f) != expected_imgs for f in cam_files_map.values()):
            return None  # Corrupt file: inconsistent image count
            
        limit = expected_imgs

        if limit < min_length:
            return None

        # Extract valid segment (skip first action for alignment)
        # Actions, rewards all start from index 1; images/proprio keep index 0
        # so that obs has T+1 entries for T transitions (imitation lib convention).
        valid_actions = raw_actions[1:].astype(np.float32)

        # NOISY rewards for training  (length = n_acts - 1, matches acts)
        valid_rewards_noisy = calculate_reward(
            log_data, 1, limit,
            num_cameras=num_cameras_for_reward,
            alpha=alpha,
            noise_model=noise_model,
        )
        
        # GROUND TRUTH rewards for evaluation  (length = n_acts - 1, matches acts)
        valid_rewards_gt = calculate_ground_truth_reward(log_data, 1, limit)
        
        T_len =  expected_imgs

        # Load proprioceptive features (small — always kept in memory)
        joint_pos = log_data['obs_joint_pos'].astype(np.float32)
        joint_vel = log_data['obs_joint_vel'].astype(np.float32)
        target_pos = log_data['obs_target_object_position'].astype(np.float32)
        obs_proprio = np.concatenate([joint_pos, joint_vel, target_pos], axis=1)
        
        # Verify shapes and truncate if needed
        if len(obs_proprio) != T_len:
            return None  # Corrupt file: proprio length mismatch

        return TrajectoryMeta(
            image_paths=cam_files_map,
            obs_proprio=np.ascontiguousarray(obs_proprio),
            acts=np.ascontiguousarray(valid_actions),
            rews=np.ascontiguousarray(valid_rewards_noisy),
            rews_ground_truth=np.ascontiguousarray(valid_rewards_gt),
            terminal=True,
            traj_path=traj_path,
        )

    except Exception as e:
        # Uncomment for debugging:
        print(f"Error loading trajectory at {traj_path}: {e}")
        return None


def get_trajectories(
    root_dir: str,
    camera_names: List[str] = ["front"],
    alpha: float = 1.5,
    noise_model: str = "tanh",
    num_cameras: int = 1,
    num_workers: int = NUM_WORKERS,
    max_trajectories: Optional[int] = None,
) -> List[TrajectoryMeta]:
    """
    Load all trajectory metadata from directory (Phase 1).
    
    Returns lightweight TrajectoryMeta objects — no images in memory.
    Images are loaded on-demand during precompute_features() (Phase 2).
    
    Args:
        camera_names: Which camera images to load
        num_cameras: Camera count for reward noise calculation (-1, 1, 2, 3)
        alpha: Base noise level
        noise_model: How noise scales ("tanh", "linear", "none", etc.)
    """
    print(f"Scanning directory: {root_dir}")
    print(f"Camera images: {camera_names}")
    print(f"Noise: num_cameras={num_cameras}, alpha={alpha}, model={noise_model}")
    
    # Find trajectory folders
    search_path = os.path.join(root_dir, "*", "env_*", "traj_*")
    traj_folders = sorted(glob.glob(search_path))
    
    if len(traj_folders) == 0:
        search_path = os.path.join(root_dir, "env_*", "traj_*")
        traj_folders = sorted(glob.glob(search_path))

    # Skip traj_000 (often incomplete)
    traj_folders = [f for f in traj_folders if not f.endswith("traj_000")]
    
    # if max_trajectories > 0:
    #     traj_folders = traj_folders[:max_trajectories * 2]

    print(f"Found {len(traj_folders)} potential trajectories.")
    if len(traj_folders) == 0:
        return []

    print(f"Loading metadata with {num_workers} workers (no images loaded)...")

    all_trajectories = []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(
                load_trajectory_meta, 
                path, 
                camera_names, 
                alpha, 
                noise_model,
                num_cameras,
            ): path for path in traj_folders
        }
        
        for future in tqdm(as_completed(futures), total=len(traj_folders), desc="Loading metadata"):
            result = future.result()
            if result is not None:
                all_trajectories.append(result)
                if max_trajectories and len(all_trajectories) >= max_trajectories:
                    break
    
    print(f"Successfully loaded {len(all_trajectories)} trajectory metadata.")
    
    if all_trajectories:
        sample = all_trajectories[0]
        print(f"  Cameras: {sample.camera_names} ({sample.num_cameras} views)")
        print(f"  Trajectory length: {sample.T}")
        print(f"  Proprio shape: {sample.obs_proprio.shape}")
        print(f"  Action shape: {sample.acts.shape}")
        print(f"  Noisy reward range: [{sample.rews.min():.4f}, {sample.rews.max():.4f}]")
        print(f"  GT reward range: [{sample.rews_ground_truth.min():.4f}, {sample.rews_ground_truth.max():.4f}]")
        
        # Estimate memory saved
        n = len(all_trajectories)
        avg_T = np.mean([t.T for t in all_trajectories])
        n_cams = sample.num_cameras
        img_mem_gb = n * avg_T * n_cams * 3 * 224 * 126 / 1e9  # uint8 @ 224×126 (16:9)
        print(f"  Memory saved by not loading images: ~{img_mem_gb:.1f} GB")
    
    return all_trajectories


# ============================================================
# FEATURE EXTRACTION (Phase 2: stream-and-extract)
# ============================================================

def get_backbone(name: str = "resnet50", device: str = "cuda"):
    """Load pretrained backbone for feature extraction."""
    if name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        feat_dim = 2048
    elif name == "resnet34":
        model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        feat_dim = 512
    elif name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        feat_dim = 512
    else:
        raise ValueError(f"Unknown backbone: {name}")
    
    model.fc = nn.Identity()
    model.to(device)
    model.eval()
    
    return model, feat_dim


@dataclass
class ProcessedTrajectory:
    """Trajectory with extracted features, for training and evaluation."""
    obs: np.ndarray                 # (T, obs_dim) combined visual + proprio features
    acts: np.ndarray                # (T, action_dim)
    rews: np.ndarray                # (T,) NOISY rewards - used for training
    rews_ground_truth: np.ndarray   # (T,) Ground truth rewards - used for evaluation
    terminal: bool = True


def _load_images_single_trajectory(
    traj_meta: TrajectoryMeta,
    target_width: int = 224,
    executor: Optional[ThreadPoolExecutor] = None,
) -> np.ndarray:
    """
    Load and resize images for a single trajectory from disk using threads.

    Returns:
        (T, 3 * num_cameras, H, W) uint8 array
    """
    def load_and_resize_image(path):
        with Image.open(path) as im:
            w, h = im.size
            new_h = int(h * (target_width / w))
            im_resized = im.resize((target_width, new_h), Image.BICUBIC)
            return np.array(im_resized)

    all_cam_images = []
    owns_executor = executor is None
    if owns_executor:
        executor = ThreadPoolExecutor(max_workers=NUM_WORKERS)

    try:
        for cam_name in traj_meta.camera_names:
            paths = traj_meta.image_paths[cam_name]
            imgs_list = list(executor.map(load_and_resize_image, paths))

            cam_obs = np.array(imgs_list, dtype=np.uint8)
            cam_obs = cam_obs.transpose(0, 3, 1, 2)  # (T, 3, H, W)
            all_cam_images.append(cam_obs)
    finally:
        if owns_executor:
            executor.shutdown(wait=False)

    # Concatenate along channel dim: (T, 3*num_cams, H, W)
    return np.concatenate(all_cam_images, axis=1)


def _extract_features_single_trajectory(
    obs_images: np.ndarray,
    backbone: nn.Module,
    normalize: T.Normalize,
    device: str,
    batch_size: int = 64,
) -> np.ndarray:
    """
    Run backbone on one trajectory's images and return feature vectors.
    
    Args:
        obs_images: (T, 3*num_cams, H, W) uint8
        
    Returns:
        (T, num_cams * feat_dim) float32 numpy array
    """
    T_len, C_total, H, W = obs_images.shape
    num_cams = C_total // 3

    # Convert to float tensor, reshape for per-camera processing
    obs_tensor = torch.from_numpy(obs_images).to(device).float() / 255.0
    obs_tensor = obs_tensor.reshape(T_len * num_cams, 3, H, W)  # Use reshape instead of view
    obs_tensor = normalize(obs_tensor)
    
    # Batch through backbone
    visual_features_list = []
    for b in range(0, len(obs_tensor), batch_size):
        batch = obs_tensor[b:b + batch_size]
        features = backbone(batch)
        visual_features_list.append(features.cpu())
    
    visual_features = torch.cat(visual_features_list, dim=0).numpy()
    
    # Reshape: (T * num_cams, feat_dim) -> (T, num_cams * feat_dim)
    feat_dim = visual_features.shape[1]
    visual_features = visual_features.reshape(T_len, num_cams * feat_dim)
    
    return visual_features


def precompute_features(
    trajectories: List[TrajectoryMeta],
    device: str = "cuda",
    backbone_name: str = "resnet50",
    batch_size: int = 64,
    include_proprio: bool = True,
) -> Tuple[List[TrajectoryWithRew], List[ProcessedTrajectory], int]:
    """
    Stream images one trajectory at a time, extract features, discard images.

    Peak image memory: ONE trajectory (~8 MB at 224px for 3 cams × 75 steps)
    rather than ALL trajectories (~24 GB at 512px).

    Uses a prefetch thread to overlap I/O with GPU feature extraction.

    Returns:
        train_trajectories: List of TrajectoryWithRew with NOISY rewards (for imitation library)
        eval_trajectories: List of ProcessedTrajectory with GROUND TRUTH rewards (for evaluation)
        obs_dim: Total observation dimension
    """
    print("\n" + "=" * 60)
    print(f"STREAMING FEATURE EXTRACTION (backbone={backbone_name})")
    print("=" * 60)

    # Load backbone once
    backbone, visual_feat_dim = get_backbone(backbone_name, device)

    normalize = T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    train_trajectories = []
    eval_trajectories = []
    proprio_dim = trajectories[0].obs_proprio.shape[1] if include_proprio else 0
    num_cams = trajectories[0].num_cameras

    # Shared thread pool for all image loading (avoids per-trajectory spin-up)
    io_pool = ThreadPoolExecutor(max_workers=NUM_WORKERS)

    # Prefetch: submit the first trajectory's images immediately
    prefetch_future: Optional[Future] = io_pool.submit(
        _load_images_single_trajectory, trajectories[0], 224, io_pool,
    )

    with torch.no_grad():
        for idx, traj in enumerate(tqdm(trajectories, desc="Extracting features")):
            # Collect prefetched images (blocks until I/O done)
            obs_images = prefetch_future.result()

            # Kick off prefetch for the NEXT trajectory while GPU works
            if idx + 1 < len(trajectories):
                prefetch_future = io_pool.submit(
                    _load_images_single_trajectory,
                    trajectories[idx + 1], 224, io_pool,
                )
            else:
                prefetch_future = None

            # --- Extract features on GPU ---
            visual_features = _extract_features_single_trajectory(
                obs_images, backbone, normalize, device, batch_size,
            )

            # Free images immediately — only features survive
            del obs_images

            # --- Combine with proprio ---
            if include_proprio:
                combined_obs = np.concatenate([visual_features, traj.obs_proprio], axis=1)
            else:
                combined_obs = visual_features

            combined_obs = combined_obs.astype(np.float32)

            # TrajectoryWithRew for imitation library (noisy rewards)
            train_traj = TrajectoryWithRew(
                obs=combined_obs,
                acts=traj.acts,
                rews=traj.rews,
                infos=None,
                terminal=traj.terminal,
            )
            train_trajectories.append(train_traj)

            # ProcessedTrajectory for evaluation (ground truth rewards)
            eval_traj = ProcessedTrajectory(
                obs=combined_obs,
                acts=traj.acts,
                rews=traj.rews,
                rews_ground_truth=traj.rews_ground_truth,
                terminal=traj.terminal,
            )
            eval_trajectories.append(eval_traj)

    io_pool.shutdown(wait=False)

    total_obs_dim = num_cams * visual_feat_dim + proprio_dim

    print(f"Feature extraction complete.")
    print(f"  Visual features: {num_cams} cams × {visual_feat_dim} = {num_cams * visual_feat_dim}")
    print(f"  Proprio features: {proprio_dim}")
    print(f"  Total obs dim: {total_obs_dim}")

    # Estimate actual memory usage of feature vectors
    total_steps = sum(t.obs.shape[0] for t in train_trajectories)
    feat_mem_mb = total_steps * total_obs_dim * 4 / 1e6  # float32
    print(f"  Feature memory: {total_steps} steps × {total_obs_dim} dims = {feat_mem_mb:.0f} MB")

    # Clean up
    del backbone
    torch.cuda.empty_cache()

    return train_trajectories, eval_trajectories, total_obs_dim


# ============================================================
# TRAIN/TEST SPLIT
# ============================================================

def train_test_split(
    trajectories: List,
    eval_trajectories: Optional[List] = None,
    train_frac: float = 0.8,
    seed: int = 42,
) -> Tuple:
    """
    Split trajectories into train and test sets.
    
    If eval_trajectories is provided, splits both in parallel (same indices).
    
    Returns:
        If eval_trajectories is None:
            (train_trajs, test_trajs)
        Else:
            (train_trajs, test_trajs, train_eval_trajs, test_eval_trajs)
    """
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(trajectories))
    
    split_idx = int(train_frac * len(trajectories))
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    train_trajs = [trajectories[i] for i in train_indices]
    test_trajs = [trajectories[i] for i in test_indices]
    
    print(f"Train/test split: {len(train_trajs)} / {len(test_trajs)}")
    
    if eval_trajectories is not None:
        train_eval = [eval_trajectories[i] for i in train_indices]
        test_eval = [eval_trajectories[i] for i in test_indices]
        return train_trajs, test_trajs, train_eval, test_eval
    
    return train_trajs, test_trajs


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def get_obs_dim(num_cameras: int, backbone: str = "resnet50", include_proprio: bool = True) -> int:
    """Calculate expected observation dimension."""
    visual_dim = num_cameras * BACKBONE_CONFIGS[backbone]["dim"]
    proprio_dim = 7 + 7 + 3  # joint_pos + joint_vel + target_pos (adjust if different)
    
    if include_proprio:
        return visual_dim + proprio_dim
    return visual_dim


def get_action_dim(sample_trajectory) -> int:
    """Get action dimension from a sample trajectory."""
    return sample_trajectory.acts.shape[1]
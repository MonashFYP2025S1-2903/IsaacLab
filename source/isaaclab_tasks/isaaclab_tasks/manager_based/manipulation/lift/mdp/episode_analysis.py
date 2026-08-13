"""
episode_analysis.py

Preference-based reward learning with systematic experimentation support.
Supports multiple architectures, proper evaluation, and organized output.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import os
import sys
import json
import warnings
import time
import argparse
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any, Tuple

from gymnasium import spaces
from imitation.data.types import TrajectoryWithRew
from imitation.algorithms import preference_comparisons
from imitation.algorithms.preference_comparisons import TrajectoryDataset
from imitation.rewards.reward_nets import RewardNet
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

from get_trajectories import (
    get_trajectories,
    precompute_features,
    train_test_split,
    ProcessedTrajectory,
    CAMERA_CONFIGS,
    BACKBONE_CONFIGS,
)

# Suppress warnings
os.environ['GYM_IGNORE_DEPRECATION_WARNINGS'] = '1'
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*")


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class ExperimentConfig:
    """
    All experiment parameters.
    
    Key parameters:
        seed: Random seed for reproducibility. Run with multiple seeds (0,1,2)
              to get mean ± std for error bars in your paper.
        
        num_cameras: -1 = random baseline (tests if learning happens at all)
                      1, 2, 3 = actual camera counts
        
        noise_model: "none" = perfect perception (upper bound)
                     "tanh" = realistic noise scaling
    """
    # Data
    num_cameras: int = 1          # -1, 1, 2, or 3
    alpha: float = 1.5            # Base noise level
    noise_model: str = "tanh"     # "none", "tanh", "linear", "exponential"
    sample_trajectories: int = 2000  # Max available
    train_frac: float = 0.8
    
    # Architecture
    architecture: str = "baseline"
    backbone: str = "resnet50"
    hid_sizes: Tuple[int, ...] = (256, 128)
    fusion: str = "concat"  # For per-camera architectures
    include_proprio: bool = True
    
    # Training
    total_comparisons: int = 2000  # Max for your setup
    fragment_length: int = 75
    epochs: int = 8
    batch_size: int = 32
    num_iterations: int = 8
    initial_comparison_frac: float = 0.4
    
    # Meta
    seed: int = 42
    device: str = "cuda"
    
    def to_dict(self) -> dict:
        d = asdict(self)
        d['hid_sizes'] = list(d['hid_sizes'])
        return d
    
    @classmethod
    def from_dict(cls, d: dict) -> 'ExperimentConfig':
        d = d.copy()
        if 'hid_sizes' in d:
            d['hid_sizes'] = tuple(d['hid_sizes'])
        return cls(**d)
    
    def experiment_name(self) -> str:
        """Generate unique experiment name."""
        cam_str = "random" if self.num_cameras == -1 else f"cam{self.num_cameras}"
        return f"{self.architecture}_{cam_str}_alpha{self.alpha}_{self.noise_model}_comp{self.total_comparisons}_seed{self.seed}"


@dataclass
class ExperimentResults:
    """Evaluation metrics."""
    # Primary metrics
    test_preference_accuracy: float
    test_return_spearman: float
    
    # Theoretical ceiling: how often noisy preferences agree with oracle
    # This measures the quality of the training signal, NOT the model.
    theoretical_preference_accuracy: float
    
    # Normalized score: how much of the available signal the model captures
    # = (model_acc - chance) / (theoretical_acc - chance)
    # 1.0 = model perfectly captures all signal in the noisy preferences
    # 0.0 = model is at chance level
    # >1.0 = model somehow exceeds theoretical ceiling (unlikely, but possible with lucky eval)
    normalized_preference_score: float
    
    # Secondary metrics
    test_timestep_pearson: float
    test_timestep_spearman: float
    
    # Training info
    training_time_seconds: float
    total_train_trajectories: int
    total_test_trajectories: int
    obs_dim: int
    action_dim: int
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @property
    def primary_score(self) -> float:
        """Combined score for ranking models."""
        return 0.6 * self.test_preference_accuracy + 0.4 * self.test_return_spearman


# ============================================================
# REWARD NETWORK ARCHITECTURES
# ============================================================

class MLPRewardNet(RewardNet):
    """Baseline MLP reward network."""
    
    def __init__(self, observation_space, action_space, hid_sizes=(256, 128), **kwargs):
        super().__init__(observation_space, action_space, normalize_images=False)
        
        obs_dim = int(np.prod(observation_space.shape))
        action_dim = int(np.prod(action_space.shape))
        input_dim = obs_dim + action_dim
        
        layers = []
        curr_dim = input_dim
        for hid in hid_sizes:
            layers.append(nn.Linear(curr_dim, hid))
            layers.append(nn.LeakyReLU(0.01))
            layers.append(nn.Dropout(0.1))
            curr_dim = hid
        
        layers.append(nn.Linear(curr_dim, 1))
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, state, action, next_state, done):
        x = torch.cat([state, action], dim=1)
        return self.mlp(x).squeeze(-1)


class BoundedRewardNet(RewardNet):
    """MLP with bounded output (tanh or sigmoid)."""
    
    def __init__(self, observation_space, action_space, 
                 hid_sizes=(256, 128),
                 output_activation="tanh",
                 **kwargs):
        super().__init__(observation_space, action_space, normalize_images=False)
        
        obs_dim = int(np.prod(observation_space.shape))
        action_dim = int(np.prod(action_space.shape))
        input_dim = obs_dim + action_dim
        
        layers = []
        curr_dim = input_dim
        for hid in hid_sizes:
            layers.append(nn.Linear(curr_dim, hid))
            layers.append(nn.LayerNorm(hid))
            layers.append(nn.LeakyReLU(0.01))
            layers.append(nn.Dropout(0.1))
            curr_dim = hid
        
        layers.append(nn.Linear(curr_dim, 1))
        
        if output_activation == "tanh":
            layers.append(nn.Tanh())
        elif output_activation == "sigmoid":
            layers.append(nn.Sigmoid())
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, state, action, next_state, done):
        x = torch.cat([state, action], dim=1)
        return self.mlp(x).squeeze(-1)


class LargerMLPRewardNet(RewardNet):
    """Wider network for high-dimensional inputs."""
    
    def __init__(self, observation_space, action_space,
                 hid_sizes=(512, 256, 128),
                 dropout=0.2,
                 **kwargs):
        super().__init__(observation_space, action_space, normalize_images=False)
        
        obs_dim = int(np.prod(observation_space.shape))
        action_dim = int(np.prod(action_space.shape))
        input_dim = obs_dim + action_dim
        
        layers = []
        curr_dim = input_dim
        for hid in hid_sizes:
            layers.append(nn.Linear(curr_dim, hid))
            layers.append(nn.LayerNorm(hid))
            layers.append(nn.LeakyReLU(0.01))
            layers.append(nn.Dropout(dropout))
            curr_dim = hid
        
        layers.append(nn.Linear(curr_dim, 1))
        layers.append(nn.Tanh())
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, state, action, next_state, done):
        x = torch.cat([state, action], dim=1)
        return self.mlp(x).squeeze(-1)


class PerCameraRewardNet(RewardNet):
    """
    Process each camera separately then fuse.
    Better for multi-camera setups.
    """
    
    def __init__(self, observation_space, action_space,
                 num_cameras: int = 1,
                 visual_feat_dim: int = 2048,
                 proprio_dim: int = 17,
                 camera_embed_dim: int = 128,
                 fusion: str = "concat",
                 share_weights: bool = True,
                 **kwargs):
        super().__init__(observation_space, action_space, normalize_images=False)
        
        self.num_cameras = num_cameras
        self.visual_feat_dim = visual_feat_dim
        self.proprio_dim = proprio_dim
        self.fusion = fusion
        
        action_dim = int(np.prod(action_space.shape))
        
        # Per-camera encoder
        if share_weights:
            self.camera_encoder = nn.Sequential(
                nn.Linear(visual_feat_dim, 512),
                nn.LayerNorm(512),
                nn.LeakyReLU(0.01),
                nn.Dropout(0.1),
                nn.Linear(512, camera_embed_dim),
                nn.LayerNorm(camera_embed_dim),
                nn.LeakyReLU(0.01),
            )
            self.camera_encoders = None
        else:
            self.camera_encoder = None
            self.camera_encoders = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(visual_feat_dim, 512),
                    nn.LayerNorm(512),
                    nn.LeakyReLU(0.01),
                    nn.Dropout(0.1),
                    nn.Linear(512, camera_embed_dim),
                    nn.LayerNorm(camera_embed_dim),
                    nn.LeakyReLU(0.01),
                )
                for _ in range(num_cameras)
            ])
        
        # Attention for fusion
        if fusion == "attention":
            self.attention = nn.Sequential(
                nn.Linear(camera_embed_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 1)
            )
        
        # Fusion dimension
        if fusion == "concat":
            fused_dim = camera_embed_dim * num_cameras
        else:
            fused_dim = camera_embed_dim
        
        # Proprio encoder
        self.proprio_encoder = nn.Sequential(
            nn.Linear(proprio_dim, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.01),
        )
        
        # Final MLP
        final_input_dim = fused_dim + 64 + action_dim
        self.reward_mlp = nn.Sequential(
            nn.Linear(final_input_dim, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 1),
            nn.Tanh()
        )
    
    def forward(self, state, action, next_state, done):
        batch_size = state.shape[0]
        
        # Split state into visual and proprio
        visual_dim = self.num_cameras * self.visual_feat_dim
        visual_features = state[:, :visual_dim]
        proprio_features = state[:, visual_dim:]
        
        # Split visual features by camera
        camera_embeddings = []
        for c in range(self.num_cameras):
            cam_feat = visual_features[:, c * self.visual_feat_dim:(c + 1) * self.visual_feat_dim]
            
            if self.camera_encoder is not None:
                emb = self.camera_encoder(cam_feat)
            else:
                emb = self.camera_encoders[c](cam_feat)
            camera_embeddings.append(emb)
        
        # Stack: (B, num_cams, embed_dim)
        stacked = torch.stack(camera_embeddings, dim=1)
        
        # Fusion
        if self.fusion == "concat":
            fused = stacked.view(batch_size, -1)
        elif self.fusion == "mean":
            fused = stacked.mean(dim=1)
        elif self.fusion == "max":
            fused = stacked.max(dim=1)[0]
        elif self.fusion == "attention":
            attn_scores = self.attention(stacked).squeeze(-1)
            attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(-1)
            fused = (stacked * attn_weights).sum(dim=1)
        else:
            fused = stacked.mean(dim=1)
        
        # Encode proprio
        proprio_emb = self.proprio_encoder(proprio_features)
        
        # Combine and predict reward
        combined = torch.cat([fused, proprio_emb, action], dim=1)
        return self.reward_mlp(combined).squeeze(-1)


class BottleneckRewardNet(RewardNet):
    """Compress high-dim input through bottleneck."""
    
    def __init__(self, observation_space, action_space,
                 bottleneck_dim: int = 256,
                 **kwargs):
        super().__init__(observation_space, action_space, normalize_images=False)
        
        obs_dim = int(np.prod(observation_space.shape))
        action_dim = int(np.prod(action_space.shape))
        
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.2),
            nn.Linear(512, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.LeakyReLU(0.01),
        )
        
        self.reward_head = nn.Sequential(
            nn.Linear(bottleneck_dim + action_dim, 128),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
            nn.Tanh()
        )
    
    def forward(self, state, action, next_state, done):
        z = self.encoder(state)
        x = torch.cat([z, action], dim=1)
        return self.reward_head(x).squeeze(-1)


# Architecture registry
ARCHITECTURES = {
    "baseline": MLPRewardNet,
    "bounded_tanh": lambda *args, **kwargs: BoundedRewardNet(*args, output_activation="tanh", **kwargs),
    "bounded_sigmoid": lambda *args, **kwargs: BoundedRewardNet(*args, output_activation="sigmoid", **kwargs),
    "larger": LargerMLPRewardNet,
    "per_camera": PerCameraRewardNet,
    "per_camera_attention": PerCameraRewardNet,  # fusion comes from config.fusion
    "per_camera_mean": PerCameraRewardNet,
    "per_camera_max": PerCameraRewardNet,
    "bottleneck": BottleneckRewardNet,
}


# ============================================================
# EVALUATION
# ============================================================

def compute_theoretical_preference_accuracy(
    test_trajectories: List[ProcessedTrajectory],
    n_pairs: int = 500,
    seed: int = 123,
) -> float:
    """
    Compute the theoretical preference accuracy ceiling for this noise level.
    
    Measures how often noisy preferences (from the simulated user at this
    camera config) agree with oracle preferences (from ground truth rewards).
    This is the CEILING the reward model can hope to achieve — it cannot do
    better than the training signal it receives.
    
    For num_cameras=-1 (random rewards), this should be ~0.50 (coin flip).
    For num_cameras=0  (no noise),       this should be ~0.89 (from paper).
    For num_cameras=1  (single view),    this should be ~0.70.
    For num_cameras=3  (multi-view),     this should be ~0.82.
    
    Args:
        test_trajectories: ProcessedTrajectory with both .rews (noisy) and 
                          .rews_ground_truth (oracle)
        n_pairs: Number of trajectory pairs to sample
        seed: Random seed
    
    Returns:
        Fraction of pairs where noisy preference agrees with oracle preference
    """
    rng = np.random.default_rng(seed)
    n_trajs = len(test_trajectories)
    
    if n_trajs < 2:
        return 0.5  # Can't compare with <2 trajectories
    
    # Precompute returns for all test trajectories
    noisy_returns = np.array([t.rews.sum() for t in test_trajectories])
    gt_returns = np.array([t.rews_ground_truth.sum() for t in test_trajectories])
    
    agree = 0
    valid = 0
    
    for _ in range(n_pairs):
        i, j = rng.choice(n_trajs, 2, replace=False)
        
        # Oracle preference (from ground truth)
        gt_pref = gt_returns[i] > gt_returns[j]
        
        # Noisy preference (from simulated user at this camera config)
        noisy_pref = noisy_returns[i] > noisy_returns[j]
        
        # Skip ties in ground truth (ambiguous — neither label is "wrong")
        if gt_returns[i] == gt_returns[j]:
            continue
        
        valid += 1
        if gt_pref == noisy_pref:
            agree += 1
    
    if valid == 0:
        return 0.5
    
    return agree / valid


def evaluate_model(
    reward_net: RewardNet,
    test_trajectories: List[ProcessedTrajectory],
    device: str,
    n_preference_pairs: int = 500,
    seed: int = 123
) -> Dict[str, float]:
    """
    Evaluate model on held-out test set using GROUND TRUTH rewards.
    
    This ensures all models (trained with different noise levels) are
    compared fairly against the same target.
    
    Args:
        reward_net: Trained reward network
        test_trajectories: ProcessedTrajectory objects with ground truth rewards
        device: cuda/cpu
        n_preference_pairs: Number of pairs to sample for preference accuracy
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary of evaluation metrics
    """
    reward_net.eval()
    rng = np.random.default_rng(seed)
    
    all_pred = []
    all_true_gt = []  # Ground truth rewards
    traj_returns_pred = []
    traj_returns_true_gt = []  # Ground truth returns
    
    with torch.no_grad():
        for traj in test_trajectories:
            obs = torch.tensor(traj.obs[:-1], dtype=torch.float32, device=device)
            acts = torch.tensor(traj.acts, dtype=torch.float32, device=device)
            
            pred = reward_net(obs, acts, None, None).cpu().numpy()
            
            all_pred.extend(pred)
            all_true_gt.extend(traj.rews_ground_truth)  # Use GROUND TRUTH
            traj_returns_pred.append(pred.sum())
            traj_returns_true_gt.append(traj.rews_ground_truth.sum())  # Use GROUND TRUTH
    
    all_pred = np.array(all_pred)
    all_true_gt = np.array(all_true_gt)
    
    # Timestep-level correlations (against ground truth)
    timestep_pearson, _ = pearsonr(all_true_gt, all_pred)
    timestep_spearman, _ = spearmanr(all_true_gt, all_pred)
    
    # Return correlation (against ground truth)
    return_spearman, _ = spearmanr(traj_returns_true_gt, traj_returns_pred)
    
    # Preference accuracy (against ground truth)
    # This measures: can the model correctly rank trajectories based on TRUE reward?
    correct = 0
    for _ in range(n_preference_pairs):
        i, j = rng.choice(len(test_trajectories), 2, replace=False)
        
        # Ground truth preference
        true_pref = traj_returns_true_gt[i] > traj_returns_true_gt[j]
        # Model's preference
        pred_pref = traj_returns_pred[i] > traj_returns_pred[j]
        
        if true_pref == pred_pref:
            correct += 1
    
    preference_accuracy = correct / n_preference_pairs
    
    return {
        "test_preference_accuracy": preference_accuracy,
        "test_return_spearman": return_spearman,
        "test_timestep_pearson": timestep_pearson,
        "test_timestep_spearman": timestep_spearman,
    }


# ============================================================
# TRAINING
# ============================================================

def create_reward_net(
    config: ExperimentConfig,
    obs_dim: int,
    action_dim: int,
    device: str
) -> RewardNet:
    """Create reward network based on config."""
    
    obs_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
    action_space = spaces.Box(-np.inf, np.inf, shape=(action_dim,), dtype=np.float32)
    
    arch_name = config.architecture
    
    if arch_name not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture: {arch_name}. Available: {list(ARCHITECTURES.keys())}")
    
    arch_cls = ARCHITECTURES[arch_name]
    
    # Build kwargs based on architecture
    kwargs = {"hid_sizes": config.hid_sizes}
    
    if "per_camera" in arch_name:
        visual_feat_dim = BACKBONE_CONFIGS[config.backbone]["dim"]
        proprio_dim = obs_dim - (config.num_cameras * visual_feat_dim)
        
        kwargs.update({
            "num_cameras": config.num_cameras,
            "visual_feat_dim": visual_feat_dim,
            "proprio_dim": proprio_dim,
            "fusion": config.fusion,
        })
    
    reward_net = arch_cls(obs_space, action_space, **kwargs)
    reward_net = reward_net.to(device)
    
    # Print model info
    n_params = sum(p.numel() for p in reward_net.parameters())
    print(f"Created {arch_name} with {n_params:,} parameters")
    
    return reward_net


def train_reward_model(
    config: ExperimentConfig,
    train_trajectories: List[TrajectoryWithRew],
    obs_dim: int,
    action_dim: int,
) -> Tuple[RewardNet, float]:
    """
    Train reward model and return trained network + training time.
    """
    device = config.device
    
    # Set seeds
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    
    # Create dataset
    rng = np.random.default_rng(config.seed)
    dataset = TrajectoryDataset(train_trajectories, rng=rng)
    
    # Create reward network
    reward_net = create_reward_net(config, obs_dim, action_dim, device)
    
    # Training setup
    fragmenter = preference_comparisons.RandomFragmenter(rng=rng, warning_threshold=0)
    gatherer = preference_comparisons.SyntheticGatherer(rng=rng)
    
    reward_trainer = preference_comparisons.BasicRewardTrainer(
        preference_model=preference_comparisons.PreferenceModel(reward_net),
        loss=preference_comparisons.CrossEntropyRewardLoss(),
        epochs=config.epochs,
        batch_size=config.batch_size,
        rng=rng,
    )
    
    pref_comparisons = preference_comparisons.PreferenceComparisons(
        trajectory_generator=dataset,
        reward_model=reward_net,
        num_iterations=config.num_iterations,
        fragmenter=fragmenter,
        preference_gatherer=gatherer,
        reward_trainer=reward_trainer,
        fragment_length=config.fragment_length,
        transition_oversampling=1.0,
        initial_comparison_frac=config.initial_comparison_frac,
        query_schedule="hyperbolic",
        allow_variable_horizon=True,
    )
    
    # Train
    print(f"Training with {config.total_comparisons} comparisons...")
    start_time = time.time()
    
    pref_comparisons.train(
        total_timesteps=0,
        total_comparisons=config.total_comparisons,
    )
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.1f}s")
    
    return reward_net, training_time


# ============================================================
# EXPERIMENT RUNNER
# ============================================================

class Logger:
    """Dual logging to console and file."""
    
    def __init__(self, filepath: str):
        self.terminal = sys.stdout
        self.log = open(filepath, "w", encoding='utf-8', buffering=1)
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()


def run_experiment(
    config: ExperimentConfig,
    data_root: str,
    output_dir: str,
) -> ExperimentResults:
    """
    Run a single experiment with given configuration.
    
    Saves:
        - config.json: Experiment configuration
        - results.json: Evaluation results
        - reward_model.pt: Trained model weights
        - training.log: Full training log
    """
    
    # Create output directory
    exp_name = config.experiment_name()
    exp_dir = Path(output_dir) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_path = exp_dir / "training.log"
    logger = Logger(str(log_path))
    old_stdout = sys.stdout
    sys.stdout = logger
    
    print("=" * 60)
    print(f"EXPERIMENT: {exp_name}")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 60)
    
    # Save config
    config_path = exp_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config.to_dict(), f, indent=2)
    print(f"Config saved to {config_path}")
    
    device = config.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    try:
        # Load trajectories
        print("\n" + "=" * 60)
        print("LOADING DATA")
        print("=" * 60)
        
        # Get camera names (for image loading)
        # num_cameras=-1 uses front camera but random rewards
        actual_num_cams = abs(config.num_cameras) if config.num_cameras != -1 else 1
        camera_names = CAMERA_CONFIGS.get(actual_num_cams, ["front"])
        
        raw_trajectories = get_trajectories(
            root_dir=data_root,
            camera_names=camera_names,
            alpha=config.alpha,
            noise_model=config.noise_model,
            num_cameras=config.num_cameras,  # Pass actual value (-1, 1, 2, 3) for reward noise
            max_trajectories=config.sample_trajectories,
        )
        
        if not raw_trajectories:
            raise ValueError("No trajectories loaded!")
        
        # Precompute features (returns both train format and eval format)
        print("\n" + "=" * 60)
        print("FEATURE EXTRACTION")
        print("=" * 60)
        
        all_train_trajs, all_eval_trajs, obs_dim = precompute_features(
            raw_trajectories, 
            device=device,
            backbone_name=config.backbone,
            include_proprio=config.include_proprio,
        )
        
        # Train/test split (splits both in parallel with same indices)
        train_trajs, test_trajs, train_eval, test_eval = train_test_split(
            all_train_trajs,
            all_eval_trajs,
            train_frac=config.train_frac, 
            seed=config.seed
        )
        
        action_dim = train_trajs[0].acts.shape[1]
        
        print(f"Observation dim: {obs_dim}")
        print(f"Action dim: {action_dim}")
        print(f"Train trajectories: {len(train_trajs)}")
        print(f"Test trajectories: {len(test_trajs)}")
        
        # Train
        print("\n" + "=" * 60)
        print("TRAINING")
        print("=" * 60)
        
        reward_net, training_time = train_reward_model(
            config=config,
            train_trajectories=train_trajs,
            obs_dim=obs_dim,
            action_dim=action_dim,
        )
        
        # Evaluate on test set using GROUND TRUTH rewards
        print("\n" + "=" * 60)
        print("EVALUATION (against ground truth)")
        print("=" * 60)
        
        eval_metrics = evaluate_model(
            reward_net=reward_net,
            test_trajectories=test_eval,  # Uses ground truth rewards
            device=device,
        )
        
        print(f"Test Preference Accuracy: {eval_metrics['test_preference_accuracy']:.4f}")
        print(f"Test Return Correlation:  {eval_metrics['test_return_spearman']:.4f}")
        print(f"Test Timestep Pearson:    {eval_metrics['test_timestep_pearson']:.4f}")
        print(f"Test Timestep Spearman:   {eval_metrics['test_timestep_spearman']:.4f}")
        
        # Compute theoretical preference accuracy ceiling
        # This measures the training signal quality, not the model
        print("\n" + "=" * 60)
        print("THEORETICAL CEILING (noisy vs oracle preference agreement)")
        print("=" * 60)
        
        theoretical_acc = compute_theoretical_preference_accuracy(
            test_eval,
            n_pairs=1000,
            seed=123,
        )
        
        # Normalized score: fraction of available signal the model captures
        # = (model_acc - 0.5) / (theoretical_acc - 0.5)
        # chance = 0.5 for binary preference comparisons
        CHANCE = 0.5
        if theoretical_acc > CHANCE:
            normalized_score = (eval_metrics['test_preference_accuracy'] - CHANCE) / (theoretical_acc - CHANCE)
        else:
            # Theoretical ceiling is at or below chance (e.g., random baseline)
            # Normalized score is 0 if model is at chance, undefined otherwise
            normalized_score = 0.0
        
        print(f"Theoretical accuracy:     {theoretical_acc:.4f}")
        print(f"Model accuracy:           {eval_metrics['test_preference_accuracy']:.4f}")
        print(f"Normalized score:         {normalized_score:.4f}")
        
        if normalized_score > 1.0:
            print(f"  (Model exceeds ceiling — likely eval variance, not real)")
        
        # Create results object
        results = ExperimentResults(
            test_preference_accuracy=eval_metrics['test_preference_accuracy'],
            test_return_spearman=eval_metrics['test_return_spearman'],
            theoretical_preference_accuracy=theoretical_acc,
            normalized_preference_score=normalized_score,
            test_timestep_pearson=eval_metrics['test_timestep_pearson'],
            test_timestep_spearman=eval_metrics['test_timestep_spearman'],
            training_time_seconds=training_time,
            total_train_trajectories=len(train_trajs),
            total_test_trajectories=len(test_trajs),
            obs_dim=obs_dim,
            action_dim=action_dim,
        )
        
        # Save results
        results_path = exp_dir / "results.json"
        with open(results_path, "w") as f:
            json.dump(results.to_dict(), f, indent=2)
        print(f"\nResults saved to {results_path}")
        
        # Save model
        model_path = exp_dir / "reward_model.pt"
        torch.save({
            "state_dict": reward_net.state_dict(),
            "config": config.to_dict(),
            "obs_dim": obs_dim,
            "action_dim": action_dim,
        }, model_path)
        print(f"Model saved to {model_path}")
        
        # Save for Isaac Lab integration
        isaaclab_model_path = exp_dir / "reward_model_isaaclab.pt"
        torch.save({
            "state_dict": reward_net.state_dict(),
            "architecture": config.architecture,
            "obs_dim": obs_dim,
            "action_dim": action_dim,
            "num_cameras": config.num_cameras,
            "backbone": config.backbone,
            "include_proprio": config.include_proprio,
            "hid_sizes": list(config.hid_sizes),
        }, isaaclab_model_path)
        print(f"Isaac Lab model saved to {isaaclab_model_path}")
        
        print("\n" + "=" * 60)
        print("EXPERIMENT COMPLETE")
        print(f"Score: {results.primary_score:.4f}")
        print(f"Normalized: {results.normalized_preference_score:.4f} "
              f"(model {results.test_preference_accuracy:.3f} / "
              f"ceiling {results.theoretical_preference_accuracy:.3f})")
        print("=" * 60)
        
        return results
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        raise
        
    finally:
        sys.stdout = old_stdout
        logger.close()


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Preference-based reward learning experiment")
    
    # Data
    parser.add_argument("--data_root", type=str, 
                        default="/datasets/work/hri-fyp2025s1-2903/work/pref_updated")
    parser.add_argument("--output_dir", type=str, default="./experiments")
    
    # Experiment config
    parser.add_argument("--num_cams", type=int, default=1,
                        help="Number of cameras: -1=random baseline, 1/2/3=actual cameras")
    parser.add_argument("--alpha", type=float, default=1.5)
    parser.add_argument("--noise_model", type=str, default="tanh",
                    choices=["none", "tanh", "linear", "exponential", "sqrt", "occlusion", "constant"])
    parser.add_argument("--sample_trajectories", type=int, default=0,
                    help="0 = use all available")
    parser.add_argument("--train_frac", type=float, default=0.8)
    
    # Architecture
    parser.add_argument("--architecture", type=str, default="baseline",
                        choices=list(ARCHITECTURES.keys()))
    parser.add_argument("--backbone", type=str, default="resnet50",
                        choices=list(BACKBONE_CONFIGS.keys()))
    parser.add_argument("--hid_sizes", type=int, nargs="+", default=[256, 128])
    parser.add_argument("--fusion", type=str, default="concat",
                        choices=["concat", "mean", "max", "attention"])
    parser.add_argument("--include_proprio", action="store_true", default=True)
    
    # Training
    parser.add_argument("--total_comparisons", type=int, default=2000)
    parser.add_argument("--fragment_length", type=int, default=75)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_iterations", type=int, default=8)
    
    # Meta
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility. Use different seeds (0,1,2) for error bars.")
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    # Validate num_cams
    if args.num_cams not in [-1, 1, 2, 3]:
        raise ValueError(f"num_cams must be -1, 1, 2, or 3. Got: {args.num_cams}")
    
    # Create config
    config = ExperimentConfig(
        num_cameras=args.num_cams,
        alpha=args.alpha,
        noise_model=args.noise_model,
        sample_trajectories=args.sample_trajectories,
        train_frac=args.train_frac,
        architecture=args.architecture,
        backbone=args.backbone,
        hid_sizes=tuple(args.hid_sizes),
        fusion=args.fusion,
        include_proprio=args.include_proprio,
        total_comparisons=args.total_comparisons,
        fragment_length=args.fragment_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_iterations=args.num_iterations,
        seed=args.seed,
        device=args.device,
    )
    
    # Run experiment
    results = run_experiment(
        config=config,
        data_root=args.data_root,
        output_dir=args.output_dir,
    )
    
    print(f"\nFinal Score: {results.primary_score:.4f}")


if __name__ == "__main__":
    main()

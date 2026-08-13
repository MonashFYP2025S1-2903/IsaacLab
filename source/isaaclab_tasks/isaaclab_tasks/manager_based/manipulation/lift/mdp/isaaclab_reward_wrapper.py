"""
isaaclab_reward_wrapper.py

Bridge between trained preference-based reward models and Isaac Lab runtime.

During training (episode_analysis.py), the pipeline is:
    images → resize → ResNet backbone → global avg pool → 2048-dim features
    features + proprio → reward MLP → scalar reward

This wrapper reconstructs that pipeline for real-time inference inside
Isaac Lab environments. It loads the reward_model_isaaclab.pt checkpoint,
sets up the backbone, and provides a simple __call__ interface:
    (camera_images, proprio, actions) → per-env reward tensor

Usage:
    from isaaclab_reward_wrapper import LearnedRewardFunction

    reward_fn = LearnedRewardFunction.from_checkpoint(
        "experiments/.../reward_model_isaaclab.pt", device="cuda"
    )

    # Inside your env step:
    rewards = reward_fn(images, proprio, actions)
    # images: (num_envs, num_cams, 3, H, W) uint8 or float
    # proprio: (num_envs, proprio_dim) float
    # actions: (num_envs, action_dim) float
    # rewards: (num_envs,) float
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from typing import Optional

import torchvision.models as models
import torchvision.transforms as T


class LearnedRewardFunction(nn.Module):
    """
    End-to-end reward function: raw images + proprio → scalar reward.

    Reconstructs the training-time pipeline:
        1. Resize images to target_width (matching training resolution)
        2. Normalize with ImageNet stats
        3. Extract features via frozen ResNet backbone
        4. Global average pool → flat feature vector per camera
        5. Concatenate [cam1_features, cam2_features, ..., proprio]
        6. Forward through trained reward MLP
        7. Return scalar reward per environment

    The backbone weights are frozen ImageNet pretrained (same as training).
    The reward MLP weights are loaded from the checkpoint.
    """

    def __init__(
        self,
        reward_net: nn.Module,
        backbone_name: str = "resnet50",
        num_cameras: int = 1,
        target_width: int = 224,
        include_proprio: bool = True,
        device: str = "cuda",
    ):
        super().__init__()

        self.num_cameras = num_cameras
        self.target_width = target_width
        self.include_proprio = include_proprio
        self._device = device

        # ---- Backbone (frozen, same as training) ----
        if backbone_name == "resnet50":
            backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self.feat_dim = 2048
        elif backbone_name == "resnet34":
            backbone = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
            self.feat_dim = 512
        elif backbone_name == "resnet18":
            backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            self.feat_dim = 512
        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")

        # Keep conv layers + adaptive pool, discard FC
        self.backbone = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

        # ---- ImageNet normalization (same as training) ----
        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        # ---- Trained reward MLP ----
        self.reward_net = reward_net
        self.reward_net.eval()
        for p in self.reward_net.parameters():
            p.requires_grad = False

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: str = "cuda",
    ) -> "LearnedRewardFunction":
        """
        Load a complete reward function from a checkpoint.

        The checkpoint (reward_model_isaaclab.pt) contains:
            - state_dict: trained reward MLP weights
            - architecture: which MLP class to use
            - obs_dim, action_dim: input dimensions
            - camera_config, num_cameras: which cameras
            - backbone: which ResNet
            - target_width: image resize resolution
            - include_proprio: whether proprio is part of obs
            - hid_sizes: MLP hidden layer sizes

        This reconstructs the reward MLP, loads weights, sets up the
        backbone, and returns a ready-to-use LearnedRewardFunction.
        """
        from episode_analysis import ExperimentConfig, create_reward_net, ARCHITECTURES
        from get_trajectories import BACKBONE_CONFIGS

        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Reconstruct config enough to create the reward net
        obs_dim = ckpt["obs_dim"]
        action_dim = ckpt["action_dim"]
        architecture = ckpt.get("architecture", "baseline")
        backbone_name = ckpt.get("backbone", "resnet50")
        num_cameras = ckpt.get("num_cameras", 1)
        camera_config = ckpt.get("camera_config", "front")
        target_width = ckpt.get("target_width", 224)
        include_proprio = ckpt.get("include_proprio", True)
        hid_sizes = tuple(ckpt.get("hid_sizes", [256, 128]))

        # Build a minimal ExperimentConfig for create_reward_net
        config = ExperimentConfig(
            camera_config=camera_config,
            architecture=architecture,
            backbone=backbone_name,
            hid_sizes=hid_sizes,
            include_proprio=include_proprio,
            target_width=target_width,
        )

        # Create and load the reward MLP
        reward_net = create_reward_net(config, obs_dim, action_dim, device)
        reward_net.load_state_dict(ckpt["state_dict"])
        reward_net.eval()

        print(f"[LearnedRewardFunction] Loaded from {checkpoint_path}")
        print(f"  Architecture: {architecture}")
        print(f"  Camera config: {camera_config} ({num_cameras} cameras)")
        print(f"  Backbone: {backbone_name} ({BACKBONE_CONFIGS[backbone_name]['dim']}-dim)")
        print(f"  Resolution: {target_width}px")
        print(f"  Obs dim: {obs_dim}, Action dim: {action_dim}")

        wrapper = cls(
            reward_net=reward_net,
            backbone_name=backbone_name,
            num_cameras=num_cameras,
            target_width=target_width,
            include_proprio=include_proprio,
            device=device,
        )
        wrapper.to(device)
        return wrapper

    def _preprocess_images(self, images: torch.Tensor) -> torch.Tensor:
        """
        Resize and normalize images to match training pipeline.

        Args:
            images: (B, N_cams, 3, H, W) uint8 [0, 255] or float [0, 1]

        Returns:
            (B, N_cams, 3, H', W') float normalized
        """
        B, N, C, H, W = images.shape

        # Convert to float [0, 1] if uint8
        if images.dtype == torch.uint8:
            images = images.float() / 255.0

        # Resize if needed (match training resolution)
        if W != self.target_width:
            # Maintain aspect ratio
            new_h = int(H * (self.target_width / W))
            images = images.reshape(B * N, C, H, W)
            images = torch.nn.functional.interpolate(
                images, size=(new_h, self.target_width),
                mode="bilinear", align_corners=False,
            )
            images = images.reshape(B, N, C, new_h, self.target_width)

        # Apply ImageNet normalization per camera
        # normalize expects (C, H, W), apply to (B*N, C, H, W)
        B, N, C, H, W = images.shape
        images = images.reshape(B * N, C, H, W)
        images = self.normalize(images)
        images = images.reshape(B, N, C, H, W)

        return images

    @torch.no_grad()
    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract visual features from preprocessed images.

        Args:
            images: (B, N_cams, 3, H, W) normalized float

        Returns:
            (B, N_cams * feat_dim) concatenated features
        """
        B, N, C, H, W = images.shape

        # Process all cameras through backbone
        # Reshape to (B*N, C, H, W) for batch efficiency
        flat_images = images.reshape(B * N, C, H, W)
        flat_features = self.backbone(flat_images)  # (B*N, feat_dim)

        # Reshape back: (B, N, feat_dim) → (B, N * feat_dim)
        features = flat_features.reshape(B, N, self.feat_dim)
        features = features.reshape(B, N * self.feat_dim)

        return features

    @torch.no_grad()
    def forward(
        self,
        images: torch.Tensor,
        proprio: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute rewards from raw camera images + proprio + actions.

        This mirrors the training pipeline exactly:
            images → resize → normalize → backbone → pool → features
            [features, proprio] → reward_net(obs, action, None, None) → reward

        Args:
            images: (B, N_cams, 3, H, W) — raw camera output (uint8 or float)
            proprio: (B, proprio_dim) — joint positions, velocities, etc.
            actions: (B, action_dim) — robot actions

        Returns:
            (B,) reward tensor
        """
        # 1. Preprocess images (resize + normalize)
        images = self._preprocess_images(images)

        # 2. Extract visual features
        visual_features = self.extract_features(images)  # (B, N_cams * feat_dim)

        # 3. Build observation vector matching training format
        if self.include_proprio and proprio is not None:
            obs = torch.cat([visual_features, proprio], dim=1)
        else:
            obs = visual_features

        # 4. Build action tensor (zero if not provided)
        if actions is None:
            actions = torch.zeros(
                obs.shape[0], 1, device=obs.device, dtype=obs.dtype
            )

        # 5. Forward through trained reward MLP
        # RewardNet.forward(state, action, next_state, done)
        rewards = self.reward_net(obs, actions, None, None)

        return rewards

    def __repr__(self) -> str:
        return (
            f"LearnedRewardFunction("
            f"cameras={self.num_cameras}, "
            f"feat_dim={self.feat_dim}, "
            f"resolution={self.target_width}px, "
            f"proprio={self.include_proprio})"
        )

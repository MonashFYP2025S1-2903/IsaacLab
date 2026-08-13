# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from . import mdp

import torchvision.models as models
import torchvision.transforms as T

##
# Scene definition
##

@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """Configuration for the lift scene with a robot and a object.
    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the target object, robot and end-effector frames
    """

    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = MISSING
    # end-effector sensor: will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING
    # target object: will be populated by agent env cfg
    object: RigidObjectCfg | DeformableObjectCfg = MISSING
    
    # Table
    #
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.25, 0, -0.5], rot=[0.707, 0, 0, 0.707]),
        spawn=UsdFileCfg(usd_path=f"/home/sh-d61-cps-hri/hri-pl-frm-mvvd/isaaclab/usd_assets/FYP_shop_table_2.usd",scale=(0.01, 0.01, 0.01),
                         semantic_tags=[("class", "table")]),
    )

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=12000.0),
    )

# ============================================================
# VISUAL OBSERVATION HELPER
# ============================================================

class VisualFeatureExtractor:
    """Singleton to hold the frozen backbone for generating observations."""
    def __init__(self, camera_names: list[str], backbone_name: str = "resnet50", device: str = "cuda"):
        import torchvision.models as models
        import torchvision.transforms as T
        
        self.camera_names = camera_names
        self.device = device
        
        # Load backbone (same architecture as ImagePolicyEncoder)
        if backbone_name == "resnet50":
            backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self.feat_dim = 2048
        elif backbone_name == "resnet18":
            backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            self.feat_dim = 512
        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")

        # Remove final FC, keep conv + pool
        self.backbone = torch.nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4,
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
        ).to(device)

        # Freeze
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ).to(device)

    def extract(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (num_envs, num_cams, 3, H, W) float [0, 1]
        Returns:
            features: (num_envs, num_cams * feat_dim)
        """
        B, N_cams, C, H, W = images.shape
        
        # Flatten batch and cams for processing: (B*N, C, H, W)
        flat_imgs = images.view(B * N_cams, C, H, W)
        
        # Normalize
        flat_imgs = self.normalize(flat_imgs)
        
        with torch.no_grad():
            feats = self.backbone(flat_imgs) # (B*N, feat_dim)
            
        # Reshape back to (B, N * feat_dim)
        return feats.view(B, N_cams * self.feat_dim)

# Global singleton
_visual_extractor: VisualFeatureExtractor | None = None

def visual_features_obs(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = None) -> torch.Tensor:
    """
    Observation function: Gathers images -> ResNet Backbone -> Feature Vector.
    """
    global _visual_extractor
    
    # 1. Define cameras to use (must match scene definition)
    # Ideally match the reward model config, but for simplicity we list them here:
    camera_names = ["camera_ext1"]
    
    # 2. Lazy init singleton
    if _visual_extractor is None:
        _visual_extractor = VisualFeatureExtractor(camera_names, device=env.device)

    # 3. Gather images
    camera_images = []
    for cam_name in _visual_extractor.camera_names:
        sensor_name = f"cam_{cam_name}"
        cam = env.scene.sensors[sensor_name]
        # Get RGB, drop alpha, reshape to (N, 3, H, W)
        rgb = cam.data.output["rgb"][..., :3].permute(0, 3, 1, 2).contiguous()
        camera_images.append(rgb)
    
    # Stack: (num_envs, num_cams, 3, H, W)
    images = torch.stack(camera_images, dim=1).float() / 255.0

    # 4. Extract features
    return _visual_extractor.extract(images)

##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,  # will be set by agent env cfg
        resampling_time_range=(5.0, 5.0),
        debug_vis=False,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.4, 0.6), pos_y=(-0.25, 0.25), pos_z=(0.25, 0.5), roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0)
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # will be set by agent env cfg
    arm_action: mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING



@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""
    
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
       
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        # camera_features = ObsTerm(func=visual_features_obs)
        target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
   
    
    # @configclass
    # class RGBCfg(ObsGroup):
    #     """Observations for image group."""
    #     image = ObsTerm(func=mdp.image, params={"sensor_cfg": SceneEntityCfg("camera"), "data_type": "rgb"})
    #     image1 = ObsTerm(func=mdp.image, params={"sensor_cfg": SceneEntityCfg("camera_ext1"), "data_type": "rgb"})
    #     image2 = ObsTerm(func=mdp.image, params={"sensor_cfg": SceneEntityCfg("camera_ext2"), "data_type": "rgb"})
    #     image3 = ObsTerm(func=mdp.image, params={"sensor_cfg": SceneEntityCfg("camera_bird"), "data_type": "rgb"})
        
    #     def __post_init__(self):
    #         self.enable_corruption = False
    #         self.concatenate_terms = False
    

    # @configclass
    # class DepthCfg(ObsGroup):
    #     """Observations for depth group."""
    #     image = ObsTerm(func=mdp.image, params={"sensor_cfg": SceneEntityCfg("camera"), "data_type": "depth"})
    #     image1 = ObsTerm(func=mdp.image, params={"sensor_cfg": SceneEntityCfg("camera_ext1"), "data_type": "depth"})
    #     image2 = ObsTerm(func=mdp.image, params={"sensor_cfg": SceneEntityCfg("camera_ext2"), "data_type": "depth"})
    #     image3 = ObsTerm(func=mdp.image, params={"sensor_cfg": SceneEntityCfg("camera_bird"), "data_type": "depth"})
        
    #     def __post_init__(self):
    #         self.enable_corruption = False
    #         self.concatenate_terms = False

    # @configclass
    # class SemanticCfg(ObsGroup):
    #     """Observations for semantic group."""
    #     image = ObsTerm(func=mdp.image, params={"sensor_cfg": SceneEntityCfg("camera"), "data_type": "semantic_segmentation"})
    #     image1 = ObsTerm(func=mdp.image, params={"sensor_cfg": SceneEntityCfg("camera_ext1"), "data_type": "semantic_segmentation"})
    #     image2 = ObsTerm(func=mdp.image, params={"sensor_cfg": SceneEntityCfg("camera_ext2"), "data_type": "semantic_segmentation"})
    #     image3 = ObsTerm(func=mdp.image, params={"sensor_cfg": SceneEntityCfg("camera_bird"), "data_type": "semantic_segmentation"})

    #     def __post_init__(self):
    #         self.enable_corruption = False
    #         self.concatenate_terms = False

    @configclass
    class PrefLogCfg(ObsGroup):
        """Observations for preflog group - position logging for manual reward evaluation."""
    
        # Log positions used in reward functions
        object_position = ObsTerm(func=mdp.log_object_position, params={"object_cfg": SceneEntityCfg("object")})
        ee_position = ObsTerm(func=mdp.log_ee_position, params={"ee_frame_cfg": SceneEntityCfg("ee_frame")})
        goal_position = ObsTerm(
            func=mdp.log_goal_position, 
            params={"command_name": "object_pose", "robot_cfg": SceneEntityCfg("robot")}
        )

        # Log joint velocities used in reward functions
        joint_velocities = ObsTerm(func=mdp.log_joint_velocities, params={"asset_cfg": SceneEntityCfg("robot")})
        # Log actions used in reward functions
        current_action = ObsTerm(func=mdp.log_current_action)
        previous_action = ObsTerm(func=mdp.log_previous_action)
        action_difference = ObsTerm(func=mdp.log_action_difference)

        # Log observations
        obs_joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        obs_joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        obs_object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        obs_target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        obs_actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False
    


    # observation groups
    policy: PolicyCfg = PolicyCfg()
    # rgb: RGBCfg = RGBCfg()
    # depth: DepthCfg = DepthCfg()
    # semantic: SemanticCfg = SemanticCfg()
    preflog: PrefLogCfg = PrefLogCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.25, 0.25), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # Primary: learned reward from preference model
    learned = RewTerm(
        func=learned_reward,
        weight=1.0,
        params={
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
            "checkpoint_path": "/home/sh-d61-cps-hri/hri-pl-frm-mvvd/isaaclab/reward_model_isaaclab.pt",  # Set at runtime via env config
        },
    )

    # Shaping: small reaching bonus to bootstrap exploration
    # (faded out via curriculum after 10k steps)
    reaching_object = RewTerm(
        func=mdp.object_ee_distance,
        params={"std": 0.1},
        weight=0.1,
    )

    # Penalties to prevent degenerate behavior
    action_rate = RewTerm(
        func=mdp.action_rate_l2,
        weight=-1e-4,
    )
    joint_vel = RewTerm(
        func=mdp.joint_vel_l1,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")}
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    """Fade out shaping rewards so learned reward dominates."""
    reaching_fade = CurrTerm(
        func=mdp.modify_reward_weight,
        params={
            "term_name": "reaching_object",
            "weight": 0.0,  # fully off after 10k steps
            "num_steps": 10_000,
        },
    )


##
# Environment configuration
##


@configclass
class LiftLearnedRewardEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the lifting environment."""

    # Scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=4, env_spacing=5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 5.0   # simulation settings
        self.sim.dt = 1/60  # 60Hz
        self.sim.render_interval = self.decimation
        
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625

        self.sim.render.enable_dl_denoiser = False
        self.sim.render.antialiasing_mode = "FXAA"
        
        self.rerender_on_reset = True

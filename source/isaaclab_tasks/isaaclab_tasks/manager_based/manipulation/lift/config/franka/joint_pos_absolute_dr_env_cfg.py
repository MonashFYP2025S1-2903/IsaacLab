# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
#
# ABSOLUTE action (JointPositionActionCfg, the no_vel_gravon family) + VELOCITY-RANDOMISED arm cap (control-dyn DR).
# Same per-episode velocity randomization as the delta-DR task, but keeps the BASE absolute action so we get an
# ABSOLUTE policy that is IN-DISTRIBUTION at slow arm speeds -> deploy at 0.1 rad/s without the OOD "EE flies up"
# failure the fixed-fast absolute policy shows at low vmax. Position-only 27-dim obs (matches no_vel deploy).
# SELF-CONTAINED. Registered as Isaac-Lift-Cube-Franka-Absolute-DR-v0.

import os

from isaaclab.assets import AssetBaseCfg

import isaaclab.sim as sim_utils

from isaaclab.managers import CurriculumTermCfg, EventTermCfg, ObservationTermCfg, RewardTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.lift import mdp

from .joint_pos_env_cfg import FrankaCubeLiftEnvCfg
from .joint_pos_delta_dr_env_cfg import randomize_arm_velocity_limit  # reuse the velocity-randomization event
from .joint_pos_delta_dr_env_cfg import widen_cube_pose_range_curriculum  # reuse the curriculum ramp
from .joint_pos_delta_dr_env_cfg import TABLE_RAISE  # reuse the sim2real table-height fix constant


@configclass
class FrankaCubeLiftAbsoluteDREnvCfg(FrankaCubeLiftEnvCfg):
    """Base ABSOLUTE action + per-episode randomised arm velocity cap [RL_VMAX_MIN, RL_VMAX_MAX]; 27-dim obs."""

    def __post_init__(self):
        super().__post_init__()
        # keep the base ABSOLUTE JointPositionActionCfg (scale 0.5, use_default_offset) -> no action override.
        self.observations.policy.joint_vel = None   # 36 -> 27, matches the no_vel deploy
        self.events.randomize_arm_vmax = EventTermCfg(
            func=randomize_arm_velocity_limit,
            mode="reset",
            params={
                "vmin": float(os.environ.get("RL_VMAX_MIN", "0.1")),
                "vmax": float(os.environ.get("RL_VMAX_MAX", "0.8")),
                "asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"]),
            },
        )
        # EXPANDED cube-position randomization -> cover more of the REACHABLE workspace (was x+/-0.1, y+/-0.25).
        # Env-var tunable; keep within the down-grasp reachable set (far/side corners may be unreachable).
        self.events.reset_object_position.params["pose_range"] = {
            "x": (float(os.environ.get("CUBE_DX_MIN", "-0.15")), float(os.environ.get("CUBE_DX_MAX", "0.15"))),
            "y": (float(os.environ.get("CUBE_DY_MIN", "-0.35")), float(os.environ.get("CUBE_DY_MAX", "0.35"))),
            "z": (0.0, 0.0)}
        self.episode_length_s = float(os.environ.get("EP_LEN_S", "12.0"))


@configclass
class FrankaCubeLiftAbsoluteDREnvCfg_PLAY(FrankaCubeLiftAbsoluteDREnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class FrankaCubeLiftAbsoluteCurrDREnvCfg(FrankaCubeLiftAbsoluteDREnvCfg):
    """Absolute + velocity-DR + CURRICULUM cube-range widening: starts at the ORIGINAL range (x+/-0.1, y+/-0.25)
    and linearly ramps to the WIDE range (CUBE_DX/DY_MIN/MAX env vars, default +/-0.15/+/-0.35) between
    CURR_START_ITER and CURR_END_ITER (env-var, default iter 200 -> 1600 of a 2500-iter run)."""

    def __post_init__(self):
        super().__post_init__()
        start_iter = int(os.environ.get("CURR_START_ITER", "200"))
        end_iter = int(os.environ.get("CURR_END_ITER", "1600"))
        x_end = (float(os.environ.get("CUBE_DX_MIN", "-0.15")), float(os.environ.get("CUBE_DX_MAX", "0.15")))
        y_end = (float(os.environ.get("CUBE_DY_MIN", "-0.35")), float(os.environ.get("CUBE_DY_MAX", "0.35")))
        self.events.reset_object_position.params["pose_range"] = {"x": (-0.1, 0.1), "y": (-0.25, 0.25), "z": (0.0, 0.0)}
        self.curriculum.widen_cube_range = CurriculumTermCfg(
            func=widen_cube_pose_range_curriculum,
            params={
                "term_name": "reset_object_position",
                "x_start": (-0.1, 0.1), "x_end": x_end,
                "y_start": (-0.25, 0.25), "y_end": y_end,
                "start_step": start_iter * 24, "end_step": end_iter * 24,
            },
        )


@configclass
class FrankaCubeLiftAbsoluteCurrDREnvCfg_PLAY(FrankaCubeLiftAbsoluteCurrDREnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class FrankaCubeLiftAbsoluteDRTableFixEnvCfg(FrankaCubeLiftAbsoluteCurrDREnvCfg):
    """Absolute + velocity-DR + cube-range + SIM2REAL TABLE HEIGHT FIX (table & cube spawn raised ~4.1cm to
    match the real matted table, measured via touch_probe 2026-08-06)."""

    def __post_init__(self):
        super().__post_init__()
        # MAT PRIM (2026-08-06, revised approach): originally moved the whole table asset up by TABLE_RAISE.
        # Both that version AND this mat-prim version first collapsed training to reward ~6-7 -- traced (2026-08-07)
        # to inheriting from the base *DREnvCfg (wide cube-range, NO curriculum) instead of *CurrDREnvCfg. This
        # class now inherits from the CurrDR class so the curriculum-widening fix is preserved; the table-height
        # change itself was never the cause. Mat kept (matches the real mat-on-table setup more faithfully than
        # repositioning the table) and extended toward the robot base (2026-08-07) so it visually covers the full
        # reachable table area, not just the cube's DR pose_range.
        self.scene.mat = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Mat",
            spawn=sim_utils.CuboidCfg(
                size=(0.92, 0.9, TABLE_RAISE),  # far edge measured via raycast to true table edge x=0.99 (2026-08-07)
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.25, 0.28)),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=[0.51, 0, TABLE_RAISE / 2]),
        )
        ox, oy, oz = self.scene.object.init_state.pos
        self.scene.object.init_state.pos = [ox, oy, oz + TABLE_RAISE]
        # REWARD/TERMINATION HEIGHT FIX (2026-08-06, caught via sanity-run reward exploit): minimal_height in
        # lifting_object/object_goal_tracking/object_goal_tracking_fine_grained, and minimum_height in the
        # object_dropping termination, are ABSOLUTE world-frame thresholds -- raising the table without also
        # raising these left the cube's raised RESTING height (0.062) already above minimal_height=0.04,
        # giving massive free reward (weight 15+16+5=36) for doing nothing. Raise all of them by TABLE_RAISE too.
        self.rewards.lifting_object.params["minimal_height"] += TABLE_RAISE
        self.rewards.object_goal_tracking.params["minimal_height"] += TABLE_RAISE
        self.rewards.object_goal_tracking_fine_grained.params["minimal_height"] += TABLE_RAISE
        self.terminations.object_dropping.params["minimum_height"] += TABLE_RAISE



@configclass
class FrankaCubeLiftAbsoluteDRTableFixEnvCfg_PLAY(FrankaCubeLiftAbsoluteDRTableFixEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class FrankaCubeLiftAbsoluteDRTableFixWrenchEnvCfg(FrankaCubeLiftAbsoluteDRTableFixEnvCfg):
    """TableFix + EE WRENCH observation/reward (2026-08-07, PROPOSAL -- not yet validated by training).
    Adds the 6D (force+torque) wrench at panda_hand as a new observation term (breaks ONNX/deploy compat with
    the non-wrench policies -- hence a separate task, not a flag on the existing one), and an asymmetric
    contact-force reward that penalizes z (downward, into-the-table) contact much harder than xy (lateral,
    can help reorient the cube pre-grasp). z_weight/xy_weight ratio and the overall RewTerm weight are first
    guesses and need tuning once a sanity run shows the penalty's actual scale relative to the other terms.
    """

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.ee_wrench = ObservationTermCfg(
            func=mdp.ee_wrench_b, params={"asset_cfg": SceneEntityCfg("robot", body_names="panda_hand")}
        )
        self.rewards.ee_contact_penalty = RewardTermCfg(
            func=mdp.ee_contact_force_penalty,
            weight=1e-5,
            params={
                "z_weight": 1.0,
                "xy_weight": 0.2,
                "asset_cfg": SceneEntityCfg("robot", body_names="panda_hand"),
            },
        )


@configclass
class FrankaCubeLiftAbsoluteDRTableFixWrenchEnvCfg_PLAY(FrankaCubeLiftAbsoluteDRTableFixWrenchEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class FrankaCubeLiftAbsoluteDRTableFixSimpleEnvCfg(FrankaCubeLiftAbsoluteDRTableFixEnvCfg):
    """Diagnostic (2026-08-07): strips ALL domain randomization down to the simplest possible task on top
    of TableFix, to isolate whether TableFix itself can converge at all. Fixed cube spawn (no position
    randomization -- pose_range pinned to (0,0)/(0,0)), curriculum widening term removed entirely (would
    otherwise fight the fixed pose every reset). Velocity DR is neutralized by pinning RL_VMAX_MIN=RL_VMAX_MAX
    at submit time (no code change needed there -- the event still runs but samples a degenerate/fixed value).
    """

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_object_position.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)}
        self.curriculum.widen_cube_range = None
        # DIAGNOSTIC (2026-08-09): log raw contact-force magnitude at panda_hand, tiny weight (1e-6, vs. the
        # 1e-5 used where this is a real training reward) purely so Episode_Reward/ee_contact_force_metric
        # shows up in the log -- divide by 1e-6 to recover the raw episode-averaged penalty value. Added to
        # investigate whether a force spike precedes the reward drop around iter ~400-420 seen across seeds.
        self.rewards.ee_contact_force_metric = RewardTermCfg(
            func=mdp.ee_contact_force_penalty,
            weight=1e-6,
            params={
                "z_weight": 1.0,
                "xy_weight": 0.2,
                "asset_cfg": SceneEntityCfg("robot", body_names="panda_hand"),
            },
        )


@configclass
class FrankaCubeLiftAbsoluteDRTableFixSimpleEnvCfg_PLAY(FrankaCubeLiftAbsoluteDRTableFixSimpleEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class FrankaCubeLiftAbsoluteDRTableFixWrenchSimpleEnvCfg(FrankaCubeLiftAbsoluteDRTableFixSimpleEnvCfg):
    """Diagnostic (2026-08-07): EE wrench observation/reward on top of the Simple (no-DR) TableFix task,
    to isolate whether the wrench addition itself converges cleanly before confounding it with DR -- same
    methodology as FrankaCubeLiftAbsoluteDRTableFixSimpleEnvCfg isolated TableFix from DR. Fixed cube spawn,
    curriculum disabled (inherited from Simple); velocity DR still needs RL_VMAX_MIN=RL_VMAX_MAX at submit time.
    """

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.ee_wrench = ObservationTermCfg(
            func=mdp.ee_wrench_b, params={"asset_cfg": SceneEntityCfg("robot", body_names="panda_hand")}
        )
        self.rewards.ee_contact_penalty = RewardTermCfg(
            func=mdp.ee_contact_force_penalty,
            weight=1e-5,
            params={
                "z_weight": 1.0,
                "xy_weight": 0.2,
                "asset_cfg": SceneEntityCfg("robot", body_names="panda_hand"),
            },
        )
        # DIAGNOSTIC (2026-08-09): see FrankaCubeLiftAbsoluteDRTableFixSimpleEnvCfg for rationale.
        self.rewards.ee_contact_force_metric = RewardTermCfg(
            func=mdp.ee_contact_force_penalty,
            weight=1e-6,
            params={
                "z_weight": 1.0,
                "xy_weight": 0.2,
                "asset_cfg": SceneEntityCfg("robot", body_names="panda_hand"),
            },
        )


@configclass
class FrankaCubeLiftAbsoluteDRTableFixWrenchSimpleEnvCfg_PLAY(FrankaCubeLiftAbsoluteDRTableFixWrenchSimpleEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class FrankaCubeLiftAbsoluteDRTableFixWrenchSimpleEnvCfg_EVAL(FrankaCubeLiftAbsoluteDRTableFixWrenchSimpleEnvCfg):
    """Eval-only variant (2026-08-08): identical observation space (33-dim, wrench included -- so the
    already-trained network loads and runs correctly) but ee_contact_penalty reward weight zeroed, so the
    logged Mean reward is the clean 'original-only' (pre-wrench) reward for the exact same trained policy,
    measured via real rollout rather than estimated from training logs.
    """

    def __post_init__(self):
        super().__post_init__()
        self.rewards.ee_contact_penalty.weight = 0.0


@configclass
class FrankaCubeLiftAbsoluteDRTableFixWrenchObsOnlySimpleEnvCfg(FrankaCubeLiftAbsoluteDRTableFixSimpleEnvCfg):
    """2x2 ablation cell (2026-08-08): wrench OBSERVATION only, original (non-wrench) reward. Isolates
    whether the wrench-simple-absolute run's big improvement over the baseline comes from the extra sensing
    channel itself, independent of the (tiny-weight) contact-penalty reward shaping.
    """

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.ee_wrench = ObservationTermCfg(
            func=mdp.ee_wrench_b, params={"asset_cfg": SceneEntityCfg("robot", body_names="panda_hand")}
        )
        # no ee_contact_penalty reward term added -- original reward function unchanged.
        # DIAGNOSTIC (2026-08-09): see FrankaCubeLiftAbsoluteDRTableFixSimpleEnvCfg for rationale.
        self.rewards.ee_contact_force_metric = RewardTermCfg(
            func=mdp.ee_contact_force_penalty,
            weight=1e-6,
            params={
                "z_weight": 1.0,
                "xy_weight": 0.2,
                "asset_cfg": SceneEntityCfg("robot", body_names="panda_hand"),
            },
        )


@configclass
class FrankaCubeLiftAbsoluteDRTableFixWrenchObsOnlySimpleEnvCfg_PLAY(FrankaCubeLiftAbsoluteDRTableFixWrenchObsOnlySimpleEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class FrankaCubeLiftAbsoluteDRTableFixWrenchRewardOnlySimpleEnvCfg(FrankaCubeLiftAbsoluteDRTableFixSimpleEnvCfg):
    """2x2 ablation cell (2026-08-08): wrench REWARD only (contact penalty), original (27-dim, no wrench)
    observation. The policy cannot see the wrench signal but is still penalized for high contact force --
    isolates whether the reward shaping alone (without the extra sensing channel) has any effect.
    """

    def __post_init__(self):
        super().__post_init__()
        self.rewards.ee_contact_penalty = RewardTermCfg(
            func=mdp.ee_contact_force_penalty,
            weight=1e-5,
            params={
                "z_weight": 1.0,
                "xy_weight": 0.2,
                "asset_cfg": SceneEntityCfg("robot", body_names="panda_hand"),
            },
        )
        # no ee_wrench observation term added -- original 27-dim observation unchanged.
        # DIAGNOSTIC (2026-08-09): duplicate metric-only term (distinct name from the real ee_contact_penalty
        # above) so the raw force magnitude is directly comparable across all four ablation cells at the same
        # nominal weight -- see FrankaCubeLiftAbsoluteDRTableFixSimpleEnvCfg for full rationale.
        self.rewards.ee_contact_force_metric = RewardTermCfg(
            func=mdp.ee_contact_force_penalty,
            weight=1e-6,
            params={
                "z_weight": 1.0,
                "xy_weight": 0.2,
                "asset_cfg": SceneEntityCfg("robot", body_names="panda_hand"),
            },
        )


@configclass
class FrankaCubeLiftAbsoluteDRTableFixWrenchRewardOnlySimpleEnvCfg_PLAY(FrankaCubeLiftAbsoluteDRTableFixWrenchRewardOnlySimpleEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False

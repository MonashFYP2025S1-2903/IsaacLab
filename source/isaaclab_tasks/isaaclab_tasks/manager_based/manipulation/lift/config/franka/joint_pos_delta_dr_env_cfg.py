# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
#
# DELTA action + VELOCITY-RANDOMISED arm cap (control-dynamics DR). Per episode, each env samples a max joint
# speed in [RL_VMAX_MIN, RL_VMAX_MAX] and writes it to the arm joints -> the policy learns to do the task across
# a RANGE of arm speeds. Fast episodes bootstrap learning (avoid the from-scratch exploration wall at a fixed slow
# cap); slow episodes add robustness -> DEPLOY at any speed in the band (e.g. 0.3, slower than the fixed-0.8 run),
# no exact vmax match needed. Plain delta action (linear -> learns), position-only 27-dim obs. SELF-CONTAINED.
# Registered as Isaac-Lift-Cube-Franka-Delta-DR-v0.

import os

from isaaclab.assets import AssetBaseCfg

import isaaclab.sim as sim_utils
import torch

from isaaclab.managers import CurriculumTermCfg, EventTermCfg, ObservationTermCfg, RewardTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.lift import mdp

from .joint_pos_env_cfg import FrankaCubeLiftEnvCfg


def randomize_arm_velocity_limit(env, env_ids, vmin: float, vmax: float, asset_cfg: SceneEntityCfg):
    """mode=reset event: sample ONE max joint velocity per env in [vmin,vmax] and write it to the arm joints,
    so training spans a range of arm speeds (models the real joint_pos_runner TRACK_VMAX at varying values)."""
    asset = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids
    n = int(env_ids.shape[0]) if hasattr(env_ids, "shape") else len(env_ids)
    nj = len(joint_ids) if hasattr(joint_ids, "__len__") else asset.num_joints
    v = torch.empty(n, 1, device=asset.device).uniform_(float(vmin), float(vmax))
    asset.write_joint_velocity_limit_to_sim(v.expand(n, nj).contiguous(), joint_ids=joint_ids, env_ids=env_ids)


@configclass
class FrankaCubeLiftDeltaDREnvCfg(FrankaCubeLiftEnvCfg):
    """Delta action + per-episode randomised arm velocity cap [RL_VMAX_MIN, RL_VMAX_MAX]; 27-dim obs."""

    def __post_init__(self):
        super().__post_init__()
        self.actions.arm_action = mdp.RelativeJointPositionActionCfg(
            asset_name="robot", joint_names=["panda_joint.*"],
            scale=float(os.environ.get("DELTA_SCALE", "0.10")), use_zero_offset=True,
        )
        self.observations.policy.joint_vel = None
        self.events.randomize_arm_vmax = EventTermCfg(
            func=randomize_arm_velocity_limit,
            mode="reset",
            params={
                "vmin": float(os.environ.get("RL_VMAX_MIN", "0.2")),
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
        self.episode_length_s = float(os.environ.get("EP_LEN_S", "10.0"))


@configclass
class FrankaCubeLiftDeltaDREnvCfg_PLAY(FrankaCubeLiftDeltaDREnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


def widen_cube_pose_range_curriculum(
    env, env_ids, term_name: str,
    x_start: tuple, x_end: tuple, y_start: tuple, y_end: tuple,
    start_step: int, end_step: int,
):
    """CurriculumTermCfg: linearly ramps reset_object_position's x/y pose_range from (x_start,y_start) at
    start_step to (x_end,y_end) at end_step (held constant outside that window). Lets the policy first master
    the task on the EASY (original) range before the DR range widens, instead of being dropped into the full
    wide range from iteration 0 -- which caused delta to plateau at reward ~6 and absolute to cap at ~70-84
    (see FYP2025S1-2903_deployment_setup_guide.md, 2026-08-05 combined-DR collapse). common_step_counter
    increments by 1 per env.step() call regardless of num_envs (NOT multiplied) -- with num_steps_per_env=24
    (agents/rsl_rl_ppo_cfg.py), iteration N corresponds to step N*24.
    """
    step = env.common_step_counter
    if step <= start_step:
        frac = 0.0
    elif step >= end_step:
        frac = 1.0
    else:
        frac = (step - start_step) / (end_step - start_step)
    x_lo = x_start[0] + frac * (x_end[0] - x_start[0])
    x_hi = x_start[1] + frac * (x_end[1] - x_start[1])
    y_lo = y_start[0] + frac * (y_end[0] - y_start[0])
    y_hi = y_start[1] + frac * (y_end[1] - y_start[1])
    term_cfg = env.event_manager.get_term_cfg(term_name)
    term_cfg.params["pose_range"]["x"] = (x_lo, x_hi)
    term_cfg.params["pose_range"]["y"] = (y_lo, y_hi)
    env.event_manager.set_term_cfg(term_name, term_cfg)
    return frac


@configclass
class FrankaCubeLiftDeltaCurrDREnvCfg(FrankaCubeLiftDeltaDREnvCfg):
    """Delta + velocity-DR + CURRICULUM cube-range widening: starts at the ORIGINAL range (x+/-0.1, y+/-0.25)
    and linearly ramps to the WIDE range (CUBE_DX/DY_MIN/MAX env vars, default +/-0.15/+/-0.35) between
    CURR_START_ITER and CURR_END_ITER (env-var, default iter 200 -> 1600 of a 2500-iter run)."""

    def __post_init__(self):
        super().__post_init__()
        start_iter = int(os.environ.get("CURR_START_ITER", "200"))
        end_iter = int(os.environ.get("CURR_END_ITER", "1600"))
        x_end = (float(os.environ.get("CUBE_DX_MIN", "-0.15")), float(os.environ.get("CUBE_DX_MAX", "0.15")))
        y_end = (float(os.environ.get("CUBE_DY_MIN", "-0.35")), float(os.environ.get("CUBE_DY_MAX", "0.35")))
        # hold the base (non-curriculum) range at the ORIGINAL/easy range; curriculum ramps it during training
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
class FrankaCubeLiftDeltaCurrDREnvCfg_PLAY(FrankaCubeLiftDeltaCurrDREnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


# SIM2REAL TABLE HEIGHT FIX (2026-08-06): the sim table sits ~4.1cm lower than the real (matted) table --
# measured empirically: sim cube settles at center z=0.021 (table-top ~-0.004) vs real touch_probe EE_pos.z=0.037
# on the real table+mat. Raises BOTH the table and the cube's spawn point by the same amount (so the cube still
# spawns comfortably above the raised table, not inside it) -- the policy's own learned "how far down is safe"
# should then transfer more accurately to real hardware, instead of relying entirely on deploy-side Z_FLOOR
# guards fighting a policy that was trained against a lower surface. Env-var tunable via TABLE_RAISE.
TABLE_RAISE = float(os.environ.get("TABLE_RAISE", "0.041"))


@configclass
class FrankaCubeLiftDeltaDRTableFixEnvCfg(FrankaCubeLiftDeltaCurrDREnvCfg):
    """Delta + velocity-DR + cube-range + SIM2REAL TABLE HEIGHT FIX (table & cube spawn raised ~4.1cm to match
    the real matted table, measured via touch_probe 2026-08-06)."""

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
class FrankaCubeLiftDeltaDRTableFixEnvCfg_PLAY(FrankaCubeLiftDeltaDRTableFixEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class FrankaCubeLiftDeltaDRTableFixActionL2EnvCfg(FrankaCubeLiftDeltaDRTableFixEnvCfg):
    """Diagnostic (2026-08-14): adds a raw-action-magnitude penalty (mdp.action_l2, NOT clip) on top of
    TableFix, to test the action-saturation finding from job 29848155 (raw action logging showed 2/8 delta
    dims pinned at huge, one-signed magnitude -- e.g. mean -36, min -72 -- on the no-wrench delta checkpoint,
    vs a bounded +/-10-25 symmetric range on the working wrench checkpoint and the working absolute checkpoint).

    Deliberately NOT a clip on the action term or agent_cfg.clip_actions: RSL-RL's rollout computes
    log_prob/action_mean/action_sigma on the RAW sampled action in PPO.act() (rsl_rl/algorithms/ppo.py
    line ~144), strictly BEFORE env.step() applies any clip (on_policy_runner.py line ~224-226; clip_actions
    is applied even later, inside RslRlVecEnvWrapper.step()). Clipping downstream of that call creates a
    sampled-vs-applied mismatch: PPO's gradient is computed against the unclipped raw action while the
    reward reflects the clipped physical outcome, so once the raw mean drifts past the clip boundary nothing
    in the gradient pulls it back (increasing it further doesn't change the clipped outcome, but PPO can't
    see that) -- a known failure mode, and matches a prior real experience of a clipped-action policy
    failing to converge for the same reason. mdp.action_l2 penalizes env.action_manager.action directly,
    i.e. the SAME raw action log_prob is computed on -- no discrepancy, the thing being penalized is exactly
    the thing PPO is optimizing and exactly the thing actually scaled+applied as the joint delta.
    ACTION_L2_WEIGHT env-var tunable; default is a first guess (small relative to the ~36-unit task-reward
    weights), needs checking against a sanity run's actual reward-component scale.
    """

    def __post_init__(self):
        super().__post_init__()
        self.rewards.action_magnitude = RewardTermCfg(
            func=mdp.action_l2, weight=-float(os.environ.get("ACTION_L2_WEIGHT", "0.01"))
        )


@configclass
class FrankaCubeLiftDeltaDRTableFixActionL2EnvCfg_PLAY(FrankaCubeLiftDeltaDRTableFixActionL2EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class FrankaCubeLiftDeltaDRTableFixWrenchEnvCfg(FrankaCubeLiftDeltaDRTableFixEnvCfg):
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
class FrankaCubeLiftDeltaDRTableFixWrenchEnvCfg_PLAY(FrankaCubeLiftDeltaDRTableFixWrenchEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class FrankaCubeLiftDeltaDRTableFixSimpleEnvCfg(FrankaCubeLiftDeltaDRTableFixEnvCfg):
    """Diagnostic (2026-08-07/08): strips ALL domain randomization down to the simplest possible task on top
    of TableFix, mirroring FrankaCubeLiftAbsoluteDRTableFixSimpleEnvCfg. Fixed cube spawn (pose_range pinned
    to (0,0)/(0,0)), curriculum widening term removed. Velocity DR neutralized by pinning
    RL_VMAX_MIN=RL_VMAX_MAX at submit time (no code change needed there).
    """

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_object_position.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)}
        self.curriculum.widen_cube_range = None


@configclass
class FrankaCubeLiftDeltaDRTableFixSimpleEnvCfg_PLAY(FrankaCubeLiftDeltaDRTableFixSimpleEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class FrankaCubeLiftDeltaDRTableFixWrenchSimpleEnvCfg(FrankaCubeLiftDeltaDRTableFixSimpleEnvCfg):
    """Diagnostic (2026-08-07/08): EE wrench observation/reward on top of the Simple (no-DR) TableFix task,
    mirroring FrankaCubeLiftAbsoluteDRTableFixWrenchSimpleEnvCfg -- isolates whether the wrench addition
    itself converges cleanly, independent of DR. Fixed cube spawn + no curriculum inherited from Simple;
    velocity DR still needs RL_VMAX_MIN=RL_VMAX_MAX at submit time.
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
class FrankaCubeLiftDeltaDRTableFixWrenchSimpleEnvCfg_PLAY(FrankaCubeLiftDeltaDRTableFixWrenchSimpleEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False

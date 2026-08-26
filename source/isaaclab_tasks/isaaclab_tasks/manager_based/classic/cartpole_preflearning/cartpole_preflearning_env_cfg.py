# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cartpole variant with an added "preflog" observation group, for use as the simplest-possible
sanity-check task in the preference-learning investigation (see
FYP2025S1-2903_deployment_setup_guide.md, 2026-08-27 "Simpler-task controls" entry). Standard
Isaac-Cartpole-v0 (isaaclab_tasks.manager_based.classic.cartpole) has no "preflog" observation
group -- on_policy_runner.py's in-training trajectory collector hardcodes a read of
infos["observations"]["preflog"], so plain Cartpole would KeyError there. Rather than touch
on_policy_runner.py or the existing cartpole_env_cfg.py, this adds the missing group via
subclassing, entirely additively.

Cartpole's whole state (cart position, pole angle, cart velocity, pole angular velocity) is
already fully captured by the existing "policy" group (joint_pos_rel + joint_vel_rel over both
joints) -- this is a fully-observed 4-dim MDP, unlike Lift's partially-observed, much higher-
dimensional privileged state. The preflog group here simply mirrors those same two generic
IsaacLab MDP functions in un-concatenated (dict) form, plus a `current_action` term (mirrors
Lift's PrefLogCfg convention exactly) so get_trajectories.py's existing, unmodified
`log_data["current_action"]` read works against this task without any change there either.
"""

from isaaclab.envs import mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.classic.cartpole.cartpole_env_cfg import CartpoleEnvCfg


@configclass
class CartpolePrefLearningObservationsCfg:
    """Observation specifications: the existing PolicyCfg group, unmodified, plus a new PrefLogCfg
    group (dict-form, unconcatenated) for preference-learning trajectory collection.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """Identical to CartpoleEnvCfg's own PolicyCfg -- duplicated here (not imported) only
        because IsaacLab's @configclass observation groups are defined inline per env cfg; the
        values themselves (mdp.joint_pos_rel, mdp.joint_vel_rel) are the same generic, unmodified
        IsaacLab MDP functions the base task uses.
        """

        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class PrefLogCfg(ObsGroup):
        """Preference-learning trajectory logging group (added, mirrors the Franka Lift task's own
        PrefLogCfg convention -- see lift_env_cfg.py -- unconcatenated dict output).
        """

        obs_joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        obs_joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        current_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    preflog: PrefLogCfg = PrefLogCfg()


@configclass
class CartpolePrefLearningEnvCfg(CartpoleEnvCfg):
    """CartpoleEnvCfg (imported, unmodified) with the preflog observation group added."""

    observations: CartpolePrefLearningObservationsCfg = CartpolePrefLearningObservationsCfg()

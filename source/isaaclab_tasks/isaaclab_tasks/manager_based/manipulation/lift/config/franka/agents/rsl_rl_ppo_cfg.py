# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class LearnedRewardSettings:
    """Parameters for the preference-based learned reward model.

    Consumed by ``rsl_rl/runners/learned_reward.py`` via ``LearnedRewardCfg``
    when ``LiftCubePPORunnerCfg.use_learned_reward`` is True.
    Checkpoints are produced by ``preference_learning/train_reward_model.py``.
    """

    rm_checkpoint: str = ""
    """Absolute path to the reward model checkpoint (.pt file).

    The checkpoint is created by ``train_reward_model.py`` and saved to the
    experiment output directory, e.g.:
        /datasets/work/hri-fyp2025s1-2903/work/.../experiments/<run_name>/reward_model.pt

    The checkpoint embeds the architecture config (hidden sizes, obs_mode,
    architecture name), so the correct network is reconstructed automatically.
    """

    reward_weight: float = 1.0
    """Scalar multiplier applied to the learned reward before replacing the env reward.

    The raw model output is an unbounded scalar (or tanh-bounded for
    ``bounded_tanh`` / ``bounded_sigmoid`` architectures). Keep at 1.0 unless
    the learned reward scale needs to be matched to the env reward magnitude.
    """

    rm_checkpoints: str = ""
    """Optional comma-separated list of reward-model checkpoint paths for ensemble mode
    (added 2026-08-24 to combat reward-model overoptimization -- see
    FYP2025S1-2903_deployment_setup_guide.md 2026-08-18 entry). When non-empty, overrides
    ``rm_checkpoint`` and loads all listed checkpoints; the effective reward becomes
    ``mean(predictions) - ensemble_penalty * std(predictions)`` across the ensemble, so a
    policy can only earn high reward where the models agree, not by exploiting one model's
    individual blind spot. Leave empty (default) for the original single-model behaviour,
    unchanged from before this field existed.
    """

    ensemble_penalty: float = 1.0
    """Weight on the cross-model standard deviation subtracted from the ensemble mean
    (only used when ``rm_checkpoints`` is set). 0.0 = plain mean (no disagreement penalty);
    higher values are more conservative/pessimistic about disagreement.
    """


@configclass
class LiftCubePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 50
    experiment_name = "franka_lift"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

    # ── Learned reward injection ────────────────────────────────────────────
    # When True, the environment reward is replaced at every PPO rollout step
    # by the output of a pretrained preference-based reward model.
    # Handled by on_policy_runner.py → learned_reward.py → LearnedRewardWrapper.
    # Leave False for standard PPO with the environment's ground-truth reward.
    use_learned_reward: bool = False

    # Populated only when use_learned_reward = True.
    # Set rm_checkpoint to the .pt file produced by train_reward_model.py.
    # Example for the apr21a_baseline run (privileged obs, fragment_length=1):
    #   learned_reward = LearnedRewardSettings(
    #       rm_checkpoint="/datasets/work/.../experiments/apr21a_baseline/reward_model.pt",
    #       reward_weight=1.0,
    #   )
    learned_reward: LearnedRewardSettings = LearnedRewardSettings()

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

    kl_penalty_weight: float = 0.0
    """Weight on a PPO-side KL(policy || reference) penalty, added 2026-08-25 (see
    FYP2025S1-2903_deployment_setup_guide.md 2026-08-24/25 entry). Handled in
    on_policy_runner.py, which loads a full second (frozen) policy network from
    ``kl_reference_checkpoint`` -- not just a reward net -- so this directly bounds how far
    the policy is allowed to drift from a known-good reference, instead of relying on the
    reward model alone to police that. 0.0 = off (default, original behaviour).
    """

    kl_reference_checkpoint: str = ""
    """Frozen reference policy .pt checkpoint (e.g. a GT-oracle's final checkpoint) that PPO
    is penalized for drifting away from. Required when ``kl_penalty_weight`` > 0.
    """

    kl_anneal_to_iter: int = 0
    """Added 2026-08-26. If > 0, linearly decay the effective KL weight from
    ``kl_penalty_weight`` (iteration 0) to 0.0 (this iteration and beyond), instead of holding
    it constant for the whole run. 0 = off (default, original constant-weight behaviour).
    """

    collect_traj_dir: str = ""
    """Added 2026-08-26. Output dir for in-training preference-learning trajectory collection
    (see ``rsl_rl/runners/traj_collector.py``) -- empty = disabled, zero behaviour change.
    Collects the same privileged ``preflog`` data ``play_collect_pref_data.py`` collects, live
    during training, for a small set of envs during selected iteration windows, instead of a
    separate post-hoc checkpoint-replay pass.
    """

    collect_traj_env_ids: str = "0,1,2,3,4,5,6,7"
    """Comma-separated env indices to collect trajectories from."""

    collect_traj_iterations: str = ""
    """Comma-separated iteration numbers; each opens a fresh ``collect_traj_window``-iteration
    collection window for the selected envs. Empty = no windows ever open.
    """

    collect_traj_window: int = 25
    """Iterations a collection window stays open once triggered."""

    online_update_interval: int = -1
    """Added 2026-08-26, default changed to -1 same day (0 read ambiguously as "update every
    iteration" rather than "disabled"). -1 (or any value <= 0) = disabled, default, no behaviour
    change. If > 0, every time a ``traj_collector`` window closes, the just-collected data is
    ingested into a growing online pool, fresh comparisons are resampled from that pool, and the
    reward net is fine-tuned in
    place -- genuine iterative preference learning across the whole run (see
    ``rsl_rl/runners/online_reward_update.py``), instead of training the reward model once,
    offline, before a single fresh PPO run. For continuous coverage, set
    ``collect_traj_iterations`` to every multiple of this value up to the run length and
    ``collect_traj_window`` equal to it, so windows tile the run with no gaps. Requires
    ``collect_traj_dir``. Not supported together with an ensemble (``rm_checkpoints`` set).
    """

    online_finetune_epochs: int = 8
    """Fixed epoch count per online round (no early stopping)."""

    online_comparisons_per_round: int = 3000
    """Comparisons freshly resampled from the accumulated online pool every round."""

    online_lr: float = 3e-4
    """Learning rate for the online fine-tuning optimizer."""

    online_val_frac: float = 0.15
    """Trajectory-level split of the online pool for round-local monitoring only."""

    online_camera_config: str = "ground_truth"
    """Passed to ``load_from_meta_dataset`` for online windows; irrelevant for
    ``obs_mode=privileged``.
    """

    online_fixed_val_meta: str = ""
    """Optional path to the original offline ``test_meta.json``. If set, accuracy against this
    fixed set is tracked every online round, independent of the shifting online pool, so
    catastrophic forgetting from repeated fine-tuning is visible directly. Empty = skip.
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

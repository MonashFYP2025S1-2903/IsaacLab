# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass

from .distillation_cfg import RslRlDistillationAlgorithmCfg, RslRlDistillationStudentTeacherCfg
from .rnd_cfg import RslRlRndCfg
from .symmetry_cfg import RslRlSymmetryCfg

#########################
# Policy configurations #
#########################


@configclass
class RslRlPpoActorCriticCfg:
    """Configuration for the PPO actor-critic networks."""

    class_name: str = "ActorCritic"
    """The policy class name. Default is ActorCritic."""

    init_noise_std: float = MISSING
    """The initial noise standard deviation for the policy."""

    noise_std_type: Literal["scalar", "log"] = "scalar"
    """The type of noise standard deviation for the policy. Default is scalar."""

    std_clamp_min: float = -1.0
    """Lower bound for an optional clamp on the action-distribution std, applied every forward
    pass. Both this and std_clamp_max must be set positive (std_clamp_max > std_clamp_min) to
    activate -- the negative sentinel default preserves prior behavior exactly (no clamp) for
    reproducibility of existing experiments. Two plain scalars, not a tuple/Optional[list]: the
    IsaacLab Hydra CLI override merges a list override via len(existing_value), which crashes on a
    None-defaulted field (TypeError: object of type 'NoneType' has no len(), found 2026-08-30/31);
    plain floats don't hit that code path. Added 2026-08-30 as Arm E1 of the action-bounding
    comparison -- in "scalar" mode (this project's default) std has zero positivity guarantee (no
    exp()), so gradient descent can push it through zero, matching the "normal expects
    std >= 0.0" crash. See FYP2025S1-2903_deployment_setup_guide.md, 2026-08-30 cont. 9/12."""

    std_clamp_max: float = -1.0
    """Upper bound -- see std_clamp_min."""

    strip_last_action_from_actor_obs: bool = False
    """Arm P1 (added 2026-08-31): if True, the actor network never sees the trailing
    `mdp.last_action` block of its own observation group (every task's PolicyCfg in this project
    appends it as the final term). Distinct from R1/R2/R3 (which change what the FROZEN REWARD NET
    sees) and from E1/E2 (which bound the sampled action after the fact) -- this changes what the
    actor itself is conditioned on, on every task, independent of which reward-net variant is in
    use. Motivation: if the actor's own action distribution drifts wide (the failure mode E1
    targets), the actor is feeding its own increasingly erratic last action back into its own next
    input -- a feedback loop distinct from anything R1/E1/E2 address. Only the actor's input is
    sliced (in ActorCritic._actor_obs); the critic has its own independent flag (see
    strip_last_action_from_critic_obs below) and the reward net's input (which reads the same
    shared observation tensor -- see FYP2025S1-2903_deployment_setup_guide.md, 2026-08-31) is left
    untouched regardless. Default False preserves exact prior behavior. Plain bool, not an
    Optional/None-defaulted field, so it doesn't hit the Hydra CLI-override type-mismatch bug that
    affected clip_actions."""

    strip_last_action_from_critic_obs: bool = False
    """Arm C1 (added 2026-08-31, Lingheng: "you don't have plan to deal with the action fed into
    critic?"): if True, the critic network never sees the trailing `mdp.last_action` block of its
    own observation either. Independent of strip_last_action_from_actor_obs (P1) -- combine both
    for the "remove last_action from both actor AND critic" test Lingheng asked for, since P1
    alone leaves the critic's input untouched (confirmed via the printed Critic MLP input dim in
    every P1 run so far) and is therefore not a complete ablation of last_action from the network.
    Motivation distinct from P1's: tests whether the CRITIC's own value estimates are thrown off
    by an increasingly erratic last_action input feature (e.g. extrapolating badly once the
    reward-net exploit pushes it outside the range seen early in training), independent of the
    target-side value-function-loss-explosion mechanism found the same day (see
    FYP2025S1-2903_deployment_setup_guide.md cont. 16). Plain bool, same no-CLI-bug reasoning as
    strip_last_action_from_actor_obs."""

    feed_action_saturation_delta: bool = False
    """Arm A1 (added 2026-08-31, Lingheng's anti-windup idea: "raw and executed action... you can
    test the delta since there is no much difference"): if True, the actor receives an EXTRA input
    -- the delta between the raw action it sampled and the actually-executed (post-`clip_actions`)
    action from its previous step -- concatenated onto its (possibly P1-stripped) observation.
    Genuinely new information, not a slice of an existing field like P1/C1, so it widens the
    actor's input by num_actions rather than narrowing it. Requires `clip_actions` to be actively
    set (e.g. `agent.clip_actions=3.0`) for the delta to ever be nonzero -- with no clip active,
    raw action always equals executed action by construction (see
    RslRlVecEnvWrapper.last_action_saturation_delta). Computed in the wrapper (not appended to the
    shared `obs`/`obs_dict["policy"]` tensor there -- that would corrupt the frozen reward net's
    expected input shape) and handed to the policy directly by `OnPolicyRunner`'s rollout loop each
    step. Default False preserves exact prior behavior. Plain bool, same no-CLI-bug reasoning as
    strip_last_action_from_actor_obs."""

    tanh_squash: bool = False
    """Arm E3 (proposed 2026-08-30 cont. 9 as the rigorous member of the E1/E2/E3 action-bounding
    comparison; implemented 2026-08-31). If True, the actor samples raw z ~ Normal(mean, std) as
    usual but returns tanh(z) * tanh_action_scale as the actual action (and act_inference returns
    tanh(mean) * tanh_action_scale) -- SAC-paper-appendix-C squashing, mirroring
    sac_plpomdp.py's SquashedGaussianMLPActor. Unlike E2's post-hoc clip_actions (which creates a
    genuine train/execution log-prob mismatch: PPO's ratio uses the RAW sampled action's log-prob
    but the environment experiences the CLIPPED one), the action is in-bounds by construction here,
    so get_actions_log_prob computes the exact log-prob of the action actually executed, no
    mismatch. Default False preserves exact prior behavior. Plain bool, same no-CLI-bug reasoning
    as strip_last_action_from_actor_obs."""

    tanh_action_scale: float = 1.0
    """Scale applied after tanh when tanh_squash is True (action bound is +-tanh_action_scale).
    Inert when tanh_squash is False. Default 1.0; set to match this investigation's established
    clip_actions=3.0 convention (E2/A1) for an apples-to-apples bound comparison."""

    actor_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the actor network."""

    critic_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the critic network."""

    activation: str = MISSING
    """The activation function for the actor and critic networks."""


@configclass
class RslRlPpoActorCriticRecurrentCfg(RslRlPpoActorCriticCfg):
    """Configuration for the PPO actor-critic networks with recurrent layers."""

    class_name: str = "ActorCriticRecurrent"
    """The policy class name. Default is ActorCriticRecurrent."""

    rnn_type: str = MISSING
    """The type of RNN to use. Either "lstm" or "gru"."""

    rnn_hidden_dim: int = MISSING
    """The dimension of the RNN layers."""

    rnn_num_layers: int = MISSING
    """The number of RNN layers."""


############################
# Algorithm configurations #
############################


@configclass
class RslRlPpoAlgorithmCfg:
    """Configuration for the PPO algorithm."""

    class_name: str = "PPO"
    """The algorithm class name. Default is PPO."""

    num_learning_epochs: int = MISSING
    """The number of learning epochs per update."""

    num_mini_batches: int = MISSING
    """The number of mini-batches per update."""

    learning_rate: float = MISSING
    """The learning rate for the policy."""

    schedule: str = MISSING
    """The learning rate schedule."""

    gamma: float = MISSING
    """The discount factor."""

    lam: float = MISSING
    """The lambda parameter for Generalized Advantage Estimation (GAE)."""

    entropy_coef: float = MISSING
    """The coefficient for the entropy loss."""

    desired_kl: float = MISSING
    """The desired KL divergence."""

    max_grad_norm: float = MISSING
    """The maximum gradient norm."""

    value_loss_coef: float = MISSING
    """The coefficient for the value loss."""

    use_clipped_value_loss: bool = MISSING
    """Whether to use clipped value loss."""

    clip_param: float = MISSING
    """The clipping parameter for the policy."""

    normalize_advantage_per_mini_batch: bool = False
    """Whether to normalize the advantage per mini-batch. Default is False.

    If True, the advantage is normalized over the mini-batches only.
    Otherwise, the advantage is normalized over the entire collected trajectories.
    """

    symmetry_cfg: RslRlSymmetryCfg | None = None
    """The symmetry configuration. Default is None, in which case symmetry is not used."""

    rnd_cfg: RslRlRndCfg | None = None
    """The configuration for the Random Network Distillation (RND) module. Default is None,
    in which case RND is not used.
    """

    gradient_regularization_coef: float = 0.0
    """Gradient regularization strength (gamma), added 2026-09-01 for the FYP2025S1-2903
    reward-hacking investigation (Ackermann, Noukhovitch, Ishida, Sugiyama 2026, arXiv:2602.18037).
    Penalizes the squared norm of the policy-loss gradient, biasing PPO toward flatter optima the
    paper connects to higher reward-model accuracy -- an alternative to KL-anchoring a reference
    policy that needs no reference policy or ensemble. Default 0.0 is a hard no-op: zero behavior
    change, zero extra compute, unless explicitly set nonzero. NOT validated in combination with
    rnd_cfg/symmetry_cfg -- PPO.__init__ raises NotImplementedError if both are set.
    """

    gradient_regularization_eps: float = 1e-3
    """Finite-difference step size (epsilon) for the gradient-regularization second-order term.
    Only used when gradient_regularization_coef > 0. Default matches the paper's own validated
    value -- used unchanged across every one of their experiment families (RLHF, AlpacaFarm,
    GSM8K/MATH reasoning) and confirmed in their own sensitivity sweep (Appendix D.1) not to need
    per-task tuning, unlike gamma which they do tune per setting (their own sweeps span
    1e-1 to 1e-3, task-dependent). Not yet tuned specifically for this codebase's manipulation
    tasks -- still smoke-test before a real run; see PPO.update()'s gradient-regularization block
    for the exact formula.
    """


#########################
# Runner configurations #
#########################


@configclass
class RslRlOnPolicyRunnerCfg:
    """Configuration of the runner for on-policy algorithms."""

    seed: int = 42
    """The seed for the experiment. Default is 42."""

    device: str = "cuda:0"
    """The device for the rl-agent. Default is cuda:0."""

    num_steps_per_env: int = MISSING
    """The number of steps per environment per update."""

    max_iterations: int = MISSING
    """The maximum number of iterations."""

    empirical_normalization: bool = MISSING
    """Whether to use empirical normalization."""

    policy: RslRlPpoActorCriticCfg | RslRlDistillationStudentTeacherCfg = MISSING
    """The policy configuration."""

    algorithm: RslRlPpoAlgorithmCfg | RslRlDistillationAlgorithmCfg = MISSING
    """The algorithm configuration."""

    clip_actions: float = -1.0
    """The clipping value for actions. If <= 0 (default -1.0), then no clipping is done.

    Changed from `float | None = None` (2026-08-31): IsaacLab's own Hydra CLI-override merge
    (`isaaclab/utils/dict.py:update_class_from_dict()`) does strict runtime type-matching against
    the CURRENT value, not the annotated type -- so a None-defaulted field can never be overridden
    with a non-None type via CLI at all (`ValueError: Incorrect type under namespace: /clip_actions.
    Expected: <class 'NoneType'>, Received: <class 'float'>`, found 2026-08-31 trying
    `agent.clip_actions=3.0`). This is a genuine, pre-existing limitation, unrelated to anything
    added this investigation -- it's why no prior job script in this whole project ever set this
    field via CLI. A negative sentinel keeps the field's runtime type as `float` always, so CLI
    overrides type-match and actually work. `RslRlVecEnvWrapper.__init__` normalizes both this
    sentinel and a genuine `None` (for any caller still passing that directly) to "disabled"
    internally, so this change is safe for every existing caller
    (train.py/train_ppo_plpomdp.py/train_td3_plpomdp.py/train_sac_plpomdp.py/etc.) without needing
    to touch each of them individually.

    .. note::
        This clipping is performed inside the :class:`RslRlVecEnvWrapper` wrapper.
    """

    save_interval: int = MISSING
    """The number of iterations between saves."""

    experiment_name: str = MISSING
    """The experiment name."""

    run_name: str = ""
    """The run name. Default is empty string.

    The name of the run directory is typically the time-stamp at execution. If the run name is not empty,
    then it is appended to the run directory's name, i.e. the logging directory's name will become
    ``{time-stamp}_{run_name}``.
    """

    logger: Literal["tensorboard", "neptune", "wandb"] = "tensorboard"
    """The logger to use. Default is tensorboard."""

    neptune_project: str = "isaaclab"
    """The neptune project name. Default is "isaaclab"."""

    wandb_project: str = "isaaclab"
    """The wandb project name. Default is "isaaclab"."""

    resume: bool = False
    """Whether to resume. Default is False."""

    load_run: str = ".*"
    """The run directory to load. Default is ".*" (all).

    If regex expression, the latest (alphabetical order) matching run will be loaded.
    """

    load_checkpoint: str = "model_.*.pt"
    """The checkpoint file to load. Default is ``"model_.*.pt"`` (all).

    If regex expression, the latest (alphabetical order) matching file will be loaded.
    """

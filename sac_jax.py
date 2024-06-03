# ruff: noqa: E402
import multiprocessing as mp

from paths import *
import argparse
import io
import os
import random
import time
from collections import deque
from distutils.util import strtobool
from functools import partial
from typing import Deque, NamedTuple, Optional, Tuple, Union, Type
from argparse import Namespace
import distrax
import flax
import flax.linen as nn
import gymnasium as gym  # type: ignore
import jax
import jax.numpy as jnp
import metaworld  # type: ignore
import numpy as np
import numpy.typing as npt
import optax  # type: ignore
import orbax.checkpoint  # type: ignore
from cleanrl_utils.buffers_metaworld import ReplayBuffer
from cleanrl_utils.evals.metaworld_jax_eval import evaluation
from flax.training import orbax_utils
from flax.training.train_state import TrainState
from jax.typing import ArrayLike
from torch.utils.tensorboard import SummaryWriter
from video_language_critic.reward import RewardCalculator
import torch
import pickle
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
import metaworld
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv
from cleanrl_utils.wrappers import metaworld_wrappers


def load_vlc_args(vlc_ckpt):
    init_model_path = os.path.join(REWARD_CKPT_DIR, vlc_ckpt)
    vlc_args_path = os.path.join(init_model_path + '_config.pkl')
    with open(vlc_args_path, 'rb') as f:
        vlc_args = pickle.load(f)['args']
    vlc_args.init_model = init_model_path
    vlc_args.resume_from_latest = False
    return vlc_args

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default=None,
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--save-model", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to save model checkpoints")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="reach-v2", help="the name of the environment")
    parser.add_argument("--total-timesteps", type=int, default=int(5e6),
        help="total timesteps of the experiment")
    parser.add_argument("--max-episode-steps", type=int, default=500,
        help="maximum number of timesteps in one episode during training")
    parser.add_argument("--buffer-size", type=int, default=int(1e6),
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=0.005, help="target smoothing coefficient (default: 0.005)")
    parser.add_argument("--sac-batch-size", type=int, default=128,
                        help="the total size of the batch to sample from the replay memory. Must be divisible by number of tasks")
    parser.add_argument("--learning-starts", type=int, default=4e3, help="timestep to start learning")
    parser.add_argument("--evaluation-frequency", type=int, default=200_000,
        help="every how many timesteps to evaluate the agent. Evaluation is disabled if 0.")
    parser.add_argument("--evaluation-num-episodes", type=int, default=50,
        help="the number episodes to run per evaluation")
    parser.add_argument("--reward-normalization-offset", default=False, action="store_true", 
        help="use the values at the 12th step as the value added to each reward")
    parser.add_argument("--reward-normalization-gymnasium", default=False, action="store_true",
        help="normalize the rewards using the gymnasium logic")
    parser.add_argument("--reward-normalization-constant", default=False, action="store_true",
        help="add a constant to each reward")
    parser.add_argument("--predict-for-partial-videos", default=False, action="store_true", help="If true, use reward predictions for steps smaller than max frames. Else use 0.")
    parser.add_argument("--stretch-partial-videos", default=False, action="store_true", help="If true, repeat frames for videos smaller than max frames. Only applies when predicting for partial videos.")
    parser.add_argument("--env-reward-weight", type=float, default=0.,
        help="the weight of the original environment reward.")
    parser.add_argument("--sparse-reward-weight", type=float, default=0.,
        help="the weight of the sparse task success reward.")
    parser.add_argument("--vlm-reward-weight", type=float, default=1.,
        help="the weight of the predicted reward.")

    parser.add_argument("--reward-normalization-constant-value", type=float, default=None,
        help="the reward normalization constant to be added to the rewards")
    parser.add_argument("--reward-episode-end-only", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True, help="If True, use r=0 for non-terminal time steps.")

    # SAC
    parser.add_argument("--policy-lr", type=float, default=3e-4,
        help="the learning rate of the policy network optimizer")
    parser.add_argument("--q-lr", type=float, default=3e-4, help="the learning rate of the Q network network optimizer")
    parser.add_argument("--target-network-frequency", type=int, default=1,
        help="the frequency of updates for the target nerworks")
    parser.add_argument("--clip-grad-norm", type=float, default=0,
        help="the value to clip the gradient norm to. Disabled if 0. Not applied to alpha gradients.")
    parser.add_argument("--actor-network", type=str, default="400,400,400", help="The architecture of the actor network")
    parser.add_argument("--critic-network", type=str, default="400,400,400", help="The architecture of the critic network")
    parser.add_argument("--transition-logging-freq", type=int, default=50_000, help="How often to log data from training")

    # vlc
    parser.add_argument('--vlc-ckpt', type=str, default=None, help='Path to video language critic checkpoint under paths/REWARD_CKPT_DIR.')
    args = parser.parse_args()
    vlc_args = None
    if args.vlm_reward_weight != 0:
        vlc_args = load_vlc_args(args.vlc_ckpt)
        vlc_args = parser.parse_args(namespace=vlc_args)

        # fmt: on
        print("vlc args:")
        for key in sorted(vlc_args.__dict__):
            print("  <<< {}: {}".format(key, vlc_args.__dict__[key]))
    print("RL args:")
    for key in sorted(args.__dict__):
        print("  <<< {}: {}".format(key, args.__dict__[key]))
    return args, vlc_args


def _make_envs_common(
    benchmark: metaworld.Benchmark,
    seed: int,
    max_episode_steps: Optional[int] = None,
    terminate_on_success: bool = False,
    run_name: str = None,
) -> gym.vector.VectorEnv:
    def init_each_env(env_cls: Type[SawyerXYZEnv], name: str, env_id: int, run_name: str) -> gym.Env:
        env = env_cls(render_mode='rgb_array')
        env = gym.wrappers.TimeLimit(env, max_episode_steps or env.max_path_length)
        if terminate_on_success:
            env = metaworld_wrappers.AutoTerminateOnSuccessWrapper(env)
            trigger = lambda t: t % 10 == 0
            env = gym.wrappers.RecordVideo(env, os.path.join(EXP_DIR, f'runs/{run_name}/eval_videos'), episode_trigger=trigger)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        tasks = [task for task in benchmark.train_tasks if task.env_name == name]
        env = metaworld_wrappers.RandomTaskSelectWrapper(env, tasks)
        env.action_space.seed(seed)
        return env
    return gym.vector.AsyncVectorEnv(
        [
            partial(init_each_env, env_cls=env_cls, name=name, env_id=env_id, run_name=run_name)
            for env_id, (name, env_cls) in enumerate(benchmark.train_classes.items())
        ]
    )

def _transform(n_px):
        return Compose(
            [
                Resize(n_px, interpolation=Image.Resampling.BICUBIC),
                CenterCrop(n_px),
                lambda image: image.convert("RGB"),
                ToTensor(),
                Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
)


class Batch(NamedTuple):
    observations: ArrayLike
    actions: ArrayLike
    rewards: ArrayLike
    next_observations: ArrayLike
    dones: ArrayLike


class TanhNormal(distrax.Transformed):
    def __init__(self, loc, scale):
        normal_dist = distrax.Normal(loc, scale)
        tanh_bijector = distrax.Tanh()
        super().__init__(distribution=normal_dist, bijector=tanh_bijector)

    def mean(self):
        return self.bijector.forward(self.distribution.mean())


def uniform_init(bound: float):
    def _init(key, shape, dtype):
        return jax.random.uniform(
            key, shape=shape, minval=-bound, maxval=bound, dtype=dtype
        )

    return _init


class Actor(nn.Module):
    num_actions: int
    hidden_dims: int = "400,400,400"

    LOG_STD_MIN: float = -20.0
    LOG_STD_MAX: float = 2.0

    @nn.compact
    def __call__(self, x: jax.Array):
        hidden_lst = [int(dim) for dim in self.hidden_dims.split(",")]
        for i, h_size in enumerate(hidden_lst):
            x = nn.Dense(
                h_size,
                kernel_init=nn.initializers.he_uniform(),
                bias_init=nn.initializers.constant(0.1),
            )(x)
            x = nn.relu(x)

        log_sigma = nn.Dense(
            self.num_actions,
            kernel_init=uniform_init(1e-3),
            bias_init=uniform_init(1e-3),
        )(x)
        mu = nn.Dense(
            self.num_actions,
            kernel_init=uniform_init(1e-3),
            bias_init=uniform_init(1e-3),
        )(x)
        log_sigma = jnp.clip(log_sigma, self.LOG_STD_MIN, self.LOG_STD_MAX)
        return TanhNormal(mu, jnp.exp(log_sigma))


@jax.jit
def sample_action(
    actor: TrainState,
    obs: ArrayLike,
    key: jax.random.PRNGKeyArray,
) -> Tuple[jax.Array, jax.random.PRNGKeyArray]:
    key, action_key = jax.random.split(key)
    dist = actor.apply_fn(actor.params, obs)
    action = dist.sample(seed=action_key)
    return action, key


@jax.jit
def sample_and_log_prob(
    actor: TrainState,
    actor_params: flax.core.FrozenDict,
    obs: ArrayLike,
    key: jax.random.PRNGKeyArray,
) -> Tuple[jax.Array, jax.Array, jax.random.PRNGKeyArray]:
    key, action_key = jax.random.split(key)
    dist = actor.apply_fn(actor_params, obs)
    action, log_prob = dist.sample_and_log_prob(seed=action_key)
    return action, log_prob, key


@jax.jit
def get_deterministic_action(
    actor: TrainState,
    obs: ArrayLike,
) -> jax.Array:
    dist = actor.apply_fn(actor.params, obs)
    return dist.mean()


class Critic(nn.Module):
    hidden_dims: int = "400,400"

    @nn.compact
    def __call__(self, state, action):
        x = jnp.hstack([state, action])
        hidden_lst = [int(dim) for dim in self.hidden_dims.split(",")]
        for i, h_size in enumerate(hidden_lst):
            x = nn.Dense(
                h_size,
                kernel_init=nn.initializers.he_uniform(),
                bias_init=nn.initializers.constant(0.1),
            )(x)
            x = nn.relu(x)

        return nn.Dense(
            1, kernel_init=uniform_init(3e-3), bias_init=uniform_init(3e-3)
        )(x)


class VectorCritic(nn.Module):
    n_critics: int = 2
    hidden_dims: int = "400,400,400"

    @nn.compact
    def __call__(self, state: jax.Array, action: jax.Array) -> jax.Array:
        vmap_critic = nn.vmap(
            Critic,
            variable_axes={"params": 0},  # parameters not shared between the critics
            split_rngs={"params": True},  # different initializations
            in_axes=None,
            out_axes=0,
            axis_size=self.n_critics,
        )
        return vmap_critic(self.hidden_dims)(state, action)


class CriticTrainState(TrainState):
    target_params: Optional[flax.core.FrozenDict] = None


@jax.jit
def get_alpha(log_alpha: jax.Array) -> jax.Array:
    return jnp.exp(log_alpha)


class Agent:
    actor: TrainState
    critic: CriticTrainState
    alpha_train_state: TrainState
    target_entropy: float

    def __init__(
        self,
        init_obs: jax.Array,
        action_space: gym.spaces.Box,
        policy_lr: float,
        q_lr: float,
        gamma: float,
        clip_grad_norm: float,
        init_key: jax.random.PRNGKeyArray,
    ):
        self._action_space = action_space
        self._gamma = gamma

        random_action = jnp.array(
            [self._action_space.sample() for _ in range(init_obs.shape[0])]
        )

        def _make_optimizer(lr: float, max_grad_norm: float = 0.0):
            optim = optax.adam(learning_rate=lr)
            if max_grad_norm != 0:
                optim = optax.chain(
                    optax.clip_by_global_norm(max_grad_norm),
                    optim,
                )
            return optim

        actor_network = Actor(
            num_actions=int(np.prod(self._action_space.shape)),
            hidden_dims=args.actor_network,
        )
        key, actor_init_key = jax.random.split(init_key)
        self.actor = TrainState.create(
            apply_fn=actor_network.apply,
            params=actor_network.init(actor_init_key, init_obs),
            tx=_make_optimizer(policy_lr, clip_grad_norm),
        )

        _, qf_init_key = jax.random.split(key, 2)
        vector_critic_net = VectorCritic(
            hidden_dims=args.critic_network
        )
        self.critic = CriticTrainState.create(
            apply_fn=vector_critic_net.apply,
            params=vector_critic_net.init(
                qf_init_key, init_obs, random_action
            ),
            target_params=vector_critic_net.init(
                qf_init_key, init_obs, random_action
            ),
            tx=_make_optimizer(q_lr, clip_grad_norm),
        )

        self.alpha_train_state = TrainState.create(
            apply_fn=get_alpha,
            params=jnp.zeros(1),  # Log alpha
            tx=_make_optimizer(q_lr, max_grad_norm=0.0),
        )
        self.target_entropy = -np.prod(self._action_space.shape).item()

    def get_action_train(
        self, obs: np.ndarray, key: jax.random.PRNGKeyArray
    ) -> Tuple[np.ndarray, jax.random.PRNGKeyArray]:
        actions, key = sample_action(self.actor, obs, key)
        return jax.device_get(actions), key

    def get_action_eval(self, obs: np.ndarray) -> np.ndarray:
        actions = get_deterministic_action(self.actor, obs)
        return jax.device_get(actions)

    @staticmethod
    @jax.jit
    def soft_update(tau: float, critic_state: CriticTrainState) -> CriticTrainState:
        qf_state = critic_state.replace(
            target_params=optax.incremental_update(
                critic_state.params, critic_state.target_params, tau
            )
        )
        return qf_state

    def soft_update_target_networks(self, tau: float):
        self.critic = self.soft_update(tau, self.critic)

    def get_ckpt(self) -> dict:
        return {
            "actor": self.actor,
            "critic": self.critic,
            "alpha": self.alpha_train_state,
            "target_entropy": self.target_entropy,
        }


@partial(jax.jit, static_argnames=("gamma", "target_entropy"))
def update(
    actor: TrainState,
    critic: CriticTrainState,
    alpha: TrainState,
    batch: Batch,
    target_entropy: float,
    gamma: float,
    key: jax.random.PRNGKeyArray,
) -> Tuple[
    Tuple[TrainState, CriticTrainState, TrainState], dict, jax.random.PRNGKeyArray
]:
    next_actions, next_action_log_probs, key = sample_and_log_prob(
        actor, actor.params, batch.next_observations, key
    )
    q_values = critic.apply_fn(
        critic.target_params, batch.next_observations, next_actions
    )

    def critic_loss(params: flax.core.FrozenDict, alpha_val: jax.Array):
        min_qf_next_target = jnp.min(q_values, axis=0).reshape(
            -1, 1
        ) - alpha_val * next_action_log_probs.sum(-1).reshape(-1, 1)
        next_q_value = jax.lax.stop_gradient(
            batch.rewards + (1 - batch.dones) * gamma * min_qf_next_target
        )

        q_pred = critic.apply_fn(
            params, batch.observations, batch.actions
        )
        return 0.5 * ((q_pred - next_q_value) ** 2).mean(axis=1).sum(), q_pred.mean()

    def update_critic(
        _critic: CriticTrainState, alpha_val: jax.Array
    ) -> Tuple[CriticTrainState, dict]:
        (critic_loss_value, qf_values), critic_grads = jax.value_and_grad(
            critic_loss, has_aux=True
        )(_critic.params, alpha_val)
        _critic = _critic.apply_gradients(grads=critic_grads)
        return _critic, {
            "losses/qf_values": qf_values,
            "losses/qf_loss": critic_loss_value,
        }

    def alpha_loss(params: jax.Array, log_probs: jax.Array):
        return (-params.reshape(-1, 1) * (log_probs.sum(-1).reshape(-1, 1) + target_entropy)).mean()

    def update_alpha(
        _alpha: TrainState, log_probs: jax.Array
    ) -> Tuple[TrainState, jax.Array, jax.Array, dict]:
        alpha_loss_value, alpha_grads = jax.value_and_grad(alpha_loss)(
            _alpha.params, log_probs
        )
        _alpha = _alpha.apply_gradients(grads=alpha_grads)
        alpha_vals = _alpha.apply_fn(_alpha.params)
        return (
            _alpha,
            alpha_vals,
            {"losses/alpha_loss": alpha_loss_value, "alpha": jnp.exp(_alpha.params).sum()},  # type: ignore
        )

    key, actor_loss_key = jax.random.split(key)

    def actor_loss(params: flax.core.FrozenDict):
        action_samples, log_probs, _ = sample_and_log_prob(
            actor, params, batch.observations, actor_loss_key
        )
        _alpha, _alpha_val, alpha_logs = update_alpha(alpha, log_probs)
        _alpha_val = jax.lax.stop_gradient(_alpha_val)
        _critic, critic_logs = update_critic(critic, _alpha_val)
        logs = {**alpha_logs, **critic_logs}

        q_values = _critic.apply_fn(
            _critic.params, batch.observations, action_samples
        )
        min_qf_values = jnp.min(q_values, axis=0)
        return (_alpha_val * log_probs.sum(-1).reshape(-1, 1) - min_qf_values).mean(), (
            _alpha,
            _critic,
            logs,
        )

    (actor_loss_value, (alpha, critic, logs)), actor_grads = jax.value_and_grad(
        actor_loss, has_aux=True
    )(actor.params)
    actor = actor.apply_gradients(grads=actor_grads)

    return (actor, critic, alpha), {**logs, "losses/actor_loss": actor_loss_value}, key


class RunningMeanStd:
    """Tracks the mean, variance and count of values."""

    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        """Tracks the mean, variance and count of values."""
        self.mean = jnp.zeros(shape)
        self.var = jnp.ones(shape)
        self.count = epsilon

    def update(self, x):
        """Updates the mean, var and count from a batch of samples."""
        batch_mean = jnp.mean(x, axis=0)
        batch_var = jnp.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        """Updates from batch mean, variance and count moments."""
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )


def update_mean_var_count_from_moments(
    mean, var, count, batch_mean, batch_var, batch_count
):
    """Updates the mean, var and count using the previous mean, var, count and batch values."""
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + jnp.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


# Training loop
if __name__ == "__main__":
    mp.set_start_method('spawn')
    args, vlc_args = parse_args()
    run_name = f"{args.env_id}_{args.exp_name}"

    if args.track:
        run = wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
        # Add wandb ID to non-wandb run name.
        run_name += f'_wb_{run.id}'
    # Add seed to non-wandb run name.
    run_name += f'_s{args.seed}'
    writer = SummaryWriter(os.path.join(EXP_DIR, f"runs/{run_name}/summaries"))
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    if args.save_model:  # Orbax checkpoints
        ckpt_options = orbax.checkpoint.CheckpointManagerOptions(
            max_to_keep=3, create=True, best_fn=lambda x: x["charts/mean_success_rate"]
        )
        checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        ckpt_manager = orbax.checkpoint.CheckpointManager(
            os.path.join(EXP_DIR, f"runs/{run_name}/checkpoints"), checkpointer, options=ckpt_options
        )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)

    # env setup
    make_envs = partial(_make_envs_common, terminate_on_success=False)
    make_eval_envs = partial(_make_envs_common, terminate_on_success=True, run_name=run_name)

    if args.env_id == "MT10":
        benchmark = metaworld.MT10(seed=args.seed)
        args.num_envs = 10
    elif args.env_id == "MT50":
        benchmark = metaworld.MT50(seed=args.seed)
        args.num_envs = 50
    else:
        benchmark = metaworld.MT1(args.env_id, seed=args.seed)
        args.num_envs = 1

    envs = make_envs(
        benchmark, args.seed, args.max_episode_steps
    )
    eval_envs = make_eval_envs(
        benchmark, args.seed, args.max_episode_steps
    )

    NUM_TASKS = len(benchmark.train_classes)

    assert isinstance(
        envs.single_action_space, gym.spaces.Box
    ), "only continuous action space is supported"

    # agent setup
    rb = ReplayBuffer(
        total_capacity=args.buffer_size,
        envs=envs,
        use_torch=False,
        seed=args.seed,
    )

    global_episodic_return: Deque[float] = deque([], maxlen=20)
    global_episodic_length: Deque[int] = deque([], maxlen=20)

    obs, _ = envs.reset()
    key, agent_init_key = jax.random.split(key)
    agent = Agent(
        init_obs=obs,
        action_space=envs.single_action_space,
        policy_lr=args.policy_lr,
        q_lr=args.q_lr,
        gamma=args.gamma,
        clip_grad_norm=args.clip_grad_norm,
        init_key=key,
    )

    if args.vlm_reward_weight != 0:
        reward_model = RewardCalculator(args=vlc_args)
        reward_model.model.eval()

        frames = torch.zeros((args.num_envs, args.max_episode_steps, 3, 224, 224))

        with open(f'{CAPTION_PATH}/raw-captions.pkl', 'rb') as f:
            descriptions = pickle.load(f)
        task_desc = [v for k, v in descriptions.items() if 'success_videos__' + args.env_id in k][0][0]
        task_desc = " ".join(task_desc)
        print('Caption:', task_desc)
        del descriptions

        pairs_text, pairs_mask, pairs_segment, choice_video_ids = reward_model.dataloader._get_text(video_id=0, caption=task_desc)
        pairs_text, pairs_mask, pairs_segment, choice_video_ids = torch.from_numpy(np.asarray(pairs_text)).to('cuda:0'), torch.from_numpy(np.asarray(pairs_mask)).to('cuda:0'), torch.from_numpy(np.asarray(pairs_segment)).to('cuda:0'), torch.from_numpy(np.asarray(choice_video_ids)).to('cuda:0')
        new_images = torch.zeros((args.num_envs, args.max_episode_steps, 3, 224, 224))
    
        transform = _transform(224)

    gamma = args.gamma
    epsilon = 1e-8
    returns = jnp.zeros(args.num_envs)
    return_rms = RunningMeanStd(shape=())

    offset = None

    start_time = time.time()
    derivatives = np.asarray([0. for _ in range(NUM_TASKS)])
    last_rewards = None 
    last_actions = None
    logging = False
    eval_success_rate_history = {}
    # TRY NOT TO MODIFY: start the game
    for global_step in range(args.total_timesteps // NUM_TASKS):
        current_t = global_step % args.max_episode_steps
        total_steps = global_step * NUM_TASKS

        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array(
                [envs.single_action_space.sample() for _ in range(NUM_TASKS)]
            )
        else:
            actions, key = agent.get_action_train(obs, key)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, og_rewards, terminations, truncations, infos = envs.step(actions)
        if args.vlm_reward_weight != 0:
            current_frames = envs.call('render')

            for idx, f in enumerate(current_frames):
                frames[idx, current_t] = transform(Image.fromarray(np.rot90(np.rot90(f).astype(np.uint8))))
            batches = torch.zeros((args.num_envs, 1, reward_model.dataloader.max_frames, 1, 3, 224, 224))

            offset_timestep = 0 if args.predict_for_partial_videos else reward_model.dataloader.max_frames - 1
            if current_t >= offset_timestep:
                if current_t >= vlc_args.max_frames or args.stretch_partial_videos:
                    images = np.linspace(0, current_t, vlc_args.max_frames, dtype=int)
                else:
                    images = np.arange(current_t + 1)
                if vlc_args.frame_indices_to_use is not None and len(images) > len(vlc_args.frame_indices_to_use):
                    images = np.stack([images[j] for j in vlc_args.frame_indices_to_use])

                for i in range(args.num_envs):
                    curr_video = frames[i][images]
                    curr_video = curr_video.unsqueeze(1)
                    curr_video = curr_video.unsqueeze(0)
                    curr_video = curr_video.unsqueeze(0)
                    batches[i][0][:len(images)] = curr_video

                episode_finished = np.logical_or(terminations, truncations)
                if np.any(episode_finished) or not args.reward_episode_end_only:
                    with torch.no_grad():
                        num_frames = len(images)
                        num_padded = reward_model.dataloader.max_frames - num_frames
                        video_mask = torch.from_numpy(np.asarray([1] * num_frames + [0] * num_padded)).unsqueeze(0).unsqueeze(0).repeat(args.num_envs, 1, 1).to('cuda:0')
                        a, b = reward_model.model.get_sequence_visual_output(pairs_text, pairs_mask, pairs_segment, batches.to('cuda:0'), video_mask)
                        scores = reward_model.model.get_similarity_logits(a, b, pairs_text, video_mask, loose_type=reward_model.model.loose_type)[0]
                    if len(scores.shape) > 2:
                        video_lengths = torch.argmax(torch.logical_not(video_mask).int(), dim=2).squeeze(1) - 1
                        final_scores = torch.zeros(args.num_envs)
                        for i in range(args.num_envs):
                            final_scores[i] = scores[0, i, video_lengths[i]]
                        scores = final_scores
                    rewards = scores.cpu().numpy()
                    if args.reward_episode_end_only:
                        rewards = np.multiply(rewards, episode_finished)
                else:
                    rewards = np.zeros_like(og_rewards)
                og_vlm_rewards = rewards.copy()

                if args.reward_normalization_offset:
                    if current_t == offset_timestep:
                        offset = rewards.copy()
                        rewards -= offset
                    elif current_t > offset_timestep:
                        rewards -= offset
                if args.reward_normalization_constant:
                    assert args.reward_normalization_constant_value is not None, "Need to set the constant to add via args"
                    rewards += args.reward_normalization_constant_value
                if args.reward_normalization_gymnasium:
                    terminated = 1 - terminations
                    returns = returns * gamma * (1 - terminated) + rewards
                    return_rms.update(returns)
                    rewards = rewards / jnp.sqrt(return_rms.var + epsilon)
                    rewards = np.asarray(rewards)
                if last_rewards is None and current_t != 0:
                    last_rewards = rewards.copy()
                    last_actions = np.asarray(actions).copy()[:, :3]
                elif current_t == 0:
                    last_rewards = None
                    last_actions = None
                else:
                    dist = np.linalg.norm(last_actions - np.asarray(actions)[:, :3], axis=1)
                    derivatives += (rewards.squeeze() - last_rewards.squeeze()) / dist

            else:
                rewards = np.asarray([0. for _ in range(NUM_TASKS)])
                og_vlm_rewards = rewards.copy()

            if global_step % args.transition_logging_freq == 0:
                logging = True
                start_step = global_step
                to_log = ['actions', 'raw_vlm_reward', 'vlm_reward', 'sparse_reward', 'success', 'original_reward',
                      'total_reward', 'state', 'next_state', 'termination', 'frames']
                episode_dict = {key: [] for key in to_log}

            if logging:
                writer.add_scalar("charts/reward_original", np.mean(og_rewards), global_step)
            rewards = rewards * args.vlm_reward_weight
            vlm_rewards = rewards.copy()
            if logging:
                writer.add_scalar("charts/reward_vlm", np.mean(rewards), global_step)
            success = None
            if 'success' in infos:
                success = infos['success'] * args.sparse_reward_weight
                rewards = rewards + success
                if logging:
                    writer.add_scalar("charts/reward_success", np.mean(success), global_step)
            rewards = rewards + og_rewards * args.env_reward_weight
        else:
            rewards = og_rewards * args.env_reward_weight
            if 'success' in infos:
                success = infos['success'] * args.sparse_reward_weight
            else:
                success = np.zeros_like(rewards)
            rewards = rewards + success

        if logging:
            writer.add_scalar("charts/reward_total", np.mean(rewards), global_step)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                # Skip the envs that are not done
                if info is None:
                    continue
                global_episodic_return.append(info["episode"]["r"])
                global_episodic_length.append(info["episode"]["l"])

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, d in enumerate(truncations):
            if d:
                real_next_obs[idx] = infos["final_observation"][idx]

        # Store data in the buffer
        rb.add(obs, real_next_obs, actions, rewards, terminations)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        if global_step % 500 == 0 and global_episodic_return:
            print(
                f"global_step={total_steps}, mean_episodic_return={np.mean(list(global_episodic_return))}"
            )
            writer.add_scalar(
                "charts/mean_episodic_return",
                np.mean(list(global_episodic_return)),
                total_steps,
            )
            writer.add_scalar(
                "charts/mean_episodic_length",
                np.mean(list(global_episodic_length)),
                total_steps,
            )

        # ALGO LOGIC: training.
        if global_step >= args.learning_starts:
            # Sample a batch from replay buffer
            data = rb.sample(args.sac_batch_size)
            batch = Batch(
                data.observations,
                data.actions,
                data.rewards,
                data.next_observations,
                data.dones,
            )

            # Update agent
            (agent.actor, agent.critic, agent.alpha_train_state), logs, key = update(
                agent.actor,
                agent.critic,
                agent.alpha_train_state,
                batch,
                agent.target_entropy,
                args.gamma,
                key,
            )
            logs = jax.device_get(logs)

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                agent.soft_update_target_networks(args.tau)

            # Logging
            if global_step % 1000 == 0:
                for _key, value in logs.items():
                    writer.add_scalar(_key, value, total_steps)
                writer.add_scalar(
                    "charts/SPS",
                    int(total_steps / (time.time() - start_time)),
                    total_steps,
                )

            # Evaluation
            if total_steps % args.evaluation_frequency == 0 and global_step > 0:
                (
                    eval_success_rate,
                    eval_returns,
                    eval_success_per_task,
                ) = evaluation(
                    agent=agent,
                    eval_envs=eval_envs,
                    num_episodes=args.evaluation_num_episodes,
                )
                eval_metrics = {
                    "charts/mean_success_rate": float(eval_success_rate),
                    "charts/mean_evaluation_return": float(eval_returns),
                } | {
                    f"charts/{env_name}_success_rate": float(eval_success_per_task[i])
                    for i, (env_name, _) in enumerate(benchmark.train_classes.items())
                }

                for k, v in eval_metrics.items():
                    writer.add_scalar(k, v, total_steps)
                print(
                    f"total_steps={total_steps}, mean evaluation success rate: {eval_success_rate:.4f}"
                    + f" return: {eval_returns:.4f}"
                )

                # Checkpointing
                if args.save_model:
                    ckpt = agent.get_ckpt()
                    ckpt["rng_key"] = key
                    ckpt["global_step"] = global_step
                    save_args = orbax_utils.save_args_from_target(ckpt)
                    ckpt_manager.save(
                        step=global_step,
                        items=ckpt,
                        save_kwargs={"save_args": save_args},
                        metrics=eval_metrics,
                    )
                    print(f"model saved to {ckpt_manager.directory}")

                eval_success_rate_history[total_steps] = float(eval_success_rate)
                recent_history = {k: v for k, v in eval_success_rate_history.items() if k >= total_steps - 10 * args.evaluation_frequency}
                recent_success_rates = list(recent_history.values())
                print("Recent success rates")
                for k, v in sorted(recent_history.items()):
                    print(f"{k}: {v}")
                print(f"mean: {np.mean(recent_success_rates)}, median: {np.median(recent_success_rates)}", )
                if total_steps > 10 * args.evaluation_frequency and np.mean(recent_success_rates) >= 0.98 and np.median(recent_success_rates) == 1 and float(eval_success_rate) >= 0.9:
                    print("Terminating early")
                    break
                else:
                    print("Not terminating early")


    envs.close()
    writer.close()

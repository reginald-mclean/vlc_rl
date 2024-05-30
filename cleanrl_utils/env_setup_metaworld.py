# ruff: noqa: E402
from functools import partial
from typing import Optional, Type

import gymnasium as gym  # type: ignore
import metaworld  # type: ignore
from cleanrl_utils.wrappers import metaworld_wrappers
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv  # type: ignore


def _make_envs_common(
    benchmark: metaworld.Benchmark,
    seed: int,
    max_episode_steps: Optional[int] = None,
    use_one_hot: bool = True,
    terminate_on_success: bool = False,
) -> gym.vector.VectorEnv:
    def init_each_env(env_cls: Type[SawyerXYZEnv], name: str, env_id: int) -> gym.Env:
        env = env_cls()
        env = gym.wrappers.TimeLimit(env, max_episode_steps or env.max_path_length)
        if terminate_on_success:
            env = metaworld_wrappers.AutoTerminateOnSuccessWrapper(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if use_one_hot:
            env = metaworld_wrappers.OneHotWrapper(
                env, env_id, len(benchmark.train_classes)
            )
        tasks = [task for task in benchmark.train_tasks if task.env_name == name]
        env = metaworld_wrappers.RandomTaskSelectWrapper(env, tasks)
        env.action_space.seed(seed)
        return env

    return gym.vector.AsyncVectorEnv(
        [
            partial(init_each_env, env_cls=env_cls, name=name, env_id=env_id)
            for env_id, (name, env_cls) in enumerate(benchmark.train_classes.items())
        ]
    )


make_envs = partial(_make_envs_common, terminate_on_success=False)
make_eval_envs = partial(_make_envs_common, terminate_on_success=True)

__all__ = ["make_envs", "make_eval_envs"]

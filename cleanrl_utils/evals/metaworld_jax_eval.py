# ruff: noqa: E402
import time
from typing import List, Optional, Tuple

import gymnasium as gym
import jax
import numpy as np
import numpy.typing as npt
import jax.numpy as jnp
import distrax


def evaluation(
    agent,
    eval_envs: gym.vector.VectorEnv,
    num_episodes: int,
    task_names: Optional[List[str]] = None,
) -> Tuple[float, float, npt.NDArray, jax.random.PRNGKey]:
    print(f"Evaluating for {num_episodes} episodes.")
    obs, _ = eval_envs.reset()

    if task_names is not None:
        successes = {task_name: 0 for task_name in set(task_names)}
        episodic_returns = {task_name: [] for task_name in set(task_names)}
        envs_per_task = {task_name: task_names.count(task_name) for task_name in set(task_names)}
    else:
        successes = np.zeros(eval_envs.num_envs)
        episodic_returns = [[] for _ in range(eval_envs.num_envs)]

    start_time = time.time()

    def eval_done(returns):
        if type(returns) is dict:
            return all(len(r) >= (num_episodes * envs_per_task[task_name]) for task_name, r in returns.items())
        else:
            return all(len(r) >= num_episodes for r in returns)

    while not eval_done(episodic_returns):
        actions = agent.get_action_eval(obs)
        obs, _, _, _, infos = eval_envs.step(actions)
        if "final_info" in infos:
            for i, info in enumerate(infos["final_info"]):
                # Skip the envs that are not done
                if info is None:
                    continue
                if task_names is not None:
                    episodic_returns[task_names[i]].append(float(info["episode"]["r"][0]))
                    if len(episodic_returns[task_names[i]]) <= num_episodes * envs_per_task[task_names[i]]:
                        successes[task_names[i]] += int(info["success"])
                else:
                    episodic_returns[i].append(float(info["episode"]["r"][0]))
                    if len(episodic_returns[i]) <= num_episodes:
                        successes[i] += int(info["success"])

    if type(episodic_returns) is dict:
        episodic_returns = {
            task_name: returns[: (num_episodes * envs_per_task[task_name])]
            for task_name, returns in episodic_returns.items()
        }
    else:
        episodic_returns = [returns[:num_episodes] for returns in episodic_returns]

    print(f"Evaluation time: {time.time() - start_time:.2f}s")

    if type(successes) is dict:
        success_rate_per_task = np.array(
            [successes[task_name] / (num_episodes * envs_per_task[task_name]) for task_name in set(task_names)]
        )
        mean_success_rate = np.mean(success_rate_per_task)
        mean_returns = np.mean(list(episodic_returns.values()))
    else:
        success_rate_per_task = successes / num_episodes
        mean_success_rate = np.mean(success_rate_per_task)
        mean_returns = np.mean(episodic_returns)

    return mean_success_rate, mean_returns, success_rate_per_task

from typing import Callable, List, NamedTuple, Optional, Union

import gymnasium as gym
import numpy as np
import numpy.typing as npt
import scipy  # type: ignore
import torch

TorchOrNumpyArray = Union[torch.Tensor, npt.NDArray]


class ReplayBufferSamples(NamedTuple):
    observations: TorchOrNumpyArray
    actions: TorchOrNumpyArray
    next_observations: TorchOrNumpyArray
    dones: TorchOrNumpyArray
    rewards: TorchOrNumpyArray

class ReplayBuffer:
    obs: npt.NDArray
    actions: npt.NDArray
    rewards: npt.NDArray
    next_obs: npt.NDArray
    dones: npt.NDArray
    pos: int

    def __init__(
        self,
        total_capacity: int,
        envs: gym.vector.VectorEnv,
        use_torch: bool = True,
        device: str = "cpu",
        seed: Optional[int] = None,
    ):
        self.capacity = total_capacity
        self.use_torch = use_torch
        self.device = device
        self._rng = np.random.default_rng(seed)
        self._obs_shape = np.array(envs.single_observation_space.shape).prod()
        self._action_shape = np.array(envs.single_action_space.shape).prod()
        self.full = False


        self.reset()  # Init buffer

    def reset(self):
        """Reinitialize the buffer."""
        self.obs = np.zeros((self.capacity, self._obs_shape), dtype=np.float32)
        self.actions = np.zeros((self.capacity, self._action_shape), dtype=np.float32)
        self.rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, self._obs_shape), dtype=np.float32)
        self.dones = np.zeros((self.capacity, 1), dtype=np.float32)
        self.pos = 0

    def add(self, obs: npt.NDArray, next_obs: npt.NDArray, action: npt.NDArray, reward: npt.NDArray, done: npt.NDArray):
        """Add a batch of samples to the buffer."""

        self.obs[self.pos] = obs.copy()
        self.actions[self.pos] = action.copy()
        self.rewards[self.pos] = reward.copy().reshape(-1, 1)
        self.next_obs[self.pos] = next_obs.copy()
        self.dones[self.pos] = done.copy().reshape(-1, 1)
        

        self.pos += 1

        if self.pos == self.capacity:
            self.full = True

        self.pos = self.pos % self.capacity

    def sample(self, batch_size: int) -> ReplayBufferSamples:
        """Sample a batch of size `single_task_batch_size`

        Args:
            batch_size (int)

        Returns:
            ReplayBufferSamples: A batch of samples of batch shape (batch_size,).
        """
        sample_idx = self._rng.integers(
            low=0,
            high=max(self.pos if not self.full else self.capacity, batch_size),
            size=(batch_size,),
        )

        batch = (
            self.obs[sample_idx],
            self.actions[sample_idx],
            self.next_obs[sample_idx],
            self.dones[sample_idx],
            self.rewards[sample_idx],
        )

        if self.use_torch:
            batch = map(lambda x: torch.tensor(x).to(self.device), batch)  # type: ignore

        return ReplayBufferSamples(*batch)

import warnings
from typing import Any, Dict, Optional, Type, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from diff_rl.common.buffers import RolloutBuffer
from diff_rl.common.on_policy_algorithm import OnPolicyAlgorithm
from diff_rl.common.policies import Diffusion_ActorCriticPolicy

from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, get_schedule_fn

class Diffusion_RL(OnPolicyAlgorithm):
    
    def __init__(
        self,
        env: Union[GymEnv, str],
        policy: Union[str, Type[Diffusion_ActorCriticPolicy]] = Diffusion_ActorCriticPolicy,
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        normalize_advantage: bool = True,
        max_grad_norm: float = 0.5,
        rollout_buffer_class: Optional[Type[RolloutBuffer]] = None,
        rollout_buffer_kwargs: Optional[Dict[str, Any]] = None,
        target_kl: Optional[float] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            max_grad_norm=max_grad_norm,
            rollout_buffer_class=rollout_buffer_class,
            rollout_buffer_kwargs=rollout_buffer_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            seed=seed,
            _init_setup_model=False,
        )

        # Sanity check, otherwise it will lead to noisy gradient and NaN
        # because of the advantage normalization
        if normalize_advantage:
            assert (
                batch_size > 1
            ), "`batch_size` must be greater than 1. See https://github.com/DLR-RM/stable-baselines3/issues/440"

        if self.env is not None:
            # Check that `n_steps * n_envs > 1` to avoid NaN
            # when doing advantage normalization
            buffer_size = self.env.num_envs * self.n_steps
            assert buffer_size > 1 or (
                not normalize_advantage
            ), f"`n_steps * n_envs` must be greater than 1. Currently n_steps={self.n_steps} and n_envs={self.env.num_envs}"
            # Check that the rollout buffer size is a multiple of the mini-batch size
            untruncated_batches = buffer_size // batch_size
            if buffer_size % batch_size > 0:
                warnings.warn(
                    f"You have specified a mini-batch size of {batch_size},"
                    f" but because the `RolloutBuffer` is of size `n_steps * n_envs = {buffer_size}`,"
                    f" after every {untruncated_batches} untruncated mini-batches,"
                    f" there will be a truncated mini-batch of size {buffer_size % batch_size}\n"
                    f"We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.\n"
                    f"Info: (n_steps={self.n_steps} and n_envs={self.env.num_envs})"
                )
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.normalize_advantage = normalize_advantage
        self.target_kl = target_kl

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()

    def train(self) -> None:

        self.policy.train(True)
        self._update_learning_rate(self.policy.optimizer)
        pg_losses = []
        q_value_losses = []
        continue_training = True
        
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions # actions is the selected_action
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                states = rollout_data.observations # s
                next_states = rollout_data.next_observations # s'
                next_values_pred = self.policy.value_estimation(next_states) # V(s')
                actions = actions.unsqueeze(1).expand(-1, self.policy.n_actions, -1) # a

                # TODO，for diffusion model, the loss should be the distance between the optimal actions and all other actions
                # difference between current optimal actions and reconstruction
                # could introduce probability later
                # actions are single but self.policy.diffusion_policy(states) is multi-actions    

                # TODO, the loss function is the difference of action itself it's a bit weird
                # TODO, should we take the currect optimal action as the optimal action?
                # TODO, or make it as standard diffusion loss, i.e., loss w.r.t reconstruction
                # actions here are the actions from rollout, which shouldn't be the target actions
                # For diffusion policy: take the optimal action as the ground truth
                # TODO, change pg_loss to advantage w.r.t current state and all actions
                # TODO, this loss only related to diffusion network
                pg_loss = F.mse_loss(actions, self.policy.diffusion_policy(states)) # Only this part has to do with policy

                # TODO, for Q_value network, the loss should be the distance between 
                # This loss w.r.t. both diffusion and policy network
                # TD_target = r + p(s', a') * Q(s', a') - R | equals to R - V
                # q_value_loss = F.mse_loss(rollout_data.returns, rollout_data.rewards + next_values_pred.squeeze()) # r + V(s') # first nothing to do with policy, but then does - R

                # TEST:
                # pg_loss.backward()
                # for param in self.policy.diffusion_policy.parameters(): # diffusion_policy
                #     print(param.grad)

                # Logging
                pg_losses.append(pg_loss.item())
                # q_value_losses.append(q_value_loss.item())

                # loss = pg_loss + q_value_loss
                loss = pg_loss

                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")

    def learn(
        self,
        total_timesteps: int = None,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "Diffusion_RL",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ):
        return super().learn(
            total_timesteps=self.n_envs * self.n_steps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )
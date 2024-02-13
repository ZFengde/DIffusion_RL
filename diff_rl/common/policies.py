"""Policies: abstract base class and concrete implementations."""

import collections
import copy
import warnings
from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch import nn
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.preprocessing import get_action_dim, get_flattened_obs_dim

class ActorCriticPolicy(nn.Module): 
    # This Q-value critic network meant to generate q-value for (s, a1, a2,...)
    def __init__(
        self,
        n_env,
        env,
        net_arch,
        activation_fn = nn.ReLU, 
        n_critics: int = 2 
    ):
        super().__init__()

        self.n_env = n_env
        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.n_critics = n_critics

        self.action_dim = get_action_dim(env.action_space)
        self.obs_dim = get_flattened_obs_dim(env.observation_space)

    def build(self):
        # Q-value networks construction
        self.q_networks = []
        for idx in range(self.n_critics):
            q_net_list = self.create_mlp(self.obs_dim + self.action_dim, 1, self.net_arch, self.activation_fn)
            q_net = nn.Sequential(*q_net_list)
            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)
        
        # Diffusion policy network construction
            self.diffusion_policy = 1

    def forward(self, obs):
        actions = self.diffusion_policy(obs)

        # Learn the features extractor using the policy loss only
        qvalue_input = self.pre_process(obs, actions)
        qvalue_output = tuple(q_net(qvalue_input) for q_net in self.q_networks)
        qvalue_final = th.max(qvalue_output[0], qvalue_output[1])
        action_indexes = th.argmax(qvalue_final, dim=1).squeeze()
        chosen_action = actions[th.arange(self.n_env), action_indexes]

        return chosen_action, qvalue_final

    def q1_forward(self, obs, actions):
        """
        Only predict the Q-value using the first network.
        This allows to reduce computation when all the estimates are not needed
        (e.g. when updating the policy in TD3).
        """
        return self.q_networks[0](th.cat([obs, actions], dim=1))
    
    def create_mlp(self, input_dim, output_dim, net_arch, activation_fn = nn.ReLU):
        if len(net_arch) > 0:
            modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
        else:
            modules = []

        for idx in range(len(net_arch) - 1):
            modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
            modules.append(activation_fn())

        
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))
        return modules

    def pre_process(self, obs, actions):
        obs = th.tensor(obs, dtype=th.float)
        actions = th.tensor(actions, dtype=th.float)
        # The dimensions have to be num_env, num_actions, obs_dim/action_dim
        obs = obs.unsqueeze(1).expand(-1, actions.size()[1], -1)
        qvalue_input = th.cat((obs, actions), dim=-1)

        return qvalue_input
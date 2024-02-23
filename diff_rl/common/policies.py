"""Policies: abstract base class and concrete implementations."""

import numpy as np
import torch as th
from torch import nn
from stable_baselines3.common.preprocessing import get_action_dim, get_flattened_obs_dim
from diff_rl.common.diffusion_policy import Diffusion_Policy, Q_networks, Networks, select_action

class Diffusion_ActorCriticPolicy(nn.Module): 

    def __init__(
        self,
        n_env,
        env,
        net_arch,
        n_actions,
        activation_fn = nn.ReLU, 
        n_critics: int = 2 
    ):
        super().__init__()

        self.n_actions = n_actions
        self.env = env
        self.n_env = n_env
        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.n_critics = n_critics

        self.action_dim = get_action_dim(self.env.action_space)
        self.obs_dim = get_flattened_obs_dim(self.env.observation_space)

        self.build()

    def build(self):
        # Q-value networks construction
        self.q_networks = Q_networks(env=self.env, n_actions=self.n_actions)
        
        # Diffusion policy network construction
        self.diffusion_policy = Diffusion_Policy(env=self.env, model=Networks, n_actions=100)

    def forward(self, state):
        actions = self.diffusion_policy(state)

        # TODO, could try other form of the values, no need to be q_min
        q_values = self.q_networks.q_min(state, actions) # q_values w.r.t each env-state: n_actions
        selected_actions = select_action(actions, q_values)

        # selected_actions: actions from all possible actions but with maxium q-value
        # actions: different actions generated with different gaussian noise
        # q_values for all possible actions
        return selected_actions, actions, q_values
    
    def q_value_estimation(self, state): # this is for estimating q_values for a given states, Q(s', a')
        actions = self.diffusion_policy(state) # s, a
        q_values = self.q_networks.q_min(state, actions) # q_values w.r.t each env-state: n_actions
        return q_values

    def q_value_evaluation(self, state, actions): # This is for evaluating q_values for a given s, a pair, Q(s, a)
        q_values = self.q_networks.q_min(state, actions) # q_values w.r.t each env-state: n_actions
        return q_values



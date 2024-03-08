"""Policies: abstract base class and concrete implementations."""

from gymnasium import spaces
import numpy as np
import torch as th
from torch import nn
from torch.distributions.normal import Normal
from stable_baselines3.common.preprocessing import get_action_dim, get_flattened_obs_dim
from diff_rl.common.diffusion_policy import Diffusion_Policy, Q_networks, Networks, select_action

class Diffusion_ActorCriticPolicy(nn.Module): 

    def __init__(
        self,
        env,
        n_actions=100,
        activation_fn = nn.ReLU, 
        n_critics: int = 2 
    ):
        super().__init__()

        self.n_actions = n_actions
        self.env = env
        self.register_buffer('action_high', th.tensor(self.env.action_space.high))
        self.register_buffer('action_low', th.tensor(self.env.action_space.low))

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
        self.optimizer = th.optim.Adam(self.parameters(), lr=0.001)
        self.gaussian_distribution = Normal(0.0, 1.0)
        
    def forward(self, state): # Multi-actions here, the value output is an expectation for all actions
        all_actions = self.diffusion_policy(state)
        if isinstance(self.env.action_space, spaces.Box):
            clipped_all_actions = th.clip(all_actions, self.action_low, self.action_high)

        # TODO, try to invlove with probability
        q_values = self.q_networks.multi_q_min(state, clipped_all_actions) # q_values w.r.t each env-state: n_actions and take mean
        # TODO, change values to q_values here, check why values work
        selected_actions = select_action(clipped_all_actions, q_values)
        values = q_values.mean(dim=1) # TODO, here should make values estimation involved with probability
        # selected_actions: actions from all possible actions but with maxium q-value
        # actions: different actions generated w
        # q-values w.r.t. all possible actions
        return selected_actions, clipped_all_actions, values
    
    def value_estimation(self, state): # Multi-actions here, this is for estimating values for a given states, Q(s', a'), should be an average
        actions = self.diffusion_policy(state).detach() # s, a, detach make it not back propagation
        values = self.q_networks.multi_q_min(state, actions).mean(dim=1) # q_values w.r.t each env-state: n_actions
        return values

    def q_value_evaluation(self, state, action): # Single cliped action here, this is for evaluating q_values for a given s, a pair, Q(s, a)
        q_values = self.q_networks.sinlge_q_min(state, action) # q_values w.r.t each env-state: n_actions
        return q_values




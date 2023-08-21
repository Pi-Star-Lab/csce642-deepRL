# Licensing Information:  You are free to use or extend this codebase for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide the following
# attribution:
# This CSCE-689 RL assignment codebase was developed at Texas A&M University.
# The core code base was developed by Guni Sharon (guni@tamu.edu).
# The PyTorch code was developed by Sheelabhadra Dey (sheelabhadra@tamu.edu).

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.optim import AdamW

from Solvers.Abstract_Solver import AbstractSolver
from lib import plotting


class ActorCriticNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes):
        super().__init__()
        sizes = [obs_dim] + hidden_sizes + [act_dim]
        self.layers = nn.ModuleList()
        # Shared layers
        for i in range(len(sizes) - 2):
            self.layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        # Actor head layers
        self.layers.append(nn.Linear(hidden_sizes[-1], hidden_sizes[-1]))
        self.layers.append(nn.Linear(hidden_sizes[-1], act_dim))
        # Critic head layers
        self.layers.append(nn.Linear(hidden_sizes[-1], hidden_sizes[-1]))
        self.layers.append(nn.Linear(hidden_sizes[-1], 1))

    def forward(self, obs):
        x = torch.cat([obs], dim=-1)
        for i in range(len(self.layers) - 4):
            x = F.relu(self.layers[i](x))
        # Actor head
        probs = F.relu(self.layers[-4](x))
        probs = F.softmax(self.layers[-3](probs), dim=-1)
        # Critic head
        value = F.relu(self.layers[-2](x))
        value = self.layers[-1](value)

        return torch.squeeze(probs, -1), torch.squeeze(value, -1)


class A2C(AbstractSolver):
    def __init__(self, env, eval_env, options):
        super().__init__(env, eval_env, options)
        # Create actor-critic network
        self.actor_critic = ActorCriticNetwork(
            env.observation_space.shape[0], env.action_space.n, self.options.layers
        )
        self.policy = self.create_greedy_policy()

        self.optimizer = AdamW(
            self.actor_critic.parameters(), lr=self.options.alpha, amsgrad=True
        )

    def create_greedy_policy(self):
        """
        Creates a greedy policy.


        Returns:
            A function that takes an observation as input and returns a vector
            of action probabilities.
        """

        def policy_fn(state):
            state = torch.as_tensor(state, dtype=torch.float32)
            return torch.argmax(self.actor_critic(state)[0]).detach().numpy()

        return policy_fn

    def train_episode(self):
        """
        Run a single episode of the A2C algorithm

        Use:
            self.actor_critic: actor-critic network that is being learned. Returns action
                probabilites and the critic value.
            np.random.choice(len(probs), probs): Randomly select an element
                from probs (a list) based on the probability distribution in probs.
            self.step(action): Performs an action in the env..
            self.options.gamma: Gamma discount factor.
        """

        state, _ = self.env.reset()
        action_probs = []  # Action probability
        deltas = []  # Advantage
        values = []  # Critic value
        target_values = []
        # Don't forget to convert the states to torch tensors to pass them through the network.
        for _ in range(self.options.steps):
            ################################
            #   YOUR IMPLEMENTATION HERE   #
            ################################

        target_values = torch.stack(target_values)
        # Normalize target values (learning speedup trick)
        target_values = (target_values - target_values.mean()) / target_values.std()
        values = torch.stack(values)
        deltas = target_values - values
        # Normalize deltas (learning speedup trick)
        deltas = (deltas - deltas.mean()) / deltas.std()
        action_probs = torch.stack(action_probs)

        # Compute loss
        actor_loss = self.actor_loss(deltas.detach(), action_probs).mean()
        critic_loss = self.critic_loss(deltas.detach(), values).mean()

        loss = actor_loss + critic_loss

        # Update actor critic
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def actor_loss(self, advantage, prob):
        """
        The policy gradient loss function.
        Note that you are required to define the Loss^PG
        which should be the integral of the policy gradient.

        args:
            advantage: advantage of the chosen action.
            prob: probability associated with the chosen action.

        Use:
            torch.log: Element-wise log.

        Returns:
            The unreduced loss (as a tensor).
        """
        ################################
        #   YOUR IMPLEMENTATION HERE   #
        ################################)

    def critic_loss(self, advantage, value):
        """
        The integral of the critic gradient

        args:
            advantage: advantage of the chosen action.
            value: Predicted state value.

        Returns:
            The unreduced loss (as a tensor).
        """
        ################################
        #   YOUR IMPLEMENTATION HERE   #
        ################################

    def __str__(self):
        return "A2C"

    def plot(self, stats, smoothing_window=20, final=False):
        plotting.plot_episode_stats(stats, smoothing_window, final=final)

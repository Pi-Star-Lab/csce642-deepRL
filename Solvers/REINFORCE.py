# Licensing Information:  You are free to use or extend this codebase for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) inform Guni Sharon at 
# guni@tamu.edu regarding your usage (relevant statistics is reported to NSF).
# The development of this assignment was supported by NSF (IIS-2238979).
# Contributors:
# The core code base was developed by Guni Sharon (guni@tamu.edu).
# The PyTorch code was developed by Sheelabhadra Dey (sheelabhadra@tamu.edu).

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.optim import Adam
from Solvers.Abstract_Solver import AbstractSolver
from lib import plotting


class PolicyNet(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes):
        super().__init__()
        sizes = [obs_dim] + hidden_sizes + [act_dim]
        self.layers = nn.ModuleList()
        # Shared layers
        for i in range(len(sizes) - 2):
            self.layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        # Policy head layers
        self.layers.append(nn.Linear(hidden_sizes[-1], act_dim))
        # Baseline head layers
        self.layers.append(nn.Linear(hidden_sizes[-1], 1))

    def forward(self, obs):
        x = torch.cat([obs], dim=-1)
        for i in range(len(self.layers) - 2):
            x = F.relu(self.layers[i](x))
        # Policy
        probs = F.softmax(self.layers[-2](x), dim=-1)
        # Baseline
        baseline = self.layers[-1](x)

        return torch.squeeze(probs, -1), torch.squeeze(baseline, -1)


class Reinforce(AbstractSolver):
    def __init__(self, env, eval_env, options):
        super().__init__(env, eval_env, options)
        # Create the policy network
        self.model = PolicyNet(
            env.observation_space.shape[0], env.action_space.n, self.options.layers
        )
        self.optimizer = Adam(self.model.parameters(), lr=self.options.alpha)

    def create_greedy_policy(self):
        """
        Creates a greedy policy.

        Returns:
            A function that takes an observation as input and returns the action with
            the highest probability.
        """

        def policy_fn(state):
            state = torch.as_tensor(state, dtype=torch.float32)
            return torch.argmax(self.model(state)[0]).detach().numpy()

        return policy_fn

    def compute_returns(self, rewards, gamma):
        """
        Compute the returns along an episode.

        Returns:
            The per step return along an episode (as a list).
        """
        ################################
        #   YOUR IMPLEMENTATION HERE   #
        ################################

    def select_action(self, state):
        """
        Selects an action given state.

        Returns:
            The selected action (as an int)
            The probability of the selected action (as a tensor)
            The baseline value (as a tensor)
        """
        state = torch.as_tensor(state, dtype=torch.float32)
        probs, baseline = self.model(state)

        probs_np = probs.detach().numpy()
        action = np.random.choice(len(probs_np), p=probs_np)

        return action, probs[action], baseline

    def update_model(self, rewards, action_probs, baselines):
        """
        Performs model update.
        """
        returns = torch.as_tensor(
            self.compute_returns(rewards, self.options.gamma), dtype=torch.float32
        )
        action_probs = torch.stack(action_probs)
        baselines = torch.stack(baselines)

        # Compute advantage (delta)
        deltas = returns - baselines

        # Compute loss
        pg_loss = self.pg_loss(deltas.detach(), action_probs).mean()
        value_loss = F.smooth_l1_loss(returns.detach(), baselines)

        loss = pg_loss + value_loss

        # Update model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train_episode(self):
        """
        Run a single episode of the Reinforce algorithm

        Use:
            self.select_action(state): Sample an action from the policy.
            self.step(action): Performs an action in the env.
            self.update_model(rewards, action_probs, baselines): Update the model.
        """

        state, _ = self.env.reset()
        rewards = []  # Reward per step
        action_probs = []  # Action probability
        baselines = []  # Value function
        for _ in range(self.options.steps):
            ################################
            #   YOUR IMPLEMENTATION HERE   #
            # Run update_model() only ONCE #
            # at the END of an episode.    #
            ################################


    def pg_loss(self, advantage, prob):
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
        ################################


    def __str__(self):
        return "REINFORCE"

    def plot(self, stats, smoothing_window, final=False):
        plotting.plot_episode_stats(stats, smoothing_window, final=final)

# Licensing Information:  You are free to use or extend this codebase for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide the following
# attribution:
# This CSCE-689 RL assignment codebase was developed at Texas A&M University.
# The core code base was developed by Guni Sharon (guni@tamu.edu).

import numpy as np
from Solvers.Abstract_Solver import AbstractSolver, Statistics


class RandomWalk(AbstractSolver):
    def __init__(self,env,options):
        super().__init__(env,options)

    def train_episode(self):
        for t in range(self.options.steps):
            action = self.env.action_space.sample()
            observation, reward, done, info = self.step(action)
            if done:
                break
        print("Episode {} finished after {} timesteps with total rewards {}".format(
            self.statistics[Statistics.Episode.value],self.statistics[Statistics.Steps.value],
            self.statistics[Statistics.Rewards.value]))

    def __str__(self):
        return "Random Walk"

    def create_greedy_policy(self):
        """
        Creates a random policy function.

        Returns:
            A function that takes an observation as input and returns a vector
            of action probabilities
        """
        nA = self.env.action_space.n
        A = np.ones(nA, dtype=float) / nA

        def policy_fn(observation):
            return A

        return policy_fn

import sys
from io import StringIO

import gymnasium as gym
import numpy as np

from gymnasium import spaces
from gymnasium.envs.toy_text.utils import categorical_sample


UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3


class WindyGridworldEnv(gym.Env):

    metadata = {"render.modes": ["human", "ansi"]}
    name = "WindyGridworld"

    def _limit_coordinates(self, coord):
        coord[0] = min(coord[0], self.shape[0] - 1)
        coord[0] = max(coord[0], 0)
        coord[1] = min(coord[1], self.shape[1] - 1)
        coord[1] = max(coord[1], 0)
        return coord

    def _calculate_transition_prob(self, current, delta, winds):
        new_position = (
            np.array(current)
            + np.array(delta)
            + np.array([-1, 0]) * winds[tuple(current)]
        )
        new_position = self._limit_coordinates(new_position).astype(int)
        new_state = np.ravel_multi_index(tuple(new_position), self.shape)
        is_done = tuple(new_position) == (3, 7)
        return [(1.0, new_state, -1.0, is_done)]

    def __init__(self):
        self.shape = (7, 10)

        nS = np.prod(self.shape)
        nA = 4

        # Wind strength
        winds = np.zeros(self.shape)
        winds[:, [3, 4, 5, 8]] = 1
        winds[:, [6, 7]] = 2

        # Calculate transition probabilities
        P = {}
        for s in range(nS):
            position = np.unravel_index(s, self.shape)
            P[s] = {a: [] for a in range(nA)}
            P[s][UP] = self._calculate_transition_prob(position, [-1, 0], winds)
            P[s][RIGHT] = self._calculate_transition_prob(position, [0, 1], winds)
            P[s][DOWN] = self._calculate_transition_prob(position, [1, 0], winds)
            P[s][LEFT] = self._calculate_transition_prob(position, [0, -1], winds)

        # We always start in state (3, 0)
        self.isd = np.zeros(nS)
        self.isd[np.ravel_multi_index((3, 0), self.shape)] = 1.0

        self.P = P

        self.observation_space = spaces.Discrete(nS)
        self.action_space = spaces.Discrete(nA)

        super(WindyGridworldEnv, self).__init__()

    def step(self, a):
        transitions = self.P[self.s][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, t = transitions[i]
        self.s = s
        self.lastaction = a

        return (int(s), r, t, False, {"prob": p})

    def reset(self, seed=123):
        self.s = categorical_sample(self.isd, self.np_random)
        return int(self.s), {"prob": 1}

    def render(self, mode="human", close=False):
        self._render(mode, close)

    def _render(self, mode="human", close=False):
        if close:
            return

        outfile = StringIO() if mode == "ansi" else sys.stdout

        for s in range(self.nS):
            position = np.unravel_index(s, self.shape)
            # print(self.s)
            if self.s == s:
                output = " x "
            elif position == (3, 7):
                output = " T "
            else:
                output = " o "

            if position[1] == 0:
                output = output.lstrip()
            if position[1] == self.shape[1] - 1:
                output = output.rstrip()
                output += "\n"

            outfile.write(output)
        outfile.write("\n")

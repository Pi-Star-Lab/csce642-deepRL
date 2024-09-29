# Licensing Information:  You are free to use or extend this codebase for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) inform Guni Sharon at 
# guni@tamu.edu regarding your usage (relevant statistics is reported to NSF).
# The development of this assignment was supported by NSF (IIS-2238979).
# Contributors:
# The autograder for the assignments was developed by Sumedh Pendurkar (sumedhpendurkar@tamu.edu)
# and Sheelabhadra Dey (sheelabhadra@tamu.edu).


import unittest
import shlex

from run import main, build_parser
from copy import deepcopy

import numpy as np
import pandas as pd
import joblib
import torch


def run_main(command_str):
    parser = build_parser()
    options, args = parser.parse_args(shlex.split(command_str))
    results = main(options)
    return results


def l2_distance_bounded(v1, v2, bound):
    distance = np.mean((v1 - v2) ** 2)
    return True if distance < bound else False


class vi(unittest.TestCase):
    points = 0

    @classmethod
    def setUpClass(self):
        command_str = "-s vi -d Gridworld -e 100 -g 0.9 --no-plots"
        self.results = run_main(command_str)

    def set_test_v(self):
        solver = self.results["solver"]
        v = np.array(
            [
                1.2086319679296675,
                1.3206763103045178,
                0.46923107590723945,
                0.8616088480733963,
                1.8455138935253883,
                2.079230999993757,
                1.9830750667821562,
                1.1425017272963585,
                1.3410148962839679,
                2.7341845187723646,
                0.10915686708240568,
                2.0887591555244995,
                0.12398618237655956,
                2.7830735037201877,
                1.8415810534076993,
                0.8549790393954977,
            ]
        )
        solver.V = v

    def test_train_episode(self):
        solver = self.results["solver"]
        self.set_test_v()
        solver.train_episode()
        updated_v = solver.V
        expected_v = np.array(
            [
                1.0877687711367008,
                0.8713078999943813,
                0.7847675601039406,
                0.02825155456672257,
                0.8713078999943813,
                1.4607660668951281,
                0.3146894602056154,
                0.8798832399720495,
                1.4607660668951281,
                1.5047661533481689,
                0.8798832399720495,
                0.8798832399720495,
                1.5047661533481689,
                1.5047661533481689,
                0.6574229480669294,
                0.769481135455948,
            ]
        )
        self.assertTrue(
            l2_distance_bounded(updated_v, expected_v, 1e-2),
            "`train_episode' function failed to provide correct output",
        )
        self.__class__.points += 4

    def test_create_greedy_policy(self):

        self.set_test_v()
        policy = self.results["solver"].create_greedy_policy()

        greedy_actions = []
        for i in range(0, 16):
            greedy_actions += [policy(i)]
        self.assertEqual(
            greedy_actions,
            [0, 2, 2, 2, 1, 2, 3, 2, 1, 2, 3, 1, 1, 2, 3, 0],
            "`create_greedy_policy' function failed to provide correct output",
        )
        self.__class__.points += 3

    def test_grid_world_1_reward(self):
        episode_rewards = self.results["stats"].episode_rewards[-1]
        expected_reward = -26.24
        self.assertEqual(
            expected_reward, episode_rewards, "got unexpected rewards for gridworld"
        )
        self.__class__.points += 1

    def test_grid_world_2_reward(self):
        command_str = "-s vi -d Gridworld -e 10 -g 0.4 --no-plots"
        results = run_main(command_str)
        episode_rewards = results["stats"].episode_rewards[-1]
        expected_reward = -18.64
        self.assertEqual(
            expected_reward, episode_rewards, "got unexpected rewards for gridworld"
        )
        self.__class__.points += 1

    def test_frozen_lake_reward(self):
        command_str = "-s vi -d FrozenLake-v1 -e 70 -g 0.9 --no-plots"
        results = run_main(command_str)
        episode_rewards = results["stats"].episode_rewards[-1]
        expected_reward = 2.176
        self.assertTrue(
            expected_reward < episode_rewards, "got unexpected rewards for frozen lake"
        )
        self.__class__.points += 1

    @classmethod
    def tearDownClass(cls):
        print("\nTotal Points: {} / 10".format(cls.points))


class avi(unittest.TestCase):
    points = 0

    @classmethod
    def setUpClass(self):
        command_str = "-s avi -d Gridworld -e 100 -g 0.5 --no-plots"
        self.results = run_main(command_str)

    def set_test_v(self):
        solver = self.results["solver"]
        v = np.array(
            [
                1.2086319679296675,
                1.3206763103045178,
                0.46923107590723945,
                0.8616088480733963,
                1.8455138935253883,
                2.079230999993757,
                1.9830750667821562,
                1.1425017272963585,
                1.3410148962839679,
                2.7341845187723646,
                0.10915686708240568,
                2.0887591555244995,
                0.12398618237655956,
                2.7830735037201877,
                1.8415810534076993,
                0.8549790393954977,
            ]
        )
        solver.V = v

    def test_train_episode_1(self):
        solver = self.results["solver"]
        self.set_test_v()
        solver.train_episode()
        updated_v = solver.V
        expected_v = np.array(
            [
                1.2086319679296675,
                1.3206763103045178,
                0.46923107590723945,
                0.8616088480733963,
                1.8455138935253883,
                2.079230999993757,
                1.9830750667821562,
                1.1425017272963585,
                1.3410148962839679,
                0.39153675186009385,
                0.10915686708240568,
                2.0887591555244995,
                0.12398618237655956,
                2.7830735037201877,
                1.8415810534076993,
                0.8549790393954977,
            ]
        )
        self.assertTrue(
            l2_distance_bounded(updated_v, expected_v, 1e-2),
            "`train_episode' function returned unexpected outputs",
        )
        self.__class__.points += 3

    def test_train_episode_2(self):
        solver = self.results["solver"]
        self.set_test_v()
        for i in range(10):
            solver.train_episode()
        updated_v = solver.V
        expected_v = np.array(
            [
                1.2086319679296675,
                -0.3396618448477411,
                0.46923107590723945,
                0.8616088480733963,
                -0.07724305323730585,
                -0.008462466608921915,
                -0.42874913635182077,
                0.04437957776224977,
                -0.32949255185801607,
                0.03961549999687852,
                0.10915686708240568,
                2.0887591555244995,
                0.12398618237655956,
                -0.8042316240699531,
                -0.07920947329615036,
                0.8549790393954977,
            ]
        )
        self.assertTrue(
            l2_distance_bounded(updated_v, expected_v, 1e-2),
            "`train_episode' function return unexpected outputs",
        )
        self.__class__.points += 4

    def test_grid_world_reward(self):
        episode_rewards = self.results["stats"].episode_rewards[-1]
        expected_reward = -20
        self.assertEqual(
            expected_reward, episode_rewards, "got unexpected rewards for grid world"
        )
        self.__class__.points += 1

    def test_frozen_lake_1_reward(self):
        command_str = "-s avi -d FrozenLake-v1 -e 60 -g 0.5 --no-plots"
        results = run_main(command_str)
        episode_rewards = results["stats"].episode_rewards[-1]
        expected_reward = 0.637
        self.assertTrue(
            expected_reward < episode_rewards, "got unexpected rewards for frozen lake"
        )
        self.__class__.points += 1

    def test_frozen_lake_2_reward(self):
        command_str = "-s avi -d FrozenLake-v1 -e 100 -g 0.7 --no-plots"
        results = run_main(command_str)
        episode_rewards = results["stats"].episode_rewards[-1]
        expected_reward = 0.978
        self.assertTrue(
            expected_reward < episode_rewards, "got unexpected rewards for frozen lake"
        )
        self.__class__.points += 1

    @classmethod
    def tearDownClass(cls):
        print("\nTotal Points: {} / 10".format(cls.points))


class pi(unittest.TestCase):
    points = 0
    fail = True
    @classmethod
    def setUpClass(self):
        command_str = "-s pi -d Gridworld -e 100 -g 0.9 --no-plots"
        self.results = run_main(command_str)

    def test_policy_eval(self):
        solver = self.results["solver"]
        policy = np.eye(16, k=1)
        policy[-1][0] = 1
        solver.policy_eval()
        v = solver.V
        expected_v = np.array(
            [
                0,
                -1.0,
                -1.9,
                -2.71,
                -1,
                -1.9,
                -2.71,
                -1.9,
                -1.9,
                -2.71,
                -1.9,
                -1.0,
                -2.71,
                -1.9,
                -1.0,
                0.0,
            ]
        )
        self.assertTrue(
            l2_distance_bounded(v, expected_v, 1e-3),
            "`policy_eval' function resulted in unexpected V",
        )
        self.__class__.points += 3

    def test_train_episode(self):
        def dummy_policy_eval():
            pass

        solver = self.results["solver"]
        solver.V = np.array(
            [
                0.1,
                -1.0,
                -1.9,
                -2.71,
                -1.1,
                -1.91,
                -2.72,
                -1.92,
                -1.93,
                -2.73,
                -1.95,
                -1.2,
                -2.74,
                -1.96,
                -1.3,
                0.0,
            ]
        )
        solver.policy_eval = dummy_policy_eval
        policy = np.argmax(solver.policy, axis=1).tolist()
        expected_policy = [0, 3, 3, 2, 0, 0, 0, 2, 0, 0, 1, 2, 0, 1, 1, 0]
        self.assertEqual(
            policy,
            expected_policy,
            "`train_episode' function return unexpected outputs",
        )
        self.__class__.points += 4


    def test_iterative(self):
        og_solver = np.linalg.solve
        calls = 0
        def custom_solver(A, b):
            nonlocal calls
            calls += 1
            og_A = np.load("TestData/pi_A.npy")
            og_b = np.load("TestData/pi_b.npy")
            if not (l2_distance_bounded(np.abs(A), np.abs(og_A), 1e-3) and l2_distance_bounded(np.abs(b), np.abs(og_b), 1e-3)):
                calls += 100
            return og_solver(A,b)

        solver = self.results["solver"]
        np.linalg.solve = custom_solver
        solver.policy_eval()
        np.linalg.solve = og_solver
        self.assertTrue(1 == calls, "Make sure you the linear equation coefficients provided to np.linalg.solve is correct")
        self.__class__.fail = False

    def test_grid_world_1_reward(self):
        episode_rewards = self.results["stats"].episode_rewards[-1]
        expected_reward = -26.24
        self.assertEqual(
            expected_reward, episode_rewards, "got unexpected rewards for grid world"
        )
        self.__class__.points += 1

    def test_grid_world_2_reward(self):
        command_str = "-s pi -d Gridworld -e 10 -g 0.4 --no-plots"
        results = run_main(command_str)
        episode_rewards = results["stats"].episode_rewards[-1]
        expected_reward = -18.64
        self.assertEqual(
            expected_reward, episode_rewards, "got unexpected rewards for grid world"
        )
        self.__class__.points += 1

    def test_frozen_lake_reward(self):
        command_str = "-s pi -d FrozenLake-v1 -e 5 -g 0.5 --no-plots"
        results = run_main(command_str)
        episode_rewards = results["stats"].episode_rewards[-1]
        expected_reward = 0.634
        self.assertTrue(
            expected_reward < episode_rewards, "got unexpected rewards for frozen lake"
        )
        self.__class__.points += 1

    @classmethod
    def tearDownClass(cls):
        points = 0 if cls.fail else cls.points
        print("\nTotal Points: {} / 10".format(points))


class mc(unittest.TestCase):
    points = 0

    @classmethod
    def setUpClass(self):
        command_str = "-s mc -d Blackjack -e 0 -g 0.9 -p 0.1 --no-plots"
        # command_str = "-s mc -d Blackjack -e 500000 -g 1.0 -p 0.1 --no-plots"
        self.results = run_main(command_str)

    def test_make_epsilon_greedy_policy(self):
        solver = self.results["solver"]
        policy = solver.make_epsilon_greedy_policy()
        solver.Q[0][0] = 0.3
        solver.Q[0][1] = 0.1
        self.assertTrue(l2_distance_bounded(np.array([0.95, 0.05]), policy(0), 1e-8))
        self.__class__.points += 2

    def test_create_greedy_policy(self):
        solver = self.results["solver"]
        policy = solver.create_greedy_policy()
        solver.Q[1][0] = 0.3
        solver.Q[1][1] = 0.1
        predict_action = policy(1)
        self.assertTrue(
            predict_action == 0, "`create_greedy_policy' returns unexpected policy"
        )
        solver.Q[1][0] = 0.1
        solver.Q[1][1] = 0.3
        predict_action = policy(1)
        self.assertTrue(
            predict_action == 1, "`create_greedy_policy' returns unexpected policy"
        )
        self.__class__.points += 1

    def test_train_episode(self):
        def dummy_policy(state):

            if state == (14, 10, False):
                return np.array([1, 0])
            else:
                return np.array([0, 1])

        def dummy_reset():
            return (14, 10, False), {}

        def dummy_step(action):

            if action == 0:
                return (14, 9, False), -1, False, ""
            else:
                return (23, 2, False), -1, True, ""

        solver = self.results["solver"]
        solver.policy = dummy_policy
        solver.env.reset = dummy_reset
        solver.step = dummy_step
        solver.train_episode()
        self.assertEqual(
            list(solver.Q[(14, 10, False)]),
            [-1.9, 0],
            "`train_episode' function return unexpected outputs",
        )
        self.assertEqual(
            list(solver.Q[(14, 9, False)]),
            [0, -1],
            "`train_episode' function return unexpected outputs",
        )
        self.assertEqual(
            list(solver.Q[(23, 2, False)]),
            [0, 0],
            "`train_episode' function return unexpected outputs",
        )
        self.__class__.points += 5

    def test_blackjack_1_reward(self):
        command_str = "-s mc -d Blackjack -e 500000 -g 1.0 -p 0.1 --no-plots"
        results = run_main(command_str)
        Q_ar = np.zeros((21, 21, 2, 2))
        solver = results["solver"]
        for key, val in solver.Q.items():
            x, y, z = key
            z = 0 if z is False else 1
            Q_ar[x - 1][y][z][0] = val[0]
            Q_ar[x - 1][y][z][1] = val[1]
        expected_Q_ar = np.load("TestData/mc_rewards_mean_ar.npy")
        self.assertTrue(
            l2_distance_bounded(expected_Q_ar, Q_ar, 0.03),
            "got unexpected rewards for blackjack",
        )
        self.__class__.points += 2

    @classmethod
    def tearDownClass(cls):
        print("\nTotal Points: {} / 10".format(cls.points))


class mcis(unittest.TestCase):
    points = 0

    @classmethod
    def setUpClass(self):
        command_str = "-s mcis -d WindyGridworld -e 0 -g 0.3 --no-plots"
        self.results = run_main(command_str)
        solver = self.results["solver"]

    def test_train_episode(self):
        solver = self.results["solver"]

        def dummy_behavior_policy(state):
            action = np.ones((4)) * 0.01 / 3
            action[state] = 0.99
            return action

        def dummy_target_policy(state):
            action = state
            return action

        def dummy_reset():
            return 0, {}

        def dummy_step(action):
            if action == 3:
                return action, 0, True, ""
            else:
                return action + 1, -1, False, ""

        solver.target_policy = dummy_target_policy
        solver.behavior_policy = dummy_behavior_policy
        solver.env.reset = dummy_reset
        solver.step = dummy_step
        solver.train_episode()
        self.assertTrue(
            l2_distance_bounded(solver.Q[0], np.array([-1.39, 0, 0, 0]), 1e-12),
            "`train_episode' function return unexpected outputs",
        )
        self.__class__.points += 2
        self.assertTrue(
            l2_distance_bounded(solver.Q[1], np.array([0, -1.3, 0, 0]), 1e-12),
            "`train_episode' function return unexpected outputs",
        )
        self.__class__.points += 2
        self.assertTrue(
            l2_distance_bounded(solver.Q[2], np.array([0, 0, -1, 0]), 1e-12),
            "`train_episode' function return unexpected outputs",
        )
        self.__class__.points += 1
        self.assertTrue(
            l2_distance_bounded(solver.Q[3], np.array([0, 0, 0, 0]), 1e-12),
            "`train_episode' function return unexpected outputs",
        )
        self.__class__.points += 1

    def test_blackjack_1_reward(self):
        command_str = "-s mcis -d Blackjack -e 500000 -g 0.6 --no-plots"
        results = run_main(command_str)
        Q_ar = np.zeros((21, 21, 2, 2))
        solver = results["solver"]
        for key, val in solver.Q.items():
            x, y, z = key
            z = 0 if z is False else 1
            Q_ar[x - 1][y][z][0] = val[0]
            Q_ar[x - 1][y][z][1] = val[1]
        expected_Q_ar = np.load("TestData/mcis_rewards_mean_ar.npy")
        self.assertTrue(
            l2_distance_bounded(expected_Q_ar, Q_ar, 0.03),
            "got unexpected rewards for blackjack",
        )
        self.__class__.points += 4

    @classmethod
    def tearDownClass(cls):
        print("\nTotal Points: {} / 10".format(cls.points))


class ql(unittest.TestCase):
    points = 0

    @classmethod
    def setUpClass(self):
        command_str = "-s ql -d Blackjack -e 0 -a 0.5 -g 0.3 -p 0.1 --no-plots"
        self.results = run_main(command_str)

    def test_make_epsilon_greedy_policy(self):
        solver = self.results["solver"]
        policy = solver.epsilon_greedy_action
        solver.Q[0][0] = 0.3
        solver.Q[0][1] = 0.1
        np.random.seed(10)
        import random

        random.seed(10)
        # test = [(sum([policy(0) for x in range(1000)])) for y in range(100)]
        self.assertTrue(
            l2_distance_bounded(np.array([0.95, 0.05]), policy(0), 1e-8),
            "`make_epsilon_greedy_policy' returns unexpected policy",
        )
        self.__class__.points += 2

    def test_create_greedy_policy(self):
        solver = self.results["solver"]
        policy = solver.create_greedy_policy()
        old_Q = deepcopy(solver.Q)
        solver.Q[1][0] = 0.3
        solver.Q[1][1] = 0.1
        predict_action = policy(1)
        self.assertTrue(
            predict_action == 0, "`create_greedy_policy' returns unexpected policy"
        )
        solver.Q[1][0] = 0.1
        solver.Q[1][1] = 0.3
        predict_action = policy(1)
        self.assertTrue(
            predict_action == 1, "`create_greedy_policy' returns unexpected policy"
        )
        self.__class__.points += 1
        solver.Q = old_Q

    def test_train_episode(self):
        def dummy_policy(state):

            if state == (14, 10, False):
                return 0
            else:
                return 1

        def dummy_reset():
            return (14, 10, False), {"prob": 1}

        def dummy_step(action):

            if action == 0:
                return (14, 9, False), -1, False, ""
            else:
                return (23, 2, False), -1, True, ""

        solver = self.results["solver"]
        solver.make_epsilong_greedy_policy = dummy_policy
        solver.env.reset = dummy_reset
        solver.step = dummy_step
        solver.train_episode()
        self.assertEqual(
            list(solver.Q[(14, 10, False)]),
            [-0.5, 0],
            "`train_episode' function return unexpected outputs",
        )
        self.assertEqual(
            list(solver.Q[(14, 9, False)]),
            [-0.5, -0.5],
            "`train_episode' function return unexpected outputs",
        )
        self.assertEqual(
            list(solver.Q[(23, 2, False)]),
            [0, 0],
            "`train_episode' function return unexpected outputs",
        )
        self.__class__.points += 5

    def test_cliff_walking_reward(self):
        command_str = "-s ql -d CliffWalking -e 500 -a 0.5 -g 1.0 -p 0.1 --no-plots"
        results = run_main(command_str)
        stats = results["stats"]
        smoothing_window = 10
        ep_len = stats.episode_lengths
        self.assertTrue(
            np.mean(ep_len[:30]) > 25 and np.mean(ep_len[150:]) < 15,
            "got unexpected rewards for cliff walking",
        )
        self.__class__.points += 1
        rewards_smoothed = (
            pd.Series(stats.episode_rewards)
            .rolling(smoothing_window, min_periods=smoothing_window)
            .mean()
        )

        self.assertTrue(
            np.max(rewards_smoothed) > -15 and np.mean(ep_len[150:]) > -40,
            "got unexpected rewards for cliff walking",
        )
        self.__class__.points += 1

    @classmethod
    def tearDownClass(cls):
        print("\nTotal Points: {} / 10".format(cls.points))


class sarsa(unittest.TestCase):
    points = 0

    @classmethod
    def setUpClass(self):
        command_str = "-s sarsa -d WindyGridworld -e 0 -a 0.5 -g 0.3 -p 0.1 --no-plots"
        self.results = run_main(command_str)

    def test_make_epsilon_greedy_policy(self):
        solver = self.results["solver"]
        policy = solver.epsilon_greedy_action
        old_Q = deepcopy(solver.Q)
        solver.Q[0][0] = 0.3
        solver.Q[0][1] = 0.1
        solver.Q[0][2] = 0.2
        solver.Q[0][3] = 0.0
        self.assertTrue(
            l2_distance_bounded(
                np.array([0.925, 0.025, 0.025, 0.025]), policy(0), 1e-8
            ),
            "`make_epsilon_greedy_policy' returns unexpected policy",
        )
        solver.Q = old_Q
        self.__class__.points += 2

    def test_create_greedy_policy(self):
        solver = self.results["solver"]
        policy = solver.create_greedy_policy()
        old_Q = deepcopy(solver.Q)
        solver.Q[1][0] = 0.3
        solver.Q[1][1] = 0.1
        solver.Q[1][2] = 0.1
        solver.Q[1][3] = 0.1
        predict_action = policy(1)
        self.assertTrue(
            predict_action == 0, "`create_greedy_policy' returns unexpected policy"
        )
        solver.Q[1][0] = 0.1
        solver.Q[1][1] = 0.3
        solver.Q[1][2] = 0.1
        solver.Q[1][3] = 0.1
        predict_action = policy(1)
        self.assertTrue(
            predict_action == 1, "`create_greedy_policy' returns unexpected policy"
        )
        solver.Q = old_Q
        self.__class__.points += 1

    def test_train_episode(self):
        def dummy_policy(state):
            action = np.zeros((4))
            action[state] = 1
            return action

        def dummy_reset():
            return 0, {"prob": 1}

        def dummy_step(action):
            if action == 3:
                return action, 0, True, ""
            else:
                return action + 1, -1, False, ""

        solver = self.results["solver"]
        solver.epsilon_greedy_action = dummy_policy
        solver.env.reset = dummy_reset
        solver.step = dummy_step
        solver.train_episode()
        solver.train_episode()
        self.assertEqual(
            list(solver.Q[0]),
            [-0.825, 0, 0, 0],
            "`train_episode' function return unexpected outputs",
        )
        self.assertEqual(
            list(solver.Q[1]),
            [0, -0.825, 0, 0],
            "`train_episode' function return unexpected outputs",
        )
        self.assertEqual(
            list(solver.Q[2]),
            [0, 0, -0.75, 0],
            "`train_episode' function return unexpected outputs",
        )
        self.assertEqual(
            list(solver.Q[3]),
            [0, 0, 0, 0],
            "`train_episode' function return unexpected outputs",
        )
        self.__class__.points += 5

    def test_cliff_walking_reward(self):
        command_str = "-s sarsa -d CliffWalking -e 500 -a 0.5 -g 1.0 -p 0.1 --no-plots"
        results = run_main(command_str)
        stats = results["stats"]
        smoothing_window = 10
        ep_len = stats.episode_lengths
        rewards_smoothed = (
            pd.Series(stats.episode_rewards)
            .rolling(smoothing_window, min_periods=smoothing_window)
            .mean()
        )
        #print(np.mean(rewards_smoothed[:10]),  np.mean(ep_len[450:]))
        self.assertTrue(
            np.mean(rewards_smoothed[:10]) < -99 and np.mean(ep_len[450:]) < 61,
            "got unexpected rewards for cliff walking",
        )
        self.__class__.points += 1
        rewards_smoothed = (
            pd.Series(stats.episode_rewards)
            .rolling(smoothing_window, min_periods=smoothing_window)
            .mean()
        )
        #print(np.max(stats.episode_rewards), np.mean(rewards_smoothed[499:]))
        self.assertTrue(
            np.max(stats.episode_rewards) > -18
            and np.mean(rewards_smoothed[499:]) > -70,
            "got unexpected rewards for cliff walking",
        )
        self.__class__.points += 1

    @classmethod
    def tearDownClass(cls):
        print("\nTotal Points: {} / 10".format(cls.points))


class aql(unittest.TestCase):
    points = 0

    @classmethod
    def setUpClass(self):
        command_str = "-s aql -d MountainCar-v0 -e 0 -g 1.0 -p 0.3 -r 100 --no-plots"
        self.results = run_main(command_str)

    def test_greedy_policy(self):
        solver = self.results["solver"]
        orig_model = deepcopy(solver.estimator)
        dummy_model_path = "TestData/test_model_aql.pkl"
        dummy_model = joblib.load(dummy_model_path)
        solver.estimator = dummy_model
        # Test 1
        dummy_state = np.array([-1.0, 0.0])
        self.assertEqual(
            solver.create_greedy_policy()(dummy_state),
            2,
            "`epsilon_greedy' returns unexpected policy",
        )
        # Test 2
        dummy_state = np.array([-0.39, -0.017])
        self.assertEqual(
            solver.create_greedy_policy()(dummy_state),
            0,
            "`epsilon_greedy' returns unexpected policy",
        )
        self.__class__.points += 2
        solver.model = orig_model

    def test_epsilon_greedy_policy(self):
        solver = self.results["solver"]
        orig_model = deepcopy(solver.estimator)
        dummy_model_path = "TestData/test_model_aql.pkl"
        dummy_model = joblib.load(dummy_model_path)
        solver.estimator = dummy_model
        # Test 1
        dummy_state = np.array([-1.0, 0.0])
        self.assertTrue(
            l2_distance_bounded(
                solver.epsilon_greedy(dummy_state), np.array([0.1, 0.1, 0.8]), 1e-4
            ),
            "`epsilon_greedy' returns unexpected policy",
        )
        # Test 2
        dummy_state = np.array([-0.39, -0.017])
        self.assertTrue(
            l2_distance_bounded(
                solver.epsilon_greedy(dummy_state), np.array([0.8, 0.1, 0.1]), 1e-4
            ),
            "`epsilon_greedy' returns unexpected policy",
        )
        self.__class__.points += 3
        solver.model = orig_model

    def test_mountain_car_reward(self):
        command_str = "-s aql -d MountainCar-v0 -e 100 -g 1.0 -p 0.2 -r 100 --no-plots"
        results = run_main(command_str)
        stats = results["stats"]
        smoothing_window = 10
        ep_len = stats.episode_lengths
        rewards_smoothed = (
            pd.Series(stats.episode_rewards)
            .rolling(smoothing_window, min_periods=smoothing_window)
            .mean()
        )
        self.assertTrue(
            np.mean(ep_len[:5]) > 500 and np.mean(ep_len[80:]) < 250,
            "got unexpected rewards for mountain car",
        )
        self.__class__.points += 1
        self.assertTrue(
            np.mean(ep_len[:5]) > 500 and np.mean(ep_len[80:]) < 180,
            "got unexpected rewards for mountain car",
        )
        self.__class__.points += 1
        rewards_smoothed = (
            pd.Series(stats.episode_rewards)
            .rolling(smoothing_window, min_periods=smoothing_window)
            .mean()
        )

        self.assertTrue(
            np.max(rewards_smoothed) > -200 and np.mean(rewards_smoothed[:20]) < -350,
            "got unexpected rewards for mountain car",
        )
        self.__class__.points += 1
        self.assertTrue(
            np.max(rewards_smoothed) > -150 and np.mean(rewards_smoothed[:20]) < -350,
            "got unexpected rewards for mountain car",
        )
        self.__class__.points += 2

    @classmethod
    def tearDownClass(cls):
        print("\nTotal Points: {} / 10".format(cls.points))


class dqn(unittest.TestCase):
    points = 0

    @classmethod
    def setUpClass(self):
        command_str = "-s dqn -t 1000 -d LunarLander-v2 -e 0 -a 0.01 -g 0.99 -p 0.1 -P 0.1 -c 1.0 -m 2000 -r 100 -N 100 -b 64 -l [128,128] --no-plots"
        self.results = run_main(command_str)

    def test_epsilon_greedy(self):
        solver = self.results["solver"]
        orig_model = deepcopy(solver.model)
        dummy_model_path = "TestData/test_model_dqn.pth"
        dummy_model = torch.load(dummy_model_path)
        solver.model = dummy_model
        # Test 1
        dummy_state = torch.tensor(
            [-1.6034, -1.1659, 0.5953, 1.2609, 0.7732, -0.3623, 1.7237, 0.9790],
            dtype=torch.float32,
        )
        self.assertEqual(
            list(solver.epsilon_greedy(dummy_state)),
            [0.025, 0.925, 0.025, 0.025],
            "`epsilon_greedy' returns unexpected policy",
        )
        # Test 2
        dummy_state = torch.tensor(
            [-2.1429, 0.8679, 0.2397, -0.5396, 0.3134, 0.5916, -1.1758, 0.0649],
            dtype=torch.float32,
        )
        self.assertEqual(
            list(solver.epsilon_greedy(dummy_state)),
            [0.025, 0.025, 0.025, 0.925],
            "`epsilon_greedy' returns unexpected policy",
        )
        self.__class__.points += 2
        solver.model = orig_model

    def test_compute_target_q_values(self):
        solver = self.results["solver"]
        orig_model = deepcopy(solver.model)
        dummy_model_path = "TestData/test_model_dqn.pth"
        dummy_model = torch.load(dummy_model_path)
        solver.model = dummy_model
        solver.target_model = dummy_model
        # Test 1
        dummy_reward = torch.tensor([10], dtype=torch.float32)
        dummy_next_state = torch.tensor(
            [-2.1429, 0.8679, 0.2397, -0.5396, 0.3134, 0.5916, -1.1758, 0.0649],
            dtype=torch.float32,
        )
        dummy_done = torch.tensor([0], dtype=torch.float32)
        dummy_next_state = dummy_next_state.reshape(1,-1)
        self.assertTrue(
            l2_distance_bounded(
                solver.compute_target_values(dummy_next_state, dummy_reward, dummy_done)
                .detach()
                .item(),
                -32.05014,
                1e-2,
            ),
            "`compute_target_values' returns unexpected values",
        )
        # Test 2
        dummy_reward = torch.tensor([10], dtype=torch.float32)
        
        dummy_next_state = torch.tensor(
            [-2.1429, 0.8679, 0.2397, -0.5396, 0.3134, 0.5916, -1.1758, 0.0649],
            dtype=torch.float32,
        )
        dummy_next_state = dummy_next_state.reshape(1,-1)
        dummy_done = torch.tensor([1], dtype=torch.float32)
        self.assertEqual(
            solver.compute_target_values(dummy_next_state, dummy_reward, dummy_done).reshape(-1)
            .detach()
            .item(),
            10,
            "`compute_target_values' returns unexpected values",
        )
        self.__class__.points += 3
        solver.model = orig_model

    def test_cartpole_reward(self):
        """ """
        command_str = "-s dqn -t 2000 -d CartPole-v1 -e 100 -a 0.01 -g 0.95 -p 1.0 -P 0.01 -c 0.95 -m 2000 -r 100 -N 100 -b 32 -l [32,32] --no-plots"
        import torch

        torch.manual_seed(200)
        import random

        random.seed(200)
        import numpy as np

        np.random.seed(200)
        import os

        os.environ["PYTHONHASHSEED"] = str(200)
        results = run_main(command_str)
        stats = results["stats"]
        smoothing_window = 10
        ep_len = stats.episode_lengths
        rewards_smoothed = (
            pd.Series(stats.episode_rewards)
            .rolling(smoothing_window, min_periods=smoothing_window)
            .mean()
        )

        self.assertTrue(
            np.mean(ep_len[:5]) < 200 and np.max(ep_len[80:]) > 800,
            "got unexpected rewards for cartpole",
        )
        self.__class__.points += 2
        rewards_smoothed = (
            pd.Series(stats.episode_rewards)
            .rolling(smoothing_window, min_periods=smoothing_window)
            .mean()
        )

        self.assertTrue(
            np.max(rewards_smoothed) > 750 and np.mean(rewards_smoothed[:20]) < 100,
            "got unexpected rewards for cartpole",
        )
        self.__class__.points += 2
        self.assertTrue(
            np.max(rewards_smoothed) > 900 and np.mean(rewards_smoothed[:20]) < 100,
            "got unexpected rewards for cartpole",
        )
        self.__class__.points += 1

    @classmethod
    def tearDownClass(cls):
        print("\nTotal Points: {} / 10".format(cls.points))


class reinforce(unittest.TestCase):
    points = 0

    @classmethod
    def setUpClass(self):
        command_str = "-s reinforce -t 1000 -d CartPole-v1 -e 0 -a 0.001 -g 0.95 -l [32] --no-plots"
        self.results = run_main(command_str)

    def test_compute_returns(self):
        solver = self.results["solver"]
        # Test 1
        test_rewards = [5]
        self.assertTrue(
            l2_distance_bounded(
                np.array(solver.compute_returns(test_rewards, 1.0)), np.array([5]), 1e-2
            ),
            "`compute_returns' returns unexpected values",
        )
        # Test 2
        test_rewards = [1, 2, 3, 4, 5]
        self.assertTrue(
            l2_distance_bounded(
                np.array(solver.compute_returns(test_rewards, 0.8)),
                np.array([8.616, 9.52, 9.4, 8.0, 5]),
                1e-2,
            ),
            "`compute_returns' returns unexpected values",
        )
        self.__class__.points += 2

    def test_pg_loss(self):
        solver = self.results["solver"]
        # Test 1
        test_advantage = torch.tensor([0.3435, 0.3512, 0.2456])
        test_prob = torch.tensor([0.4799, 0.1807, 0.3393])
        self.assertTrue(
            l2_distance_bounded(
                solver.pg_loss(test_advantage, test_prob).detach().numpy(),
                np.array([0.2522, 0.6008, 0.2654]),
                1e-2,
            ),
            "`pg_loss' returns unexpected values.",
        )
        # Test 2
        test_advantage = torch.tensor([-0.6934, 1.0605, 0.0662, -0.7866, -1.4868])
        test_prob = torch.tensor([0.1898, 0.1443, 0.3548, 0.2178, 0.0933])
        self.assertTrue(
            l2_distance_bounded(
                solver.pg_loss(test_advantage, test_prob).detach().numpy(),
                np.array([-1.1524, 2.0528, 0.0686, -1.1990, -3.5259]),
                1e-2,
            ),
            "`pg_loss' returns unexpected values.",
        )
        # Test 3
        test_advantage = torch.tensor([10, 20])
        test_prob = torch.tensor([0.999, 0.001])
        self.assertTrue(
            l2_distance_bounded(
                solver.pg_loss(test_advantage, test_prob).detach().numpy(),
                np.array([0.010005, 138.16]),
                1e-2,
            ),
            "`pg_loss' returns unexpected values.",
        )
        self.__class__.points += 3

    def test_cartpole_reward(self):
        command_str = "-s reinforce -t 1000 -d CartPole-v1 -e 2000 -a 0.001 -g 0.95 -l [64,64] --no-plots"
        import torch

        torch.manual_seed(200)
        import random

        random.seed(200)
        import numpy as np

        np.random.seed(200)
        import os

        os.environ["PYTHONHASHSEED"] = str(200)
        results = run_main(command_str)
        stats = results["stats"]
        smoothing_window = 10
        ep_len = stats.episode_lengths
        rewards_smoothed = (
            pd.Series(stats.episode_rewards)
            .rolling(smoothing_window, min_periods=smoothing_window)
            .mean()
        )
        self.assertTrue(
            np.mean(ep_len[:5]) < 200 and np.max(ep_len[90:]) > 600,
            "got unexpected rewards for cartpole",
        )
        self.__class__.points += 1
        self.assertTrue(
            np.mean(ep_len[:5]) < 200 and np.max(ep_len[80:]) > 900,
            "got unexpected rewards for cartpole",
        )
        self.__class__.points += 1
        rewards_smoothed = (
            pd.Series(stats.episode_rewards)
            .rolling(smoothing_window, min_periods=smoothing_window)
            .mean()
        )

        self.assertTrue(
            np.max(rewards_smoothed) > 900 and np.mean(rewards_smoothed[:20]) < 100,
            "got unexpected rewards for cartpole",
        )
        self.__class__.points += 1
        self.assertTrue(
            np.min(rewards_smoothed[1900:]) > 200
            and np.mean(rewards_smoothed[:20]) < 100,
            "got unexpected rewards for cartpole",
        )
        self.__class__.points += 2

    @classmethod
    def tearDownClass(cls):
        print("\nTotal Points: {} / 10".format(cls.points))


class a2c(unittest.TestCase):
    points = 0

    @classmethod
    def setUpClass(self):
        command_str = (
            "-s a2c -t 1000 -d CartPole-v1 -e 0 -a 0.001 -g 0.95 -l [32] --no-plots"
        )
        self.results = run_main(command_str)

    def test_actor_loss(self):
        solver = self.results["solver"]
        # Test 1
        test_advantage = torch.tensor([0.0536, 0.8327, 0.3142, 0.8665, 0.2751])
        test_prob = torch.tensor([0.0519, 0.2371, 0.2226, 0.5396, 0.6266])
        self.assertTrue(
            l2_distance_bounded(
                solver.actor_loss(test_advantage, test_prob).detach().numpy(),
                np.array([0.1586, 1.1985, 0.4720, 0.5346, 0.1286]),
                1e-2,
            ),
            "`actor_loss' returns unexpected values.",
        )
        # Test 2
        test_advantage = torch.tensor([2.3142, -1.2101, 1.1669, -0.7585])
        test_prob = torch.tensor([0.1593, 0.4562, 0.2559, 0.1285])
        self.assertTrue(
            l2_distance_bounded(
                solver.actor_loss(test_advantage, test_prob).detach().numpy(),
                np.array([4.2505, -0.9498, 1.5903, -1.5561]),
                1e-2,
            ),
            "actor_loss' returns unexpected values.",
        )
        self.__class__.points += 3

    def test_critic_loss(self):
        solver = self.results["solver"]
        # Test 1
        test_advantage = torch.tensor([0.3742, 0.4920, -2.5846, 0.5786, 0.4693])
        test_value = torch.tensor([0.0049, 0.0342, 0.5687, 0.4458, 0.7628])
        self.assertTrue(
            l2_distance_bounded(
                solver.critic_loss(test_advantage, test_value).detach().numpy(),
                np.array([-0.0018, -0.0168, 1.4699, -0.2579, -0.3580]),
                1e-2,
            ),
            "`critic_loss' returns unexpected values.",
        )
        # Test 2
        test_advantage = torch.tensor([0.6318, 0.5236, 0.0985, 0.9992])
        test_value = torch.tensor([0.0521, -1.7944, 0.3403, 0.7393])
        self.assertTrue(
            l2_distance_bounded(
                solver.critic_loss(test_advantage, test_value).detach().numpy(),
                np.array([-0.0329, 0.9396, -0.0335, -0.7388]),
                1e-2,
            ),
            "`critic_loss' returns unexpected values.",
        )
        self.__class__.points += 2

    def test_cartpole_reward(self):
        command_str = " -s a2c -t 1000 -d CartPole-v1 -e 3000 -a 0.0001 -g 0.95 -l [64,64] --no-plots"

        import torch

        torch.manual_seed(200)
        import random

        random.seed(200)
        import numpy as np

        np.random.seed(200)
        import os

        os.environ["PYTHONHASHSEED"] = str(200)
        results = run_main(command_str)
        stats = results["stats"]
        smoothing_window = 10
        ep_len = stats.episode_lengths
        rewards_smoothed = (
            pd.Series(stats.episode_rewards)
            .rolling(smoothing_window, min_periods=smoothing_window)
            .mean()
        )
        self.assertTrue(
            np.mean(ep_len[:5]) < 200 and np.max(ep_len[90:]) > 600,
            "got unexpected rewards for cartpole",
        )
        self.__class__.points += 1
        self.assertTrue(
            np.mean(ep_len[:5]) < 200 and np.max(ep_len[80:]) > 900,
            "got unexpected rewards for cartpole",
        )
        self.__class__.points += 1
        rewards_smoothed = (
            pd.Series(stats.episode_rewards)
            .rolling(smoothing_window, min_periods=smoothing_window)
            .mean()
        )

        self.assertTrue(
            np.max(rewards_smoothed) > 900 and np.mean(rewards_smoothed[:20]) < 100,
            "got unexpected rewards for cartpole",
        )
        self.__class__.points += 1
        self.assertTrue(
            np.min(rewards_smoothed[2900:]) > 200
            and np.mean(rewards_smoothed[:20]) < 100,
            "got unexpected rewards for cartpole",
        )
        self.__class__.points += 2

    @classmethod
    def tearDownClass(cls):
        print("\nTotal Points: {} / 10".format(cls.points))


class ddpg(unittest.TestCase):
    points = 0

    @classmethod
    def setUpClass(self):
        command_str = (
            "-s ddpg -t 1000 -d HalfCheetah-v4 -e 0 -a 0.001 -g 0.99 -l [256,256] -m 1000000 -b 100 --no-plots"
        )
    
    def test_compute_target_values(self):
       
        command_str = (
            "-s ddpg -t 1000 -d LunarLanderContinuous-v2 -e 0 -a 0.001 -g 0.99 -l [64,64] -b 100 --no-plots"
        )

        results = run_main(command_str)
        solver = results["solver"]
        solver.actor_critic = torch.load('TestData/test_ddpg_ac_lunar_lander.pth')
        solver.target_actor_critic = torch.load('TestData/test_ddpg_ac_target_lunar_lander.pth')
        states = torch.Tensor(np.load('TestData/ddpg_states_lander.npy'))
        rewards = torch.Tensor(np.load('TestData/ddpg_rewards_lander.npy'))
        dones = torch.Tensor(np.load('TestData/ddpg_dones_lander.npy'))
        values = solver.compute_target_values(states, rewards, dones)
        target = np.load("TestData/test_ddpg_ctv_lander.npy")
        self.assertTrue(
            l2_distance_bounded(
                values.detach().numpy(),
                target,
                1e-2,
            ),
            "`test_compute_target_values' returns unexpected values.",
        )
        self.__class__.points += 4

    def test_pi_loss(self):
       
        command_str = (
            "-s ddpg -t 1000 -d LunarLanderContinuous-v2 -e 1 -a 0.001 -g 0.99 -l [64,64] -b 100 --no-plots"
        )
        results = run_main(command_str)
        solver = results["solver"]
        solver.actor_critic = torch.load('TestData/test_ddpg_ac_lunar_lander.pth')
        states = torch.Tensor(np.load('TestData/ddpg_states_lander.npy'))
        target = np.load("TestData/test_ddpg_pi_loss_lander.npy")
        self.assertTrue(
            l2_distance_bounded(
                solver.pi_loss(states).detach().numpy(),
                target,
                1e-3,
            ),
            "`test_pi_loss' returns unexpected values.",
        )
        self.__class__.points += 2
    
    def test_lander_rewards(self):
        command_str = (
            "-s ddpg -t 1000 -d LunarLanderContinuous-v2 -e 1000 -a 0.001 -g 0.99 -l [64,64] -b 100 --no-plots"
        )
        results = run_main(command_str)
        
        stats = results["stats"]
        smoothing_window = 10
        rewards_smoothed = (
            pd.Series(stats.episode_rewards)
            .rolling(smoothing_window, min_periods=smoothing_window)
            .mean()
        )

        self.assertTrue(
            np.mean(rewards_smoothed[:15]) < -150 and 
            np.mean(rewards_smoothed[550:]) > -150 and 
            np.max(rewards_smoothed[700:]) > 150,
            "got unexpected rewards for lunar_lander; verify implementation of ``train_episode''",
        )
        self.__class__.points += 4

    @classmethod
    def tearDownClass(cls):
        print("\nTotal Points: {} / 10".format(cls.points))


if __name__ == "__main__":
    import sys

    assert len(sys.argv) == 2
    unittest.main()

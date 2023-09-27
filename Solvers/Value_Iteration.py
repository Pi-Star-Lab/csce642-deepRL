# Licensing Information:  You are free to use or extend this codebase for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) inform Guni Sharon at 
# guni@tamu.edu regarding your usage (relevant statistics is reported to NSF).
# The development of this assignment was supported by NSF (IIS-2238979).
# Contributors:
# The core code base was developed by Guni Sharon (guni@tamu.edu).

import numpy as np
import heapq
from Solvers.Abstract_Solver import AbstractSolver, Statistics


class ValueIteration(AbstractSolver):
    def __init__(self, env, eval_env, options):
        assert str(env.observation_space).startswith("Discrete"), (
            str(self) + " cannot handle non-discrete state spaces"
        )
        assert str(env.action_space).startswith("Discrete"), (
            str(self) + " cannot handle non-discrete action spaces"
        )
        super().__init__(env, eval_env, options)
        self.V = np.zeros(env.observation_space.n)

    def train_episode(self):
        """
        Inputs: (Available/Useful variables)
            self.env
                this the OpenAI GYM environment
                     see https://gymnasium.farama.org/index.html

            state, _ = self.env.reset():
                Resets the environment and returns the starting state

            self.env.observation_space.n:
                number of states in the environment

            self.env.action_space.n:
                number of actions in the environment

            for probability, next_state, reward, done in self.env.P[state][action]:
                `probability` will be probability of `next_state` actually being the next state
                `reward` is the short-term/immediate reward for achieving that next state
                `done` is a boolean of whether or not that next state is the last/terminal state

                Every action has a chance (at least theoretically) of different outcomes (states)
                This is why `self.env.P[state][action]` is a list of outcomes and not a single outcome

            self.options.gamma:
                The discount factor (gamma from the slides)

        Outputs: (what you need to update)
            self.V:
                This is a numpy array, but you can think of it as a dictionary
                `self.V[state]` should return a floating point value that
                represents the value of a state. This value should become
                more accurate with each episode.

                How should this be calculated?
                    look at the value iteration algorithm
                    Ref: Sutton book eq. 4.10.
                Once those values have been updated, that's it for this function/class
        """

        # you can add variables here if it is helpful

        # Update the estimated value of each state
        for each_state in range(self.env.observation_space.n):
            # Do a one-step lookahead to find the best action
            # Update the value function. Ref: Sutton book eq. 4.10.
            ################################
            #   YOUR IMPLEMENTATION HERE   #
            ################################

        # Dont worry about this part
        self.statistics[Statistics.Rewards.value] = np.sum(self.V)
        self.statistics[Statistics.Steps.value] = -1

    def __str__(self):
        return "Value Iteration"

    def one_step_lookahead(self, state: int):
        """
        Helper function to calculate the value for all actions from a given state.
        Args:
            state: The state to consider (int)
            V: The value to use as an estimator, Vector of length self.env.observation_space.n
        Returns:
            A vector of length self.env.action_space.n containing the expected value of each action.
        """
        A = np.zeros(self.env.action_space.n)
        for a in range(self.env.action_space.n):
            for prob, next_state, reward, done in self.env.P[state][a]:
                A[a] += prob * (reward + self.options.gamma * self.V[next_state])
        return A

    def create_greedy_policy(self):
        """
        Creates a greedy policy based on state values.
        Use:
            self.env.action_space.n: Number of actions in the environment.
        Returns:
            A function that takes an observation as input and returns a Greedy
               action
        """

        def policy_fn(state):
            """
            What is this function?
                This function is the part that decides what action to take

            Inputs: (Available/Useful variables)
                self.V[state]
                    the estimated long-term value of getting to a state

                values = self.one_step_lookahead(state)
                    len(values) will be the number of actions (self.env.nA)
                    values[action] will be the expected value of that action (float)

                for probability, next_state, reward, done in self.env.P[state][action]:
                    `probability` will be the probability of `next_state` actually being the next state
                    `reward` is the short-term/immediate reward for achieving that next state
                    `done` is a boolean of whether or not that next state is the last/terminal state

                    Every action has a chance (at least theoretically) of different outcomes (states)
                    This is why `self.env.P[state][action]` is a list of outcomes and not a single outcome

                self.self.env.observation_space.n:
                    number of states in the environment

                self.self.env.action_space.n:
                    number of actions in the environment

                self.options.gamma:
                    The discount factor (gamma from the slides)

            Outputs: (what you need to output)
                return action as an integer
            """
            ################################
            #   YOUR IMPLEMENTATION HERE   #
            ################################
            

        return policy_fn


class AsynchVI(ValueIteration):
    def __init__(self, env, eval_env, options):
        super().__init__(env, eval_env, options)
        # list of States to be updated by priority
        self.pq = PriorityQueue()
        # A mapping from each state to all states potentially leading to it in a single step
        self.pred = {}
        for s in range(self.env.observation_space.n):
            # Do a one-step lookahead to find the best action
            A = self.one_step_lookahead(s)
            best_action_value = np.max(A)
            self.pq.push(s, -abs(self.V[s] - best_action_value))
            for a in range(self.env.action_space.n):
                for prob, next_state, reward, done in self.env.P[s][a]:
                    if prob > 0:
                        if next_state not in self.pred.keys():
                            self.pred[next_state] = set()
                        if s not in self.pred[next_state]:
                            try:
                                self.pred[next_state].add(s)
                            except KeyError:
                                self.pred[next_state] = set()

    def train_episode(self):
        """
        What is this?
            same as other `train_episode` function above, but for Asynch value iteration

        New Inputs:

            self.pq.update(state, priority)
                priority is a number BUT more-negative == higher priority

            state = self.pq.pop()
                this gets the state with the highest priority

        Update:
            self.V
                this is still the same as the previous
        """

        #########################################################
        # YOUR IMPLEMENTATION HERE                              #
        # Choose state with the maximal value change potential  #
        # Do a one-step lookahead to find the best action       #
        # Update the value function. Ref: Sutton book eq. 4.10. #
        #########################################################

        # you can ignore this part
        self.statistics[Statistics.Rewards.value] = np.sum(self.V)
        self.statistics[Statistics.Steps.value] = -1

    def __str__(self):
        return "Asynchronous VI"


class PriorityQueue:
    """
    Implements a priority queue data structure. Each inserted item
    has a priority associated with it and the client is usually interested
    in quick retrieval of the lowest-priority item in the queue. This
    data structure allows O(1) access to the lowest-priority item.
    """

    def __init__(self):
        self.heap = []
        self.count = 0

    def push(self, item, priority):
        entry = (priority, self.count, item)
        heapq.heappush(self.heap, entry)
        self.count += 1

    def pop(self):
        (_, _, item) = heapq.heappop(self.heap)
        return item

    def isEmpty(self):
        return len(self.heap) == 0

    def update(self, item, priority):
        # If item already in priority queue with higher priority, update its priority and rebuild the heap.
        # If item already in priority queue with equal or lower priority, do nothing.
        # If item not in priority queue, do the same thing as self.push.
        for index, (p, c, i) in enumerate(self.heap):
            if i == item:
                if p <= priority:
                    break
                del self.heap[index]
                self.heap.append((priority, c, item))
                heapq.heapify(self.heap)
                break
        else:
            self.push(item, priority)

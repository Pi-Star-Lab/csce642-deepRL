# Licensing Information:  You are free to use or extend this codebase for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide the following
# attribution:
# This CSCE-689 RL assignment codebase was developed at Texas A&M University.
# The core code base was developed by Guni Sharon (guni@tamu.edu).

solvers = ['random', 'vi', 'pi', 'mc','avi', 'mcis', 'ql', 'sarsa', 'aql', 'dqn', 'reinforce', 'a2c']

def get_solver_class(name):
    if name == solvers[0]:
        from Solvers.Random_Walk import RandomWalk
        return RandomWalk
    elif name == solvers[1]:
        from Solvers.Value_Iteration import ValueIteration
        return ValueIteration
    elif name == solvers[2]:
        from Solvers.Policy_Iteration import PolicyIteration
        return PolicyIteration
    elif name == solvers[3]:
        from Solvers.Monte_Carlo import MonteCarlo
        return MonteCarlo
    elif name == solvers[4]:
        from Solvers.Value_Iteration import AsynchVI
        return AsynchVI
    elif name == solvers[5]:
        from Solvers.Monte_Carlo import OffPolicyMC
        return OffPolicyMC
    elif name == solvers[6]:
        from Solvers.Q_Learning import QLearning
        return QLearning
    elif name == solvers[7]:
        from Solvers.SARSA import Sarsa
        return Sarsa
    elif name == solvers[8]:
        from Solvers.Q_Learning import ApproxQLearning
        return ApproxQLearning
    elif name == solvers[9]:
        from Solvers.DQN import DQN
        return DQN
    elif name == solvers[10]:
        from Solvers.REINFORCE import Reinforce
        return Reinforce
    elif name == solvers[11]:
        from Solvers.A2C import A2C
        return A2C
    else:
        assert False, "unknown solver name {}. solver must be from {}".format(name, str(solvers))

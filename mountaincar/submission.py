import math, random
from collections import defaultdict
from typing import List, Callable, Tuple, Dict, Any, Optional, Iterable, Set
import gymnasium as gym
import numpy as np

import util
from util import ContinuousGymMDP, StateT, ActionT
from custom_mountain_car import CustomMountainCarEnv

############################################################
# Problem 3a
# Implementing Value Iteration on Number Line (from Problem 1)
def valueIteration(succAndRewardProb: Dict[Tuple[StateT, ActionT], List[Tuple[StateT, float, float]]], discount: float, epsilon: float = 0.001):
    '''
    Given transition probabilities and rewards, computes and returns V and
    the optimal policy pi for each state.
    - succAndRewardProb: Dictionary mapping tuples of (state, action) to a list of (nextState, prob, reward) Tuples.
    - Returns: Dictionary mapping each state to an action.
    '''
    # Define a mapping from states to Set[Actions] so we can determine all the actions that can be taken from s.
    # You may find this useful in your approach.
    stateActions = defaultdict(set)
    for state, action in succAndRewardProb.keys():
        stateActions[state].add(action)

    def computeQ(V: Dict[StateT, float], state: StateT, action: ActionT) -> float:
        # Return Q(state, action) based on V(state)
        # Calculate Q(state, action) = sum_{s'} P(s'|s,a) * [R(s,a,s') + discount * V(s')]
        return sum(prob * (reward + discount * V[nextState]) for nextState, prob, reward in succAndRewardProb[(state, action)])

    def computePolicy(V: Dict[StateT, float]) -> Dict[StateT, ActionT]:
        # Return the policy given V.
        # Remember the policy for a state is the action that gives the greatest Q-value
        policy = {}
        for state in stateActions:
            # Choose the action that maximizes Q-value, break ties by choosing larger action
            policy[state] = max(stateActions[state], key=lambda a: (computeQ(V, state, a), a))
        return policy

    print('Running valueIteration...')
    V = defaultdict(float) # This will return 0 for states not seen (handles terminal states)
    numIters = 0
    while True:
        newV = defaultdict(float) # This will return 0 for states not seen (handles terminal states)
        # update V values using the computeQ function above.
        # repeat until the V values for all states do not change by more than epsilon.
        maxDiff = 0
        for state in stateActions:
            if stateActions[state]:  # Only update if state has actions
                newV[state] = max(computeQ(V, state, action) for action in stateActions[state])
                maxDiff = max(maxDiff, abs(newV[state] - V[state]))
        if maxDiff < epsilon:
            break
        V = newV
        numIters += 1
    V_opt = V
    print(("valueIteration: %d iterations" % numIters))
    return computePolicy(V_opt)

############################################################
# Problem 3b
# Model-Based Monte Carlo

# Runs value iteration algorithm on the number line MDP
# and prints out optimal policy for each state.
def run_VI_over_numberLine(mdp: util.NumberLineMDP):
    succAndRewardProb = {
        (-mdp.n + 1, 1): [(-mdp.n + 2, 0.2, mdp.penalty), (-mdp.n, 0.8, mdp.leftReward)],
        (-mdp.n + 1, 2): [(-mdp.n + 2, 0.3, mdp.penalty), (-mdp.n, 0.7, mdp.leftReward)],
        (mdp.n - 1, 1): [(mdp.n - 2, 0.8, mdp.penalty), (mdp.n, 0.2, mdp.rightReward)],
        (mdp.n - 1, 2): [(mdp.n - 2, 0.7, mdp.penalty), (mdp.n, 0.3, mdp.rightReward)]
    }

    for s in range(-mdp.n + 2, mdp.n - 1):
        succAndRewardProb[(s, 1)] = [(s+1, 0.2, mdp.penalty), (s - 1, 0.8, mdp.penalty)]
        succAndRewardProb[(s, 2)] = [(s+1, 0.3, mdp.penalty), (s - 1, 0.7, mdp.penalty)]

    pi = valueIteration(succAndRewardProb, mdp.discount)
    return pi


class ModelBasedMonteCarlo(util.RLAlgorithm):
    def __init__(self, actions: List[ActionT], discount: float, calcValIterEvery: int = 10000,
                 explorationProb: float = 0.2,) -> None:
        self.actions = actions
        self.discount = discount
        self.calcValIterEvery = calcValIterEvery
        self.explorationProb = explorationProb
        self.numIters = 0

        # (state, action) -> {nextState -> ct} for all nextState
        self.tCounts = defaultdict(lambda: defaultdict(int))
        # (state, action) -> {nextState -> totalReward} for all nextState
        self.rTotal = defaultdict(lambda: defaultdict(float))

        self.pi = {} # Optimal policy for each state. state -> action

    # This algorithm will produce an action given a state.
    # Here we use the epsilon-greedy algorithm: with probability |explorationProb|, take a random action.
    # Should return random action if the given state is not in self.pi.
    # The input boolean |explore| indicates whether the RL algorithm is in train or test time. If it is false (test), we
    # should always follow the policy if available.
    # HINT: Use random.random() (not np.random()) to sample from the uniform distribution [0, 1]
    def getAction(self, state: StateT, explore: bool = True) -> ActionT:
        if explore:
            self.numIters += 1
        explorationProb = self.explorationProb
        if self.numIters < 2e4: # Always explore
            explorationProb = 1.0
        elif self.numIters > 1e6: # Lower the exploration probability by a logarithmic factor.
            explorationProb = explorationProb / math.log(self.numIters - 100000 + 1)
        
        if not explore or random.random() > explorationProb:
            # Exploit: return action from policy if available, otherwise random
            return self.pi.get(state, random.choice(self.actions))
        else:
            # Explore: choose random action
            return random.choice(self.actions)

    # We will call this function with (s, a, r, s'), which is used to update tCounts and rTotal.
    # For every self.calcValIterEvery steps, runs value iteration after estimating succAndRewardProb.
    def incorporateFeedback(self, state: StateT, action: ActionT, reward: int, nextState: StateT, terminal: bool):

        self.tCounts[(state, action)][nextState] += 1
        self.rTotal[(state, action)][nextState] += reward

        if self.numIters % self.calcValIterEvery == 0:
            # Estimate succAndRewardProb based on self.tCounts and self.rTotal.
            # Hint 1: prob(s, a, s') = (counts of transition (s,a) -> s') / (total transtions from (s,a))
            # Hint 2: Reward(s, a, s') = (total reward of (s,a) -> s') / (counts of transition (s,a) -> s')
            # Then run valueIteration and update self.pi.
            succAndRewardProb = defaultdict(list)
            for (s, a), next_states in self.tCounts.items():
                total_count = sum(next_states.values())
                if total_count > 0:  # Only include if we have observations
                    for ns, count in next_states.items():
                        prob = count / total_count
                        avg_reward = self.rTotal[(s, a)][ns] / count
                        succAndRewardProb[(s, a)].append((ns, prob, avg_reward))
            
            # Update policy using value iteration
            self.pi = valueIteration(succAndRewardProb, self.discount)

############################################################
# Problem 4a
# Performs Tabular Q-learning. Read util.RLAlgorithm for more information.
class TabularQLearning(util.RLAlgorithm):
    def __init__(self, actions: List[ActionT], discount: float,
                 explorationProb: float = 0.2, initialQ: float = 0):
        '''
        - actions: the list of valid actions
        - discount: a number between 0 and 1, which determines the discount factor
        - explorationProb: the epsilon value indicating how frequently the policy returns a random action
        - intialQ: the value for intializing Q values.
        '''
        self.actions = actions
        self.discount = discount
        self.explorationProb = explorationProb
        self.Q = defaultdict(lambda: initialQ)
        self.numIters = 0

    # This algorithm will produce an action given a state.
    # Here we use the epsilon-greedy algorithm: with probability |explorationProb|, take a random action.
    # The input boolean |explore| indicates whether the RL algorithm is in train or test time. If it is false (test), we
    # should always choose the maximum Q-value action.
    # HINT 1: You can access Q-value with self.Q[state, action]
    # HINT 2: Use random.random() to sample from the uniform distribution [0, 1]
    def getAction(self, state: StateT, explore: bool = True) -> ActionT:
        if explore:
            self.numIters += 1
        explorationProb = self.explorationProb
        if self.numIters < 2e4: # explore
            explorationProb = 1.0
        elif self.numIters > 1e5: # Lower the exploration probability by a logarithmic factor.
            explorationProb = explorationProb / math.log(self.numIters - 100000 + 1)
        
        if not explore or random.random() > explorationProb:
            # Choose action with highest Q-value
            return max(self.actions, key=lambda a: self.Q[state, a])
        else:
            # Random exploration
            return random.choice(self.actions)

    # Call this function to get the step size to update the weights.
    def getStepSize(self) -> float:
        return 0.1

    # We will call this function with (s, a, r, s'), which you should use to update |Q|.
    # Note that if s' is a terminal state, then terminal will be True.  Remember to check for this.
    # You should update the Q values using self.getStepSize() 
    # HINT 1: The target V for the current state is a combination of the immediate reward
    # and the discounted future value.
    # HINT 2: V for terminal states is 0
    def incorporateFeedback(self, state: StateT, action: ActionT, reward: float, nextState: StateT, terminal: bool) -> None:
        # Get current Q-value
        step_size = self.getStepSize()
        
        # Calculate target value
        if terminal:
            target = reward
        else:
            # Target is reward + discount * max_{a'} Q(s', a')
            next_q_values = [self.Q[nextState, next_action] for next_action in self.actions]
            target = reward + self.discount * max(next_q_values)
        
        # Update Q-value
        self.Q[state, action] = (1 - step_size) * self.Q[state, action] + step_size * target

############################################################
# Problem 4b: Fourier feature extractor

def fourierFeatureExtractor(
        state: StateT,
        maxCoeff: int = 5,
        scale: Optional[Iterable] = None
    ) -> np.ndarray:
    '''
    For state (x, y, z), maxCoeff 2, and scale [2, 1, 1], this should output (in any order):
    [1, cos(pi * 2x), cos(pi * y), cos(pi * z),
     cos(pi * (2x + y)), cos(pi * (2x + z)), cos(pi * (y + z)),
     cos(pi * (4x)), cos(pi * (2y)), cos(pi * (2z)),
     cos(pi*(4x + y)), cos(pi * (4x + z)), ..., cos(pi * (4x + 2y + 2z))]
    '''
    if scale is None:
        scale = np.ones_like(state)
    features = None

    # Below, implement the fourier feature extractor as similar to the doc string provided.
    # The return shape should be 1 dimensional ((maxCoeff+1)^(len(state)),).
    #
    # HINT: refer to util.polynomialFeatureExtractor as a guide for
    # doing efficient arithmetic broadcasting in numpy.

    # Create the dimensions for our state space
    dims = [np.arange(maxCoeff + 1) for _ in range(len(state))]
    
    # Create all combinations of coefficients using meshgrid
    coeffs = np.meshgrid(*dims)
    coeffs = np.vstack([c.ravel() for c in coeffs]).T
    
    # Scale the state and compute dot product with coefficients
    scaled_state = np.array(state) * np.array(scale)
    features = np.cos(np.pi * np.dot(coeffs, scaled_state))

    return features

############################################################
# Problem 4c: Q-learning with Function Approximation
# Performs Function Approximation Q-learning. Read util.RLAlgorithm for more information.
class FunctionApproxQLearning(util.RLAlgorithm):
    def __init__(self, featureDim: int, featureExtractor: Callable, actions: List[int],
                 discount: float, explorationProb=0.2):
        '''
        - featureDim: the dimensionality of the output of the feature extractor
        - featureExtractor: a function that takes a state and returns a numpy array representing the feature.
        - actions: the list of valid actions
        - discount: a number between 0 and 1, which determines the discount factor
        - explorationProb: the epsilon value indicating how frequently the policy returns a random action
        '''
        self.featureDim = featureDim
        self.featureExtractor = featureExtractor
        self.actions = actions
        self.discount = discount
        self.explorationProb = explorationProb
        self.W = np.random.standard_normal(size=(featureDim, len(actions)))
        self.numIters = 0

    def getQ(self, state: np.ndarray, action: int) -> float:
        # Extract features for the state
        features = self.featureExtractor(state)
        # Calculate Q-value using features and weights
        return float(features.dot(self.W[:, action]))

    # This algorithm will produce an action given a state.
    # Here we use the epsilon-greedy algorithm: with probability |explorationProb|, take a random action.
    # The input boolean |explore| indicates whether the RL algorithm is in train or test time. If it is false (test), we
    # should always choose the maximum Q-value action.
    # HINT: This function should be the same as your implementation for 4a.
    def getAction(self, state: np.ndarray, explore: bool = True) -> int:
        if explore:
            self.numIters += 1
        explorationProb = self.explorationProb
        if self.numIters < 2e4: # Always explore
            explorationProb = 1.0
        elif self.numIters > 1e5: # Lower the exploration probability by a logarithmic factor.
            explorationProb = explorationProb / math.log(self.numIters - 100000 + 1)

        if not explore or random.random() > explorationProb:
            # Choose action with highest Q-value
            return int(max(range(len(self.actions)), key=lambda a: self.getQ(state, a)))
        else:
            # Random exploration
            return random.randrange(len(self.actions))

    # Call this function to get the step size to update the weights.
    def getStepSize(self) -> float:
        return 0.005 * (0.99)**(self.numIters / 500)

    # We will call this function with (s, a, r, s'), which you should use to update |weights|.
    # Note that if s' is a terminal state, then terminal will be True.  Remember to check for this.
    # You should update W using self.getStepSize()
    # HINT 1: this part will look similar to 4a, but you are updating self.W
    # HINT 2: review the function approximation module for the update rule
    def incorporateFeedback(self, state: np.ndarray, action: int, reward: float, nextState: np.ndarray, terminal: bool) -> None:
        # Get current Q-value
        features = self.featureExtractor(state)
        
        # Calculate target value
        if terminal:
            target = reward
        else:
            # Target is reward + discount * max_{a'} Q(s', a')
            next_q_values = [self.getQ(nextState, a) for a in range(len(self.actions))]
            target = reward + self.discount * max(next_q_values)
        
        # Calculate prediction
        prediction = self.getQ(state, action)
        
        # Update weights using gradient descent
        step_size = self.getStepSize()
        self.W[:, action] += step_size * (target - prediction) * features

############################################################
# Problem 5c: Constrained Q-learning

class ConstrainedQLearning(FunctionApproxQLearning):
    def __init__(self, featureDim: int, featureExtractor: Callable, actions: List[int],
                 discount: float, force: float, gravity: float,
                 max_speed: Optional[float] = None,
                 explorationProb=0.2):
        super().__init__(featureDim, featureExtractor, actions,
                         discount, explorationProb)
        self.force = force
        self.gravity = gravity
        self.max_speed = max_speed

    # This algorithm will produce an action given a state.
    # Here we use the epsilon-greedy algorithm: with probability |explorationProb|, take a random action.
    # The input boolean |explore| indicates whether the RL algorithm is in train or test time. If it is false (test), we
    # should always choose the maximum Q-value action that is valid.
    def getAction(self, state: np.ndarray, explore: bool = True) -> int:
        if explore:
            self.numIters += 1
        explorationProb = self.explorationProb
        if self.numIters < 2e4: # Always explore
            explorationProb = 1.0
        elif self.numIters > 1e5: # Lower the exploration probability by a logarithmic factor.
            explorationProb = explorationProb / math.log(self.numIters - 100000 + 1)

        def is_valid_action(action, current_velocity):
            # Get next velocity based on mountain car physics
            velocity = current_velocity
            position = state[0]
            if action == 0: 
                action_force = -1.0
            elif action == 2: 
                action_force = 1.0
            else: 
                action_force = 0.0
            
            next_velocity = velocity + action_force * self.force - math.cos(3 * position) * self.gravity
            if self.max_speed is not None and abs(next_velocity) > self.max_speed:
                return False
            return True

        # Get valid actions
        valid_actions = [a for a in range(len(self.actions)) if is_valid_action(a, state[1])]
        
        if not valid_actions:
            # If no valid actions (rare edge case), return action that keeps velocity closest to constraint
            return 1  # No acceleration
        
        if not explore or random.random() > explorationProb:
            # Choose valid action with highest Q-value
            return max(valid_actions, key=lambda a: self.getQ(state, a))
        else:
            # Random exploration among valid actions
            return random.choice(valid_actions)

############################################################
# This is helper code for comparing the predicted optimal
# actions for 2 MDPs with varying max speed constraints
gym.register(
    id="CustomMountainCar-v0",
    entry_point="custom_mountain_car:CustomMountainCarEnv",
    max_episode_steps=1000,
    reward_threshold=-110.0,
)

mdp1 = ContinuousGymMDP("CustomMountainCar-v0", discount=0.999, timeLimit=1000)
mdp2 = ContinuousGymMDP("CustomMountainCar-v0", discount=0.999, timeLimit=1000)

# This is a helper function for 5c. This function runs
# ConstrainedQLearning, then simulates various trajectories through the MDP
# and compares the frequency of various optimal actions.
def compare_MDP_Strategies(mdp1: ContinuousGymMDP, mdp2: ContinuousGymMDP):
    rl1 = ConstrainedQLearning(
        36,
        lambda s: fourierFeatureExtractor(s, maxCoeff=5, scale=[1, 15]),
        mdp1.actions,
        mdp1.discount,
        mdp1.env.unwrapped.force,
        mdp1.env.unwrapped.gravity,
        10000,
        explorationProb=0.2,
    )
    rl2 = ConstrainedQLearning(
        36,
        lambda s: fourierFeatureExtractor(s, maxCoeff=5, scale=[1, 15]),
        mdp2.actions,
        mdp2.discount,
        mdp2.env.unwrapped.force,
        mdp2.env.unwrapped.gravity,
        0.065,
        explorationProb=0.2,
    )
    sampleKRLTrajectories(mdp1, rl1)
    sampleKRLTrajectories(mdp2, rl2)

def sampleKRLTrajectories(mdp: ContinuousGymMDP, rl: ConstrainedQLearning):
    accelerate_left, no_accelerate, accelerate_right = 0, 0, 0
    for n in range(100):
        traj = util.sample_RL_trajectory(mdp, rl)
        accelerate_left = traj.count(0)
        no_accelerate = traj.count(1)
        accelerate_right = traj.count(2)

    print(f"\nRL with MDP -> start state:{mdp.startState()}, max_speed:{rl.max_speed}")
    print(f"  *  total accelerate left actions: {accelerate_left}, total no acceleration actions: {no_accelerate}, total accelerate right actions: {accelerate_right}")


##command for running the code=C:/Users/Asus/AppData/Local/Programs/Python/Python313/python.exe train.py --agent value-iteration
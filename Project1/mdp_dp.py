### MDP Value Iteration and Policy Iteration
### Reference: https://web.stanford.edu/class/cs234/assignment1/index.html 

import numpy as np

np.set_printoptions(precision=3)

"""
For policy_evaluation, policy_improvement, policy_iteration and value_iteration,
the parameters P, nS, nA, gamma are defined as follows:

	P: nested dictionary
		From gym.core.Environment
		For each pair of states in [1, nS] and actions in [1, nA], P[state][action] is a
		tuple of the form (probability, nextstate, reward, terminal) where
			- probability: float
				the probability of transitioning from "state" to "nextstate" with "action"
			- nextstate: int
				denotes the state we transition to (in range [0, nS - 1])
			- reward: int
				either 0 or 1, the reward for transitioning from "state" to
				"nextstate" with "action"
			- terminal: bool
			  True when "nextstate" is a terminal state (hole or goal), False otherwise
	nS: int
		number of states in the environment
	nA: int
		number of actions in the environment
	gamma: float
		Discount factor. Number in range [0, 1)
"""

def policy_evaluation(P, nS, nA, policy, gamma=0.9, tol=1e-8):
    """Evaluate the value function from a given policy.

    Parameters:
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    policy: np.array[nS,nA]
        The policy to evaluate. Maps states to actions.
    tol: float
        Terminate policy evaluation when
            max |value_function(s) - prev_value_function(s)| < tol
    Returns:
    -------
    value_function: np.ndarray[nS]
        The value function of the given policy, where value_function[s] is
        the value of state s
    """
    
    value_function = np.zeros(nS)
    ############################
    # YOUR IMPLEMENTATION HERE #
    
    #    delta = 0 
    #    for s in range(nS):
    #        v = value_function[s]
    #        Vnew = sum over a of policy(A|S) sum over sprime and r of  prob * (sprime r given s a)[r + gamma * V(s')]
    #        delta  = max( delta, v - Vnew)
    #    Stop when delta < tol
    while True:
        delta = 0
        for s in range(nS):
            v = value_function[s]
            va = getValue(P,value_function,s,policy[s],gamma)
            value_function[s] = sum(np.multiply(policy[s], va )) 
            delta = max(delta,abs(v - value_function[s]))
        if delta < tol: break    
    ############################
    return value_function

def getValue(P,V,s,a,gamma):
    """
    Get the Value of a give S for all a:
    S: int
    A: array of choices  1 x nA 

    """
    va = []
    # ap - Probablity of any action given the policy
    # action - range[0,nA]
    for ap,action in zip(a,range(0,len(a))):
        tmp = 0
        for p,s_, r, _ in P[s][action]:
            tmp += p*(r + gamma * V[s_])
        va.append(tmp)
    return va




def policy_improvement(P, nS, nA, value_from_policy, gamma=0.9):
    """Given the value function from policy improve the policy.

    Parameters:
    -----------
    P, nS, nA, gamma:
        defined at beginning of file
    value_from_policy: np.ndarray
        The value calculated from the policy
    Returns:
    --------
    new_policy: np.ndarray[nS,nA]
        A 2D array of floats. Each float is the probability of the action
        to take in that state according to the environment dynamics and the 
        given value function.
    """

    new_policy = np.ones([nS, nA]) / nA
	############################
	# YOUR IMPLEMENTATION HERE #
    
    
    for s in range(0,nS):
        #print(old_action)
        new_policy[s] = np.eye(4)[np.argmax(getValue(P,value_from_policy,s,a = [0,1,2,3],gamma = gamma))]   

	############################
    return new_policy


def policy_iteration(P, nS, nA, policy, gamma=0.9, tol=1e-8):
    """Runs policy iteration.

    You should call the policy_evaluation() and policy_improvement() methods to
    implement this method.

    Parameters
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    policy: policy to be updated
    tol: float
        tol parameter used in policy_evaluation()
    Returns:
    ----------
    new_policy: np.ndarray[nS,nA]
    V: np.ndarray[nS]
    """
    new_policy = policy.copy()
	############################
	# YOUR IMPLEMENTATION HERE #
    while True:
        policy_stable = True
        old_policy = new_policy
        V = policy_evaluation(P,nS,nA,new_policy,gamma,tol)
        new_policy = policy_improvement(P, nS, nA, V, gamma)
        if not np.array_equal(old_policy,new_policy): policy_stable = False
        if policy_stable: break
        
        #print(new_policy)
    
	############################
    return new_policy, V

def value_iteration(P, nS, nA, V, gamma=0.9, tol=1e-8):
    """
    Learn value function and policy by using value iteration method for a given
    gamma and environment.

    Parameters:
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    V: value to be updated
    tol: float
        Terminate value iteration when
            max |value_function(s) - prev_value_function(s)| < tol
    Returns:
    ----------
    policy_new: np.ndarray[nS,nA]
    V_new: np.ndarray[nS]
    """
    
    V_new = V.copy()
    ############################
    # YOUR IMPLEMENTATION HERE #
    policy_new = np.zeros([nS,nA])
    while True:
        delta = 0
        for s in range(0,nS):
            v = V_new[s]
            va = getValue(P, V_new, s, [0, 1, 2, 3], gamma)
            V_new[s] = max(va)
            delta = max(delta,abs( v - V_new[s]))
            bestAction = np.argmax(va)
            policy_new[s] = np.eye(nA)[bestAction]
        if delta < tol:break
    ############################
    return policy_new, V_new

    
def render_single(env, policy, render = False, n_episodes=100):
    """
    Given a game envrionemnt of gym package, play multiple episodes of the game.
    An episode is over when the returned value for "done" = True.
    At each step, pick an action and collect the reward and new state from the game.

    Parameters:
    ----------
    env: gym.core.Environment
      Environment to play on. Must have nS, nA, and P as attributes.
    policy: np.array of shape [env.nS, env.nA]
      The action to take at a given state
    render: whether or not to render the game(it's slower to render the game)
    n_episodes: the number of episodes to play in the game. 
    Returns:
    ------
    total_rewards: the total number of rewards achieved in the game.
    """
    total_rewards = 0
    for _ in range(n_episodes):
        ob = env.reset() # initialize the episode
        done = False
        while not done:
            if render:
                env.render() # render the game
            ############################
            # YOUR IMPLEMENTATION HERE #
            action = np.argmax(policy[ob])
            ob, reward,done,prob = env.step(action)
            total_rewards+=reward
            
            
            
    return total_rewards




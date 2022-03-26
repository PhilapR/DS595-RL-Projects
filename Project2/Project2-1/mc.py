#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%%
import numpy as np
import random
from collections import defaultdict

#-------------------------------------------------------------------------
'''
    Monte-Carlo
    In this problem, you will implememnt an AI player for Blackjack.
    The main goal of this problem is to get familar with Monte-Carlo algorithm.
    You could test the correctness of your code
    by typing 'nosetests -v mc_test.py' in the terminal.

    You don't have to follow the comments to write your code. They are provided
    as hints in case you need.
'''
#-------------------------------------------------------------------------
#%%
def initial_policy(observation):
    """A policy that sticks if the player score is >= 20 and hits otherwise

    Parameters:
    -----------
    observation

    Returns:
    --------
    action: 0 or 1
        0: STICK
        1: HIT
    """
    ############################
    # YOUR IMPLEMENTATION HERE #
    # get parameters from observation
    score, dealer_score, usable_ace = observation
    action = 1
    if score >= 20:
        action = 0

    ############################
    return action
#%%
def mc_prediction(policy, env, n_episodes, gamma = 1.0):
    """Given policy using sampling to calculate the value function
        by using Monte Carlo first visit algorithm.

    Parameters:
    -----------
    policy: function
        A function that maps an obversation to action probabilities
    env: function
        OpenAI gym environment
    n_episodes: int
        Number of episodes to sample
    gamma: float
        Gamma discount factor
    Returns:
    --------
    V: defaultdict(float)
        A dictionary that maps from state to value

    Note: at the begining of each episode, you need initialize the environment using env.reset()
    """
    # initialize empty dictionaries
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    
    returns = defaultdict(list)
    
    # a nested dictionary that maps state -> value
    V = defaultdict(float)

    ############################
    # YOUR IMPLEMENTATION HERE #
    # loop each episode
    for _ in range(n_episodes):
        #episode list in for of [[state,action,reward] ... [state,action,reward]]
        epi_list = gen_episode(policy,env)
        G = 0
       
        for t in range(len(epi_list)-1,-1,-1):
            s, a, r = epi_list[t]
            G = gamma * G + r
            if not s in [x[0] for x in epi_list[:t]]: 
                
                returns_count[s] += 1
                returns_sum[s] += G
                V[s] =  returns_sum[s] / returns_count[s]

   
    return V

    ############################
#%%

def gen_episode(policy, env):
    # initialize the episode
    obs = env.reset()
    # generate empty episode list
    epi_list = []
    # loop until episode generation is done
    while True:
        # select an action
        # return a reward and new state
        # append state, action, reward to episode
        # update state to new state
        action = policy(obs)
        new_obs, reward, done, _ = env.step(action)
        epi_list.append([obs, action, reward])
        obs = new_obs
        if done:
            break
    return epi_list
        
        
#%%
def epsilon_greedy(Q, state, nA, epsilon = 0.1):
    """Selects epsilon-greedy action for supplied state.

    Parameters:
    -----------
    Q: dict()
        A dictionary  that maps from state -> action-values,
        where Q[s][a] is the estimated action value corresponding to state s and action a.
    state: int
        current state
    nA: int
        Number of actions in the environment
    epsilon: float
        The probability to select a random action, range between 0 and 1

    Returns:
    --------
    action: int
        action based current state
    Hints:
    ------
    With probability (1 − epsilon) choose the greedy action.
    With probability epsilon choose an action at random.
    """
    ############################
    # YOUR IMPLEMENTATION HERE #
    action = None
    e = np.random.random()
    if e < 1 - epsilon: # P(Greedy) = 1-e
        bestq = -1000
        bestA = None
        for a in range(nA):
            q = Q[state][a]
            if q > bestq:
                bestA = a
                bestq = q
        action = bestA
        
    else:
        action = np.random.randint(nA)

    ############################
    return action

def mc_control_epsilon_greedy(env, n_episodes, gamma = 1.0, epsilon = 0.1):
    """Monte Carlo control with exploring starts.
        Find an optimal epsilon-greedy policy.

    Parameters:
    -----------
    env: function
        OpenAI gym environment
    n_episodes: int
        Number of episodes to sample
    gamma: float
        Gamma discount factor
    epsilon: float
        The probability to select a random action, range between 0 and 1
    Returns:
    --------
    Q: dict()
        A dictionary  that maps from state -> action-values,
        where Q[s][a] is the estimated action value corresponding to state s and action a.
    Hint:
    -----
    You could consider decaying epsilon, i.e. epsilon = epsilon-(0.1/n_episodes) during each episode
    and episode must > 0.
    """

    returns_sum = defaultdict(lambda: np.zeros(env.action_space.n))
    returns_count = defaultdict(lambda: np.zeros(env.action_space.n))
    # a nested dictionary that maps state -> (action -> action-value)
    # e.g. Q[state] = np.darrary(nA)
    
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    ############################
    # YOUR IMPLEMENTATION HERE #
    for i in range(n_episodes):
        
        # define decaying epsilon
        
        # initialize the episode
        obs = env.reset()
        # generate empty episode list
        epi_list = []
        # loop until one episode generation is done
        i=0
        while True:
            i += 1
            # get an action from epsilon greedy policy
            action = epsilon_greedy(Q, obs,2,  epsilon = epsilon * (.99 ** i))
            # return a reward and new state
            new_obs, reward, done, _ = env.step(action)
            # append state, action, reward to episode
            epi_list.append([obs,action, reward])
            # update state to new state
            obs = new_obs
            if done:
                break



    
        G = 0

        for t in range(len(epi_list)-1, -1, -1):
            s, a, r = epi_list[t]
            G = gamma * G + r
            if not (s,a) in [(x[0],x[1]) for x in epi_list[:t]]:

                returns_count[s][a] += 1
                returns_sum[s][a] += G
                Q[s][a] = returns_sum[s][a] / returns_count[s][a]

    return Q

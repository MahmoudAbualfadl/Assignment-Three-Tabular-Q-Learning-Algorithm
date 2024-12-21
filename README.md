# Assignment-Three-Tabular-Q-Learning-Algorithm

🚀 developing a reinforcement learning agent using Tabular Q-learning to control the actions of a virtual agent in the Cliff Walking environment.

<img src="https://github.com/user-attachments/assets/311743b4-d552-423d-9ac8-4423cc8619a3">

## Task:
 > Your task is to implement two versions of the agent: one without the epsilon-greedy
algorithm and one with the epsilon-greedy algorithm. Evaluate the performance of the agent
using appropriate metrics and plot the learning curve.

## The Cliff Walking Environment

This is a simple implementation of the Gridworld Cliff reinforcement learning task.
Adapted from Example 6.6 (page 106) from [Reinforcement Learning: An Introduction by Sutton and Barto](http://incompleteideas.net/book/bookdraft2018jan1.pdf).

## Description

The board is a 4x12 matrix, with (using NumPy matrix indexing):

    [3, 0] as the start at bottom-left

    [3, 11] as the goal at bottom-right

    [3, 1..10] as the cliff at bottom-center


Actions

There are 4 discrete deterministic actions:

    0: move up

    1: move right

    2: move down

    3: move left

Observations

There are 3x12 + 1 possible states. In fact, the agent cannot be at the cliff, nor at the goal (as this results in the end of the episode). It remains all the positions of the first 3 rows plus the bottom-left cell. The observation is simply the current position encoded as flattened index.
Reward

Each time step incurs -1 reward, and stepping into the cliff incurs -100 reward.
Arguments

gym.make('CliffWalking-v0')

Version History

    v0: Initial version release

## Overview of Q-learning 

Q-learning is an off-policy RL algorithm that learns the value of the optimal action independently of the policy being followed. It aims to learn the optimal action-value function, Q*(s,a) which gives the maximum expected future reward for an action a taken in state s. The update rule for Q-learning is:
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha (r_{t+1} + \gamma \max_{a'}Q(s_{t+1}a') - Q(s_t, a_t))

### Update Rules

-Q-learning: Uses the max operator to update Q-values, focusing on the best possible action.

### Let's break down the code step by step to understand how these Q-tables were generated:

### 1-Step 1: Import Libraries

```
import gym
import numpy as np
```

### Step 2: Initialize Environment

<Initializes the CliffWalking-v0 environment.>

```
env = gym.make('CliffWalking-v0')
```
### Step 3: Define Hyperparameters

 > Sets the learning rate, discount factor, epsilon-greedy parameter, and the number of episodes for training.

```
alpha = 0.1  # Learning rate
gamma = 0.99  # Discount factor
epsilon = 0.1  # Epsilon-greedy parameter
episodes = 500  # Number of episodes

```
### Step 4: Initialize Q-tables

> Creates two Q-tables, one for Q-learning and one for SARSA, initialized to zeros.

```
q_table_q_learning = np.zeros((env.observation_space.n, env.action_space.n))
q_table_sarsa = np.zeros((env.observation_space.n, env.action_space.n))

```

### Step 5: Define Epsilon-greedy Policy

> Defines a policy that chooses an action based on the epsilon-greedy approach.

```
def epsilon_greedy_policy(state, q_table):
    if np.random.rand() < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(q_table[state])
```

### Step 6: Q-learning Algorithm

> Trains the agent using the Q-learning algorithm by updating the Q-table based on the maximum expected future rewards.

```
def q_learning():
    for episode in range(episodes):
        state = env.reset()
        done = False
        
        while not done:
            action = epsilon_greedy_policy(state, q_table_q_learning)
            next_state, reward, done, _ = env.step(action)
            
            best_next_action = np.argmax(q_table_q_learning[next_state])
            td_target = reward + gamma * q_table_q_learning[next_state][best_next_action]
            td_error = td_target - q_table_q_learning[state][action]
            q_table_q_learning[state][action] += alpha * td_error
            
            state = next_state
```


### Step 7: Test the Policies

Tests the learned policies by running a single episode using the learned Q-tables and prints the total rewards received.

###   result

<img src="https://github.com/user-attachments/assets/8a4f3c1f-1fe3-417c-9e74-6c783fdb6343">






























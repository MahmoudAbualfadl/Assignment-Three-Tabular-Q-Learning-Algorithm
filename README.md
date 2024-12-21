# Assignment-Three-Tabular-Q-Learning-Algorithm
## 🚀 Overview
This project implements a reinforcement learning solution for the Cliff Walking environment using both standard Q-learning and epsilon-greedy approaches. The implementation provides comprehensive training, analysis, and visualization capabilities for comparing different Q-learning strategies.


<img src="https://github.com/user-attachments/assets/311743b4-d552-423d-9ac8-4423cc8619a3">

## Task:
 > Your task is to implement two versions of the agent: one without the epsilon-greedy
algorithm and one with the epsilon-greedy algorithm. Evaluate the performance of the agent
using appropriate metrics and plot the learning curve.

## The Cliff Walking Environment
## 📋 Features
- **Dual Implementation**: Both standard Q-learning and epsilon-greedy approaches
- **Automated Training**: Configurable training parameters with progress tracking
- **Performance Metrics**: Comprehensive metrics including rewards, steps, and success rates
- **Visualization Tools**: Advanced plotting capabilities with reward curves and Q-value heatmaps
- **Flexible Configuration**: Easily adjustable hyperparameters
- **Progress Monitoring**: Real-time training progress logging

## 🔧 Installation

### Prerequisites
- Python 3.8+
- OpenAI Gym
- NumPy
- Matplotlib
- Seaborn

```bash
pip install gym numpy matplotlib seaborn
```

## 💻 Usage

### Basic Usage
```python
from cliff_walking import CliffWalkingExperiment

# Create experiment instance
experiment = CliffWalkingExperiment("My Experiment")

# Train the agents
results = experiment.train()

# Visualize results
experiment.visualize_results()
```

### Configuration
You can customize the training parameters when initializing the experiment:

```python
params = {
    'alpha': 0.1,           # Learning rate
    'gamma': 0.99,          # Discount factor
    'epsilon': 0.1,         # Initial exploration rate
    'decay_rate': 0.995,    # Epsilon decay rate
    'min_epsilon': 0.01,    # Minimum exploration rate
    'episodes': 500         # Number of training episodes
}

experiment = CliffWalkingExperiment(exp_name="Custom Experiment")
experiment.params = params
```

## 🌟 The Cliff Walking Environment

The Cliff Walking task is a classic reinforcement learning problem from Sutton and Barto's book "Reinforcement Learning: An Introduction" (Example 6.6, page 106).

### Observations
There are 3x12 + 1 possible states. In fact, the agent cannot be at the cliff, nor at the goal (as this results in the end of the episode). It remains all the positions of the first 3 rows plus the bottom-left cell. The observation is simply the current position encoded as flattened index.
Reward

Each time step incurs -1 reward, and stepping into the cliff incurs -100 reward.
Arguments

gym.make('CliffWalking-v0')

Version History

    v0: Initial version release
    ## Description


### Environment Details
- **Grid Size**: 4x12 matrix
- **Start Position**: [3, 0] (bottom-left)
- **Goal Position**: [3, 11] (bottom-right)
- **Cliff**: [3, 1..10] (bottom-center)

### Actions
Four discrete deterministic actions:
- 0: Move up
- 1: Move right
- 2: Move down
- 3: Move left

### Rewards
- -1 per time step
- -100 for stepping into the cliff
- Episode ends upon reaching the goal or falling off the cliff

## 📊 Implementation Details

### Overview of Q-learning 

Q-learning is an off-policy RL algorithm that learns the value of the optimal action independently of the policy being followed. It aims to learn the optimal action-value function, Q*(s,a) which gives the maximum expected future reward for an action a taken in state s. The update rule for Q-learning is:
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha (r_{t+1} + \gamma \max_{a'}Q(s_{t+1}a') - Q(s_t, a_t))

### Q-Learning Algorithm
The implementation uses the Q-learning update rule:

```
Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
```

Where:
- `s`: Current state
- `a`: Current action
- `r`: Reward
- `s'`: Next state
- `α`: Learning rate
- `γ`: Discount factor

### Epsilon-Greedy Strategy
The epsilon-greedy approach balances exploration and exploitation:
- With probability ε: Choose random action (exploration)
- With probability 1-ε: Choose action with highest Q-value (exploitation)
- ε decays over time to reduce exploration

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

## 📈 Performance Metrics

The implementation tracks several key metrics:
- Episode rewards
- Q-value evolution

## 🎯 Results Visualization
This is a simple implementation of the Gridworld Cliff reinforcement learning task.
Adapted from Example 6.6 (page 106) from [Reinforcement Learning: An Introduction by Sutton and Barto](http://incompleteideas.net/book/bookdraft2018jan1.pdf).


###   result

<img src="https://github.com/user-attachments/assets/8a4f3c1f-1fe3-417c-9e74-6c783fdb6343">
<img src="https://github.com/user-attachments/assets/48fcb2d7-b718-45b6-b7fc-9adc97b120d9">



The `visualize_results()` method provides comprehensive visualization including:
- Reward curves for both approaches
- Q-value heatmaps

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## ✨ Acknowledgments
- Based on OpenAI Gym's Cliff Walking environment
- Inspired by Sutton and Barto's Reinforcement Learning textbook
- Implementation structure influenced by modern RL practices













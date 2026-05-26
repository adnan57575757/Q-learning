# Q-Learning 📘

Q-Learning is a **model-free reinforcement learning algorithm** that helps an agent learn the optimal action-selection policy in an environment. It estimates the expected cumulative reward of taking an action in a state and following the best policy thereafter.

---

## 🔑 Key Concepts
- **State (s):** Current situation of the agent.
- **Action (a):** Choice available to the agent.
- **Reward (r):** Feedback received after taking an action.
- **Q-value (Q(s,a)):** Expected future reward of taking action `a` in state `s`.
- **Policy (π):** Strategy that defines which action to take in each state.

---

## ⚙️ Algorithm
1. Initialize Q-table with zeros.
2. For each episode:
   - Start from an initial state.
   - Choose an action using ε-greedy (explore vs exploit).
   - Perform the action, observe reward and next state.
   - Update Q-value:

   

\[
   Q(s,a) \leftarrow Q(s,a) + \alpha \Big[ r + \gamma \max_{a'} Q(s',a') - Q(s,a) \Big]
   \]



   where:
   - \(\alpha\) = learning rate  
   - \(\gamma\) = discount factor  
   - \(s'\) = next state  
   - \(a'\) = next action  

3. Repeat until convergence.

---

## 🧩 Example (Python)
```python
import numpy as np
import gym

env = gym.make("CliffWalking-v0")
Q = np.zeros((env.observation_space.n, env.action_space.n))

alpha = 0.1   # learning rate
gamma = 0.99  # discount factor
epsilon = 0.1 # exploration rate
episodes = 500

for episode in range(episodes):
    state = env.reset()[0]
    done = False
    
    while not done:
        if np.random.rand() < epsilon:
            action = env.action_space.sample()  # explore
        else:
            action = np.argmax(Q[state])        # exploit
        
        next_state, reward, done, _, _ = env.step(action)
        
        Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
        state = next_state

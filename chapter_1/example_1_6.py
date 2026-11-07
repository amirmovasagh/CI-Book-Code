# Example 1.6: Reinforcement learning with Q-Learning in a simple corridor
import numpy as np
 
n_states, n_actions = 6, 2     # states 0..5 ; action 0 = left, 1 = right
goal = 5                       # reaching state 5 is rewarded
Q = np.zeros((n_states, n_actions))
alpha, gamma, epsilon = 0.1, 0.9, 0.2
rng = np.random.default_rng(0)
 
for episode in range(500):
    s = 0
    while s != goal:
        # ε-greedy policy: explore sometimes, exploit most of the time
        if rng.random() < epsilon:
            a = int(rng.integers(n_actions))      # explore
        else:
            a = int(np.argmax(Q[s]))              # exploit
        s_next = max(0, s - 1) if a == 0 else min(goal, s + 1)
        r = 1.0 if s_next == goal else -0.01   # reward
        Q[s, a] += alpha * (r + gamma * Q[s_next].max() - Q[s, a])
        s = s_next
 
np.set_printoptions(precision=3, suppress=True)
print("Q-table (rows = states, cols = [left, right]):\n", Q)
print("Learned policy:", ["L" if np.argmax(q) == 0 else "R" for q in Q[:goal]])

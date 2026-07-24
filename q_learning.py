import random
from collections import defaultdict


class QLearningAgent:
    """
    Tabular Q-learning agent.

    Keeps one Q-value for every (state, action) pair in a table.
    Works well for small mazes where the number of states is manageable.
    """

    def __init__(self, n_actions=4, alpha=0.2, gamma=0.95,
                 epsilon=1.0, epsilon_decay=0.99, epsilon_min=0.05):
        self.n_actions = n_actions
        self.alpha = alpha                  # learning rate
        self.gamma = gamma                  # discount factor
        self.epsilon = epsilon              # current exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Q[state] is a list of one value per action.
        # defaultdict means unseen states start at all zeros automatically.
        self.Q = defaultdict(lambda: [0.0] * n_actions)

    def choose_action(self, state):
        """Epsilon-greedy: mostly pick the best known action, sometimes explore."""
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        return self._best_action(state)

    def _best_action(self, state):
        """Index of the highest Q-value, ties broken randomly."""
        values = self.Q[state]
        best = max(values)
        candidates = [i for i, v in enumerate(values) if v == best]
        return random.choice(candidates)

def update(self, state, action, reward, next_state, done):
    """
    Q-learning update:
        Q(s, a) <- Q(s, a) + alpha * (target - Q(s, a))
    where target = reward + gamma * max_a' Q(s', a')

    If the episode truly ended at the goal there is no future,
    so the target is just the reward.
    """
    current = self.Q[state][action]

    if done:
        target = reward
    else:
        target = reward + self.gamma * max(self.Q[next_state])

    self.Q[state][action] = current + self.alpha * (target - current)

def decay_epsilon(self):
    self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
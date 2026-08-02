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

    def train(self, env, episodes=1000, on_step=None):
        """
        Run many episodes, learning as it goes. Returns per-episode logs.

        on_step, if given, gets called after every environment step as
        on_step(episode, env) - lets a caller (e.g. a CLI display) watch
        training happen without this method needing to know anything
        about rendering.
        """
        reward_history = []
        step_history = []

        for episode in range(episodes):
            state = env.reset()
            total_reward = 0.0

            while True:
                action = self.choose_action(state)
                next_state, reward, done, truncated = env.step(action)

                self.update(state, action, reward, next_state, done)

                state = next_state
                total_reward += reward

                if on_step is not None:
                    on_step(episode, env)

                if done or truncated:
                    break

            self.decay_epsilon()
            reward_history.append(total_reward)
            step_history.append(env.steps)


        return reward_history, step_history


    def greedy_path(self, env, max_length=1000):
        """
        Follow the learned policy with no exploration.
        Returns (path, length) or (None, None) if it fails to reach the goal.
        """
        state = env.reset()
        path = [state]

        for _ in range(max_length):
            action = self._best_action(state)
            state, reward, done, truncated = env.step(action)
            path.append(state)
            if done:
                return path, len(path) - 1
            if truncated:
                break

        return None, None
import random
from collections import deque
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class QNetwork(nn.Module):
    """
    Small feedforward network that approximates Q-values.

    Input: a state, represented as vector.
    Output: one Q-value for each action.
    """

    def __init__(self, input_size, n_actions, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    """
    Stores past transitions so the network can train on a random batch
    of past experience instead of only the most recent step.
    
    Training on the single most recent transition would mean consecutive
    training examples are highly correlated (each state is one move away
    from the last), which makes gradient descent unstable. Sampling a
    random batch breaks that correlation.
    """
    
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    DQN agent: a QNetwork trained with a replay buffer and a target network.

    encode_fn: a function state -> list[float], turning a maze position
    into whatever vector the network actually sees. Swappable on purpose —
    this is exactly the choice the state-representation experiment tests.
    """

    def __init__(self, input_size, encode_fn, n_actions=4,
             lr=5e-3, gamma=0.95,
             epsilon=1.0, epsilon_decay=0.99, epsilon_min=0.05,
             buffer_capacity=10000, batch_size=64,
             target_update_freq=100,
             reward_clip=1.0, lr_decay=0.999):
        self.encode_fn = encode_fn
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.reward_clip = reward_clip

        self.policy_net = QNetwork(input_size, n_actions)
        self.target_net = copy.deepcopy(self.policy_net)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=lr_decay)
        self.buffer = ReplayBuffer(capacity=buffer_capacity)
        self.train_steps = 0

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            vec = torch.tensor(self.encode_fn(state), dtype=torch.float32)
            q_values = self.policy_net(vec)
            return int(torch.argmax(q_values).item())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return  # not enough experience yet

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        states = torch.tensor([self.encode_fn(s) for s in states], dtype=torch.float32)
        next_states = torch.tensor([self.encode_fn(s) for s in next_states], dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        q_values = self.policy_net(states)
        q_selected = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = self.target_net(next_states)
            max_next_q = next_q.max(dim=1).values
            targets = rewards + self.gamma * max_next_q * (1.0 - dones)

        loss = F.mse_loss(q_selected, targets)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.train_steps += 1
        if self.train_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def train(self, env, episodes=500, on_step=None):
        """
        on_step, if given, gets called after every environment step as
        on_step(episode, env) - lets a caller (e.g. a CLI display) watch
        training happen without this method needing to know anything
        about rendering.
        """
        reward_history = []
        step_history = []

        for ep in range(episodes):
            state = env.reset()
            total_reward = 0.0

            while True:
                action = self.choose_action(state)
                next_state, reward, done, truncated = env.step(action)

                clipped_reward = max(-self.reward_clip, min(self.reward_clip, reward))
                self.buffer.push(state, action, clipped_reward, next_state, done)
                self.train_step()

                state = next_state
                total_reward += reward   # log the TRUE reward, not the clipped one

                if on_step is not None:
                    on_step(ep, env)

                if done or truncated:
                    break

            self.decay_epsilon()
            self.scheduler.step()
            reward_history.append(total_reward)
            step_history.append(env.steps)

        return reward_history, step_history

    def greedy_path(self, env, max_len=1000):
        state = env.reset()
        path = [state]
        for _ in range(max_len):
            with torch.no_grad():
                vec = torch.tensor(self.encode_fn(state), dtype=torch.float32)
                action = int(torch.argmax(self.policy_net(vec)).item())
            state, reward, done, truncated = env.step(action)
            path.append(state)
            if done:
                return path, len(path) - 1
            if truncated:
                break
        return None, None


if __name__ == "__main__":
    import random
    import torch
    from maze_env import load_maze
    from utils import astar, plot_steps

    random.seed(3)
    torch.manual_seed(3)

    env = load_maze("mazes/maze_10x10_.txt", max_steps=150)
    print("free cells:", len(env.free_cells()))

    _, optimal = astar(env)
    print("A* optimal:", optimal)

    def coord_encode(state):
        r, c = state
        return [r / (env.rows - 1), c / (env.cols - 1)]

    agent = DQNAgent(input_size=2, encode_fn=coord_encode, epsilon_decay=0.99)
    rewards, steps = agent.train(env, episodes=400)

    print("first 5 episodes:", steps[:5])
    print("last 15 episodes:", steps[-15:])

    path, length = agent.greedy_path(env)
    print("greedy path length:", length)

    plot_steps(steps, optimal=optimal,
               title="DQN with (row, col) coordinate input",
               filename="results/dqn_coord_encoding.png")
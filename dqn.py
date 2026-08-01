import random
from collections import deque

import torch
import torch.nn as nn


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

if __name__ == "__main__":
    input_size = 10  # placeholder — real size depends on state representation

    net = QNetwork(input_size=input_size, n_actions=4)

    dummy_state = torch.zeros(input_size)
    output = net(dummy_state)
    print("single state output:", output)
    print("output shape:", output.shape)

    batch = torch.zeros((32, input_size))
    batch_output = net(batch)
    print("batch output shape:", batch_output.shape)

    buf = ReplayBuffer(capacity=100)
    for i in range(50):
        buf.push([0.0] * input_size, i % 4, -0.01, [0.0] * input_size, False)
    print("buffer length:", len(buf))

    states, actions, rewards, next_states, dones = buf.sample(16)
    print("sampled batch sizes:", len(states), len(actions), len(rewards))
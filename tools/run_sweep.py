"""
Runs the full experiment sweep for the report: trains each agent on its
target mazes and collects steps-to-goal, final greedy path length vs A*
optimal, and saved plot files into results/sweep_summary.txt.

Run: python tools/run_sweep.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import torch

from maze_env import load_maze
from q_learning import QLearningAgent
from dqn import DQNAgent
from utils import astar, compute_max_steps, plot_steps, plot_rewards
from run_custom import coord_encode_fn

SEED = 3

# (maze path, agent type, episodes) - Q-learning is the baseline on the
# smaller mazes per the proposal; DQN entries for larger mazes get added
# once their max_steps budget is confirmed to actually work.
RUNS = [
    ("mazes/maze_7x7_1.txt", "q", 500),
    ("mazes/maze_7x7_2.txt", "q", 500),
    ("mazes/maze_7x7_3.txt", "q", 500),
    ("mazes/maze_8x8.txt", "q", 500),
    ("mazes/maze_8x8_1.txt", "q", 500),
]


def run_one(maze_path, agent_type, episodes):
    random.seed(SEED)
    torch.manual_seed(SEED)

    env = load_maze(maze_path, max_steps=1)
    _, optimal = astar(env)
    if optimal is None:
        print(f"skipping {maze_path}: unsolvable")
        return None

    multiplier = 12 if agent_type == "dqn" else 5
    env.max_steps = compute_max_steps(env, multiplier=multiplier)

    if agent_type == "dqn":
        encode_fn, input_size = coord_encode_fn(env)
        agent = DQNAgent(input_size=input_size, encode_fn=encode_fn, epsilon_decay=0.99)
    else:
        agent = QLearningAgent()

    rewards, steps = agent.train(env, episodes=episodes)
    _, greedy_length = agent.greedy_path(env)

    maze_name = os.path.splitext(os.path.basename(maze_path))[0]
    plot_steps(steps, optimal=optimal,
               title=f"Steps to goal per episode ({maze_name}, {agent_type})",
               filename=f"results/{maze_name}_{agent_type}_steps.png", window=50)
    plot_rewards(rewards,
                 title=f"Total reward per episode ({maze_name}, {agent_type})",
                 filename=f"results/{maze_name}_{agent_type}_rewards.png", window=50)

    matches = greedy_length == optimal
    line = (f"{maze_name:14s} agent={agent_type:4s} max_steps={env.max_steps:5d} "
            f"optimal={optimal:4d} greedy={str(greedy_length):>4s} matches_optimal={'yes' if matches else 'no'}")
    print(line)
    return line


def main():
    os.makedirs("results", exist_ok=True)
    summary_lines = []

    for maze_path, agent_type, episodes in RUNS:
        line = run_one(maze_path, agent_type, episodes)
        if line is not None:
            summary_lines.append(line)

    with open("results/sweep_summary.txt", "w") as f:
        f.write("\n".join(summary_lines) + "\n")

    print("\nsaved results/sweep_summary.txt")


if __name__ == "__main__":
    main()

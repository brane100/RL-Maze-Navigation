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
from run_custom import coord_encode_fn, make_onehot_encode_fn

SEED = 3

# (maze path, agent type, episodes, encoding) - Q-learning is the baseline
# on the smaller mazes per the proposal. encoding is only used for dqn.
#
# maze_15x15.txt / maze_20x20.txt are deliberately NOT here: at 449 free
# cells, plain epsilon-greedy DQN never reaches the goal even once in 300
# full episodes (confirmed, not assumed - see results/sweep_summary.txt
# history / day10 findings). That's a documented limitation, not a bug -
# exploration cost scales with state-space size, not path length, and
# fixing it needs a smarter exploration strategy that's out of scope here.
RUNS = [
    ("mazes/maze_7x7_1.txt", "q", 500, None),
    ("mazes/maze_7x7_2.txt", "q", 500, None),
    ("mazes/maze_7x7_3.txt", "q", 500, None),
    ("mazes/maze_8x8.txt", "q", 500, None),
    ("mazes/maze_8x8_1.txt", "q", 500, None),
    ("mazes/maze_8x8_1.txt", "dqn", 300, "onehot"),
]


def run_one(maze_path, agent_type, episodes, encoding):
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
        if encoding == "onehot":
            encode_fn, input_size = make_onehot_encode_fn(env)
        else:
            encode_fn, input_size = coord_encode_fn(env)
        agent = DQNAgent(input_size=input_size, encode_fn=encode_fn, epsilon_decay=0.99)
    else:
        agent = QLearningAgent()

    rewards, steps = agent.train(env, episodes=episodes)
    _, greedy_length = agent.greedy_path(env)

    maze_name = os.path.splitext(os.path.basename(maze_path))[0]
    tag = agent_type if agent_type == "q" else f"{agent_type}_{encoding}"
    plot_steps(steps, optimal=optimal,
               title=f"Steps to goal per episode ({maze_name}, {tag})",
               filename=f"results/{maze_name}_{tag}_steps.png", window=50)
    plot_rewards(rewards,
                 title=f"Total reward per episode ({maze_name}, {tag})",
                 filename=f"results/{maze_name}_{tag}_rewards.png", window=50)

    matches = greedy_length == optimal
    line = (f"{maze_name:14s} agent={tag:10s} max_steps={env.max_steps:5d} "
            f"optimal={optimal:4d} greedy={str(greedy_length):>4s} matches_optimal={'yes' if matches else 'no'}")
    print(line)
    return line


def main():
    os.makedirs("results", exist_ok=True)
    summary_lines = []

    for maze_path, agent_type, episodes, encoding in RUNS:
        line = run_one(maze_path, agent_type, episodes, encoding)
        if line is not None:
            summary_lines.append(line)

    with open("results/sweep_summary.txt", "w") as f:
        f.write("\n".join(summary_lines) + "\n")

    print("\nsaved results/sweep_summary.txt")


if __name__ == "__main__":
    main()

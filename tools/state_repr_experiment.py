"""
Reproduces the coordinate-vs-one-hot DQN state-encoding comparison on
maze_7x7_1.txt across three seeds. Writes a plain-text summary to
results/state_representation_sweep.txt.

Run: python tools/state_repr_experiment.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import statistics
import torch

from maze_env import load_maze
from dqn import DQNAgent
from run_custom import coord_encode_fn, make_onehot_encode_fn

MAZE = "mazes/maze_7x7_1.txt"
EPISODES = 400
MAX_STEPS = 400  # ~14.3x A* optimal (28) - matches the original run this reproduces
SEEDS = [3, 7, 11]


def run_one(encoding_name, encode_factory, seed):
    random.seed(seed)
    torch.manual_seed(seed)

    env = load_maze(MAZE, max_steps=MAX_STEPS)
    encode_fn, input_size = encode_factory(env)
    agent = DQNAgent(input_size=input_size, encode_fn=encode_fn, epsilon_decay=0.99)

    _, steps = agent.train(env, episodes=EPISODES)
    _, greedy_length = agent.greedy_path(env)

    reached = sum(1 for s in steps if s < MAX_STEPS)
    return {
        "encoding": encoding_name,
        "seed": seed,
        "greedy_length": greedy_length,
        "reached": reached,
        "episodes": EPISODES,
        "mean_last_50": round(statistics.mean(steps[-50:]), 1),
        "last_15": steps[-15:],
    }


def main():
    os.makedirs("results", exist_ok=True)

    encodings = [("coord", coord_encode_fn), ("onehot", make_onehot_encode_fn)]
    all_results = []

    for encoding_name, encode_factory in encodings:
        for seed in SEEDS:
            result = run_one(encoding_name, encode_factory, seed)
            all_results.append(result)
            print(f"{encoding_name:7s} seed={seed:2d} greedy={result['greedy_length']} "
                  f"reached={result['reached']}/{result['episodes']} "
                  f"mean(last50)={result['mean_last_50']}")

    with open("results/state_representation_sweep.txt", "w") as f:
        f.write(f"maze={MAZE} episodes={EPISODES} max_steps={MAX_STEPS} seeds={SEEDS}\n\n")
        for encoding_name, _ in encodings:
            f.write(f"{encoding_name}:\n")
            for r in all_results:
                if r["encoding"] != encoding_name:
                    continue
                f.write(f"  seed={r['seed']} greedy={r['greedy_length']} "
                        f"reached={r['reached']}/{r['episodes']} "
                        f"mean(last50)={r['mean_last_50']} last15={r['last_15']}\n")
            f.write("\n")

    print("\nsaved results/state_representation_sweep.txt")


if __name__ == "__main__":
    main()

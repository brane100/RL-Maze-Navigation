import argparse
import os
import sys

from maze_env import load_maze
from utils import astar, build_path


def main():
    ap = argparse.ArgumentParser(description="Train an RL agent on a custom maze file.")
    ap.add_argument("--maze", required=True)
    ap.add_argument("--agent", choices=["q", "dqn"], default="q")
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--check-only", action="store_true",
                    help="load the maze and report the A* optimum, then exit without training")
    args = ap.parse_args()

    args.checknot os.path.exists(args.maze):
        print(f"Maze file not found: {args.maze}")
        sys.exit(1)

    env = load_maze(args.maze)
    print(f"Maze: {args.maze}")


if __name__ == "__main__":
    main()
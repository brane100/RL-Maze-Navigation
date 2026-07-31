import argparse
import os
import sys
import time

from maze_env import load_maze
from q_learning import QLearningAgent
from utils import astar, plot_steps, plot_rewards


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def animate_path(env, path, delay):
    """Replay a sequence of (row, col) states in the terminal, one frame per move."""
    for step, state in enumerate(path):
        clear_screen()
        env.agent_pos = state
        print(f"step {step}/{len(path) - 1}")
        env.render()
        time.sleep(delay)


def main():
    ap = argparse.ArgumentParser(description="Train an RL agent on a custom maze file.")
    ap.add_argument("--maze", required=True)
    ap.add_argument("--agent", choices=["q", "dqn"], default="q")
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--check-only", action="store_true",
                    help="load the maze and report the A* optimum, then exit without training")
    ap.add_argument("--animate", action="store_true",
                    help="replay the trained agent's greedy path in the terminal")
    ap.add_argument("--delay", type=float, default=0.2,
                    help="seconds between animation frames (used with --animate)")
    args = ap.parse_args()

    if not os.path.exists(args.maze):
        print(f"Maze file not found: {args.maze}")
        sys.exit(1)

    if args.agent == "dqn":
        print("--agent dqn is not implemented yet")
        sys.exit(1)

    os.makedirs("results", exist_ok=True)

    try:
        env = load_maze(args.maze, max_steps=args.max_steps)
    except ValueError as e:
        print(f"Failed to load maze: {e}")
        sys.exit(1)

    print(f"Maze: {args.maze}")

    _, optimal = astar(env)
    print("A* optimal:", optimal)

    if args.check_only:
        if optimal is None:
            print("UNSOLVABLE")
            sys.exit(1)
        return

    agent = QLearningAgent()
    rewards, steps = agent.train(env, episodes=args.episodes)

    final_path, final_length = agent.greedy_path(env)

    print("first 5 episodes (steps):", steps[:5])
    print("last 5 episodes (steps):", steps[-5:])
    print("final epsilon:", round(agent.epsilon, 4))
    print("final greedy path length:", final_length)
    print("matches optimal?", final_length == optimal)

    if args.animate:
        if final_path is None:
            print("agent never reached the goal, nothing to animate")
        else:
            animate_path(env, final_path, args.delay)

    maze_name = os.path.splitext(os.path.basename(args.maze))[0]
    plot_steps(steps, optimal=optimal,
               title=f"Steps to goal per episode ({maze_name})",
               filename=f"results/{maze_name}_steps.png", window=50)
    plot_rewards(rewards,
                 title=f"Total reward per episode ({maze_name})",
                 filename=f"results/{maze_name}_rewards.png", window=50)


if __name__ == "__main__":
    main()
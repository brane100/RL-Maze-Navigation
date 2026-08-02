import argparse
import os
import sys
import time

from maze_env import load_maze
from q_learning import QLearningAgent
from dqn import DQNAgent
from utils import astar, plot_steps, plot_rewards


# ANSI color codes. Nothing fancy, just enough to tell the pieces apart.
RESET = "\033[0m"
AGENT_COLOR = "\033[1;91m"   # bright red
GOAL_COLOR = "\033[1;92m"    # bright green
PATH_COLOR = "\033[34m"      # dark blue, cells the agent already stepped on


def enable_ansi():
    """
    cmd.exe on Windows ignores ANSI escape codes unless something pokes it
    first. Calling os.system("") with an empty string is a weird trick but
    it's the one that actually works without pulling in colorama.
    """
    if os.name == "nt":
        os.system("")


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def move_cursor_home():
    # jumps the cursor back to the top-left WITHOUT clearing the screen,
    # so the next frame gets typed over the last one instead of the
    # terminal wiping + repainting (that's what was causing the flashing)
    sys.stdout.write("\033[H")


def draw_frame(lines):
    move_cursor_home()
    for line in lines:
        # \033[K wipes any leftover characters from the previous, possibly
        # longer, line so old text doesn't stick around at the end
        sys.stdout.write(line + "\033[K\n")
    sys.stdout.flush()


def render_maze(env, visited):
    """Build the maze as a list of printable, colored strings."""
    lines = []
    for r in range(env.rows):
        row_chars = []
        for c in range(env.cols):
            here = (r, c)
            if here == env.agent_pos:
                row_chars.append(AGENT_COLOR + "A" + RESET)
            elif here == env.goal:
                row_chars.append(GOAL_COLOR + "G" + RESET)
            elif here == env.start:
                # keep the start cell marked even after the agent has
                # walked away from it and the trail has covered it
                row_chars.append("S")
            elif here in visited:
                row_chars.append(PATH_COLOR + "*" + RESET)
            else:
                row_chars.append(env.grid[r][c])
        lines.append(" ".join(row_chars))
    return lines


def animate_path(env, path, delay):
    """Replay a sequence of (row, col) states in the terminal, one frame per move."""
    clear_screen()
    visited = set()
    for step, state in enumerate(path):
        env.agent_pos = state
        visited.add(state)

        header = [
            f"step {step}/{len(path) - 1}",
            "(red = agent, green = goal, dark blue = path so far, S = start)",
            "",
        ]
        draw_frame(header + render_maze(env, visited))
        time.sleep(delay)


class TrainingDisplay:
    """
    Draws the maze during training so you can actually watch the agent
    bump around and (eventually) find the goal, instead of just staring
    at a wall of numbers at the end.

    Made this a class instead of a loose function because I needed to
    keep track of the delay + whether the screen's been cleared yet
    without passing that around to a function every time.
    """

    def __init__(self, env, delay):
        self.env = env
        self.delay = delay
        self.screen_cleared = False

    def draw(self, episode, total_episodes, step_count, visited):
        if not self.screen_cleared:
            clear_screen()
            self.screen_cleared = True

        header = [
            f"episode {episode + 1}/{total_episodes}   step {step_count}",
            "(red = agent, green = goal, dark blue = path so far, S = start)",
            "",
        ]
        draw_frame(header + render_maze(self.env, visited))
        time.sleep(self.delay)


def train_with_display(env, agent, episodes, show_every, delay):
    """
    Runs agent.train() and watches it happen, whichever agent it is.
    Both QLearningAgent and DQNAgent take an on_step(episode, env) hook now,
    called after every environment step - so this doesn't need its own copy
    of the episode loop, it just tracks the visited trail and asks the
    display to draw every show_every-th episode.
    """
    display = TrainingDisplay(env, delay)
    visited = set()
    state = {"episode": -1}

    def on_step(episode, env):
        if episode != state["episode"]:
            state["episode"] = episode
            visited.clear()

        visited.add(env.agent_pos)

        if episode % show_every == 0 or episode == episodes - 1:
            display.draw(episode, episodes, env.steps, visited)

    return agent.train(env, episodes=episodes, on_step=on_step)


def coord_encode_fn(env):
    """(row, col) -> a 0..1 scaled vector, the input DQNAgent's network sees."""
    def encode(state):
        r, c = state
        return [r / (env.rows - 1), c / (env.cols - 1)]
    return encode


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
    ap.add_argument("--show-training", action="store_true",
                    help="watch the agent move during training, not just the final path")
    ap.add_argument("--show-every", type=int, default=50,
                    help="only draw every Nth episode with --show-training, otherwise it's too slow")
    args = ap.parse_args()

    enable_ansi()

    if not os.path.exists(args.maze):
        print(f"Maze file not found: {args.maze}")
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

    if args.agent == "dqn":
        agent = DQNAgent(input_size=2, encode_fn=coord_encode_fn(env), epsilon_decay=0.99)
    else:
        agent = QLearningAgent()

    if args.show_training:
        rewards, steps = train_with_display(env, agent, args.episodes, args.show_every, args.delay)
    else:
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
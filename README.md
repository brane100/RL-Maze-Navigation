# Maze Navigation Using Reinforcement Learning

## Overview

A maze navigation agent that reaches a certain goal within the maze (maze exit or a desired cell). It can navigate in four directions and cannot move through walls. A small amount of reward is given for each step and reaching the goal pays out, meaning the agent is encouraged to find the shortest paths rather than any path.

## Learning Without Prior Knowledge

The agent has no previous knowledge of the maze. It learns from the transitions it experiences—each one captured as a tuple of (state, action, reward, next_state).

**State:** The agent's current observation, which includes its position and the goal position within the maze grid.

**Action:** One of the four possible movements: up, down, left, or right.

**Reward:** A scalar value received after taking an action. Reaching the goal yields a positive reward. Each step incurs a cost of −0.01. Colliding with walls carries a penalty of −0.1.

**Next State:** The resulting state after the action is executed, representing the agent's updated position and observation of the environment.

## Validation Method

A shortest-path solver is used at the beginning to verify the maze is solvable and at the end to check how close the learned route came to optimal. Importantly, it is never used during training.

## Setup

```
pip install -r requirements.txt
```

Requires `numpy`, `matplotlib`, and `torch`.

## Files

| File | What it does |
|---|---|
| `maze_env.py` | The environment: `MazeEnv` (state, rewards, `step`/`reset`) and `load_maze()` for reading a maze from a text file. |
| `q_learning.py` | Tabular Q-learning agent. |
| `dqn.py` | DQN agent: Q-network, replay buffer, target network, reward/gradient clipping, LR decay. |
| `utils.py` | A* baseline solver, `compute_max_steps` (scales the step budget from A*'s optimal length instead of a fixed guess), and plotting helpers. |
| `run_custom.py` | Main CLI — trains either agent on any maze file and can display the run live in the terminal. |
| `generate_maze.py` | Generates a random solvable maze (randomized recursive backtracking) and writes it as a text file. |
| `svg_to_grid.py` | Converts an SVG maze (e.g. from mazegenerator.net) into the same text-grid format. |
| `tools/run_sweep.py` | Reproduces the full experiment sweep across mazes and writes `results/sweep_summary.txt`. |
| `test_utils.py` | Unit tests for `utils.py`. |

## Usage

Train Q-learning on a maze:

```
python run_custom.py --maze mazes/maze_7x7_1.txt --agent q --episodes 500
```

Train DQN, choosing the state encoding:

```
python run_custom.py --maze mazes/maze_7x7_1.txt --agent dqn --encoding onehot --episodes 400
```

(DQN only converges reliably on this smallest maze — see "Known limitation" below.)

Watch it learn live in the terminal (colored: red = agent, green = goal, dark blue = visited path, S = start):

```
python run_custom.py --maze mazes/maze_7x7_1.txt --agent q --episodes 500 --show-training --show-every 5 --delay 0.05
```

Replay the trained agent's final path after training:

```
python run_custom.py --maze mazes/maze_7x7_1.txt --agent q --episodes 500 --animate
```

Just check whether a maze is solvable and what A*'s optimal path length is, without training:

```
python run_custom.py --maze mazes/maze_7x7_1.txt --check-only
```

Other flags: `--seed` (reproducible runs), `--max-steps` (override the auto-computed step budget), `--max-steps` omitted lets `compute_max_steps` scale it from A*'s optimal length automatically.

Every run saves `results/{maze}_{steps,rewards}.png` — training curves with a moving average, plotted against the A* optimal line.

## Reproducing the full experiment sweep

```
python tools/run_sweep.py
```

Trains Q-learning on all small mazes and DQN on the designated larger maze, with a fixed seed, and writes a plain-text summary table to `results/sweep_summary.txt` alongside all the plots.

## Results

Q-learning reaches the exact A* optimal path on every small maze tested:

| Maze | Optimal | Agent's path | Matches optimal |
|---|---|---|---|
| maze_7x7_1 | 28 | 28 | yes |
| maze_7x7_2 | 32 | 32 | yes |
| maze_7x7_3 | 48 | 48 | yes |
| maze_8x8 | 32 | 32 | yes |
| maze_8x8_1 | 56 | 56 | yes |

DQN converges reliably only on the smallest maze tested. On `maze_7x7_1.txt`
(97 free cells) it reaches the optimal 28-step path (see
`results/day10_findings.md`). On `maze_8x8_1.txt` (127 free cells, only 30%
more) it does not: `results/sweep_summary.txt` shows `greedy=None
matches_optimal=no` after 300 episodes, even with the more reliable one-hot
encoding. This confirms the effect crosses over somewhere between 97 and 127
free cells, not at some much larger scale.

**Known limitation:** DQN with plain epsilon-greedy exploration does not
converge outside the smallest maze tested, and completely fails on
`maze_15x15.txt` / `maze_20x20.txt` (flat at the step cap for the full
300-episode run). This isn't a bug in the implementation — a random walk's
expected time to find the goal scales with the number of free cells (97 on
the one maze that works, 127 on the one that doesn't, 449 on the mazes that
fail outright), not the optimal path length. Tabular Q-learning avoids this
because its zero-initialized table makes unvisited cells always look at
least as good as visited ones, biasing exploration toward the frontier for
free; a shared-weight network has no equivalent, so it degrades toward an
undirected random walk as the state space grows. Fixing this would need a
smarter exploration strategy (e.g. reward shaping or count-based
exploration), which is outside this project's scope — see
`results/day10_findings.md` for the full analysis.
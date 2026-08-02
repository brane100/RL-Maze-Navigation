# State representation: coordinate vs. one-hot encoding for DQN

## Setup

`maze_7x7_1.txt` (7x7 maze, 97 free cells, A* optimal = 28), `max_steps=400` (~14.3x
optimal), 400 training episodes, seeds 3/7/11. Reproduce with:

```
python tools/state_repr_experiment.py
```

## Results

| seed | encoding | reached goal / 400 episodes | mean steps (last 50) | greedy path length |
|---|---|---|---|---|
| 3 | coord | 342 | 34.0 | 28 |
| 3 | onehot | 388 | 31.3 | 28 |
| 7 | coord | **16** | **400.0** | **None (never converged)** |
| 7 | onehot | 334 | 37.9 | 28 |
| 11 | coord | 344 | 29.7 | 28 |
| 11 | onehot | 364 | 30.2 | 28 |

## Finding

Across three seeds on `maze_7x7_1.txt`, one-hot state encoding converged to the
optimal 28-step path in all three runs. Coordinate encoding converged in two of
three. On seed 7 it never converged at all: it sat at the 400-step cap for the
entire final 15 episodes, and the greedy policy failed to reach the goal
(`greedy_path` returned `None`).

Where coordinate encoding did converge, seeds 3 and 11, its final-50-episode
average was comparable to one-hot's. On seed 11 it was even marginally better,
29.7 vs 30.2. So when both converge, episode-by-episode performance is roughly
a wash. The real difference is reliability. Coordinate encoding has a failure
mode one-hot didn't hit in this sample.

## Why

Coordinate encoding represents a state as `(row / (rows-1), col / (cols-1))`,
two numbers. This bakes in an assumption: cells with similar coordinates
should have similar Q-values. That assumption breaks across a wall. Two cells
one grid-step apart can require a long detour to get between, but the network
sees them as nearly identical inputs. One-hot encoding gives every free cell
its own dedicated input dimension, so it carries no such assumption.

This connects to a separate finding from the same project. Tabular Q-learning
converges reliably on every maze tested (see `results/sweep_summary.txt`),
while DQN with coordinate encoding does not. The Q-table initializes to zero,
and every pre-goal reward is negative, so unvisited state-action pairs always
look at least as good as visited ones. That's an implicit optimism that biases
exploration toward the frontier. A shared-weight network has no equivalent
per-state memory, so it degrades toward something closer to a random walk.
Coordinate encoding's cross-wall generalization error makes that degradation
worse. One-hot removes that extra failure mode, though it doesn't fix the
underlying exploration weakness, which is also why plain epsilon-greedy DQN
fails outright on the larger mazes regardless of encoding
(`maze_15x15.txt`, 449 free cells; see the README's "Known limitation" section).

## Honest limitations

- Three seeds is a small sample. A 1-in-3 failure rate for coordinate
  encoding could be sampling noise on its own. But a complete failure to
  reach the goal even once in the last 15 episodes of training is a stronger
  signal than a small mean difference would be.
- This was only tested on one maze size. Whether the gap holds, widens, or
  disappears at other sizes is untested.
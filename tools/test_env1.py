# test_env1.py (or a new tests/test_maze_env.py)
import random
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from maze_env import MazeEnv

def test_goal_reached():
    maze = ["S.G"]
    env = MazeEnv(maze, max_steps=10)
    env.reset()
    pos, reward, done, truncated = env.step(3)  # right
    pos, reward, done, truncated = env.step(3)  # right, onto G
    assert pos == env.goal
    assert done is True
    assert truncated is False
    assert reward > 0  # includes REWARD_GOAL
    print("goal case OK:", pos, reward, done, truncated)

def test_timeout_without_goal():
    maze = ["S....G"]
    env = MazeEnv(maze, max_steps=2)  # too few steps to ever reach G
    env.reset()
    pos, reward, done, truncated = env.step(3)
    pos, reward, done, truncated = env.step(3)
    assert pos != env.goal
    assert done is False
    assert truncated is True
    print("timeout case OK:", pos, reward, done, truncated)

def test_wall_bump():
    maze = [
        "S#",
        ".G",
    ]
    env = MazeEnv(maze, max_steps=10)
    env.reset()
    pos, reward, done, truncated = env.step(3)  # right, into wall
    assert pos == env.start  # didn't move
    assert reward == env.REWARD_STEP + env.REWARD_WALL
    assert done is False
    assert truncated is False
    print("wall bump OK:", pos, reward, done, truncated)

if __name__ == "__main__":
    test_goal_reached()
    test_timeout_without_goal()
    test_wall_bump()
    print("all step() tests passed")
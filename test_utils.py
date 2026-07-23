"""
Correctness checks for utils.py: unique_path, astar, and the plotting helpers.
Run with: python test_utils.py
"""
import os
from maze_env import MazeEnv
from utils import astar, unique_path, plot_rewards, plot_steps


# ---------- astar ----------

def test_astar_simple_path():
    maze = ["S...G"]
    env = MazeEnv(maze)
    path, length = astar(env)
    assert length == 4, f"expected 4 moves, got {length}"
    assert path[0] == env.start
    assert path[-1] == env.goal
    print("astar simple path OK:", length, path)


def test_astar_forces_detour():
    maze = [
        "S#G",
        ".#.",
        "...",
    ]
    env = MazeEnv(maze)
    path, length = astar(env)
    manhattan_dist = abs(env.start[0] - env.goal[0]) + abs(env.start[1] - env.goal[1])
    assert length is not None, "should have found a path"
    assert length >= manhattan_dist, "path can't be shorter than straight-line distance"
    print("astar detour OK:", length, "(manhattan lower bound:", manhattan_dist, ")")


def test_astar_unreachable():
    maze = [
        "S#G",
        "###",
    ]
    env = MazeEnv(maze)
    path, length = astar(env)
    assert path is None and length is None, "should report unreachable"
    print("astar unreachable OK: correctly returned (None, None)")


def test_astar_matches_known_maze():
    maze = [
        "S..#.",
        ".#..#",
        ".#.#.",
        "...#.",
        "#...G",
    ]
    env = MazeEnv(maze)
    path, length = astar(env)
    assert path[0] == (0, 0)
    assert path[-1] == (4, 4)
    for (r1, c1), (r2, c2) in zip(path, path[1:]):
        step_dist = abs(r1 - r2) + abs(c1 - c2)
        assert step_dist == 1, f"non-adjacent step in path: {(r1,c1)} -> {(r2,c2)}"
        assert not env.is_wall(r2, c2), f"path walks through a wall at {(r2,c2)}"
    print("astar path validity OK:", length, "moves, all steps adjacent and wall-free")


# ---------- unique_path ----------

def test_unique_path_no_collision():
    os.makedirs("results", exist_ok=True)
    target = "results/collision_test.png"
    if os.path.exists(target):
        os.remove(target)
    if os.path.exists("results/collision_test_1.png"):
        os.remove("results/collision_test_1.png")

    p1 = unique_path(target)
    assert p1 == target, "first call should return the plain filename"

    open(p1, "w").close()
    p2 = unique_path(target)
    assert p2 == "results/collision_test_1.png", f"expected _1 suffix, got {p2}"
    print("unique_path OK:", p1, "->", p2)


# ---------- plotting ----------

def test_plotting_runs_and_produces_real_files():
    import random
    fake_steps = [200 - i * 0.15 + random.randint(-20, 20) for i in range(500)]

    out = plot_steps(fake_steps, optimal=8, filename="results/_test_steps.png")
    assert os.path.exists(out), "plot file was not created"
    assert os.path.getsize(out) > 5000, "plot file suspiciously small, likely blank"
    print("plot_steps smoke test OK:", out, os.path.getsize(out), "bytes")

    out2 = plot_rewards(fake_steps, filename="results/_test_rewards.png")
    assert os.path.exists(out2)
    assert os.path.getsize(out2) > 5000
    print("plot_rewards smoke test OK:", out2, os.path.getsize(out2), "bytes")


if __name__ == "__main__":
    test_astar_simple_path()
    test_astar_forces_detour()
    test_astar_unreachable()
    test_astar_matches_known_maze()
    test_unique_path_no_collision()
    test_plotting_runs_and_produces_real_files()
    print("\nall utils.py tests passed")
"""
generate_maze.py — generate random, solvable mazes and save them as text
files compatible with run_custom.py's load_maze().

Algorithm: randomized recursive backtracking (a.k.a. randomized DFS) —
the standard algorithm for "perfect mazes" (no loops, exactly one path
between any two cells, fully connected by construction).

Usage:
    python generate_maze.py --rows 10 --cols 10 --out .\mazes\mymaze.txt
    python generate_maze.py --rows 15 --cols 25 --seed 7 --out big.txt

TODO markers below mark the parts left for you to implement.
"""

import argparse
import random
from collections import deque


# --------------------------------------------------------------------------
# Step 1: Grid representation
# --------------------------------------------------------------------------
# We generate on a LOGICAL grid of `rows x cols` cells, but carve into a
# PHYSICAL character grid twice the size (+1), because every wall segment
# between two cells needs its own character:
#
#   physical_rows = 2 * rows + 1
#   physical_cols = 2 * cols + 1
#
# Logical cell (r, c) always lives at physical position (2r+1, 2c+1).
# The physical cell directly between two adjacent logical cells is the
# "wall" you knock down when you carve a passage between them.

WALL = "#"
FREE = "."


def make_blank_grid(rows: int, cols: int):
    """Return a physical grid, entirely walls, sized for `rows x cols` cells."""
    phys_rows = 2 * rows + 1
    phys_cols = 2 * cols + 1
    return [[WALL for _ in range(phys_cols)] for _ in range(phys_rows)]


def cell_to_physical(r: int, c: int):
    """Logical cell (r, c) -> physical (row, col) coordinate."""
    return 2 * r + 1, 2 * c + 1


# --------------------------------------------------------------------------
# Step 2: The carving algorithm — YOUR TODO
# --------------------------------------------------------------------------

def get_unvisited_neighbors(r: int, c: int, rows: int, cols: int, visited):
    """
    Return a list of LOGICAL (nr, nc) neighbor coordinates of (r, c) that
    are inside the grid (0 <= nr < rows, 0 <= nc < cols) and not yet in
    `visited`.

    TODO: check all four directions in LOGICAL coordinates — i.e. the
    neighbor "east" of (r, c) is (r, c+1), NOT (r, c+2). The +2 jump only
    applies once you're working with physical coordinates.
    """

    neighbors = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
            neighbors.append((nr, nc))
    return neighbors


def carve_maze(rows: int, cols: int, rnd: random.Random):
    """
    Run randomized recursive backtracking and return the finished physical
    grid (list of lists of '#'/'.').

    TODO, roughly:
      grid = make_blank_grid(rows, cols)
      start = (0, 0)                      # or a random logical cell
      visited = {start}
      # carve the start cell itself into FREE at its physical position
      stack = [start]

      while stack:
          r, c = stack[-1]                # current = top of stack
          neighbors = get_unvisited_neighbors(r, c, rows, cols, visited)
          if neighbors:
              nr, nc = rng.choice(neighbors)
              # 1. mark (nr, nc) visited
              # 2. carve (nr, nc)'s own physical cell to FREE
              # 3. carve the WALL BETWEEN (r,c) and (nr,nc) to FREE —
              #    that wall's physical coord is the midpoint of the two
              #    cells' physical coords: ((pr+pnr)//2, (pc+pnc)//2)
              stack.append((nr, nc))
          else:
              stack.pop()                 # dead end -> backtrack

      return grid
    """
    
    grid = make_blank_grid(rows, cols)
    start = (0, random.randrange(cols))  # start at a random cell in the first row
    visited = {start}

    sr, sc = cell_to_physical(*start)
    grid[sr][sc] = FREE

    stack = [start]

    while stack:
        r, c = stack[-1]
        neighbors = get_unvisited_neighbors(r, c, rows, cols, visited)
        
        if neighbors:
            nr, nc = rnd.choice(neighbors)
            visited.add((nr, nc))

            # Carve the neighbor cell
            pr, pc = cell_to_physical(nr, nc)
            pnr, pnc = cell_to_physical(nr, nc)
            wall_r, wall_c = (pr + pnr) // 2, (pc + pnc) // 2
            
            grid[pnr][pnc] = FREE
            grid[wall_r][wall_c] = FREE

            stack.append((nr, nc))
        else:
             stack.pop()  # dead end -> backtrack

    return grid



# --------------------------------------------------------------------------
# Step 3: Validation (done for you — reused from maze-generation-patterns)
# --------------------------------------------------------------------------

def bfs_path_exists(grid, start, goal):
    """BFS over the physical grid; True if goal is reachable from start."""
    rows, cols = len(grid), len(grid[0])
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        if (r, c) == goal:
            return True
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if (0 <= nr < rows and 0 <= nc < cols
                    and grid[nr][nc] != WALL
                    and (nr, nc) not in seen):
                seen.add((nr, nc))
                q.append((nr, nc))
    return False


def generate_with_retry(rows: int, cols: int, seed: int, max_attempts: int = 5):
    """
    Generate + validate a maze, retrying with a derived seed on failure.
    Recursive backtracking always produces a fully-connected maze, so this
    should never actually need to retry — it's a cheap safety net, in case
    you later change carve_maze() to something that can fail (e.g. braiding,
    room carving, imperfect mazes with loops).
    """
    for attempt in range(1, max_attempts + 1):
        rng = random.Random(seed + attempt)
        grid = carve_maze(rows, cols, rng)
        start = cell_to_physical(0, 0)
        goal = cell_to_physical(rows - 1, cols - 1)
        if bfs_path_exists(grid, start, goal):
            return grid, start, goal
        print(f"attempt {attempt}: generated maze failed validation, retrying...")
    raise RuntimeError(f"Could not generate a valid maze in {max_attempts} attempts")


# --------------------------------------------------------------------------
# Step 4: Render + save in the exact format run_custom.py's load_maze() wants
# --------------------------------------------------------------------------

def to_text(grid, start, goal) -> str:
    chars = [row[:] for row in grid]
    sr, sc = start
    gr, gc = goal
    chars[sr][sc] = "S"
    chars[gr][gc] = "G"
    return "\n".join("".join(row) for row in chars)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=10, help="maze height in cells")
    ap.add_argument("--cols", type=int, default=10, help="maze width in cells")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default="mymaze.txt")
    args = ap.parse_args()

    seed = args.seed if args.seed is not None else random.randrange(10**6)
    grid, start, goal = generate_with_retry(args.rows, args.cols, seed)
    text = to_text(grid, start, goal)

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text + "\n")

    print(f"Wrote {args.rows}x{args.cols}-cell maze (seed={seed}) to {args.out}")
    print(text)


if __name__ == "__main__":
    main()

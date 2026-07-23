import heapq

def astar(env):
    """
    Find the shortest path from start to goal.

    This is A* search - it explores the maze smartly instead of
    checking every possible path. It always tries the move that
    looks closest to the goal first.

    Returns (path, length):
      - path: list of (row, col) cells from start to goal
      - length: number of moves it takes
      - (None, None) if there's no way to reach the goal
    """
    start = env.start
    goal = env.goal

    def distance_guess(cell):
        # How many moves left if there were no walls at all.
        # This can never guess too high, which is exactly why A* works.
        r, c = cell
        gr, gc = goal
        return abs(r - gr) + abs(c - gc)

    # Cells we still need to check, sorted so the most promising is on top.
    # Each entry: (guess_total_cost, moves_so_far, cell)
    to_check = [(distance_guess(start), 0, start)]

    # For each cell, remember which cell we came from - lets us rebuild
    # the path once we hit the goal.
    came_from = {start: None}

    # Cheapest number of moves found so far to reach each cell.
    best_known = {start: 0}

    while to_check:
        _, moves, current = heapq.heappop(to_check)

        if current == goal:
            return build_path(came_from, goal)

        # We might have already found a shorter way here since this
        # entry was added. If so, skip it - it's outdated.
        if moves > best_known.get(current, float('inf')):
            continue

        row, col = current
        for dr, dc in env.ACTIONS.values():
            next_row, next_col = row + dr, col + dc

            if env.is_wall(next_row, next_col):
                continue  # can't walk through walls or off the grid

            next_cell = (next_row, next_col)
            moves_to_next = moves + 1  # one more step to get here

            if moves_to_next < best_known.get(next_cell, float('inf')):
                best_known[next_cell] = moves_to_next
                came_from[next_cell] = current
                priority = moves_to_next + distance_guess(next_cell)
                heapq.heappush(to_check, (priority, moves_to_next, next_cell))

    return None, None  # ran out of cells to check - goal is unreachable


def build_path(came_from, goal):
    # Walk backwards from the goal to the start, then flip it around.
    path = []
    cell = goal
    while cell is not None:
        path.append(cell)
        cell = came_from[cell]
    path.reverse()
    return path, len(path) - 1  # number of cells minus 1 = number of moves

if __name__ == "__main__":
    from maze_env import MazeEnv

    maze = [
        "S..#.",
        ".#..#",
        ".#.#.",
        "...#.",
        "#...G",
    ]
    env = MazeEnv(maze)
    path, length = astar(env)
    print("Optimal length:", length)
    print("Path:", path)
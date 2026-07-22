class MazeEnv:
    WALL = '#'
    FREE = '.'
    START = 'S'
    GOAL = 'G'

    def __init__(self, grid, max_steps=500):
        self.grid = [list(row) for row in grid]
        self.rows = len(self.grid)
        self.cols = len(self.grid[0])
        self.max_steps = max_steps

        self.start = self._find(self.START)
        self.goal = self._find(self.GOAL)
        if self.start is None or self.goal is None:
            raise ValueError("maze must contain exactly one S and one G")

        self.agent_pos = self.start
        self.steps = 0

    def _find(self, char):
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r][c] == char:
                    return (r, c)
        return None

    def is_wall(self, r, c):
        if r < 0 or r >= self.rows or c < 0 or c >= self.cols:
            return True
        return self.grid[r][c] == self.WALL

    def free_cells(self):
        return [(r, c) for r in range(self.rows)
                for c in range(self.cols) if not self.is_wall(r, c)]


def load_maze(path, max_steps=500):
    rows = []
    with open(path) as f:
        for line in f:
            cleaned = ''.join(ch for ch in line if not ch.isspace())
            if cleaned:
                rows.append(cleaned)

    width = len(rows[0])
    for i, row in enumerate(rows):
        if len(row) != width:
            raise ValueError(f"row {i} has length {len(row)}, expected {width}")

    return MazeEnv(rows, max_steps=max_steps)

    def reset(self):
        self.agent_pos = self.start
        self.steps = 0
        return self.agent_pos
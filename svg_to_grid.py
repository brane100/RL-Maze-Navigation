"""
Converts an SVG maze (as downloaded from mazegenerator.net) into the
text-grid format used by this project (# wall, . free, S start, G goal,
space-separated characters, one row per line).

mazegenerator.net renders each wall as its own straight line segment
(confirmed by their own docs: importing the SVG into Blender gives you
"a set of curve objects, which is the number of wall segments"). 
So the approach here is:

1. Parse every <line> and straight-line <path> element out of the SVG.
2. Snap all coordinates onto a grid by inferring the cell size from the
     segment endpoints themselves (rather than assuming a fixed pixel
     size, since that can change between exports).
3. Build the standard (2*rows-1) x (2*cols-1) block grid: odd
     row/col indices are cells, even indices are the wall-or-passage
     between two adjacent cells.
4. Find the two gaps in the outer boundary wall -> these are S and G

USAGE
    python svg_to_grid.py path/to/maze.svg mazes/maze_7x7_1.txt --rows 7 --cols 7

"""

import argparse
import re
import sys
from collections import defaultdict
 
NUM = r"-?\d+(?:\.\d+)?"
 
 
def _parse_lines(svg_text):
    """Return a list of (x1, y1, x2, y2) for every straight segment in the SVG."""
    segments = []
 
    # <line x1=".." y1=".." x2=".." y2=".."/>
    for m in re.finditer(
        rf'<line[^>]*\bx1="({NUM})"[^>]*\by1="({NUM})"[^>]*\bx2="({NUM})"[^>]*\by2="({NUM})"',
        svg_text,
    ):
        x1, y1, x2, y2 = map(float, m.groups())
        segments.append((x1, y1, x2, y2))

    # <path d="M x1 y1 L x2 y2 [L x3 y3 ...]"/>  -- split polylines into segments
    for m in re.finditer(r'<path[^>]*\bd="([^"]+)"', svg_text):
        d = m.group(1)
        pts = re.findall(rf"({NUM})[,\s]+({NUM})", d)
        pts = [(float(x), float(y)) for x, y in pts]
        for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
            segments.append((x1, y1, x2, y2))
 
    return segments

def _infer_cell_size(segments):
    """From the spacing of coordinates that appear."""
    xs = sorted(set(round(c, 2) for x1, y1, x2, y2 in segments for c in (x1, x2)))
    ys = sorted(set(round(c, 2) for x1, y1, x2, y2 in segments for c in (y1, y2)))

    def min_gap(vals):
        gaps = [b - a for a, b in zip(vals, vals[1:]) if b - a >  0.5]
        return min(gaps) if gaps else 1.0

    return min(min_gap(xs), min_gap(ys)), min(xs), min(ys)


def convert(svg_path, rows, cols):
    with open(svg_path, "r", encoding="utf-8") as f:
        svg_text = f.read()

    segments = _parse_lines(svg_text)
    if not segments:
        raise ValueError(
            "No <line> or straight <path> segments found. Paste me the SVG "
            "source and I'll adapt the parser to the actual markup."
        )

    cell, x0, y0 = _infer_cell_size(segments)

    def snap(v, origin):
        return round((v - origin) / cell)

    # Grid: block[r][c], size (2*rows-1) x (2*cols-1). '#' = wall by default
    # for the outer frame; interior between-cell slots start as '.' (open)
    # and get a '#' only where a wall segment actually crosses that slot.
    H, W = 2 * rows - 1, 2 * cols - 1
    block = [["#" if r % 2 == 0 and c % 2 == 0 else "." for c in range(W)] for r in range(H)]
 
    # Every wall segment is either horizontal (blocks vertical movement,
    # i.e. sits between row i and row i+1) or vertical (blocks horizontal
    # movement, between col j and col j+1). A segment spanning exactly one
    # cell-width/height maps to exactly one slot in the block grid.
    for x1, y1, x2, y2 in segments:
        gx1, gy1 = snap(x1, x0), snap(y1, y0)
        gx2, gy2 = snap(x2, x0), snap(y2, y0)
 
        if gy1 == gy2 and gx1 != gx2:  # horizontal segment
            row = gy1  # even index -> between cell-row (row/2 - 1) and (row/2)
            lo, hi = sorted((gx1, gx2))
            for gx in range(lo, hi):
                col = gx * 2 + 1  # this wall sits at an odd column, over a cell column
                if 0 <= row < H and 0 <= col < W:
                    block[row][col] = "#"
        elif gx1 == gx2 and gy1 != gy2:  # vertical segment
            col = gx1 * 2
            lo, hi = sorted((gy1, gy2))
            for gy in range(lo, hi):
                row = gy * 2 + 1
                if 0 <= row < H and 0 <= col < W:
                    block[row][col] = "#"
 
    # Fill remaining interior odd/odd positions (actual cells) as free.
    for r in range(1, H, 2):
        for c in range(1, W, 2):
            block[r][c] = "."
 
    # Find S/G as gaps in the outer boundary wall.
    gaps = []
    for c in range(1, W, 2):
        if block[0][c] == ".":
            gaps.append((0, c))
        if block[H - 1][c] == ".":
            gaps.append((H - 1, c))
    for r in range(1, H, 2):
        if block[r][0] == ".":
            gaps.append((r, 0))
        if block[r][W - 1] == ".":
            gaps.append((r, W - 1))
 
    if len(gaps) != 2:
        print(
            f"WARNING: found {len(gaps)} boundary gaps (expected 2 for S/G). "
            "Open the .txt output and place S and G manually.",
            file=sys.stderr,
        )
        s_cell = g_cell = None
    else:
        def nearest_cell(pos):
            r, c = pos
            r = min(max(r, 1), H - 2)
            c = min(max(c, 1), W - 2)
            if r % 2 == 0:
                r += 1
            if c % 2 == 0:
                c += 1
            return r, c
 
        s_cell, g_cell = nearest_cell(gaps[0]), nearest_cell(gaps[1])
        block[s_cell[0]][s_cell[1]] = "S"
        block[g_cell[0]][g_cell[1]] = "G"
 
    return block
 
 
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("svg_path")
    ap.add_argument("out_path")
    ap.add_argument("--rows", type=int, required=True)
    ap.add_argument("--cols", type=int, required=True)
    args = ap.parse_args()
 
    block = convert(args.svg_path, args.rows, args.cols)
    text = "\n".join(" ".join(row) for row in block)
 
    with open(args.out_path, "w", encoding="utf-8") as f:
        f.write(text + "\n")
 
    print(f"Wrote {args.out_path} ({len(block)}x{len(block[0])} block grid)")
    print(text)
 
 
if __name__ == "__main__":
    main()
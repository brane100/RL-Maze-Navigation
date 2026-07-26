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

def _infer_cell_size(segments)

def convert()

def main()
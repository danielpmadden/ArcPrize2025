# ----------------------------
# REAP - Recusrive Emergent Abstraction Program
# ----------------------------

import json, time, hashlib, argparse, sys
import csv, os
from concurrent.futures import ThreadPoolExecutor as ProcessPoolExecutor, as_completed
from pathlib import Path
from collections import deque
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass

# ----------------------------
# Grid Type Definitions
# ----------------------------
# In ARC, a Grid is just a 2D list of integers (colors 0–9)

Grid = List[List[int]]

@dataclass
class Example:
    input: Grid
    output: Optional[Grid] = None

# ----------------------------
# Grid Encoder Utilities
# ----------------------------
class GridEncoder:
    """Base encoder for converting grids to/from text."""

    @staticmethod
    def to_text(grid: Grid) -> str:
        raise NotImplementedError

    @staticmethod
    def to_grid(text: str) -> Grid:
        raise NotImplementedError


class MinimalGridEncoder(GridEncoder):
    """Encodes grids as plain text digits with no separators."""

    @staticmethod
    def to_text(grid: Grid) -> str:
        return "\n".join("".join(str(x) for x in row) for row in grid)

    @staticmethod
    def to_grid(text: str) -> Grid:
        return [[int(x) for x in line] for line in text.strip().splitlines()]


class GridWithSeparationEncoder(GridEncoder):
    """Encodes grids with a separator (e.g. '|') between cells."""

    def __init__(self, split_symbol: str = "|"):
        self.split_symbol = split_symbol

    def to_text(self, grid: Grid) -> str:
        return "\n".join(self.split_symbol.join(str(x) for x in row) for row in grid)

    def to_grid(self, text: str) -> Grid:
        return [[int(x) for x in line.split(self.split_symbol)] for line in text.strip().splitlines()]


class GridCodeBlockEncoder(GridEncoder):
    """Wraps another encoder and puts the grid inside a code block."""

    def __init__(self, base_encoder: GridEncoder):
        self.encoder = base_encoder

    def to_text(self, grid: Grid) -> str:
        return f"```grid\n{self.encoder.to_text(grid)}\n```"

    def to_grid(self, text: str) -> Grid:
        grid_text = text.split("```grid\n")[1].split("\n```")[0]
        return self.encoder.to_grid(grid_text)


# Default encoder to use across the system
DEFAULT_ENCODER = MinimalGridEncoder()

def test_encoder(encoder: GridEncoder):
    """Quick sanity check: encode → decode → compare"""
    sample = [[1,0,0],[0,1,0],[0,0,1]]
    text = encoder.to_text(sample)
    back = encoder.to_grid(text)
    print("Encoded:\n", text)
    print("Roundtrip OK:", back == sample)

# Example: test_encoder(DEFAULT_ENCODER)
# ----------------------------

# ----------------------------
# Core Type Aliases & Files
# ----------------------------
Color = int
Coord = Tuple[int, int]
Object = Dict[str, Any]
Template = Tuple[Tuple[int, ...], ...]
TrainPair = Any
TestItem = Any

# Files for memory & failures
MEMORY_DB = "memory_db.json"
FAIL_LOG = "missing_ops.jsonl"

# ----------------------------
# Utility Grid Functions
# ----------------------------
def dims(g: Grid) -> Tuple[int, int]:
    return (0, 0) if not g or not g[0] else (len(g), len(g[0]))

def deepcopy_grid(g: Grid) -> Grid:
    return [row[:] for row in g]

def eq_grid(a: Grid, b: Grid) -> bool:
    return a == b

def valid_grid(g: Grid) -> bool:
    if not isinstance(g, list) or not g:
        return True
    if not all(isinstance(row, list) for row in g):
        return False
    if not g[0]:
        return True
    w = len(g[0])
    return all(
        len(row) == w and all(isinstance(v, int) and 0 <= v <= 9 for v in row)
        for row in g
    )

def make_grid(h: int, w: int, fill: int = 0) -> Grid:
    return [[fill] * w for _ in range(h)]

def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(v, hi))

def grid_to_template(g: Grid) -> Template:
    return tuple(tuple(row) for row in g)

# ----------------------------
# DSL Primitives (rotate, flip, map, etc.)
# ----------------------------

def enforce_invariants(fn):
    """Decorator to ensure grids are valid after every op."""
    def wrapper(*args, **kwargs):
        out = fn(*args, **kwargs)
        assert isinstance(out, list), f"{fn.__name__}: Grid must be list"
        if out:
            w = len(out[0])
            for row in out:
                assert len(row) == w, f"{fn.__name__}: inconsistent row width"
                for v in row:
                    assert isinstance(v, int), f"{fn.__name__}: non-int cell"
                    assert 0 <= v <= 9, f"{fn.__name__}: invalid color {v}"
        return out
    return wrapper


@enforce_invariants
def rotate(grid: Grid, angle: int) -> Grid:
    H, W = dims(grid)
    angle %= 360
    if angle == 0: return deepcopy_grid(grid)
    if angle == 90:
        out = make_grid(W, H)
        for r in range(H):
            for c in range(W):
                out[c][H-1-r] = grid[r][c]
        return out
    if angle == 180:
        out = make_grid(H, W)
        for r in range(H):
            for c in range(W):
                out[H-1-r][W-1-c] = grid[r][c]
        return out
    if angle == 270:
        out = make_grid(W, H)
        for r in range(H):
            for c in range(W):
                out[W-1-c][r] = grid[r][c]
        return out
    return deepcopy_grid(grid)


@enforce_invariants
def flip(grid: Grid, axis: str) -> Grid:
    H, W = dims(grid)
    out = make_grid(H, W)
    if axis == 'h':
        for r in range(H):
            for c in range(W):
                out[r][W-1-c] = grid[r][c]
    elif axis == 'v':
        for r in range(H):
            for c in range(W):
                out[H-1-r][c] = grid[r][c]
    else:
        out = deepcopy_grid(grid)
    return out


@enforce_invariants
def pad(grid: Grid, top: int, bottom: int, left: int, right: int, value: int = 0) -> Grid:
    H, W = dims(grid)
    out = make_grid(H+top+bottom, W+left+right, value)
    for r in range(H):
        for c in range(W):
            out[r+top][c+left] = grid[r][c]
    return out


@enforce_invariants
def crop(grid: Grid, r0: int, c0: int, r1: int, c1: int) -> Grid:
    H, W = dims(grid)
    r0, r1 = clamp(r0, 0, H), clamp(r1, 0, H)
    c0, c1 = clamp(c0, 0, W), clamp(c1, 0, W)
    if r1 <= r0 or c1 <= c0: return []
    return [row[c0:c1] for row in grid[r0:r1]]


@enforce_invariants
def overlay(base: Grid, top: Grid, mode: str = "top_nonzero_over") -> Grid:
    Hb, Wb = dims(base)
    Ht, Wt = dims(top)
    if (Hb, Wb) != (Ht, Wt): return deepcopy_grid(base)
    out = deepcopy_grid(base)
    for r in range(Hb):
        for c in range(Wb):
            v = top[r][c]
            if mode == "top_nonzero_over" and v != 0:
                out[r][c] = v
    return out


@enforce_invariants
def map_color(grid: Grid, color_map: Dict[Color, Color]) -> Grid:
    H, W = dims(grid)
    return [[color_map.get(grid[r][c], grid[r][c]) for c in range(W)] for r in range(H)]


@enforce_invariants
def tile_to_target(grid: Grid, target_h: int, target_w: int) -> Grid:
    H, W = dims(grid)
    if H == 0 or W == 0 or target_h <= 0 or target_w <= 0: return []
    out = make_grid(target_h, target_w)
    for r in range(target_h):
        row = grid[r % H]
        for c in range(target_w):
            out[r][c] = row[c % W]
    return out


@enforce_invariants
def repeat_scale(grid: Grid, k: int) -> Grid:
    if k <= 1: return deepcopy_grid(grid)
    H, W = dims(grid)
    out = make_grid(H*k, W*k)
    for r in range(H):
        for c in range(W):
            v = grid[r][c]
            rs, cs = r*k, c*k
            for dr in range(k):
                for dc in range(k):
                    out[rs+dr][cs+dc] = v
    return out


@enforce_invariants
def fill_holes(grid: Grid, fill_color: int = 0) -> Grid:
    H, W = dims(grid)
    out = deepcopy_grid(grid)
    visited = [[False]*W for _ in range(H)]
    dirs = [(1,0),(-1,0),(0,1),(0,-1)]
    for r in range(H):
        for c in range(W):
            if out[r][c] != 0 or visited[r][c]:
                continue
            region = []
            is_border = False
            q = deque([(r, c)])
            visited[r][c] = True
            while q:
                pr, pc = q.popleft()
                region.append((pr, pc))
                if pr in (0, H-1) or pc in (0, W-1):
                    is_border = True
                for dr, dc in dirs:
                    nr, nc = pr+dr, pc+dc
                    if 0 <= nr < H and 0 <= nc < W and not visited[nr][nc] and out[nr][nc] == 0:
                        visited[nr][nc] = True
                        q.append((nr, nc))
            if not is_border:
                for pr, pc in region:
                    out[pr][pc] = fill_color
    return out


@enforce_invariants
def mirror_symmetry(grid: Grid, axis: str) -> Grid:
    H, W = dims(grid)
    out = deepcopy_grid(grid)
    if axis == 'h':
        for r in range(H):
            for c in range(W // 2):
                out[r][W-1-c] = grid[r][c]
    elif axis == 'v':
        for r in range(H // 2):
            for c in range(W):
                out[H-1-r][c] = grid[r][c]
    return out


### NEW DSL OPERATORS ###

import numpy as np

@enforce_invariants
def compress_block(grid: Grid, factor: int) -> Grid:
    h, w = len(grid), len(grid[0])
    out_h, out_w = h // factor, w // factor
    out = np.zeros((out_h, out_w), dtype=int)
    for i in range(out_h):
        for j in range(out_w):
            block = [grid[ii][jj]
                     for ii in range(i*factor, (i+1)*factor)
                     for jj in range(j*factor, (j+1)*factor)]
            vals = [v for v in block if v != 0]
            out[i, j] = max(set(vals), key=vals.count) if vals else 0
    return out.tolist()


@enforce_invariants
def expand_block(grid: Grid, factor: int) -> Grid:
    arr = np.array(grid)
    out = arr.repeat(factor, axis=0).repeat(factor, axis=1)
    return out.tolist()


@enforce_invariants
def repeat_pattern(grid: Grid, stride: int) -> Grid:
    arr = np.array(grid)
    h, w = arr.shape
    out = arr.copy()
    for i in range(h):
        for j in range(w):
            if arr[i, j] != 0:
                for k in range(1, stride):
                    if j + k < w:
                        out[i, j + k] = arr[i, j]
    return out.tolist()


@enforce_invariants
def tile_with_padding(grid: Grid, pad: int, direction: str = "right") -> Grid:
    arr = np.array(grid)
    if direction == "right":
        out = np.pad(arr, ((0, 0), (pad, pad)), constant_values=0)
    elif direction == "down":
        out = np.pad(arr, ((pad, pad), (0, 0)), constant_values=0)
    else:
        out = np.pad(arr, ((pad, pad), (pad, pad)), constant_values=0)
    return out.tolist()


@enforce_invariants
def replace_region(grid: Grid, color: int, bounds: Tuple[int,int,int,int]) -> Grid:
    arr = np.array(grid)
    r1, r2, c1, c2 = bounds
    arr[r1:r2, c1:c2] = color
    return arr.tolist()


@enforce_invariants
def grow_block(grid: Grid, target_color: int) -> Grid:
    arr = np.array(grid)
    h, w = arr.shape
    out = arr.copy()
    for i in range(h):
        for j in range(w):
            if arr[i, j] != 0:
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < h and 0 <= nj < w and out[ni, nj] == 0:
                            out[ni, nj] = target_color
    return out.tolist()

### END NEW DSL OPERATORS ###


# ----------------------------
# New Object-Centric Operators
# ----------------------------

def translate_object(grid: Grid, dr: int, dc: int) -> Grid:
    H, W = dims(grid)
    objs = extract_objects(grid)
    if not objs:
        return deepcopy_grid(grid)
    out = make_grid(H, W)
    for obj in objs:
        r0, c0, _, _ = obj['bbox']
        gh, gw = dims(obj['grid'])
        for rr in range(gh):
            for cc in range(gw):
                v = obj['grid'][rr][cc]
                if v != 0:
                    nr, nc = r0 + rr + dr, c0 + cc + dc
                    if 0 <= nr < H and 0 <= nc < W:
                        out[nr][nc] = v
    return out

def place_in_center(grid: Grid) -> Grid:
    H, W = dims(grid)
    objs = extract_objects(grid)
    if not objs:
        return deepcopy_grid(grid)
    out = make_grid(H, W)
    for obj in objs:
        gh, gw = dims(obj['grid'])
        r0 = max(0, (H - gh) // 2)
        c0 = max(0, (W - gw) // 2)
        for rr in range(gh):
            for cc in range(gw):
                v = obj['grid'][rr][cc]
                if v != 0:
                    out[r0+rr][c0+cc] = v
    return out

def place_in_corner(grid: Grid, which: str = "tl") -> Grid:
    H, W = dims(grid)
    objs = extract_objects(grid)
    if not objs:
        return deepcopy_grid(grid)
    out = make_grid(H, W)
    for obj in objs:
        gh, gw = dims(obj['grid'])
        if which == "tl":
            r0, c0 = 0, 0
        elif which == "tr":
            r0, c0 = 0, W-gw
        elif which == "bl":
            r0, c0 = H-gh, 0
        else:  # "br"
            r0, c0 = H-gh, W-gw
        for rr in range(gh):
            for cc in range(gw):
                v = obj['grid'][rr][cc]
                if v != 0:
                    out[r0+rr][c0+cc] = v
    return out

def remove_color(grid: Grid, color: int) -> Grid:
    H, W = dims(grid)
    return [[0 if grid[r][c] == color else grid[r][c] for c in range(W)] for r in range(H)]

def keep_largest_object(grid: Grid) -> Grid:
    objs = extract_objects(grid)
    if not objs:
        return deepcopy_grid(grid)
    largest = max(objs, key=lambda o: o['size'])
    H, W = dims(grid)
    return draw_objects(H, W, [largest])

def outline_object(grid: Grid) -> Grid:
    H, W = dims(grid)
    objs = extract_objects(grid)
    out = make_grid(H, W)
    dirs = [(1,0),(-1,0),(0,1),(0,-1)]
    for obj in objs:
        for r in range(H):
            for c in range(W):
                if obj['grid'][r][c] != 0:
                    for dr, dc in dirs:
                        nr, nc = r+dr, c+dc
                        if 0 <= nr < H and 0 <= nc < W and obj['grid'][nr][nc] == 0:
                            out[r][c] = obj['grid'][r][c]
    return out

def fill_object(grid: Grid, color: int) -> Grid:
    H, W = dims(grid)
    objs = extract_objects(grid)
    out = make_grid(H, W)
    for obj in objs:
        for r in range(H):
            for c in range(W):
                if obj['grid'][r][c] != 0:
                    out[r][c] = color
    return out

# ----------------------------
# Object Extraction & Transformation
# ----------------------------

from functools import lru_cache

@lru_cache(maxsize=4096)
def _cached_extract_objects(grid_tuple: Tuple[Tuple[int, ...], ...], connectivity: int = 4):
    """Internal cached version of extract_objects using tuple-hashable grids."""
    H, W = len(grid_tuple), len(grid_tuple[0]) if grid_tuple else 0
    grid = [list(row) for row in grid_tuple]
    visited = set()
    objs = []
    dirs = [(1,0),(-1,0),(0,1),(0,-1)] if connectivity == 4 else [
        (1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)
    ]
    for r in range(H):
        for c in range(W):
            if grid[r][c] != 0 and (r,c) not in visited:
                color = grid[r][c]
                q = deque([(r, c)])
                pixels = {(r,c)}
                visited.add((r, c))
                while q:
                    pr, pc = q.popleft()
                    for dr, dc in dirs:
                        nr, nc = pr+dr, pc+dc
                        if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in visited and grid[nr][nc] == color:
                            visited.add((nr, nc))
                            pixels.add((nr, nc))
                            q.append((nr, nc))
                rs = [p[0] for p in pixels]
                cs = [p[1] for p in pixels]
                bbox = (min(rs), min(cs), max(rs)+1, max(cs)+1)
                obj_grid = [[grid[y][x] if (y,x) in pixels else 0 for x in range(W)] for y in range(H)]
                obj_crop = crop(obj_grid, *bbox)
                shape = [[1 if v != 0 else 0 for v in row] for row in obj_crop]
                objs.append({'color': color, 'pixels': pixels, 'bbox': bbox, 'grid': obj_crop,
                             'shape': grid_to_template(shape), 'size': len(pixels)})
    return objs


def extract_objects(grid: Grid, connectivity: int = 4) -> List[Object]:
    """Wrapper around cached object extraction."""
    if not grid:
        return []
    grid_tuple = tuple(tuple(row) for row in grid)
    return _cached_extract_objects(grid_tuple, connectivity)

def draw_objects(H: int, W: int, objects: List[Object], bg: int = 0) -> Grid:
    out = make_grid(H, W, bg)
    for obj in objects:
        r0, c0, _, _ = obj['bbox']
        gh, gw = dims(obj['grid'])
        for rr in range(gh):
            for cc in range(gw):
                val = obj['grid'][rr][cc]
                if val != 0 and 0 <= r0+rr < H and 0 <= c0+cc < W:
                    out[r0+rr][c0+cc] = val
    return out

def learn_object_transformation_maps(task: Any) -> Dict[Template, Dict[str, Any]]:
    mapping = {}
    for pair in task.train:
        in_o = extract_objects(pair.input)
        out_o = extract_objects(pair.output)
        if len(in_o) != len(out_o):
            continue
        for i in in_o:
            cand = [o for o in out_o if o['color'] == i['color']]
            if len(cand) == 1:
                o2 = cand[0]
                shape = i['shape']
                rule = mapping.setdefault(shape, {})
                if i['color'] != o2['color']:
                    rule['color_map'] = {i['color']: o2['color']}
                ishaped = [[1 if v != 0 else 0 for v in row] for row in i['grid']]
                oshaped = [[1 if v != 0 else 0 for v in row] for row in o2['grid']]
                for a in (90,180,270):
                    if eq_grid(rotate(ishaped, a), oshaped):
                        rule['rotate'] = a
                        break
    return mapping

def transform_by_object_template(grid: Grid, transform_map: Dict[Template, Dict[str, Any]]) -> Grid:
    H, W = dims(grid)
    in_o = extract_objects(grid)
    newo = []
    for i in in_o:
        rule = transform_map.get(i['shape'])
        ng = deepcopy_grid(i['grid'])
        if rule:
            if 'color_map' in rule:
                ng = map_color(ng, rule['color_map'])
            if 'rotate' in rule:
                ng = rotate(ng, rule['rotate'])
        newo.append({'color': i['color'], 'pixels': None, 'bbox': i['bbox'], 'grid': ng})
    return draw_objects(H, W, newo)

# ----------------------------
# Color Map Inference & Fill
# ----------------------------
def infer_color_map(grid_in: Grid, grid_out: Grid) -> Optional[Dict[Color, Color]]:
    if dims(grid_in) != dims(grid_out):
        return None
    mapping, mapped_values = {}, set()
    H, W = dims(grid_in)
    for r in range(H):
        for c in range(W):
            ci, co = grid_in[r][c], grid_out[r][c]
            if ci == co:
                continue
            if ci in mapping and mapping[ci] != co:
                return None
            mapping[ci] = co
            mapped_values.add(co)
    if len(mapped_values) < len(mapping):
        return None
    return mapping

def infer_color_maps_from_train(task: Any) -> List[Dict[int, int]]:
    per_pair = []
    for tp in task.train:
        if dims(tp.input) != dims(tp.output):
            return []
        m = infer_color_map(tp.input, tp.output)
        if m is None:
            return []
        per_pair.append(m)
    if not per_pair:
        return []
    merged = {}
    for m in per_pair:
        for k, v in m.items():
            if k in merged and merged[k] != v:
                return []
            merged[k] = v
    return [merged] if merged else []

def fill_holes(grid: Grid, fill_color: int = 0) -> Grid:
    H, W = dims(grid)
    out = deepcopy_grid(grid)
    visited = [[False]*W for _ in range(H)]
    dirs = [(1,0),(-1,0),(0,1),(0,-1)]
    for r in range(H):
        for c in range(W):
            if out[r][c] != 0 or visited[r][c]:
                continue
            region = []
            is_border = False
            q = deque([(r, c)])
            visited[r][c] = True
            while q:
                pr, pc = q.popleft()
                region.append((pr, pc))
                if pr in (0, H-1) or pc in (0, W-1):
                    is_border = True
                for dr, dc in dirs:
                    nr, nc = pr+dr, pc+dc
                    if 0 <= nr < H and 0 <= nc < W and not visited[nr][nc] and out[nr][nc] == 0:
                        visited[nr][nc] = True
                        q.append((nr, nc))
            if not is_border:
                for pr, pc in region:
                    out[pr][pc] = fill_color
    return out

def mirror_symmetry(grid: Grid, axis: str) -> Grid:
    H, W = dims(grid)
    out = deepcopy_grid(grid)
    if axis == 'h':
        for r in range(H):
            for c in range(W // 2):
                out[r][W-1-c] = grid[r][c]
    elif axis == 'v':
        for r in range(H // 2):
            for c in range(W):
                out[H-1-r][c] = grid[r][c]
    return out

# ----------------------------
# New Object-Centric Operators
# ----------------------------

def translate_object(grid: Grid, dr: int, dc: int) -> Grid:
    H, W = dims(grid)
    objs = extract_objects(grid)
    if not objs:
        return deepcopy_grid(grid)
    out = make_grid(H, W)
    for obj in objs:
        r0, c0, _, _ = obj['bbox']
        gh, gw = dims(obj['grid'])
        for rr in range(gh):
            for cc in range(gw):
                v = obj['grid'][rr][cc]
                if v != 0:
                    nr, nc = r0 + rr + dr, c0 + cc + dc
                    if 0 <= nr < H and 0 <= nc < W:
                        out[nr][nc] = v
    return out

def place_in_center(grid: Grid) -> Grid:
    H, W = dims(grid)
    objs = extract_objects(grid)
    if not objs:
        return deepcopy_grid(grid)
    out = make_grid(H, W)
    for obj in objs:
        gh, gw = dims(obj['grid'])
        r0 = max(0, (H - gh) // 2)
        c0 = max(0, (W - gw) // 2)
        for rr in range(gh):
            for cc in range(gw):
                v = obj['grid'][rr][cc]
                if v != 0:
                    out[r0+rr][c0+cc] = v
    return out

def place_in_corner(grid: Grid, which: str = "tl") -> Grid:
    H, W = dims(grid)
    objs = extract_objects(grid)
    if not objs:
        return deepcopy_grid(grid)
    out = make_grid(H, W)
    for obj in objs:
        gh, gw = dims(obj['grid'])
        if which == "tl":
            r0, c0 = 0, 0
        elif which == "tr":
            r0, c0 = 0, W-gw
        elif which == "bl":
            r0, c0 = H-gh, 0
        else:  # "br"
            r0, c0 = H-gh, W-gw
        for rr in range(gh):
            for cc in range(gw):
                v = obj['grid'][rr][cc]
                if v != 0:
                    out[r0+rr][c0+cc] = v
    return out

def remove_color(grid: Grid, color: int) -> Grid:
    H, W = dims(grid)
    return [[0 if grid[r][c] == color else grid[r][c] for c in range(W)] for r in range(H)]

def keep_largest_object(grid: Grid) -> Grid:
    objs = extract_objects(grid)
    if not objs:
        return deepcopy_grid(grid)
    largest = max(objs, key=lambda o: o['size'])
    H, W = dims(grid)
    return draw_objects(H, W, [largest])

def outline_object(grid: Grid) -> Grid:
    H, W = dims(grid)
    objs = extract_objects(grid)
    out = make_grid(H, W)
    dirs = [(1,0),(-1,0),(0,1),(0,-1)]
    for obj in objs:
        for r in range(H):
            for c in range(W):
                if obj['grid'][r][c] != 0:
                    for dr, dc in dirs:
                        nr, nc = r+dr, c+dc
                        if 0 <= nr < H and 0 <= nc < W and obj['grid'][nr][nc] == 0:
                            out[r][c] = obj['grid'][r][c]
    return out

def fill_object(grid: Grid, color: int) -> Grid:
    H, W = dims(grid)
    objs = extract_objects(grid)
    out = make_grid(H, W)
    for obj in objs:
        for r in range(H):
            for c in range(W):
                if obj['grid'][r][c] != 0:
                    out[r][c] = color
    return out


# ----------------------------
# Program & Search Infrastructure
# ----------------------------
class Op:
    def __init__(self, name: str, params: Dict[str, Any]):
        self.name = name
        self.params = params

class Program:
    def __init__(self, ops: List[Op]):
        self.ops = ops

    def apply(self, g: Grid) -> Grid:
        out = deepcopy_grid(g)
        for op in self.ops:
            fn = FUNCTION_REGISTRY.get(op.name)
            if not fn:
                return make_grid(1,1,0)
            try:
                out = fn(out, **op.params)
            except Exception:
                return make_grid(1,1,0)
        return out

    def signature(self) -> Tuple[str, ...]:
        return tuple(op.name for op in self.ops)

    def cost(self) -> float:
        return len(self.ops) + 0.01 * sum(len(str(v)) for op in self.ops for v in op.params.values())

    def __repr__(self):
        return " -> ".join(op.name for op in self.ops)

def translate_all_by_centroid(grid: Grid, target_r: int, target_c: int) -> Grid:
    H, W = dims(grid)
    objs = extract_objects(grid)
    if not objs:
        return deepcopy_grid(grid)
    all_pixels = [p for obj in objs for p in obj['pixels']]
    cy = sum(p[0] for p in all_pixels) // len(all_pixels)
    cx = sum(p[1] for p in all_pixels) // len(all_pixels)
    dy, dx = target_r - cy, target_c - cx
    out = make_grid(H, W)
    for r, c in all_pixels:
        v = grid[r][c]
        nr, nc = r + dy, c + dx
        if 0 <= nr < H and 0 <= nc < W:
            out[nr][nc] = v
    return out

# ----------------------------
# FUNCTION REGISTRY
# ----------------------------
FUNCTION_REGISTRY = {
    "rotate": rotate, "flip": flip, "pad": pad, "crop": crop,
    "overlay": overlay, "map_color": map_color, "transform_by_object_template": transform_by_object_template,
    "translate_all_by_centroid": translate_all_by_centroid,
    "tile_to_target": tile_to_target, "repeat_scale": repeat_scale,
    "fill_holes": fill_holes, "mirror_symmetry": mirror_symmetry,
}

# Add new object-centric ops
FUNCTION_REGISTRY.update({
    "translate_object": translate_object,
    "place_in_center": place_in_center,
    "place_in_corner": place_in_corner,
    "remove_color": remove_color,
    "keep_largest_object": keep_largest_object,
    "outline_object": outline_object,
    "fill_object": fill_object,
})

# ----------------------------
# Search Diagnostics Helpers
# ----------------------------
from dataclasses import dataclass, field

def hamming_like_distance(a: Grid, b: Grid) -> int:
    """Compare grids of same dims; if dims differ, penalize heavily."""
    if dims(a) != dims(b):
        ha, wa = dims(a); hb, wb = dims(b)
        # penalize by total area difference + max area (keeps score positive & comparable)
        return (ha*wa + hb*wb)
    H, W = dims(a)
    d = 0
    for r in range(H):
        for c in range(W):
            if a[r][c] != b[r][c]:
                d += 1
    return d

@dataclass
class SearchStats:
    time_elapsed: float = 0.0
    depth_reached: int = 0
    total_candidates: int = 0
    kept_after_filter: int = 0
    valids_found: int = 0
    survivors_per_depth: list[int] = field(default_factory=list)
    best_firstpair_distance: Optional[int] = None
    best_dims_match: bool = False
    best_objcount_match: bool = False
    crashes_during_apply: int = 0

# ----------------------------
# Search & Enumeration
# ----------------------------
cache_transform_maps: Dict[str, Dict[Template, Dict[str, Any]]] = {}

class SearchConfig:
    max_depth = 7
    beam_size = 256
    time_budget_s = 8.0
    allow_ops: List[str] = []

def program_fits_all(p: Program, train: List[Any]) -> bool:
    return all(eq_grid(p.apply(tp.input), tp.output) for tp in train)

def enumerate_params_for_op(op_name: str, task: Any) -> List[Dict[str, Any]]:
    params = []
    H_in, W_in = dims(task.train[0].input)
    H_out, W_out = dims(task.train[0].output)

    if op_name == "rotate":
        params = [{"angle": a} for a in (90, 180, 270)]

    elif op_name == "flip":
        params = [{"axis": ax} for ax in ("h", "v")]

    elif op_name == "map_color":
        for cmap in infer_color_maps_from_train(task)[:3]:  # 🚀 cap at 3 mappings
            params.append({"color_map": cmap})

    elif op_name == "transform_by_object_template":
        key_hash = hashlib.md5(json.dumps([tp.input for tp in task.train]).encode()).hexdigest()
        tmap = cache_transform_maps.get(key_hash)
        if not tmap:
            cache_transform_maps[key_hash] = learn_object_transformation_maps(task)
            tmap = cache_transform_maps[key_hash]
        if tmap:
            params = [{"transform_map": tmap}]

    elif op_name == "translate_all_by_centroid":
        params = [{"target_r": H_in // 2, "target_c": W_in // 2}]

    elif op_name == "tile_to_target":
        params = [{"target_h": H_out, "target_w": W_out}]

    elif op_name == "repeat_scale":
        ks = {2}
        if H_in and W_in and H_out % H_in == 0 and W_out % W_in == 0 and (H_out // H_in) == (W_out // W_in):
            ks.add(H_out // H_in)
        params = [{"k": k} for k in sorted(ks) if k > 0]

    elif op_name == "pad":
        # 🚀 only single-sided pads (instead of all 16 combos)
        params = [
            {"top": 1, "bottom": 0, "left": 0, "right": 0, "value": 0},
            {"top": 0, "bottom": 1, "left": 0, "right": 0, "value": 0},
            {"top": 0, "bottom": 0, "left": 1, "right": 0, "value": 0},
            {"top": 0, "bottom": 0, "left": 0, "right": 1, "value": 0},
        ]

    elif op_name == "crop":
        H, W = H_in, W_in
        params = []
        if H > 2:
            params.append({"r0": 1, "c0": 0, "r1": H, "c1": W})
            params.append({"r0": 0, "c0": 0, "r1": H - 1, "c1": W})
        if W > 2:
            params.append({"r0": 0, "c0": 1, "r1": H, "c1": W})
            params.append({"r0": 0, "c0": 0, "r1": H, "c1": W - 1})
        if H > 2 and W > 2:
            params.append({"r0": 1, "c0": 1, "r1": H - 1, "c1": W - 1})

    elif op_name == "fill_holes":
        colors = {v for tp in task.train for row in tp.output for v in row}
        trial = [0] + sorted([c for c in colors if c != 0])[:2]  # 🚀 at most 2 colors
        params = [{"fill_color": c} for c in trial]

    elif op_name == "mirror_symmetry":
        params = [{"axis": "h"}, {"axis": "v"}]

    # ----------------------------
    # Object-centric operators
    # ----------------------------
    elif op_name == "translate_object":
        # 🚀 smaller range: only -1, 0, +1 instead of -2..2
        params = [{"dr": dr, "dc": dc} for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]

    elif op_name == "place_in_center":
        params = [{}]

    elif op_name == "place_in_corner":
        params = [{"which": w} for w in ("tl", "tr", "bl", "br")]

    elif op_name == "remove_color":
        colors = {v for tp in task.train for row in tp.input for v in row}
        params = [{"color": c} for c in colors if c != 0]

    elif op_name == "keep_largest_object":
        params = [{}]

    elif op_name == "outline_object":
        params = [{}]

    elif op_name == "fill_object":
        colors = {v for tp in task.train for row in tp.output for v in row if v != 0}
        params = [{"color": c} for c in colors]

    return params


# ----------------------------
# Search & Enumeration
# ----------------------------
class SearchStats:
    def __init__(self):
        self.total_candidates = 0
        self.crashes_during_apply = 0
        self.kept_after_filter = 0
        self.survivors_per_depth: List[int] = []
        self.depth_reached = 0
        self.time_elapsed = 0.0
        self.valids_found = 0
        self.best_firstpair_distance: Optional[int] = None
        self.best_dims_match: Optional[bool] = None


def hamming_like_distance(a: Grid, b: Grid) -> int:
    """Compare two grids by counting differing cells. Returns large penalty if dims mismatch."""
    if dims(a) != dims(b):
        return 9999
    H, W = dims(a)
    d = 0
    for r in range(H):
        for c in range(W):
            if a[r][c] != b[r][c]:
                d += 1
    return d


def bfs_synthesize(task: Any, cfg: SearchConfig) -> Tuple[List[Program], SearchStats]:
    start = time.time()
    stats = SearchStats()
    first_in = task.train[0].input
    first_out = task.train[0].output
    target_dims = dims(first_out)

    def firstpair_score(prog: Program) -> tuple[int, bool, bool]:
        """Return (distance, dims_match, objcount_match). Lower distance is better."""
        out = prog.apply(first_in)
        d = hamming_like_distance(out, first_out)
        dm = (dims(out) == target_dims)
        oc = False
        try:
            eo, et = extract_objects(out), extract_objects(first_out)
            oc = (len(eo) == len(et))
        except Exception:
            pass
        return d, dm, oc

    def fits_first(prog: Program) -> bool:
        """Balanced pruning: must at least match dims or be close in distance."""
        d, dm, _ = firstpair_score(prog)

        # Track best stats
        if stats.best_firstpair_distance is None or d < stats.best_firstpair_distance:
            stats.best_firstpair_distance = d
            stats.best_dims_match = dm

        # 🚀 Dimension pruning
        last_op = prog.ops[-1].name if prog.ops else None
        resize_ops = {
            "compress_block", "expand_block",
            "tile_with_padding", "pad", "crop",
            "tile_to_target", "repeat_scale"
        }
        if not dm and last_op not in resize_ops and d > 3:
            return False

        return dm or d <= 5

    beam: List[Program] = [Program([])]
    valids: List[Program] = []

    if program_fits_all(Program([]), task.train):
        valids.append(Program([]))

    # depth loop
    for depth in range(cfg.max_depth):
        if time.time() - start > cfg.time_budget_s:
            break
        next_beam: List[Program] = []
        survivors_this_depth = 0

        for prog in beam:
            for op_name in cfg.allow_ops:
                for params in enumerate_params_for_op(op_name, task):
                    cand = Program(prog.ops + [Op(op_name, params)])
                    stats.total_candidates += 1
                    try:
                        if not fits_first(cand):
                            continue
                    except Exception:
                        stats.crashes_during_apply += 1
                        continue
                    survivors_this_depth += 1
                    if program_fits_all(cand, task.train):
                        valids.append(cand)
                    next_beam.append(cand)

        # 🚦 Hard cap
        hard_cap = max(cfg.beam_size * 5, cfg.beam_size)
        if len(next_beam) > hard_cap:
            def rank_key(p: Program) -> tuple[float, int]:
                d, *_ = firstpair_score(p)
                return (p.cost(), d)
            next_beam = sorted(next_beam, key=rank_key)[:hard_cap]

        # Beam pruning
        def final_rank_key(p: Program) -> tuple[float, int]:
            d, *_ = firstpair_score(p)
            return (p.cost(), d)

        next_beam.sort(key=final_rank_key)
        stats.kept_after_filter += len(next_beam)
        beam = next_beam[:cfg.beam_size]
        stats.survivors_per_depth.append(len(beam))
        stats.depth_reached = depth + 1

        if time.time() - start > cfg.time_budget_s:
            break

    stats.time_elapsed = time.time() - start
    stats.valids_found = len(valids)
    uniq = {p.signature(): p for p in sorted(valids, key=lambda p: p.cost())}
    return list(uniq.values()), stats



def pick_two(pl: List[Program]) -> Tuple[Optional[Program], Optional[Program]]:
    if not pl:
        return None, None
    return pl[0], (pl[1] if len(pl) > 1 else None)

def classify_task(task: Any) -> Dict[str, Any]:
    f = {}
    f['same_shape'] = all(dims(tp.input) == dims(tp.output) for tp in task.train)
    f['sym_h'] = all(eq_grid(tp.input, flip(tp.input, 'h')) for tp in task.train)
    f['sym_v'] = all(eq_grid(tp.input, flip(tp.input, 'v')) for tp in task.train)
    try:
        in_o = [extract_objects(tp.input) for tp in task.train]
        out_o = [extract_objects(tp.output) for tp in task.train]
        f['obj_count_same'] = all(len(i) == len(o) for i, o in zip(in_o, out_o))
    except Exception:
        f['obj_count_same'] = False
    return f


def micro_tune(output: Grid, target_shape: Tuple[int, int]) -> Grid:
    if dims(output) == target_shape:
        return output
    for fn in (
        lambda g: rotate(g, 90),
        lambda g: rotate(g, 180),
        lambda g: rotate(g, 270),
        lambda g: flip(g, 'h'),
        lambda g: flip(g, 'v'),
    ):
        cand = fn(output)
        if dims(cand) == target_shape:
            return cand
    return output


def solve_task(task: Any, time_budget_s: float = 10.0) -> Tuple[List[Dict[str, Grid]], SearchStats]:
    feats = classify_task(task)

    # Base ops always available
    ops = ["rotate", "flip", "map_color", "fill_holes", "mirror_symmetry"]

    # Object-based heuristics
    if feats.get("obj_count_same"):
        ops += [
            "transform_by_object_template",
            "translate_object",
            "place_in_center",
            "place_in_corner",
            "keep_largest_object",
            "outline_object",
            "fill_object",
        ]

    # Shape differences → compression / expansion / tiling
    if not feats["same_shape"]:
        ops += [
            "pad", "crop", "tile_to_target", "repeat_scale",
            "compress_block", "expand_block", "tile_with_padding"
        ]
    else:
        ops += ["translate_all_by_centroid"]

    # Always include remove_color (cheap filter)
    ops.append("remove_color")

    # Add repetition / block replacement ops unconditionally
    ops += [
        "repeat_pattern",
        "replace_region",
        "grow_block",
    ]

    # Deduplicate ops
    ops = list(dict.fromkeys(ops))

    # Configure search adaptively
    cfg = SearchConfig()
    cfg.time_budget_s = time_budget_s
    cfg.allow_ops = ops

    # Default beam/depth
    cfg.max_depth = 4
    cfg.beam_size = 64

    # Expand search if object counts match (safe to go deeper)
    if feats.get("obj_count_same"):
        cfg.max_depth = max(cfg.max_depth, 5)
        cfg.beam_size = max(cfg.beam_size, 128)

    # Expand search if shapes differ (compression/expansion cases)
    if not feats["same_shape"]:
        cfg.max_depth = max(cfg.max_depth, 6)
        cfg.beam_size = max(cfg.beam_size, 128)

    # Search (instrumented + pruning)
    try:
        progs, stats = bfs_synthesize(task, cfg)
    except Exception as e:
        print(f"[WARN] bfs_synthesize crashed: {e}")
        progs, stats = [], SearchStats()

    p1, p2 = pick_two(progs)

    # Fallbacks if no valid programs
    if not p1:
        out_dims = dims(task.train[0].output) if task.train else (0, 0)
        p1 = Program([Op("tile_to_target", {"target_h": out_dims[0], "target_w": out_dims[1]})])
        p2 = Program([Op("repeat_scale", {"k": 2})])
    elif not p2:
        p2 = Program([Op("rotate", {"angle": 180})])

    # Apply to test cases
    attempts: List[Dict[str, Grid]] = []
    for ti in task.test:
        tgt = dims(task.train[0].output) if task.train else dims(ti.input)
        o1 = p1.apply(ti.input)
        o1 = micro_tune(o1, tgt)
        o2 = p2.apply(ti.input)
        o2 = micro_tune(o2, tgt)
        attempts.append({"attempt_1": o1, "attempt_2": o2})

    return attempts, stats


# ----------------------------
# Memory & Logging Functions
# ----------------------------
def hash_train(train: List[Any]) -> str:
    return hashlib.md5(json.dumps(train, sort_keys=True).encode()).hexdigest()

def load_memory_db() -> Dict[str, List[Grid]]:
    return json.load(open(MEMORY_DB)) if Path(MEMORY_DB).exists() else {}

def save_memory_db(db: Dict[str, List[Grid]]):
    json.dump(db, open(MEMORY_DB, "w"), indent=2)

def log_missing(task_id: str, task: Any):
    entry = {
        "task_id": task_id,
        "train": [{"input": p.input, "output": p.output} for p in task.train],
        "test": [{"input": ti.input} for ti in task.test]
    }
    with open(FAIL_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ----------------------------
# Main Pipeline
# ----------------------------
def main():
    parser = argparse.ArgumentParser("REAP v8.1 Solver")
    parser.add_argument("--infile", required=True, help="ARC tasks JSON input")
    parser.add_argument("--outfile", default="submission.json", help="Output predictions file")
    parser.add_argument("--time_per_task", type=float, default=20.0, help="Time budget per task (sec)")
    parser.add_argument("--max-workers", type=int, default=1, help="Number of worker processes (parallelism)")
    args = parser.parse_args()

    raw = json.load(open(args.infile))
    memory_db = load_memory_db()
    submission = {}

    print("REAP v8.1 starting...")

    task_ids = list(raw.keys())
    total_tasks = len(task_ids)

    # Open CSV log file
    stats_csv_path = Path(args.outfile).with_suffix(".stats.csv")
    csv_file = open(stats_csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "task_id", "elapsed", "from_mem",
        "depth_reached", "valids_found", "best_firstpair_distance",
        "survivors_per_depth", "kept_after_filter",
        "total_candidates", "crashes_during_apply",
        "best_dims_match"
    ])

    ### Parallel Execution with Full CPU Utilization ###
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing

    # If user did not set --max-workers, use all CPU cores
    if args.max_workers <= 0:
        args.max_workers = multiprocessing.cpu_count()

    with ProcessPoolExecutor(max_workers=args.max_workers) as ex:
        futures = {
            ex.submit(solve_single_task, tid, raw[tid], memory_db, args.time_per_task): tid
            for tid in task_ids
        }

        for i, fut in enumerate(as_completed(futures), start=1):
            tid = futures[fut]
            try:
                result = fut.result()
            except Exception as e:
                print(f"[{i}/{total_tasks}] Task {tid} crashed: {e}")
                continue

            submission[tid] = result["submission"]

            if result["from_mem"]:
                print(f"[{i}/{total_tasks}] Task {tid} solved from memory in {result['elapsed']:.2f}s.")
                csv_writer.writerow([tid, result["elapsed"], True, "", "", "", "", "", "", "", ""])
            else:
                stats = result["stats"]
                if not stats:
                    print(f"[{i}/{total_tasks}] Task {tid} failed (no stats).")
                    csv_writer.writerow([tid, result["elapsed"], False, "", "", "", "", "", "", "", ""])
                else:
                    print(
                        f"[{i}/{total_tasks}] Task {tid} done in {result['elapsed']:.2f}s | "
                        f"depth={stats.depth_reached} valids={stats.valids_found} "
                        f"best_d={stats.best_firstpair_distance} beam={stats.survivors_per_depth}"
                    )
                    csv_writer.writerow([
                        tid, result["elapsed"], False,
                        getattr(stats, "depth_reached", ""),
                        getattr(stats, "valids_found", ""),
                        getattr(stats, "best_firstpair_distance", ""),
                        getattr(stats, "survivors_per_depth", ""),
                        getattr(stats, "kept_after_filter", ""),
                        getattr(stats, "total_candidates", ""),
                        getattr(stats, "crashes_during_apply", ""),
                        getattr(stats, "best_dims_match", ""),
                    ])

                # Train/test correctness check if counts match
                if "train" in result:
                    train = result["train"]
                    if len(train) == len(raw[tid]["test"]):
                        try:
                            correct = all(
                                eq_grid(r["attempt_1"], tp.output)
                                for r, tp in zip(result["submission"], train)
                            )
                        except Exception:
                            correct = False
                        if correct:
                            print(f"   -> Correct on training! Added to memory.")
                            memory_db[result["train_hash"]] = [
                                r["attempt_1"] for r in result["submission"]
                            ]
                        else:
                            print(f"   -> Incorrect. Logging for DSL review.")
                            log_missing(tid, type("Spec", (), {"train": train, "test": []}))

    save_memory_db(memory_db)
    json.dump(submission, open(args.outfile, "w"), indent=2)
    csv_file.close()
    print("\nSubmissions saved to", args.outfile)
    print(f"Per-task stats saved to {stats_csv_path}")


# ----------------------------
# Worker for Parallel Execution
# ----------------------------

def solve_single_task(tid: str, spec: dict, memory_db: dict, time_per_task: float):
    import time
    t0 = time.time()

    # Parse spec into train/test objects using Example dataclass
    spec_parsed = type("Spec", (), {})()
    spec_parsed.train = [Example(input=p["input"], output=p["output"]) for p in spec["train"]]
    spec_parsed.test = [Example(input=t["input"]) for t in spec["test"]]

    # Memory shortcut
    train_hash = hash_train([{"input": p.input, "output": p.output} for p in spec_parsed.train])
    if train_hash in memory_db:
        outs = memory_db[train_hash]
        elapsed = time.time() - t0
        return {
            "tid": tid,
            "submission": [{"attempt_1": g, "attempt_2": g} for g in outs],
            "from_mem": True,
            "elapsed": elapsed,
            "stats": None,
            "train_hash": train_hash,
        }

    # Solve task
    res, stats = solve_task(spec_parsed, time_budget_s=time_per_task)
    elapsed = time.time() - t0
    return {
        "tid": tid,
        "submission": res,
        "from_mem": False,
        "elapsed": elapsed,
        "stats": stats,
        "train_hash": train_hash,
        "train": spec_parsed.train,
    }



if __name__ == "__main__":
    main()

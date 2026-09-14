"""Bridges a `clearing.Scene` (clearing-scenes/clearing/scene.py) to the
`Graph`/`Searcher.graphSearch` world the 4 approaches expect, without
touching either side's own code.

The two data models don't line up 1:1, so this module absorbs every
adaptation (see the approved plan's "Key technical findings" for the why):

- `Node`/`Edge`/`Graph` are otherwise-opaque containers to `graphSearch`, so
  a scene vertex's xyz is a fine stand-in for the grid `pos` GraphBuilder
  normally fills in.
- `graphSearch`'s `startNodes` is a *count*: nodes `0..startNodes-1` are
  valid spanning-tree roots. Scene vertex ids are arbitrary, so nodes are
  renumbered here with the user-chosen start vertices first.
- `cellpriors[cell[0], cell[1]]` / `D[node.idx]` (see any `Searcher.py`)
  index detection-set cells as 2-tuples into a 2D array -- GraphBuilder's
  grid cells are `(x, y)`. A scene cell is a single int id, so it is
  represented here as `(cell_id, 0)` into an `(n_cells, 1)` prior array; no
  code in `Searcher.py` needs to change for that.
"""

from __future__ import annotations

import base64
import json
import os
import re
import sys

import numpy as np
import scipy.sparse
from scipy.sparse.csgraph import dijkstra as sparse_dijkstra
from scipy.spatial import cKDTree

_CLEARING_SCENES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "clearing-scenes"
)
if _CLEARING_SCENES_DIR not in sys.path:
    sys.path.insert(0, _CLEARING_SCENES_DIR)

from clearing import roster as clearing_roster  # noqa: E402

from strategy_service.approaches import Graph  # noqa: E402

# The scene's cell lattice (topology + edge lengths) never changes once
# loaded, only the hotspot source cells do -- so the sparse adjacency matrix
# is built once per scene and reused, rather than re-assembled from
# `evader_edges` on every /api/priors preview call. Keyed by id(scene) since
# server.py already caches one Scene object per name for the process
# lifetime; nothing here needs to survive past that.
_adjacency_cache: dict[int, "scipy.sparse.csr_matrix"] = {}


def _cell_adjacency(scene) -> "scipy.sparse.csr_matrix | None":
    key = id(scene)
    if key not in _adjacency_cache:
        a, b = scene.evader_edges
        if a.size:
            w = np.linalg.norm(scene.cell_xyz[a] - scene.cell_xyz[b], axis=1)
            _adjacency_cache[key] = scipy.sparse.coo_matrix(
                (np.concatenate([w, w]), (np.concatenate([a, b]), np.concatenate([b, a]))),
                shape=(scene.n_cells, scene.n_cells),
            ).tocsr()
        else:
            _adjacency_cache[key] = None
    return _adjacency_cache[key]


# The viewer's shipped per-scene JS (viewer/data/<scene>.js) renders a
# DECIMATED point cloud for the walkable surface -- its point count and
# indexing do not match `Scene.cell_xyz` at all (confirmed empirically:
# same scene, different point count, and index i is a different physical
# location in each). A click in the viewer can only be turned into a real
# Scene cell id via its 3D position, not by reusing whatever index the
# viewer's own raycast happened to hit. One KD-tree per scene, cached like
# the adjacency matrix above.
_kdtree_cache: dict[int, cKDTree] = {}


def _cell_kdtree(scene) -> cKDTree:
    key = id(scene)
    if key not in _kdtree_cache:
        _kdtree_cache[key] = cKDTree(scene.cell_xyz)
    return _kdtree_cache[key]


def nearest_cell(scene, xyz) -> int:
    """The Scene cell id whose `cell_xyz` is closest to a clicked point."""
    _dist, idx = _cell_kdtree(scene).query(xyz)
    return int(idx)


# The same decimation/reindexing problem, the other direction: a value
# computed per real Scene cell (like the prior heatmap) has to be shown on
# the viewer's own differently-sized, differently-ordered surface point
# cloud. Reads the exact same viewer/data/<scene>.js the browser loads,
# decodes its "surface" points with the same convention app.js's own
# dequantise() uses, and maps each one to its nearest real cell -- once per
# scene, cached, so a heatmap response can be reindexed to line up with
# `S.surf` with no changes needed on the JS side.
_viewer_cell_map_cache: dict[str, np.ndarray] = {}


def viewer_surface_cell_map(scene, scene_name: str, viewer_data_dir: str) -> np.ndarray:
    if scene_name in _viewer_cell_map_cache:
        return _viewer_cell_map_cache[scene_name]

    path = os.path.join(viewer_data_dir, f"{scene_name}.js")
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    pattern = r"window\.SCENES\[" + re.escape(json.dumps(scene_name)) + r"\]\s*=\s*(\{.*\});?\s*$"
    m = re.search(pattern, content, re.S)
    if not m:
        raise ValueError(f"could not find viewer data for scene {scene_name!r} in {path}")
    payload = json.loads(m.group(1))

    centre = np.array(payload["centre"], dtype=np.float64)
    surf = payload["surface"]
    q = np.frombuffer(base64.b64decode(surf["q"]), dtype="<u2").astype(np.float64).reshape(-1, 3)
    points = q * np.array(surf["scale"]) + np.array(surf["offset"]) + centre

    _dist, idx = _cell_kdtree(scene).query(points)
    mapping = idx.astype(np.int64)
    _viewer_cell_map_cache[scene_name] = mapping
    return mapping


def compute_priors(scene, hotspots, l, sigma, epsilon=0.05) -> np.ndarray:
    """The same math as GraphBuilder.py's PART 1 (per-category weights, a
    Gaussian mixture by shortest-path distance, normalize, blend with a
    uniform background), over the scene's own cell adjacency instead of a
    grid.

    This has to be a per-CELL field, not a per-VERTEX one: a hotspot can sit
    on any of a scene's ~30k-290k walkable cells (wherever it was clicked),
    not only on one of its ~30-150 graph vertices, and the heatmap needs a
    value everywhere on the surface, not just at those vertices. The scene's
    own precomputed `travel_seconds` is a vertex-to-vertex table and has no
    notion of an arbitrary cell at all, so it cannot stand in here -- the
    single-source Dijkstra below (one call per hotspot, over the cached
    lattice from `_cell_adjacency`) is what actually produces that field.
    `Node.prior` (used by the search itself) is then just this field summed
    over each node's detection set, in `build_graph` below.

    hotspots: list of (cell_id: int, category: 0/1/2).
    Returns an (n_cells,) array of per-cell prior mass.
    """
    n = scene.n_cells
    priors = np.zeros(n, dtype=np.float64)
    weights = [1.0, float(l), float(l) ** 2]
    sigma = max(float(sigma), 1e-6)

    if hotspots and n:
        adj = _cell_adjacency(scene)
        if adj is not None:
            sources = [int(cell) for cell, _ in hotspots]
            dist = sparse_dijkstra(adj, directed=False, indices=sources)
            for (_, category), d in zip(hotspots, dist):
                reachable = np.isfinite(d)
                priors[reachable] += weights[category] * np.exp(
                    -(d[reachable] ** 2) / (2 * sigma ** 2)
                )

    total = priors.sum()
    uniform = 1.0 / n if n else 0.0
    if total > 0:
        priors /= total
        priors = (1 - epsilon) * priors + epsilon * uniform
    else:
        priors[:] = uniform
    return priors


def build_graph(scene, start_vertex_ids, priors):
    """Build a Graph over every scene vertex, "start" vertices sorted first.

    Returns (G, D, startNodes, idx_to_vertex):
    - D[node_idx] is that node's detection set, as {(cell_id, 0), ...} --
      Searcher.py only ever reads `D[node.idx]` as an iterable of 2-tuples.
    - idx_to_vertex[node_idx] is the original scene vertex id, for
      translating results back.
    """
    start = {int(v) for v in start_vertex_ids}
    order = sorted(start) + [v for v in range(scene.n_vertices) if v not in start]
    idx_to_vertex = order
    vertex_to_idx = {v: i for i, v in enumerate(order)}

    G = Graph()
    D = []
    for idx, vid in enumerate(idx_to_vertex):
        cells = scene.detection[vid]
        prior = float(priors[cells].sum()) if len(cells) else 0.0
        G.add_node(idx, tuple(float(c) for c in scene.vertex_xyz[vid]), prior)
        D.append({(int(c), 0) for c in cells})

    for e in range(scene.edge_ij.shape[0]):
        if scene.edge_shady[e]:
            continue
        i, j = int(scene.edge_ij[e, 0]), int(scene.edge_ij[e, 1])
        ni, nj = vertex_to_idx[i], vertex_to_idx[j]
        t_ij = float(scene.travel_seconds[i, j])
        t_ji = float(scene.travel_seconds[j, i])
        if np.isfinite(t_ij):
            G.add_edge(G.nodes[ni], G.nodes[nj], t_ij)
        if np.isfinite(t_ji):
            G.add_edge(G.nodes[nj], G.nodes[ni], t_ji)

    start_nodes = len(start) if start else scene.n_vertices
    return G, D, start_nodes, idx_to_vertex


def travel_time_matrix(scene, idx_to_vertex) -> np.ndarray:
    """The all-pairs travel-time table, reindexed to node order -- fed to
    graphSearch as `travelTime` so it can skip its own aStar precompute."""
    idx = np.asarray(idx_to_vertex)
    return scene.travel_seconds[np.ix_(idx, idx)]


def build_moves(scene, idx_to_vertex, strategy):
    """Convert a returned strategy (node-index based, see any Searcher.py's
    graphSearch docstring) into original-vertex-id moves with real walked
    paths, plus the initial placement and each vertex's first-visit time.

    A move's source/target are travel-time-table neighbours (from
    graphSearch's own `flagDistance`/"nearest flag" reasoning), not
    necessarily adjacent in the guard graph -- exactly like the grid version
    (`TrajectoryPlanning.computeTrajectory`) runs `aStar` point-to-point for
    every move rather than only along G's edges. `roster.walked_paths`
    (clearing-scenes/clearing/roster.py) is the scene-native equivalent: it
    reuses the shipped guard-graph polyline where a move happens to be one,
    and otherwise reconstructs the walk via Dijkstra over the same walkable
    lattice `Scene.evader_edges` is built from.

    Returns (root, moves, visit_time):
    - root: {"vertex": int, "robots": int}
    - moves: [{"source", "target", "robots", "t_departure", "t_arrival",
               "path": [[x,y,z], ...]}, ...]
    - visit_time: {vertex_id: earliest time a robot is standing there}
    """
    root = None
    entries = []
    for entry in strategy:
        if len(entry) == 3:
            _, root_idx, robots = entry
            root = {"vertex": idx_to_vertex[root_idx], "robots": int(robots)}
            continue
        entries.append(entry)

    pairs = [(idx_to_vertex[e[0]], idx_to_vertex[e[1]]) for e in entries]
    paths, _worst_disagreement_m = clearing_roster.walked_paths(scene, pairs)

    moves = []
    visit_time = {}
    if root is not None:
        visit_time[root["vertex"]] = 0.0

    for (src_idx, dst_idx, robots, t_departure, t_arrival) in entries:
        src_vertex, dst_vertex = idx_to_vertex[src_idx], idx_to_vertex[dst_idx]
        path = paths.get((src_vertex, dst_vertex), np.zeros((0, 3), np.float32))
        moves.append({
            "source": src_vertex,
            "target": dst_vertex,
            "robots": int(robots),
            "t_departure": float(t_departure),
            "t_arrival": float(t_arrival),
            "path": path.tolist(),
        })
        t = float(t_arrival)
        if dst_vertex not in visit_time or t < visit_time[dst_vertex]:
            visit_time[dst_vertex] = t

    return root, moves, visit_time

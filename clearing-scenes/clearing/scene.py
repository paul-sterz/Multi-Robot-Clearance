"""Loading a scenario: the YAML, the geometry beside it, and one lattice rule.

A `Scene` is everything a clearing algorithm needs and nothing else:

    cells        the walkable surface, one point per cell
    detection    D(p) per vertex, as cell indices -- what a searcher there sees
    edges        the guard-region graph, regular or shady
    travel       vertex-to-vertex time and distance for the platform

All of it is already inside the scenario boundary. The cut happens once, in
`scripts/export_from_e1.py`, against the lines drawn and approved upstream in
`e1-clearing-graph`; by the time a scene is loaded there is no outside left to
exclude, and no algorithm in this package has to know the boundary exists.
`Scene.boundary` is the line itself, carried for drawing and for the record.

Two things are computed here rather than shipped, because both are short and
shipping them would let a file drift away from the rule that defines it:
the evader adjacency (`Scene.evader_edges`) and the all-pairs travel times
read back out of the YAML.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np
import yaml

SCENES_DIR = Path(__file__).resolve().parent.parent / "scenes"


# ---------------------------------------------------------------------------
# the lattice
# ---------------------------------------------------------------------------


def lattice_edges(cell_ij: np.ndarray, height: np.ndarray, cell_size: float,
                  max_step: float):
    """8-connected edges over the walkable cells, as (a, b, length).

    Two cells are neighbours when their columns touch in the 8-neighbourhood
    AND their heights differ by no more than `max_step`. The length is the 3D
    step, so a ramp costs its slope rather than its plan projection.

    THE HEIGHT TEST IS NOT DECORATION. The surface is per-voxel, so one column
    can carry two cells -- the ground under an arcade and the terrace over it.
    Plain (i, j) adjacency would make those neighbours and let an evader step
    four metres vertically through a stone vault. For the same reason a column
    is joined over its FULL range of cells and not just its first one, or the
    upper layer of an overhang ends up with no neighbours at all.
    """
    n = cell_ij.shape[0]
    if n == 0:
        return (np.zeros(0, np.int64),) * 2 + (np.zeros(0, np.float64),)

    ny = int(cell_ij[:, 1].max()) + 2
    key = cell_ij[:, 0].astype(np.int64) * ny + cell_ij[:, 1].astype(np.int64)
    order = np.argsort(key, kind="stable")
    ks = key[order]

    A, B, W = [], [], []
    for di, dj in ((1, 0), (0, 1), (1, 1), (1, -1)):
        flat = float(np.hypot(di, dj) * cell_size)
        target = key + di * ny + dj
        lo = np.searchsorted(ks, target, side="left")
        hi = np.searchsorted(ks, target, side="right")
        cnt = hi - lo
        total = int(cnt.sum())
        if total == 0:
            continue
        a = np.repeat(np.arange(n, dtype=np.int64), cnt)
        step = np.arange(total, dtype=np.int64) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        b = order[np.repeat(lo, cnt) + step]
        dz = np.abs(height[a] - height[b])
        keep = dz <= max_step
        a, b, dz = a[keep], b[keep], dz[keep]
        A.append(a)
        B.append(b)
        W.append(np.sqrt(flat * flat + dz * dz))
    if not A:
        return (np.zeros(0, np.int64),) * 2 + (np.zeros(0, np.float64),)
    return np.concatenate(A), np.concatenate(B), np.concatenate(W)


def _csr(a: np.ndarray, b: np.ndarray, n: int):
    """An undirected CSR over the given edge list."""
    from scipy import sparse

    g = sparse.coo_matrix((np.ones(a.size, np.uint8), (a, b)),
                          shape=(n, n)).tocsr()
    return g + g.T


def _keep_the_surface_connected(a: np.ndarray, b: np.ndarray, n: int,
                                drop: np.ndarray) -> tuple[np.ndarray, int]:
    """Un-drop the speck fragments that are holding the surface together.

    Returns the corrected mask and how many fragments came back. A port of
    `e2.scene._keep_the_surface_connected` in `e1-clearing-graph`, kept
    line-for-line comparable to it so the two cannot drift.

    The test is not "does this fragment touch two pieces" but "does removing it
    RAISE the number of pieces", which is the same question asked in the
    presence of every other removal. Fragments are offered least-area-first and
    a union-find over the surviving pieces records what has already been
    reconnected, so a threshold that is one of two parallel doorways is
    correctly dropped once the other has been kept.

    ONE PASS IS ENOUGH. Putting a fragment back can only merge pieces, never
    split them, so a fragment that was not needed before cannot become needed
    after -- the union-find is monotone and one pass sees every fragment.
    """
    from scipy.sparse.csgraph import connected_components

    if not drop.any():
        return drop, 0

    def label(mask: np.ndarray) -> np.ndarray:
        m = mask[a] & mask[b]
        return connected_components(_csr(a[m], b[m], n), directed=False)[1]

    piece = label(~drop)                 # the pieces the filter would leave
    frag = label(drop)                   # the speck fragments themselves

    # incidence: which surviving pieces does each fragment touch?
    at_edge = drop[a] ^ drop[b]
    f_side = frag[np.where(drop[a], a, b)[at_edge]]
    p_side = piece[np.where(drop[a], b, a)[at_edge]]

    rows = np.flatnonzero(drop)
    f_of_row = frag[rows]
    size = np.bincount(f_of_row, minlength=int(frag.max()) + 1)

    parent = np.arange(n, dtype=np.int64)          # union-find over piece ids

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    order = np.argsort(f_side, kind="stable")
    f_sorted, p_sorted = f_side[order], p_side[order]
    bounds = np.flatnonzero(np.r_[True, f_sorted[1:] != f_sorted[:-1], True])
    groups = [(int(f_sorted[lo]), p_sorted[lo:hi])
              for lo, hi in zip(bounds[:-1], bounds[1:])]
    groups.sort(key=lambda q: int(size[q[0]]))     # least area back first

    back = np.zeros(size.size, dtype=bool)
    for fid, sides in groups:
        roots = {find(int(q)) for q in np.unique(sides)}
        if len(roots) < 2:
            continue
        it = iter(roots)
        r0 = next(it)
        for r in it:
            parent[r] = r0
        back[fid] = True

    if not back.any():
        return drop, 0
    out = drop.copy()
    out[rows[back[f_of_row]]] = False
    return out, int(back.sum())


# ---------------------------------------------------------------------------
# the scene
# ---------------------------------------------------------------------------


@dataclass
class Scene:
    name: str
    title: str
    doc: dict                       # the YAML, verbatim

    cell_xyz: np.ndarray            # (n_cells, 3) surface points [m]
    cell_ij: np.ndarray             # (n_cells, 2) grid indices
    cell_size: float

    detection: list[np.ndarray]     # per vertex: cell indices it sees
    #: per vertex, the cells of `dD(p)`: the detection-set boundary, which is
    #: what a guard region is carved out of
    detection_boundary: list[np.ndarray]
    vertex_xyz: np.ndarray            # (n_vertices, 3) sensor positions
    vertex_cell: np.ndarray           # (n_vertices,) the cell each vertex stands on

    edge_ij: np.ndarray             # (n_edges, 2) guard-region edges
    edge_shady: np.ndarray          # (n_edges,) bool
    #: per edge, the cells of the guard region `G_ij = dD(p_i) n D(p_j)`
    guard_region: list[np.ndarray]

    travel_seconds: np.ndarray      # (n_vertices, n_vertices), inf where unreachable
    travel_metres: np.ndarray
    uncoverable: np.ndarray         # cells no vertex sees, at any team size

    #: The scenario boundary as line segments in world XY, `(n, 2, 2)`: the
    #: faces between a cell the approved boundary retains and one it does not.
    #: Empty for a scene exported with `--no-boundary`.
    #:
    #: It is geometry to draw and nothing else. Nothing in this package tests
    #: a cell against it, because everything here is already inside it -- and
    #: a second, approximate copy of a cut that has already been made is how
    #: two answers to the same question get into one repository.
    site_boundary_seg: np.ndarray = field(
        repr=False, default_factory=lambda: np.zeros((0, 2, 2), np.float32))
    site_boundary_cell_size: float = 0.0

    #: per guard-graph edge, the path the platform walks between its two
    #: vertices: the true shortest route over the walkable surface, simplified
    #: for drawing. Empty where the two vertices cannot reach each other.
    routes: list[np.ndarray] = field(repr=False, default_factory=list)

    cloud_xyz: np.ndarray = field(repr=False, default=None)

    #: connected uncoverable fragments below this area are taken out of the
    #: evader space entirely. 0 keeps every cell. See `excluded`.
    speck_max_area_m2: float = 4.0

    # -- sizes --------------------------------------------------------------

    @property
    def n_cells(self) -> int:
        return int(self.cell_xyz.shape[0])

    @property
    def n_vertices(self) -> int:
        return len(self.detection)

    @property
    def cell_area(self) -> float:
        return self.cell_size ** 2

    @property
    def bounded(self) -> bool:
        """Whether a scenario boundary was applied when this scene was cut."""
        return bool((self.doc.get("site_boundary") or {}).get("bounded"))

    def area(self, mask_or_idx) -> float:
        a = np.asarray(mask_or_idx)
        n = int(a.sum()) if a.dtype == bool else int(a.size)
        return n * self.cell_area

    # -- derived ------------------------------------------------------------

    @cached_property
    def max_step(self) -> float:
        return float(self.doc["surface"]["max_step_m"])

    @cached_property
    def evader_edges(self) -> tuple[np.ndarray, np.ndarray]:
        """How the evader moves: the walkable lattice, as an edge list.

        Cells taken out by `excluded` are taken out of the graph with their
        edges: a cell that is not in the evader space cannot be walked through
        either. That is sound only because `excluded` is now forbidden to
        remove a fragment that is HOLDING THE SURFACE TOGETHER -- see the
        connectivity paragraph there. Before that, this built a wall wherever
        the fringe of a detection set failed.
        """
        a, b, _ = lattice_edges(self.cell_ij, self.cell_xyz[:, 2],
                                self.cell_size, self.max_step)
        ex = self.excluded
        keep = ~ex[a] & ~ex[b]
        return a[keep].astype(np.int32), b[keep].astype(np.int32)

    @cached_property
    def lattice(self) -> tuple[np.ndarray, np.ndarray]:
        """The walkable lattice before anything is excluded from it.

        `excluded` is decided against this, not against `evader_edges`, which
        would be circular.
        """
        a, b, _ = lattice_edges(self.cell_ij, self.cell_xyz[:, 2],
                                self.cell_size, self.max_step)
        return a.astype(np.int32), b.astype(np.int32)

    @cached_property
    def excluded(self) -> np.ndarray:
        """Uncoverable specks removed from the evader space, as a cell mask.

        Connected fragments of the uncoverable set smaller than
        `speck_max_area_m2`. These are wall-fringe slivers, every one within a
        voxel of masonry, and they are an artefact of two things: where the
        sampler stopped adding vertices, and a detection predicate that
        deliberately fails at a wall corner when one of its five rays clips the
        stone. E1 probed 300 of them against the nearest forty admissible
        sensor positions: 78% are fully visible from a position the sampler
        simply never placed, 14% reach four rays of five, and none reach zero.
        They are not places nothing can see.

        WHY THE EVADER SPACE AND NOT JUST THE FAILURE COUNT. A speck left in
        the lattice is unobserved and adjacent to contamination, so the flood
        puts it straight back and every step has to hold its boundary. On these
        scenes that is the difference between a team of twenty and a team of
        ninety -- the strategy spends its robots watching the fringe of a
        crack, and the fringe is the artefact, not the crack.

        THE EXACT REPAIR is to sample the vertices that were never placed,
        which removes each speck and its boundary at once and needs no tolerance.
        That is what `--vertex-set repaired-ground` now ships, and it is why
        the amount left here is small; `scenes/*.yaml` records it.

        AND IT MAY NOT CUT THE SITE UP. A fragment is justified for removal by
        VISIBILITY -- nothing can see it -- and that says nothing about whether
        it is load-bearing. Some of them are thresholds: a 0.40 m2 gate, a
        0.32 m2 step-over, unwatchable and holding two halves of a courtyard
        together. Take one out and the propagator stops being able to walk an
        evader through the doorway, so a strategy that leaves a leak on the far
        side is recorded as clearing the site. Measured on Blenheim Palace,
        that was the difference between 11 055 m2 contaminated and `cleared`.

        So `_keep_the_surface_connected` puts back every fragment whose removal
        would raise the number of pieces the surface is in, cheapest first, and
        those stay in the evader space as permanent contamination sources --
        which is honest, and which is exactly the ground the repaired vertex
        set puts a sensor on. `e1-clearing-graph` found this on its own side
        and fixed it the same way; see `docs/guard_graph_components.md` there.
        """
        out = np.zeros(self.n_cells, dtype=bool)
        if self.speck_max_area_m2 <= 0 or self.uncoverable.size == 0:
            return out
        from scipy.sparse.csgraph import connected_components

        a, b = self.lattice
        rows = self.uncoverable
        g = _csr(a, b, self.n_cells)
        _, lab = connected_components(g[rows][:, rows], directed=False)
        areas = np.bincount(lab) * self.cell_area
        out[rows[(areas < self.speck_max_area_m2)[lab]]] = True
        return _keep_the_surface_connected(a, b, self.n_cells, out)[0]

    @cached_property
    def coverable(self) -> np.ndarray:
        """Cells some vertex sees. Its complement is what no team can clear."""
        out = np.zeros(self.n_cells, dtype=bool)
        for d in self.detection:
            out[d] = True
        return out

    def observed(self, vertices) -> np.ndarray:
        """Union of the detection sets of the given vertices, as a cell mask."""
        out = np.zeros(self.n_cells, dtype=bool)
        for v in vertices:
            out[self.detection[int(v)]] = True
        return out

    @cached_property
    def site_boundary_fence(self) -> tuple[float, float]:
        """`(z0, z1)`: the height to extrude the boundary line to, for drawing.

        A boundary drawn as a line on the floor is invisible in both renderers
        here -- the walkable surface is drawn as points about a cell wide and
        covers it, which is exactly the ground the line is meant to bound. So
        it is extruded into a low fence whose FOOT is still exactly the line.

        ONE HEIGHT, EVERYWHERE, and not the ground under each segment. The
        boundary is a 2D region with no height, and a fence draped over the
        surface would assert one it does not have -- and the surface it would
        drape over is not terrain: the outline runs along and through buildings,
        so a column under an arcade reports the terrace above it. The ground
        within a metre of the outline spans 5.0 m at Blenheim and 6.4 m at Christ
        Church, which is what a draped fence would be following.

        So the extent is measured once per scene instead: the foot just under
        the lowest ground the outline runs along, and the rail the taller of five
        metres -- the height e1-clearing-graph uses, chosen to stay under the
        eaves -- and a metre above the highest, so the rail clears the Christ
        Church terrace rather than disappearing into it and leaving the
        boundary looking like it has gaps.

        Computed here rather than in each renderer, so the viewer and the GIFs
        cannot draw the same boundary at two different heights.
        """
        seg = np.asarray(self.site_boundary_seg, dtype=np.float64)
        if not len(seg):
            return (0.0, 0.0)
        from scipy.spatial import cKDTree

        near = cKDTree(self.cell_xyz[:, :2]).query_ball_point(seg.mean(axis=1), 1.0)
        hit = [np.asarray(i, dtype=np.int64) for i in near if len(i)]
        z = (self.cell_xyz[np.unique(np.concatenate(hit)), 2] if hit
             else self.cell_xyz[:, 2])
        z0 = float(np.percentile(z, 2)) - 0.3
        return z0, max(z0 + 5.0, float(np.percentile(z, 98)) + 1.0)

    def route(self, i: int, j: int) -> np.ndarray:
        """The walked path between two vertices, or an empty array.

        Only guard-graph edges carry one -- those are the pairs the viewer and
        the strategies ask about. `travel_metres` and `travel_seconds` cover
        every pair; the polylines cover the edges.
        """
        i, j = int(i), int(j)
        hit = np.flatnonzero(((self.edge_ij[:, 0] == i) & (self.edge_ij[:, 1] == j))
                             | ((self.edge_ij[:, 0] == j) & (self.edge_ij[:, 1] == i)))
        if hit.size == 0:
            return np.zeros((0, 3), np.float32)
        return self.routes[int(hit[0])]

    def neighbours(self, v: int) -> np.ndarray:
        """Vertices sharing a guard-region edge with `v`."""
        m = self.edge_ij[:, 0] == v
        n = self.edge_ij[:, 1] == v
        return np.unique(np.concatenate([self.edge_ij[m, 1], self.edge_ij[n, 0]]))


def load(name: str, scenes_dir: Path | str = SCENES_DIR,
         with_cloud: bool = False, speck_max_area_m2: float = 4.0) -> Scene:
    """Read `scenes/<name>.yaml` and the geometry it points at.

    `speck_max_area_m2` is the evader-space tolerance documented on
    `Scene.excluded`; pass 0 to keep every cell.
    """
    scenes_dir = Path(scenes_dir)
    doc = yaml.safe_load((scenes_dir / f"{name}.yaml").read_text())
    z = np.load(scenes_dir / doc["surface"]["geometry_file"])

    def csr(prefix):
        off, val = z[f"{prefix}_offsets"], z[f"{prefix}_cells"]
        return [val[off[i]:off[i + 1]] for i in range(off.size - 1)]

    roff, rpts = z["edge_route_offsets"], z["edge_route_points"]
    routes = [rpts[roff[k]:roff[k + 1]] for k in range(roff.size - 1)]

    return Scene(
        name=doc["scene"], title=doc["title"], doc=doc,
        cell_xyz=z["cell_xyz"], cell_ij=z["cell_ij"],
        cell_size=float(z["cell_size"]),
        detection=csr("detection"),
        detection_boundary=csr("detection_boundary"),
        vertex_xyz=z["vertex_xyz"], vertex_cell=z["vertex_cell"],
        edge_ij=z["edge_ij"], edge_shady=z["edge_shady"].astype(bool),
        guard_region=csr("guard_region"),
        travel_seconds=z["travel_seconds"], travel_metres=z["travel_metres"],
        uncoverable=z["uncoverable"], routes=routes,
        site_boundary_seg=(z["site_boundary_seg"] if "site_boundary_seg" in z.files
                           else np.zeros((0, 2, 2), np.float32)),
        site_boundary_cell_size=(float(z["site_boundary_cell_size"])
                                 if "site_boundary_cell_size" in z.files else 0.0),
        cloud_xyz=z["cloud_xyz"] if with_cloud else None,
        speck_max_area_m2=float(speck_max_area_m2),
    )


def available(scenes_dir: Path | str = SCENES_DIR) -> list[str]:
    return sorted(p.stem for p in Path(scenes_dir).glob("*.yaml"))

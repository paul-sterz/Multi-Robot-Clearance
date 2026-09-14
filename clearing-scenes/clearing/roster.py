"""Who each machine is, and the ground it walks between two steps.

A strategy says which vertices are occupied at every step and nothing else. It
is a sequence of SETS, and a set has no memory: step 12 holding {3, 9, 41} and
step 13 holding {3, 9, 44} does not say whether the machine at 41 walked to 44 or
whether the one at 9 did and 41 was abandoned. Without that, a viewer can only
blink occupied vertices on and off, and a reader cannot follow one machine
through the plan.

This module puts identities on a strategy. A fleet of machines, each with an id
it keeps from the step it walks on to the site until the end; step to step the
fleet is matched to the vertices the strategy wants by least total travel
(`scipy.optimize.linear_sum_assignment`); and every leg comes with the path the
machine actually walks over the walkable surface, not the straight line.

THE FLEET IS THE PEAK TEAM
--------------------------
A machine the next step does not need PARKS where it stands and stays in the
fleet, and the step after that may re-task it. The alternative -- letting
surplus machines disappear and charging a fresh one to walk on from the entry
whenever the team grows again -- is what `verify.makespan` used to do, and on
these scenes it invents machines wholesale: Bodleian Library peaks at 21 and
would need 53 distinct machines, Christ Church 16 against 42. Those numbers are
an artefact of the bookkeeping, not of the plan.

Parking instead makes the fleet exactly the peak team, which is the number the
tables report, and it is also the cheaper strategy: a parked machine two vertices
away is a candidate the assignment can pick over a fresh one at the entry.
`verify.makespan` is now this module, so the time and the identities cannot
disagree.

WHAT PARKING IS NOT
-------------------
A parked machine is standing at a vertex and therefore does see `D(v)` -- but the
verifier credits only the vertices the strategy names, so that sight is free
and uncounted. The guarantee is unaffected in the safe direction: every cell
`verify.propagate` calls cleared is cleared by the named vertices alone.

Nor does the walking enter the guarantee. Steps are instantaneous in the model
and a machine in transit watches nothing; the evader is arbitrarily fast, so
whether the scene is cleared is a combinatorial statement about the detection
sets and carries no time at all. The legs here are what a strategy COSTS and
what it LOOKS LIKE, never what makes it correct.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .scene import Scene, lattice_edges
from .verify import Strategy


# ---------------------------------------------------------------------------
# the roster
# ---------------------------------------------------------------------------


@dataclass
class Leg:
    """One machine's leg into one step: where it came from and what it cost."""

    machine: int
    frm: int
    to: int
    seconds: float
    metres: float
    entered: bool = False           # walked on to the site at this step

    @property
    def stays(self) -> bool:
        return self.frm == self.to


@dataclass
class Roster:
    """A strategy with identities: per step, one `Leg` per occupied vertex."""

    steps: list[list[Leg]] = field(default_factory=list)
    parked: list[list[int]] = field(default_factory=list)      # per step
    positions: list[list[int]] = field(default_factory=list)   # per step, per machine
    step_seconds: list[float] = field(default_factory=list)
    entry: int = -1
    n_machines: int = 0

    @property
    def makespan_s(self) -> float:
        """A step takes as long as its slowest machine; the run is their sum."""
        return float(sum(self.step_seconds))

    def track(self, machine: int) -> list[tuple[int, Leg]]:
        """Every step this machine was given a vertex, with the leg it walked."""
        return [(t, m) for t, step in enumerate(self.steps)
                for m in step if m.machine == machine]

    def pairs(self) -> list[tuple[int, int]]:
        """The distinct (from, to) legs walked, for `walked_paths`."""
        seen = {(m.frm, m.to) for step in self.steps for m in step if not m.stays}
        return sorted(seen)


def assign(scene: Scene, strategy: Strategy, entry: int | None = None) -> Roster:
    """Match a fleet to a strategy, step by step, by least total travel.

    Robots on site -- working or parked -- are the rows of the cost matrix and
    the step's vertices are its columns; a machine that keeps its vertex travels
    nothing. Extra rows are added only when the step wants more machines than the
    fleet has, and those walk on at `entry`, by default the first vertex of the
    first step. Since a row is added only against a column that needs it, the
    fleet ends up exactly as large as the largest step: `n_machines == n_searchers`.

    Least TOTAL travel, while the makespan is a sum of per-step MAXIMA -- the
    assignment optimises the fleet's work, not the step's clock. Matching for
    the minimax instead is a different problem (a bottleneck assignment) and a
    different claim about what the team is doing; this keeps the objective the
    makespan has always used.

    Vertices a machine cannot reach at all cost `BIG`, so the assignment avoids
    them where it can; such a leg is recorded with an infinite time and left
    out of its step's clock, exactly as the old makespan did.
    """
    from scipy.optimize import linear_sum_assignment

    strategy.validate(scene)
    steps = [tuple(int(v) for v in st) for st in strategy.steps]
    if not steps:
        return Roster()

    T = np.asarray(scene.travel_seconds, dtype=float)
    D = np.asarray(scene.travel_metres, dtype=float)
    BIG = 1e9
    start = int(steps[0][0]) if entry is None else int(entry)

    at: list[int] = []                      # where each machine of the fleet is
    out = Roster(entry=start)

    for want in steps:
        spare = max(0, len(want) - len(at))
        at.extend([start] * spare)          # the new machines, still at the gate
        cost = np.empty((len(at), len(want)))
        for j, v in enumerate(want):
            for i, u in enumerate(at):
                c = T[u, v]
                cost[i, j] = float(c) if np.isfinite(c) else BIG
        rows, cols = linear_sum_assignment(cost)

        legs, working = [], set()
        for i, j in zip(rows, cols):
            v, u = want[j], at[i]
            entered = i >= len(at) - spare
            legs.append(Leg(machine=int(i), frm=int(u), to=int(v),
                              seconds=float(T[u, v]), metres=float(D[u, v]),
                              entered=bool(entered)))
            working.add(int(i))
        for m in legs:
            at[m.machine] = m.to

        walked = [m.seconds for m in legs if np.isfinite(m.seconds)]
        out.steps.append(sorted(legs, key=lambda m: m.machine))
        out.parked.append([r for r in range(len(at)) if r not in working])
        out.positions.append(list(at))
        out.step_seconds.append(max(walked) if walked else 0.0)

    out.n_machines = len(at)
    for row in out.positions:               # machines that walk on later
        row.extend([-1] * (out.n_machines - len(row)))
    return out


# ---------------------------------------------------------------------------
# the ground under a leg
# ---------------------------------------------------------------------------


def simplify(pts: np.ndarray, eps: float) -> np.ndarray:
    """Ramer-Douglas-Peucker, iteratively. Returns the indices to keep.

    A lattice path is long straight runs with a corner every so often, so this
    takes a 500-point walk down to a couple of dozen points, visually losslessly
    at an epsilon below the cell size. The same routine `scripts/export_from_e1.py`
    ran over the graph edges, kept here so that `clearing/` needs nothing from
    `scripts/`; the distances quoted anywhere come from the FULL path.
    """
    n = len(pts)
    if n < 3:
        return np.arange(n)
    keep = np.zeros(n, dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        seg = pts[j] - pts[i]
        rel = pts[i + 1:j] - pts[i]
        length = float(np.linalg.norm(seg))
        if length < 1e-9:
            d = np.linalg.norm(rel, axis=1)
        else:
            u = seg / length
            d = np.linalg.norm(rel - np.outer(rel @ u, u), axis=1)
        m = int(np.argmax(d))
        if d[m] > eps:
            k = i + 1 + m
            keep[k] = True
            stack.append((i, k))
            stack.append((k, j))
    return np.flatnonzero(keep)


def _oriented(pts: np.ndarray, scene: Scene, i: int) -> np.ndarray:
    """The polyline, running away from vertex `i` rather than towards it."""
    if pts.shape[0] < 2:
        return pts
    a = scene.vertex_xyz[i]
    head = float(np.linalg.norm(pts[0] - a))
    tail = float(np.linalg.norm(pts[-1] - a))
    return pts if head <= tail else pts[::-1].copy()


def walked_paths(scene: Scene, pairs, eps: float = 0.12, chunk: int = 16,
                 tol: float = 0.05) -> tuple[dict[tuple[int, int], np.ndarray], float]:
    """The path walked for each `(i, j)`, as a polyline oriented from `i`.

    `scenes/*.npz` ships one polyline per GUARD-GRAPH edge, and those are used
    unchanged wherever a leg happens to be one -- the viewer must not draw two
    different lines for the same walk. A roster's legs are not confined to the
    graph, though: a machine re-tasked across the site walks between two vertices
    that share no guard region and carry no shipped route, so the rest are
    reconstructed here from a Dijkstra predecessor tree over the same walkable
    lattice `Scene.evader_edges` is built from, with the same step rule.

    The reconstructed length is checked against `travel_seconds`' own distance
    matrix and the worst disagreement is returned; anything above `tol` raises,
    because a drifted lattice would draw machines walking a route their own time
    was never made of. Unreachable pairs come back as empty arrays.
    """
    from scipy import sparse
    from scipy.sparse.csgraph import dijkstra

    want = sorted({(int(i), int(j)) for i, j in pairs if int(i) != int(j)})
    out: dict[tuple[int, int], np.ndarray] = {}
    todo: dict[int, list[int]] = {}
    for i, j in want:
        shipped = scene.route(i, j)
        if shipped.shape[0] >= 2:
            out[(i, j)] = _oriented(np.asarray(shipped, np.float32), scene, i)
        else:
            todo.setdefault(i, []).append(j)

    worst = 0.0
    if todo:
        a, b, w = lattice_edges(scene.cell_ij, scene.cell_xyz[:, 2],
                                scene.cell_size, scene.max_step)
        n = scene.n_cells
        g = sparse.coo_matrix((w, (a, b)), shape=(n, n)).tocsr()
        g = g + g.T
        srcs = sorted(todo)
        for c0 in range(0, len(srcs), chunk):
            block = srcs[c0:c0 + chunk]
            _, pred = dijkstra(g, indices=scene.vertex_cell[block], directed=False,
                               return_predecessors=True)
            for row, src in enumerate(block):
                p, s0 = pred[row], int(scene.vertex_cell[src])
                for j in todo[src]:
                    path, q = [int(scene.vertex_cell[j])], int(scene.vertex_cell[j])
                    while q != s0:
                        q = int(p[q])
                        if q < 0:
                            path = []
                            break
                        path.append(q)
                    if not path:
                        out[(src, j)] = np.zeros((0, 3), np.float32)
                        continue
                    pts = scene.cell_xyz[path[::-1]].astype(np.float64)
                    walked = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
                    quoted = float(scene.travel_metres[src, j])
                    if np.isfinite(quoted):
                        worst = max(worst, abs(walked - quoted))
                    out[(src, j)] = pts[simplify(pts, eps)].astype(np.float32)

    if worst > tol:
        raise ValueError(f"reconstructed legs disagree with the travel matrix "
                         f"by up to {worst:.3f} m; the lattices have drifted apart")
    return out, worst

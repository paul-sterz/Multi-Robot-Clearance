"""Does a strategy actually clear the scene? The contamination-set verifier.

THE MODEL
---------
State is the set of cells that could still hold the evader. It starts as the
whole walkable surface. A strategy is a sequence of steps; each step names the
vertices that are occupied. Per step:

    O := union of D(v) over the occupied vertices
    C := C \\ O                    what is seen right now is clear
    C := flood(C, allowed = ~O)   the evader runs anywhere it can still reach

and the strategy clears the scene iff C is empty at the end.

THE EVADER IS NEVER SIMULATED. There is no agent and no random seed. The
guarantee is the emptiness of C, which quantifies over every arbitrarily fast,
omniscient evader at once.

WHY THE FLOOD IS CONNECTED COMPONENTS AND NOT ITERATED DILATION
---------------------------------------------------------------
"Arbitrarily fast" means the evader crosses any distance between two steps, so
contamination is not dilated by one cell per step -- it fills every cell of ~O
reachable from a contaminated one. That is exactly a connected-component
labelling of the subgraph induced on ~O: one pass, and it cannot stop early the
way a capped dilation loop can.

The verifier is deliberately thinner than the graph. It knows cells, adjacency
and detection sets -- not guard regions, not the regular/shady classification.
That is the point: if a graph-level strategy claims to clear and this reports a
residual, the guard-region abstraction is wrong for that geometry, and the
verifier has to be able to say so without assuming the thing under test.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import sparse
from scipy.sparse import csgraph

from .scene import Scene


@dataclass
class Strategy:
    """A sequence of steps, each naming the vertices a searcher stands on.

    The paper's own object. A searcher has no identity here and no position
    between two steps: `{3, 9, 41}` followed by `{3, 9, 44}` says which places
    are held and nothing about who moved. Putting machines on it is
    `clearing.roster`, and that is a different question with a different word
    for its answer -- see `docs/glossary.md`.
    """

    steps: list[tuple[int, ...]]
    name: str = ""

    def __len__(self) -> int:
        return len(self.steps)

    @property
    def n_searchers(self) -> int:
        """How many searchers the strategy costs: the peak over its steps.

        This is the number Kolling et al. report and the one GSST minimises.
        """
        return max((len(s) for s in self.steps), default=0)

    def validate(self, scene: Scene) -> None:
        for t, step in enumerate(self.steps):
            for v in step:
                if not 0 <= int(v) < scene.n_vertices:
                    raise ValueError(
                        f"step {t}: vertex {v} is not in the graph "
                        f"(0..{scene.n_vertices - 1})")


@dataclass
class Result:
    contaminated: list[np.ndarray] = field(default_factory=list)  # per step
    residual: np.ndarray = None
    #: per step, how many vertices are occupied. Its maximum is what the
    #: paper reports and what `Strategy.n_searchers` returns.
    searchers: list[int] = field(default_factory=list)
    recontaminated: list[int] = field(default_factory=list)
    absorbed: list[int] = field(default_factory=list)      # per step
    absorb_m2: float = 0.0
    makespan_s: float = 0.0
    fleet: int = 0

    @property
    def cleared(self) -> bool:
        return not bool(self.residual.any())

    def cleared_except_uncoverable(self, scene: Scene) -> bool:
        """Cleared apart from cells no vertex of this graph can ever see.

        The honest weaker claim. `uncoverable` is a property of the sampled
        vertex set, not a theorem about the geometry -- a denser sampling could
        shrink it -- so it is reported as a tolerance and never folded silently
        into `cleared`.
        """
        return not bool((self.residual & scene.coverable).any())

    @property
    def recontamination_events(self) -> int:
        return int(sum(1 for c in self.recontaminated if c))

    def metrics(self, scene: Scene) -> dict:
        return {
            "steps": len(self.searchers),
            "n_searchers": int(max(self.searchers)) if self.searchers else 0,
            "searchers_median": int(np.median(self.searchers)) if self.searchers else 0,
            "cleared": self.cleared,
            "cleared_except_uncoverable": self.cleared_except_uncoverable(scene),
            "residual_cells": int(self.residual.sum()),
            "residual_m2": round(scene.area(self.residual), 2),
            "recontamination_events": self.recontamination_events,
            "recontaminated_cells": int(sum(self.recontaminated)),
            "absorb_m2": self.absorb_m2,
            "absorbed_cells": int(sum(self.absorbed)),
            "makespan_s": round(self.makespan_s, 1),
            "fleet": self.fleet,
        }


# ---------------------------------------------------------------------------
# the flood
# ---------------------------------------------------------------------------


def _graph(scene: Scene) -> sparse.csr_matrix:
    a, b = scene.evader_edges
    n = scene.n_cells
    g = sparse.coo_matrix((np.ones(a.size, np.uint8), (a, b)), shape=(n, n)).tocsr()
    return g + g.T


def flood(scene: Scene, seed: np.ndarray, allowed: np.ndarray,
          graph: sparse.csr_matrix | None = None) -> np.ndarray:
    """Every cell of `allowed` reachable from a `seed` cell, to a fixed point."""
    g = _graph(scene) if graph is None else graph
    idx = np.flatnonzero(allowed)
    if idx.size == 0:
        return np.zeros(scene.n_cells, dtype=bool)
    n_comp, lab = csgraph.connected_components(g[idx][:, idx], directed=False)
    hot = np.zeros(n_comp, dtype=bool)
    hot[lab[seed[idx]]] = True
    out = np.zeros(scene.n_cells, dtype=bool)
    out[idx[hot[lab]]] = True
    return out


def frontier(scene: Scene, cleared: np.ndarray) -> np.ndarray:
    """Cells of `cleared` with a walkable neighbour outside it.

    The only thing that has to be watched. A cell deep inside the cleared
    region cannot be reached without crossing the frontier; a cell on the
    frontier is where the flood comes back in.
    """
    a, b = scene.evader_edges
    out = np.zeros(scene.n_cells, dtype=bool)
    ca, cb = cleared[a], cleared[b]
    out[a[ca & ~cb]] = True
    out[b[cb & ~ca]] = True
    return out


def absorb_fragments(scene: Scene, contaminated: np.ndarray,
                     below_m2: float) -> tuple[np.ndarray, np.ndarray]:
    """Drop enclosed contaminated fragments under `below_m2`, every step.

    WHAT IT IS, AND HOW IT DIFFERS FROM THE SPECK RULE ALREADY HERE
    ----------------------------------------------------------------
    `Scene.excluded` takes fragments of the UNCOVERABLE set -- cells no vertex
    can ever see -- out of the evader space once, before anything runs. This
    takes fragments of the CONTAMINATED set out at every step: ground the
    searchers have already enclosed, too small to be worth a machine, whether or
    not anybody can see it. The two are independent and this repository now
    applies both, at 4.0 m2 and 1.0 m2 respectively, which is what
    `e1-clearing-graph` measures its own rows under.

    WHY IT IS NEEDED HERE AT ALL. Upstream, at the peak step of one strategy,
    18 searchers stood over a frontier of 36 m2 that was 99 separate fragments of
    median 0.37 m2. Team size correlates with the NUMBER of frontier fragments
    and barely with its area: the site is not what costs 18, the crumbs at the
    edges of the detection sets are.

    WHAT IT ASSERTS, AND WHERE THAT STOPS. Most of what it takes is smaller
    than the target's own footprint, pi * r_t^2 = 0.20 m2, and for those it
    asserts nothing the model does not already say -- a person does not fit.
    Above that it is E1's `A_min` argument applied to a transiently severed
    piece of an admitted component rather than to a standalone one, which is
    weaker, and it is an operator decision rather than a derivation. Pass
    `below_m2 = 0.20` to take only what needs no decision, or `0.0` for the
    propagator without it.

    Returns the reduced set and the mask of what was taken.
    """
    gone = np.zeros(scene.n_cells, dtype=bool)
    if below_m2 <= 0.0 or not contaminated.any():
        return contaminated, gone

    # Built from the edge list restricted to contaminated cells rather than
    # from the whole-surface CSR: this runs once per step, and on the Bodleian
    # that is a 287 000-cell matrix rebuilt some thousands of times.
    a, b = scene.evader_edges
    keep = contaminated[a] & contaminated[b]
    n = scene.n_cells
    g = sparse.coo_matrix(
        (np.ones(int(keep.sum()), dtype=np.uint8), (a[keep], b[keep])),
        shape=(n, n)).tocsr()
    _, lab = csgraph.connected_components(g + g.T, directed=False)
    rows = np.flatnonzero(contaminated)
    sizes = np.bincount(lab[rows]) * scene.cell_area
    gone[rows[sizes[lab[rows]] < below_m2]] = True
    return contaminated & ~gone, gone


# ---------------------------------------------------------------------------
# propagation
# ---------------------------------------------------------------------------


def propagate(scene: Scene, strategy: Strategy, verbose: bool = False,
              absorb_m2: float = 1.0) -> Result:
    """Run a strategy through the contamination set and report what is left.

    `absorb_m2` drops enclosed contaminated fragments smaller than that at
    every step -- see `absorb_fragments`. It defaults to the 1.0 m2 that
    `e1-clearing-graph` judges every one of its own rows under, so that a
    number here and a number there are the same number; `absorb_m2 = 0` is the
    propagator without it, which is what this repository reported before the
    two were reconciled.

    IT IS NOT A ROUNDING TOLERANCE and the difference is not small. A half
    square metre of contamination left enclosed behind the front is a SEED: the
    step after next, when the region it sits in reopens, it floods everything
    the front has passed. On Blenheim Palace two of the three cycle-edge
    variants clear at 1.0 m2 and leave 11 100 m2 at 0.0 -- one crumb, the whole
    site. Which of those is the honest verdict is the question `absorb_fragments`
    documents; what is not defensible is quoting one of them against a table
    computed with the other.
    """
    strategy.validate(scene)
    g = _graph(scene)

    C = ~scene.excluded          # excluded cells are never contaminated
    C, _ = absorb_fragments(scene, C, absorb_m2)
    res = Result()
    res.absorb_m2 = float(absorb_m2)
    prev = C.copy()
    for t, step in enumerate(strategy.steps):
        O = scene.observed(step)
        C = flood(scene, C & ~O, ~O, graph=g)
        # after the flood, not before it: a fragment counts as enclosed only
        # once the evader has been given every move it has
        C, gone = absorb_fragments(scene, C, absorb_m2)
        res.absorbed.append(int(gone.sum()))
        res.contaminated.append(C.copy())
        res.searchers.append(len(step))
        res.recontaminated.append(int((C & ~prev).sum()))
        prev = C
        if verbose and (t + 1) % 10 == 0:
            print(f"    step {t + 1}: {len(step)} searchers, "
                  f"contaminated {int(C.sum()):,}", flush=True)
    res.residual = C
    # one matching, used twice: how long the walking takes and how many
    # distinct machines do it
    from .roster import assign      # local: roster.py imports Strategy from here

    roster = assign(scene, strategy)
    res.makespan_s = roster.makespan_s
    res.fleet = roster.n_machines
    return res


# ---------------------------------------------------------------------------
# how long it takes
# ---------------------------------------------------------------------------


def makespan(scene: Scene, strategy: Strategy) -> float:
    """Wall-clock time of the strategy, from the travel times in the YAML.

    One line, because the matching that produces it is the same matching that
    gives every searcher an identity, and two implementations of it would be two
    strategies: `clearing.roster.assign` puts a fleet on the strategy -- searchers
    matched to each step's vertices by minimum total travel, a searcher that keeps
    its vertex travelling nothing, a searcher the step does not need parking where
    it stands, and a fresh one walking on at the entry point only when the team
    grows past the fleet. A step takes as long as its slowest searcher.

    This is the strategy's SECONDARY cost. It does not enter the guarantee at
    all: the evader is arbitrarily fast, so whether the scene is cleared is a
    combinatorial statement about the detection sets and carries no time.
    """
    from .roster import assign      # local: roster.py imports Strategy from here

    return assign(scene, strategy).makespan_s


def fleet(scene: Scene, strategy: Strategy) -> int:
    """How many distinct MACHINES the strategy needs, over the whole run.

    Equal to `n_searchers` by construction -- see `clearing.roster.assign`, which
    adds a searcher only when a step wants one the fleet has not got. It is
    reported anyway, because the two numbers being equal is precisely the claim
    the searcher view makes and a silent regression in it would be invisible.
    """
    from .roster import assign

    return assign(scene, strategy).n_machines


def all_at_once(scene: Scene) -> Strategy:
    """Every vertex occupied at once, forever. The feasibility ceiling.

    Not a comparison: an unlimited team held simultaneously. A residual that
    survives this survives every strategy over this vertex set, of any size.
    """
    return Strategy([tuple(range(scene.n_vertices))], name="all vertices at once")

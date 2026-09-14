"""Kolling et al. (2010) as published: GSST over random spanning trees.

WHAT THIS IS, AND WHAT IT IS NOT
---------------------------------
It is NOT GRAPH-CLEAR. The 2010 paper cites GRAPH-CLEAR (Kolling & Carpin,
IROS 2007) in related work as *the alternative graph model it does not use*,
and its planner never looks at a cell. A module implementing that other
lineage used to sit beside this one and has been removed; `docs/gsst.md`
records what it measured.

What the paper actually runs is this:

  * contamination lives on VERTICES, following the robotics-adapted edge-search
    variant of Hollinger et al. (2008) rather than classical Parsons edge
    search;
  * strategies come from the label-based tree algorithm of Barriere et al.
    (2002), which yields contiguous strategies without recontamination;
  * SLIDING IS FORBIDDEN. A searcher driving cross-country between two strategic
    locations cannot promise that its path covers the detection-set boundaries
    on the way, so only `place` and `remove` survive. That costs one searcher
    in the leaf case and the label recursion records it (see `label_no_slide`);
  * the anytime layer is GSST (Hollinger et al.): generate many random
    depth-first spanning trees, solve each, convert the tree strategy back to a
    graph strategy by leaving a searcher wherever a CYCLE EDGE leads into a
    contaminated vertex, and keep the cheapest.

The geometry -- sampling, detection sets, guard regions, the regular/shady
classification -- is upstream and unchanged. That is the paper's own
architecture: all the terrain realism is absorbed before the graph exists, and
the clearing theory is reused almost as written.

THE THREE CYCLE-EDGE VARIANTS
------------------------------
The paper's first headline finding is that shady edges can be dropped. It
compares exactly three treatments, and so does `gsst`:

    naive           regular and shady edges treated alike
    regular         only regular edges count as cycle edges, which is
                    equivalent to deleting every shady edge outside the tree
    regular_biased  additionally bias the depth-first construction toward
                    regular edges, pushing shady edges out of the tree where
                    the previous variant can then drop them

Reported as 2-3 searchers in the minimum across all tested conditions, with
lower variance. Whether that reproduces here is the question this module is
for.

WHAT THE NUMBER IS, AND WHAT IT IS NOT
---------------------------------------
`gsst` returns a number of searchers and a strategy that clear the GUARD GRAPH
under vertex contamination. That is the paper's claim and the paper stops there. It
is not a cell-level guarantee: an edge is a coverage relation between a sensor
position and a piece of boundary, and whether holding every such obligation
also holds the surface is a separate question that `clearing.verify` answers
without being told any of this. Both numbers are reported. They are not the
same number and folding them together would be the whole error.

AND THERE IS A THIRD NUMBER, which is the one that clears these sites. The
strategy above is a sequence of SETS, and the model behind it moves every
searcher between two of them in no time. `clearing.execute` drives it one
MACHINE at a time -- keeping the paper's prohibition, so a mover is credited
with nothing while it drives -- and `clearing.clock` charges for the interval.
A searcher is the paper's token; a machine is a thing with an id and a route,
and `docs/glossary.md` keeps the two apart.
Atomically the published method leaves 7 749 m2 at Christ Church; driven, it
leaves none. Nothing about the strategy changed between those two figures.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .scene import Scene
from .verify import Strategy

VARIANTS = ("naive", "regular", "regular_biased")


# ---------------------------------------------------------------------------
# the guard graph, per variant
# ---------------------------------------------------------------------------


def adjacency(scene: Scene) -> tuple[list[list[int]], list[list[bool]]]:
    """The guard graph as neighbour lists, with each neighbour's shadiness.

    Undirected. The export has already collapsed the two directions of a pair
    into one edge, taking regular over shady where they disagree -- a decision
    made upstream and recorded in `docs/scene-format.md`, not re-made here.

    EACH NEIGHBOUR LIST IS SORTED, and that is not tidiness. `random_dfs_forest`
    draws `rng.permutation(len(adj[u]))` and reads `adj[u]` through it, so the
    ORDER of a neighbour list is part of what a seed means: the same seed over
    a list in edge order and over the same list sorted picks different
    neighbours and grows a different forest. Sorting is what makes a draw a
    property of the graph rather than of the order the edges happened to be
    written in, and it is what `e2.graph_guided.GraphGuide.from_edges` does
    upstream, so the two agree draw for draw at equal seed.
    """
    n = scene.n_vertices
    pairs: list[list[tuple[int, bool]]] = [[] for _ in range(n)]
    for k in range(len(scene.edge_ij)):
        i, j = (int(v) for v in scene.edge_ij[k])
        s = bool(scene.edge_shady[k])
        pairs[i].append((j, s))
        pairs[j].append((i, s))
    adj: list[list[int]] = []
    shady: list[list[bool]] = []
    for row in pairs:
        row.sort()
        adj.append([v for v, _ in row])
        shady.append([q for _, q in row])
    return adj, shady


# ---------------------------------------------------------------------------
# a random depth-first spanning forest
# ---------------------------------------------------------------------------


def random_dfs_forest(adj: list[list[int]], shady: list[list[bool]],
                      rng: np.random.Generator, bias_regular: bool = False,
                      ) -> tuple[list[list[int]], np.ndarray, list[int]]:
    """One random DFS spanning forest: children per vertex, parents, roots.

    A forest and not a tree, because the guard graph of a real site need not be
    connected and pretending otherwise silently drops components.

    Randomness is the whole point of GSST: one spanning tree is one strategy,
    and the anytime claim is that a hundred cheap draws beat one clever one.
    Neighbours are visited in a random order; with `bias_regular` the regular
    neighbours are drawn first, which is the paper's third variant.
    """
    n = len(adj)
    parent = -np.ones(n, dtype=np.int64)
    children: list[list[int]] = [[] for _ in range(n)]
    seen = np.zeros(n, dtype=bool)
    roots: list[int] = []

    for start in rng.permutation(n):
        start = int(start)
        if seen[start]:
            continue
        roots.append(start)
        seen[start] = True
        stack = [start]
        while stack:
            u = stack.pop()
            nb = np.array(adj[u], dtype=np.int64)
            if nb.size == 0:
                continue
            order = rng.permutation(nb.size)
            if bias_regular:
                # regular first: a stable sort on the shadiness flag keeps the
                # random order inside each class
                sh = np.array(shady[u], dtype=np.int64)[order]
                order = order[np.argsort(sh, kind="stable")]
            # a stack reverses, so push in reverse to visit in `order`
            for k in order[::-1]:
                v = int(nb[k])
                if not seen[v]:
                    seen[v] = True
                    parent[v] = u
                    children[u].append(v)
                    stack.append(v)
    return children, parent, roots


# ---------------------------------------------------------------------------
# the label, without sliding
# ---------------------------------------------------------------------------


def label_no_slide(children: list[list[int]], root: int) -> dict[int, int]:
    """The Barriere et al. tree label, with the paper's no-sliding correction.

    Writing rho_1 >= rho_2 >= ... for the labels of the subtrees at a vertex,
    classical edge search gives

        lambda = max{rho_1, rho_2 + 1}

    and the 2.5D version uses

        lambda = rho_1 + 1        if rho_1 = 1
                 max{rho_1, rho_2 + 1}   otherwise

    THE DIFFERENCE IS THE LEAF CASE, and it is worth seeing why. Children are
    cleared cheapest-first, so the expensive subtree is entered last when the
    fewest siblings are still contaminated. Entering the last child costs a
    guard standing on the junction plus one searcher placed on the child's
    root; the guard can then leave, because the child's root is occupied and
    every other child is clear. So that step costs 2, and the rest of the last
    subtree costs rho_1. When rho_1 >= 2 the 2 is absorbed. When rho_1 = 1 --
    the last child is a leaf -- it is not, and the label goes to 2.

    With sliding the guard would move INTO the last subtree and clear it on the
    way, and the 2 would never appear. A searcher crossing open terrain cannot
    sweep a detection-set boundary while it drives, so it does appear here.

    Returned per vertex, entered from its parent. Leaves get 1.
    """
    lab: dict[int, int] = {}
    # iterative post-order: these forests are shallow but a 112-vertex path is
    # a real possibility and Python's recursion limit is a poor place to meet it
    stack: list[tuple[int, bool]] = [(root, False)]
    while stack:
        v, done = stack.pop()
        if not done:
            stack.append((v, True))
            for c in children[v]:
                stack.append((c, False))
            continue
        kids = sorted((lab[c] for c in children[v]), reverse=True)
        if not kids:
            lab[v] = 1
        else:
            r1 = kids[0]
            r2 = kids[1] if len(kids) > 1 else 0
            lab[v] = r1 + 1 if r1 == 1 else max(r1, r2 + 1)
    return lab


# ---------------------------------------------------------------------------
# the tree strategy: place and remove, never slide
# ---------------------------------------------------------------------------


def tree_strategy(children: list[list[int]], lab: dict[int, int],
                  root: int) -> list[frozenset[int]]:
    """The move sequence the label counts, as the occupied set per step.

    Each step is the set of vertices a searcher stands on. Steps are not single
    moves: a handover places one searcher and lifts another, which is two searchers
    and one instant, and the peak of `len(step)` over the sequence is exactly
    the label at the root.

    The recursion, for a subtree at `v` entered from a cleared parent while
    `held` is standing guard elsewhere:

        1. place a searcher on v -- v is now swept
        2. clear the children cheapest-first; while any child is still
           contaminated a guard stands on v, so the cleared siblings and the
           parent side cannot be reached
        3. for the LAST child, place a searcher on it while the guard is still
           on v, then lift the guard: the child's root is occupied, every other
           child is clear, and v is safe without anybody on it

    Step 3 is the handover the no-sliding rule forces, and it is where the
    extra searcher of `label_no_slide` is spent. When the call returns, the
    whole subtree is clear and no searcher is left inside it -- which is sound on
    a TREE, where the parent edge is the only way back in, and is exactly the
    assumption `to_graph_strategy` then has to pay for on a graph.
    """
    steps: list[frozenset[int]] = []

    def clear(v: int, held: frozenset[int]) -> None:
        steps.append(held | {v})
        kids = sorted(children[v], key=lambda c: lab[c])
        for i, c in enumerate(kids):
            if i < len(kids) - 1:
                clear(c, held | {v})
            else:
                mark = len(steps)
                clear(c, held)              # v is NOT held inside the last child
                steps[mark] = steps[mark] | {v}     # ... except for the handover

    import sys
    limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(limit, 10 * len(children) + 1000))
    try:
        clear(root, frozenset())
    finally:
        sys.setrecursionlimit(limit)
    return steps


# ---------------------------------------------------------------------------
# back to the graph: cycle edges are guarding obligations
# ---------------------------------------------------------------------------


def _flood(contaminated: np.ndarray, blocked: np.ndarray,
           adj: list[list[int]]) -> np.ndarray:
    """Every unblocked vertex reachable from a contaminated one, to a fixpoint.

    The evader is arbitrarily fast, so this is reachability and not one step of
    it: the same statement `clearing.verify.flood` makes about cells, made
    about vertices.
    """
    out = contaminated & ~blocked
    stack = list(np.flatnonzero(out))
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if not out[v] and not blocked[v]:
                out[v] = True
                stack.append(v)
    return out


def to_graph_strategy(n: int, steps: list[frozenset[int]],
                      tree_adj: list[list[int]],
                      cycle_adj: list[list[int]],
                      contaminated: np.ndarray | None = None,
                      ) -> tuple[list[tuple[int, ...]],
                                 list[frozenset[int]], np.ndarray]:
    """Leave a searcher wherever a cycle edge leads into a contaminated vertex.

    A tree strategy walks away from a cleared subtree because a tree has no
    other way back in. A graph does: every edge outside the spanning tree is a
    door the tree strategy does not know about. GSST's conversion is to station
    a searcher on the cleared side of any such door that is still open.

    CONTAMINATION IS READ OFF THE TREE, not off the graph, and that is not a
    shortcut -- it is the only order in which the question can be asked. On the
    graph a door vertex is already contaminated by the time you look at it,
    because the flood came through the door; the tree state is the strategy's
    INTENTION, and the guards are what make the graph honour it.

    It also makes the conversion sound rather than merely plausible. Suppose
    the graph contamination is inside the tree contamination at some step, as
    it is at the first. Any edge from a tree-contaminated vertex to a
    tree-clear one is either a tree edge -- and then the tree flood would have
    crossed it unless the far end is occupied by the step itself -- or a cycle
    edge, and then this rule has just put a searcher on the far end. Either way
    the boundary is occupied, the graph flood cannot cross it, and the
    containment survives the step.

    THE STATE IS THREADED THROUGH THE COMPONENTS, not restarted at each. A
    forest is cleared one component at a time, but the contamination is one set
    over the whole graph: restarting it would re-mark a finished component as
    contaminated. It buys no guard -- there are no edges between components, so
    nothing there could ever put a searcher on a door -- but it makes the recorded
    state a lie, and the state is what a picture of the strategy is a picture
    of. Returned alongside the strategy for that reason.
    """
    tree_contaminated = (np.ones(n, dtype=bool) if contaminated is None
                         else contaminated.copy())
    out: list[tuple[int, ...]] = []
    dirty: list[frozenset[int]] = []
    for step in steps:
        occ = np.zeros(n, dtype=bool)
        occ[list(step)] = True
        tree_contaminated = _flood(tree_contaminated, occ, tree_adj)
        for u in range(n):
            if not occ[u] and not tree_contaminated[u] \
                    and any(tree_contaminated[w] for w in cycle_adj[u]):
                occ[u] = True
        out.append(tuple(np.flatnonzero(occ).tolist()))
        dirty.append(frozenset(int(v) for v in np.flatnonzero(tree_contaminated)))
    return out, dirty, tree_contaminated


def graph_clears(n: int, steps: list[tuple[int, ...]],
                 adj: list[list[int]]) -> tuple[bool, int]:
    """Does this strategy clear the graph? The vertex-level judge.

    Deliberately independent of everything above: it is handed a sequence of
    occupied sets and an adjacency, and it believes only those. A strategy that
    claims its label and fails here has a bug in the recursion, not a
    disagreement about geometry.
    """
    contaminated = np.ones(n, dtype=bool)
    for step in steps:
        occ = np.zeros(n, dtype=bool)
        occ[list(step)] = True
        contaminated = _flood(contaminated, occ, adj)
    return not bool(contaminated.any()), int(contaminated.sum())


# ---------------------------------------------------------------------------
# the anytime layer
# ---------------------------------------------------------------------------


@dataclass
class Draw:
    """One spanning forest, solved."""

    seed_index: int
    tree_label: int                 # max over components of the root label
    searchers: int                       # peak of the graph strategy
    steps: list[tuple[int, ...]]
    tree_edges: int
    shady_in_tree: int
    cycle_edges: int
    cleared: bool
    residual_vertices: int

    #: The forest itself, so a strategy can be DRAWN as the tree it was read
    #: off rather than guessed at from the occupied sets. Vertex ids are the
    #: scene's own -- this module never renumbers.
    children: tuple[tuple[int, ...], ...] = ()
    parent: tuple[int, ...] = ()
    roots: tuple[int, ...] = ()
    labels: tuple[int, ...] = ()
    #: Per step, the contamination the TREE strategy believes in. This is the
    #: state the cycle-edge rule was computed against, and the only one a
    #: picture of the tree is a picture of; the GRAPH state at the same step is
    #: a subset of it and is what `graph_clears` judges.
    tree_dirty: tuple[frozenset[int], ...] = ()
    shady_in_best_tree: tuple[tuple[int, int], ...] = ()


@dataclass
class GsstResult:
    variant: str
    scene: str
    best: Draw
    searchers_per_draw: list[int] = field(default_factory=list)       # per draw
    labels: list[int] = field(default_factory=list)
    n_trees: int = 0

    @property
    def strategy(self) -> Strategy:
        return Strategy(self.best.steps, name=f"gsst/{self.variant}")

    def summary(self) -> dict:
        t = np.array(self.searchers_per_draw, dtype=float)
        return {
            "variant": self.variant,
            "trees": self.n_trees,
            "searchers_min": int(t.min()),
            "searchers_mean": round(float(t.mean()), 2),
            "searchers_sd": round(float(t.std(ddof=1)) if t.size > 1 else 0.0, 2),
            "searchers_max": int(t.max()),
            "tree_label_min": int(min(self.labels)),
            "tree_label_at_best": self.best.tree_label,
            "steps": len(self.best.steps),
            "shady_in_best_tree": self.best.shady_in_tree,
            "cycle_edges_in_best": self.best.cycle_edges,
            "graph_cleared": self.best.cleared,
        }


def run(scene: Scene, variant: str = "regular", n_trees: int = 100,
        seed: int = 1, verbose: bool = False) -> GsstResult:
    """GSST: draw `n_trees` random spanning forests, keep the cheapest.

    The team of a draw is the peak of its graph strategy, not its tree label.
    The label is what the tree costs; the cycle edges are what the graph adds
    on top, and on a dense guard graph the second term is the one that decides.

    Ties on team size are broken by the shorter strategy -- an arbitrary but
    fixed rule, so that a rerun with the same seed returns the same strategy.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}, expected one of {VARIANTS}")

    n = scene.n_vertices
    adj, shady = adjacency(scene)
    shady_of = {(min(int(i), int(j)), max(int(i), int(j))): bool(sh)
                for (i, j), sh in zip(scene.edge_ij, scene.edge_shady)}
    rng = np.random.default_rng(seed)
    drop_shady = variant in ("regular", "regular_biased")
    bias = variant == "regular_biased"

    best: Draw | None = None
    searchers_per_draw, labels = [], []

    for k in range(n_trees):
        children, parent, roots = random_dfs_forest(adj, shady, rng, bias_regular=bias)

        in_tree = {(min(int(parent[v]), v), max(int(parent[v]), v))
                   for v in range(n) if parent[v] >= 0}

        # three adjacencies, and they are three different questions: what the
        # tree strategy plans on, what it must additionally hold, and what
        # contamination actually spreads along when the verifier is asked
        tree_adj: list[list[int]] = [[] for _ in range(n)]
        cycle_adj: list[list[int]] = [[] for _ in range(n)]
        graph_adj: list[list[int]] = [[] for _ in range(n)]
        shady_in_tree = 0
        cycle_edges = 0
        for e in range(len(scene.edge_ij)):
            i, j = (int(v) for v in scene.edge_ij[e])
            is_shady = bool(scene.edge_shady[e])
            tree_edge = (min(i, j), max(i, j)) in in_tree
            if tree_edge:
                shady_in_tree += is_shady
                tree_adj[i].append(j)
                tree_adj[j].append(i)
            elif drop_shady and is_shady:
                continue                    # deleted from the instance entirely
            else:
                cycle_edges += 1
                cycle_adj[i].append(j)
                cycle_adj[j].append(i)
            graph_adj[i].append(j)
            graph_adj[j].append(i)

        # one component at a time: they are cleared in sequence, so the team is
        # the largest of them and not their sum
        steps: list[tuple[int, ...]] = []
        dirty: list[frozenset[int]] = []
        labels = [0] * n
        tree_label = 0
        carried = np.ones(n, dtype=bool)
        for r in roots:
            lab = label_no_slide(children, r)
            tree_label = max(tree_label, lab[r])
            for v, q in lab.items():
                labels[v] = int(q)
            st, dy, carried = to_graph_strategy(
                n, tree_strategy(children, lab, r), tree_adj, cycle_adj,
                contaminated=carried)
            steps.extend(st)
            dirty.extend(dy)

        ok, residual = graph_clears(n, steps, graph_adj)
        searchers = max((len(s) for s in steps), default=0)
        draw = Draw(seed_index=k, tree_label=tree_label,
                    searchers=searchers, steps=steps,
                    tree_edges=len(in_tree), shady_in_tree=shady_in_tree,
                    cycle_edges=cycle_edges, cleared=ok, residual_vertices=residual,
                    children=tuple(tuple(c) for c in children),
                    parent=tuple(int(q) for q in parent),
                    roots=tuple(int(q) for q in roots),
                    labels=tuple(labels), tree_dirty=tuple(dirty),
                    shady_in_best_tree=tuple(sorted(
                        e for e in in_tree if shady_of.get(e, False))))
        searchers_per_draw.append(searchers)
        labels.append(tree_label)
        if best is None or (searchers, len(steps)) < (best.searchers,
                                                       len(best.steps)):
            best = draw
            if verbose:
                print(f"    draw {k:>3}: {searchers:>3} searchers "
                      f"(label {tree_label}), "
                      f"{len(steps)} steps  <- best", flush=True)

    return GsstResult(variant=variant, scene=scene.name, best=best,
                      searchers_per_draw=searchers_per_draw, labels=labels, n_trees=n_trees)

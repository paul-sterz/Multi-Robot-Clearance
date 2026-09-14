# The vocabulary, and which half of it is the paper's

Every name in this repository is either **the paper's**, or **deliberately not
the paper's**. There is no third category, and the point of that rule is that a
reader with Kolling et al. (2010) open can tell, from the name alone, whether
they are looking at something the paper defines or something this repository
added.

The reference is Kolling, Kleiner, Lewis & Sycara, *Pursuit-Evasion in 2.5D
Based on Team-Visibility*, IROS 2010, together with the two lineages it builds
its strategy layer from: Barrière et al. (2002) for the tree label, and
Hollinger et al. for GSST.

---

## The paper's words

| word | symbol | what it is | where |
|---|---|---|---|
| **searcher** | — | one token of the strategy: a thing that can be *placed* on a vertex and *removed* from it. It has no identity and no position between two moves. | `gsst.py`, `verify.py` |
| **vertex** | `p` | a sampled sensor position on the walkable surface | everywhere |
| **detection set** | `D(p)` | the cells a searcher standing at `p` certifies empty | `Scene.detection` |
| **detection-set boundary** | `dD(p)` | the cells of `D(p)` with a walkable neighbour outside it | `Scene.detection_boundary` |
| **guard region** | `G_ij = dD(p_i) n D(p_j)` | the part of `i`'s boundary a searcher standing at `j` can watch. Non-empty is exactly what makes `(i, j)` an edge. | `Scene.guard_region` |
| **regular / shady** | — | an edge is *shady* when its guard region is strictly contained in another one, *regular* otherwise | `Scene.edge_shady` |
| **contaminated / cleared** | — | the state, which is a SET of cells and never a position | `verify.py` |
| **recontamination** | — | cells that were cleared and are not any more | `Result.recontaminated` |
| **strategy** | — | the plan: a sequence of steps, each naming the occupied vertices | `Strategy` |
| **place / remove / slide** | — | the three moves of edge search. The 2010 formulation forbids the third, because it has no model of what a moving sensor sees. | `gsst.tree_strategy` |
| **sliding** | — | the forbidden move. Every driven record carries `sliding_credited: false`, so no row can be read as having quietly lifted the prohibition. | `execute.py`, `clock.py` |
| **label** | `lambda` | how many searchers the tree recursion says a subtree needs | `gsst.label_no_slide` |
| **cycle edge** | — | an edge outside the spanning tree | `gsst.to_graph_strategy` |
| **evader** | — | arbitrarily fast, omniscient, continuous. Never simulated. | `Scene.evader_edges` |

The number a strategy is judged by is `Strategy.n_searchers` — the largest
number of vertices occupied at any one step.

**One note on "searcher" against "robot".** Kolling et al. write in robotics
terms and count *robots*; the edge-search lineage their strategy layer comes
from — Parsons, Barrière et al., Hollinger et al. — writes *searcher*. This
repository uses **searcher** everywhere for the abstract token, and never
"robot", because "robot" would otherwise have to cover three different things
here: the token, the machine with an id, and the platform whose sensor defines
the scenario. Those are `searcher`, `machine` and `platform`.

## The file formats carry the same words

`scenes/<scene>.yaml` and `scenes/geometry/<scene>.npz` were renamed with the
code, so nothing has to be translated on the way in:

| npz array | what it holds |
|---|---|
| `vertex_xyz`, `vertex_cell` | where each vertex is, and the cell it stands on |
| `detection_offsets` / `detection_cells` | `D(p)` per vertex, CSR |
| `detection_boundary_offsets` / `detection_boundary_cells` | `dD(p)` per vertex, CSR |
| `guard_region_offsets` / `guard_region_cells` | `G_ij` per edge, CSR |
| `edge_ij`, `edge_shady` | the guard graph and its regular/shady split |
| `site_boundary_seg`, `site_boundary_cell_size` | the scenario cut, for drawing |

In `examples/<scene>.gsst.yaml`, `searchers_min/mean/sd/max` are the paper's
number over the drawn forests; `driven_no_sliding` and `driven_on_the_clock`
are the two blocks that are not the paper's, and both carry
`sliding_credited: false`.

---

## The words that are deliberately not the paper's

These name things the 2010 formulation has no model of. They are different
words on purpose, so that nothing here can be mistaken for a citation.

| word | what it is | why the paper has no word for it |
|---|---|---|
| **machine** | a physical platform with an id, a position, and a route it walks | a searcher has no identity; `{3, 9, 41}` followed by `{3, 9, 44}` does not say which one moved |
| **fleet** | how many machines the run needs | equal to `n_searchers`, except where the executor had to spend one |
| **leg** | one machine walking from one vertex to another | there is no motion in the strategy model at all, which is *why* sliding is forbidden |
| **roster** | machines matched to a strategy's steps | `clearing/roster.py` |
| **driven** | a strategy executed one machine at a time | `clearing/execute.py` |
| **the clock** | the wall-clock referee that charges for the interval between two steps | `clearing/clock.py` |
| **frontier** | cells of the *cleared region* with a walkable neighbour outside | a region frontier is a cell-level notion; the paper's boundary is per detection set |
| **speck** | a connected fragment of the uncoverable set below `speck_max_area_m2` | a tolerance, not a model |
| **absorb** | dropping an enclosed contaminated fragment below `absorb_m2` | a tolerance, not a model |
| **platform** | the ground robot the scenario is posed for: sensor height, range, speed | E1's word, upstream of the graph |
| **site boundary** | where the scenario stops, drawn by hand upstream | not in the paper; and named at length so it can never be read as `dD(p)` |

**The one collision worth naming.** "Boundary" means two things on a site: the
boundary of a detection set, and the line the scenario is cut to. They are
`detection_boundary` and `site_boundary` everywhere, never plain `boundary`,
and the npz arrays carry the same two names for the same reason.

---

## Where the two meet

`n_searchers` is a property of the strategy; `fleet` is a property of the run
that executes it. They are equal unless a machine had to be **spent** — placed
because no machine on site could make a drive without opening the frontier,
which is the paper's own `place` used as a fallback. `machines_spent` counts
them, and it is the one number that makes the two columns differ.

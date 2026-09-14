# Kolling et al. (2010) as published, and what it does on these scenes

This page is the paper's own machinery, implemented in `clearing/gsst.py` and
run by `scripts/run_gsst.py`. It exists to answer one question — *how far does
the published method get on real sites* — and the answer moved twice while it
was being answered.

**It moved once on the vertex set.** These scenes used to ship E1's raw sample,
which is what a sampler optimising *coverage* produces; coverage is not the
property a clearing argument needs, and what it left behind — unwatched
fragments holding the surface together, and ground no *ground* vertex can see —
is repaired now. See [scene-format.md](scene-format.md).

**It moved again on what a step is.** Atomically a step is instantaneous and
nobody is ever in transit. Charge for the interval instead, keeping the paper's
own prohibition on sweeping while driving, and the same strategies clear every
scene. `clearing/execute.py` and `clearing/clock.py` are that layer, and
Finding 2 below is the two readings side by side.

A second planner used to live beside this one — a GRAPH-CLEAR label and a
greedy cover of the **cell** frontier — as the thing the published method was
measured against. It was never the 2010 paper's and it has been removed; what
that costs is recorded at "What would have to change", item 3.

The vocabulary is deliberate: every name is either the paper's or deliberately
not the paper's, and [glossary.md](glossary.md) says which is which.

---

## What was implemented

The paper's architecture is deliberate: **all the terrain realism is absorbed
before the graph exists, and the clearing theory is then reused almost
unchanged.** Steps 1–3 already live in `e1-clearing-graph` and ship in
`scenes/`; this is step 4.

| step | status |
|---|---|
| sample positions greedily until free space is covered | upstream, then repaired — see [scene-format.md](scene-format.md) |
| detection sets `D(p)` from sensor height, target height, range | upstream, unchanged |
| detection-set boundaries `dD(p)` | upstream, unchanged |
| guard regions `G_ij = dD(p_i) ∩ D(p_j)`; regular vs shady | upstream, unchanged |
| **strategy on the graph** | **`clearing/gsst.py`** |
| verifying it, steps as instants | `clearing/verify.py` |
| *executing* it, one machine at a time | `clearing/execute.py` — **not in the paper** |
| charging for the interval between two steps | `clearing/clock.py` — **not in the paper** |

The last two rows are marked because the 2010 formulation has no model of a
searcher in transit at all: that is exactly why it forbids sliding. They keep
the prohibition rather than lift it, and their vocabulary is deliberately not
the paper's — *machine*, *leg*, *fleet* — so the boundary stays visible.

The strategy layer is the other lineage from GRAPH-CLEAR, exactly as the paper
specifies it:

* **contamination on vertices**, following Hollinger et al.'s robotics-adapted
  edge search rather than classical Parsons edge search;
* the **label-based tree recursion of Barrière et al.**, which yields contiguous
  strategies without recontamination;
* **sliding forbidden.** A searcher driving between two strategic locations cannot
  promise its path covers the detection-set boundaries on the way, so only
  `place` and `remove` survive. The label pays for it in the leaf case:

  ```
  classical   lambda = max{rho_1, rho_2 + 1}
  2.5D        lambda = rho_1 + 1              if rho_1 = 1
                       max{rho_1, rho_2 + 1}  otherwise
  ```

* the **GSST anytime layer**: draw many random depth-first spanning forests,
  solve each, convert back to a graph strategy by stationing a searcher wherever a
  cycle edge leads into a contaminated vertex, keep the cheapest.

`tests/test_gsst.py` checks the pairing that carries the whole thing: the move
sequence `tree_strategy` emits spends *exactly* the label `label_no_slide`
predicts, on every rooting of every hand-checkable tree, and clears it.

### Where the guard rule reads its contamination

The conversion has one subtlety worth stating, because getting it wrong is
silent. Contamination for the cycle-edge rule is read off the **tree**, not off
the graph. On the graph a door vertex is already contaminated by the time you
look at it — the flood came through the door. The tree state is the strategy's
*intention*, and the guards are what make the graph honour it.

It is also what makes the conversion sound rather than merely plausible. Any
edge from a tree-contaminated vertex to a tree-clear one is either a tree edge,
and the tree flood would have crossed it unless the far end is occupied by the
step itself, or a cycle edge, and the rule has just put a searcher on the far end.
Either way the boundary is occupied and the containment survives the step.

---

## Finding 1: dropping shady edges helps, and by more than the published amount

The paper compares three treatments of cycle edges and reports that using only
regular ones is worth **2–3 searchers in the minimum across all tested
conditions**. Over 100 random forests per variant per scene, on the
`repaired-ground` vertex set:

| scene | n | naive | regular | regular + biased tree | saved |
|---|--:|--:|--:|--:|--:|
| blenheim-palace | 47 | 13 | **10** | 12 | +3 |
| bodleian-library | 151 | 30 | **25** | **25** | +5 |
| christ-church | 81 | 14 | **11** | 12 | +3 |
| hb-allen-centre | 32 | 16 | 14 | **13** | +3 |
| keble-college | 34 | 12 | **8** | **8** | +4 |
| observatory-quarter | 60 | 21 | **17** | 18 | +4 |
| | | | | **mean** | **+3.7** |

**Reproduced, and then some.** +3 to +5, mean +3.7. It is worth naming why it
is larger here than on the released vertex set, where the same measurement gave
+2.2: the repairs added 10 to 39 vertices per scene, a denser guard graph has
more cycle edges, and what dropping shady edges saves is cycle edges. The
finding is about the graph, and the repair made the graph more of what the
finding is about.

The **variance** claim now reproduces on five scenes of six. Standard deviation
over the 100 draws falls from naive to regular everywhere except
observatory-quarter (3.30→2.96, 6.76→5.94, 3.02→2.76, 1.82→1.77, 3.40→2.81, and
3.17→3.42 is the rise).

**The biased tree still does not pay.** The paper's third variant wins once
(hb-allen-centre, 14→13), loses three times (blenheim 10→12, christ-church
11→12, observatory 17→18) and ties twice. On this scene class, biasing the
depth-first construction toward regular edges is not worth implementing;
dropping shady cycle edges is.

---

## Finding 2: the graph clears, the ground does not — until the clock is run

Every strategy above provably clears the **guard graph** under vertex
contamination — `gsst.graph_clears` is an independent check and it passes for
every draw reported. The paper stops here. This repository has a cell-level
verifier that was told nothing about guard regions, so it does not have to; and
it now has two of them, because a step is not an instant.

| scene | variant | searchers | atomic residual | driven fleet | machines spent | driven, avoidable |
|---|---|--:|--:|--:|--:|--:|
| blenheim-palace | regular | 10 | **0 m²** | 10 | 0 | **0.00 m²** |
| bodleian-library | regular_biased | 25 | 10 920 m² | 29 | 4 | **0.00 m²** |
| christ-church | regular | 11 | 7 749 m² | 13 | 2 | **0.00 m²** |
| hb-allen-centre | regular_biased | 13 | 1 185 m² | 13 | 0 | **0.00 m²** |
| keble-college | regular | 8 | **0 m²** | 8 | 0 | **0.00 m²** |
| observatory-quarter | regular_biased | 18 | 2 745 m² | 18 | 0 | **0.00 m²** |

Christ Church is 8 133 m² of walkable surface and the graph strategy leaves
7 749 m² of it contaminated **when every step is treated as an instant**. That
was never a data artefact — `all vertices occupied at once` leaves 1–11 m² on
these scenes, all of it cells no ground vertex can see — and it is not a defect
in the strategy either. It is what the atomic model cannot see.

**Driven, all eighteen rows clear.** One machine moves at a time; the mover is
credited with nothing between its two posts, which is the paper's own
prohibition; the interval is charged under the strict rule. Three scenes keep a
residual that is not avoidable — 10.52, 1.12 and 5.52 m² of ground no ground
vertex can see at any team size — and it is reported separately rather than
folded in.

**What that says about the atomic column, stated carefully.** It does NOT say
the atomic verdict was wrong. Both are correct answers to different questions,
and the paper asks the atomic one. What it says is that the enormous residuals
are an artefact of *when the machines are allowed to be somewhere*, not of the
strategy: the same posts, in the same order, held by machines that arrive one
at a time instead of all at once, hold the site. The published method's
difficulty on this geometry was a modelling difficulty.

**And it is not free.** Bodleian needs 29 machines where its atomic peak is 25,
Christ Church 13 where the peak is 11: four and two machines that had to be
*spent*, because at some move no searcher on site could make the drive without
opening the frontier. `machines_spent` is in every row.

---

## Why: an edge is not a door, it is part of one

The measurement is `scripts/boundary_coverage.py`, and it is a property of the
exported geometry alone — no strategy, no execution, no seed.

An edge exists when `G_ij = dD(p_i) ∩ D(p_j)` is non-empty: a searcher at `j` can
watch **some** of `dD(p_i)`. Every graph-level clearing argument then treats
*"i is clear and a neighbour is occupied"* as enough to keep it clear. So: how
much of the boundary is *some*?

On the `repaired-ground` vertex set:

| scene | n | best single neighbour (median) | every neighbour at once (median) | min | boundaries closed |
|---|--:|--:|--:|--:|--:|
| blenheim-palace | 47 | 0.70 | 0.96 | 0.11 | 0/47 |
| bodleian-library | 151 | 0.56 | 0.96 | 0.51 | 0/151 |
| christ-church | 81 | 0.56 | 0.89 | 0.46 | 0/81 |
| hb-allen-centre | 32 | 0.74 | 0.96 | 0.37 | 0/32 |
| keble-college | 34 | 0.82 | 0.97 | 0.07 | 0/34 |
| observatory-quarter | 60 | 0.60 | 0.94 | 0.43 | 1/60 |
| **all** | **405** | **0.60** | **0.96** | **0.07** | **1/405** |

The median neighbour watches **60%** of the boundary it is supposed to guard, and
occupying *every* neighbour of a vertex simultaneously **still** leaves a gap —
on 404 of the 405 vertices of all six sites. Exactly one boundary, at
observatory-quarter, is closed.

**The repair moved these numbers and did not change the conclusion**, which is
the useful part. On the released vertex set the same measurement gave a median
of 0.54 with 0 of 293 boundaries closed, and the worst boundaries were visible from
nowhere but their own vertex (a minimum of 0.00 on two scenes). Adding 10 to 39
vertices per scene lifts the median to 0.60, lifts the *minimum* from 0.00 to
0.07, and closes exactly one boundary. More vertices watch more of every
boundary and close essentially none of them.

**The second column is the ceiling, not just a neighbour statistic**, and that
is what makes this structural. A vertex that watches any part of `dD(i)` has a
non-empty guard region with `i` and is therefore *by definition* a neighbour of
`i`: "every neighbour" and "every other vertex" are the same set. So **no set
of vertices other than `i` closes `dD(p_i)`**, on 404 vertices of 405, however
many vertices are added.

<!-- a column measuring the definition rather than the geometry -->
There is an obvious third column — the boundary covered by the *whole* vertex set —
and it is **vacuous**: `dD(i)` is a subset of `D(i)`, so any union that
includes `i` covers it outright and reports 1.00 everywhere. It was in
`boundary_coverage.py` for one commit and is recorded here so it is not added back.

And this is exactly where the tree strategy dies. `dD(p_i)` is by definition a
subset of `D(p_i)`, so **a searcher standing at `i` watches the whole of its own
boundary** — while it is there, `D(p_i)` cannot be re-entered at all. The gap opens
the moment it leaves, which is precisely what a tree strategy does: `clear(v)`
returns with no searcher left anywhere in the subtree, on the argument that the
parent edge is the only way back in. On the graph that argument is sound. On
the surface the vertex was holding a boundary that its neighbours cannot between
them close, and nobody is holding it now.

So the guard graph records **who can watch a piece of a boundary**, never
**whether the boundary is closed**. A strategy that discharges every edge
obligation has still left a hole in every detection set it cleared, and an
arbitrarily fast evader needs one hole.

The gap is therefore not a tuning failure of GSST. **A detection set on this
geometry cannot be held by anybody except the searcher standing in it** — so any
strategy whose vertices are regions and whose guards are neighbours is unsound
here, whatever order it visits them in. What can be repaired, and at what
price, is [the next section](#what-would-have-to-change).

This is the structural reason `clearing/kolling.py` covers the **cell**
frontier with a set cover over vertices instead of discharging edges: a set
cover can hold a boundary that no single edge, and no set of edges, describes.

---

## What would have to change

Three repairs follow from the measurement, and they attack it at three
different places. Only the third is cheap.

### 1. Never *remove* a searcher — sound, and it costs the whole vertex set

If no boundary can be held by anyone but the vertex itself, the honest fix inside
the 2010 model is to never release a swept vertex. That clears: it is
`verify.all_at_once`, which leaves only the 4–18 m² no vertex can see. But the
release rule was the only thing keeping the team small, so the team becomes the
vertex set: **62 searchers at Christ Church against the cell planner's 12.**

This is not a straw man — it is roughly what A7 already does. Its traversal
holds nearly every visited node, it *does* clear the cell model, and it pays
three to five times the cell planner's team. The two results agree, and
together they say the cost of soundness under per-vertex thinking is a factor
of three to five at best.

### 2. Sample so that boundaries *are* closable — the upstream repair

The gap is a property of **where the vertices are**, and the sampler never
tried to close it: it places positions greedily until free space is *covered*,
which is an art-gallery objective and says nothing about whether each `D(p)`
has its boundary watched by the others. A sampler carrying the extra constraint

> for every vertex `i`, `dD(i)` must be covered by `∪ D(j)` over `j ≠ i`

would make the guard graph mean what the clearing argument assumes it means,
and the 2010 machinery would then be sound on it unchanged.

**Implemented and measured** — `e1-clearing-graph`, `e1/rim_closure.py` and
`scripts/run_rim_closure.py`: add vertices drawn from the open boundary itself
(a boundary cell nobody else sees is walkable, and a sensor standing on it sees it),
recompute the guard graph, re-run GSST, verify on cells.

| scene | vertices | edges | team, GSST regular | cell planner | cleared on cells |
|---|--:|--:|--:|--:|:--|
| | before → after | before → after | before → after | | before → after |
| hb-allen-centre | 20 → 140 | 84 → 4 261 | 7 → **62** | 10 | 940 m² → **0.0** |
| keble-college | 23 → 143 | 84 → 2 946 | 5 → **33** | 5 | 5 982 m² → **0.0** |
| observatory-quarter | 39 → 283 | 203 → 8 983 | 12 → **75** | 13 | 2 702 m² → **0.0** |
| christ-church | 57 → 447 | 245 → 11 216 | 8 → **67** | 12 | 7 692 m² → **0.0** |
| blenheim-palace | 36 → 227 | 167 → 5 029 | 9 → **54** | 8 | 8 258 m² → **0.0** |
| bodleian-library | 112 → 830 | 808 → 38 254 | 15 → **146** | 22 | 10 578 m² → **0.0** |

**The repair works, on all six scenes.** The published method goes from leaving
most of the site contaminated to clearing it outright — residual **0.0 m²**
under the *strict* test, with no tolerance applied, in 17 of the 18
scene × variant runs. The 2010 machinery is untouched; only the vertex set
changed.

**And it is not affordable.** It costs **6.2–7.8× the vertices** and
**6.0–9.7× the team**, against a cell-frontier planner that clears the same
scenes with 5–22 searchers. The mechanism is the paper's own second finding
arriving from a new direction: more vertices make the guard graph denser —
the Bodleian goes from 808 edges to 38 254 — and a graph strategy pays per
cycle edge held, not per area seen. **Repairing the sampling so the
abstraction becomes sound makes the instance the abstraction is expensive on.**

Four qualifications, all against the result:

* **The fixpoint does not close.** Each added vertex brings its own boundary. Round
  one is decisive — Christ Church goes from 2 987 open boundary cells to 318 for
  100 vertices — and then it stalls: eleven further rounds and 290 more
  vertices only reach 200. Runs stop at a 12-round budget with 60–476 cells
  still open on a third to a half of the vertices. **They cleared anyway**,
  because what is left is 2.4–19 m² spread over many boundaries. So full boundary closure
  was sufficient by a margin, not necessary, and the vertex counts above are an
  *upper* bound on what a smarter repair would need.
* **One run of eighteen still fails.** `regular_biased` on Christ Church leaves
  499.6 m² avoidable. The repair makes the method clear; it does not make every
  variant of it clear.
* **The baseline differs slightly.** This starts from Stage 5's raw ground
  sample; the Stage 6 graph the "before" column quotes has
  `repair_joint_residual` applied on top, which is why Keble reads 23 here and
  24 there, and Christ Church 57 against 62. The repaired graph therefore
  starts from *fewer* vertices than its own baseline — the result is not
  flattered by the difference.
* **The tolerance in the repair runner is weaker than Stage 6's** — with no air
  family present, `coverable` is the ground union alone, so an air-only cell is
  classified uncoverable rather than avoidable. Every claim above is the strict
  `cleared`, which that asymmetry cannot touch.

### 3. Hold the frontier of the cleared *region*, not the boundaries of its parts

> **THE CODE THIS SECTION MEASURES IS NO LONGER IN THE REPOSITORY, and the
> numbers are kept because they were measured.** `clearing/kolling.py` -- a
> GRAPH-CLEAR label over a maximum spanning forest, planning against a greedy
> cover of the cell frontier -- was removed when this repository narrowed to
> Kolling et al. (2010) as published. It was never the 2010 paper's planner
> and said so in its own docstring; `e1-clearing-graph` has no counterpart for
> it; and keeping a second, better planner around as the thing the published
> method is quietly compared against was the arrangement that made the
> comparison easy to misread.
>
> **What goes with it is a real comparison, and this is the cost.** It was
> this repository's only same-scene, same-vertex-set demonstration that a
> *cell*-frontier planner clears what a *graph* strategy does not, at 5-13
> searchers against 62. Nothing here replaces that. What does exist now is the
> comparison between the published method read ATOMICALLY and the same method
> DRIVEN, which is the one the geometry turns on and which
> `scripts/run_gsst.py --drive` measures.
>
> The code, the schedules it wrote (`examples/*.clearing.yaml`) and the table
> below are recoverable at commit `a7e0cbf`. The table below was measured on
> the RELEASED vertex set, before the repairs described above; it has not been
> re-measured on `repaired-ground` and must not be quoted as though it had.

The reason it wins is arithmetic rather than cleverness. When two cleared detection sets touch, the shared part
of their boundaries stops being a boundary — but per-vertex thinking pays for it
anyway. Measured at each scene's peak step, over the vertices whose detection
sets lie wholly inside the cleared region:

| scene | Σ boundary cells of the cleared vertices | frontier of their union | cancelled |
|---|--:|--:|--:|
| blenheim-palace | 14 791 | 1 852 | 87% |
| bodleian-library | 93 393 | 2 228 | **98%** |
| christ-church | 23 375 | 368 | **98%** |
| hb-allen-centre | 8 649 | 176 | **98%** |
| keble-college | 5 344 | 991 | 81% |
| observatory-quarter | 18 822 | 751 | 96% |

**81–98% of the boundary a per-vertex rule would guard is interior.** The
region frontier is 2–19% of the sum of the boundaries, and it is coverable — which is
why the cell planner cleared Christ Church with 12 searchers where holding every
boundary would need 62.

That measurement is about the GEOMETRY and survives the planner that made it:
the boundaries of a set of cleared detection sets cancel against each other whoever
is doing the clearing. What does not survive is the 12.

<callout>
<b>All three are now measured, and they agree on the shape of the answer.</b>
(1) never lift: sound, costs the whole vertex set. (2) close the boundaries: sound,
clears, costs 6–9× the team. (3) hold the region frontier: clears at 5–13
searchers, and is what this project already does. The 2010 formulation can be
made correct on this geometry in two different ways, and both of them cost
roughly an order of magnitude — which is the argument for (3) stated as a
measurement rather than as a preference.
</callout>

---

## What was not run

**The sensing-range sweep.** The paper's second finding — that on complex
terrain a *longer* sensing range needs *more* searchers, because it complicates
the detection-set boundary rather than enlarging it — needs the same scenes
exported at several values of `s_r`. These ship at `R = 30 m` only, and
re-exporting requires the survey clouds and the raycaster in
`e1-clearing-graph` (`scripts/export_from_e1.py --R ...`), whose `data/` is
empty here. It is the most interesting thing left undone, and the boundary-coverage
table above predicts its mechanism would show up here too.

**Heterogeneity.** No UAV, as in the rest of this repository. The 2010 paper is
homogeneous and this is exactly its setting.

**Cell-level components.** Guard-graph components are cleared in sequence, which
is sound on the graph, where a component has no edges leaving it. Two graph
components can still be adjacent on the walkable surface. Where that happens it
is one more contributor to the residual above, and it is not separated out.

<!-- separated out upstream on 2026-08-25 -->
**It happens.** Measured in `e1-clearing-graph` on the same six sites
(`docs/guard_graph_components.md`): christ-church's walkable surface is one
connected piece of 8 133 m² and its ground guard graph is in three, the whole
split carried by a 0.40 m² gate and a 0.32 m² step-over that no vertex watches.
The two regions are 0.9 m of walking apart, so the sequential argument is false
there. Placing one vertex per threshold connects the graph on all six scenes
for 25–85% more vertices — and does not clear them, because the boundary gap above
is untouched by it. The same probe found E2's speck filter deleting those
thresholds, which is why neither the graph nor the cell verifier could see the
leak; that is fixed upstream and it costs blenheim-palace its one clearing
result.

---

## Agreement with `e1-clearing-graph`, and the one place it stops

The published method is implemented on both sides of the fence -- here and as
`e2.gsst` upstream -- and the two are held to agreeing rather than to looking
similar. At seed 1 over 100 forests, on all six scenes in all three variants:

* the **schedules are byte-identical** -- same forests, same steps, same team.
  That is not a statistical claim about 100 draws; both sides draw
  `np.random.default_rng(1)` over neighbour lists sorted the same way, so a
  single differing neighbour would show up as a differing schedule. (It did:
  this repository used to build its adjacency in edge order, which makes a seed
  mean something different. `gsst.adjacency` now sorts, and says why.)
* the **guard graphs are identical**, edge for edge and kind for kind. The
  export builds it with `e1.graph.build_graph` and checks the result against
  `e2.gsst.ground_guide` before writing anything.
* the **atomic verdicts agree**: exact zeros match exactly, and the non-zero
  residuals agree to within 0.1--0.8 %.

**Where they stop agreeing is the definition of "nothing can see this", and it
is a difference in the PROBLEM rather than in the code.** E1's scene is
bipartite; a cell only a UAV can see is *coverable* there. This scenario has no
UAV, so the same cell is coverable by nothing, and the speck filter -- which
takes connected fragments of the uncoverable set under 4 m² out of the evader
space -- therefore takes a little more out here:

| scene | extra ground excluded | E1's driven team | this repository's | |
|---|--:|--:|--:|--:|
| HB Allen Centre | 10.0 m² | 13 | 13 | |
| Keble College | 8.8 m² | 8 | 8 | |
| Blenheim Palace | 14.2 m² | 10 | 10 | |
| Observatory Quarter | 17.9 m² | 17 | 18 | +1 |
| Christ Church | 20.5 m² | 12 | 13 | +1 |
| Bodleian Library | 76.8 m² | 25 | 29 | +4 |

The two columns are ordered the same way, and the attribution is not inferred
from that ordering: injecting E1's exclusion mask into this repository's
executor and changing nothing else returns E1's answer exactly -- Observatory
17 with no machine spent, against 18 with seven spent on the mask computed here.

Which mask is right depends on which problem is being posed, and for a
ground-only scenario it is this one. The consequence is worth stating plainly:
**the teams here are the same as E1's on three scenes and higher on three, and
where they are higher it is because this scenario is harder, not because the
method did worse.**

---

## Reproduction

```bash
python scripts/run_gsst.py --trees 100          # Finding 1, atomically
python scripts/run_gsst.py --drive --exact      # and driven, on the clock
python scripts/boundary_coverage.py                  # the boundary measurement, seconds
python -m pytest tests/test_gsst.py tests/test_execute.py
```

Per-scene schedules and every metric land in `examples/<scene>.gsst.yaml`,
which carries both verdicts for each variant.

`--exact` floods once per arrival rather than every `dt`. Under the no-sliding
rule that is not an approximation of `--dt 1 --sub 8`: with no credit for a
moving sensor, `O(t)` changes only when a mover arrives, so its intersection
over an interval IS its value at the interval's left end. The two agree on
every scene and `tests/test_execute.py` asserts it; `--exact` is a great deal
cheaper, and the sampled form is kept because it is the referee
`e1-clearing-graph` judges its own rows with.

"""Verify a driven strategy on a wall clock, with no credit for the drive.

WHY A SECOND VERIFIER
---------------------
`clearing.verify` asks the atomic question: between two steps the whole team
teleports, so there is no moment at which anybody is in transit and no interval
to charge for. That is the model Kolling et al. (2010) work in and it is the
model this repository has always reported.

It is also optimistic in a way this geometry punishes. A step is not an
instant: somebody drives, and for the length of that drive the frontier is held
by whoever stayed behind. On the repaired vertex sets the published method's
atomic verdict and its clock verdict are not close -- Christ Church leaves
10 990 m2 atomically and 0.00 driven -- and the difference is entirely in which
question was asked.

THE RULE, AND WHY IT IS THE STRICT ONE
---------------------------------------
`rule="instant"` samples the drive every `dt` seconds and asks nothing about
the OPEN interval between two samples: it credits the guards standing at `tau`
with having stood there since `tau - dt`. An evader that is arbitrarily fast is
arbitrarily fast inside `dt` too, so that is the atomic model's optimism shrunk
rather than removed.

`rule="strict"` charges for the interval. A cell survives an interval only if
it is observed at EVERY instant of it:

    M_I  :=  intersection over tau in I of O(tau)
    C    :=  flood(C \\ M_I,  ~M_I)      the open interval
    C    :=  flood(C \\ O(t_end), ~O)    the instant at its end

Both are sound on their own -- a cell watched throughout an interval cannot
hold an evader at the end of it, a cell watched at an instant cannot hold one
at that instant -- and the composition is sound because the second is applied
to the output of the first.

WITH NO SWEEP THE STRICT RULE IS EXACT, WHICH IT IS NOT UPSTREAM
-----------------------------------------------------------------
Under the 2010 prohibition a machine in transit is credited with nothing. So
within one transition `O(tau)` changes only when a mover ARRIVES: it is a step
function of time, non-decreasing, with one jump per arrival. Its intersection
over an interval is therefore its value at the interval's left end, computed
and not approximated.

Upstream this is not true -- `e2.transit` credits the driver with what it sees
along the leg, `M_I` is approximated from `sub` sub-samples, and the rule is
optimistic by an amount only refinement can bound. Here `exact=True` floods
once per inter-arrival interval and `dt` and `sub` stop being tolerances.

`dt` and `sub` are kept anyway, and are the default, because a number that
agrees with `e1-clearing-graph` at `dt = 1 s, sub = 8` is a number that has
been checked against the referee every other row in that project is judged by.
`exact=True` must agree with them, and `tests` asserts it does.

WHAT IT DOES NOT DO
-------------------
It does not plan. It is handed a strategy and a roster and believes only those
plus the scene's own detection sets and adjacency -- it is told nothing about
guard regions, spanning forests or who chose the order. A strategy that clears
in `clearing.execute`'s own bookkeeping and not here has a bug in the planner,
not a disagreement about geometry.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse import csgraph

from .roster import Roster
from .scene import Scene
from .verify import Strategy, _graph, absorb_fragments, flood


def component_areas(scene: Scene, cells: np.ndarray) -> np.ndarray:
    """Areas of the connected components of `cells`, largest first."""
    idx = np.asarray(cells, dtype=np.int64)
    if idx.size == 0:
        return np.zeros(0, dtype=float)
    mask = np.zeros(scene.n_cells, dtype=bool)
    mask[idx] = True
    a, b = scene.evader_edges
    keep = mask[a] & mask[b]
    g = sparse.coo_matrix(
        (np.ones(int(keep.sum()), dtype=np.uint8), (a[keep], b[keep])),
        shape=(scene.n_cells,) * 2).tocsr()
    _, lab = csgraph.connected_components(g + g.T, directed=False)
    sizes = np.bincount(lab[idx]) * scene.cell_area
    return np.sort(sizes[sizes > 0])[::-1]


def timeline_propagate(scene: Scene, strategy: Strategy, roster: Roster, *,
                       dt: float = 1.0, sub: int = 8, rule: str = "strict",
                       absorb_m2: float = 1.0, exact: bool = False,
                       max_substeps: int = 200_000,
                       record=None, verbose: bool = False) -> dict:
    """Every `dt` seconds, all machines at once; a machine in transit sees nothing.

    Each machine is placed at the arc length its own quoted travel time has
    carried it to -- but since it is credited with nothing until it gets there,
    the only thing that matters about that position is whether it has arrived.
    The transition clock is the slowest mover's, because a step ends when the
    last machine is in place; departures are simultaneous.
    """
    if dt <= 0:
        raise ValueError("dt must be positive")
    if rule not in ("instant", "strict"):
        raise ValueError(f"unknown rule {rule!r}, expected instant or strict")
    strict = rule == "strict"
    if strict and sub < 1:
        raise ValueError("sub must be at least 1: the interval needs both ends")
    if len(roster.steps) != len(strategy.steps):
        raise ValueError(f"roster has {len(roster.steps)} steps and the "
                         f"strategy {len(strategy.steps)}")
    strategy.validate(scene)
    g = _graph(scene)

    C = ~scene.excluded
    C, _ = absorb_fragments(scene, C, absorb_m2)

    substeps = 0
    regained: list[int] = []
    trace: list[int] = []
    exposure: list[dict] = []
    seconds_elapsed = 0.0

    def observed_of(vertices) -> np.ndarray:
        m = np.zeros(scene.n_cells, dtype=bool)
        for v in vertices:
            m[scene.detection[int(v)]] = True
        return m

    #: `O(tau)` by the set of vertices standing, not by `tau`. Under the
    #: no-sliding rule `O` is a step function of time with one jump per
    #: arrival, so a whole transition has at most `movers + 1` distinct masks
    #: however finely it is sampled -- and the strict rule at `dt = 1 s` with
    #: `sub = 8` asks for one nine times a second. Rebuilding the union each
    #: time was most of the verifier's cost and none of its content.
    _mask_cache: dict[tuple[int, ...], np.ndarray] = {}

    def team_mask(legs, tau: float) -> np.ndarray:
        """`O(tau)`: what the team observes at one instant of one transition.

        Pure. A mover that has not arrived contributes nothing -- that is the
        whole of the no-sliding rule, and it is why no geometry beyond the
        travel time is needed to evaluate this.
        """
        held = []
        for m in legs:
            if m.stays:
                held.append(m.to)
            elif m.entered:
                continue                        # walking on site: blind
            elif (not np.isfinite(m.seconds) or m.seconds <= 0
                  or tau >= m.seconds):
                held.append(m.to)
        key = tuple(sorted(held))
        got = _mask_cache.get(key)
        if got is None:
            got = observed_of(key)
            _mask_cache[key] = got
        return got

    def advance(mask: np.ndarray, kind: str, t: int, tau: float) -> None:
        nonlocal C, substeps
        C = flood(scene, C & ~mask, ~mask, graph=g)
        C, _ = absorb_fragments(scene, C, absorb_m2)
        substeps += 1
        if record is not None:
            record({"kind": kind, "step": t, "tau": tau,
                    "clock": seconds_elapsed + tau,
                    "contaminated": C.copy(), "observed": mask})

    for t, step in enumerate(strategy.steps):
        if t:
            legs = roster.steps[t]
            _mask_cache.clear()
            clock = max((m.seconds for m in legs
                         if not m.stays and np.isfinite(m.seconds)),
                        default=0.0)
            before = int(C.sum())
            worst = before
            if clock > 0.0:
                if exact:
                    # The only instants at which O(tau) changes are arrivals.
                    # Flooding at each of them, plus tau = 0, IS the strict
                    # rule: the intersection over an inter-arrival interval is
                    # the mask at its left end, and re-applying that mask at
                    # the right end is idempotent.
                    taus = sorted({0.0} | {
                        float(m.seconds) for m in legs
                        if not m.stays and m.entered is False
                        and np.isfinite(m.seconds) and 0.0 < m.seconds <= clock})
                    if substeps + len(taus) > max_substeps:
                        raise ValueError(
                            f"this strategy needs more than {max_substeps:,} "
                            "sub-steps; raise max_substeps deliberately")
                    for tau in taus:
                        advance(team_mask(legs, tau), "interval", t, tau)
                        worst = max(worst, int(C.sum()))
                else:
                    n = int(np.ceil(clock / dt))
                    if substeps + (2 * n if strict else n) > max_substeps:
                        raise ValueError(
                            f"dt={dt} would need more than {max_substeps:,} "
                            "sub-steps on this strategy; raise max_substeps "
                            "deliberately or coarsen dt")
                    prev = team_mask(legs, 0.0) if strict else None

                    def interval_mask(lo, hi, at_lo):
                        # `&` allocates, so the cached arrays are never written
                        M, span = at_lo, hi - lo
                        for j in range(1, sub + 1):
                            other = team_mask(legs, lo + span * (j / sub))
                            if other is not M:
                                M = M & other
                        return M

                    for s in range(1, n):        # tau = clock is the step itself
                        tau = s * dt
                        mask = team_mask(legs, tau)
                        if strict:
                            advance(interval_mask(tau - dt, tau, prev),
                                    "interval", t, tau)
                        advance(mask, "drive", t, tau)
                        prev = mask
                        worst = max(worst, int(C.sum()))
                    if strict:
                        # The tail, from the last sample to the arrival, is a
                        # shorter interval and is charged the same way.
                        # Skipping it would leave the one interval in which
                        # every mover reaches its destination -- the moment the
                        # frontier changes hands -- uncharged.
                        lo = (n - 1) * dt
                        advance(interval_mask(lo, clock, prev),
                                "interval", t, clock)
                        worst = max(worst, int(C.sum()))
                seconds_elapsed += clock
            regained.append(max(0, int(C.sum()) - before))
            exposure.append({"transition": t - 1, "seconds": round(clock, 1),
                             "peak_contaminated": worst,
                             "net_given_back": regained[-1]})

        advance(scene.observed(step), "step", t, 0.0)
        trace.append(int(C.sum()))
        if verbose and (t + 1) % 20 == 0:
            print(f"  [clock] step {t + 1}/{len(strategy)}: contaminated "
                  f"{int(C.sum()):,} at t={seconds_elapsed:,.0f}s", flush=True)

    left = np.flatnonzero(C).astype(np.int64)
    avoidable = np.flatnonzero(C & scene.coverable).astype(np.int64)
    comps = component_areas(scene, avoidable)
    a_min = scene.speck_max_area_m2
    over = comps[comps >= a_min]
    return {
        "dt": dt, "rule": rule, "sub": int(sub) if strict else None,
        "exact": bool(exact), "sliding_credited": False, "substeps": substeps,
        "mission_seconds": round(seconds_elapsed, 1),
        "cleared": bool(left.size == 0),
        "residual_cells": int(left.size),
        "residual_m2": round(scene.area(left), 2),
        "residual_avoidable_m2": round(scene.area(avoidable), 2),
        "cleared_within_tolerance": bool(over.size == 0),
        "a_min_m2": float(a_min),
        "avoidable_components": int(comps.size),
        "avoidable_components_over_a_min": int(over.size),
        "avoidable_largest_component_m2": (round(float(comps[0]), 2)
                                           if comps.size else 0.0),
        "transitions": len(regained),
        "transitions_giving_area_back": int(sum(1 for q in regained if q)),
        "cells_given_back": int(sum(regained)),
        "contaminated_after_step": trace,
        "exposure": exposure,
    }

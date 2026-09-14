"""Execute an atomic strategy one machine at a time, under the no-sliding rule.

WHY AN ATOMIC STRATEGY NEEDS EXECUTING AT ALL
----------------------------------------------
`clearing.gsst` produces a sequence of SETS of occupied vertices, and the model
behind it moves every searcher between two of those sets in no time. Real
machines do not. Between step `t` and step `t + 1` somebody drives, and while
it drives it is not standing anywhere -- so the team that is left is smaller
than either step, and the frontier one of them was holding may be open for the
length of a drive.

That gap is not a detail on this geometry. Judged atomically on the repaired
vertex sets, the published method leaves 1 194 m2 at HB Allen and 10 990 at
Christ Church; the same strategies, driven one machine at a time and judged on a
clock, clear both. Splitting a step into single legs is what closes it: while
one machine drives, every other one is still standing where the step wanted it.

WHAT IS AND IS NOT DECIDED HERE
--------------------------------
This module does not decide whether its strategy clears. It orders the legs
and hands back a strategy and the roster that walks it; `clearing.clock` is the
verdict, at a fine `dt`, and it is independent of everything below. A planner
that graded itself would be grading its own sampling rate.

NO SLIDING, AND WHY THAT MAKES THIS CHEAP
------------------------------------------
Kolling et al. (2010) forbid a driving searcher to sweep a detection-set
boundary:
that formulation has no model of what a moving sensor sees, so only `place` and
`remove` survive. This module keeps the prohibition rather than reporting
against it -- a drive is admissible only when the STANDING team covers the
frontier by itself, and the mover is credited with nothing between its two
posts.

The prohibition is what lets this live in this repository. Crediting the driver
would need the raycaster, the occupancy volume and the survey data, none of
which are here; crediting it with nothing needs only the detection sets and the
travel times `scenes/` already ships. `e1-clearing-graph` runs both, and the
sliding column costs one machine on one of the six scenes.

`spend_machines` is the fallback the paper itself supplies. When no searcher on
site can make a drive without opening the frontier, `place` a new one on the
destination and leave the one that would have leaked where it stands. It walks
in blind, which the standing team can cover precisely because nobody has left.
It costs one machine, permanently, and `machines_spent` counts them.

WHICH MACHINE DRIVES IS A CHOICE, AND IT IS MADE HERE
------------------------------------------------------
A step names a SET of vertices, not an assignment of machines to them.
`clearing.roster.assign` picks one by least total travel, which knows nothing
about what a departure opens. This picks it by the same test that decides the
order: every (machine on site, post still to be filled) pair is scored by how
much frontier its drive would leave unwatched, and the least-leaking one goes.
It changes no vertex, no step and no fleet -- the same machines go to the same
set of posts -- only which of two interchangeable machines drives.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .roster import Leg, Roster, assign
from .scene import Scene
from .verify import Strategy, _graph, absorb_fragments, flood, frontier


@dataclass
class ExecReport:
    """A driven strategy: the legs, who made them, and what each one cost."""

    strategy: Strategy
    roster: Roster
    #: per move: the mover, where it went, what it cost, what it conceded
    legs: list[dict] = field(default_factory=list)
    fleet_curve: list[int] = field(default_factory=list)
    notes: dict = field(default_factory=dict)

    @property
    def fleet(self) -> int:
        return int(self.notes.get("fleet", 0))

    def metrics(self) -> dict:
        con = [m["conceded_m2"] for m in self.legs]
        return {
            "legs": len(self.strategy),
            "fleet": self.fleet,
            "legs_holding": int(sum(1 for q in con if q <= 0.0)),
            "legs_conceding": int(sum(1 for q in con if q > 0.0)),
            "conceded_m2_total": round(float(sum(con)), 2),
            "conceded_m2_worst": round(float(max(con)) if con else 0.0, 2),
            "walking_s": round(self.roster.makespan_s, 1),
            **self.notes,
        }


def sequentialise(scene: Scene, strategy: Strategy, *,
                  spend_machines: bool = True, absorb_m2: float = 1.0,
                  verbose: bool = False) -> ExecReport:
    """Drive an atomic strategy, one machine at a time, crediting no sliding.

    The order within a step is greedy by the cheap test the admission rule
    uses: among the legs still to be made, take the one that leaves the fewest
    frontier cells unwatched for the duration of its own drive, ties broken by
    the shorter drive.

    There is no sampling rate here and there is deliberately none. Upstream the
    drive is cut into `k` samples because the mover's detection set legs with
    it; with no sliding the observed set is constant for the whole drive, so one
    flood reaches the fixed point and any further pass is idempotent. The
    verdict is `clearing.clock`'s in either case.

    EVERY ROBOT ON SITE IS CREDITED, NOT ONLY THE STEP'S OWN. An atomic step
    lists the vertices it is USING; a machine the step has no work for is still
    standing where it last stopped and still sees what it sees. The atomic
    model had no reason to count it, the operator is paying for it either way,
    and the fleet metric counts it. So the observed set here is over every
    machine on site, which is strictly more than the atomic strategy was verified
    with -- and the verdict is recomputed from scratch downstream, so this
    cannot quietly become a weaker claim.
    """
    strategy.validate(scene)
    ros0 = assign(scene, strategy)
    g = _graph(scene)

    T = np.asarray(scene.travel_seconds, dtype=float)
    D = np.asarray(scene.travel_metres, dtype=float)

    at: list[int | None] = [None] * ros0.n_machines
    spent = 0
    steps: list[tuple[int, ...]] = []
    rsteps: list[list[Leg]] = []
    records: list[dict] = []
    fleet_curve: list[int] = []
    where: list[list[int]] = []      # per move, every machine's vertex or -1

    C = ~scene.excluded
    C, _ = absorb_fragments(scene, C, absorb_m2)

    def occupied() -> tuple[int, ...]:
        return tuple(sorted(v for v in at if v is not None))

    def standing(skip: int | None = None) -> np.ndarray:
        m = np.zeros(scene.n_cells, dtype=bool)
        for i, v in enumerate(at):
            if v is not None and i != skip:
                m[scene.detection[v]] = True
        return m

    def commit(rid: int, to: int, kind: str) -> None:
        nonlocal C
        frm = at[rid]
        entered = frm is None
        s = 0.0 if entered else float(T[frm, to])
        me = 0.0 if entered else float(D[frm, to])
        if not np.isfinite(s):
            s, me = 0.0, 0.0
        legs = [Leg(i, v, v, 0.0, 0.0)
                 for i, v in enumerate(at) if v is not None and i != rid]
        legs.append(Leg(rid, to if entered else frm, to, s, me,
                          entered=entered))

        before = ~C
        # The drive. A machine walking on to the site is blind, and so -- under
        # the 2010 prohibition -- is one driving between two posts: for the
        # verifier the two are the same event, and the standing team has to
        # hold alone for as long as it lasts.
        others = standing(rid)
        # ONCE, not once per sample. With no sliding the observed set does not
        # change while a machine drives, and the flood is a fixed point of it:
        # the second pass returns the components of `~others` that still hold a
        # seed, which is what the first pass already left. Upstream this loop
        # runs `k` times because there the mask legs with the driver.
        C = flood(scene, C & ~others, ~others, graph=g)
        C, _ = absorb_fragments(scene, C, absorb_m2)
        at[rid] = to
        O = scene.observed(occupied())
        C = flood(scene, C & ~O, ~O, graph=g)
        C, _ = absorb_fragments(scene, C, absorb_m2)

        n = sum(1 for v in at if v is not None)
        # Snapshotted rather than reconstructed from the move list: machines do
        # not enter in id order -- the least-leaking candidate may be any free
        # slot -- so "sort the legs by machine" would not put machine i at index i.
        where.append([v if v is not None else -1 for v in at])
        steps.append(occupied())
        rsteps.append(sorted(legs, key=lambda m: m.machine))
        records.append({"kind": kind, "machine": int(rid),
                        "frm": None if entered else int(frm), "to": int(to),
                        "seconds": round(s, 1),
                        "conceded_m2": round(float((before & C).sum())
                                             * scene.cell_area, 2),
                        "contaminated": int(C.sum()), "fleet": n})
        fleet_curve.append(n)

    def score(rid: int, to: int, F: np.ndarray) -> tuple[int, float]:
        """Frontier cells left unwatched while this machine makes this drive."""
        frm = at[rid]
        if frm is None:
            # a fresh machine walks in blind, so it holds nothing on the way and
            # takes nothing away: admissible exactly when the standing team
            # already covers the frontier by itself
            return int((F & ~standing()).sum()), float("inf")
        return int((F & ~standing(rid)).sum()), float(T[frm, to])

    for t, want in enumerate(strategy.steps):
        want = [int(v) for v in want]
        while True:
            todo = [v for v in want if v not in at]
            if not todo:
                break
            F = frontier(scene, ~C)
            # A machine already standing on a post this step wants is not a
            # candidate: moving it would undo the step.
            cand = [(rid, v) for v in todo for rid, u in enumerate(at)
                    if u is None or u not in want]
            scored = sorted(((*score(rid, v, F), rid, v) for rid, v in cand),
                            key=lambda z: (z[0], z[1], z[2], z[3]))
            if not scored or (scored[0][0] > 0 and spend_machines):
                # `place`: nobody on site can make this drive without opening
                # the frontier, so bring one more searcher in and leave the one
                # that would have leaked where it stands.
                at.append(None)
                spent += 1
                commit(len(at) - 1, todo[0], "enter")
                continue
            leak, _s, rid, to = scored[0]
            commit(rid, to, "enter" if at[rid] is None
                   else ("hold" if leak == 0 else "concede"))
        if verbose and (t + 1) % 10 == 0:
            print(f"    [seq] atomic step {t + 1}/{len(strategy.steps)}: "
                  f"{len(steps)} legs, contaminated {int(C.sum()):,}",
                  flush=True)

    ros = Roster(steps=rsteps, entry=ros0.entry, n_machines=len(at),
                 positions=[row + [-1] * (len(at) - len(row)) for row in where],
                 # every machine on site is holding a post in a driven strategy,
                 # so nobody is parked: the distinction belongs to the atomic
                 # model, where a step may want fewer vertices than the fleet
                 parked=[[] for _ in rsteps],
                 step_seconds=[max((m.seconds for m in st if not m.stays),
                                   default=0.0) for st in rsteps])
    return ExecReport(
        Strategy(steps, "driven one machine at a time, no sliding"), ros,
        legs=records, fleet_curve=fleet_curve,
        notes={"fleet": len(at), "machines_spent": int(spent),
               "atomic_searchers": strategy.n_searchers,
               "absorb_m2": float(absorb_m2),
               "sliding_credited": False,
               "order": "atomic-sequentialised-no-sliding"})

"""Ground-platform clearing scenarios, and Kolling et al. (2010) on them.

`gsst` is the published method: the no-sliding Barriere label recursion over
random depth-first spanning forests, converted back to a graph strategy by
GSST's cycle-edge rule. `verify` judges a strategy atomically, the way the
paper's own model does; `execute` and `clock` judge it as something machines
drive, keeping the paper's prohibition on sweeping while in transit.

The GRAPH-CLEAR + cell-frontier baseline that used to sit here as `kolling` is
gone, and `docs/gsst.md` records what went with it.

ONE WORD PER THING. A **searcher** is the paper's token: it is placed on a
vertex and removed from it, and has no identity and no position in between. A
**machine** is a physical platform with an id and a route, which is what
`roster`, `execute` and `clock` deal in and what the paper has no model of. A
**platform** is the ground robot the scenario is posed for -- its sensor height,
range and speed. Nothing here is called a "robot". See `docs/glossary.md`.
"""

from . import clock, execute, gsst
from .scene import Scene, available, load
from .verify import Strategy, all_at_once, propagate

__all__ = ["Scene", "Strategy", "all_at_once", "available", "clock", "execute",
           "gsst", "load", "propagate"]

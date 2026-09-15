"""Local backend for the 3D-scene strategy tool.

Serves the (extended) viewer as static files and one JSON endpoint that
computes a DP/Greedy strategy on a real scene. Run with:

    python -m strategy_service.server

then open http://localhost:8000 . Purely additive: nothing here is imported
by, or changes the behaviour of, the per-approach `streamlit run test.py`
2D-grid workflow -- see strategy_service/adapter.py and the patched
Searcher.py's `travelTime=None` default for how that is kept intact.
"""

from __future__ import annotations

import base64
import json
import os
import sys
import time
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEARING_SCENES_DIR = os.path.join(REPO_DIR, "clearing-scenes")
VIEWER_DIR = os.path.join(CLEARING_SCENES_DIR, "viewer")

# One small JSON file per scene, holding whatever hotspot layout was last
# saved for it from the viewer -- the "meaningful, hand-placed priors per
# scene" 3DTest.py's Monte Carlo comparisons read instead of an arbitrary
# placeholder layout (see 3DTest.py's default3DHotspots()).
SCENE_PRIORS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scene_priors")

if CLEARING_SCENES_DIR not in sys.path:
    sys.path.insert(0, CLEARING_SCENES_DIR)

import clearing  # noqa: E402

from strategy_service import adapter, approaches  # noqa: E402

app = FastAPI(title="clearing-scenes strategy service")

_scene_cache: dict[str, "clearing.Scene"] = {}


def _get_scene(name: str) -> "clearing.Scene":
    if name not in _scene_cache:
        if name not in clearing.available(scenes_dir=os.path.join(CLEARING_SCENES_DIR, "scenes")):
            raise HTTPException(404, f"unknown scene {name!r}")
        _scene_cache[name] = clearing.load(name, scenes_dir=os.path.join(CLEARING_SCENES_DIR, "scenes"))
    return _scene_cache[name]


class Hotspot(BaseModel):
    # A clicked point on the viewer's surface, in absolute Scene coordinates
    # (the viewer's own render points are a differently-indexed, decimated
    # cloud -- see adapter.viewer_surface_cell_map -- so a raw point index
    # from the browser is not a valid Scene cell id; the browser sends the
    # 3D position it actually clicked, and the nearest real cell is looked
    # up here instead).
    x: float
    y: float
    z: float
    category: Literal[0, 1, 2]


class Stopping(BaseModel):
    mode: Literal["time", "trees"]
    value: float = Field(gt=0)


class PriorsRequest(BaseModel):
    scene: str
    hotspots: list[Hotspot] = []
    prior_l: float = Field(default=3.0, gt=0)
    prior_radius_m: float = Field(default=5.0, gt=0)
    epsilon: float = Field(default=0.05, ge=0, le=1)


class RunRequest(PriorsRequest):
    approach: Literal["dp", "dp-blabel", "greedy", "greedy-blabel"]
    start_vertices: list[int] = []
    available_robots: int = Field(gt=0)
    stopping: Stopping


class HotspotConfig(PriorsRequest):
    start_vertices: list[int] = []


@app.get("/api/scenes")
def list_scenes():
    return {"scenes": clearing.available(scenes_dir=os.path.join(CLEARING_SCENES_DIR, "scenes"))}


@app.get("/api/approaches")
def list_approaches():
    return {key: approaches.APPROACH_LABELS[key] for key in approaches.APPROACH_DIRS}


# Same conversion as each approach's own Streamlit test.py: pick sigma so the
# hotspot's true location falls inside the marked radius in 50% of cases.
def _sigma_from_radius(radius_m: float) -> float:
    return radius_m / np.sqrt(2 * np.log(50))


def _compute_priors(scene, req: PriorsRequest) -> np.ndarray:
    hotspot_cells = [
        (adapter.nearest_cell(scene, (h.x, h.y, h.z)), h.category) for h in req.hotspots
    ]
    sigma = _sigma_from_radius(req.prior_radius_m)
    return adapter.compute_priors(scene, hotspot_cells, req.prior_l, sigma, req.epsilon)


def _quantize_priors(priors: np.ndarray, remap: np.ndarray | None = None) -> dict:
    """Priors as a base64-packed uint16 array -- a heatmap only needs relative
    intensity, and packing keeps a quarter-million-cell scene's worth of
    priors a few hundred KB instead of a multi-MB JSON float array.

    Priors PART 1's own math (GraphBuilder.py) is a Gaussian mixture blended
    with a small uniform background: on a real scene's fine cell grid, a
    hotspot's Gaussian bump covers a tiny fraction of the cells, so almost
    every cell sits within a whisker of the uniform floor and only a handful
    sit near the peak -- linear min/max scaling puts ~everything at one end
    of the ramp and a single pixel at the other, which reads as "barely any
    colour, and no visible spread". Scaling in LOG space instead turns that
    exponential falloff back into something close to linear-in-distance (a
    Gaussian's log is a parabola in distance), which is what actually renders
    as a smooth, visible gradient around each hotspot. `lo` is a low
    percentile rather than the true min so a handful of unreachable/underflow
    cells can't blow out the range; `hi` is left as the true max since it is
    the one cell that should read as fully "hot", and clipping it earlier
    only concentrates more cells at the saturated top end (checked
    empirically against 99.9/99.5 percentile alternatives).
    """
    positive = priors[priors > 0]
    floor = float(positive.min()) if positive.size else 1.0
    logp = np.log(np.maximum(priors, floor * 1e-3))
    lo = float(np.percentile(logp, 1))
    hi = float(logp.max())
    rng = hi - lo
    if rng > 0:
        t = np.clip((logp - lo) / rng, 0, 1)
        q = np.round(t * 65535).astype("<u2")
    else:
        q = np.zeros(priors.shape[0], dtype="<u2")
    # The viewer's own surface point cloud is a differently-sized,
    # differently-indexed decimation of the real cells (see
    # adapter.viewer_surface_cell_map) -- reindex onto it here so the caller
    # can drop `q[i]` straight onto its `S.surf` point `i` with no mapping of
    # its own to do.
    if remap is not None:
        q = q[remap]
    return {
        "q": base64.b64encode(q.tobytes()).decode("ascii"),
        "min": float(priors.min()),
        "max": float(priors.max()),
    }


def _viewer_remap(scene, scene_name: str) -> np.ndarray:
    return adapter.viewer_surface_cell_map(scene, scene_name, os.path.join(VIEWER_DIR, "data"))


@app.post("/api/priors")
def preview_priors(req: PriorsRequest):
    scene = _get_scene(req.scene)
    priors = _compute_priors(scene, req)
    return {"priors": _quantize_priors(priors, _viewer_remap(scene, req.scene))}


def _hotspot_config_path(scene_name: str) -> str:
    # scene names are already a closed, known set (validated by _get_scene
    # against clearing.available()), so this never sees attacker-controlled
    # path segments in practice -- still resolved through _get_scene first
    # in both endpoints below rather than trusted blindly.
    return os.path.join(SCENE_PRIORS_DIR, f"{scene_name}.json")


@app.get("/api/hotspots/{scene_name}")
def load_hotspot_config(scene_name: str):
    _get_scene(scene_name)  # 404s on an unknown scene name
    path = _hotspot_config_path(scene_name)
    if not os.path.isfile(path):
        return {"hotspots": [], "start_vertices": [], "prior_l": None, "prior_radius_m": None}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@app.post("/api/hotspots/{scene_name}")
def save_hotspot_config(scene_name: str, req: HotspotConfig):
    _get_scene(scene_name)
    os.makedirs(SCENE_PRIORS_DIR, exist_ok=True)
    payload = {
        "hotspots": [h.model_dump() for h in req.hotspots],
        "start_vertices": req.start_vertices,
        "prior_l": req.prior_l,
        "prior_radius_m": req.prior_radius_m,
    }
    with open(_hotspot_config_path(scene_name), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return {"saved": True}


@app.post("/api/run")
def run_strategy(req: RunRequest):
    scene = _get_scene(req.scene)

    n_vertices = scene.n_vertices
    for v in req.start_vertices:
        if not (0 <= v < n_vertices):
            raise HTTPException(400, f"start vertex {v} out of range [0, {n_vertices})")
    if req.available_robots > n_vertices:
        raise HTTPException(400, "available_robots exceeds the number of vertices in the scene")

    priors = _compute_priors(scene, req)
    G, D, start_nodes, idx_to_vertex = adapter.build_graph(scene, req.start_vertices, priors)
    travel_time = adapter.travel_time_matrix(scene, idx_to_vertex)
    cellpriors = priors.reshape(-1, 1)

    graph_search = approaches.graph_search_for(req.approach)
    available_time = req.stopping.value if req.stopping.mode == "time" else None
    max_trees = int(req.stopping.value) if req.stopping.mode == "trees" else None

    # FHPE_SA's own hardcoded default horizon is tuned for the small
    # synthetic Streamlit grids, where an edge takes a handful of steps --
    # far too small next to a real scene's walking-SECOND edge weights
    # (tens of seconds between guard-graph vertices here). Left at that
    # default, a robot has no reachable neighbour within the horizon at
    # all, so the fallback used whenever clearance is infeasible always
    # produced zero real moves (a 0-second "strategy"). Size it off the
    # scene's own edges instead, so it always has real room to move --
    # FHPE_SA's separate maxHops cap is what keeps this tractable even
    # though horizon now spans several hops.
    edge_times = [e.time for e in G.edges.values()]
    fhpe_horizon = float(np.median(edge_times)) * 4 if edge_times else 60.0

    t0 = time.time()
    strategy, _best_tree, checked_trees, fitness = graph_search(
        G, available_time, req.available_robots, start_nodes,
        None, None, None, cellpriors, D,
        maxTrees=max_trees, travelTime=travel_time, horizon=fhpe_horizon,
    )
    elapsed_s = time.time() - t0

    if not strategy:
        raise HTTPException(
            422,
            f"no feasible strategy found for {req.available_robots} robots "
            f"within the given budget ({checked_trees} trees checked) -- try "
            "more robots, a larger budget, or fewer/lower-weight hotspots.",
        )

    root, moves, visit_time = adapter.build_moves(scene, idx_to_vertex, strategy)
    mission_seconds = max((m["t_arrival"] for m in moves), default=0.0)

    return {
        "scene": req.scene,
        "approach": req.approach,
        "root": root,
        "moves": moves,
        "visit_time": {str(v): t for v, t in visit_time.items()},
        "priors": _quantize_priors(priors, _viewer_remap(scene, req.scene)),
        "metrics": {
            "checked_trees": checked_trees,
            "fitness": None if not np.isfinite(fitness) else fitness,
            # True when the approach found no feasible clearance strategy
            # and graphSearch fell back to the FHPE_SA patrol strategy.
            "no_clearance_found": not np.isfinite(fitness),
            "mission_seconds": mission_seconds,
            "available_robots": req.available_robots,
            "vertices_visited": len(visit_time),
            "n_vertices": n_vertices,
            "compute_seconds": elapsed_s,
        },
    }


# Static viewer last: it defines "/", which would otherwise shadow /api/*.
app.mount("/", StaticFiles(directory=VIEWER_DIR, html=True), name="viewer")


def main():
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()

# clearing-scenes (copied in)

A snapshot of the scenario data and loader from the
[`clearing-scenes`](https://github.com/) repo, copied in on 2026-09-14 so this
project can use the six real-site clearing scenarios without depending on that
repo living alongside it. This is a **plain copy, not a submodule** — pulling
updates from upstream means re-copying by hand.

## Layout

```
scenes/            the six scenarios: graph, guard regions, routes (YAML)
scenes/geometry/    the arrays each YAML points at -- cells, detection sets, cloud (npz)
examples/           worked strategies per scene, step by step, with metrics
clearing/           the Python loader + the published method's implementation
viewer/             the standalone 3D viewer (open viewer/index.html, no server needed)
docs/               scene-format.md, glossary.md, gsst.md -- the format and method explained
requirements.txt    numpy, scipy, pyyaml (install these to use `clearing`)
```

## Loading a scene

```python
from clearing import load

scene = load("christ-church")          # or any of: blenheim-palace,
                                        # bodleian-library, hb-allen-centre,
                                        # keble-college, observatory-quarter

scene.vertex_xyz            # (n_vertices, 3) sensor positions
scene.edge_ij                # (n_edges, 2) the guard-region graph
scene.detection[v]           # cell indices D(p): what a searcher at vertex v sees
scene.travel_seconds[i, j]   # shortest-path time between any two vertices
scene.route(i, j)            # the actual walked polyline for a graph edge
```

See [docs/scene-format.md](docs/scene-format.md) for what every field means and
[docs/glossary.md](docs/glossary.md) for the notation (D(p), dD(p), G_ij, ...).

## Viewer

```bash
open viewer/index.html
```

No build step, no server -- it reads `viewer/data/<scene>.js`, which is a
separate, downsampled/quantised export of the same scenes (built by
`clearing-scenes/scripts/build_web.py` upstream). If you regenerate or add
scenes here, the viewer's `data/` files will not update automatically; that
requires the upstream repo's build script.

## Setup

```bash
pip install -r requirements.txt
```

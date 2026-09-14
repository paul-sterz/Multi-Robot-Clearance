# The scenario format

One scene is two files:

```
scenes/<scene>.yaml              the scenario -- readable, hand-editable
scenes/geometry/<scene>.npz      the arrays it points at
```

The split is by size, not by importance. The YAML carries the graph and every
number a reader needs to understand the instance; the npz carries the cell-level
payload -- a detection set is ten thousand cell indices and a hundred of them do
not belong in a file a person is meant to read.

`clearing.scene.load(name)` returns both as one `Scene`.

---

## The YAML

```yaml
scene: christ-church
title: Christ Church

source:                    # where the geometry came from
  dataset: Oxford Spires Dataset (Tao et al., IJRR 2025)
  survey: terrestrial laser scan, merged cloud + individual E57 setups
  pipeline: e1-clearing-graph, stages 2-6
  occupancy: O-carved (unobserved space blocks)
  graph_file: carved_huav20_R30_seed1
  vertex_set: repaired-ground

boundary:                  # where the scenario stops
  bounded: true
  stamp: 4b19deaef44d      # hash of the drawn lines; matches E1's approval
  approved: 2026-08-22
  source: e1-clearing-graph configs/site_boundary.yaml, drawn by hand
  cuts: 2                  # hand-drawn lines across the openings
  held_line_m: 9.5         # how much of the perimeter those lines are
  cells_dropped: 45658     # walkable cells outside it
  walkable_dropped_m2: 1826.3
  excluded_m2: 2012.0      # E1's figure, on its 0.4 m raster, not this 0.2 m one
  rim_segments: 4424
  rim_length_m: 1769.6     # the whole outline, in the npz as boundary_seg

platform:
  kind: ground
  sensor_height_m: 1.0     # the sensor sits this far above the surface
  detection_range_m: 30.0
  speed_m_s: 1.0

target:                    # what has to be detected
  height_m: 1.0
  radius_m: 0.25
  visibility: all 5 rays to the target cylinder must pass

surface:
  geometry_file: geometry/christ-church.npz
  cell_size_m: 0.2
  cells: 203314
  area_m2: 8132.56
  bbox_min: [-102.7, -98.72, -2.63]
  bbox_max: [21.3, 59.48, 3.83]
  max_step_m: 0.25         # two neighbouring cells may differ by this much
  evader_edges: 790583     # 8-connected under that rule; recomputed, not stored
  uncoverable_cells: 5134  # seen by no vertex, at any team size
  uncoverable_area_m2: 205.36

graph:
  vertices: 62
  edges: 245
  edges_regular: 158
  edges_shady: 87
  vertices_list:
    - {id:   0, xyz: [-29.3, -27.92, -0.32], detection_m2: 2652.8, boundary_cells: 1392}
    ...
  edges_list:
    - {i:   0, j:   1, type: regular, guard_cells: 421, guard_m2: 16.84}
    ...

routes:
  reachable_pairs: 1891
  unreachable_pairs: 0
  speed_m_s: 1.0
  polylines_for_graph_edges: 245   # the walked path per edge, in the npz
  columns: [i, j, distance_m, time_s]
  values:
    - [  0,   1,    52,      52]
    ...
```

Every count above is *inside the boundary*. The surveyed surface at Christ
Church is 9 959 m² over 248 972 cells with 90 vertices; the two drawn lines
take 1 826 m² and 28 vertices off it before anything else is computed, and
nothing downstream ever sees the difference.

### What the fields mean

**The boundary** is where the scenario stops. It is drawn by hand upstream in
`e1-clearing-graph` and approved there; `stamp` is a hash over the lines that
were drawn, so a scene here can be matched to the decision that produced it. By
the time a scene loads, the cut has already been made — the surface, the
vertices, the guard regions and the routes are all what survived it — and
nothing in `clearing/` tests a cell against the line. `boundary_seg` in the npz
is the line itself, carried for drawing and for the record.

Two areas are quoted because they are measured on two grids: `walkable_dropped_m2`
is this surface's own 0.2 m cells, and `excluded_m2` is E1's figure on its 0.4 m
boundary raster. Use the first with anything else on this page.

`cuts` counts the lines somebody drew; `rim_segments` and `rim_length_m` are the
whole retained outline, most of which is masonry rather than a drawn line. At
Christ Church that is 9.5 m of decision against 1 770 m of wall.

**A vertex** is a sampled sensor position on the walkable surface. `xyz` is
where the sensor is -- `sensor_height_m` above the cell the searcher stands on.
`detection_m2` is the area of `D(p)`, the part of the surface a searcher there can
certify empty; `detection_boundary_cells` is the size of `dD(p)`, the cells of `D(p)` with
a walkable neighbour outside it.

**`vertex_set` says which vertices, and it is not E1's sample.** The sampler
upstream places positions until free space is COVERED -- the art-gallery
objective the 2010 formulation inherits -- and coverage is not the property a
clearing argument needs. Two things it leaves behind were measured upstream and
are repaired before the export:

* fragments that SEVER the walkable surface, watched by nothing. Christ
  Church's surface is one connected piece and its ground guard graph was in
  three, the whole split carried by a 0.40 m² gate and a 0.32 m² step-over.
  One ground sensor per threshold closes it.
* ground that no GROUND vertex can see -- 16 m² at HB Allen up to 116 m² at
  the Bodleian, every square metre of it seen by the air family E1 also has
  and this scenario does not. For a homogeneous ground team that is a
  permanent contamination source and a floor no strategy can go under.

`repaired-ground` is both repairs, and it is what these files ship. Ground
vertices against E1's raw sample: HB Allen 20 → 32, Keble 24 → 34, Blenheim
36 → 47, Observatory 39 → 60, Christ Church 62 → 81, Bodleian 112 → 151.
`scripts/export_from_e1.py --vertex-set released` reproduces the raw sample,
which is the control every claim about the repaired one is a claim against.

It is a CHANGE TO THE PROBLEM, not a better solution of it, and the field is
in every scene file so that a number can never be quoted without it.

**An edge** `(i, j)` exists where a searcher at `j` can watch part of the boundary of
`D(p_i)`: the guard region `G_ij = dD(p_i) ∩ D(p_j)` of Kolling et al. 2010.
It is `shady` when its guard region is strictly contained in another vertex's
and `regular` otherwise. Undirected edges take regular over shady.

An edge is a *coverage* relation between a position and a piece of boundary, not
adjacency between regions. That is worth keeping in mind when reading the graph:
two vertices can be edge-joined across a courtyard they are nowhere near each
other in.

**A route** is the platform's shortest path over the walkable surface,
8-connected with the same `max_step_m` rule, weighted by 3D step length so a
ramp costs its slope rather than its plan projection. `time_s` is
`distance_m / speed_m_s`. Pairs the platform cannot reach at all are omitted and
counted in `unreachable_pairs`.

Routes are lattice paths at `cell_size_m`, so they overestimate a smoothed path
by a few per cent, and they ignore vehicle dynamics entirely.

The YAML gives the *time and distance* for every pair. The **path itself** is in
the npz, for the pairs that are also edges of the graph -- `edge_route_points`,
one polyline per edge, reachable through `Scene.route(i, j)` and drawn by the
viewer. That is the set worth carrying: an edge of the graph is a line of sight,
and the ground between its two ends may be a building. On Blenheim Palace, edge
3-21 spans 22.0 m of straight line and 71 m of walking.

The polylines are simplified (Ramer-Douglas-Peucker at 0.12 m, below the cell
size) so a 500-point lattice walk becomes a couple of dozen points. Simplifying
shortens a path by up to about 5%, which is why every length quoted anywhere
comes from `travel_metres` and never from measuring the polyline.

### What is NOT in the file, on purpose

`evader_edges` is a count, not a list. The adjacency is recomputed from the
cells and `max_step_m` by `clearing.scene.lattice_edges`, so the rule and the
data cannot drift apart -- and it saves about 8 MB per scene.

---

## The npz

| key | shape | what it is |
|---|---|---|
| `cell_xyz` | (n_cells, 3) f32 | walkable surface points, world coordinates |
| `cell_ij` | (n_cells, 2) i32 | grid indices; two cells may share a column |
| `cell_size` | scalar f32 | 0.2 m |
| `node_xyz` | (n_nodes, 3) f32 | sensor positions |
| `node_cell` | (n_nodes,) i32 | the cell each vertex stands on |
| `detection_offsets` / `detection_cells` | CSR | `D(v)` per vertex, as cell indices |
| `rim_offsets` / `rim_cells` | CSR | `dD(v)` per vertex |
| `edge_ij` | (n_edges, 2) i32 | the guard-region graph |
| `edge_shady` | (n_edges,) u8 | 1 = shady |
| `guard_offsets` / `guard_cells` | CSR | `G_ij` per edge |
| `travel_seconds` / `travel_metres` | (n, n) f32 | all pairs, `inf` where unreachable |
| `travel_component` | (n_nodes,) i32 | vertices sharing one can reach each other |
| `edge_route_points` / `edge_route_offsets` | polylines | per graph edge, the path the platform walks |
| `uncoverable` | (k,) i32 | cells no vertex sees |
| `boundary_seg` | (n_seg, 2, 2) f32 | the scenario boundary, as line segments in world XY |
| `boundary_cell_size` | scalar f32 | the raster the boundary was cut on, 0.4 m |
| `cloud_xyz` | (m, 3) f32 | the survey cloud, decimated, for context |

CSR here means the usual pair: row `i` is `values[offsets[i]:offsets[i+1]]`.

---

## Making your own scene

Nothing in `clearing/` knows where the geometry came from. To add a scene,
write the two files. The checks in `tests/test_scene.py` are the contract:
every count in the YAML has to match the arrays, guard regions have to lie
inside the detection set of one of their endpoints, and travel times have to be
distance over speed.

To regenerate the six shipped scenes from the upstream pipeline:

```bash
python scripts/export_from_e1.py --e1-root ~/e1-clearing-graph --R 30 --seed 1
```

That is the only script that needs the survey clouds and the raycaster.

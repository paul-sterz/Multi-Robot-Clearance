/* The scene viewer.
 *
 * Everything it draws is a point cloud or a line set, so there is one geometry
 * per layer and no scene graph to speak of. The only slightly involved parts
 * are `paint()`, which decides what colour each walkable cell gets -- its
 * height, the detection set of a selected vertex, or the state of a strategy
 * step -- and the machine view below it, which puts the fleet on the strategy:
 * one marker with an id per machine, the ground it walks between two steps, and
 * the surface the one you click on watches.
 */

const $ = (s) => document.querySelector(s);

/* ---------------------------------------------------------------- decoding */

function decode(b64, Type) {
  const bin = atob(b64);
  const buf = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) buf[i] = bin.charCodeAt(i);
  return new Type(buf.buffer);
}

/** uint16 back to metres, relative to the scene centre. */
function dequantise(pack) {
  const q = decode(pack.q, Uint16Array);
  const out = new Float32Array(q.length);
  const [sx, sy, sz] = pack.scale, [ox, oy, oz] = pack.offset;
  for (let i = 0; i < q.length; i += 3) {
    out[i] = q[i] * sx + ox;
    out[i + 1] = q[i + 1] * sy + oy;
    out[i + 2] = q[i + 2] * sz + oz;
  }
  return out;
}

/** The same, for the packed XY of the boundary line: two axes, not three. */
function dequantise2(pack) {
  const q = decode(pack.q, Uint16Array);
  const out = new Float32Array(q.length);
  const [sx, sy] = pack.scale, [ox, oy] = pack.offset;
  for (let i = 0; i < q.length; i += 2) {
    out[i] = q[i] * sx + ox;
    out[i + 1] = q[i + 1] * sy + oy;
  }
  return out;
}

/** Run lengths back to a Uint8 mask; the first run is always "off". */
function unrle(runs, n) {
  const m = new Uint8Array(n);
  let at = 0, on = 0;
  for (const r of runs) {
    if (on) m.fill(1, at, at + r);
    at += r;
    on ^= 1;
  }
  return m;
}

/* ------------------------------------------------------------------ colour */

function ramp(t, stops) {
  t = Math.max(0, Math.min(1, t));
  for (let i = 0; i < stops.length - 1; i++) {
    const [t0, c0] = stops[i], [t1, c1] = stops[i + 1];
    if (t <= t1) {
      const f = (t - t0) / Math.max(t1 - t0, 1e-9);
      return [c0[0] + (c1[0] - c0[0]) * f,
              c0[1] + (c1[1] - c0[1]) * f,
              c0[2] + (c1[2] - c0[2]) * f];
    }
  }
  return stops[stops.length - 1][1];
}

const SURFACE = [[0, [0.07, 0.31, 0.36]], [0.5, [0.14, 0.58, 0.58]],
                 [1, [0.89, 0.84, 0.63]]];
const CLOUD = [[0, [0.25, 0.28, 0.32]], [0.6, [0.45, 0.49, 0.54]],
               [1, [0.69, 0.71, 0.76]]];
const CLEARED = [0.14, 0.58, 0.64];
const DIRTY = [0.36, 0.18, 0.27];
const WATCHED = [0.98, 0.69, 0.29];
const SEEN_BY = [0.98, 0.55, 0.22];
// Violet and pink, not green: the surface is teal and the graph edges are
// amber, and a route has to be told apart from both at a glance.
const ROUTE = 0x9d7bff;      // every route, drawn faintly
const ROUTE_HOT = 0xff5fbf;  // the selected one
// Routes are lifted clear of the surface. The walkable cells are drawn as
// sprites nearly two cells wide, so a polyline sitting on the surface is
// hidden by the very floor it runs over; the selected one is lifted further
// again so it reads over the rest.
const ROUTE_LIFT = 0.45;
const ROUTE_LIFT_SEL = 0.75;

// The scenario boundary: two rails at the foot and the top of a low fence,
// bright, with faint posts every couple of metres between them. That is how
// e1-clearing-graph draws it, and the shape is worth keeping.
//
// THE COLOUR IS NOT. E1 draws the boundary amber because nothing else in its
// viewers is; here amber is the guard-region edges and pale yellow is the
// vertices, and an amber fence read as more graph -- a ring of edges around
// the site, which is the one thing it is not. Cool blue instead, for the same
// reason the routes above are violet: it has to be told apart from the teal
// surface and the amber graph at a glance, and blue is the hue this palette
// has not spent.
//
// It is a fence and not a line on the floor because the floor is what it
// bounds: the walkable cells are sprites nearly two cells across and a line
// lying on them is covered by them, which is invisible from exactly the
// overhead view the site is read from. The FOOT of the fence is still exactly
// the boundary; the rest of it is only what makes the foot findable in three
// dimensions. Posts are drawn faint because at any distance a fence drawn
// evenly reads as a solid wall, and a wall hides the site it bounds.
const BOUND = 0x8fd4ff;
const POST_EVERY_M = 2.4;

// The machine view. A step is a set of vertices and says nothing about which
// machine stands where, so the identities come from the roster shipped beside
// the strategy (clearing/roster.py): a fleet as large as the peak team, matched
// step to step by least total travel.
const TRANSIT = 0.75;        // of a play interval spent walking, the rest held
const STEP_MS = 550;         // one step, in the plain strategy playback
const WALK_MS = 1500;        // one step, when the walking is being animated

/* ---------------------------------------------------------- computed strategy
 *
 * A strategy computed here (DP/Greedy/... over /api/run) is a continuous-time
 * move list, not the discrete "steps: sets of held vertices" shape the shipped
 * strategy above uses -- see strategy_service/adapter.py. So it gets its own
 * small playback model: a scrubber over mission seconds instead of step index,
 * "packs" of robots (a move's `robots` travel and arrive together, and the
 * algorithms never give them individual identity) drawn at nodes or in transit
 * along a move's path, and cleared/dirty painted from each vertex's first-visit
 * time against the detection sets already loaded for this scene.
 */
const HOTSPOT_COLOR = [0xffe066, 0xff9d3d, 0xff3d3d];   // low / medium / high
const START_COLOR = 0x3dff9d;
const PACK_COLOR = 0x8fd4ff;
const MARKER_LIFT = 0.55;
const PACK_LIFT = 0.9;
// dark blue -> violet -> red -> amber -> pale yellow: same "cool to hot" read
// as the 2D grid tool's prior heatmap, over the same ramp() helper `paint()`
// uses for height/cloud colouring.
const HEAT = [[0, [0.06, 0.06, 0.28]], [0.35, [0.32, 0.08, 0.42]],
              [0.6, [0.77, 0.15, 0.18]], [0.85, [0.98, 0.55, 0.12]],
              [1, [1.0, 0.96, 0.35]]];

/* ------------------------------------------------------------------- scene */

let renderer, camera, world, raycaster;
let S = null;               // the loaded scene payload
let G = {};                 // the THREE objects of the current scene
let labels = [];            // one id sprite per machine of the current scene
let selMachine = -1;
let playing = false, phase = 0, lastMs = 0;

// hotspot/start-vertex placement, and the strategy /api/run last computed --
// all scene-specific, reset in build()
let hotspots = [];          // { cell, category, xyz: [x,y,z] }
let startVertices = [];     // { vertex, xyz: [x,y,z] }
let computed = null;        // the /api/run response, or null
let compLabelPool = new Map();  // pack key -> id sprite, reused while stable
let compPlaying = false, compLastMs = 0, compSpeed = 1;
let priorsView = null;      // { codes: Uint16Array, min, max } -- last computed/previewed priors
let priorsPreviewTimer = null;

function init() {
  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  $("#view").appendChild(renderer.domElement);
  camera = new THREE.PerspectiveCamera(52, 1, 0.2, 6000);
  world = new THREE.Scene();
  world.background = new THREE.Color(0x0b0f14);
  raycaster = new THREE.Raycaster();
  raycaster.params.Points.threshold = 1.6;

  const sel = $("#scene");
  for (const s of window.SCENE_INDEX) {
    const o = document.createElement("option");
    o.value = s.name;
    o.textContent = `${s.title}  (${s.mb} MB)`;
    sel.appendChild(o);
  }
  sel.onchange = () => loadScene(sel.value);

  for (const id of ["cloud", "surface", "edges", "shady", "vertices", "routes",
                    "boundary"]) {
    $("#l-" + id).onchange = applyLayers;
  }
  $("#route-from").onchange = () => { fillRouteTo(); drawRoute(); };
  $("#route-to").onchange = drawRoute;
  $("#vertex").onchange = () => { $("#show-strat").checked = false;
                                  selMachine = -1; paint(); };
  $("#step").oninput = onScrub;
  $("#show-strat").onchange = () => { $("#vertex").value = "-1"; paint(); };
  $("#show-machines").onchange = () => { $("#vertex").value = "-1"; paint(); };
  $("#fleet-list").onclick = (e) => {
    const row = e.target.closest("[data-r]");
    if (row) selectRobot(+row.dataset.r === selMachine ? -1 : +row.dataset.r);
  };
  $("#play").onclick = togglePlay;

  $("#c-stop-mode").onchange = () => {
    $("#c-stop-label").textContent = $("#c-stop-mode").value === "time" ? "Seconds" : "Trees";
  };
  $("#c-markers").onclick = (e) => {
    const btn = e.target.closest("[data-rm]");
    if (!btn) return;
    const i = parseInt(btn.dataset.rm.slice(1), 10);
    if (btn.dataset.rm[0] === "h") removeHotspot(i); else removeStart(i);
  };
  $("#c-run").onclick = runStrategy;
  $("#c-play").onclick = toggleCompPlay;
  $("#c-time").oninput = () => {
    compPlaying = false;
    $("#c-play").textContent = "Play";
    paintComputed(parseFloat($("#c-time").value));
  };
  $("#c-show-priors").onchange = () => {
    if ($("#c-show-priors").checked && !priorsView) refreshPriorsPreview();
    else updateSurfaceView();
  };
  $("#c-prior-l").oninput = schedulePriorsPreview;
  $("#c-prior-radius").oninput = schedulePriorsPreview;

  addEventListener("keydown", (e) => {
    if (e.key === "r" || e.key === "R") frame();
    if (e.key === "ArrowLeft") stepBy(-1);
    if (e.key === "ArrowRight") stepBy(1);
  });
  addEventListener("resize", resize);
  renderer.domElement.addEventListener("click", pick);
  orbit(renderer.domElement);

  resize();
  const want = hashState();
  sel.value = window.SCENE_INDEX.some((s) => s.name === want.scene)
    ? want.scene : window.SCENE_INDEX[0].name;
  loadScene(sel.value);
  // The loop is a clock, not just a repaint: playing the strategy with the
  // machines on walks them along their paths, and that has to be told how much
  // time went by rather than how many frames.
  (function loop(ms) {
    requestAnimationFrame(loop);
    if (playing && S) {
      const dt = lastMs ? Math.min(ms - lastMs, 250) : 0;
      lastMs = ms;
      phase += dt / (robotMode() ? WALK_MS : STEP_MS);
      if (phase >= 1) { phase = 0; stepBy(1); }
      else if (robotMode()) placeRobots(phase);
    } else lastMs = 0;

    if (compPlaying && computed) {
      const dt = compLastMs ? Math.min(ms - compLastMs, 250) : 0;
      compLastMs = ms;
      const el = $("#c-time");
      let t = parseFloat(el.value) + (dt / 1000) * compSpeed;
      const max = +el.max;
      if (t >= max) { t = max; compPlaying = false; $("#c-play").textContent = "Play"; }
      el.value = t;
      paintComputed(t);
    } else compLastMs = 0;

    renderer.render(world, camera);
  })();
}

function resize() {
  const w = innerWidth, h = innerHeight;
  renderer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}

/* -------------------------------------------------------------- loading */

const loaded = new Set();

function loadScene(name) {
  $("#loading").classList.remove("hidden");
  const go = () => { build(window.SCENES[name]); $("#loading").classList.add("hidden"); };
  if (loaded.has(name)) { setTimeout(go, 0); return; }
  const meta = window.SCENE_INDEX.find((s) => s.name === name);
  const tag = document.createElement("script");
  tag.src = meta.file;
  tag.onload = () => { loaded.add(name); go(); };
  document.body.appendChild(tag);
}

function build(data) {
  stopPlay();
  selMachine = -1;
  for (const sp of labels) {
    world.remove(sp);
    sp.material.map.dispose();
    sp.material.dispose();
  }
  // The computed-strategy pack labels are keyed and reused across frames
  // (see placeComputedPacks), so they are never swept up by the generic
  // Object.keys(G) cleanup below -- without this they survive a scene
  // switch as orphaned sprites still sitting at the OLD scene's coordinates,
  // which is exactly why they seemed to "float outside" a newly loaded scene.
  for (const sp of compLabelPool.values()) {
    world.remove(sp);
    sp.material.map.dispose();
    sp.material.dispose();
  }
  labels = [];
  for (const k of Object.keys(G)) {
    world.remove(G[k]);
    G[k].geometry && G[k].geometry.dispose();
    G[k].material && G[k].material.dispose();
  }
  G = {};
  S = data;

  // survey cloud
  const cloud = dequantise(data.cloud);
  G.cloud = points(cloud, colourByHeight(cloud, CLOUD), 0.32);

  // walkable surface -- its colours are rewritten by paint()
  S.surf = dequantise(data.surface);
  S.nCells = S.surf.length / 3;
  S.surfBase = colourByHeight(S.surf, SURFACE);
  G.surface = points(S.surf, S.surfBase.slice(), data.cellSize * 1.7);

  // graph
  const vertices = decode(data.vertices, Float32Array);
  const edges = decode(data.edges, Uint16Array);
  const shady = decode(data.edgeShady, Uint8Array);
  S.vertices = vertices;
  S.edges = edges;
  S.shady = shady;
  const lift = 0.6;
  for (const kind of [0, 1]) {
    const pos = [];
    for (let e = 0; e < shady.length; e++) {
      if (shady[e] !== kind) continue;
      for (const v of [edges[2 * e], edges[2 * e + 1]]) {
        pos.push(vertices[3 * v], vertices[3 * v + 1], vertices[3 * v + 2] + lift);
      }
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
    const m = new THREE.LineBasicMaterial({
      color: kind ? 0x7a5c46 : 0xfab04a, transparent: true,
      opacity: kind ? 0.5 : 0.75,
    });
    G[kind ? "shady" : "regular"] = new THREE.LineSegments(g, m);
  }
  const np = new Float32Array(vertices.length);
  for (let i = 0; i < vertices.length; i += 3) {
    np[i] = vertices[i]; np[i + 1] = vertices[i + 1]; np[i + 2] = vertices[i + 2] + lift;
  }
  S.nodeDraw = np;
  G.vertices = points(np, null, 2.4, 0xffeca8);

  // the walked routes, one polyline per edge
  S.routePts = dequantise(data.routes);
  S.routeOff = decode(data.routes.offsets, Uint32Array);
  S.routeSecs = decode(data.routes.seconds, Float32Array);
  S.routeMetres = decode(data.routes.metres, Float32Array);
  G.routes = segments(routeSegments(S), ROUTE, 0.7);
  G.routeSel = segments(new Float32Array(0), ROUTE_HOT, 1.0);
  G.routeDots = points(new Float32Array(0), null, 1.6, ROUTE_HOT);

  // the scenario boundary -- where the site stops, and where the surface,
  // the vertices and the routes above were all cut to
  const bnd = data.boundary;
  if (bnd) {
    const seg = dequantise2(bnd);
    const [z0, z1] = bnd.fence;
    // Every segment is one boundary cell wide and they come out of the mask in
    // raster order, which runs along the rim, so every k-th segment lands
    // roughly every k cells of line rather than all of them in one corner.
    const kpost = Math.max(1, Math.round(POST_EVERY_M / (bnd.cellSize || 0.4)));
    const rails = [], posts = [];
    for (let q = 0; q < seg.length / 4; q++) {
      const ax = seg[4 * q], ay = seg[4 * q + 1];
      const bx = seg[4 * q + 2], by = seg[4 * q + 3];
      rails.push(ax, ay, z0, bx, by, z0, ax, ay, z1, bx, by, z1);
      if (q % kpost === 0) posts.push(ax, ay, z0, ax, ay, z1);
    }
    G.bound = segments(new Float32Array(rails), BOUND, 0.9);
    G.boundPosts = segments(new Float32Array(posts), BOUND, 0.28);
  }

  G.held = points(new Float32Array(0), null, 4.2, 0xff6a3d);

  // hotspot / start-vertex markers, and the packs of a strategy computed here
  hotspots = [];
  startVertices = [];
  computed = null;
  compPlaying = false;
  compPaintKey = null;
  compLabelPool = new Map();
  priorsView = null;
  clearTimeout(priorsPreviewTimer);
  G.hotspots = points(new Float32Array(0), new Float32Array(0), 3.6);
  G.starts = points(new Float32Array(0), new Float32Array(0), 3.6);
  G.compPacks = points(new Float32Array(0), new Float32Array(0), 3.2);

  // the fleet: a marker per machine, the legs of the current step in the machines'
  // own colours, and the selected machine's walked track
  S.roster = data.strategy ? data.strategy.roster : null;
  S.legCache = new Map();
  S.robotIds = [];
  if (S.roster) {
    S.machinePts = dequantise(S.roster.paths);
    S.machineOff = decode(S.roster.paths.offsets, Uint32Array);
    S.legOf = S.roster.legs.map((step) => {
      const m = new Map();
      for (const mv of step) m.set(mv.r, mv);
      return m;
    });
  }
  G.halo = discs(6.4, 0.5);          // behind the marker of the selected machine
  G.machines = discs(3.6);
  G.legs = vsegments(new Float32Array(0), new Float32Array(0), 0.95);
  G.legDots = points(new Float32Array(0), new Float32Array(0), 1.5);
  G.track = segments(new Float32Array(0), 0xffffff, 0.55);
  G.trackDots = points(new Float32Array(0), null, 1.6, 0xffffff);

  for (const k of Object.keys(G)) world.add(G[k]);

  // detection sets, decoded lazily -- most are never looked at
  S.dsetCache = new Map();

  // How big an id sprite has to be to read at the framing `frame()` picks: the
  // camera sits about one site-span away, so a label of a fortieth of the span
  // lands at roughly twenty pixels however large the site is.
  const bb = new THREE.Box3().setFromBufferAttribute(
    G.surface.geometry.attributes.position);
  S.span = Math.max(bb.getSize(new THREE.Vector3()).x,
                    bb.getSize(new THREE.Vector3()).y);
  S.labelSize = S.span * 0.025;
  if (S.roster) makeLabels(S.roster.nMachines);

  fillPanel();
  rebuildMarkers();
  fillMarkerList();
  $("#c-playback").style.display = "none";
  $("#c-status").textContent = "";
  applyLayers();
  const want = hashState();
  if (want.vertex !== null) $("#vertex").value = want.vertex;
  if (want.step !== null) { $("#step").value = want.step; $("#show-strat").checked = true; }
  if (want.machine !== null && S.roster && want.machine < S.roster.nMachines) {
    $("#show-machines").checked = true;
    selMachine = want.machine;
  }
  if (want.from !== null) {
    $("#route-from").value = want.from;
    fillRouteTo();
    if (want.to !== null) $("#route-to").value = want.to;
    $("#l-routes").checked = true;
    drawRoute();
  }
  paint();
  frame();
}

/** `#scene=christ-church&vertex=12`, `&step=7`, `&machine=3` -- a linkable view. */
function hashState() {
  const q = new URLSearchParams(location.hash.replace(/^#/, ""));
  const num = (k) => (q.has(k) ? parseInt(q.get(k), 10) : null);
  return { scene: q.get("scene"), vertex: num("vertex"), step: num("step"),
           from: num("from"), to: num("to"), machine: num("machine") };
}

/** Every edge index, as the default "all routes" selection. */
function allEdges(sc) {
  return Array.from({ length: sc.routeOff.length - 1 }, (_, k) => k);
}

/** The given edges' polylines, flattened into consecutive segment endpoints. */
function routeSegments(sc, which, lift) {
  const keep = which || allEdges(sc);
  const dz = lift === undefined ? ROUTE_LIFT : lift;
  const out = [];
  for (const k of keep) {
    const a = sc.routeOff[k], b = sc.routeOff[k + 1];
    for (let i = a; i + 1 < b; i++) {
      out.push(sc.routePts[3 * i], sc.routePts[3 * i + 1], sc.routePts[3 * i + 2] + dz,
               sc.routePts[3 * i + 3], sc.routePts[3 * i + 4], sc.routePts[3 * i + 5] + dz);
    }
  }
  return new Float32Array(out);
}

/** The same polylines, resampled at a fixed spacing, as drawable points. */
function routeBeads(sc, which, spacing, lift) {
  const out = [];
  for (const k of which || []) {
    const a = sc.routeOff[k], b = sc.routeOff[k + 1];
    let carry = 0;
    for (let i = a; i + 1 < b; i++) {
      const p = [sc.routePts[3 * i], sc.routePts[3 * i + 1], sc.routePts[3 * i + 2]];
      const q = [sc.routePts[3 * i + 3], sc.routePts[3 * i + 4], sc.routePts[3 * i + 5]];
      const d = Math.hypot(q[0] - p[0], q[1] - p[1], q[2] - p[2]);
      for (let t = carry; t < d; t += spacing) {
        const f = t / Math.max(d, 1e-9);
        out.push(p[0] + (q[0] - p[0]) * f, p[1] + (q[1] - p[1]) * f,
                 p[2] + (q[2] - p[2]) * f + lift);
      }
      carry = d > 0 ? (spacing - ((d - carry) % spacing)) % spacing : carry;
    }
  }
  return new Float32Array(out);
}

function segments(xyz, colour, opacity) {
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(xyz, 3));
  return new THREE.LineSegments(g, new THREE.LineBasicMaterial({
    color: colour, transparent: true, opacity }));
}

function vsegments(xyz, colours, opacity) {
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(xyz, 3));
  g.setAttribute("color", new THREE.Float32BufferAttribute(colours, 3));
  return new THREE.LineSegments(g, new THREE.LineBasicMaterial({
    vertexColors: true, transparent: true, opacity }));
}

/** Machine markers: round, so that a machine is never a vertex seen up close.
 *
 * Every other layer here is a square point sprite -- the surface cells, the
 * graph vertices, the beads along a route. A disc is the one shape that is not
 * already spoken for, and the fleet has to be picked out of all of them.
 */
function discs(size, opacity) {
  const c = document.createElement("canvas");
  c.width = c.height = 64;
  const g = c.getContext("2d");
  g.beginPath();
  g.arc(32, 32, 26, 0, 2 * Math.PI);
  g.fillStyle = "#fff";
  g.fill();
  g.lineWidth = 7;
  g.strokeStyle = "rgba(7,11,16,0.85)";
  g.stroke();
  const geom = new THREE.BufferGeometry();
  geom.setAttribute("position", new THREE.Float32BufferAttribute([], 3));
  geom.setAttribute("color", new THREE.Float32BufferAttribute([], 3));
  return new THREE.Points(geom, new THREE.PointsMaterial({
    size, sizeAttenuation: true, vertexColors: true,
    map: new THREE.CanvasTexture(c), transparent: true, alphaTest: 0.35,
    opacity: opacity === undefined ? 1 : opacity }));
}

function setGeom(obj, pos, col) {
  obj.geometry.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  obj.geometry.attributes.position.needsUpdate = true;
  if (!col) return;
  obj.geometry.setAttribute("color", new THREE.Float32BufferAttribute(col, 3));
  obj.geometry.attributes.color.needsUpdate = true;
}

function points(xyz, colours, size, colour) {
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(xyz, 3));
  const opts = { size, sizeAttenuation: true };
  if (colours) { g.setAttribute("color", new THREE.Float32BufferAttribute(colours, 3)); opts.vertexColors = true; }
  else opts.color = colour === undefined ? 0xffffff : colour;
  return new THREE.Points(g, new THREE.PointsMaterial(opts));
}

function colourByHeight(xyz, stops) {
  let lo = Infinity, hi = -Infinity;
  for (let i = 2; i < xyz.length; i += 3) { lo = Math.min(lo, xyz[i]); hi = Math.max(hi, xyz[i]); }
  const out = new Float32Array(xyz.length);
  for (let i = 0; i < xyz.length; i += 3) {
    const c = ramp((xyz[i + 2] - lo) / Math.max(hi - lo, 1e-6), stops);
    out[i] = c[0]; out[i + 1] = c[1]; out[i + 2] = c[2];
  }
  return out;
}

/* --------------------------------------------------------------- painting */

function dset(v) {
  if (!S.dsetCache.has(v)) S.dsetCache.set(v, unrle(S.detection[v], S.nCells));
  return S.dsetCache.get(v);
}

function paint() {
  const col = G.surface.geometry.attributes.color.array;
  col.set(S.surfBase);

  const v = parseInt($("#vertex").value, 10);
  const t = parseInt($("#step").value, 10);
  const showStrat = $("#show-strat").checked;
  let held = [];

  if (v >= 0) {
    const m = dset(v);
    for (let i = 0; i < S.nCells; i++) {
      if (!m[i]) continue;
      col[3 * i] = SEEN_BY[0]; col[3 * i + 1] = SEEN_BY[1]; col[3 * i + 2] = SEEN_BY[2];
    }
    held = [v];
    $("#step-label").textContent = "";
  } else if (S.strategy && showStrat) {
    const dirty = unrle(S.strategy.contaminated[t], S.nCells);
    held = S.strategy.steps[t];
    const seen = new Uint8Array(S.nCells);
    for (const q of held) {
      const m = dset(q);
      for (let i = 0; i < S.nCells; i++) if (m[i]) seen[i] = 1;
    }
    for (let i = 0; i < S.nCells; i++) {
      const c = seen[i] ? WATCHED : (dirty[i] ? DIRTY : CLEARED);
      col[3 * i] = c[0]; col[3 * i + 1] = c[1]; col[3 * i + 2] = c[2];
    }
    const secs = S.roster ? S.roster.stepSeconds[t] : null;
    $("#step-label").textContent =
      `step ${t + 1} of ${S.strategy.steps.length} — ${held.length} machines`
      + (secs ? ` — ${fmtTime(secs)} of walking` : "");
  } else {
    $("#step-label").textContent = "";
  }

  // The one machine the reader is following: what it watches from where it
  // stands, in its own colour, over whatever the strategy painted underneath.
  if (v < 0 && robotMode() && selMachine >= 0) {
    const rv = S.roster.positions[t][selMachine];
    if (rv >= 0) {
      const m = dset(rv), c = robotColour(selMachine);
      for (let i = 0; i < S.nCells; i++) {
        if (!m[i]) continue;
        col[3 * i] = c.r * 0.92; col[3 * i + 1] = c.g * 0.92;
        col[3 * i + 2] = c.b * 0.92;
      }
    }
  }

  G.surface.geometry.attributes.color.needsUpdate = true;

  const hp = new Float32Array(held.length * 3);
  held.forEach((q, k) => {
    hp[3 * k] = S.nodeDraw[3 * q];
    hp[3 * k + 1] = S.nodeDraw[3 * q + 1];
    hp[3 * k + 2] = S.nodeDraw[3 * q + 2];
  });
  G.held.geometry.setAttribute("position", new THREE.Float32BufferAttribute(hp, 3));
  G.held.geometry.attributes.position.needsUpdate = true;

  updateRobots();
}

/* ------------------------------------------------------------------ machines */

/* A strategy step is a SET of vertices: it says which places are held and
 * nothing about which machine holds them, so nine dots blinking on and off is
 * all it can honestly be drawn as. `clearing/roster.py` matches a fleet across
 * the steps -- least total travel, a machine the step does not need parking where
 * it stands -- and ships one leg per occupied vertex with the path it walks.
 * That is what everything below draws. The fleet is exactly the peak team, so
 * the ids run R0..R8 on Blenheim and R0..R20 on the Bodleian and no further.
 */

function roster() { return S && S.roster ? S.roster : null; }
function robotMode() { return !!roster() && $("#show-machines").checked; }
function stepAt() { return parseInt($("#step").value, 10) || 0; }
function legOf(t, r) {
  const m = S.legOf && S.legOf[t];
  return m ? m.get(r) : undefined;
}

/* Three hues are already spoken for and a machine may not borrow any of them:
 * amber is "watched now" and the graph edges, teal is "cleared" and the
 * surface, mauve is "contaminated". A machine painted in one of those is a machine
 * the reader has to look at twice -- and the selected machine's detection set is
 * painted in its own colour, straight over that palette. So the hue circle is
 * cut into the arcs that are left. */
const HUE_ARCS = [[0.22, 0.44], [0.56, 0.87], [0.97, 1.05]];
const HUE_SPAN = HUE_ARCS.reduce((a, [lo, hi]) => a + (hi - lo), 0);

/** One hue per machine, walked along those arcs by the golden angle.
 *
 * Any fleet size comes out with neighbouring ids far apart in colour, which is
 * what matters when R6 and R7 stand two vertices from each other. The
 * lightness reads darker here than it looks: three.js takes these as sRGB and
 * renders them through its working space, which lifts them by about a fifth.
 */
function robotColour(r) {
  let p = ((r * 0.381966) % 1) * HUE_SPAN;
  for (const [lo, hi] of HUE_ARCS) {
    if (p <= hi - lo) return new THREE.Color().setHSL((lo + p) % 1, 0.95, 0.42);
    p -= hi - lo;
  }
  return new THREE.Color().setHSL(HUE_ARCS[0][0], 0.95, 0.42);
}

/** The id sprites. One per machine, made once per scene and then only moved. */
function makeLabels(n) {
  for (let r = 0; r < n; r++) {
    const sp = labelSprite(`R${r}`, robotColour(r));
    sp.visible = false;
    labels.push(sp);
    world.add(sp);
  }
}

function labelSprite(text, colour) {
  const c = document.createElement("canvas");
  const g = c.getContext("2d");
  const font = "600 44px ui-sans-serif, system-ui, -apple-system, sans-serif";
  g.font = font;
  c.width = Math.ceil(g.measureText(text).width) + 28;   // resets the context
  c.height = 64;
  g.font = font;
  g.textAlign = "center";
  g.textBaseline = "middle";
  g.lineWidth = 7;
  g.strokeStyle = "rgba(7,11,16,0.9)";   // an id crossing pale stone still reads
  g.strokeText(text, c.width / 2, 34);
  g.fillStyle = "#" + colour.getHexString();
  g.fillText(text, c.width / 2, 34);
  const sp = new THREE.Sprite(new THREE.SpriteMaterial({
    map: new THREE.CanvasTexture(c), transparent: true, depthTest: false }));
  sp.renderOrder = 10;                       // ids are never hidden by geometry
  sp.userData.aspect = c.width / c.height;
  return sp;
}

/** The polyline of one leg, lifted to meet the vertex markers at both ends.
 *
 * The shipped path runs over the ground; a vertex is drawn at sensor height.
 * Blending the two offsets along the walk is what makes a machine leave its
 * marker and arrive at the next one without a step in the air.
 */
function leg(k, frm, to) {
  if (S.legCache.has(k)) return S.legCache.get(k);
  const a = S.machineOff[k], n = S.machineOff[k + 1] - a;
  if (!n) {                       // the two vertices cannot reach each other
    const empty = { pts: new Float32Array(0), cum: new Float64Array(0), n: 0, len: 0 };
    S.legCache.set(k, empty);
    return empty;
  }
  const pts = new Float32Array(n * 3);
  const dz0 = S.nodeDraw[3 * frm + 2] - S.machinePts[3 * a + 2];
  const dz1 = S.nodeDraw[3 * to + 2] - S.machinePts[3 * (a + n - 1) + 2];
  for (let i = 0; i < n; i++) {
    const u = n > 1 ? i / (n - 1) : 0;
    pts[3 * i] = S.machinePts[3 * (a + i)];
    pts[3 * i + 1] = S.machinePts[3 * (a + i) + 1];
    pts[3 * i + 2] = S.machinePts[3 * (a + i) + 2] + dz0 + (dz1 - dz0) * u;
  }
  const cum = new Float64Array(n);
  for (let i = 1; i < n; i++) {
    cum[i] = cum[i - 1] + Math.hypot(pts[3 * i] - pts[3 * i - 3],
                                     pts[3 * i + 1] - pts[3 * i - 2],
                                     pts[3 * i + 2] - pts[3 * i - 1]);
  }
  const out = { pts, cum, n, len: n ? cum[n - 1] : 0 };
  S.legCache.set(k, out);
  return out;
}

function nodePoint(v) {
  return [S.nodeDraw[3 * v], S.nodeDraw[3 * v + 1], S.nodeDraw[3 * v + 2]];
}

/** `g` of the way along a leg, by walked distance and not by vertex count. */
function alongLeg(mv, g) {
  const L = leg(mv.p, mv.f, mv.v);
  if (L.n < 2 || !L.len) return nodePoint(mv.v);
  const want = g * L.len;
  let i = 1;
  while (i < L.n - 1 && L.cum[i] < want) i++;
  const u = (want - L.cum[i - 1]) / Math.max(L.cum[i] - L.cum[i - 1], 1e-9);
  return [0, 1, 2].map((c) =>
    L.pts[3 * (i - 1) + c] + (L.pts[3 * i + c] - L.pts[3 * (i - 1) + c]) * u);
}

/** Where machine `r` is at step `t`, or `f` of the way into the step after it. */
function robotPos(r, t, f) {
  const R = roster();
  const at = R.positions[t][r];
  const mv = f > 0 ? legOf(t + 1, r) : undefined;
  if (mv) return mv.p >= 0 ? alongLeg(mv, Math.min(f / TRANSIT, 1)) : nodePoint(mv.v);
  return at >= 0 ? nodePoint(at) : null;
}

/** Draw the fleet. `f` is the walk into the next step, 0 while a step is held. */
function placeRobots(f) {
  const R = roster();
  if (!R) return;
  const t = stepAt();
  const pos = [], col = [], ids = [], halo = [];
  for (let r = 0; r < R.nMachines; r++) {
    const p = robotPos(r, t, f);
    if (!p) { labels[r].visible = false; continue; }
    const c = robotColour(r);
    const hot = r === selMachine;
    pos.push(p[0], p[1], p[2]);
    col.push(c.r, c.g, c.b);
    if (hot) halo.push(p[0], p[1], p[2]);
    ids.push(r);
    const sp = labels[r];
    sp.visible = true;
    // Three heights, by id: a team crowds into a courtyard corner, and two
    // ids at the same height there are one unreadable word.
    sp.position.set(p[0], p[1], p[2] + S.labelSize * (1.1 + 0.75 * (r % 3)));
    const size = S.labelSize * (hot ? 1.35 : 1);
    sp.scale.set(size * sp.userData.aspect, size, 1);
  }
  S.robotIds = ids;
  setGeom(G.machines, pos, col);
  setGeom(G.halo, halo, halo.length ? [1, 1, 1] : []);
}

/** The legs into one step: what the team walked to take up these vertices. */
function drawLegs(t) {
  const R = roster();
  const pos = [], col = [], dots = [], dotCol = [];
  for (const mv of (R && R.legs[t]) || []) {
    if (mv.p < 0) continue;
    const L = leg(mv.p, mv.f, mv.v), c = robotColour(mv.r);
    for (let i = 0; i + 1 < L.n; i++) {
      pos.push(L.pts[3 * i], L.pts[3 * i + 1], L.pts[3 * i + 2],
               L.pts[3 * i + 3], L.pts[3 * i + 4], L.pts[3 * i + 5]);
      col.push(c.r, c.g, c.b, c.r, c.g, c.b);
    }
    const before = dots.length / 3;
    beads(L, 1.2, dots);
    for (let k = before; k < dots.length / 3; k++) dotCol.push(c.r, c.g, c.b);
  }
  setGeom(G.legs, pos, col);
  setGeom(G.legDots, dots, dotCol);
}

/** Everything the selected machine has walked up to and including step `t`. */
function drawTrack(r, t) {
  const R = roster();
  const pos = [], dots = [];
  if (r >= 0 && R) {
    for (let k = 0; k <= t && k < R.legs.length; k++) {
      const mv = legOf(k, r);
      if (!mv || mv.p < 0) continue;
      const L = leg(mv.p, mv.f, mv.v);
      for (let i = 0; i + 1 < L.n; i++) {
        pos.push(L.pts[3 * i], L.pts[3 * i + 1], L.pts[3 * i + 2],
                 L.pts[3 * i + 3], L.pts[3 * i + 4], L.pts[3 * i + 5]);
      }
      beads(L, 1.5, dots);
    }
    const c = robotColour(r);
    G.track.material.color.copy(c);
    G.trackDots.material.color.copy(c);
  }
  setGeom(G.track, pos);
  setGeom(G.trackDots, dots);
}

/** A polyline resampled at a fixed spacing: a one-pixel line disappears over a
 *  dense surface once the whole site is in frame, and beads do not. */
function beads(L, spacing, out) {
  let carry = 0;
  for (let i = 0; i + 1 < L.n; i++) {
    const d = L.cum[i + 1] - L.cum[i];
    for (let t = carry; t < d; t += spacing) {
      const f = t / Math.max(d, 1e-9);
      for (let c = 0; c < 3; c++) {
        out.push(L.pts[3 * i + c] + (L.pts[3 * i + 3 + c] - L.pts[3 * i + c]) * f);
      }
    }
    carry = d > 0 ? (spacing - ((d - carry) % spacing)) % spacing : carry;
  }
}

function selectRobot(r) {
  selMachine = r;
  if (r >= 0) {
    $("#vertex").value = "-1";
    $("#show-machines").checked = true;
  }
  paint();
}

/** The fleet list: one row per machine, what it is doing at this step. */
function fillFleet(t) {
  const R = roster();
  const box = $("#fleet");
  box.style.display = robotMode() ? "" : "none";
  if (!robotMode()) return;

  const rows = [];
  for (let r = 0; r < R.nMachines; r++) {
    const at = R.positions[t][r], mv = legOf(t, r), c = robotColour(r);
    const idle = at < 0 || !mv;
    const what = at < 0 ? "off site"
      : !mv ? "parked"
      : mv.s == null ? "no route"
      : mv.p < 0 ? (mv.e ? "walks on" : "holds")
      : `${mv.e ? "walks on " : ""}${mv.m.toFixed(0)} m · ${fmtTime(mv.s)}`;
    rows.push(`<div class="machine${r === selMachine ? " sel" : ""}`
      + `${idle ? " idle" : ""}" data-r="${r}" `
      + `style="color:#${c.getHexString()}">`
      + `<i style="background:#${c.getHexString()}"></i>`
      + `<b>R${r}</b><span>${at < 0 ? "—" : "v" + at}</span>`
      + `<span>${what}</span></div>`);
  }
  $("#fleet-list").innerHTML = rows.join("");

  const walked = R.legs.flat().reduce((a, mv) => a + (mv.m || 0), 0);
  const sel = selMachine >= 0
    ? ` · R${selMachine} has walked ${
        R.legs.slice(0, t + 1).flat()
          .filter((mv) => mv.r === selMachine)
          .reduce((a, mv) => a + (mv.m || 0), 0).toFixed(0)} m`
    : "";
  // A parked machine is standing at a vertex and does see it. The verifier
  // credits only the vertices the step names, so what is painted for a parked
  // machine is free sight the guarantee never used -- which is worth saying,
  // because the picture cannot show the difference.
  const idle = selMachine >= 0 && R.positions[t][selMachine] >= 0
    && !legOf(t, selMachine);
  $("#fleet-meta").textContent =
    `${R.nMachines} machines, R0–R${R.nMachines - 1}, on site at vertex ${R.entry}; `
    + `${(walked / 1000).toFixed(1)} km walked in all${sel}.`
    + (idle ? ` R${selMachine} is parked at this step: this step's guarantee `
              + `does not count what it sees.` : "");
}

/** Show or hide everything the machine view owns. */
function showRobots(on) {
  G.machines.visible = on;
  G.halo.visible = on && selMachine >= 0;
  G.legs.visible = on;
  G.legDots.visible = on;
  G.track.visible = on && selMachine >= 0;
  G.trackDots.visible = on && selMachine >= 0;
  G.held.visible = !on;
  if (!on) for (const sp of labels) sp.visible = false;   // placeRobots re-shows
}

function updateRobots() {
  const on = robotMode();
  showRobots(on);
  fillFleet(stepAt());
  if (!on) return;
  placeRobots(0);
  // Standing at a step, the legs shown are the ones that got the team here;
  // playing, they are the ones being walked right now.
  drawLegs(playing ? stepAt() + 1 : stepAt());
  drawTrack(selMachine, stepAt());
}

/* ----------------------------------------------------------------- panel */

function fillPanel() {
  const s = S.stats;
  $("#stats").innerHTML = [
    ["walkable area", `${s.areaM2.toLocaleString()} m²`],
    ["outside the boundary", S.boundary
      ? `${s.excludedM2.toLocaleString()} m² · ${S.boundary.cuts} cuts`
      : "—"],
    ["cells", s.cells.toLocaleString()],
    ["vertices", s.vertices],
    ["edges", `${s.edges} (${s.edgesShady} shady)`],
    ["detection range", `${s.range} m`],
    ["seen by nobody", `${s.uncoverableM2} m²`],
  ].map(([k, v]) => `<div><span>${k}</span><span>${v}</span></div>`).join("");

  const sel = $("#vertex");
  sel.innerHTML = '<option value="-1">— none —</option>';
  for (let v = 0; v < S.stats.vertices; v++) {
    const o = document.createElement("option");
    o.value = v;
    o.textContent = `vertex ${v}`;
    sel.appendChild(o);
  }

  const from = $("#route-from");
  from.innerHTML = '<option value="-1">— none —</option>';
  for (let v = 0; v < S.stats.vertices; v++) {
    const o = document.createElement("option");
    o.value = v;
    o.textContent = `vertex ${v}`;
    from.appendChild(o);
  }
  fillRouteTo();
  drawRoute();

  const pb = $("#playback");
  if (!S.strategy) {
    pb.style.display = "none";
    $("#fleet").style.display = "none";
    return;
  }
  pb.style.display = "";
  // Data written before the roster existed still loads; the toggle is simply
  // not offered, rather than offered and dead.
  $("#show-machines").disabled = !S.roster;
  $("#show-machines").parentElement.style.display = S.roster ? "" : "none";
  // The strategy shipped is the DRIVEN one -- one machine moving at a time,
  // nobody credited in transit -- so the line names the fleet and the avoidable
  // residual, which are what the clock verdict is about. `atomic_searchers` is
  // beside them because the two readings of the same plan disagree and hiding
  // that is the one thing this panel must not do.
  const m = S.strategy.metrics;
  const spent = m.machines_spent ? `, ${m.machines_spent} spent` : "";
  // BOTH residuals, never just the tolerant one. `avoidable` discounts every
  // cell no vertex of this graph can see, and it is 0.00 on all six scenes --
  // quoting it alone would read as "nothing is left", when what is left is
  // ground nobody could ever have watched. The raw figure is beside it.
  const rest = m.residual_m2 > m.residual_avoidable_m2
    ? ` (+${(m.residual_m2 - m.residual_avoidable_m2).toFixed(2)} m² unseeable)`
    : "";
  $("#strat-meta").textContent =
    `${S.strategy.key}: ${m.legs} legs, fleet ${m.fleet}` +
    ` (atomic peak ${m.atomic_searchers}${spent}), ` +
    `avoidable residual ${m.residual_avoidable_m2} m²${rest}, ` +
    `${(m.mission_seconds / 3600).toFixed(2)} h`;
  $("#step").max = S.strategy.steps.length - 1;
  $("#step").value = 0;
}

/** Which edges leave the chosen vertex, cheapest first. */
function edgesFrom(v) {
  const out = [];
  for (let k = 0; k < S.edges.length / 2; k++) {
    const i = S.edges[2 * k], j = S.edges[2 * k + 1];
    if (i === v || j === v) out.push({ k, other: i === v ? j : i });
  }
  out.sort((a, b) => S.routeSecs[a.k] - S.routeSecs[b.k]);
  return out;
}

function fillRouteTo() {
  const v = parseInt($("#route-from").value, 10);
  const to = $("#route-to");
  to.innerHTML = "";
  if (v < 0) {
    to.innerHTML = '<option value="-1">—</option>';
    return;
  }
  const all = document.createElement("option");
  all.value = "-1";
  all.textContent = "— every route from here —";
  to.appendChild(all);
  for (const { k, other } of edgesFrom(v)) {
    const o = document.createElement("option");
    o.value = other;                       // a vertex id, so #from=12&to=61 works
    o.textContent = `vertex ${other}  —  ${S.routeMetres[k].toFixed(0)} m, `
      + fmtTime(S.routeSecs[k]);
    to.appendChild(o);
  }
}

function fmtTime(s) {
  return s < 90 ? `${s.toFixed(0)} s` : `${(s / 60).toFixed(1)} min`;
}

function drawRoute() {
  const v = parseInt($("#route-from").value, 10);
  const w = parseInt($("#route-to").value, 10);
  const hit = v >= 0 ? edgesFrom(v).find((e) => e.other === w) : null;
  let which = [];
  let text = "";

  if (hit) {
    const k = hit.k;
    which = [k];
    const i = v, j = w;
    const straight = Math.hypot(
      S.vertices[3 * i] - S.vertices[3 * j],
      S.vertices[3 * i + 1] - S.vertices[3 * j + 1],
      S.vertices[3 * i + 2] - S.vertices[3 * j + 2]);
    const detour = S.routeMetres[k] / Math.max(straight, 1e-6);
    text = `${i} → ${j}: walks ${S.routeMetres[k].toFixed(1)} m in `
      + `${fmtTime(S.routeSecs[k])} — ${detour.toFixed(2)}× the straight line `
      + `of ${straight.toFixed(1)} m.`;
  } else if (v >= 0) {
    const es = edgesFrom(v);
    which = es.map((e) => e.k);
    const secs = es.map((e) => S.routeSecs[e.k]);
    text = es.length
      ? `${es.length} routes from vertex ${v}: `
        + `${fmtTime(Math.min(...secs))} to ${fmtTime(Math.max(...secs))}.`
      : `vertex ${v} has no graph edges.`;
  }

  $("#route-meta").textContent = text;
  G.routeSel.geometry.setAttribute("position",
    new THREE.Float32BufferAttribute(
      routeSegments(S, which, ROUTE_LIFT_SEL), 3));
  G.routeSel.geometry.attributes.position.needsUpdate = true;

  // A one-pixel line disappears over a dense surface once the whole site is in
  // frame. Beads along the path do not: they scale with the view.
  G.routeDots.geometry.setAttribute("position",
    new THREE.Float32BufferAttribute(
      routeBeads(S, which, 1.5, ROUTE_LIFT_SEL), 3));
  G.routeDots.geometry.attributes.position.needsUpdate = true;
}

function applyLayers() {
  G.cloud.visible = $("#l-cloud").checked;
  G.surface.visible = $("#l-surface").checked;
  G.regular.visible = $("#l-edges").checked;
  G.shady.visible = $("#l-edges").checked && $("#l-shady").checked;
  G.vertices.visible = $("#l-vertices").checked;
  G.routes.visible = $("#l-routes").checked;
  // A scene exported with --no-boundary has no line to draw. The toggle is
  // hidden rather than left checked over nothing, because an empty layer that
  // is switched on reads as a site with no boundary rather than as a scene
  // that was cut without one.
  const on = $("#l-boundary").checked && !!G.bound;
  $("#l-boundary").parentElement.style.display = G.bound ? "" : "none";
  if (G.bound) { G.bound.visible = on; G.boundPosts.visible = on; }
}

function togglePlay() {
  if (playing) {
    stopPlay();
    paint();
    return;
  }
  playing = true;
  phase = 0;
  lastMs = 0;
  $("#play").textContent = "Pause";
  $("#vertex").value = "-1";
  if (!robotMode()) $("#show-strat").checked = true;
  paint();
}

function stopPlay() {
  playing = false;
  phase = 0;
  $("#play").textContent = "Play";
}

/** Step by `d`, wrapping. The slider, the arrow keys and playback share it. */
function stepBy(d) {
  if (!S || !S.strategy) return;
  const el = $("#step");
  const n = +el.max + 1;
  el.value = ((+el.value + d) % n + n) % n;
  onScrub();
}

function onScrub() {
  $("#vertex").value = "-1";
  if (!robotMode()) $("#show-strat").checked = true;
  paint();
}

/* ------------------------------------------------------------- picking */

function pick(ev) {
  const r = renderer.domElement.getBoundingClientRect();
  const m = new THREE.Vector2(
    ((ev.clientX - r.left) / r.width) * 2 - 1,
    -((ev.clientY - r.top) / r.height) * 2 + 1);
  raycaster.setFromCamera(m, camera);

  const mode = $("#c-mode").value;
  if (mode !== "off") {
    if (mode === "start") {
      const hit = raycaster.intersectObject(G.vertices, false)[0];
      if (hit) addStartVertex(hit.index);
    } else {
      const hit = raycaster.intersectObject(G.surface, false)[0];
      if (hit) addHotspot(hit.index, parseInt(mode.split("-")[1], 10));
    }
    return;
  }

  if (robotMode()) {
    const who = raycaster.intersectObject(G.machines, false)[0];
    if (who) {
      const r = S.robotIds[who.index];
      stopPlay();
      selectRobot(r === selMachine ? -1 : r);
      return;
    }
  }
  const hit = raycaster.intersectObject(G.vertices, false)[0];
  if (!hit) return;
  $("#vertex").value = hit.index;
  selMachine = -1;
  stopPlay();
  paint();
}

/* ----------------------------------------------------- compute a strategy */

function hexToRgb01(hex) {
  return [((hex >> 16) & 255) / 255, ((hex >> 8) & 255) / 255, (hex & 255) / 255];
}

function addHotspot(cellIndex, category) {
  if (hotspots.some((h) => h.cell === cellIndex)) return;
  const xyz = [S.surf[3 * cellIndex], S.surf[3 * cellIndex + 1], S.surf[3 * cellIndex + 2]];
  hotspots.push({ cell: cellIndex, category, xyz });
  rebuildMarkers();
  fillMarkerList();
  schedulePriorsPreview();
}

function addStartVertex(vertexIndex) {
  if (startVertices.some((s) => s.vertex === vertexIndex)) return;
  const xyz = [S.vertices[3 * vertexIndex], S.vertices[3 * vertexIndex + 1],
              S.vertices[3 * vertexIndex + 2]];
  startVertices.push({ vertex: vertexIndex, xyz });
  rebuildMarkers();
  fillMarkerList();
}

function removeHotspot(i) {
  hotspots.splice(i, 1);
  rebuildMarkers();
  fillMarkerList();
  schedulePriorsPreview();
}
function removeStart(i) { startVertices.splice(i, 1); rebuildMarkers(); fillMarkerList(); }

function rebuildMarkers() {
  const hp = [], hc = [];
  for (const h of hotspots) {
    hp.push(h.xyz[0], h.xyz[1], h.xyz[2] + MARKER_LIFT);
    hc.push(...hexToRgb01(HOTSPOT_COLOR[h.category]));
  }
  setGeom(G.hotspots, hp, hc);

  const sp = [], sc = [];
  for (const s of startVertices) {
    sp.push(s.xyz[0], s.xyz[1], s.xyz[2] + MARKER_LIFT);
    sc.push(...hexToRgb01(START_COLOR));
  }
  setGeom(G.starts, sp, sc);
}

/** The hotspot/start-vertex list under the placement picker. */
function fillMarkerList() {
  const rows = [];
  const cat = ["low", "medium", "high"];
  hotspots.forEach((h, i) => {
    rows.push(`<div class="marker-row">`
      + `<i style="background:#${HOTSPOT_COLOR[h.category].toString(16).padStart(6, "0")}"></i>`
      + `<span>hotspot (${cat[h.category]}) &mdash; cell ${h.cell}</span>`
      + `<button data-rm="h${i}" title="remove">&times;</button></div>`);
  });
  startVertices.forEach((s, i) => {
    rows.push(`<div class="marker-row">`
      + `<i style="background:#${START_COLOR.toString(16).padStart(6, "0")}"></i>`
      + `<span>start &mdash; vertex ${s.vertex}</span>`
      + `<button data-rm="s${i}" title="remove">&times;</button></div>`);
  });
  $("#c-markers").innerHTML = rows.join("")
    || '<p class="hint">No markers placed yet.</p>';
}

async function runStrategy() {
  const body = {
    scene: S.name,
    approach: $("#c-approach").value,
    hotspots: hotspots.map((h) => ({ cell: h.cell, category: h.category })),
    start_vertices: startVertices.map((s) => s.vertex),
    available_robots: parseInt($("#c-robots").value, 10),
    stopping: {
      mode: $("#c-stop-mode").value,
      value: parseFloat($("#c-stop-value").value),
    },
    prior_l: parseFloat($("#c-prior-l").value),
    prior_radius_m: parseFloat($("#c-prior-radius").value),
  };

  $("#c-run").disabled = true;
  $("#c-status").textContent = "running…";
  try {
    const res = await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || res.statusText);
    setComputed(data);
    const m = data.metrics;
    $("#c-status").textContent = `done — ${m.checked_trees} trees checked, `
      + `${fmtTime(m.mission_seconds)} mission, ${m.vertices_visited}/${m.n_vertices} `
      + `vertices reached${m.fitness == null ? "" : `, fitness ${m.fitness.toFixed(2)}`}.`;
  } catch (err) {
    $("#c-status").textContent = "error: " + err.message;
  } finally {
    $("#c-run").disabled = false;
  }
}

function setComputed(data) {
  compPlaying = false;
  $("#c-play").textContent = "Play";
  for (const sp of compLabelPool.values()) {
    world.remove(sp);
    sp.material.map.dispose();
    sp.material.dispose();
  }
  compLabelPool = new Map();

  // Every coordinate already in S (vertices, surface, routes, ...) is stored
  // relative to S.centre -- that is what keeps float32/quantised precision
  // tight over a survey given in large absolute (real-world) coordinates. The
  // backend has no notion of that packing convention and rightly returns
  // plain absolute Scene coordinates, so a move's path has to be shifted into
  // the same centred frame here, once, before anything renders it -- this is
  // exactly the "packs floating outside the scene" bug: a move's in-transit
  // marker used to sit at raw-minus-nothing, offset from every other layer by
  // the full `centre` vector (tens of metres).
  const [cx, cy, cz] = S.centre;
  for (const mv of data.moves) {
    mv.path = mv.path.map((p) => [p[0] - cx, p[1] - cy, p[2] - cz]);
  }

  computed = data;
  compPaintKey = null;
  compSpeed = Math.max(1, data.metrics.mission_seconds / 20);
  if (data.priors) priorsView = decodedPriors(data.priors);

  const m = data.metrics;
  $("#c-meta").textContent = `${data.approach}: ${data.moves.length} moves, `
    + `${m.available_robots} robots, ${m.vertices_visited}/${m.n_vertices} `
    + `vertices reached, ${fmtTime(m.mission_seconds)} mission.`;

  const el = $("#c-time");
  el.max = String(Math.max(0, Math.ceil(data.metrics.mission_seconds)));
  el.value = "0";
  $("#c-playback").style.display = "";
  paintComputed(0);
}

/** How many robots are standing at each vertex at time `t`: a departed move's
 *  robots leave the source count and are in neither count until they arrive,
 *  exactly mirroring TrajectoryPlanning.computeRobotCounts. */
function robotCountsAt(t) {
  const counts = new Map();
  const add = (v, n) => counts.set(v, (counts.get(v) || 0) + n);
  if (computed.root) add(computed.root.vertex, computed.root.robots);
  for (const mv of computed.moves) {
    if (mv.t_departure <= t) add(mv.source, -mv.robots);
    if (mv.t_arrival <= t) add(mv.target, mv.robots);
  }
  return counts;
}

/** `f` (0..1) of the way along a [x,y,z][] polyline, by walked distance. */
function pointAlongPath(path, f) {
  f = Math.max(0, Math.min(1, f));
  if (path.length === 1) return path[0];
  const segLen = [];
  let total = 0;
  for (let i = 0; i + 1 < path.length; i++) {
    const a = path[i], b = path[i + 1];
    const d = Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]);
    segLen.push(d);
    total += d;
  }
  if (!total) return path[0];
  let want = f * total, acc = 0;
  for (let i = 0; i < segLen.length; i++) {
    if (acc + segLen[i] >= want || i === segLen.length - 1) {
      const u = segLen[i] ? (want - acc) / segLen[i] : 0;
      const a = path[i], b = path[i + 1];
      return [a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u, a[2] + (b[2] - a[2]) * u];
    }
    acc += segLen[i];
  }
  return path[path.length - 1];
}

/** Every "pack" of robots to draw at time `t`: one per occupied vertex (idle,
 *  by node position) and one per move currently in transit (by path
 *  position), each carrying the robot count it represents. */
function packPositionsAt(t) {
  const packs = [];
  for (const [v, n] of robotCountsAt(t)) {
    if (n > 0) packs.push({ pos: nodePoint(v), count: n, key: "v" + v });
  }
  for (const mv of computed.moves) {
    if (t < mv.t_departure || t >= mv.t_arrival) continue;
    const key = "m" + mv.source + "-" + mv.target + "-" + mv.t_departure;
    if (!mv.path || mv.path.length < 2) {
      packs.push({ pos: nodePoint(mv.target), count: mv.robots, key });
      continue;
    }
    const f = (t - mv.t_departure) / Math.max(mv.t_arrival - mv.t_departure, 1e-9);
    packs.push({ pos: pointAlongPath(mv.path, f), count: mv.robots, key });
  }
  return packs;
}

function placeComputedPacks(t) {
  const packs = packPositionsAt(t);
  const pos = [], col = [];
  const seen = new Set();
  const c = new THREE.Color(PACK_COLOR);
  for (const p of packs) {
    seen.add(p.key);
    pos.push(p.pos[0], p.pos[1], p.pos[2] + PACK_LIFT);
    col.push(c.r, c.g, c.b);

    const text = "×" + p.count;
    let sp = compLabelPool.get(p.key);
    if (!sp || sp.userData.text !== text) {
      if (sp) { world.remove(sp); sp.material.map.dispose(); sp.material.dispose(); }
      sp = labelSprite(text, c);
      sp.userData.text = text;
      compLabelPool.set(p.key, sp);
      world.add(sp);
    }
    sp.visible = true;
    sp.position.set(p.pos[0], p.pos[1], p.pos[2] + PACK_LIFT + S.labelSize * 1.1);
    const size = S.labelSize;
    sp.scale.set(size * sp.userData.aspect, size, 1);
  }
  for (const [key, sp] of compLabelPool) if (!seen.has(key)) sp.visible = false;
  setGeom(G.compPacks, pos, col);
}

/** Paint the surface for the strategy computed here: cleared (ever seen by a
 *  visited vertex), watched now (seen by a currently-occupied one), or still
 *  dirty -- the same three colours `paint()` uses for the shipped strategy,
 *  driven by `visit_time` instead of the shipped `contaminated[]` mask.
 *
 * The per-cell scan below is O(nCells) per held vertex, so it is skipped
 * unless the qualifying vertex sets actually changed since the last call --
 * which, at 30-60 calls/second while playing, is most of them: a scene's
 * cell count is in the hundreds of thousands and the held/watched sets only
 * change at the (far rarer) departure/arrival events. Only the pack markers
 * (`placeComputedPacks`, cheap: O(moves)) move every frame regardless.
 */
let compPaintKey = null;

function paintComputed(t) {
  if (!computed) return;

  const held = [];
  for (const [vStr, vt] of Object.entries(computed.visit_time)) {
    if (vt <= t) held.push(+vStr);
  }
  const heldNow = [];
  for (const [v, n] of robotCountsAt(t)) if (n > 0) heldNow.push(v);

  const showPriors = $("#c-show-priors").checked && priorsView;
  // Prefixed with the view kind, so toggling the heatmap on/off always
  // invalidates the cache even when the held/watched sets themselves didn't
  // change between the two calls.
  const key = (showPriors ? "H|" : "C|") + held.join(",") + "|"
    + heldNow.sort((a, b) => a - b).join(",");
  if (key !== compPaintKey) {
    compPaintKey = key;

    if (showPriors) {
      paintPriorHeatmap();
    } else {
      const col = G.surface.geometry.attributes.color.array;
      col.set(S.surfBase);

      const cleared = new Uint8Array(S.nCells);
      for (const v of held) {
        const m = dset(v);
        for (let i = 0; i < S.nCells; i++) if (m[i]) cleared[i] = 1;
      }
      const watchedNow = new Uint8Array(S.nCells);
      for (const v of heldNow) {
        const m = dset(v);
        for (let i = 0; i < S.nCells; i++) if (m[i]) watchedNow[i] = 1;
      }
      for (let i = 0; i < S.nCells; i++) {
        const c = watchedNow[i] ? WATCHED : (cleared[i] ? CLEARED : DIRTY);
        col[3 * i] = c[0]; col[3 * i + 1] = c[1]; col[3 * i + 2] = c[2];
      }
      G.surface.geometry.attributes.color.needsUpdate = true;
    }

    const hp = new Float32Array(held.length * 3);
    held.forEach((q, k) => {
      hp[3 * k] = S.nodeDraw[3 * q]; hp[3 * k + 1] = S.nodeDraw[3 * q + 1];
      hp[3 * k + 2] = S.nodeDraw[3 * q + 2];
    });
    setGeom(G.held, hp);
  }

  placeComputedPacks(t);

  $("#c-time-label").textContent =
    `${fmtTime(t)} of ${fmtTime(computed.metrics.mission_seconds)}`;
}

/* -------------------------------------------------------- prior heatmap */

function decodedPriors(packed) {
  return { codes: decode(packed.q, Uint16Array), min: packed.min, max: packed.max };
}

function paintPriorHeatmap() {
  const col = G.surface.geometry.attributes.color.array;
  const codes = priorsView.codes;
  for (let i = 0; i < S.nCells; i++) {
    const c = ramp(codes[i] / 65535, HEAT);
    col[3 * i] = c[0]; col[3 * i + 1] = c[1]; col[3 * i + 2] = c[2];
  }
  G.surface.geometry.attributes.color.needsUpdate = true;
}

/** Repaint the surface from whatever is currently available: a computed
 *  strategy (cleared/dirty or, if toggled, the heatmap), the heatmap alone
 *  (placed hotspots but no run yet), or the plain height ramp. */
function updateSurfaceView() {
  if (computed) {
    paintComputed(parseFloat($("#c-time").value));
    return;
  }
  if ($("#c-show-priors").checked && priorsView) {
    paintPriorHeatmap();
  } else {
    const col = G.surface.geometry.attributes.color.array;
    col.set(S.surfBase);
    G.surface.geometry.attributes.color.needsUpdate = true;
  }
}

/** Debounced live preview: refetches priors shortly after the last hotspot
 *  edit or l/radius change, the same reactivity the 2D grid tool gets for
 *  free from Streamlit rerunning on every widget change. Cheap on its own
 *  (no graph search) -- separate from /api/run so placing a hotspot doesn't
 *  have to wait for a full strategy computation. */
function schedulePriorsPreview() {
  if (!$("#c-show-priors").checked) return;
  clearTimeout(priorsPreviewTimer);
  priorsPreviewTimer = setTimeout(refreshPriorsPreview, 250);
}

async function refreshPriorsPreview() {
  const body = {
    scene: S.name,
    hotspots: hotspots.map((h) => ({ cell: h.cell, category: h.category })),
    prior_l: parseFloat($("#c-prior-l").value),
    prior_radius_m: parseFloat($("#c-prior-radius").value),
  };
  try {
    const res = await fetch("/api/priors", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await res.json();
    if (!res.ok) return;
    priorsView = decodedPriors(data.priors);
    updateSurfaceView();
  } catch {
    // A stale preview is not worth surfacing as an error -- the next edit,
    // or Run itself, will refresh it.
  }
}

function toggleCompPlay() {
  if (!computed) return;
  if (compPlaying) {
    compPlaying = false;
    $("#c-play").textContent = "Play";
    return;
  }
  if (parseFloat($("#c-time").value) >= +$("#c-time").max) $("#c-time").value = "0";
  compPlaying = true;
  compLastMs = 0;
  $("#c-play").textContent = "Pause";
}

/* -------------------------------------------------------------- camera */

let dist = 200, az = 0.9, el = 0.55, target = new THREE.Vector3();

function place() {
  camera.position.set(
    target.x + dist * Math.cos(el) * Math.cos(az),
    target.y + dist * Math.cos(el) * Math.sin(az),
    target.z + dist * Math.sin(el));
  camera.up.set(0, 0, 1);
  camera.lookAt(target);
}

function frame() {
  const box = new THREE.Box3().setFromBufferAttribute(
    G.surface.geometry.attributes.position);
  const size = box.getSize(new THREE.Vector3());
  target.copy(box.getCenter(new THREE.Vector3()));
  dist = Math.max(size.x, size.y) * 0.95;
  az = 0.9; el = 0.55;
  place();
}

function orbit(el0) {
  let down = null;
  el0.addEventListener("pointerdown", (e) => {
    down = { x: e.clientX, y: e.clientY, b: e.button };
    el0.setPointerCapture(e.pointerId);
  });
  el0.addEventListener("pointerup", (e) => {
    down = null;
    el0.releasePointerCapture(e.pointerId);
  });
  el0.addEventListener("pointermove", (e) => {
    if (!down) return;
    const dx = e.clientX - down.x, dy = e.clientY - down.y;
    down.x = e.clientX; down.y = e.clientY;
    if (down.b === 0) {
      az -= dx * 0.006;
      el = Math.max(-1.45, Math.min(1.45, el + dy * 0.006));
    } else {
      const s = dist * 0.0016;
      const right = new THREE.Vector3(-Math.sin(az), Math.cos(az), 0);
      const up = new THREE.Vector3(
        -Math.cos(az) * Math.sin(el), -Math.sin(az) * Math.sin(el), Math.cos(el));
      target.addScaledVector(right, -dx * s).addScaledVector(up, dy * s);
    }
    place();
  });
  el0.addEventListener("contextmenu", (e) => e.preventDefault());
  el0.addEventListener("wheel", (e) => {
    e.preventDefault();
    dist = Math.max(3, Math.min(4000, dist * Math.exp(e.deltaY * 0.0012)));
    place();
  }, { passive: false });
}

init();

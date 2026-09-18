/* The scene viewer.
 *
 * Everything it draws is a point cloud or a line set, so there is one geometry
 * per layer and no scene graph to speak of. Generic scene inspection (survey
 * layers, an edge's walked route, a vertex's detection set) works on any
 * scene with no strategy involved. The "Compute a strategy" panel drives
 * /api/run (strategy_service/) with a chosen approach, placed hotspots/start
 * vertices and search parameters, and plays back whatever it returns --
 * `paintComputed()` is the strategy-specific coloring, the rest below it is
 * the pack/marker rendering for it.
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

/* ---------------------------------------------------------- computed strategy
 *
 * A strategy computed here (DP/Greedy/... over /api/run) is a continuous-time
 * move list, not a discrete "steps: sets of held vertices" shape -- see
 * strategy_service/adapter.py. So it gets its own small playback model: a
 * scrubber over mission seconds, "packs" of robots (a move's `robots` travel
 * and arrive together, and the algorithms never give them individual
 * identity) drawn at nodes or in transit along a move's path, and
 * cleared/dirty/watched painted from each vertex's first-visit time against
 * the detection sets already loaded for this scene.
 */
const HOTSPOT_COLOR = [0xffe066, 0xff9d3d, 0xff3d3d];   // low / medium / high
const START_COLOR = 0x3dff9d;
const PACK_COLOR = 0x8fd4ff;
const MARKER_LIFT = 0.55;
const PACK_LIFT = 0.9;
// dark blue -> violet -> red -> amber -> pale yellow: same "cool to hot" read
// as the 2D grid tool's prior heatmap, over the same ramp() helper used for
// height/cloud colouring.
const HEAT = [[0, [0.06, 0.06, 0.28]], [0.35, [0.32, 0.08, 0.42]],
              [0.6, [0.77, 0.15, 0.18]], [0.85, [0.98, 0.55, 0.12]],
              [1, [1.0, 0.96, 0.35]]];

/* ------------------------------------------------------------------- scene */

// The scene's normal background -- also what "Export image" restores after a
// transparent/white capture, so it's named once rather than repeated as a
// literal at both call sites.
const BG_COLOR = 0x0b0f14;

let renderer, camera, world, raycaster;
let S = null;               // the loaded scene payload
let G = {};                 // the THREE objects of the current scene

// hotspot/start-vertex placement, and the strategy /api/run last computed --
// all scene-specific, reset in build()
let hotspots = [];          // { cell, category, xyz: [x,y,z] }
let startVertices = [];     // { vertex, xyz: [x,y,z] }
let computed = null;        // the /api/run response, or null
let compLabelPool = new Map();  // pack key -> id sprite, reused while stable
let compPlaying = false, compLastMs = 0, compSpeed = 1, compTime = 0;
let priorsView = null;      // { codes: Uint16Array, min, max } -- last computed/previewed priors
let priorsPreviewTimer = null;

function init() {
  // alpha + preserveDrawingBuffer cost nothing while world.background stays
  // opaque (which it does for every normal frame), but both are required for
  // exportScreenshot() below: alpha to make a transparent capture possible at
  // all, preserveDrawingBuffer so the buffer toDataURL reads isn't already
  // cleared for the next frame by the time it's called.
  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  $("#view").appendChild(renderer.domElement);
  camera = new THREE.PerspectiveCamera(52, 1, 0.2, 6000);
  world = new THREE.Scene();
  world.background = new THREE.Color(BG_COLOR);
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
  $("#vertex").onchange = () => { paint(); };

  $("#c-stop-mode").onchange = () => {
    $("#c-stop-label").textContent = $("#c-stop-mode").value === "time" ? "Seconds" : "Trees";
  };
  $("#c-markers").onclick = (e) => {
    const btn = e.target.closest("[data-rm]");
    if (!btn) return;
    const i = parseInt(btn.dataset.rm.slice(1), 10);
    if (btn.dataset.rm[0] === "h") removeHotspot(i); else removeStart(i);
  };
  $("#c-export").onclick = exportScreenshot;
  $("#c-save-hotspots").onclick = saveHotspots;
  $("#c-run").onclick = runStrategy;
  $("#c-play").onclick = toggleCompPlay;
  $("#c-time").oninput = () => {
    compPlaying = false;
    $("#c-play").textContent = "Play";
    compTime = parseFloat($("#c-time").value);
    paintComputed(compTime);
  };
  $("#c-show-detection").onchange = () => paintComputed(compTime);
  $("#c-show-priors").onchange = () => {
    if ($("#c-show-priors").checked && !priorsView) refreshPriorsPreview();
    else updateSurfaceView();
  };
  $("#c-prior-l").oninput = schedulePriorsPreview;
  $("#c-prior-radius").oninput = schedulePriorsPreview;
  $("#c-epsilon").oninput = schedulePriorsPreview;

  addEventListener("keydown", (e) => {
    if (e.key === "r" || e.key === "R") frame();
  });
  addEventListener("resize", resize);
  renderer.domElement.addEventListener("click", pick);
  orbit(renderer.domElement);

  resize();
  const want = hashState();
  sel.value = window.SCENE_INDEX.some((s) => s.name === want.scene)
    ? want.scene : window.SCENE_INDEX[0].name;
  loadScene(sel.value);
  // The loop is a clock, not just a repaint: playing a computed strategy
  // walks its packs along their paths, and that has to be told how much time
  // went by rather than how many frames.
  //
  // compTime -- not the #c-time slider's own .value -- is the actual clock.
  // A range input snaps whatever it's assigned to the nearest `step` (every
  // browser, confirmed for both Chromium and WebKit): at 60fps each frame's
  // advance is a fraction of a second, so reading .value back as the running
  // total silently discarded it below the step every single frame and the
  // clock never moved -- "the robots move a little, then nothing happens"
  // was this rounding, not a stall. The slider still gets that value on
  // every frame, but only to *display* it; nothing reads it back.
  (function loop(ms) {
    requestAnimationFrame(loop);
    if (compPlaying && computed) {
      const dt = compLastMs ? Math.min(ms - compLastMs, 250) : 0;
      compLastMs = ms;
      compTime += (dt / 1000) * compSpeed;
      const max = +$("#c-time").max;
      if (compTime >= max) { compTime = max; compPlaying = false; $("#c-play").textContent = "Play"; }
      $("#c-time").value = compTime;
      paintComputed(compTime);
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

/** Save the current view as a PNG, at a chosen background and resolution --
 *  the viewer's own dark scene background is right for orbiting a site
 *  on-screen but wrong for a figure meant to sit in a paper or slide, so this
 *  renders one extra frame with the background swapped (or dropped for a
 *  transparent PNG) and the pixel ratio bumped for supersampling, then puts
 *  everything back exactly as resize()/init() left it. */
function exportScreenshot() {
  if (!S) return;
  const status = $("#c-export-status");
  status.textContent = "rendering…";

  try {
    const w = innerWidth, h = innerHeight;
    const basePixelRatio = Math.min(devicePixelRatio, 2);
    const scale = parseFloat($("#c-export-scale").value);
    const bg = $("#c-export-bg").value;

    if (bg === "transparent") {
      world.background = null;
      renderer.setClearColor(0x000000, 0);
    } else {
      world.background = new THREE.Color(bg === "white" ? 0xffffff : BG_COLOR);
    }

    renderer.setPixelRatio(basePixelRatio * scale);
    renderer.setSize(w, h, false);   // false: keep the on-screen canvas size, grow only the drawing buffer
    camera.updateProjectionMatrix();
    renderer.render(world, camera);

    // A blob + object URL downloads reliably everywhere; a bare data: URI on
    // an <a download> does not -- Safari in particular just navigates to it
    // instead of saving, and at 4x on a large scene the data: URI can run to
    // tens of megabytes of base64, which some browsers silently refuse.
    renderer.domElement.toBlob((blob) => {
      // Put the canvas back the way resize()/init() left it before anything
      // else touches it, whether or not the encode below succeeds.
      world.background = new THREE.Color(BG_COLOR);
      renderer.setClearColor(0x000000, 1);
      renderer.setPixelRatio(basePixelRatio);
      renderer.setSize(w, h, false);
      camera.updateProjectionMatrix();
      renderer.render(world, camera);

      if (!blob) {
        status.textContent = "error: could not encode PNG";
        return;
      }
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${S.name}-${bg}-${scale}x.png`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
      status.textContent = "saved.";
    }, "image/png");
  } catch (err) {
    status.textContent = "error: " + err.message;
  }
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

  // walkable surface -- its colours are rewritten by paint()/paintComputed()
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
    // Width in cellSize units -- the same physical/quantisation scale already
    // used for the surface point size just above -- rather than in screen
    // pixels, which the GPU would clamp to 1 regardless of what's asked for.
    const halfWidth = data.cellSize * (kind ? 0.28 : 0.4);
    G[kind ? "shady" : "regular"] = ribbon(
      pos, halfWidth, kind ? 0x9c6b34 : 0xffa100, kind ? 0.7 : 1.0);
  }
  const np = new Float32Array(vertices.length);
  for (let i = 0; i < vertices.length; i += 3) {
    np[i] = vertices[i]; np[i + 1] = vertices[i + 1]; np[i + 2] = vertices[i + 2] + lift;
  }
  S.nodeDraw = np;
  G.vertices = points(np, null, 3.4, 0xffe14d);

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

  // the currently-held vertices (parked packs), drawn as small dots
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

  fillPanel();
  rebuildMarkers();
  fillMarkerList();
  $("#c-playback").style.display = "none";
  $("#c-status").textContent = "";
  $("#c-no-clearance-warning").hidden = true;
  $("#c-save-status").textContent = "";
  loadSavedHotspots();
  applyLayers();
  const want = hashState();
  if (want.vertex !== null) $("#vertex").value = want.vertex;
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

/** `#scene=christ-church&vertex=12`, `&from=3&to=9` -- a linkable view. */
function hashState() {
  const q = new URLSearchParams(location.hash.replace(/^#/, ""));
  const num = (k) => (q.has(k) ? parseInt(q.get(k), 10) : null);
  return { scene: q.get("scene"), vertex: num("vertex"), from: num("from"), to: num("to") };
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

/** A LineSegments-alike drawn as flat ribbon triangles instead of GL lines.
 *  `LineBasicMaterial.linewidth` is silently clamped to 1px on most desktop
 *  drivers (Windows ANGLE and macOS both do this for core-profile GL
 *  contexts, which is what WebGL uses) -- no material setting fixes that, the
 *  width has to be real geometry. The offset is taken in the XY plane
 *  (perpendicular to each segment's ground projection), which is the right
 *  ribbon orientation for edges running over near-planar terrain seen mostly
 *  from above, and keeps this independent of render resolution: it reads the
 *  same width whether on screen or supersampled for an export PNG. */
function ribbon(pairs, halfWidth, colour, opacity) {
  const tri = [];
  for (let i = 0; i + 5 < pairs.length; i += 6) {
    const ax = pairs[i], ay = pairs[i + 1], az = pairs[i + 2];
    const bx = pairs[i + 3], by = pairs[i + 4], bz = pairs[i + 5];
    let dx = bx - ax, dy = by - ay;
    const len = Math.hypot(dx, dy) || 1;
    dx /= len; dy /= len;
    const px = -dy * halfWidth, py = dx * halfWidth;
    tri.push(
      ax + px, ay + py, az, ax - px, ay - py, az, bx + px, by + py, bz,
      bx + px, by + py, bz, ax - px, ay - py, az, bx - px, by - py, bz,
    );
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(new Float32Array(tri), 3));
  const m = new THREE.MeshBasicMaterial({ color: colour, transparent: true, opacity, side: THREE.DoubleSide });
  return new THREE.Mesh(g, m);
}

function segments(xyz, colour, opacity) {
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(xyz, 3));
  return new THREE.LineSegments(g, new THREE.LineBasicMaterial({
    color: colour, transparent: true, opacity }));
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

/** The generic "pick any vertex, see what it sees" view -- independent of
 *  any strategy, computed or otherwise. */
function paint() {
  const col = G.surface.geometry.attributes.color.array;
  col.set(S.surfBase);

  const v = parseInt($("#vertex").value, 10);
  let held = [];

  if (v >= 0) {
    const m = dset(v);
    for (let i = 0; i < S.nCells; i++) {
      if (!m[i]) continue;
      col[3 * i] = SEEN_BY[0]; col[3 * i + 1] = SEEN_BY[1]; col[3 * i + 2] = SEEN_BY[2];
    }
    held = [v];
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
}

function nodePoint(v) {
  return [S.nodeDraw[3 * v], S.nodeDraw[3 * v + 1], S.nodeDraw[3 * v + 2]];
}

/** A robot pack's marker: a filled disc, with the count inside only when
 *  more than one robot travels together -- most moves here carry exactly
 *  one, and a lone disc reads as "a robot" far better than a bare "×1" ever
 *  did floating over a node. */
function packSprite(count) {
  const c = document.createElement("canvas");
  const size = 64;
  c.width = c.height = size;
  const g = c.getContext("2d");
  const r = size / 2 - 5;
  g.beginPath();
  g.arc(size / 2, size / 2, r, 0, 2 * Math.PI);
  g.fillStyle = "#" + new THREE.Color(PACK_COLOR).getHexString();
  g.fill();
  g.lineWidth = 4;
  g.strokeStyle = "rgba(7,11,16,0.85)";
  g.stroke();
  if (count > 1) {
    g.font = "700 30px ui-sans-serif, system-ui, -apple-system, sans-serif";
    g.textAlign = "center";
    g.textBaseline = "middle";
    g.fillStyle = "#0b0f14";
    g.fillText(String(count), size / 2, size / 2 + 1);
  }
  const sp = new THREE.Sprite(new THREE.SpriteMaterial({
    map: new THREE.CanvasTexture(c), transparent: true, depthTest: false }));
  sp.renderOrder = 10;
  return sp;
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
      if (hit) addHotspot([hit.point.x, hit.point.y, hit.point.z], parseInt(mode.split("-")[1], 10));
    }
    return;
  }

  const hit = raycaster.intersectObject(G.vertices, false)[0];
  if (!hit) return;
  $("#vertex").value = hit.index;
  paint();
}

/* ----------------------------------------------------- compute a strategy */

function hexToRgb01(hex) {
  return [((hex >> 16) & 255) / 255, ((hex >> 8) & 255) / 255, (hex & 255) / 255];
}

function addHotspot(xyz, category) {
  // A duplicate click is one that lands within a cell of an existing marker,
  // not one at the exact same float coordinates -- there is no "the" cell
  // index to compare any more, on purpose (see the request payload below).
  const dup = hotspots.some((h) => Math.hypot(
    h.xyz[0] - xyz[0], h.xyz[1] - xyz[1], h.xyz[2] - xyz[2]) < S.cellSize);
  if (dup) return;
  hotspots.push({ category, xyz });
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
      + `<span>hotspot (${cat[h.category]}) &mdash; `
      + `${h.xyz[0].toFixed(1)}, ${h.xyz[1].toFixed(1)}, ${h.xyz[2].toFixed(1)}</span>`
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

/** Hotspots for a request body, in absolute Scene coordinates -- S itself
 *  only ever holds the centred ones (see setComputed for why). */
function hotspotsPayload() {
  const [cx, cy, cz] = S.centre;
  return hotspots.map((h) => ({
    x: h.xyz[0] + cx, y: h.xyz[1] + cy, z: h.xyz[2] + cz, category: h.category,
  }));
}

/** Save the currently-placed hotspots/start vertices/prior settings for this
 *  scene, so they come back next time it's opened -- and so the Monte Carlo
 *  3D tests (Monte Carlo Simulation/3DTest.py) use a deliberately-placed
 *  layout instead of the arbitrary placeholder one. */
async function saveHotspots() {
  $("#c-save-hotspots").disabled = true;
  $("#c-save-status").textContent = "saving…";
  try {
    const res = await fetch(`/api/hotspots/${encodeURIComponent(S.name)}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        scene: S.name,
        hotspots: hotspotsPayload(),
        start_vertices: startVertices.map((s) => s.vertex),
        prior_l: parseFloat($("#c-prior-l").value),
        prior_radius_m: parseFloat($("#c-prior-radius").value),
        epsilon: parseFloat($("#c-epsilon").value),
      }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || res.statusText);
    $("#c-save-status").textContent =
      `saved ${hotspots.length} hotspot${hotspots.length === 1 ? "" : "s"}`
      + (startVertices.length
        ? `, ${startVertices.length} start ${startVertices.length === 1 ? "vertex" : "vertices"}`
        : "") + ".";
  } catch (err) {
    $("#c-save-status").textContent = "error: " + err.message;
  } finally {
    $("#c-save-hotspots").disabled = false;
  }
}

/** The counterpart to saveHotspots(), called once per scene load: restores
 *  whatever was last saved for it, if anything. */
async function loadSavedHotspots() {
  const sceneAtRequest = S.name;
  try {
    const res = await fetch(`/api/hotspots/${encodeURIComponent(sceneAtRequest)}`);
    if (!res.ok) return;
    const data = await res.json();
    if (S.name !== sceneAtRequest) return;  // the scene changed again while this was in flight

    const [cx, cy, cz] = S.centre;
    for (const h of data.hotspots || []) {
      hotspots.push({ category: h.category, xyz: [h.x - cx, h.y - cy, h.z - cz] });
    }
    for (const v of data.start_vertices || []) {
      if (v >= 0 && v < S.stats.vertices && !startVertices.some((s) => s.vertex === v)) {
        startVertices.push({
          vertex: v,
          xyz: [S.vertices[3 * v], S.vertices[3 * v + 1], S.vertices[3 * v + 2]],
        });
      }
    }
    if (data.prior_l != null) $("#c-prior-l").value = data.prior_l;
    if (data.prior_radius_m != null) $("#c-prior-radius").value = data.prior_radius_m;
    if (data.epsilon != null) $("#c-epsilon").value = data.epsilon;

    if (hotspots.length || startVertices.length) {
      rebuildMarkers();
      fillMarkerList();
      if ($("#c-show-priors").checked) refreshPriorsPreview();
    }
  } catch {
    // No saved config yet (or offline) -- markers just stay empty, same as
    // any other fresh scene.
  }
}

async function runStrategy() {
  const body = {
    scene: S.name,
    approach: $("#c-approach").value,
    hotspots: hotspotsPayload(),
    start_vertices: startVertices.map((s) => s.vertex),
    available_robots: parseInt($("#c-robots").value, 10),
    stopping: {
      mode: $("#c-stop-mode").value,
      value: parseFloat($("#c-stop-value").value),
    },
    prior_l: parseFloat($("#c-prior-l").value),
    prior_radius_m: parseFloat($("#c-prior-radius").value),
    epsilon: parseFloat($("#c-epsilon").value),
    prob_budget: parseFloat($("#c-prob-budget").value),
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
    $("#c-no-clearance-warning").hidden = !m.no_clearance_found;
  } catch (err) {
    $("#c-status").textContent = "error: " + err.message;
    $("#c-no-clearance-warning").hidden = true;
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
  // the same centred frame here, once, before anything renders it.
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

  compTime = 0;
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
  const seen = new Set();
  for (const p of packs) {
    seen.add(p.key);

    let sp = compLabelPool.get(p.key);
    if (!sp || sp.userData.count !== p.count) {
      if (sp) { world.remove(sp); sp.material.map.dispose(); sp.material.dispose(); }
      sp = packSprite(p.count);
      sp.userData.count = p.count;
      compLabelPool.set(p.key, sp);
      world.add(sp);
    }
    sp.visible = true;
    sp.position.set(p.pos[0], p.pos[1], p.pos[2] + PACK_LIFT);
    const size = S.labelSize * (p.count > 1 ? 1.3 : 1);
    sp.scale.set(size, size, 1);
  }
  for (const [key, sp] of compLabelPool) if (!seen.has(key)) sp.visible = false;
}

/** Paint the surface for the strategy computed here: cleared (ever seen by a
 *  visited vertex), watched now (seen by a currently-occupied one -- only
 *  while "Show detection sets" is on), or still dirty. The per-cell scan is
 *  O(nCells) per held vertex, so it is skipped unless the qualifying vertex
 *  sets (or the heatmap/detection toggle) actually changed since the last
 *  call -- which, at 30-60 calls/second while playing, is most of them: a
 *  scene's cell count is in the hundreds of thousands and the held/watched
 *  sets only change at the (far rarer) departure/arrival events. Only the
 *  pack markers (`placeComputedPacks`, cheap: O(moves)) move every frame
 *  regardless.
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
  heldNow.sort((a, b) => a - b);

  const showPriors = $("#c-show-priors").checked && priorsView;
  const showDetection = $("#c-show-detection").checked;
  const mode = showPriors ? "H" : (showDetection ? "W" : "C");
  const key = mode + "|" + held.join(",") + "|" + heldNow.join(",");
  if (key !== compPaintKey) {
    compPaintKey = key;

    const col = G.surface.geometry.attributes.color.array;
    if (showPriors) {
      paintPriorHeatmap();
    } else {
      col.set(S.surfBase);
      const cleared = new Uint8Array(S.nCells);
      for (const v of held) {
        const m = dset(v);
        for (let i = 0; i < S.nCells; i++) if (m[i]) cleared[i] = 1;
      }
      let watchedNow = null;
      if (showDetection) {
        watchedNow = new Uint8Array(S.nCells);
        for (const v of heldNow) {
          const m = dset(v);
          for (let i = 0; i < S.nCells; i++) if (m[i]) watchedNow[i] = 1;
        }
      }
      for (let i = 0; i < S.nCells; i++) {
        const c = (watchedNow && watchedNow[i]) ? WATCHED : (cleared[i] ? CLEARED : DIRTY);
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
    paintComputed(compTime);
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
    hotspots: hotspotsPayload(),
    prior_l: parseFloat($("#c-prior-l").value),
    prior_radius_m: parseFloat($("#c-prior-radius").value),
    epsilon: parseFloat($("#c-epsilon").value),
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
  if (compTime >= +$("#c-time").max) compTime = 0;
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

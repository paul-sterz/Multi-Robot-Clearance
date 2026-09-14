"""A small software renderer, so the README GIFs need nothing but numpy.

Points only -- the survey cloud, the walkable surface, the graph -- projected
with a pinhole camera and painted back to front into a z-buffer. Lines are
drawn by sampling points along them, which costs nothing extra and keeps the
whole rasteriser to one function.

There is no OpenGL context anywhere in this repository, which is the point: the
GIFs regenerate on a headless machine, in CI, or over ssh, with no display and
no browser. The interactive view lives in `web/` and uses three.js; this is for
the still frames.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# colour
# ---------------------------------------------------------------------------


def ramp(t: np.ndarray, stops: list[tuple[float, tuple[int, int, int]]]) -> np.ndarray:
    """Piecewise-linear colour ramp; `t` in [0, 1]."""
    t = np.clip(np.asarray(t, dtype=np.float64), 0, 1)
    out = np.zeros((t.size, 3), dtype=np.float64)
    for (t0, c0), (t1, c1) in zip(stops, stops[1:]):
        m = (t >= t0) & (t <= t1)
        if not m.any():
            continue
        f = ((t[m] - t0) / max(t1 - t0, 1e-9))[:, None]
        out[m] = np.array(c0) * (1 - f) + np.array(c1) * f
    return out


#: Height ramp for the walkable surface: deep teal to warm sand.
SURFACE = [(0.0, (18, 78, 92)), (0.5, (36, 148, 148)), (1.0, (226, 214, 160))]


# ---------------------------------------------------------------------------
# camera
# ---------------------------------------------------------------------------


@dataclass
class Camera:
    eye: np.ndarray
    target: np.ndarray
    up: np.ndarray = None
    fov_deg: float = 55.0
    near: float = 0.3

    def matrix(self) -> np.ndarray:
        up = np.array([0.0, 0.0, 1.0]) if self.up is None else np.asarray(self.up)
        f = np.asarray(self.target, float) - np.asarray(self.eye, float)
        f /= max(np.linalg.norm(f), 1e-9)
        if abs(f @ up) > 0.999:                       # looking straight down
            up = np.array([0.0, 1.0, 0.0])
        r = np.cross(f, up)
        r /= max(np.linalg.norm(r), 1e-9)
        u = np.cross(r, f)
        return np.stack([r, u, f])                    # rows: right, up, forward


def project(pts: np.ndarray, cam: Camera, w: int, h: int):
    """World points to pixel coordinates plus depth. Behind-camera points drop."""
    M = cam.matrix()
    rel = (np.asarray(pts, dtype=np.float32) - np.asarray(cam.eye, np.float32))
    cs = rel @ M.T.astype(np.float32)                 # (right, up, forward)
    z = cs[:, 2]
    ok = z > cam.near
    focal = 0.5 * h / np.tan(np.radians(cam.fov_deg) * 0.5)
    x = np.full(z.shape, -1e9, np.float32)
    y = np.full(z.shape, -1e9, np.float32)
    x[ok] = w * 0.5 + focal * cs[ok, 0] / z[ok]
    y[ok] = h * 0.5 - focal * cs[ok, 1] / z[ok]
    return x, y, z, ok


# ---------------------------------------------------------------------------
# the rasteriser
# ---------------------------------------------------------------------------


class Frame:
    """An RGB buffer with a depth test, painted one point cloud at a time."""

    def __init__(self, w: int, h: int, background=(11, 15, 20)):
        self.w, self.h = w, h
        self.rgb = np.tile(np.array(background, np.uint8), (h, w, 1))
        self.depth = np.full((h, w), np.inf, np.float32)

    def points(self, xyz: np.ndarray, cam: Camera, colour, radius: int = 0,
               fade: tuple[float, float] | None = None,
               size_m: float | None = None) -> None:
        """Draw points. `colour` is one RGB triple or one per point.

        `fade` dims distant points towards the background between the two given
        distances, which is what stops the far side of a site from reading as
        loudly as the near side and flattening the picture.

        `size_m` gives the points a physical size, so a 0.2 m surface cell two
        metres from the camera is drawn as the several pixels it covers instead
        of as one. Without it a walk-through over a lattice surface is a moire
        pattern of gaps rather than a floor.
        """
        if len(xyz) == 0:
            return
        if size_m is not None:
            focal = 0.5 * self.h / np.tan(np.radians(cam.fov_deg) * 0.5)
            _, _, z, _ = project(xyz, cam, self.w, self.h)
            with np.errstate(divide="ignore", invalid="ignore"):
                px = np.where(z > cam.near, focal * size_m / np.maximum(z, 1e-6), 0)
            band = np.clip(np.rint(px * 0.5), 0, 3).astype(np.int32)
            col = np.asarray(colour, np.float64)
            for r in np.unique(band):
                m = band == r
                sub = col if col.ndim == 1 else col[m]
                self.points(xyz[m], cam, sub, radius=int(max(r, radius)),
                            fade=fade)
            return
        x, y, z, ok = project(xyz, cam, self.w, self.h)
        ok &= (x >= -radius) & (x < self.w + radius)
        ok &= (y >= -radius) & (y < self.h + radius)
        if not ok.any():
            return
        idx = np.flatnonzero(ok)
        col = np.asarray(colour, np.float64)
        col = np.tile(col, (idx.size, 1)) if col.ndim == 1 else col[idx].astype(float)
        if fade is not None:
            lo, hi = fade
            k = np.clip((z[idx] - lo) / max(hi - lo, 1e-9), 0, 1)[:, None]
            col = col * (1 - 0.85 * k) + np.array([11, 15, 20]) * (0.85 * k)

        # back to front, so the depth test is a plain comparison
        order = np.argsort(-z[idx])
        idx, col = idx[order], col[order]
        px, py, pz = x[idx], y[idx], z[idx]
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                cx = np.rint(px + dx).astype(np.int32)
                cy = np.rint(py + dy).astype(np.int32)
                m = (cx >= 0) & (cx < self.w) & (cy >= 0) & (cy < self.h)
                if not m.any():
                    continue
                cx, cy, cz = cx[m], cy[m], pz[m]
                nearer = cz < self.depth[cy, cx]
                cx, cy, cz = cx[nearer], cy[nearer], cz[nearer]
                self.rgb[cy, cx] = col[m][nearer].astype(np.uint8)
                self.depth[cy, cx] = cz

    def lines(self, a: np.ndarray, b: np.ndarray, cam: Camera, colour,
              radius: int = 0, per_metre: float = 4.0) -> None:
        """Draw segments by sampling points along them."""
        a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
        if len(a) == 0:
            return
        length = np.linalg.norm(b - a, axis=1)
        n = int(np.clip(np.ceil(length.max() * per_metre), 2, 400))
        t = np.linspace(0, 1, n)[:, None, None]
        pts = (a[None] * (1 - t) + b[None] * t).reshape(-1, 3)
        col = np.asarray(colour, np.float64)
        if col.ndim == 2:
            col = np.tile(col, (n, 1))
        self.points(pts, cam, col, radius=radius)


# ---------------------------------------------------------------------------
# the camera path
# ---------------------------------------------------------------------------


def camera_path(scene, n_orbit: int = 48, n_fly: int = 0,
                elevation_deg: float = 30.0) -> list[Camera]:
    """One turn around the site, and optionally a pass through it afterwards.

    The turn is what the README GIFs use: it says how big the place is, how the
    walkable surface is laid out in it, and where the graph sits on top.

    The pass through (`n_fly > 0`) puts the camera at sensor height and walks it
    along the site's principal axis, which shows what the surface looks like
    from where the machine stands. It is off by default: at the size a GIF is read
    at, it costs more frames than it earns.
    """
    xyz = scene.cell_xyz
    centre = xyz.mean(axis=0)
    lo, hi = xyz.min(axis=0), xyz.max(axis=0)
    radius = float(np.linalg.norm((hi - lo)[:2])) * 0.55
    ground = float(np.median(xyz[:, 2]))

    cams = []
    for k in range(n_orbit):
        a = 2 * np.pi * k / n_orbit
        e = np.radians(elevation_deg)
        cams.append(Camera(
            eye=centre + np.array([radius * np.cos(a) * np.cos(e),
                                   radius * np.sin(a) * np.cos(e),
                                   radius * np.sin(e)]),
            target=centre + np.array([0.0, 0.0, -radius * 0.06]),
            fov_deg=50.0))

    # the pass through: along the surface's principal axis, at the height a
    # sensor sits at, looking a little ahead and a little down
    d = xyz[:, :2] - centre[:2]
    _, V = np.linalg.eigh(np.cov(d.T))
    axis = V[:, -1]
    span = float(np.percentile(d @ axis, 96) - np.percentile(d @ axis, 4)) * 0.5
    for k in range(n_fly):
        t = k / max(n_fly - 1, 1)
        s = (t * 2 - 1) * span
        eye = np.array([centre[0] + axis[0] * s, centre[1] + axis[1] * s,
                        ground + 9.0])
        ahead = np.array([eye[0] + axis[0] * 26, eye[1] + axis[1] * 26,
                          ground + 1.0])
        cams.append(Camera(eye=eye, target=ahead, fov_deg=62.0))
    return cams


# ---------------------------------------------------------------------------
# one frame of a scene
# ---------------------------------------------------------------------------


def draw_scene(scene, cam: Camera, w: int, h: int, cloud_stride: int = 1,
               surface_stride: int = 1) -> np.ndarray:
    """Survey cloud, walkable surface, guard graph -- in that order."""
    f = Frame(w, h)
    far = float(np.linalg.norm(scene.cell_xyz.max(0) - scene.cell_xyz.min(0)))

    if scene.cloud_xyz is not None:
        c = scene.cloud_xyz[::cloud_stride]
        z = c[:, 2]
        t = (z - np.percentile(z, 2)) / max(np.ptp(np.percentile(z, [2, 98])), 1e-6)
        col = ramp(t, [(0.0, (64, 71, 82)), (0.6, (116, 124, 138)),
                       (1.0, (176, 182, 194))])
        f.points(c, cam, col, fade=(far * 0.35, far * 1.6), size_m=0.28)

    s = scene.cell_xyz[::surface_stride]
    z = s[:, 2]
    t = (z - z.min()) / max(np.ptp(z), 1e-6)
    f.points(s, cam, ramp(t, SURFACE), fade=(far * 0.3, far * 1.4),
             size_m=scene.cell_size)

    # the scenario boundary, under the graph: the site stops here, and the
    # surface above stops with it. Two rails at the foot and the top of a low
    # fence -- see `Scene.site_boundary_fence` -- in the blue the viewer uses, which
    # is not the amber of the graph for the reason the viewer gives.
    if len(scene.site_boundary_seg):
        seg = np.asarray(scene.site_boundary_seg, np.float64)
        z = np.full((len(seg), 1), 0.0)
        for zc in scene.site_boundary_fence:
            f.lines(np.column_stack([seg[:, 0], z + zc]),
                    np.column_stack([seg[:, 1], z + zc]),
                    cam, (143, 212, 255), radius=0)

    lift = np.array([0.0, 0.0, 0.6])
    a = scene.vertex_xyz[scene.edge_ij[:, 0]] + lift
    b = scene.vertex_xyz[scene.edge_ij[:, 1]] + lift
    reg = ~scene.edge_shady
    f.lines(a[reg], b[reg], cam, (250, 176, 74), radius=0)
    f.lines(a[~reg], b[~reg], cam, (128, 96, 72), radius=0)
    f.points(scene.vertex_xyz + lift, cam, (255, 236, 168), radius=2)
    return f.rgb

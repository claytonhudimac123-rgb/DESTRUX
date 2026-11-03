"""
DESTRUX 2  —  auto-updating client (pairs with server.py build 8)

    pip install moderngl pygame-ce numpy scipy
    python destrux.py

Controls
    WASD move   SHIFT sprint   CTRL crouch   SPACE jump/jet
    LMB tool    1/2 pickaxe/launcher   E shop / multiplayer PC
    ESC pause   F5 save world   F9 load world   F11 fullscreen
    F1 hide HUD (trailers)   F3 settings (shaders + render dist)
    F10 +500 coins (OP only)   F2 spectator (OP only)

Multiplayer: lobby PC → E → join Official / localhost
Progress (coins/items): singleplayer_save.json offline; server_save/ online

Protocol 2 / client build 8:
  - join carries proto, client_build, epoch, since (resume)
  - server pushes sync + sync_end; dig/boom/collapse always edit MAIN
    world even while the player is in lobby/shop
  - client events carry cid; server acks with the stamped eid
  - render distance 64–452 m; insane shader pack is YT/trailer grade
"""
import base64
import hashlib
import json
import math
import os
import random
import select
import subprocess
import socket
import struct
import sys
import threading
import time
import uuid

SAVE_VERSION = 3
SAVE_SLOT = "world"                      # full world .npz basename inside client_save/
SP_PROGRESS_NAME = "progress"            # offline progress JSON basename
CLIENT_SAVE_DIR = "client_save"          # CLIENT-ONLY folder (never used by server)
SERVER_SAVE_DIR = "server_save"          # player progress on dedicated/host
SERVER_WORLD_DIR = "server_world"        # persistent dig/boom log on dedicated/host
AUTOSAVE_INTERVAL = 30.0  # coins/gear local + push to server

import numpy as np
import pygame
import moderngl

# Multiplayer defaults — playit.gg public tunnel for the 24/7 server
OFFICIAL_HOST = "shrimp-ruth.tun.ply.gg"
OFFICIAL_PORT = 58819
# Local dedicated server always binds this port (playit tunnels TO this):
LOCAL_SERVER_PORT = 27015
NET_TICK = 1.0 / 20.0
NET_RESYNC = 5.0
NET_EVENT_HISTORY = 120
NET_MAGIC = b"DX2\n"
CLIENT_UPDATE_TIMEOUT = 12.0
COLLAPSE_MAX_CELLS = 8000   # max cells in one collapse packet
PROTOCOL = 2                # must match server PROTOCOL
CLIENT_BUILD = 8            # advertised to dedicated server version gate
NET_BUILD = 8

try:
    from scipy import ndimage
except Exception:          # works without scipy, just slower collapse checks
    ndimage = None

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
VS = 0.25                       # voxel size in metres (old game used 1.0)
CS = 32                         # chunk size in voxels (8 m)
WIN_W, WIN_H = 1280, 720
HUD_W, HUD_H = 1280, 720
GRAVITY = 26.0
SENS = 0.0022
FOV = 75.0
MAX_DEBRIS = 14000
VIEW_DIST = 120.0
VIEW_DIST_MIN = 64.0
VIEW_DIST_MAX = 452.0

(AIR, GRASS, DIRT, STONE, ORE, BEDROCK, BRICK, CONCRETE, SAND, SNOW, WOOD, LEAF,
 TILE_A, TILE_B, PORTAL_A, PORTAL_B, LIGHT, SCREEN, METAL, RED, GOLD, PLANK,
 WATER, WHITE) = range(24)

PALETTE = np.array([
    (0, 0, 0), (0.30, 0.60, 0.24), (0.45, 0.31, 0.19), (0.50, 0.50, 0.53),
    (0.95, 0.78, 0.18), (0.13, 0.13, 0.16), (0.68, 0.30, 0.24), (0.72, 0.72, 0.76),
    (0.85, 0.78, 0.52), (0.93, 0.95, 0.98), (0.40, 0.27, 0.14), (0.18, 0.45, 0.16),
    (0.80, 0.78, 0.72), (0.36, 0.38, 0.46), (0.30, 0.85, 1.00), (1.00, 0.62, 0.20),
    (1.00, 0.93, 0.70), (0.35, 0.90, 1.00), (0.58, 0.60, 0.65), (0.80, 0.15, 0.12),
    (0.95, 0.76, 0.25), (0.62, 0.46, 0.26), (0.20, 0.50, 0.85), (0.95, 0.95, 0.95),
    (0.66, 0.78, 0.34), (0.78, 0.62, 0.30), (0.72, 0.43, 0.24),
    (0.78, 0.69, 0.35), (0.48, 0.67, 0.30), (0.42, 0.72, 0.48),
    (0.76, 0.62, 0.72), (0.16, 0.34, 0.22), (0.31, 0.48, 0.34),
    (0.22, 0.43, 0.28), (0.31, 0.46, 0.39), (0.18, 0.53, 0.42),
    (0.24, 0.64, 0.45), (0.58, 0.68, 0.73), (0.76, 0.86, 0.94),
    (0.58, 0.26, 0.17), (0.38, 0.29, 0.29), (0.49, 0.51, 0.58),
    (0.52, 0.73, 0.78), (0.72, 0.76, 0.70),
], dtype=np.float32)
EMIS = np.zeros(len(PALETTE), bool)
EMIS[[PORTAL_A, PORTAL_B, LIGHT, SCREEN]] = True
HARDNESS = np.array([0, .6, .5, 1.0, 1.6, 1e9, 1.2, 1.8, .4, .3, .7, .2, 1.4, 1.4,
                     1, 1, 1, 1, 2.0, 1.0, 1.5, .6, .1, 1] + [0.45] * 20, dtype=np.float64)
VALUE = np.array([0, 1, 1, 2, 10, 0, 3, 4, 1, 1, 1, .5, 0, 0, 0, 0, 0, 0, 0, 0, 25, 8, 0, 0] + [1.5] * 20,
                 dtype=np.float64)  # index 20 GOLD, 21 PLANK (loot crates)


# ----------------------------------------------------------------------------
# Math helpers
# ----------------------------------------------------------------------------
def perspective(fov_deg, aspect, near, far):
    f = 1.0 / math.tan(math.radians(fov_deg) / 2.0)
    m = np.zeros((4, 4), dtype=np.float64)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = 2.0 * far * near / (near - far)
    m[3, 2] = -1.0
    return m


def look_at(eye, forward, up=(0.0, 1.0, 0.0)):
    f = np.asarray(forward, dtype=np.float64)
    f = f / np.linalg.norm(f)
    s = np.cross(f, up)
    s = s / np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.eye(4, dtype=np.float64)
    m[0, :3], m[1, :3], m[2, :3] = s, u, -f
    m[0, 3] = -np.dot(s, eye)
    m[1, 3] = -np.dot(u, eye)
    m[2, 3] = np.dot(f, eye)
    return m


def frustum_planes(vp):
    p = np.array([vp[3] + vp[0], vp[3] - vp[0], vp[3] + vp[1],
                  vp[3] - vp[1], vp[3] + vp[2], vp[3] - vp[2]])
    return p / np.linalg.norm(p[:, :3], axis=1, keepdims=True)


def make_cube():
    verts = []
    for axis in range(3):
        u, v = [a for a in range(3) if a != axis]
        for sign in (0, 1):
            n = [0, 0, 0]
            n[axis] = 1 if sign else -1
            quad = []
            for a, b in [(0, 0), (1, 0), (1, 1), (0, 1)]:
                p = [0, 0, 0]
                p[axis], p[u], p[v] = sign, a, b
                quad.append(p)
            for i in (0, 1, 2, 0, 2, 3):
                verts.append(quad[i] + n)
    return np.array(verts, dtype=np.float32)


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def value_noise(w, d, cells, rng):
    g = rng.random((cells + 2, cells + 2))
    xs = np.linspace(0, cells, w, endpoint=False)
    zs = np.linspace(0, cells, d, endpoint=False)
    xi, zi = xs.astype(int), zs.astype(int)
    xf, zf = xs - xi, zs - zi
    xf, zf = xf * xf * (3 - 2 * xf), zf * zf * (3 - 2 * zf)
    a, b = g[xi][:, zi], g[xi + 1][:, zi]
    c, e = g[xi][:, zi + 1], g[xi + 1][:, zi + 1]
    top = a * (1 - xf)[:, None] + b * xf[:, None]
    bot = c * (1 - xf)[:, None] + e * xf[:, None]
    return top * (1 - zf)[None, :] + bot * zf[None, :]


def fbm(w, d, rng, base=4, octaves=6, gain=0.5):
    out, amp, tot = np.zeros((w, d)), 1.0, 0.0
    for i in range(octaves):
        out += amp * value_noise(w, d, base * 2 ** i, rng)
        tot += amp
        amp *= gain
    return out / tot


# ----------------------------------------------------------------------------
# Chunk mesher: only exposed faces, one VBO per 32^3 chunk
# ----------------------------------------------------------------------------
_QUADS = [
    [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)],
    [(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)],
    [(0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)],
    [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)],
    [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)],
    [(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)],
]
FACE_V = [np.array(q, np.float32)[[0, 1, 2, 0, 2, 3]] for q in _QUADS]
OFFS7 = [(0, 0, 0), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]


def mesh_chunk(world, key):
    x0, y0, z0 = key[0] * CS, key[1] * CS, key[2] * CS
    P = world.b[x0:x0 + CS + 2, y0:y0 + CS + 2, z0:z0 + CS + 2]
    inner = P[1:-1, 1:-1, 1:-1]
    if not inner.any():
        return None
    solid = inner != 0
    nbs = (P[2:, 1:-1, 1:-1], P[:-2, 1:-1, 1:-1], P[1:-1, 2:, 1:-1],
           P[1:-1, :-2, 1:-1], P[1:-1, 1:-1, 2:], P[1:-1, 1:-1, :-2])
    org = np.array([x0, y0, z0], np.int64)
    out = []
    for d in range(6):
        mask = solid & (nbs[d] == 0)
        idx = np.argwhere(mask)
        n = len(idx)
        if n == 0:
            continue
        t = inner[mask]
        g = idx + org
        h = ((g[:, 0] * 73856093) ^ (g[:, 1] * 19349663) ^ (g[:, 2] * 83492791)) & 1023
        shade = (0.88 + 0.16 * (h / 1023.0)).astype(np.float32)
        col = PALETTE[t] * shade[:, None]
        nid = np.where(EMIS[t], 6.0, float(d)).astype(np.float32)
        arr = np.empty((n, 6, 7), np.float32)
        arr[:, :, 0:3] = (g[:, None, :] + FACE_V[d][None, :, :]) * VS
        arr[:, :, 3:6] = col[:, None, :]
        arr[:, :, 6] = nid[:, None]
        out.append(arr.reshape(-1, 7))
    return np.concatenate(out) if out else None


# ----------------------------------------------------------------------------
# World (fine voxel grid with a 1-voxel solid border so meshing needs no edge cases)
# ----------------------------------------------------------------------------
class World:
    def __init__(self, W, H, D, destructible):
        self.W, self.H, self.D = W, H, D
        self.destructible = destructible
        self.b = np.zeros((W + 2, H + 2, D + 2), np.uint8)
        b = self.b
        b[0, :, :] = BEDROCK
        b[-1, :, :] = BEDROCK
        b[:, 0, :] = BEDROCK
        b[:, :, 0] = BEDROCK
        b[:, :, -1] = BEDROCK
        self.blocks = b[1:-1, 1:-1, 1:-1]
        self.nc = (W // CS, H // CS, D // CS)
        self.renderer = None

    # --- queries (metres) -------------------------------------------------
    def box_hit(self, x0, y0, z0, x1, y1, z1):
        W, H, D = self.W, self.H, self.D
        a0 = min(max(int(math.floor(x0 / VS)), -1), W)
        a1 = min(max(int(math.floor(x1 / VS)), -1), W)
        b0 = min(max(int(math.floor(y0 / VS)), -1), H)
        b1 = min(max(int(math.floor(y1 / VS)), -1), H)
        c0 = min(max(int(math.floor(z0 / VS)), -1), D)
        c1 = min(max(int(math.floor(z1 / VS)), -1), D)
        return bool(self.b[a0 + 1:a1 + 2, b0 + 1:b1 + 2, c0 + 1:c1 + 2].any())

    def solid_at(self, p):
        x = min(max(int(math.floor(p[0] / VS)), -1), self.W)
        y = min(max(int(math.floor(p[1] / VS)), -1), self.H)
        z = min(max(int(math.floor(p[2] / VS)), -1), self.D)
        return self.b[x + 1, y + 1, z + 1] != 0

    def solid_many(self, p):
        i = np.floor(p / VS).astype(np.int64)
        np.clip(i[:, 0], -1, self.W, out=i[:, 0])
        np.clip(i[:, 1], -1, self.H, out=i[:, 1])
        np.clip(i[:, 2], -1, self.D, out=i[:, 2])
        return self.b[i[:, 0] + 1, i[:, 1] + 1, i[:, 2] + 1] != 0

    def raycast(self, origin, direction, max_dist):
        """Voxel DDA. Returns (x, y, z) voxel index of first solid voxel or None."""
        ox, oy, oz = (float(v) / VS for v in origin)
        dx, dy, dz = (float(v) for v in direction)
        x, y, z = math.floor(ox), math.floor(oy), math.floor(oz)
        sx, sy, sz = (1 if dx > 0 else -1), (1 if dy > 0 else -1), (1 if dz > 0 else -1)
        inf = float("inf")
        tdx = abs(1.0 / dx) if dx else inf
        tdy = abs(1.0 / dy) if dy else inf
        tdz = abs(1.0 / dz) if dz else inf
        tmx = ((x + (1 if sx > 0 else 0)) - ox) / dx if dx else inf
        tmy = ((y + (1 if sy > 0 else 0)) - oy) / dy if dy else inf
        tmz = ((z + (1 if sz > 0 else 0)) - oz) / dz if dz else inf
        maxt, t = max_dist / VS, 0.0
        W, H, D, b = self.W, self.H, self.D, self.b
        while t <= maxt:
            if 0 <= x < W and 0 <= y < H and 0 <= z < D and b[x + 1, y + 1, z + 1]:
                return (x, y, z)
            if tmx < tmy and tmx < tmz:
                x += sx
                t = tmx
                tmx += tdx
            elif tmy < tmz:
                y += sy
                t = tmy
                tmy += tdy
            else:
                z += sz
                t = tmz
                tmz += tdz
        return None

    # --- editing ---------------------------------------------------------
    def touch(self, lo, hi):
        if self.renderer is not None:
            self.renderer.mark(lo, hi)

    def bm(self, x0, y0, z0, x1, y1, z1, t):
        """Fill a box given in metres."""
        a = [int(round(v / VS)) for v in (x0, y0, z0, x1, y1, z1)]
        lim = (self.W, self.H, self.D)
        lo = [max(0, min(a[i], lim[i])) for i in range(3)]
        hi = [max(0, min(a[i + 3], lim[i])) for i in range(3)]
        if all(hi[i] > lo[i] for i in range(3)):
            self.blocks[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = t

    def sphere(self, c, r, t, mode="all"):
        """Fill a sphere (centre/radius in voxels)."""
        lim = (self.W, self.H, self.D)
        lo = [max(int(math.floor(c[i] - r - 1)), 0) for i in range(3)]
        hi = [min(int(math.ceil(c[i] + r + 2)), lim[i]) for i in range(3)]
        if any(hi[i] <= lo[i] for i in range(3)):
            return
        sub = self.blocks[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        ax = [np.arange(lo[i], hi[i]) + 0.5 - c[i] for i in range(3)]
        d2 = ax[0][:, None, None] ** 2 + ax[1][None, :, None] ** 2 + ax[2][None, None, :] ** 2
        m = d2 < r * r
        if mode == "air":
            m &= (sub == 0)
        elif mode == "stone":
            m &= (sub == STONE)
        else:
            m &= (sub != BEDROCK)
        sub[m] = t
        self.touch(lo, hi)

    def sph(self, cx, cy, cz, r, t, mode="all"):
        self.sphere((cx / VS, cy / VS, cz / VS), r / VS, t, mode)

    def cyl(self, cx, cz, r_out, y0, y1, t, r_in=0.0):
        """Vertical cylinder / ring, metres."""
        i0, i1 = max(int((cx - r_out) / VS) - 1, 0), min(int((cx + r_out) / VS) + 2, self.W)
        j0, j1 = max(int((cz - r_out) / VS) - 1, 0), min(int((cz + r_out) / VS) + 2, self.D)
        ya, yb = max(int(round(y0 / VS)), 0), min(int(round(y1 / VS)), self.H)
        if i1 <= i0 or j1 <= j0 or yb <= ya:
            return
        X = (np.arange(i0, i1) + 0.5) * VS
        Z = (np.arange(j0, j1) + 0.5) * VS
        rr = np.hypot(X[:, None] - cx, Z[None, :] - cz)
        m = (rr < r_out) & (rr >= r_in)
        sub = self.blocks[i0:i1, ya:yb, j0:j1]
        sub[np.broadcast_to(m[:, None, :], sub.shape)] = t

    def carve_sphere(self, c, r, rough=0.0, seed=None):
        """Remove voxels in a sphere (voxel coords). Returns (types, idx, lo, hi)."""
        lim = (self.W, self.H, self.D)
        lo = [max(int(math.floor(c[i] - r - 1)), 0) for i in range(3)]
        hi = [min(int(math.ceil(c[i] + r + 2)), lim[i]) for i in range(3)]
        empty = (np.zeros(0, np.int64), np.zeros((0, 3), np.int64), lo, hi)
        if any(hi[i] <= lo[i] for i in range(3)):
            return empty
        sub = self.blocks[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        ax = [np.arange(lo[i], hi[i]) + 0.5 - c[i] for i in range(3)]
        d2 = ax[0][:, None, None] ** 2 + ax[1][None, :, None] ** 2 + ax[2][None, None, :] ** 2
        rng = np.random.default_rng(seed) if seed is not None else np.random
        rr = r + (rng.uniform(-rough, rough, d2.shape) if rough else 0.0)
        mask = (d2 < rr * rr) & (sub != 0) & (sub != BEDROCK)
        idx = np.argwhere(mask)
        if len(idx) == 0:
            return empty
        types = sub[mask].astype(np.int64)
        sub[mask] = 0
        self.touch(lo, hi)
        return types, idx + np.array(lo), lo, hi


def collapse_region(world, lo, hi, margin):
    """Remove and return voxels no longer connected to the ground near a damaged area."""
    W, H, D = world.W, world.H, world.D
    x0, x1 = max(lo[0] - margin, 0), min(hi[0] + margin, W)
    z0, z1 = max(lo[2] - margin, 0), min(hi[2] + margin, D)
    y0 = max(lo[1] - 24, 0)
    col = world.blocks[x0:x1, y0:H, z0:z1]
    ys = np.flatnonzero(col.any(axis=(0, 2)))
    if len(ys) == 0:
        return None
    sub = col[:, :ys[-1] + 1, :]
    solid = sub != 0
    if ndimage is not None:
        lab, nl = ndimage.label(solid)
        ok = np.zeros(nl + 1, bool)
        for f in (lab[0], lab[-1], lab[:, 0], lab[:, :, 0], lab[:, :, -1]):
            ok[np.unique(f)] = True
        ok[0] = True
        loose = solid & ~ok[lab]
    else:
        sup = np.zeros_like(solid)
        sup[0] = solid[0]
        sup[-1] |= solid[-1]
        sup[:, 0] |= solid[:, 0]
        sup[:, :, 0] |= solid[:, :, 0]
        sup[:, :, -1] |= solid[:, :, -1]
        for _ in range(500):
            g = sup.copy()
            g[1:] |= sup[:-1]
            g[:-1] |= sup[1:]
            g[:, 1:] |= sup[:, :-1]
            g[:, :-1] |= sup[:, 1:]
            g[:, :, 1:] |= sup[:, :, :-1]
            g[:, :, :-1] |= sup[:, :, 1:]
            g &= solid
            if np.array_equal(g, sup):
                break
            sup = g
        loose = solid & ~sup
    if not loose.any():
        return None
    idx = np.argwhere(loose) + np.array([x0, y0, z0])
    types = sub[loose].astype(np.int64)
    sub[loose] = 0
    return idx, types


# ----------------------------------------------------------------------------
# Chunk renderer
# ----------------------------------------------------------------------------
class ChunkRenderer:
    def __init__(self, ctx, prog, world):
        self.ctx, self.prog, self.world = ctx, prog, world
        world.renderer = self
        self.meshes = {}
        nx, ny, nz = world.nc
        self.dirty = {(x, y, z) for x in range(nx) for y in range(ny) for z in range(nz)}
        self._keys, self._cent, self._stale = [], np.zeros((0, 3)), True
        self._dirty_order = []
        self._dirty_cursor = 0
        self._order_key = None
        self.idle_skip = 0

    def mark(self, lo, hi):
        self.idle_skip = 0
        self._order_key = None
        n = self.world.nc
        r = [range(max((int(lo[i]) - 1) // CS, 0), min(int(hi[i]) // CS, n[i] - 1) + 1)
             for i in range(3)]
        for x in r[0]:
            for y in r[1]:
                for z in r[2]:
                    self.dirty.add((x, y, z))

    def mark_cells(self, cells):
        self.idle_skip = 0
        self._order_key = None
        n = np.array(self.world.nc) - 1
        for off in OFFS7:
            c = np.clip((cells + np.array(off)) // CS, 0, n)
            self.dirty.update(map(tuple, np.unique(c, axis=0).tolist()))

    def build(self, key):
        data = mesh_chunk(self.world, key)
        old = self.meshes.pop(key, None)
        if old:
            old[1].release()
            old[0].release()
        if data is not None and len(data):
            vbo = self.ctx.buffer(data.tobytes())
            vao = self.ctx.vertex_array(
                self.prog, [(vbo, "3f 3f 1f", "in_pos", "in_col", "in_nid")])
            self.meshes[key] = (vbo, vao, len(data))
        self._stale = True

    def update(self, cam, budget, maxd):
        if not self.dirty:
            return
        if self.idle_skip > 0:
            self.idle_skip -= 1
            return
        cs = CS * VS
        t0 = time.perf_counter()
        lim = maxd * maxd
        cam_chunk = tuple(int(math.floor(v / cs)) for v in cam)
        order_key = (cam_chunk, float(maxd))

        def d2(k):
            return ((k[0] + .5) * cs - cam[0]) ** 2 + ((k[1] + .5) * cs - cam[1]) ** 2 + \
                   ((k[2] + .5) * cs - cam[2]) ** 2
        if self._order_key != order_key:
            self._dirty_order = sorted((k for k in self.dirty if d2(k) < lim), key=d2)
            self._dirty_cursor = 0
            self._order_key = order_key
        while self._dirty_cursor < len(self._dirty_order):
            k = self._dirty_order[self._dirty_cursor]
            self._dirty_cursor += 1
            if k not in self.dirty:
                continue
            self.build(k)
            self.dirty.discard(k)
            if time.perf_counter() - t0 > budget:
                return
        if self._dirty_cursor >= len(self._dirty_order):
            self.idle_skip = 15

    def draw(self, planes, cam, maxd):
        if self._stale:
            self._keys = list(self.meshes.keys())
            self._cent = ((np.array(self._keys, np.float64) + 0.5) * (CS * VS)
                          if self._keys else np.zeros((0, 3)))
            self._stale = False
        if not self._keys:
            return
        rad = CS * VS * 0.87
        d = self._cent @ planes[:, :3].T + planes[:, 3]
        vis = (d > -rad).all(axis=1)
        vis &= np.linalg.norm(self._cent - cam, axis=1) < maxd + rad
        for i in np.flatnonzero(vis):
            _, vao, n = self.meshes[self._keys[i]]
            vao.render(moderngl.TRIANGLES, vertices=n)


# ----------------------------------------------------------------------------
# Debris / rubble: vectorised, bounces, slides, and settles back into the world
# ----------------------------------------------------------------------------
class Debris:
    def __init__(self):
        M = MAX_DEBRIS
        self.n = 0
        self.pos = np.zeros((M, 3), np.float32)
        self.vel = np.zeros((M, 3), np.float32)
        self.size = np.zeros(M, np.float32)
        self.col = np.zeros((M, 3), np.float32)
        self.life = np.zeros(M, np.float32)
        self.grav = np.ones(M, np.float32)
        self.typ = np.zeros(M, np.uint8)
        self.settle = np.zeros(M, bool)
        self.rest = np.zeros(M, np.float32)
        self._arrs = [self.pos, self.vel, self.size, self.col, self.life, self.grav,
                      self.typ, self.settle, self.rest]

    def clear(self):
        self.n = 0

    def add(self, pos, vel, size, col, life, grav=1.0, typ=0, settle=False):
        pos = np.atleast_2d(np.asarray(pos, np.float32))
        k = len(pos)
        if k == 0:
            return
        bc = np.broadcast_to
        vel = bc(np.asarray(vel, np.float32), (k, 3))
        col = bc(np.asarray(col, np.float32), (k, 3))
        size = bc(np.asarray(size, np.float32), (k,))
        life = bc(np.asarray(life, np.float32), (k,))
        grav = bc(np.asarray(grav, np.float32), (k,))
        typ = bc(np.asarray(typ, np.uint8), (k,))
        settle = bc(np.asarray(settle, bool), (k,))
        if k > MAX_DEBRIS:
            sl = slice(-MAX_DEBRIS, None)
            pos, vel, col, size, life, grav, typ, settle = (
                a[sl] for a in (pos, vel, col, size, life, grav, typ, settle))
            k = MAX_DEBRIS
        if self.n + k > MAX_DEBRIS:
            drop = self.n + k - MAX_DEBRIS
            for arr in self._arrs:
                arr[:self.n - drop] = arr[drop:self.n].copy()
            self.n -= drop
        s = slice(self.n, self.n + k)
        self.pos[s], self.vel[s], self.size[s], self.col[s] = pos, vel, size, col
        self.life[s], self.grav[s], self.typ[s], self.settle[s] = life, grav, typ, settle
        self.rest[s] = 0.0
        self.n += k

    def update(self, dt, world):
        n = self.n
        if n == 0:
            return
        p, v = self.pos[:n], self.vel[:n]
        v[:, 1] -= GRAVITY * self.grav[:n] * dt
        v *= max(0.0, 1.0 - 0.12 * dt)
        ground = np.zeros(n, bool)
        for ax in range(3):
            q = p.copy()
            q[:, ax] += v[:, ax] * dt
            hit = world.solid_many(q)
            p[~hit, ax] = q[~hit, ax]
            if hit.any():
                if ax == 1:
                    ground = hit & (v[:, 1] < 0)
                v[hit, ax] *= -0.28
                f = 0.72 if ax == 1 else 0.9
                for a in range(3):
                    if a != ax:
                        v[hit, a] *= f
        slow = ground & (np.abs(v[:, 1]) < 1.4)
        v[slow, 1] = 0.0
        resting = ground & ((v * v).sum(1) < 0.9)
        self.rest[:n][resting] += dt
        self.rest[:n][~resting] = 0.0

        cand = np.flatnonzero(self.settle[:n] & (self.rest[:n] > 0.5))[:350]
        if len(cand):
            cells = np.floor(p[cand] / VS).astype(np.int64)
            cx, cy, cz = cells[:, 0], cells[:, 1], cells[:, 2]
            inb = (cx >= 0) & (cx < world.W) & (cy >= 0) & (cy < world.H) & \
                  (cz >= 0) & (cz < world.D)
            cx, cy, cz, cand_i = cx[inb], cy[inb], cz[inb], cand[inb]
            if len(cx):
                free = (world.b[cx + 1, cy + 1, cz + 1] == 0) & (world.b[cx + 1, cy, cz + 1] != 0)
                world.blocks[cx[free], cy[free], cz[free]] = self.typ[cand_i[free]]
                if free.any() and world.renderer is not None:
                    world.renderer.mark_cells(np.stack([cx[free], cy[free], cz[free]], 1))
            self.life[cand] = 0.0

        self.life[:n] -= dt
        alive = self.life[:n] > 0
        if not alive.all():
            m = int(alive.sum())
            for arr in self._arrs:
                arr[:m] = arr[:n][alive]
            self.n = m

    def instances(self):
        n = self.n
        data = np.empty((n, 7), np.float32)
        sz = self.size[:n] * np.clip(self.life[:n] * 2.0, 0.0, 1.0)
        data[:, 0:3] = self.pos[:n] - sz[:, None] / 2.0
        data[:, 3] = sz
        data[:, 4:7] = self.col[:n]
        return data


# ----------------------------------------------------------------------------
# Procedural sound
# ----------------------------------------------------------------------------
SR = 44100


def _noise(dur, decay, smooth=1):
    t = np.linspace(0, dur, int(SR * dur), False)
    n = np.random.uniform(-1, 1, len(t))
    if smooth > 1:
        n = np.convolve(n, np.ones(smooth) / smooth, "same") * math.sqrt(smooth)
    return n * np.exp(-t * decay), t


def _tone(freq, dur, decay):
    t = np.linspace(0, dur, int(SR * dur), False)
    return np.sin(2 * np.pi * freq * t) * np.exp(-t * decay)


def _soft_tone(freq, dur, attack=0.08, release=0.25):
    """Gentle sustained tone with attack/release (no click)."""
    n = max(1, int(SR * dur))
    t = np.linspace(0, dur, n, False)
    wave = np.sin(2 * np.pi * freq * t)
    wave += 0.35 * np.sin(2 * np.pi * freq * 1.003 * t)
    wave += 0.12 * np.sin(2 * np.pi * freq * 2.0 * t)
    a = max(1, int(SR * attack))
    r = max(1, int(SR * release))
    env = np.ones(n, dtype=np.float64)
    env[:a] = np.linspace(0, 1, a)
    if r < n:
        env[-r:] = np.linspace(1, 0, r)
    return (wave * env).astype(np.float64)


def _make_bgm_loop(seconds=20.0):
    """Procedural chill exploration bed — warm pads + soft pulse + sparse arps."""
    n = int(SR * seconds)
    out = np.zeros(n, dtype=np.float64)
    bpm = 78.0
    beat = 60.0 / bpm
    # Dm – Bb – F – C
    chords = [
        (146.83, 174.61, 220.00),
        (116.54, 174.61, 233.08),
        (174.61, 220.00, 261.63),
        (130.81, 196.00, 261.63),
    ]
    bar = beat * 4.0
    for i, chord in enumerate(chords * 2):
        start = int(i * bar * SR)
        if start >= n:
            break
        dur = min(bar * 1.05, (n - start) / SR)
        pad = np.zeros(int(SR * dur))
        for f in chord:
            pad += _soft_tone(f, dur, attack=0.35, release=0.9)
            pad += 0.25 * _soft_tone(f * 0.5, dur, attack=0.5, release=1.0)
        pad /= max(1.0, len(chord))
        end = min(n, start + len(pad))
        out[start:end] += 0.22 * pad[: end - start]

    roots = [73.42, 58.27, 87.31, 65.41] * 2
    for i, f in enumerate(roots):
        start = int(i * bar * SR)
        if start >= n:
            break
        dur = min(beat * 1.5, (n - start) / SR)
        bass = _soft_tone(f, dur, attack=0.05, release=0.6)
        end = min(n, start + len(bass))
        out[start:end] += 0.16 * bass[: end - start]

    arp_notes = [440.0, 523.25, 587.33, 659.25, 587.33, 523.25]
    for k, f in enumerate(arp_notes * 6):
        start = int((0.5 + k * beat * 0.5) * SR)
        if start >= n - 1000:
            break
        if k % 3 == 2:
            continue
        dur = min(0.35, (n - start) / SR)
        note = _soft_tone(f, dur, attack=0.02, release=0.25)
        end = min(n, start + len(note))
        out[start:end] += 0.07 * note[: end - start]

    rng = np.random.default_rng(42)
    air = rng.standard_normal(n)
    ksz = 400
    air = np.convolve(air, np.ones(ksz) / ksz, mode="same")
    out += 0.03 * air

    fade = int(0.6 * SR)
    out[:fade] *= np.linspace(0, 1, fade)
    out[-fade:] *= np.linspace(1, 0, fade)
    peak = np.max(np.abs(out)) or 1.0
    out = out / peak * 0.35
    return out.astype(np.float64)


class Sfx:
    def __init__(self):
        self.ok = False
        self.sounds = {}
        self.music = None
        self.music_on = True
        self._music_channel = None
        try:
            pygame.mixer.init(SR, -16, 2, 512)
            pygame.mixer.set_num_channels(16)
            self.ok = True
        except pygame.error:
            return
        n, _ = _noise(0.10, 35, 3)
        self._add("mine", 0.8 * n + 0.5 * _tone(170, 0.10, 40), 0.5)
        n, _ = _noise(0.35, 9, 10)
        self._add("break", n * 1.4, 0.6)
        self._add("coin", np.concatenate([_tone(988, 0.07, 18), _tone(1319, 0.16, 14)]), 0.35)
        n, _ = _noise(1.5, 2.8, 60)
        self._add("boom", n * 2.5 + 0.9 * _tone(52, 1.5, 2.5), 0.9)
        n, _ = _noise(0.4, 7, 6)
        self._add("launch", n * 1.5, 0.6)
        n, _ = _noise(0.14, 14, 3)
        self._add("jet", n * 0.6, 0.35)
        self._add("deny", _tone(140, 0.18, 12), 0.5)
        t = np.linspace(0, 0.6, int(SR * 0.6), False)
        self._add("portal", np.sin(2 * np.pi * (260 * t + 700 * t * t)) * np.exp(-t * 3), 0.4)
        try:
            wave = _make_bgm_loop(20.0)
            arr = np.ascontiguousarray((np.stack([wave, wave], axis=1) * 32767).astype(np.int16))
            self.music = pygame.sndarray.make_sound(arr)
            self.music.set_volume(0.38)
            self._music_channel = pygame.mixer.Channel(0)
        except Exception as e:
            print(f"[audio] bgm build failed: {e}")
            self.music = None

    def _add(self, name, wave, vol):
        wave = np.clip(wave, -1, 1) * vol
        arr = np.ascontiguousarray((np.stack([wave, wave], axis=1) * 32767).astype(np.int16))
        self.sounds[name] = pygame.sndarray.make_sound(arr)

    def play(self, name):
        if self.ok and name in self.sounds:
            ch = pygame.mixer.find_channel(True)
            if ch is not None and ch != self._music_channel:
                ch.play(self.sounds[name])
            else:
                self.sounds[name].play()

    def start_music(self):
        if not self.ok or not self.music or not self.music_on:
            return
        try:
            ch = self._music_channel or pygame.mixer.Channel(0)
            self._music_channel = ch
            if not ch.get_busy():
                ch.play(self.music, loops=-1, fade_ms=1200)
                ch.set_volume(0.38)
        except Exception as e:
            print(f"[audio] music start failed: {e}")

    def stop_music(self, fade_ms=800):
        try:
            if self._music_channel is not None:
                self._music_channel.fadeout(fade_ms)
        except Exception:
            pass

    def toggle_music(self):
        self.music_on = not self.music_on
        if self.music_on:
            self.start_music()
            return True
        self.stop_music()
        return False


# ----------------------------------------------------------------------------
# Multiplayer (TCP, length-prefixed JSON). Host runs a thread; clients connect.
# Players + dig events sync. Edit OFFICIAL_HOST for your public 24/7 box.
# ----------------------------------------------------------------------------
def _net_send(sock, obj):
    try:
        data = json.dumps(obj, separators=(",", ":")).encode("utf-8")
        sock.sendall(struct.pack(">I", len(data)) + data)
        return True
    except Exception:
        return False


def _net_recv_buf(sock, buf):
    out = []
    try:
        chunk = sock.recv(65536)
        if not chunk:
            return None, buf
        buf += chunk
    except (BlockingIOError, InterruptedError):
        return out, buf
    except (TimeoutError, socket.timeout):
        return out, buf
    except Exception:
        return None, buf
    while len(buf) >= 4:
        (n,) = struct.unpack(">I", buf[:4])
        if n > 2_000_000:
            return None, buf
        if len(buf) < 4 + n:
            break
        raw = buf[4:4 + n]
        buf = buf[4 + n:]
        try:
            out.append(json.loads(raw.decode("utf-8")))
        except Exception:
            continue
    return out, buf


def _host_world_paths():
    base = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    world_dir = os.path.join(base, SERVER_WORLD_DIR)
    os.makedirs(world_dir, exist_ok=True)
    return (
        world_dir,
        os.path.join(world_dir, "world_events.jsonl"),
        os.path.join(world_dir, "world_state.json"),
    )


def _host_load_world_events():
    _, events_path, state_path = _host_world_paths()
    events = []
    next_eid = 1
    try:
        if os.path.isfile(state_path):
            with open(state_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            next_eid = int(meta.get("next_eid", 1))
    except Exception:
        pass
    try:
        if os.path.isfile(events_path):
            with open(events_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        ev = json.loads(line)
                        if isinstance(ev, dict) and ev.get("t") in ("dig", "boom", "collapse"):
                            events.append(ev)
                    except Exception:
                        continue
    except Exception:
        pass
    if len(events) > NET_EVENT_HISTORY:
        events = events[-NET_EVENT_HISTORY:]
    if events:
        top = max((int(e.get("eid") or 0) for e in events), default=0)
        next_eid = max(next_eid, top + 1)
    return events, next_eid


def _host_append_world_event(ev, next_eid):
    _, events_path, state_path = _host_world_paths()
    slim = {k: v for k, v in ev.items() if k != "cells"}
    try:
        with open(events_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(slim, separators=(",", ":")) + "\n")
    except Exception:
        pass
    try:
        tmp = state_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"next_eid": int(next_eid), "saved_at": time.time()}, f)
        os.replace(tmp, state_path)
    except Exception:
        pass


class NetServer:
    """In-game host. World events get unique ids (eid) so clients never apply one twice.
    World edits persist under server_world/ (separate from client_save/)."""

    def __init__(self, port=LOCAL_SERVER_PORT, name="Destrux Host"):
        self.port = port
        self.name = name
        self.players = {}
        self.clients = {}
        self.send_locks = {}
        self.lock = threading.Lock()
        self.running = False
        self.sock = None
        self.thread = None
        self.host_id = None
        self.events = []   # world events from remote clients for the local host to apply
        self.history = []  # recent world events (no bulky cell lists) for catch-up
        self.next_eid = 1

    def start(self, host_id, host_name):
        self.host_id = host_id
        self.players[host_id] = {
            "id": host_id, "name": host_name, "x": 0, "y": 0, "z": 0,
            "yaw": 0, "pitch": 0, "scene": "lobby", "tool": 0,
        }
        # Load server-local world edits (never from client_save/)
        hist, next_eid = _host_load_world_events()
        self.history = hist
        self.next_eid = next_eid
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self.sock.bind(("0.0.0.0", self.port))
        except OSError:
            try:
                self.sock.close()
            except Exception:
                pass
            return False
        self.sock.listen(12)
        self.sock.settimeout(0.5)
        self.running = True
        self.thread = threading.Thread(target=self._accept_loop, daemon=True)
        self.thread.start()
        return True

    def stop(self):
        self.running = False
        with self.lock:
            for c in list(self.clients.values()):
                try:
                    c.close()
                except Exception:
                    pass
            self.clients.clear()
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass

    def _accept_loop(self):
        while self.running:
            try:
                conn, addr = self.sock.accept()
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                threading.Thread(target=self._client_loop, args=(conn,), daemon=True).start()
            except socket.timeout:
                continue
            except Exception:
                if self.running:
                    time.sleep(0.1)

    def _send(self, pid, conn, obj):
        lk = self.send_locks.get(pid)
        if lk is None:
            return False
        with lk:  # one writer per socket at a time
            return _net_send(conn, obj)

    def _client_loop(self, conn):
        buf = b""
        pid = None
        try:
            conn.settimeout(2.0)
            join = None
            deadline = time.time() + 20.0
            while time.time() < deadline and join is None:
                msgs, buf = _net_recv_buf(conn, buf)
                if msgs is None:
                    conn.close()
                    return
                for m in msgs:
                    if m.get("t") == "join":
                        join = m
                        break
            if join is None:
                conn.close()
                return
            pname = (join.get("name") or "Player")[:16]
            with self.lock:
                pid = str(join.get("id") or uuid.uuid4())[:8]
                if pid in self.players:
                    pid = str(uuid.uuid4())[:8]
                player = {
                    "id": pid, "name": pname, "x": 24, "y": 6, "z": 33,
                    "yaw": 0, "pitch": 0, "scene": "lobby", "tool": 0,
                }
                self.clients[pid] = conn
                self.players[pid] = player
                self.send_locks[pid] = threading.Lock()
                snapshot = [dict(p) for p in self.players.values()]
                hist = list(self.history)
            self._send(pid, conn, {"t": "welcome", "id": pid, "players": snapshot,
                                   "server": self.name, "port": self.port,
                                   "history": hist})
            self._broadcast({"t": "join", "player": dict(player)}, exclude=pid)
            conn.settimeout(10.0)  # send timeout; reads are gated by select()
            while self.running:
                readable, _, _ = select.select([conn], [], [], 0.5)
                if not readable:
                    continue
                msgs, buf = _net_recv_buf(conn, buf)
                if msgs is None:
                    break
                for m in msgs:
                    self._handle(pid, m)
        except Exception:
            pass
        finally:
            with self.lock:
                self.clients.pop(pid, None)
                self.send_locks.pop(pid, None)
                self.players.pop(pid, None)
            try:
                conn.close()
            except Exception:
                pass
            if pid:
                self._broadcast({"t": "leave", "id": pid})

    def _stamp(self, msg):
        """Unique event id + remember + persist under server_world/ (not client_save/)."""
        with self.lock:
            msg["eid"] = self.next_eid
            self.next_eid += 1
            slim = {k: v for k, v in msg.items() if k != "cells"}
            self.history.append(slim)
            if len(self.history) > NET_EVENT_HISTORY:
                self.history = self.history[-NET_EVENT_HISTORY:]
            eid = self.next_eid
        _host_append_world_event(slim, eid)

    def _handle(self, pid, m):
        t = m.get("t")
        if t == "state":
            with self.lock:
                p = self.players.get(pid)
                if not p:
                    return
                for k in ("x", "y", "z", "yaw", "pitch", "scene", "tool"):
                    if k in m:
                        p[k] = m[k]
                payload = dict(p)
            self._broadcast({"t": "state", "player": payload}, exclude=pid)
        elif t == "progress":
            try:
                name = (m.get("player_name") or
                        (self.players.get(pid) or {}).get("name") or "Player")
                safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(name))[:24]
                base = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
                d = os.path.join(base, SERVER_SAVE_DIR)
                os.makedirs(d, exist_ok=True)
                path = os.path.join(d, f"{safe}.json")
                data = {k: v for k, v in m.items() if k != "t"}
                data.setdefault("saved_at", time.time())
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(data, f, separators=(",", ":"))
            except Exception:
                pass
        elif t == "resync":
            try:
                since = int(m.get("since") or 0)
            except Exception:
                since = 0
            with self.lock:
                conn = self.clients.get(pid)
                missed = [e for e in self.history if e.get("eid", 0) > since]
            if conn and missed:
                self._send(pid, conn, {"t": "sync", "events": missed})
        elif t in ("dig", "rocket", "boom", "collapse"):
            msg = dict(m)
            msg["t"] = t
            msg["id"] = pid
            if t in ("dig", "boom", "collapse"):
                self._stamp(msg)
            with self.lock:
                self.events.append(dict(msg))  # host process applies it too
            self._broadcast(msg, exclude=pid)
        elif t == "chat":
            with self.lock:
                name = self.players.get(pid, {}).get("name", "?")
                msg = {"t": "chat", "id": pid, "name": name,
                       "msg": str(m.get("msg") or "")[:80]}
                self.events.append(dict(msg))  # host process shows it too
            self._broadcast(msg, exclude=pid)

    def _broadcast(self, obj, exclude=None):
        with self.lock:
            targets = [(cid, c) for cid, c in self.clients.items() if cid != exclude]
        dead = []
        for cid, conn in targets:  # network sends happen outside the global lock
            if not self._send(cid, conn, obj):
                dead.append(cid)
        for cid in dead:
            with self.lock:
                conn = self.clients.pop(cid, None)
                self.players.pop(cid, None)
            if conn:
                try:
                    conn.close()
                except Exception:
                    pass

    def update_host_state(self, state):
        with self.lock:
            if self.host_id and self.host_id in self.players:
                self.players[self.host_id].update(state)
                payload = dict(self.players[self.host_id])
            else:
                payload = dict(state)
        self._broadcast({"t": "state", "player": payload})

    def drain_events(self):
        with self.lock:
            ev = self.events
            self.events = []
        return ev

    def player_list(self):
        with self.lock:
            return list(self.players.values())

    def host_broadcast(self, obj):
        """Host-originated world event (already applied locally): stamp + send."""
        if obj.get("t") in ("dig", "boom", "collapse"):
            self._stamp(obj)
        self._broadcast(obj)


class NetClient:
    """TCP client compatible with dedicated server build 7 / protocol 2."""

    def __init__(self):
        self.sock = None
        self.buf = b""
        self.connected = False
        self.player_id = None
        self.remote = {}
        self.lock = threading.Lock()
        self.thread = None
        self.server_name = ""
        self.epoch = None
        self.head = 0
        self.resumed = False
        self.pending = []
        self.last_error = ""
        self.welcome = None
        self.progress = None
        self.sync_done = False

    def connect(self, host, port, name, player_id, epoch=None, since=0):
        """Blocking TCP join — call from a worker thread so the game loop stays alive.

        Waits for welcome + the full sync stream (sync_end) so the main world is
        fully caught up before the join worker returns. Epoch/since enable resume
        without replaying the entire edit log.
        """
        self.close()
        self.last_error = ""
        self.welcome = None
        self.progress = None
        self.sync_done = False
        self.resumed = False
        self.head = 0
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(8.0)
            s.connect((host, int(port)))
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.sock = s
            self.player_id = player_id
            join_msg = {
                "t": "join",
                "id": player_id,
                "name": name,
                "proto": PROTOCOL,
                "client_build": CLIENT_BUILD,
                "client_sha256": _local_game_sha256(),
            }
            if epoch is not None:
                try:
                    join_msg["epoch"] = int(epoch)
                    join_msg["since"] = int(max(0, since))
                except Exception:
                    pass
            if not _net_send(s, join_msg):
                self.last_error = "Connected but failed to send join"
                s.close()
                self.connected = False
                return False
            # Stay in the handshake until we see sync_end (or timeout / error).
            # Server build 7 sends welcome (history=[]), then chunked sync packets,
            # then sync_end — all before any live traffic.
            deadline = time.time() + 45.0
            buf = b""
            welcomed = False
            while time.time() < deadline:
                try:
                    readable, _, _ = select.select([s], [], [], 0.25)
                except Exception:
                    readable = [s]
                if not readable:
                    continue
                msgs, buf = _net_recv_buf(s, buf)
                if msgs is None:
                    self.last_error = "Server closed the connection (is server.py running?)"
                    break
                for m in msgs:
                    t = m.get("t")
                    if t == "error":
                        self.last_error = m.get("msg", "Server error")
                        try:
                            s.close()
                        except Exception:
                            pass
                        self.connected = False
                        return False
                    if t == "welcome":
                        welcomed = True
                        self.welcome = m
                        self.player_id = m.get("id", player_id)
                        self.server_name = m.get("server", host)
                        self.epoch = m.get("epoch")
                        try:
                            self.head = int(m.get("head") or 0)
                        except Exception:
                            self.head = 0
                        self.resumed = bool(m.get("resumed"))
                        with self.lock:
                            for p in m.get("players", []):
                                if p.get("id") != self.player_id:
                                    self.remote[p["id"]] = p
                            # legacy servers may still embed history in welcome
                            hist = m.get("history") or []
                            if hist:
                                self.pending.append({"t": "sync", "events": hist})
                            if m.get("progress"):
                                self.progress = m["progress"]
                                self.pending.append({"t": "progress", "data": m["progress"]})
                        continue
                    if t == "sync":
                        with self.lock:
                            self.pending.append(m)
                        continue
                    if t == "sync_end":
                        try:
                            self.head = int(m.get("head") or self.head or 0)
                        except Exception:
                            pass
                        if m.get("epoch") is not None:
                            self.epoch = m.get("epoch")
                        with self.lock:
                            self.pending.append(m)
                        self.sync_done = True
                        # Hand the socket to the background receiver for live traffic
                        self.buf = buf
                        s.settimeout(10.0)
                        self.connected = True
                        self.thread = threading.Thread(target=self._recv_loop, daemon=True)
                        self.thread.start()
                        return True
                    # Anything else during handshake (state, join, chat, …) is queued
                    with self.lock:
                        self.pending.append(m)
            if welcomed and not self.sync_done:
                # Older / partial servers: proceed with whatever we got
                self.buf = buf
                s.settimeout(10.0)
                self.connected = True
                self.sync_done = True
                self.thread = threading.Thread(target=self._recv_loop, daemon=True)
                self.thread.start()
                return True
            if not welcomed:
                self.last_error = self.last_error or (
                    "No welcome from server (tunnel up but nothing on local :27015?)")
            try:
                s.close()
            except Exception:
                pass
        except Exception as e:
            self.last_error = str(e)
        self.connected = False
        self.sock = None
        return False

    def _recv_loop(self):
        sock = self.sock
        while self.connected and sock is not None:
            try:
                readable, _, _ = select.select([sock], [], [], 0.25)
            except Exception:
                self.connected = False
                break
            if not readable:
                continue
            msgs, self.buf = _net_recv_buf(sock, self.buf)
            if msgs is None:
                self.connected = False
                break
            with self.lock:
                self.pending.extend(msgs)

    def poll(self):
        with self.lock:
            msgs = self.pending
            self.pending = []
        return msgs

    def _send(self, obj):
        if self.connected and self.sock:
            if not _net_send(self.sock, obj):
                self.connected = False  # a failed/partial send corrupts the stream; reconnect

    def send_state(self, state):
        st = dict(state)
        st["t"] = "state"
        self._send(st)

    def _new_cid(self):
        return uuid.uuid4().hex[:16]

    def send_dig(self, x, y, z, r, seed=None, cid=None):
        msg = {"t": "dig", "x": x, "y": y, "z": z, "r": r, "seed": seed,
               "cid": cid or self._new_cid()}
        self._send(msg)
        return msg["cid"]

    def send_rocket(self, pos, vel, life=5.0):
        self._send({
            "t": "rocket",
            "x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2]),
            "vx": float(vel[0]), "vy": float(vel[1]), "vz": float(vel[2]),
            "life": float(life),
        })

    def send_boom(self, pos, radius, seed=None, cid=None):
        msg = {
            "t": "boom",
            "x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2]),
            "r": float(radius), "seed": seed,
            "cid": cid or self._new_cid(),
        }
        self._send(msg)
        return msg["cid"]

    def send_collapse(self, lo, hi, margin, cells=None, cid=None):
        msg = {
            "t": "collapse",
            "x0": int(lo[0]), "y0": int(lo[1]), "z0": int(lo[2]),
            "x1": int(hi[0]), "y1": int(hi[1]), "z1": int(hi[2]),
            "m": int(margin),
            "cid": cid or self._new_cid(),
        }
        if cells is not None:
            msg["cells"] = cells
        self._send(msg)
        return msg["cid"]

    def request_resync(self, since=0):
        self._send({"t": "resync", "since": int(since)})

    def send_chat(self, msg):
        msg = str(msg or "").strip()[:80]
        if msg:
            self._send({"t": "chat", "msg": msg})

    def send_fp(self, fp):
        self._send({"t": "fp", "fp": str(fp)})

    def send_progress(self, data):
        msg = dict(data)
        msg["t"] = "progress"
        self._send(msg)

    def close(self):
        self.connected = False
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass
        self.sock = None
        with self.lock:
            self.remote.clear()
            self.pending.clear()
        self.buf = b""
        self.welcome = None
        self.progress = None
        self.sync_done = False



def default_server_list():
    return [
        {"name": "Destrux Official 24/7", "host": OFFICIAL_HOST, "port": OFFICIAL_PORT,
         "motd": "playit tunnel — requires server.py on the host PC", "official": True},
        {"name": "Localhost (this PC)", "host": "127.0.0.1", "port": LOCAL_SERVER_PORT,
         "motd": "Direct join to local server.py / in-game host", "official": False},
    ]


# ----------------------------------------------------------------------------
# Scenes: lobby + shop (indestructible) and the massive main map
# ----------------------------------------------------------------------------
GLYPH = {k: v.split("/") for k, v in {
    "D": "11110/10001/10001/10001/10001/10001/11110",
    "E": "11111/10000/10000/11110/10000/10000/11111",
    "S": "01111/10000/10000/01110/00001/00001/11110",
    "T": "11111/00100/00100/00100/00100/00100/00100",
    "R": "11110/10001/10001/11110/10100/10010/10001",
    "U": "10001/10001/10001/10001/10001/10001/01110",
    "X": "10001/10001/01010/00100/01010/10001/10001",
    "H": "10001/10001/10001/11111/10001/10001/10001",
    "O": "01110/10001/10001/10001/10001/10001/01110",
    "P": "11110/10001/10001/11110/10000/10000/10000",
}.items()}


def write_text(w, text, xc, ytop, z0, z1, t, px=0.5):
    total = len(text) * 6 - 1
    xs = xc - total * px / 2
    for ci, ch in enumerate(text):
        for ry, row in enumerate(GLYPH[ch]):
            for rx, c in enumerate(row):
                if c == "1":
                    x = xs + (ci * 6 + rx) * px
                    y = ytop - (ry + 1) * px
                    w.bm(x, y, z0, x + px, y + px, z1, t)


def build_arch(w, cx, cz, rot, s, y0, frame, glow, label, target, key):
    """Portal arch. s = which way the front (approach side) faces along its depth axis."""
    def co(u0, u1, ya, yb, v0, v1):
        if rot == 0:
            xa, xb = cx + u0, cx + u1
            za, zb = sorted((cz + s * v0, cz + s * v1))
        else:
            xa, xb = sorted((cx + s * v0, cx + s * v1))
            za, zb = cz + u0, cz + u1
        return (xa, y0 + ya, za, xb, y0 + yb, zb)
    for u0, u1 in ((-3.4, -1.9), (1.9, 3.4)):
        w.bm(*co(u0, u1, 0, 10, 0, 1.6), frame)
    w.bm(*co(-3.4, 3.4, 10, 11.4, 0, 1.6), frame)
    w.bm(*co(-1.9, 1.9, 0, 10, 0.6, 1.0), glow)
    for u0, u1 in ((-3.0, -2.6), (2.6, 3.0)):
        w.bm(*co(u0, u1, 1, 9.5, 1.6, 1.75), LIGHT)
    w.bm(*co(-3.4, 3.4, 11.4, 11.7, 0, 1.6), METAL)
    t = co(-1.8, 1.8, 0.1, 9.4, 0.9, 2.3)
    lab = co(-0.1, 0.1, 12.0, 12.2, 0.8, 1.0)
    return {"box": (t[0], t[1], t[2], t[3], t[4], t[5]), "target": target, "key": key,
            "label": label,
            "label_pos": ((lab[0] + lab[3]) / 2, lab[1], (lab[2] + lab[5]) / 2)}


class Scene:
    pass


def build_lobby():
    w = World(192, 128, 192, False)
    G = 6.0
    GY = int(G / VS)
    ix, iz = np.meshgrid(np.arange(192), np.arange(192), indexing="ij")
    r = np.hypot(ix + 0.5 - 96, iz + 0.5 - 96) * VS
    on = r < 22.0
    thick = np.clip((22.0 - r) / 22.0, 0, 1)
    depth = np.minimum((3 + 21 * thick ** 0.8).astype(int), GY)
    y = np.arange(128)[None, :, None]
    w.blocks[(y < GY) & (y >= GY - depth[:, None, :]) & on[:, None, :]] = STONE
    for yy in (GY - 3, GY - 2):
        w.blocks[:, yy, :][on] = DIRT
    chk = ((ix // 12 + iz // 12) % 2)
    top = np.where(r < 17.5, np.where(chk == 0, TILE_A, TILE_B), GRASS).astype(np.uint8)
    w.blocks[:, GY - 1, :][on] = top[on]
    ring = (r >= 20.4) & (r < 21.3)
    for yy in range(GY, GY + 10):
        w.blocks[:, yy, :][ring] = TILE_B
    for k in range(12):
        a = k * math.pi / 6
        w.sph(24 + 20.85 * math.cos(a), G + 2.9, 24 + 20.85 * math.sin(a), 0.5, LIGHT)
    # fountain
    w.cyl(24, 24, 3.0, G, G + 1.0, TILE_B, 2.4)
    w.cyl(24, 24, 2.4, G, G + 0.5, WATER)
    w.cyl(24, 24, 0.5, G, G + 3.5, TILE_A)
    w.sph(24, G + 4.1, 24, 0.8, PORTAL_A)
    # trees on the grass ring
    for k in range(7):
        a = 0.4 + k * 2 * math.pi / 7
        tx, tz = 24 + 19.0 * math.cos(a), 24 + 19.0 * math.sin(a)
        if abs(tx - 24) < 5 and tz < 14:
            continue
        w.cyl(tx, tz, 0.35, G, G + 4.0, WOOD)
        w.sph(tx, G + 5.0, tz, 2.4, LEAF, "air")
    # lamp posts
    for k in range(8):
        a = (k + 0.5) * math.pi / 4
        lx, lz = 24 + 13.5 * math.cos(a), 24 + 13.5 * math.sin(a)
        w.bm(lx - 0.15, G, lz - 0.15, lx + 0.15, G + 3.6, lz + 0.15, METAL)
        w.bm(lx - 0.35, G + 3.6, lz - 0.35, lx + 0.35, G + 4.2, lz + 0.35, LIGHT)
    portals = [
        build_arch(w, 24, 11, 0, 1, G, BRICK, PORTAL_A, "DROP ZONE", "main", "default"),
        build_arch(w, 38, 24, 1, -1, G, BRICK, PORTAL_B, "SHOP", "shop", "default"),
    ]
    # Multiplayer terminal PC (interact with E)
    pcx, pcz = 15.0, 30.0
    w.bm(pcx - 0.6, G, pcz - 0.4, pcx + 0.6, G + 0.9, pcz + 0.4, METAL)
    w.bm(pcx - 0.55, G + 0.9, pcz - 0.35, pcx + 0.55, G + 1.7, pcz + 0.15, SCREEN)
    w.bm(pcx - 0.15, G + 0.9, pcz + 0.15, pcx + 0.15, G + 1.15, pcz + 0.35, METAL)
    w.sph(pcx, G + 1.95, pcz - 0.1, 0.18, LIGHT)
    write_text(w, "DESTRUX", 24, G + 16.0, 10.0, 10.5, LIGHT)
    sc = Scene()
    sc.name, sc.world, sc.portals, sc.terminal = "lobby", w, portals, None
    sc.server_pc = (pcx, G, pcz)
    sc.title = "LOBBY"
    sc.spawns = {"default": ((24, G + 0.02, 33), 0.0),
                 "from_shop": ((33.2, G + 0.02, 24), -math.pi / 2)}
    sc.kill_y = G - 2.0
    sc.env = dict(top=(0.10, 0.14, 0.38), hor=(0.95, 0.58, 0.40), sun=(-0.3, 0.14, -0.9),
                  suncol=(1.0, 0.62, 0.42), amb=(0.62, 0.55, 0.72), gnd=(0.36, 0.30, 0.38),
                  fogd=0.009, stars=1.0)
    return sc


def build_shop():
    w = World(128, 128, 128, False)
    F = 1.0
    ix, iz = np.meshgrid(np.arange(128), np.arange(128), indexing="ij")
    w.blocks[:, 0:3, :] = TILE_B
    w.blocks[:, 3, :] = np.where(((ix // 8 + iz // 8) % 2) == 0, TILE_A, TILE_B)
    H = 8.0
    w.bm(0, 0, 0, 0.75, H + 1, 32, BRICK)
    w.bm(31.25, 0, 0, 32, H + 1, 32, BRICK)
    w.bm(0, 0, 0, 32, H + 1, 0.75, BRICK)
    w.bm(0, 0, 31.25, 32, H + 1, 32, BRICK)
    w.bm(0, H, 0, 32, H + 1, 32, CONCRETE)
    for px in (8, 16, 24):
        for pz in (8, 16, 24):
            w.bm(px - 1.5, H - 0.5, pz - 1.5, px + 1.5, H, pz + 1.5, LIGHT)
    for k in range(4, 30, 8):
        for wall in ((0.75, 1.25, k, k + 1.2), (30.75, 31.25, k, k + 1.2)):
            w.bm(wall[0], F, wall[2], wall[1], H, wall[3], TILE_A)
        w.bm(k, F, 0.75, k + 1.2, H, 1.25, TILE_A)
    # counter + terminal + shopkeeper bot
    w.bm(13, F, 14.5, 19, F + 1.1, 16, BRICK)
    w.bm(12.8, F + 1.1, 14.3, 19.2, F + 1.3, 16.2, PLANK)
    w.bm(14.8, F, 13.5, 17.2, F + 1.4, 14.2, METAL)
    w.bm(14.9, F + 1.4, 13.6, 17.1, F + 2.8, 13.9, SCREEN)
    w.bm(15.5, F, 12.2, 16.5, F + 1.5, 13.0, METAL)
    w.sph(16, F + 1.9, 12.6, 0.45, WHITE)
    w.bm(15.75, F + 1.85, 13.0, 16.25, F + 2.0, 13.15, SCREEN)
    # pedestals with item models
    for px in (8, 16, 24):
        w.cyl(px, 3.6, 1.2, F, F + 0.8, TILE_A)
        w.cyl(px, 3.6, 0.9, F + 0.8, F + 0.9, GOLD)
    w.cyl(8, 3.6, 0.32, F + 0.9, F + 2.8, RED)
    for i, rr in enumerate((0.27, 0.2, 0.13, 0.06)):
        w.cyl(8, 3.6, rr, F + 2.8 + i * 0.15, F + 2.95 + i * 0.15, WHITE)
    w.bm(7.3, F + 0.9, 3.55, 8.7, F + 1.5, 3.65, RED)
    w.bm(7.95, F + 0.9, 2.9, 8.05, F + 1.5, 4.3, RED)
    w.cyl(8, 3.6, 0.2, F + 0.9, F + 1.0, PORTAL_B)
    w.cyl(15.6, 3.6, 0.3, F + 1.2, F + 2.6, METAL)
    w.cyl(16.4, 3.6, 0.3, F + 1.2, F + 2.6, METAL)
    w.bm(15.3, F + 1.9, 3.8, 16.7, F + 2.1, 3.95, RED)
    w.cyl(15.6, 3.6, 0.2, F + 1.0, F + 1.2, PORTAL_B)
    w.cyl(16.4, 3.6, 0.2, F + 1.0, F + 1.2, PORTAL_B)
    w.bm(23.9, F + 0.9, 3.5, 24.1, F + 3.1, 3.7, WOOD)
    w.bm(23.0, F + 2.9, 3.45, 25.0, F + 3.2, 3.75, METAL)
    w.bm(22.8, F + 2.6, 3.45, 23.1, F + 2.95, 3.75, METAL)
    w.bm(24.9, F + 2.6, 3.45, 25.2, F + 2.95, 3.75, METAL)
    # crates and gold
    for cx, cz in ((3, 3), (3, 5.2), (29, 3), (29, 29), (3, 29)):
        w.bm(cx - 1, F, cz - 1, cx + 1, F + 1.8, cz + 1, PLANK)
        w.bm(cx - 0.7, F + 1.8, cz - 0.7, cx + 0.7, F + 2.6, cz + 0.7, PLANK)
    for k in range(3):
        w.bm(26.5, F + k * 0.3, 24 + k * 0.2, 28.5, F + k * 0.3 + 0.3, 25.5 + k * 0.2, GOLD)
    write_text(w, "SHOP", 16, 7.6, 1.0, 1.5, LIGHT, 0.5)
    portals = [build_arch(w, 16, 30.0, 0, -1, F, BRICK, PORTAL_A, "LOBBY", "lobby", "from_shop")]
    sc = Scene()
    sc.name, sc.world, sc.portals = "shop", w, portals
    sc.terminal = (16.0, F, 15.3)
    sc.title = "THE SHOP"
    sc.spawns = {"default": ((16, F + 0.02, 25), 0.0)}
    sc.kill_y = -1.0
    sc.env = dict(top=(0.08, 0.09, 0.14), hor=(0.10, 0.11, 0.16), sun=(0.0, 1.0, 0.25),
                  suncol=(0.38, 0.36, 0.33), amb=(0.72, 0.72, 0.82), gnd=(0.50, 0.48, 0.55),
                  fogd=0.004, stars=0.0)
    return sc


def build_building(w, x0, z0, bw, bd, y0, bh):
    B = w.blocks
    t = 3
    B[x0:x0 + bw, y0 - 2:y0, z0:z0 + bd] = CONCRETE
    B[x0:x0 + bw, y0:y0 + bh, z0:z0 + bd] = BRICK
    B[x0 + t:x0 + bw - t, y0:y0 + bh - t, z0 + t:z0 + bd - t] = AIR
    B[x0:x0 + bw, y0 + bh - t:y0 + bh, z0:z0 + bd] = CONCRETE
    for fy in range(12, bh - t, 12):
        B[x0 + t:x0 + bw - t, y0 + fy:y0 + fy + 2, z0 + t:z0 + bd - t] = CONCRETE
    for cx in range(x0 + t + 14, x0 + bw - t - 4, 26):
        for cz in range(z0 + t + 14, z0 + bd - t - 4, 26):
            B[cx:cx + 4, y0:y0 + bh - t, cz:cz + 4] = CONCRETE
    ys = np.arange(bh)
    wy = ((ys % 12) >= 5) & ((ys % 12) < 10) & (ys >= 5) & (ys < bh - 6)
    xs, zs = np.arange(bw), np.arange(bd)
    wx = ((xs % 10) >= 3) & ((xs % 10) < 7) & (xs >= 5) & (xs < bw - 5)
    wz = ((zs % 10) >= 3) & ((zs % 10) < 7) & (zs >= 5) & (zs < bd - 5)
    m = wx[:, None] & wy[None, :]
    B[x0:x0 + bw, y0:y0 + bh, z0:z0 + t][m] = AIR
    B[x0:x0 + bw, y0:y0 + bh, z0 + bd - t:z0 + bd][m] = AIR
    m2 = wy[:, None] & wz[None, :]
    B[x0:x0 + t, y0:y0 + bh, z0:z0 + bd][:, m2] = AIR
    B[x0 + bw - t:x0 + bw, y0:y0 + bh, z0:z0 + bd][:, m2] = AIR
    dx = x0 + bw // 2
    B[dx - 6:dx + 6, y0:y0 + 20, z0:z0 + t] = AIR
    top = y0 + bh
    if top + 4 < w.H:
        B[x0:x0 + bw, top:top + 3, z0:z0 + bd] = CONCRETE
        B[x0 + 3:x0 + bw - 3, top:top + 3, z0 + 3:z0 + bd - 3] = AIR


def build_main(progress):
    rng = np.random.default_rng(1337)
    r = random.Random(1337)
    # Keep the world large enough to feel open, but not so large that startup
    # stalls on a giant full-world generation pass. The world is still streamed
    # in chunks as the player moves, which keeps loading near-instant.
    W, H, D = 1024, 220, 1024
    w = World(W, H, D, True)
    progress("Shaping terrain", 0.05)
    n = fbm(W, D, rng, 4, 6)
    n = (n - n.min()) / (n.max() - n.min())
    top = (24 + 66 * smoothstep(0.2, 0.9, n) + 3 * fbm(W, D, rng, 40, 2)).astype(np.int32)
    cxv, czv = W // 2, D // 2
    ix, iz = np.meshgrid(np.arange(W), np.arange(D), indexing="ij")
    rc = np.hypot(ix - cxv, iz - czv)
    hc = 52
    # Larger central plateau for the primary city
    top = np.where(rc < 120, hc, np.where(rc < 220,
                   (hc + (top - hc) * smoothstep(120, 220, rc)).astype(np.int32), top))
    # buildings – cities are rare and only appear in 3 biomes
    progress("Planning rare cities", 0.12)
    blds, tries = [], 0
    # Main dense city only in the centre (will be biome 0)
    while len(blds) < 14 and tries < 6000:
        tries += 1
        bw, bd, bh = r.randint(14, 30), r.randint(14, 30), r.randint(34, 120)
        x0, z0 = r.randint(cxv - 105, cxv + 105 - bw), r.randint(czv - 105, czv + 105 - bd)
        mx, mz = x0 + bw / 2, z0 + bd / 2
        if math.hypot(mx - cxv, mz - czv) > 100:
            continue
        if any(abs(mx - o[0] - o[2] / 2) < (bw + o[2]) / 2 + 9 and
               abs(mz - o[1] - o[3] / 2) < (bd + o[3]) / 2 + 9 for o in blds):
            continue
        blds.append((x0, z0, bw, bd, bh, 0))  # tag as city biome 0
    # Sparse small villages / outposts only in the two secondary city biomes
    for secondary_biome in (4, 5):
        for _ in range(2):  # keep secondary settlements sparse for fast world setup
            for attempt in range(350):
                bw, bd, bh = r.randint(8, 15), r.randint(8, 15), r.randint(18, 44)
                x0 = r.randint(50, W - 50 - bw)
                z0 = r.randint(50, D - 50 - bd)
                mx, mz = x0 + bw / 2, z0 + bd / 2
                # Keep them well away from the central city and from each other
                if math.hypot(mx - cxv, mz - czv) < 300:
                    continue
                if any(abs(mx - o[0] - o[2] / 2) < (bw + o[2]) / 2 + 40 and
                       abs(mz - o[1] - o[3] / 2) < (bd + o[3]) / 2 + 40 for o in blds):
                    continue
                blds.append((x0, z0, bw, bd, bh, secondary_biome))
                break
    pad = np.zeros((W, D), bool)
    plan = []
    for x0, z0, bw, bd, bh, _biome_tag in blds:
        sl = (slice(max(0, x0 - 5), min(W, x0 + bw + 5)), slice(max(0, z0 - 5), min(D, z0 + bd + 5)))
        y0 = int(top[sl].mean()) + 1
        top[sl] = y0 - 1
        pad[sl] = True
        plan.append((x0, z0, bw, bd, y0, min(bh, H - 8 - y0)))
    # surface materials (initial pass – final materials applied after biome sculpt)
    gx, gz = np.gradient(top.astype(np.float64))
    surf = np.full((W, D), GRASS, np.uint8)
    surf[top < 34] = SAND
    surf[top > 82] = SNOW
    surf[np.hypot(gx, gz) > 1.15] = STONE
    surf[pad] = CONCRETE
    surf[rc < 70] = CONCRETE
    dirtd = (3 + 5 * fbm(W, D, rng, 24, 2)).astype(np.int16)
    # fill terrain in slabs (keeps temporaries small)
    progress("Laying terrain", 0.2)
    topi = top.astype(np.int16)
    yv = np.arange(H, dtype=np.int16)[None, :, None]
    for x0 in range(0, W, 64):
        t = topi[x0:x0 + 64][:, None, :]
        depth = t - yv
        dd = dirtd[x0:x0 + 64][:, None, :]
        sf = surf[x0:x0 + 64][:, None, :]
        blk = np.where(depth < 0, np.uint8(0),
                       np.where(depth == 0, sf,
                                np.where(depth < dd, np.uint8(DIRT), np.uint8(STONE))))
        blk = blk.astype(np.uint8)
        blk[:, :2, :] = BEDROCK
        w.blocks[x0:x0 + 64] = blk
        progress("Laying terrain", 0.2 + 0.2 * (x0 + 64) / W)
    # Ore is seeded later (after final surface heights are known) so deposits
    # stay underground and never punch holes in the walkable surface.
    progress("Preparing natural regions", 0.7)
    forest_mask = np.zeros((W, D), dtype=bool)
    progress("Raising buildings", 0.85)
    for x0, z0, bw, bd, y0, bh in plan:
        build_building(w, x0, z0, bw, bd, y0, bh)
    sc = Scene()
    sc.name, sc.world, sc.portals, sc.terminal = "main", w, [], None
    sc.title = "THE DROP ZONE"

    # Twenty real surface biomes. A noisy Voronoi field gives each biome a
    # large contiguous homeland instead of a checkerboard of tiny patches.
    progress("Creating 20 biomes", 0.90)
    BIOME_NAMES = [
        "Central City", "Sandy Desert", "Red Dune Desert", "Badlands", "Open Savanna",
        "Grassland Plains", "Flower Meadow", "Dark Forest", "Pine Forest",
        "Redwood Forest", "Misty Swamp", "Tropical Jungle", "Bamboo Forest",
        "Frozen Tundra", "Glacier Mountains", "Volcanic Highlands", "Ashen Wastes",
        "Crystal Mountains", "Moonstone Desert", "Rocky Coast"
    ]
    BIOME_COLORS = [
        (105, 115, 137), (220, 183, 105), (197, 124, 75), (174, 91, 63),
        (190, 177, 91), (112, 166, 88), (190, 166, 194), (47, 91, 56),
        (82, 119, 91), (61, 111, 74), (87, 125, 112), (45, 139, 111),
        (82, 173, 117), (137, 159, 169), (202, 225, 241), (157, 73, 48),
        (103, 77, 75), (123, 130, 148), (125, 183, 192), (169, 181, 169)
    ]
    # Carefully spaced seed locations for MASSIVE territories.
    # City anchored at centre; other biomes get large contiguous homelands.
    seeds = [(cxv, czv)]
    min_seed_dist = 210
    for _ in range(19):
        for attempt in range(800):
            qx, qz = r.randint(60, W - 60), r.randint(60, D - 60)
            if math.hypot(qx - cxv, qz - czv) < 260:
                continue
            if all(math.hypot(qx - ax, qz - az) > min_seed_dist for ax, az in seeds):
                seeds.append((qx, qz))
                break
        else:
            seeds.append((r.randint(50, W - 50), r.randint(50, D - 50)))
    # Use broad Voronoi territories, gently distorted by low-frequency noise.
    distort_x = (fbm(W, D, rng, 4, 3) - 0.5) * 70
    distort_z = (fbm(W, D, rng, 6, 3) - 0.5) * 70
    biome = np.zeros((W, D), dtype=np.uint8)
    best = np.full((W, D), np.inf, dtype=np.float32)
    xx = np.arange(W, dtype=np.float32)[:, None] + distort_x
    zz = np.arange(D, dtype=np.float32)[None, :] + distort_z
    for bi, (bx, bz) in enumerate(seeds):
        score = (xx - bx) ** 2 + (zz - bz) ** 2
        if bi == 0:
            score *= 0.28          # give the central city a generous footprint
        replace = score < best
        biome[replace] = bi
        best[replace] = score[replace]
    # Keep the designed downtown footprint as city biome 0.
    # Secondary rare settlements are forced into biomes 4 / 5 so cities remain rare.
    biome[(pad & (rc < 150)) | (rc < 110)] = 0
    # Force the sparse secondary settlement pads into the two secondary city biomes.
    for x0, z0, bw, bd, bh, tag in blds:
        if tag in (4, 5):
            biome[x0:x0 + bw, z0:z0 + bd] = tag
            biome[max(0, x0 - 6):min(W, x0 + bw + 6),
                  max(0, z0 - 6):min(D, z0 + bd + 6)] = tag
    # Real biome geography with SOFT borders so neighbouring regions blend.
    progress("Sculpting natural biome terrain", 0.91)
    broad = fbm(W, D, rng, 14, 5)
    fine = fbm(W, D, rng, 40, 4)
    ridges = 1.0 - np.abs(2.0 * fbm(W, D, rng, 9, 5) - 1.0)
    original_top = top.copy()

    # Soft Voronoi weights: each cell gets influence from its nearest seeds so
    # height offsets blend across biome borders instead of forming cliffs.
    scores = np.empty((20, W, D), dtype=np.float32)
    for bi, (bx, bz) in enumerate(seeds):
        scores[bi] = (xx - bx) ** 2 + (zz - bz) ** 2
        if bi == 0:
            scores[bi] *= 0.28
    # Invert & normalise so closer seeds get higher weight
    inv = 1.0 / np.maximum(scores, 1.0)
    # Sharpen a little so interiors stay distinct, but borders stay soft
    inv = inv ** 1.35
    wsum = inv.sum(axis=0, keepdims=True)
    weights = inv / wsum

    profiles = {
        0:  (0.0, 0.0),    # city plateau already set
        1:  (1.0, 4.0),    # sandy desert
        2:  (4.0, 14.0),   # red dunes
        3:  (7.0, 22.0),   # badlands
        4:  (4.5, 10.0),   # savanna
        5:  (2.0, 5.5),    # grassland
        6:  (2.5, 7.0),    # flower meadow
        7:  (8.0, 16.0),   # dark forest
        8:  (12.0, 24.0),  # pine / taiga
        9:  (10.0, 20.0),  # redwood
        10: (0.0, 2.5),    # swamp
        11: (9.0, 18.0),   # jungle
        12: (3.5, 10.0),   # bamboo
        13: (2.5, 7.0),    # tundra
        14: (22.0, 42.0),  # glacier
        15: (12.0, 28.0),  # volcanic
        16: (4.5, 14.0),   # ashen
        17: (14.0, 30.0),  # crystal
        18: (0.5, 2.5),    # moonstone
        19: (2.0, 10.0),   # coast
    }
    # Build a continuous height offset field as a weighted sum of every biome shape.
    height_offset = np.zeros((W, D), dtype=np.float32)
    for bi, (amplitude, detail) in profiles.items():
        if amplitude == 0 and detail == 0:
            continue
        if bi == 1:
            shape = (broad - 0.5) * amplitude * 2.0 + (fine - 0.5) * 2.5
        elif bi == 2:
            shape = (ridges ** 1.4) * detail + (broad - 0.5) * amplitude
        elif bi == 3:
            shape = (ridges ** 2.4) * detail + (broad - 0.5) * amplitude
        elif bi == 14:
            shape = (ridges ** 1.6) * detail + (fine - 0.5) * amplitude
        elif bi == 15:
            shape = (ridges ** 1.9) * detail + (broad - 0.5) * amplitude
        elif bi == 19:
            shape = (broad - 0.5) * detail + np.sin(ix / 28.0) * 3.0 + (fine - 0.5) * 3.0
        else:
            shape = (broad - 0.5) * amplitude + (fine - 0.5) * detail
        height_offset += weights[bi] * shape.astype(np.float32)

    new_top = original_top.astype(np.float32) + height_offset
    # Light blur so residual seams disappear
    if ndimage is not None:
        new_top = ndimage.uniform_filter(new_top, size=5)
    else:
        # cheap 3x3 box blur without scipy
        pad_t = np.pad(new_top, 1, mode="edge")
        new_top = (pad_t[:-2, :-2] + pad_t[:-2, 1:-1] + pad_t[:-2, 2:] +
                   pad_t[1:-1, :-2] + pad_t[1:-1, 1:-1] + pad_t[1:-1, 2:] +
                   pad_t[2:, :-2] + pad_t[2:, 1:-1] + pad_t[2:, 2:]) / 9.0
    new_top = np.clip(np.rint(new_top), 14, H - 28).astype(np.int16)
    top[:] = new_top

    # Surface materials – hard assignment in biome interiors, soft blend at borders.
    surface_of = np.full(20, GRASS, np.uint8)
    surface_of[0] = CONCRETE
    surface_of[1] = SAND
    surface_of[2] = np.uint8(25)
    surface_of[3] = STONE
    surface_of[4] = np.uint8(26)
    surface_of[5] = GRASS
    surface_of[6] = np.uint8(27)
    surface_of[7] = np.uint8(28)
    surface_of[8] = np.uint8(29)
    surface_of[9] = np.uint8(30)
    surface_of[10] = np.uint8(31)
    surface_of[11] = np.uint8(32)
    surface_of[12] = np.uint8(33)
    surface_of[13] = SNOW
    surface_of[14] = SNOW
    surface_of[15] = STONE
    surface_of[16] = np.uint8(34)
    surface_of[17] = STONE
    surface_of[18] = SAND
    surface_of[19] = STONE

    # Border mask: where the winning biome is not dominant, prefer stone/dirt
    # or the second-place surface so transitions look natural, not wall-like.
    top2 = np.partition(weights, -2, axis=0)
    dominance = top2[-1] - top2[-2]          # high = deep interior, low = border
    border = dominance < 0.18

    progress("Rebuilding landforms", 0.92)
    gx, gz = np.gradient(top.astype(np.float64))
    # Only treat truly steep natural slopes as stone faces (not artificial seams).
    steep_map = (np.abs(gx) + np.abs(gz)) > 2.2
    dirtd = (3 + 4 * fbm(W, D, rng, 22, 2)).astype(np.int16)

    for x0 in range(0, W, 64):
        x1 = min(x0 + 64, W)
        t = top[x0:x1][:, None, :]
        depth = t - np.arange(H, dtype=np.int16)[None, :, None]
        dd = dirtd[x0:x1][:, None, :]
        bi_slice = biome[x0:x1]
        surf = surface_of[bi_slice]
        # On soft borders, mix toward grass/sand/stone instead of a hard material wall.
        border_slice = border[x0:x1]
        # Prefer a neutral transition surface on borders (grass or sand depending on height).
        neutral = np.where(top[x0:x1] < 40, np.uint8(SAND),
                           np.where(top[x0:x1] > 85, np.uint8(SNOW), np.uint8(GRASS)))
        surf = np.where(border_slice, neutral, surf)
        surf = surf[:, None, :]
        steep_slice = steep_map[x0:x1][:, None, :]
        blk = np.where(depth < 0, np.uint8(0),
                       np.where(depth == 0,
                                np.where(steep_slice, np.uint8(STONE), surf),
                                np.where(depth < dd, np.uint8(DIRT), np.uint8(STONE))))
        blk = blk.astype(np.uint8)
        blk[:, :2, :] = BEDROCK
        existing = w.blocks[x0:x1]
        pad_slice = pad[x0:x1][:, None, :]
        has_structure = np.any(
            (existing != 0) & (existing != BEDROCK) & (existing != DIRT) &
            (existing != STONE) & (existing != SAND) & (existing != SNOW) &
            (existing != GRASS) & (existing != WATER) & (existing < 24), axis=1)
        keep = pad_slice | has_structure[:, None, :]
        w.blocks[x0:x1] = np.where(keep, existing, blk)
        progress("Rebuilding landforms", 0.92 + 0.02 * (x1 / W))

    # City core always paved.
    city_x, city_z = np.where(biome == 0)
    w.blocks[city_x, top[city_x, city_z], city_z] = CONCRETE
    for bi in (4, 5):
        cx, cz = np.where((biome == bi) & pad)
        if len(cx):
            w.blocks[cx, top[cx, cz], cz] = CONCRETE

    # Region-specific plants and landmarks, all placed on the actual terrain.
    progress("Growing biome vegetation", 0.94)
    def plant_tree(x, z, height, crown, leaf_id=LEAF):
        y = int(top[x, z]) + 1
        if y + height + crown + 2 >= H:
            return
        trunk_w = 2 if height > 30 else 1
        w.blocks[x:x+trunk_w, y:y+height, z:z+trunk_w] = WOOD
        w.sphere((x + trunk_w / 2, y + height, z + trunk_w / 2),
                 crown, leaf_id, "air")

    # Dense natural forests – scaled for the massive world.
    wooded_ids = {7, 8, 9, 10, 11, 12}
    for _ in range(1200):
        x, z = r.randint(10, W - 11), r.randint(10, D - 11)
        bi = int(biome[x, z])
        if bi not in wooded_ids or pad[x, z]:
            continue
        if bi == 7:       # darkwood
            plant_tree(x, z, r.randint(26, 42), r.randint(11, 15))
        elif bi == 8:     # tall pines
            plant_tree(x, z, r.randint(34, 54), r.randint(7, 11))
        elif bi == 9:     # giant redwoods
            plant_tree(x, z, r.randint(50, 78), r.randint(14, 22))
        elif bi == 10:    # swamp
            if r.random() < 0.38:
                plant_tree(x, z, r.randint(18, 30), r.randint(8, 12))
        elif bi == 11:    # jungle
            plant_tree(x, z, r.randint(30, 52), r.randint(13, 18))
        else:             # bamboo
            y = int(top[x, z]) + 1
            h = r.randint(20, 40)
            if y + h < H - 2:
                w.blocks[x:x+1, y:y+h, z:z+1] = np.uint8(35)

    # Cactus fields across the large desert.
    for _ in range(900):
        x, z = r.randint(6, W - 7), r.randint(6, D - 7)
        if biome[x, z] != 1:
            continue
        y = int(top[x, z]) + 1
        h = r.randint(5, 14)
        cactus = np.uint8(35)
        if y + h + 8 < H:
            w.blocks[x:x+2, y:y+h, z:z+2] = cactus
            if h >= 8 and r.random() < 0.55:
                arm_y = y + r.randint(3, h - 2)
                w.blocks[x+2:x+4, arm_y:arm_y+2, z:z+2] = cactus
                w.blocks[x+2:x+4, arm_y+2:arm_y+5, z:z+2] = cactus
                w.blocks[x:x+2, arm_y+2:arm_y+5, z+2:z+4] = cactus

    # Sparse savanna acacias.
    for _ in range(380):
        x, z = r.randint(10, W - 11), r.randint(10, D - 11)
        if biome[x, z] != 4 or pad[x, z]:
            continue
        plant_tree(x, z, r.randint(14, 22), r.randint(11, 16))

    # Sparse conifers on tundra / pine edges; glaciers stay mostly bare.
    for _ in range(700):
        x, z = r.randint(10, W - 11), r.randint(10, D - 11)
        if biome[x, z] not in (8, 13):
            continue
        if r.random() < 0.48:
            plant_tree(x, z, r.randint(24, 42), r.randint(7, 11))

    # --- Deep caves (well below final surface so no open pits) -------------
    progress("Carving deep caves", 0.95)
    for _ in range(48):
        x = r.randint(50, W - 50)
        z = r.randint(50, D - 50)
        if math.hypot(x - cxv, z - czv) < 100:
            continue
        surface_y = int(top[x, z])
        # Start deep under the final surface
        p = np.array([float(x), float(r.randint(8, max(9, surface_y - 18))), float(z)])
        d = np.array([r.uniform(-1, 1), r.uniform(-0.25, 0.15), r.uniform(-1, 1)])
        d /= max(np.linalg.norm(d), 1e-5)
        for _ in range(36):
            rad = r.uniform(4.5, 8.0)
            # Never carve within 10 voxels of the final surface
            if p[1] + rad < surface_y - 10:
                w.carve_sphere(p, rad)
            d = d + np.random.uniform(-0.3, 0.3, 3) * np.array([1, 0.4, 1])
            d /= max(np.linalg.norm(d), 1e-5)
            p = p + d * rad * 0.75
            p[1] = min(max(p[1], 6), max(7, surface_y - 14))
            p[0] = min(max(p[0], 25), W - 25)
            p[2] = min(max(p[2], 25), D - 25)
            surface_y = int(top[min(W - 1, max(0, int(p[0]))),
                                 min(D - 1, max(0, int(p[2])))])

    # Ore deposits deep under the surface only
    progress("Seeding underground ore", 0.96)
    for _ in range(3200):
        x, z = r.randint(8, W - 8), r.randint(8, D - 8)
        sy = int(top[x, z])
        if sy < 16:
            continue
        yy = r.randint(5, max(6, sy - 10))
        w.sphere((x, yy, z), r.uniform(2, 4.5), ORE, "stone")

    # --- Fun props: ruins, loot crates, jump pads, ore markers -------------
    progress("Placing landmarks & loot", 0.97)
    fun_props = []  # consumed later by Game for interaction

    # Ancient ruins (non-city biomes) – small broken walls + pillar
    for _ in range(55):
        x = r.randint(20, W - 20)
        z = r.randint(20, D - 20)
        bi = int(biome[x, z])
        if bi == 0 or pad[x, z]:
            continue
        y = int(top[x, z]) + 1
        if y + 12 >= H:
            continue
        # broken wall ring
        for dx, dz in ((-3, -3), (-3, 3), (3, -3), (3, 3), (-3, 0), (3, 0), (0, -3), (0, 3)):
            if r.random() < 0.7:
                h = r.randint(2, 7)
                w.blocks[x + dx:x + dx + 2, y:y + h, z + dz:z + dz + 2] = STONE
        # centre pillar with gold tip (visible landmark)
        w.blocks[x:x + 2, y:y + 8, z:z + 2] = STONE
        w.blocks[x:x + 2, y + 8:y + 10, z:z + 2] = GOLD
        fun_props.append({"kind": "ruin", "x": x, "z": z, "y": y})

    # Loot crates – break for coins
    for _ in range(90):
        x = r.randint(15, W - 15)
        z = r.randint(15, D - 15)
        if pad[x, z] or biome[x, z] == 0:
            continue
        y = int(top[x, z]) + 1
        if y + 3 >= H:
            continue
        w.blocks[x:x + 2, y:y + 2, z:z + 2] = PLANK
        w.blocks[x:x + 2, y + 2:y + 3, z:z + 2] = GOLD
        fun_props.append({"kind": "crate", "x": x, "z": z, "y": y, "opened": False})

    # Jump pads (portal-coloured plates) – bounce the player
    for _ in range(40):
        x = r.randint(20, W - 20)
        z = r.randint(20, D - 20)
        if pad[x, z]:
            continue
        y = int(top[x, z])
        w.blocks[x:x + 3, y:y + 1, z:z + 3] = PORTAL_B
        fun_props.append({"kind": "jump", "x": x, "z": z, "y": y})

    # Surface ore outcrops (small visible gold/ore rocks to mine)
    for _ in range(70):
        x = r.randint(10, W - 10)
        z = r.randint(10, D - 10)
        if pad[x, z] or biome[x, z] == 0:
            continue
        y = int(top[x, z]) + 1
        if y + 3 >= H:
            continue
        w.blocks[x:x + 2, y:y + 2, z:z + 2] = ORE
        if r.random() < 0.4:
            w.blocks[x:x + 1, y + 2:y + 3, z:z + 1] = GOLD

    # Seal any accidental air at the exact surface height (vectorised)
    progress("Sealing surface", 0.98)
    sy = top.astype(np.int32)
    xx_i = np.arange(W, dtype=np.int32)[:, None]
    zz_i = np.arange(D, dtype=np.int32)[None, :]
    # surface air -> proper surface block
    air_surf = w.blocks[xx_i, sy, zz_i] == 0
    if np.any(air_surf):
        fill = np.full((W, D), GRASS, np.uint8)
        fill[biome == 0] = CONCRETE
        fill[np.isin(biome, [1, 18])] = SAND
        fill[np.isin(biome, [13, 14])] = SNOW
        xi, zi = np.nonzero(air_surf)
        w.blocks[xi, sy[xi, zi], zi] = fill[xi, zi]
    # one layer under surface if open
    under = np.clip(sy - 1, 0, H - 1)
    air_under = (sy > 2) & (w.blocks[xx_i, under, zz_i] == 0)
    if np.any(air_under):
        xi, zi = np.nonzero(air_under)
        w.blocks[xi, under[xi, zi], zi] = DIRT

    forest_mask = np.isin(biome, list(wooded_ids))
    sc = Scene()
    sc.name, sc.world, sc.portals, sc.terminal = "main", w, [], None
    sc.title = "THE DROP ZONE"
    sc.biome_map = biome
    sc.biome_names = BIOME_NAMES
    sc.biome_colors = BIOME_COLORS
    sc.biome_seeds = seeds
    sc.map_size = (W, D)
    sc.fun_props = fun_props
    sc.map_destinations = []
    for bi, (bx, bz) in enumerate(seeds):
        sc.map_destinations.append({"name": BIOME_NAMES[bi], "kind": "BIOME", "biome": bi,
                                    "x": int(bx), "z": int(bz)})
    for i, (x0, z0, bw, bd, y0, bh) in enumerate(plan):
        sc.map_destinations.append({
            "name": f"Building {i + 1}", "kind": "POI",
            "x": x0 + bw // 2, "z": z0 + bd // 2
        })
    # Every biome gets a named monster-den quest marker in its own territory.
    quest_sites = [
        "Buried Tomb", "Red Dune Den", "Badlands Bunker", "Savanna Nest",
        "Greenway Cave", "Bloom Shrine", "Darkwood Hollow", "Pinewatch Burrow",
        "Redwood Heart", "Sunken Mire Lair", "Jungle Temple", "Bamboo Hollow",
        "Tundra Station", "Glacier Vault", "Volcanic Core", "Ashfall Nest",
        "Crystal Mine", "Moonstone Vault", "Coast Wreck"
    ]
    for bi, (sx, sz) in enumerate(seeds):
        if bi == 0:
            continue
        qx, qz = int(sx), int(sz)
        for radius in range(8, 46, 3):
            candidates = [(int(sx + math.cos(a) * radius), int(sz + math.sin(a) * radius))
                          for a in np.linspace(0, math.tau, 24, endpoint=False)]
            valid = [(x, z) for x, z in candidates
                     if 2 <= x < W - 2 and 2 <= z < D - 2 and biome[x, z] == bi]
            if valid:
                qx, qz = valid[0]
                break
        sc.map_destinations.append({"name": quest_sites[bi - 1], "kind": "QUEST",
                                    "biome": bi, "x": qx, "z": qz})
    sy = (int(top[cxv, czv]) + 1) * VS + 0.02
    sc.spawns = {"default": ((cxv * VS, sy, czv * VS), 0.0)}
    sc.kill_y = -1.0
    sc.env = dict(top=(0.20, 0.42, 0.85), hor=(0.72, 0.84, 0.95), sun=(0.5, 0.8, 0.35),
                  suncol=(0.66, 0.62, 0.55), amb=(0.52, 0.60, 0.74), gnd=(0.34, 0.32, 0.30),
                  fogd=0.0085, stars=0.0)
    return sc


# ----------------------------------------------------------------------------
# Shaders
# ----------------------------------------------------------------------------
COMMON = """
uniform vec3 u_cam;
uniform vec3 u_sun;
uniform vec3 u_suncol;
uniform vec3 u_amb;
uniform vec3 u_gnd;
uniform vec3 u_fogcol;
uniform float u_fogd;
uniform float u_time;
uniform float u_quality;
uniform float u_lighting;
uniform float u_fog;
uniform float u_grade;
uniform float u_emissive;
uniform float u_style;

vec3 stylize(vec3 color)
{
    if(u_style < 0.5)
    {
        return color;
    }
    if(u_style < 1.5)
    {
        return floor(clamp(color, 0.0, 1.0) * 5.0 + 0.5) / 5.0;
    }
    if(u_style < 2.5)
    {
        float lum = dot(color, vec3(0.299, 0.587, 0.114));
        return vec3(clamp((lum - 0.45) * 1.5 + 0.5, 0.0, 1.0));
    }
    if(u_style < 3.5)
    {
        return color * vec3(1.18, 0.92, 0.72) + vec3(0.04, 0.01, 0.0);
    }
    return color * vec3(0.78, 0.96, 1.2) + vec3(0.0, 0.015, 0.04);
}

vec3 ACES(vec3 x)
{
    if(u_grade < 0.5)
    {
        return x;
    }

    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;

    return clamp(
        (x * (a * x + b)) /
        (x * (c * x + d) + e),
        0.0,
        1.0
    );
}

vec3 light(vec3 albedo, vec3 n, vec3 wp)
{
    if(u_lighting < 0.5)
    {
        return albedo;
    }

    vec3 L = normalize(u_sun);
    vec3 V = normalize(u_cam - wp);
    vec3 H = normalize(L + V);

    float q = u_quality;

    float ndl = smoothstep(-0.05, 1.0, dot(n, L));

    vec3 skyCol = mix(
        vec3(0.45, 0.56, 0.95),
        vec3(0.65, 0.78, 1.0),
        clamp(q * 0.5, 0.0, 1.0)
    );

    vec3 ambient =
        mix(
            u_gnd,
            skyCol,
            n.y * 0.5 + 0.5
        );

    float bounce = max(-n.y, 0.0);
    ambient += vec3(0.82, 0.68, 0.48) * bounce * 0.18;

    float wrap =
        max(
            dot(n, L) * 0.5 + 0.5,
            0.0
        );

    float fresnel =
        pow(
            1.0 - max(dot(n, V), 0.0),
            5.0
        );

    float spec =
        pow(
            max(dot(n, H), 0.0),
            mix(
                24.0,
                128.0,
                clamp(q * 0.5, 0.0, 1.0)
            )
        );

    float rim =
        pow(
            1.0 - max(dot(n, V), 0.0),
            2.0
        );

    vec3 color =
        albedo *
        (
            ambient +
            ndl * u_suncol * 1.2 +
            wrap * u_suncol * 0.14
        );

    color += u_suncol * spec * 0.35;
    color += albedo * rim * 0.15;

    color +=
        fresnel *
        mix(
            vec3(0.4, 0.55, 0.8),
            vec3(1.0, 0.8, 0.45),
            max(L.y, 0.0)
        ) * 0.25;

    if(q > 1.5)
    {
        float edge =
            pow(
                1.0 - abs(n.y),
                1.8
            );

        color +=
            vec3(
                0.95,
                0.68,
                0.22
            ) *
            edge *
            0.12;

        color +=
            u_suncol *
            ndl *
            ndl *
            0.2;
    }

    return color;
}

vec3 fogged(vec3 col, vec3 wp)
{
    if(u_fog < 0.5)
    {
        return col;
    }

    float dist =
        length(wp - u_cam);

    float fog =
        1.0 -
        exp(
            -pow(
                dist * u_fogd,
                1.55
            )
        );

    float hfog =
        clamp(
            (32.0 - wp.y) / 42.0,
            0.0,
            1.0
        );

    fog +=
        hfog *
        0.25 *
        u_quality;

    vec3 V =
        normalize(wp - u_cam);

    float scatter =
        pow(
            max(
                dot(
                    V,
                    normalize(u_sun)
                ),
                0.0
            ),
            8.0
        );

    vec3 fogCol =
        u_fogcol +
        vec3(
            1.0,
            0.75,
            0.45
        ) *
        scatter *
        0.3;

    return mix(
        col,
        fogCol,
        clamp(
            fog,
            0.0,
            1.0
        )
    );
}
"""

CHUNK_VERT = """
#version 330
uniform mat4 u_vp;
in vec3 in_pos;
in vec3 in_col;
in float in_nid;
out vec3 v_col;
out vec3 v_world;
flat out float v_nid;
void main() {
    v_col = in_col;
    v_world = in_pos;
    v_nid = in_nid;
    gl_Position = u_vp * vec4(in_pos, 1.0);
}
"""

CHUNK_FRAG = "#version 330\n" + COMMON + """
in vec3 v_col;
in vec3 v_world;
flat in float v_nid;

out vec4 f_color;

const vec3 NORMALS[6] = vec3[6](
    vec3( 1, 0, 0),
    vec3(-1, 0, 0),
    vec3( 0, 1, 0),
    vec3( 0,-1, 0),
    vec3( 0, 0, 1),
    vec3( 0, 0,-1)
);

void main()
{
    vec3 lit;

    if(v_nid > 5.5)
    {
        if(u_emissive < 0.5)
        {
            lit = v_col;
        }
        else
        {
        float pulse =
            0.9 +
            0.1 *
            sin(
                u_time * 2.5 +
                dot(
                    v_world,
                    vec3(3.0)
                )
            );

        lit =
            v_col *
            pulse *
            mix(
                1.3,
                2.0,
                u_quality * 0.5
            );

        lit +=
            v_col *
            0.6 *
            u_quality *
            pulse;
        }
    }
    else
    {
        lit =
            light(
                v_col,
                NORMALS[
                    int(min(v_nid + 0.5, 5.0))
                ],
                v_world
            );
    }

    lit = ACES(lit);

    float lum =
        dot(
            lit,
            vec3(
                0.299,
                0.587,
                0.114
            )
        );

    lit =
        mix(
            vec3(lum),
            lit,
            1.22
        );
    lit.g *= 1.01;
    lit.b *= 0.97;

    f_color =
        vec4(
            stylize(fogged(lit, v_world)),
            1.0
        );
}
"""

INST_FRAG = "#version 330\n" + COMMON + """
in vec3 v_color;
in vec3 v_norm;
in vec3 v_world;

out vec4 f_color;

void main()
{
    vec3 lit;

    if(
        max(
            v_color.r,
            max(
                v_color.g,
                v_color.b
            )
        ) > 1.001
    )
    {
        lit = v_color;
    }
    else
    {
        lit =
            light(
                v_color,
                normalize(v_norm),
                v_world
            );
    }

    lit = ACES(lit);

    float lum =
        dot(
            lit,
            vec3(
                0.299,
                0.587,
                0.114
            )
        );

    lit =
        mix(
            vec3(lum),
            lit,
            1.2
        );

    f_color =
        vec4(
            stylize(fogged(lit, v_world)),
            1.0
        );
}
"""
INST_VERT = """
#version 330
uniform mat4 u_vp;
in vec3 in_vert;
in vec3 in_norm;
in vec3 in_pos;
in float in_scale;
in vec3 in_color;
out vec3 v_color;
out vec3 v_norm;
out vec3 v_world;
void main() {
    vec3 w = in_vert * in_scale + in_pos;
    v_world = w;
    v_color = in_color;
    v_norm = in_norm;
    gl_Position = u_vp * vec4(w, 1.0);
}
"""

INST_FRAG = """
#version 330
""" + COMMON + """

in vec3 v_color;
in vec3 v_norm;
in vec3 v_world;

out vec4 f_color;

void main()
{
    vec3 lit;

    lit = light(
        v_color,
        normalize(v_norm),
        v_world
    );

    lit = ACES(lit);

    f_color = vec4(
        stylize(fogged(lit, v_world)),
        1.0
    );
}
"""

SKY_VERT = """
#version 330
in vec2 in_pos;
out vec2 v_ndc;
void main() {
    v_ndc = in_pos;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

SKY_FRAG = """
#version 330

uniform vec3 u_fwd;
uniform vec3 u_right;
uniform vec3 u_up;
uniform vec3 u_sun;
uniform vec2 u_tan;
uniform vec3 u_top;
uniform vec3 u_hor;
uniform float u_stars;
uniform float u_style;

in vec2 v_ndc;

out vec4 f_color;

void main()
{
    vec3 d = normalize(
        u_fwd +
        v_ndc.x * u_tan.x * u_right +
        v_ndc.y * u_tan.y * u_up
    );

    vec3 col =
        mix(
            u_hor,
            u_top,
            pow(
                clamp(
                    d.y,
                    0.0,
                    1.0
                ),
                0.55
            )
        );

    if(d.y < 0.0)
    {
        col =
            mix(
                u_hor,
                u_hor * 0.75,
                clamp(
                    -d.y * 3.0,
                    0.0,
                    1.0
                )
            );
    }

    float s =
        max(
            dot(
                d,
                normalize(u_sun)
            ),
            0.0
        );

    float sunDisk =
        pow(
            s,
            2500.0
        );

    float corona =
        pow(
            s,
            16.0
        );

    col +=
        vec3(
            1.0,
            0.85,
            0.6
        ) *
        corona *
        0.25;

    col +=
        vec3(
            1.0,
            0.95,
            0.8
        ) *
        sunDisk *
        8.0;

    vec3 cell =
        floor(
            d * 260.0
        );

    float h =
        fract(
            sin(
                dot(
                    cell,
                    vec3(
                        12.9898,
                        78.233,
                        37.719
                    )
                )
            ) *
            43758.5453
        );

    col +=
        vec3(
            step(
                0.9987,
                h
            )
        ) *
        u_stars *
        smoothstep(
            0.02,
            0.35,
            d.y
        );

    if(u_style > 0.5 && u_style < 1.5)
    {
        col = floor(clamp(col, 0.0, 1.0) * 5.0 + 0.5) / 5.0;
    }
    else if(u_style > 1.5 && u_style < 2.5)
    {
        float lum = dot(col, vec3(0.299, 0.587, 0.114));
        col = vec3(clamp((lum - 0.45) * 1.5 + 0.5, 0.0, 1.0));
    }
    else if(u_style > 2.5 && u_style < 3.5)
    {
        col = col * vec3(1.18, 0.92, 0.72) + vec3(0.04, 0.01, 0.0);
    }
    else if(u_style >= 3.5)
    {
        col = col * vec3(0.78, 0.96, 1.2) + vec3(0.0, 0.015, 0.04);
    }

    f_color =
        vec4(
            col,
            1.0
        );
}
"""
OVERLAY_VERT = """
#version 330

in vec2 in_pos;

out vec2 v_uv;

void main()
{
    v_uv = in_pos * 0.5 + 0.5;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""
OVERLAY_FRAG = """
#version 330
uniform sampler2D u_tex;
in vec2 v_uv;
out vec4 f_color;
void main() {
    f_color = texture(u_tex, v_uv);
}
"""


def setu(prog, name, value):
    if name in prog:
        prog[name].value = value


# ----------------------------------------------------------------------------
# Game
# ----------------------------------------------------------------------------



def _frozen():
    return bool(getattr(sys, "frozen", False)) or hasattr(sys, "_MEIPASS")


def _app_dir():
    """Folder next to the .exe (frozen) or the .py file (source)."""
    if _frozen():
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


def _client_file_path():
    """Updatable game script the auto-updater writes to.

    Source runs: this .py file.
    Frozen runs: destrux_live.py beside the .exe (so updates don't need a new binary).
    """
    if _frozen():
        return os.path.join(_app_dir(), "destrux_live.py")
    return os.path.abspath(__file__)


def _local_game_sha256():
    try:
        with open(_client_file_path(), "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except Exception:
        try:
            with open(os.path.abspath(__file__), "rb") as f:
                return hashlib.sha256(f.read()).hexdigest()
        except Exception:
            return "unknown"


def _seed_live_payload():
    """First launch of a frozen build: copy bundled source next to the exe if missing."""
    if not _frozen():
        return
    live = _client_file_path()
    if os.path.isfile(live) and os.path.getsize(live) > 1000:
        return
    candidates = []
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidates.append(os.path.join(meipass, "destrux.py"))
        candidates.append(os.path.join(meipass, "destrux_live.py"))
    candidates.append(os.path.join(_app_dir(), "destrux.py"))
    for src in candidates:
        try:
            if os.path.isfile(src) and os.path.getsize(src) > 1000:
                with open(src, "rb") as f:
                    data = f.read()
                with open(live, "wb") as f:
                    f.write(data)
                print(f"[boot] seeded live payload -> {live} ({len(data)} bytes)")
                return
        except Exception as e:
            print(f"[boot] seed from {src} failed: {e}")
    print("[boot] WARNING: no live payload found; auto-update must supply destrux_live.py")


def check_for_game_update():
    """Fetch versions/latest.py from the game server before starting the game."""
    target = _client_file_path()
    host = OFFICIAL_HOST
    port = OFFICIAL_PORT
    try:
        with open(target, "rb") as f:
            current_bytes = f.read()
        current_hash = hashlib.sha256(current_bytes).hexdigest()
    except Exception as e:
        print(f"[UPDATE] Could not read current game file: {e}")
        return

    sock = None
    try:
        sock = socket.create_connection((host, int(port)), timeout=CLIENT_UPDATE_TIMEOUT)
        sock.settimeout(CLIENT_UPDATE_TIMEOUT)
        if not _net_send(sock, {"t": "update_check", "sha256": current_hash}):
            raise ConnectionError("Could not send update request")
        deadline = time.time() + CLIENT_UPDATE_TIMEOUT
        recv_buf = b""
        response = None
        while time.time() < deadline and response is None:
            msgs, recv_buf = _net_recv_buf(sock, recv_buf)
            if msgs is None:
                raise ConnectionError("Server closed during update check")
            if msgs:
                response = msgs[0]
        if response is None:
            print("[UPDATE] No update response; starting installed version.")
            return
        if response.get("t") == "no_update":
            print(f"[UPDATE] Game is up to date ({response.get('version', 'current')}).")
            return
        if response.get("t") != "update_file":
            print(f"[UPDATE] Server did not provide an update: {response.get('msg', response.get('t'))}")
            return

        encoded = response.get("content", "")
        new_bytes = base64.b64decode(encoded, validate=True)
        expected_hash = str(response.get("sha256", ""))
        actual_hash = hashlib.sha256(new_bytes).hexdigest()
        if not expected_hash or actual_hash != expected_hash:
            raise ValueError("Downloaded update failed its SHA-256 check")
        if not new_bytes.startswith(b'"""') and b"import " not in new_bytes[:4096]:
            raise ValueError("Downloaded latest.py does not look like a Python game file")

        temp_path = target + ".update_tmp"
        backup_path = target + ".update_backup"
        with open(temp_path, "wb") as f:
            f.write(new_bytes)
            f.flush()
            os.fsync(f.fileno())

        # Keep a rollback copy until the replacement has completed.
        try:
            if os.path.exists(backup_path):
                os.remove(backup_path)
            os.replace(target, backup_path)
            os.replace(temp_path, target)
        except Exception:
            if os.path.exists(backup_path) and not os.path.exists(target):
                os.replace(backup_path, target)
            raise

        print(f"[UPDATE] Installed server version {response.get('version', actual_hash[:12])}; restarting game...")
        # Frozen: relaunch the .exe (it will load the new destrux_live.py).
        # Source: relaunch this .py with the same interpreter.
        if _frozen():
            subprocess.Popen([sys.executable, *sys.argv[1:]], cwd=_app_dir())
        else:
            subprocess.Popen([sys.executable, target, *sys.argv[1:]],
                             cwd=os.path.dirname(target) or ".")
        raise SystemExit(0)
    except SystemExit:
        raise
    except Exception as e:
        print(f"[UPDATE] Update check failed; starting installed version: {e}")
    finally:
        if sock is not None:
            try:
                sock.close()
            except Exception:
                pass

class Game:
    def __init__(self):
        pygame.mixer.pre_init(SR, -16, 2, 512)
        pygame.init()
        gl = pygame.display.gl_set_attribute
        gl(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
        gl(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        gl(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
        gl(pygame.GL_DEPTH_SIZE, 24)
        self.win_w, self.win_h = WIN_W, WIN_H
        flags = pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE
        for samples in (4, 0):          # MSAA smooths the small voxel edges
            gl(pygame.GL_MULTISAMPLEBUFFERS, 1 if samples else 0)
            gl(pygame.GL_MULTISAMPLESAMPLES, samples)
            try:
                try:
                    pygame.display.set_mode((WIN_W, WIN_H), flags, vsync=1)
                except TypeError:
                    pygame.display.set_mode((WIN_W, WIN_H), flags)
                break
            except pygame.error:
                continue
        pygame.display.set_caption("DESTRUX")
        self.ctx = moderngl.create_context()
        ctx = self.ctx

        self.cprog = ctx.program(vertex_shader=CHUNK_VERT, fragment_shader=CHUNK_FRAG)
        self.iprog = ctx.program(vertex_shader=INST_VERT, fragment_shader=INST_FRAG)
        self.sprog = ctx.program(vertex_shader=SKY_VERT, fragment_shader=SKY_FRAG)
        self.oprog = ctx.program(vertex_shader=OVERLAY_VERT, fragment_shader=OVERLAY_FRAG)
        self.cube_vbo = ctx.buffer(make_cube().tobytes())
        self.dyn_vbo = ctx.buffer(reserve=(MAX_DEBRIS + 80) * 28)
        self.dyn_vao = ctx.vertex_array(self.iprog, [
            (self.cube_vbo, "3f 3f", "in_vert", "in_norm"),
            (self.dyn_vbo, "3f 1f 3f/i", "in_pos", "in_scale", "in_color")])
        quad = np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype=np.float32)
        self.quad_vbo = ctx.buffer(quad.tobytes())
        self.sky_vao = ctx.vertex_array(self.sprog, [(self.quad_vbo, "2f", "in_pos")])
        self.overlay_vao = ctx.vertex_array(self.oprog, [(self.quad_vbo, "2f", "in_pos")])
        self.hud_tex = ctx.texture((HUD_W, HUD_H), 4)
        self.hud_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.oprog["u_tex"].value = 0
        self.hud = pygame.Surface((HUD_W, HUD_H), pygame.SRCALPHA)
        fam = "consolas,dejavusansmono,couriernew,monospace"
        self.font = pygame.font.SysFont(fam, 20)
        self.font_big = pygame.font.SysFont(fam, 30, bold=True)
        self.font_small = pygame.font.SysFont(fam, 15)
        self.font_title = pygame.font.SysFont(fam, 72, bold=True)
        self.tcache = {}
        self.vp = np.eye(4)
        self.sfx = Sfx()
        self.sfx.start_music()

        # --- build all maps
        self.loading("Building lobby", 0.02)
        lobby = build_lobby()
        self.loading("Building shop", 0.05)
        shop = build_shop()
        self.scenes = {"lobby": lobby, "shop": shop}
        self.loading("Preparing renderer", 0.92)
        for sc in self.scenes.values():
            sc.rend = ChunkRenderer(ctx, self.cprog, sc.world)
        self.scene = lobby

        # --- state
        self.debris = Debris()
        self.rockets = []
        self.pos = np.zeros(3)
        self.vel = np.zeros(3)
        self.yaw, self.pitch = 0.0, -0.1
        self.h = 1.8
        self.grounded = False
        self.coyote = 0.0
        self.step_off = 0.0
        self.coins = 0.0
        self.pick_lvl = 1
        self.has_jet = False
        self.tank_lvl = 1
        self.fuel = 100.0
        self.has_launcher = False
        self.rocket_ammo = 0
        self.blast_lvl = 1
        self.has_tnt = False
        self.tnt_ammo = 0
        self.tnt_lvl = 1
        self.tnts = []  # {pos, fuse, radius, remote}
        self.tool = 0
        # Extended upgrade levels (shop catalog)
        self.upg = {
            "pick_fortune": 0, "pick_speed": 0, "pick_reach": 0, "pick_vein": 0,
            "jet_thrust": 0, "jet_efficiency": 0, "jet_hover": 0, "jet_dash": 0,
            "rocket_speed": 0, "rocket_multi": 0, "rocket_cluster": 0, "rocket_sticky": 0,
            "move_speed": 0, "sprint_boost": 0, "jump_boost": 0, "air_control": 0,
            "double_jump": 0, "slide": 0,
            "loot_bonus": 0, "ore_sense": 0, "auto_smelt": 0, "magnet": 0,
            "monster_dmg": 0, "monster_range": 0, "thorns": 0, "lifesteal": 0,
            "map_range": 0, "quest_bonus": 0, "portal_cd": 0, "reveal_boost": 0,
            "shake_reduce": 0, "night_eye": 0, "storm_resist": 0, "pad_boost": 0,
            "crate_luck": 0, "ruin_radar": 0, "coin_interest": 0, "starter_pack": 0,
            "ammo_cap": 0, "fuel_cap": 0, "regen_boost": 0, "melee_sweep": 0,
            "drill_mode": 0, "quake": 0, "chain_mine": 0, "treasure_hunter": 0,
            "tnt_blast": 0, "tnt_fuse": 0, "tnt_chain": 0, "tnt_pack": 0,
        }
        self.extra_jumps = 0
        self.shop_item = 0          # selected catalog item index
        self.shop_scroll = 0
        self.shop_buy_rects = []
        self.shop_item_rects = []
        self.mining_cell = None
        self.progress = 0.0
        self.target = None
        self.mine_sound_t = self.jet_sound_t = self.cooldown = 0.0
        self.shake = 0.0
        self.col_box = None
        self.col_margin = 24
        self.col_timer = 0.0
        self.shop_open = False
        self.paused = False
        self.shop_rects = []
        self.map_open = False
        self.map_selected = None
        self.map_list_rects = []
        self.map_travel_rect = pygame.Rect(0, 0, 0, 0)
        self.map_quest_rect = pygame.Rect(0, 0, 0, 0)
        self.map_scroll = 0
        self.map_zoom = 1.0
        self.map_pan = [0.0, 0.0]
        self.map_dragging = False
        self.map_drag_last = None
        mw, md = 2560, 2560
        self.map_explored = np.zeros((mw, md), dtype=bool)
        self.map_base = None
        self.monsters = []
        self.quest_states = {i: {"kills": 0, "complete": False, "claimed": False,
                                  "started": False}
                             for i in range(20)}
        self.active_quest = None
        self.quest_banner_t = 0.0
        self.message, self.message_t = "", 0.0
        self.gain_text, self.gain_t, self.gain_acc = "", 0.0, 0.0
        self.time = 0.0
        self.fps = 60.0
        self.fade, self.fade_dir, self.pending = 1.0, -1, None
        self.l_hold = 0.0
        self.set_shop(False)

        # --- main menu / modes / weather --------------------------------
        self.in_menu = True
        self.menu_sel = 0          # 0 play, 1 mode, 2 quit
        self.game_mode = "sandbox"  # sandbox | survival | creative | chaos
        self.mode_list = ["sandbox", "survival", "creative", "chaos"]
        self.mode_desc = {
            "sandbox":  "Classic free play. Earn DCOIN, upgrade gear, explore all biomes.",
            "survival": "Start broke. No free coins. Monsters hit harder. Fuel drains faster.",
            "creative": "All tools unlocked, infinite fuel & rockets, free shop purchases.",
            "chaos":    "Bigger explosions, aggressive mobs, faster action.",
        }
        # Weather: clear | light | fog | rain | snow | storm | hail  (no lightning)
        self.weather = "clear"
        self.weather_t = 0.0
        self.weather_next = 90.0 + random.uniform(0, 60)
        self.weather_particles = []  # list of [x, y, z, vx, vy, vz, life, kind]
        self.WEATHER_ORDER = ["clear", "light", "fog", "rain", "snow", "storm", "hail"]
        # multiplayer
        # One permanent full UUID per installation. It is stored beside local saves
        # and reused for every multiplayer connection and game restart.
        identity_path = os.path.join(self._save_dir(), "player_identity.json")
        identity = {}
        try:
            with open(identity_path, "r", encoding="utf-8") as identity_file:
                identity = json.load(identity_file)
        except (OSError, ValueError, TypeError):
            identity = {}

        try:
            saved_id = str(identity.get("player_uuid", ""))
            parsed_id = uuid.UUID(saved_id)
            self.player_id = str(parsed_id)
        except (ValueError, TypeError, AttributeError):
            self.player_id = str(uuid.uuid4())

        # The server's save filenames are keyed by player_name (max 16 chars).
        # Derive a stable, practically unique name from the permanent UUID.
        self.player_name = "Pilot" + self.player_id.replace("-", "")[-11:]
        identity_payload = {
            "player_uuid": self.player_id,
            "player_name": self.player_name,
        }
        try:
            os.makedirs(os.path.dirname(identity_path), exist_ok=True)
            temp_identity = identity_path + ".tmp"
            with open(temp_identity, "w", encoding="utf-8") as identity_file:
                json.dump(identity_payload, identity_file, separators=(",", ":"))
                identity_file.flush()
                os.fsync(identity_file.fileno())
            os.replace(temp_identity, identity_path)
        except OSError as identity_error:
            print(f"[IDENTITY] Could not persist player UUID: {identity_error}")
        self.net_server = None
        self.net_client = NetClient()
        self.is_host = False
        self.mp_open = False
        self.mp_servers = default_server_list()
        self.mp_selected = 0
        self.mp_status = "Idle"
        self.mp_remote = {}  # id -> player dict (from client or server view)
        self.mp_send_t = 0.0
        self.mp_resync_t = 0.0
        self.chat_open = False
        self.chat_buf = ""
        self.chat_log = []          # (name, msg, time) recent lines
        self.chat_log_ttl = 12.0
        self.is_op = False
        self.spectator = False
        self.settings_open = False
        self.shader_pack = "insane"   # default | pro | insane
        self.SHADER_PACKS = ("default", "pro", "insane", "retro", "noir", "sunset", "arctic")
        self.shader_effects = {
            "sky": True,
            "lighting": True,
            "fog": True,
            "grade": True,
            "emissive": True,
        }
        self.singleplayer_features = {
            "peaceful": False,
            "infinite_ammo": False,
            "moon_gravity": False,
        }
        self.render_distance = float(VIEW_DIST)
        self.load_settings()
        self.settings_rects = []
        self.hide_hud = False
        self.world_epoch = None   # server world-reset counter we are in sync with
        self.mp_seen = set()      # event ids already applied
        self.mp_last_eid = 0      # newest event id seen (for resync{since})
        self._join_thread = None
        self._join_result = None
        self.mp_join_host = OFFICIAL_HOST
        self.mp_join_port = str(OFFICIAL_PORT)
        self.mp_rects = []
        self.autosave_t = 0.0
        self.autosave_dirty = False
        pygame.mouse.set_visible(True)
        pygame.event.set_grab(False)

    def make_world_map(self):
        """Build a bright, detailed atlas texture from the actual generated biome grid."""
        sc = self.scenes["main"]
        colors = np.asarray(sc.biome_colors, dtype=np.float32)
        ids = sc.biome_map.astype(np.int16)
        surface = colors[ids]

        # Subtle deterministic terrain grain makes each region feel like a map,
        # while keeping the biome colors clear and readable at overview scale.
        rng = np.random.default_rng(42020)
        grain = rng.normal(0.0, 5.0, ids.shape).astype(np.float32)
        surface = np.clip(surface + grain[:, :, None], 0, 255)

        # Draw bright, thin borders where neighbouring biome territories meet.
        edge = np.zeros(ids.shape, dtype=bool)
        edge[1:, :] |= ids[1:, :] != ids[:-1, :]
        edge[:-1, :] |= ids[:-1, :] != ids[1:, :]
        edge[:, 1:] |= ids[:, 1:] != ids[:, :-1]
        edge[:, :-1] |= ids[:, :-1] != ids[:, 1:]
        surface[edge] = np.clip(surface[edge] * 0.68 + 78, 0, 255)

        # City streets/building pads stand out as a pale urban core.
        surface[ids == 0] = np.clip(surface[ids == 0] * 0.72 + 45, 0, 255)
        return pygame.surfarray.make_surface(surface.astype(np.uint8)).convert()

    def spawn_biome_monsters(self):
        """Place three hostile cave creatures around each biome landmark."""
        if self.singleplayer_feature_enabled("peaceful"):
            return
        sc = self.scenes["main"]
        mw, md = sc.map_size
        rng = random.Random(81427)
        for bi, (sx, sz) in enumerate(sc.biome_seeds):
            made = 0
            for _ in range(100):
                x = min(mw - 3, max(3, sx + rng.randint(-42, 42)))
                z = min(md - 3, max(3, sz + rng.randint(-42, 42)))
                if sc.biome_map[x, z] != bi:
                    continue
                col = sc.world.blocks[x, :, z]
                solid = np.where(col != 0)[0]
                if not len(solid):
                    continue
                y = (int(solid[-1]) + 1) * VS
                self.monsters.append({"pos": np.array([x * VS, y + 0.72, z * VS], float),
                                      "biome": bi, "hp": 3, "max_hp": 3,
                                      "phase": rng.random() * 6.28, "alive": True,
                                      "active": False})
                made += 1
                if made == 3:
                    break

    def start_biome_quest(self, biome_id):
        """Start a biome quest only when the player explicitly accepts it."""
        if biome_id is None or not 0 <= int(biome_id) < 20:
            return
        biome_id = int(biome_id)
        state = self.quest_states[biome_id]
        if state["complete"]:
            self.say("That biome quest is already complete")
            return
        if state["started"]:
            self.say("Quest already active: " + self.scene.biome_names[biome_id])
            return
        state["started"] = True
        self.active_quest = biome_id
        for monster in self.monsters:
            if monster["biome"] == biome_id and monster["alive"]:
                monster["active"] = True
        self.say("QUEST STARTED: " + self.quest_info(biome_id), 4.0)

    def current_biome(self):
        if self.scene.name != "main":
            return None
        mw, md = self.scene.map_size
        x = min(mw - 1, max(0, int(self.pos[0] / VS)))
        z = min(md - 1, max(0, int(self.pos[2] / VS)))
        return int(self.scene.biome_map[x, z])

    def quest_info(self, bi):
        names = [
            "Downtown lockdown: eliminate the street mutants", "Buried Tomb: clear the desert cave of monsters",
            "Dune Stalkers: purge the red-dune den", "Badlands Bunker: defeat the raiders",
            "Savanna Hunt: clear the predator nest", "Greenway: remove the cave crawlers",
            "Bloomguard: cleanse the flower-field shrine", "Darkwood: hunt the forest lurkers",
            "Pinewatch: clear the frozen burrow", "Redwood Heart: defeat the root beasts",
            "Sunken Mire: clear the swamp lair", "Jungle Ruins: defeat the temple guards",
            "Bamboo Hollow: clear the hidden den", "Tundra Station: defeat the ice horrors",
            "Glacier Vault: clear the frozen cave", "Volcanic Core: defeat the ember beasts",
            "Ashfall: clear the ash-waste nest", "Crystal Mine: defeat the crystal guardians",
            "Moonstone Vault: clear the underground den", "Coast Wreck: defeat the wreck scavengers"
        ]
        return names[bi]

    def damage_monster(self, monster, damage=1):
        if not monster["alive"] or not monster.get("active", False):
            return
        monster["hp"] -= damage
        if monster["hp"] <= 0:
            monster["alive"] = False
            bi = monster["biome"]
            state = self.quest_states[bi]
            state["kills"] += 1
            self.add_coins(35 + bi * 2)
            if self.upg.get("lifesteal", 0) and self.has_jet:
                self.fuel = min(self.max_fuel, self.fuel + 8 * self.upg["lifesteal"])
            self.sfx.play("break")
            if state["kills"] >= 3 and not state["complete"]:
                state["complete"] = True
                bonus = int(250 * (1.0 + 0.2 * self.upg.get("quest_bonus", 0)))
                self.add_coins(bonus)
                self.say(f"QUEST COMPLETE: {self.scene.biome_names[bi]} +{bonus} DCOIN", 4.0)
            else:
                self.say(f"Monster defeated ({state['kills']}/3) - +{35 + bi * 2} DCOIN")

    def update_monsters(self, dt):
        if self.scene.name != "main" or self.singleplayer_feature_enabled("peaceful"):
            return
        for m in self.monsters:
            if not m["alive"] or not m.get("active", False):
                continue
            m["phase"] += dt * 3.0
            delta = self.pos - m["pos"]
            delta[1] = 0
            dist = float(np.linalg.norm(delta))
            if 2.2 < dist < 18.0:
                delta /= max(dist, 1e-5)
                spd = 2.2
                if self.game_mode == "chaos":
                    spd = 3.6
                elif self.game_mode == "survival":
                    spd = 2.8
                m["pos"] += delta * min(spd * dt, dist - 2.0)

    def monster_target(self, reach=7.0):
        eye, fwd = self.eye(), self.forward()
        best, best_score = None, reach
        for m in self.monsters:
            if not m["alive"] or not m.get("active", False):
                continue
            d = m["pos"] - eye
            dist = float(np.linalg.norm(d))
            if dist < 0.1 or dist > reach:
                continue
            alignment = float(np.dot(d / dist, fwd))
            if alignment > 0.965 and dist < best_score:
                best, best_score = m, dist
        return best

    def reveal_map(self):
        if self.scene.name != "main":
            return
        mw, md = self.scene.map_size
        x = int(self.pos[0] / VS)
        z = int(self.pos[2] / VS)
        radius = 22 + 6 * self.upg.get("map_range", 0)
        x0, x1 = max(0, x - radius), min(mw, x + radius + 1)
        z0, z1 = max(0, z - radius), min(md, z + radius + 1)
        xx = np.arange(x0, x1)[:, None]
        zz = np.arange(z0, z1)[None, :]
        self.map_explored[x0:x1, z0:z1] |= (
            (xx - x) ** 2 + (zz - z) ** 2 <= radius * radius)

    def open_map(self, opened):
        self.map_open = opened
        if not opened:
            self.map_selected = None
        pygame.mouse.set_visible(opened or self.shop_open or self.paused)
        pygame.event.set_grab(not (opened or self.shop_open or self.paused))
        pygame.mouse.get_rel()
        if opened:
            self.reveal_map()

    def map_destinations(self):
        # Keep every destination in the atlas. Exploration changes its detail,
        # not whether the player can see that the location exists.
        if self.scene.name != "main":
            return []
        mw, md = self.scene.map_size
        for d in self.scene.map_destinations:
            x, z = d["x"], d["z"]
            xa, xb = max(0, x - 12), min(mw, x + 13)
            za, zb = max(0, z - 12), min(md, z + 13)
            d["discovered"] = bool(self.map_explored[xa:xb, za:zb].any())
        return self.scene.map_destinations

    def quick_travel(self, destination):
        if self.scene.name != "main":
            return
        mw, md = self.scene.map_size
        x, z = destination["x"], destination["z"]
        y = int(np.max(np.where(self.scene.world.blocks[x, :, z] != 0)[0], initial=0)) + 1
        target = np.array([x * VS, y * VS + 0.03, z * VS], dtype=float)
        if self.scene.world.box_hit(target[0] - .5, target[1], target[2] - .5,
                                    target[0] + .5, target[1] + self.h, target[2] + .5):
            for radius in range(1, 22):
                candidates = [(x + dx, z + dz) for dx in range(-radius, radius + 1)
                              for dz in range(-radius, radius + 1)
                              if abs(dx) == radius or abs(dz) == radius]
                valid = [(a, b) for a, b in candidates
                         if 2 <= a < mw - 2 and 2 <= b < md - 2]
                for a, b in valid:
                    yy = int(np.max(np.where(self.scene.world.blocks[a, :, b] != 0)[0], initial=0)) + 1
                    p = np.array([a * VS, yy * VS + .03, b * VS])
                    if not self.scene.world.box_hit(p[0]-.3, p[1], p[2]-.3,
                                                    p[0]+.3, p[1]+self.h, p[2]+.3):
                        target = p
                        break
                else:
                    continue
                break
        self.pos = target
        self.vel[:] = 0
        self.grounded = False
        self.reveal_map()
        self.open_map(False)
        self.say(f"Travelled to {destination['name']}")

    def draw_map(self):
        s = self.hud
        overlay = pygame.Surface((HUD_W, HUD_H), pygame.SRCALPHA)
        overlay.fill((4, 7, 13, 246))
        s.blit(overlay, (0, 0))
        self.t("WORLD ATLAS", 28, 20, (245, 248, 255), self.font_title)
        self.t("DRAG TO PAN  |  WHEEL TO ZOOM  |  M / ESC CLOSE", HUD_W - 470, 42,
               (175, 190, 210), self.font_small)

        map_rect = pygame.Rect(16, 100, 680, 575)
        pygame.draw.rect(s, (9, 15, 23), map_rect, border_radius=10)
        pygame.draw.rect(s, (64, 83, 112), map_rect, 2, border_radius=10)
        mw, md = self.scene.map_size
        # Fit the complete world, then apply user zoom and pan.
        base_scale = min((map_rect.w - 12) / mw, (map_rect.h - 12) / md)
        scale = base_scale * self.map_zoom
        draw_w = int(mw * scale)
        draw_h = int(md * scale)
        base = pygame.transform.scale(self.map_base, (draw_w, draw_h))
        draw_x = int(map_rect.centerx - draw_w / 2 + self.map_pan[0])
        draw_y = int(map_rect.centery - draw_h / 2 + self.map_pan[1])
        old_clip = s.get_clip()
        s.set_clip(map_rect)
        s.blit(base, (draw_x, draw_y))
        # Keep the full atlas readable from the start. Unexplored territory is
        # only slightly muted; it is never hidden behind opaque black squares.
        fog = pygame.Surface((mw, md), pygame.SRCALPHA)
        alpha = pygame.surfarray.pixels_alpha(fog)
        alpha[:] = np.where(self.map_explored, 0, 28)
        del alpha
        fog = pygame.transform.scale(fog, (draw_w, draw_h))
        s.blit(fog, (draw_x, draw_y))
        destinations = self.map_destinations()
        for d in destinations:
            px = draw_x + d["x"] * scale
            py = draw_y + d["z"] * scale
            if not map_rect.collidepoint(int(px), int(py)):
                continue
            if d["kind"] == "POI":
                col = (255, 190, 75)
                radius = 4
            elif d["kind"] == "QUEST":
                col = (255, 75, 105)
                radius = 6
            else:
                col = self.scene.biome_colors[d.get("biome", 0)]
                radius = 7 if d.get("discovered") else 5
            pygame.draw.circle(s, (8, 12, 18), (int(px), int(py)), radius + 3)
            pygame.draw.circle(s, col, (int(px), int(py)), radius)
            if d.get("kind") == "BIOME":
                label = self.font_small.render(d["name"], True, (255, 255, 245))
                label_bg = label.get_rect(topleft=(int(px) + 10, int(py) - 9)).inflate(8, 5)
                pygame.draw.rect(s, (12, 19, 27), label_bg, border_radius=4)
                s.blit(label, (label_bg.x + 4, label_bg.y + 2))
            elif self.map_zoom >= 1.0:
                label = self.font_small.render(d["name"], True, (255, 224, 190))
                s.blit(label, (int(px) + 9, int(py) - 8))
        px = draw_x + self.pos[0] / VS * scale
        py = draw_y + self.pos[2] / VS * scale
        pygame.draw.circle(s, (255, 65, 75), (int(px), int(py)), 8)
        pygame.draw.circle(s, (255, 255, 255), (int(px), int(py)), 11, 2)
        s.set_clip(old_clip)

        panel = pygame.Rect(710, 100, 536, 575)
        pygame.draw.rect(s, (17, 23, 34), panel, border_radius=10)
        self.t("20 BIOMES  /  QUEST LOG", 730, 119, (240, 245, 255), self.font_big)
        self.t("All regions are shown. Explore to reveal details and complete quests.",
               730, 154, (165, 180, 200), self.font_small)
        self.map_list_rects = []
        # Keep all twenty biome entries visible at once. Quest caves and POIs
        # remain individually selectable from their map markers.
        biome_destinations = [d for d in destinations if d["kind"] == "BIOME"]
        for i, d in enumerate(biome_destinations[:20]):
            rect = pygame.Rect(728, 181 + i * 21, 498, 19)
            self.map_list_rects.append((rect, d))
            selected = self.map_selected is d
            pygame.draw.rect(s, (56, 76, 108) if selected else (27, 36, 51),
                             rect, border_radius=4)
            col = self.scene.biome_colors[d.get("biome", 0)]
            pygame.draw.circle(s, col, (rect.x + 12, rect.centery), 5)
            self.t(d["name"], rect.x + 25, rect.y + 1, (245, 248, 255), self.font_small)
            bi = d["biome"]
            qs = self.quest_states[bi]
            tag = ("CLEARED" if qs["complete"] else
                   (f"{qs['kills']}/3 MONSTERS" if qs["started"] else "QUEST AVAILABLE"))
            self.t(tag, rect.right - 112, rect.y + 1,
                   (115, 235, 180) if qs["complete"] else (190, 205, 222), self.font_small)
        selected_name = self.map_selected["name"] if self.map_selected else "Select a biome to view its quest"
        self.t(selected_name[:48], 730, 602, (255, 220, 145), self.font_small)
        selected_biome = self.map_selected.get("biome") if self.map_selected else None
        is_biome = self.map_selected is not None and self.map_selected.get("kind") == "BIOME"
        quest_started = bool(is_biome and self.quest_states[selected_biome]["started"])
        quest_done = bool(is_biome and self.quest_states[selected_biome]["complete"])
        self.map_quest_rect = pygame.Rect(728, 628, 242, 36)
        self.map_travel_rect = pygame.Rect(982, 628, 244, 36)
        quest_label = ("QUEST COMPLETE" if quest_done else
                       ("QUEST ACTIVE" if quest_started else "START QUEST"))
        pygame.draw.rect(s, (43, 105, 83) if (is_biome and not quest_done and not quest_started)
                         else (48, 57, 70), self.map_quest_rect, border_radius=7)
        self.t(quest_label, self.map_quest_rect.centerx, self.map_quest_rect.y + 8,
               (255, 255, 255) if is_biome else (135, 145, 160), self.font_small, True)
        can_travel = self.map_selected in destinations
        pygame.draw.rect(s, (28, 130, 105) if can_travel else (55, 60, 70),
                         self.map_travel_rect, border_radius=7)
        self.t("QUICK TRAVEL" if can_travel else "SELECT A LOCATION",
               self.map_travel_rect.centerx, self.map_travel_rect.y + 8,
               (255, 255, 255) if can_travel else (145, 150, 160), self.font_small, True)
        self.t("QUESTS ARE OPTIONAL: select a biome and press START QUEST  |  Pickaxe attacks  |  Rockets deal splash damage",
               18, 690, (170, 185, 205), self.font_small)

    # ------------------------------------------------------------------ utils
    @property
    def max_fuel(self):
        return 100.0 + 40.0 * (self.tank_lvl - 1) + 25.0 * self.upg.get("fuel_cap", 0)

    @property
    def blast_radius(self):
        r = 2.6 + 0.7 * self.blast_lvl + 0.35 * self.upg.get("rocket_cluster", 0)
        if self.game_mode == "chaos":
            r *= 1.45
        return r

    @property
    def pick_radius(self):
        return 1.6 + 0.45 * self.pick_lvl + 0.25 * self.upg.get("pick_reach", 0)

    @property
    def max_ammo(self):
        return 30 + 10 * self.upg.get("ammo_cap", 0)

    @property
    def max_tnt(self):
        return 8 + 4 * self.upg.get("tnt_pack", 0)

    @property
    def tnt_radius(self):
        return 3.2 + 0.55 * (self.tnt_lvl - 1) + 0.4 * self.upg.get("tnt_blast", 0)

    @property
    def tnt_fuse_time(self):
        # Lower is faster fuse; upgrade shortens it
        return max(0.55, 2.4 - 0.35 * self.upg.get("tnt_fuse", 0))

    def eye(self):
        return self.pos + np.array([0.0, self.h - 0.18 + self.step_off, 0.0])

    def forward(self):
        cp = math.cos(self.pitch)
        return np.array([math.sin(self.yaw) * cp, math.sin(self.pitch), -math.cos(self.yaw) * cp])

    def say(self, text, t=2.0):
        self.message, self.message_t = text, t

    def _push_chat(self, name, msg):
        line = f"{name}: {msg}"
        self.chat_log.append([name, msg, self.chat_log_ttl])
        if len(self.chat_log) > 8:
            self.chat_log = self.chat_log[-8:]
        self.say(line, 4.0)

    def set_spectator(self, on):
        if on and not self.is_op:
            self.say("Spectator requires OP (server: op <name>)", 3.0)
            self.sfx.play("deny")
            return
        self.spectator = bool(on)
        if self.spectator:
            self.vel[:] = 0
            self.say("SPECTATOR ON — noclip fly  (F2 off)", 3.0)
        else:
            self.say("Spectator OFF", 2.0)

    def set_shader_pack(self, name):
        name = name if name in self.SHADER_PACKS else "default"
        self.shader_pack = name
        self.save_settings()
        labels = {
            "default": "Shader pack: DEFAULT",
            "pro": "Shader pack: PRO — richer light + specular",
            "insane": "Shader pack: INSANE — cinematic YT look (balanced glow, softer contrast)",
            "retro": "Shader pack: RETRO — chunky, posterized colors",
            "noir": "Shader pack: NOIR — high-contrast monochrome",
            "sunset": "Shader pack: SUNSET — warm amber color grade",
            "arctic": "Shader pack: ARCTIC — crisp cool color grade",
        }
        self.say(labels.get(name, name), 3.0)

    def load_settings(self):
        path = os.path.join(_app_dir(), "client_settings.json")
        try:
            with open(path, "r", encoding="utf-8") as settings_file:
                data = json.load(settings_file)
            if data.get("shader_pack") in self.SHADER_PACKS:
                self.shader_pack = data["shader_pack"]
            effects = data.get("shader_effects", {})
            for name in self.shader_effects:
                if name in effects:
                    self.shader_effects[name] = bool(effects[name])
            features = data.get("singleplayer_features", {})
            for name in self.singleplayer_features:
                if name in features:
                    self.singleplayer_features[name] = bool(features[name])
            distance = float(data.get("render_distance", self.render_distance))
            self.render_distance = max(VIEW_DIST_MIN, min(VIEW_DIST_MAX, distance))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass

    def save_settings(self):
        path = os.path.join(_app_dir(), "client_settings.json")
        data = {
            "shader_pack": self.shader_pack,
            "shader_effects": self.shader_effects,
            "singleplayer_features": self.singleplayer_features,
            "render_distance": self.render_distance,
        }
        try:
            with open(path, "w", encoding="utf-8") as settings_file:
                json.dump(data, settings_file, separators=(",", ":"))
        except OSError:
            pass

    def toggle_shader_effect(self, name):
        if name not in self.shader_effects:
            return
        self.shader_effects[name] = not self.shader_effects[name]
        self.save_settings()

    def toggle_singleplayer_feature(self, name):
        if name not in self.singleplayer_features:
            return
        self.singleplayer_features[name] = not self.singleplayer_features[name]
        if name == "peaceful" and self.singleplayer_feature_enabled(name):
            self.monsters.clear()
        self.save_settings()

    def singleplayer_feature_enabled(self, name):
        return not self._in_multiplayer() and bool(self.singleplayer_features.get(name, False))

    def open_settings(self, open_=None):
        self.settings_open = (not self.settings_open) if open_ is None else bool(open_)
        if self.settings_open:
            self.shop_open = False
            self.mp_open = False
            self.map_open = False
            self.chat_open = False
            pygame.mouse.set_visible(True)
            pygame.event.set_grab(False)
        elif not self.in_menu and not self.paused and not self.shop_open and not self.mp_open:
            pygame.mouse.set_visible(False)
            pygame.event.set_grab(True)

    def draw_settings(self):
        s = self.hud
        dim = pygame.Surface((HUD_W, HUD_H), pygame.SRCALPHA)
        dim.fill((4, 6, 12, 210))
        s.blit(dim, (0, 0))
        panel = pygame.Rect(120, 80, HUD_W - 240, HUD_H - 160)
        pygame.draw.rect(s, (14, 18, 28), panel, border_radius=14)
        pygame.draw.rect(s, (90, 130, 200), panel, 2, border_radius=14)
        self.t("SETTINGS", panel.x + 24, panel.y + 18, (255, 230, 160), self.font_big)
        self.t("Esc close", panel.right - 120, panel.y + 24, (140, 155, 175), self.font_small)
        if self.in_menu:
            back = pygame.Rect(panel.right - 148, panel.bottom - 48, 124, 32)
            pygame.draw.rect(s, (36, 48, 68), back, border_radius=6)
            self.t("BACK", back.centerx, back.y + 7, (235, 240, 250), self.font_small, True)
            self.settings_rects = [("back", None, back)]
        self.t("SHADER PACK", panel.x + 24, panel.y + 70, (140, 180, 230), self.font_small)
        if not self.in_menu:
            self.settings_rects = []
        hx, hy = self.mouse_hud()
        desc = {
            "default": "Classic voxel palette",
            "pro": "Rich light and specular",
            "insane": "Cinematic contrast",
            "retro": "Posterized pixel colors",
            "noir": "High-contrast monochrome",
            "sunset": "Warm amber tones",
            "arctic": "Cool blue highlights",
        }
        for index, pack in enumerate(self.SHADER_PACKS):
            col, row_index = index % 2, index // 2
            row = pygame.Rect(panel.x + 24 + col * 236, panel.y + 100 + row_index * 64, 224, 56)
            sel = pack == self.shader_pack
            hover = row.collidepoint(hx, hy)
            pygame.draw.rect(s, (45, 75, 120) if sel else ((30, 40, 58) if hover else (18, 24, 36)), row, border_radius=8)
            pygame.draw.rect(s, (120, 200, 255) if sel else (50, 60, 80), row, 1, border_radius=8)
            self.t(pack.upper(), row.x + 10, row.y + 7, (255, 255, 255), self.font)
            self.t(desc[pack], row.x + 10, row.y + 31, (160, 175, 195), self.font_small)
            self.settings_rects.append(("shader", pack, row))
        effects_x = panel.x + 530
        self.t("SHADER EFFECTS", effects_x, panel.y + 70, (140, 180, 230), self.font_small)
        effect_labels = {
            "sky": "Sky rendering",
            "lighting": "Dynamic lighting",
            "fog": "Distance fog",
            "grade": "Color grading",
            "emissive": "Emissive animation",
        }
        for index, (name, label) in enumerate(effect_labels.items()):
            row = pygame.Rect(effects_x, panel.y + 100 + index * 42, 470, 34)
            hover = row.collidepoint(hx, hy)
            pygame.draw.rect(s, (30, 40, 58) if hover else (18, 24, 36), row, border_radius=6)
            checked = self.shader_effects[name]
            box = pygame.Rect(row.x + 10, row.y + 7, 20, 20)
            pygame.draw.rect(s, (120, 200, 255) if checked else (50, 60, 80), box, 2, border_radius=3)
            if checked:
                pygame.draw.line(s, (230, 245, 255), (box.x + 4, box.y + 10), (box.x + 8, box.y + 15), 2)
                pygame.draw.line(s, (230, 245, 255), (box.x + 8, box.y + 15), (box.x + 17, box.y + 4), 2)
            self.t(label, row.x + 42, row.y + 8, (230, 235, 245), self.font_small)
            self.settings_rects.append(("effect", name, row))
        self.t("SINGLEPLAYER EXTRAS", effects_x, panel.y + 340, (140, 180, 230), self.font_small)
        feature_labels = {
            "peaceful": "Peaceful exploration",
            "infinite_ammo": "Infinite rockets and TNT",
            "moon_gravity": "Moon gravity",
        }
        for index, (name, label) in enumerate(feature_labels.items()):
            row = pygame.Rect(effects_x, panel.y + 370 + index * 42, 470, 34)
            hover = row.collidepoint(hx, hy)
            pygame.draw.rect(s, (30, 40, 58) if hover else (18, 24, 36), row, border_radius=6)
            checked = self.singleplayer_features[name]
            box = pygame.Rect(row.x + 10, row.y + 7, 20, 20)
            pygame.draw.rect(s, (120, 200, 255) if checked else (50, 60, 80), box, 2, border_radius=3)
            if checked:
                pygame.draw.line(s, (230, 245, 255), (box.x + 4, box.y + 10), (box.x + 8, box.y + 15), 2)
                pygame.draw.line(s, (230, 245, 255), (box.x + 8, box.y + 15), (box.x + 17, box.y + 4), 2)
            self.t(label, row.x + 42, row.y + 8, (230, 235, 245), self.font_small)
            self.settings_rects.append(("sp", name, row))
        self.t("Only active in single-player.", effects_x, panel.y + 510, (130, 145, 165), self.font_small)
        y = panel.y + 370
        self.t("RENDER DISTANCE", panel.x + 24, y, (140, 180, 230), self.font_small)
        y += 28
        rd = int(getattr(self, "render_distance", VIEW_DIST))
        self.t(f"{rd} m   (64 – 452)", panel.x + 24, y, (255, 255, 255), self.font)
        y += 30
        # preset buttons
        presets = [80, 120, 160, 220, 300, 380, 452]
        bx = panel.x + 24
        for val in presets:
            r = pygame.Rect(bx, y, 64, 32)
            sel = abs(rd - val) < 1
            pygame.draw.rect(s, (45, 75, 120) if sel else (22, 28, 42), r, border_radius=6)
            pygame.draw.rect(s, (120, 200, 255) if sel else (50, 60, 80), r, 1, border_radius=6)
            self.t(str(val), r.centerx, r.y + 6, (255, 255, 255), self.font_small, True)
            self.settings_rects.append(("rd", val, r))
            bx += 68
        y += 48
        self.t("Higher distance = heavier CPU/GPU. 452 is trailer-cam range.",
               panel.x + 24, y, (130, 145, 165), self.font_small)

    def set_render_distance(self, meters):
        self.render_distance = float(max(VIEW_DIST_MIN, min(VIEW_DIST_MAX, meters)))
        self.save_settings()
        self.say(f"Render distance {int(self.render_distance)} m", 2.0)

    def _send_chat_line(self, text):
        text = (text or "").strip()[:80]
        if not text:
            return
        if not self._in_multiplayer():
            self.say("Chat only works in multiplayer", 2.0)
            return
        # Echo locally immediately; server will not bounce it back to us with exclude
        # (server broadcasts exclude sender — so we must show our own line).
        self._push_chat(self.player_name, text)
        if self.net_client.connected:
            self.net_client.send_chat(text)
        elif self.is_host and self.net_server:
            self.net_server._broadcast({"t": "chat", "id": self.player_id,
                                        "name": self.player_name, "msg": text})

    def add_coins(self, n):
        if n <= 0:
            return
        self.coins += n
        if self.gain_t <= 0:
            self.gain_acc = 0.0
        self.gain_acc += n
        self.gain_text = f"+{int(self.gain_acc)}"
        self.gain_t = 1.0
        self.mark_progress_dirty()

    # ---------------------------------------------------------- menu / modes
    def start_game(self):
        """Leave the main menu and apply the selected game mode."""
        # Singleplayer must never stay attached to a server/host session
        if self.net_client.connected or self.is_host:
            try:
                if self.net_server:
                    self.net_server.stop()
                    self.net_server = None
                self.net_client.close()
            except Exception:
                pass
            self.is_host = False
            self.mp_remote = {}
            self.mp_status = "Disconnected"
        self.in_menu = False
        mode = self.game_mode
        # reset player loadout
        self.coins = 0.0
        self.pick_lvl = 1
        self.has_jet = False
        self.tank_lvl = 1
        self.fuel = 100.0
        self.has_launcher = False
        self.rocket_ammo = 0
        self.blast_lvl = 1
        self.has_tnt = False
        self.tnt_ammo = 0
        self.tnt_lvl = 1
        self.tnts = []
        self.tool = 0
        if mode == "creative":
            self.coins = 99999
            self.pick_lvl = 6
            self.has_jet = True
            self.tank_lvl = 5
            self.fuel = self.max_fuel
            self.has_launcher = True
            self.rocket_ammo = 30
            self.blast_lvl = 5
            self.has_tnt = True
            self.tnt_ammo = 20
            self.tnt_lvl = 5
        elif mode == "sandbox":
            self.coins = 50
        elif mode == "survival":
            self.coins = 0
        elif mode == "chaos":
            self.coins = 100
            self.has_launcher = True
            self.rocket_ammo = 10
            self.blast_lvl = 2
            self.has_tnt = True
            self.tnt_ammo = 4
            self.tnt_lvl = 2
        self.autosave_t = 0.0
        # Restore coins/gear from offline client_save only
        if mode != "creative":
            if self.load_progress(quiet=True):
                self.say(f"Mode: {mode.upper()}  ·  progress restored ({int(self.coins)} DCOIN)", 4.0)
            else:
                self.say(f"Mode: {mode.upper()}  –  walk into a portal to begin", 4.0)
        else:
            self.say(f"Mode: {mode.upper()}  –  walk into a portal to begin", 4.0)
        # CRITICAL: load client_save world only — never keep multiplayer holes
        self.enter_singleplayer_world()
        self.switch_scene("lobby", "default")
        self.fade, self.fade_dir = 1.0, -1
        pygame.mouse.set_visible(False)
        pygame.event.set_grab(True)
        pygame.mouse.get_rel()

    def cycle_mode(self, delta=1):
        i = self.mode_list.index(self.game_mode)
        self.game_mode = self.mode_list[(i + delta) % len(self.mode_list)]
        self.sfx.play("coin")


    def draw_menu(self):
        if self.settings_open:
            self.draw_settings()
            return
        s = self.hud
        s.fill((0, 0, 0, 0))
        # dark backdrop
        overlay = pygame.Surface((HUD_W, HUD_H), pygame.SRCALPHA)
        overlay.fill((6, 8, 14, 245))
        s.blit(overlay, (0, 0))
        # title
        self.t("DESTRUX", HUD_W // 2, 110, (255, 210, 60), self.font_title, True)
        self.t("voxel destruction sandbox", HUD_W // 2, 195, (160, 175, 195), self.font, True)

        items = [
            ("PLAY", "Start in the lobby"),
            (f"MODE: {self.game_mode.upper()}", self.mode_desc[self.game_mode]),
            ("SETTINGS", "Shaders, effects and single-player extras"),
            ("QUIT", "Exit the game"),
        ]
        y0 = 242
        hx, hy = self.mouse_hud()
        self.menu_rects = []
        for i, (label, desc) in enumerate(items):
            rect = pygame.Rect(HUD_W // 2 - 260, y0 + i * 88, 520, 68)
            self.menu_rects.append(rect)
            hover = rect.collidepoint(hx, hy) or self.menu_sel == i
            if hover:
                self.menu_sel = i
            pygame.draw.rect(s, (55, 85, 130) if hover else (28, 34, 48), rect, border_radius=10)
            pygame.draw.rect(s, (255, 200, 70) if hover else (60, 70, 90), rect, 2, border_radius=10)
            self.t(label, rect.centerx, rect.y + 12, (255, 255, 255) if hover else (200, 210, 225),
                   self.font_big, True)
            self.t(desc[:62], rect.centerx, rect.y + 44, (170, 185, 205), self.font_small, True)

        self.t("Arrow keys / mouse  ·  Enter or click to select  ·  Left/Right cycle mode  ·  N music",
               HUD_W // 2, HUD_H - 50, (130, 145, 165), self.font_small, True)
        self.t(f"Weather systems & biomes active  ·  20 regions  ·  {self.game_mode} mode",
               HUD_W // 2, HUD_H - 28, (110, 125, 145), self.font_small, True)

    def set_shop(self, open_):
        self.shop_open = open_
        locked = open_ or getattr(self, "paused", False) or getattr(self, "map_open", False) or getattr(self, "in_menu", False)
        pygame.mouse.set_visible(locked)
        pygame.event.set_grab(not locked)
        pygame.mouse.get_rel()

    def set_paused(self, p):
        self.paused = p
        locked = p or getattr(self, "map_open", False) or getattr(self, "in_menu", False)
        pygame.mouse.set_visible(locked)
        pygame.event.set_grab(not locked)
        pygame.mouse.get_rel()

    # --------------------------------------------------------------- save/load
    def _save_dir(self):
        """Client-only save root: client_save/ — never shared with the server."""
        base = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
        d = os.path.join(base, CLIENT_SAVE_DIR)
        os.makedirs(d, exist_ok=True)
        return d

    def _save_path(self, slot=None):
        slot = slot or SAVE_SLOT
        return os.path.join(self._save_dir(), f"{slot}.npz")

    def _player_blob(self):
        return {
            "coins": float(self.coins),
            "pick_lvl": int(self.pick_lvl),
            "has_jet": bool(self.has_jet),
            "tank_lvl": int(self.tank_lvl),
            "fuel": float(self.fuel),
            "has_launcher": bool(self.has_launcher),
            "rocket_ammo": int(self.rocket_ammo),
            "blast_lvl": int(self.blast_lvl),
            "has_tnt": bool(self.has_tnt),
            "tnt_ammo": int(self.tnt_ammo),
            "tnt_lvl": int(self.tnt_lvl),
            "tool": int(self.tool),
            "upg": dict(self.upg),
            "pos": [float(self.pos[0]), float(self.pos[1]), float(self.pos[2])],
            "yaw": float(self.yaw),
            "pitch": float(self.pitch),
            "scene": self.scene.name,
            "game_mode": self.game_mode,
            "player_name": self.player_name,
            "active_quest": self.active_quest,
            "quest_states": {str(k): dict(v) for k, v in self.quest_states.items()},
        }

    def save_game(self, slot=None):
        """Save world + player into client_save/ (singleplayer only — never while online)."""

        if self.net_client.connected or self.is_host:
            self.say("World save is client-only — disconnect from multiplayer first")
            return False

        if self.in_menu:
            self.say("Start a game before saving")
            return False

        path = self._save_path(slot)

        try:
            self.say("Saving...", 1.5)

            player = self._player_blob()

            payload = {
                "version": SAVE_VERSION,
                "kind": "full",
                "player_json": json.dumps(player),
            }

            arrays = {}
            arrays["map_explored"] = self.map_explored.astype(np.uint8)

            main = self.scenes["main"]

            arrays["blocks"] = main.world.blocks
            arrays["biome_map"] = main.biome_map.astype(np.int16)

            payload["world_size"] = json.dumps({
                "W": main.world.W,
                "H": main.world.H,
                "D": main.world.D,
                "map_w": int(main.map_size[0]),
                "map_d": int(main.map_size[1]),
            })

            np.savez_compressed(
                path,
                **arrays,
                meta=np.array([json.dumps(payload)])
            )

            size_mb = os.path.getsize(path) / (1024 * 1024)

            self.say(f"Saved full world ({size_mb:.1f} MB)", 3.0)
            self.sfx.play("coin")
            return True

        except Exception as e:
            self.say(f"Save failed: {e}", 4.0)
            self.sfx.play("deny")
            return False

    def load_game(self, slot=None):
        """Load client_save/ world — never while connected to multiplayer."""
        if self.net_client.connected or self.is_host:
            self.say("World load is client-only — disconnect first")
            return False

        if self.in_menu:
            self.say("Press PLAY first, then load")
            return False

        path = self._save_path(slot)

        if not os.path.isfile(path):
            self.say("No save file found (F5 to save)")
            self.sfx.play("deny")
            return False

        try:
            self.say("Loading...", 1.5)

            data = np.load(path, allow_pickle=True)
            meta = json.loads(str(data["meta"][0]))
            player = json.loads(meta["player_json"])

            self.coins = float(player.get("coins", 0))
            self.pick_lvl = int(player.get("pick_lvl", 1))
            self.has_jet = bool(player.get("has_jet", False))
            self.tank_lvl = int(player.get("tank_lvl", 1))
            self.fuel = float(player.get("fuel", 100))
            self.has_launcher = bool(player.get("has_launcher", False))
            self.rocket_ammo = int(player.get("rocket_ammo", 0))
            self.blast_lvl = int(player.get("blast_lvl", 1))
            self.has_tnt = bool(player.get("has_tnt", False))
            self.tnt_ammo = int(player.get("tnt_ammo", 0))
            self.tnt_lvl = int(player.get("tnt_lvl", 1))
            self.tool = int(player.get("tool", 0))
            self.game_mode = player.get("game_mode", self.game_mode)

            if player.get("player_name"):
                self.player_name = player["player_name"]

            for k, v in (player.get("upg") or {}).items():
                self.upg[k] = v

            qs = player.get("quest_states") or {}

            for k, v in qs.items():
                try:
                    bi = int(k)
                    self.quest_states[bi] = dict(v)
                except Exception:
                    pass

            self.active_quest = player.get("active_quest")

            if "map_explored" in data.files:
                exp = data["map_explored"].astype(bool)
                if exp.shape == self.map_explored.shape:
                    self.map_explored[:] = exp

            main = self.scenes["main"]

            if "blocks" in data.files:
                blocks = data["blocks"]

                if blocks.shape == main.world.blocks.shape:
                    main.world.blocks[:] = blocks

            if "biome_map" in data.files:
                bm = data["biome_map"]

                if bm.shape == main.biome_map.shape:
                    main.biome_map[:] = bm

            main.world.touch(
                (0, 0, 0),
                (main.world.W, main.world.H, main.world.D)
            )

            if main.rend:
                main.rend.dirty = {
                    (x, y, z)
                    for x in range(main.world.nc[0])
                    for y in range(main.world.nc[1])
                    for z in range(main.world.nc[2])
                }

            self.map_base = self.make_world_map()
            self.spawn_biome_monsters()

            scene_name = player.get("scene", "lobby")

            if scene_name in self.scenes:

                if scene_name != self.scene.name:
                    self.switch_scene(scene_name, "default")

                pp = player.get("pos")

                if pp and len(pp) == 3:
                    self.pos = np.array(pp, float)
                    self.vel[:] = 0

                self.yaw = float(player.get("yaw", self.yaw))
                self.pitch = float(player.get("pitch", self.pitch))

            self.debris.clear()
            self.rockets = []
            self.col_box = None

            self.say("Game loaded", 2.5)
            self.sfx.play("portal")
            self.set_paused(False)

            return True

        except Exception as e:
            self.say(f"Load failed: {e}", 4.0)
            self.sfx.play("deny")
            return False

    def _progress_path(self, multiplayer=False):
        """Client paths only under client_save/. Server has its own server_save/ folder."""
        if multiplayer:
            # Local cache of MP progress (not the SP world/progress)
            safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in self.player_name)[:24]
            d = os.path.join(self._save_dir(), "mp_cache")
            os.makedirs(d, exist_ok=True)
            return os.path.join(d, f"{safe}.json")
        return os.path.join(self._save_dir(), f"{SP_PROGRESS_NAME}.json")

    def _in_multiplayer(self):
        return bool(self.net_client.connected or self.is_host)

    def save_progress(self, quiet=True):
        """Lightweight autosave: money, gear, upgrades, quests (no world blocks).
        Offline → saves/singleplayer_save.json
        Online  → push to server + local server_save/ cache (never touches singleplayer_save).
        """
        if self.in_menu:
            return False
        try:
            blob = self._player_blob()
            blob["version"] = SAVE_VERSION
            blob["saved_at"] = time.time()
            online = self._in_multiplayer()
            path = self._progress_path(multiplayer=online)
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(blob, f, separators=(",", ":"))
            os.replace(tmp, path)
            self.autosave_dirty = False
            # Push to dedicated server / host so server.py keeps a copy under server_save/
            try:
                if self.net_client.connected and self.net_client.sock:
                    msg = dict(blob)
                    msg["t"] = "progress"
                    _net_send(self.net_client.sock, msg)
            except Exception:
                pass
            if not quiet:
                self.say("Progress saved", 1.5)
            return True
        except Exception as e:
            if not quiet:
                self.say(f"Progress save failed: {e}", 3.0)
            return False

    def load_progress(self, quiet=False):
        """Restore offline progress only (singleplayer_save.json).
        Multiplayer progress comes from the server welcome payload, never this file.
        """
        if self._in_multiplayer():
            if not quiet:
                self.say("Using server progress (isolated from singleplayer_save)")
            return False
        path = self._progress_path(multiplayer=False)
        if not os.path.isfile(path):
            if not quiet:
                self.say("No singleplayer_save yet")
            return False
        try:
            with open(path, "r", encoding="utf-8") as f:
                player = json.load(f)
            self._apply_progress_dict(player)
            if not quiet:
                self.say(f"Progress loaded · {int(self.coins)} DCOIN", 2.5)
            return True
        except Exception as e:
            if not quiet:
                self.say(f"Progress load failed: {e}", 3.0)
            return False

    def mark_progress_dirty(self):
        self.autosave_dirty = True

    def _apply_progress_dict(self, player):
        """Apply coins/gear from local file or server progress payload."""
        if not isinstance(player, dict):
            return
        if "coins" in player:
            self.coins = float(player.get("coins", self.coins))
        if "pick_lvl" in player:
            self.pick_lvl = int(player.get("pick_lvl", self.pick_lvl))
        if "has_jet" in player:
            self.has_jet = bool(player.get("has_jet", self.has_jet))
        if "tank_lvl" in player:
            self.tank_lvl = int(player.get("tank_lvl", self.tank_lvl))
        if "fuel" in player:
            self.fuel = float(player.get("fuel", self.fuel))
        if "has_launcher" in player:
            self.has_launcher = bool(player.get("has_launcher", self.has_launcher))
        if "rocket_ammo" in player:
            self.rocket_ammo = int(player.get("rocket_ammo", self.rocket_ammo))
        if "blast_lvl" in player:
            self.blast_lvl = int(player.get("blast_lvl", self.blast_lvl))
        if "has_tnt" in player:
            self.has_tnt = bool(player.get("has_tnt", self.has_tnt))
        if "tnt_ammo" in player:
            self.tnt_ammo = int(player.get("tnt_ammo", self.tnt_ammo))
        if "tnt_lvl" in player:
            self.tnt_lvl = int(player.get("tnt_lvl", self.tnt_lvl))
        if "tool" in player:
            self.tool = int(player.get("tool", self.tool))
        for k, v in (player.get("upg") or {}).items():
            self.upg[k] = v
        qs = player.get("quest_states") or {}
        for k, v in qs.items():
            try:
                self.quest_states[int(k)] = dict(v)
            except Exception:
                pass
        if player.get("active_quest") is not None:
            self.active_quest = player.get("active_quest")
        self.autosave_dirty = False


    # ----------------------------------------------------------------- scenes
    def teleport(self, target, key):
        if self.fade_dir != 0 or self.pending:
            return
        self.pending = (target, key)
        self.fade_dir = 1
        self.sfx.play("portal")

    def switch_scene(self, name, key):
        self.scene = self.scenes[name]
        self.open_map(False)
        (px, py, pz), yaw = self.scene.spawns[key]
        self.pos = np.array([px, py, pz], float)
        self.vel[:] = 0
        self.yaw, self.pitch, self.step_off = yaw, -0.1, 0.0
        self.debris.clear()
        self.rockets = []
        self.col_box = None
        self.mining_cell, self.progress = None, 0.0
        self.set_shop(False)
        self.scene.rend.update(self.eye(), 0.008, 70.0)
        pygame.mouse.get_rel()

    # ------------------------------------------------------------------- shop
    def shop_catalog(self):
        """Categories shown in the left column. Each has a list of upgrades."""
        return [
            {
                "id": "pickaxe", "name": "Pickaxe", "tag": "TOOL",
                "color": (255, 200, 90),
                "blurb": "Your main dig tool. Upgrade power, fortune and reach.",
                "upgrades": [
                    {"key": "pick_lvl", "name": "Pick Power", "max": 6, "base": 40, "scale": 1.9,
                     "desc": "Mine faster, wider dig radius, better base payout."},
                    {"key": "pick_speed", "name": "Swift Strikes", "max": 5, "base": 55, "scale": 1.7,
                     "desc": "Reduces time per swing. Stacks with power."},
                    {"key": "pick_fortune", "name": "Fortune", "max": 5, "base": 70, "scale": 1.75,
                     "desc": "Bonus DCOIN from every block destroyed."},
                    {"key": "pick_reach", "name": "Extended Reach", "max": 4, "base": 90, "scale": 1.8,
                     "desc": "Larger carve sphere while mining."},
                    {"key": "pick_vein", "name": "Vein Finder", "max": 3, "base": 120, "scale": 1.9,
                     "desc": "Chance to auto-break neighbouring ore."},
                    {"key": "drill_mode", "name": "Drill Mode", "max": 3, "base": 160, "scale": 1.85,
                     "desc": "Hold LMB longer for a continuous drill beam."},
                    {"key": "chain_mine", "name": "Chain Reaction", "max": 3, "base": 200, "scale": 1.9,
                     "desc": "Breaking ore can chain to nearby ore clusters."},
                    {"key": "melee_sweep", "name": "Sweeping Strike", "max": 3, "base": 110, "scale": 1.7,
                     "desc": "Pickaxe hits multiple monsters in a cone."},
                ],
            },
            {
                "id": "jetpack", "name": "Jetpack", "tag": "MOBILITY",
                "color": (255, 140, 60),
                "blurb": "Aerial freedom. Unlock, expand the tank, push harder.",
                "upgrades": [
                    {"key": "has_jet", "name": "Jetpack Core", "max": 1, "base": 150, "scale": 1.0,
                     "desc": "Hold SPACE in the air to fly."},
                    {"key": "tank_lvl", "name": "Fuel Tank", "max": 5, "base": 80, "scale": 1.6,
                     "desc": "Larger fuel capacity. Requires Jetpack Core."},
                    {"key": "fuel_cap", "name": "Aux Tanks", "max": 4, "base": 100, "scale": 1.65,
                     "desc": "Extra reserve fuel on top of the main tank."},
                    {"key": "jet_thrust", "name": "Afterburner", "max": 4, "base": 110, "scale": 1.7,
                     "desc": "Stronger upward thrust while jetting."},
                    {"key": "jet_efficiency", "name": "Eco Burner", "max": 4, "base": 95, "scale": 1.65,
                     "desc": "Uses less fuel per second of flight."},
                    {"key": "regen_boost", "name": "Rapid Refuel", "max": 4, "base": 85, "scale": 1.6,
                     "desc": "Faster fuel regen while grounded."},
                    {"key": "jet_hover", "name": "Hover Stabiliser", "max": 3, "base": 140, "scale": 1.75,
                     "desc": "Hold crouch mid-air to hover using little fuel."},
                    {"key": "jet_dash", "name": "Air Dash", "max": 3, "base": 180, "scale": 1.8,
                     "desc": "Double-tap a direction mid-air for a short dash."},
                ],
            },
            {
                "id": "launcher", "name": "Rocket Launcher", "tag": "EXPLOSIVE",
                "color": (255, 90, 70),
                "blurb": "Heavy demolition. Ammo, blast size and payload upgrades.",
                "upgrades": [
                    {"key": "has_launcher", "name": "Launcher Frame", "max": 1, "base": 250, "scale": 1.0,
                     "desc": "Unlock the rocket launcher (press 2). Includes 5 rockets."},
                    {"key": "rocket_ammo_pack", "name": "Rocket Pack x5", "max": 99, "base": 40, "scale": 1.0,
                     "desc": "Buy 5 rockets. Capacity grows with Ammo Bay."},
                    {"key": "blast_lvl", "name": "Blast Charge", "max": 5, "base": 200, "scale": 1.8,
                     "desc": "Larger explosion radius and more rubble."},
                    {"key": "ammo_cap", "name": "Ammo Bay", "max": 4, "base": 120, "scale": 1.7,
                     "desc": "Raise max carried rockets (+10 per level)."},
                    {"key": "rocket_speed", "name": "Hyper Propellant", "max": 4, "base": 130, "scale": 1.65,
                     "desc": "Rockets fly faster and travel farther."},
                    {"key": "rocket_multi", "name": "Double Tap", "max": 3, "base": 220, "scale": 1.85,
                     "desc": "Chance to fire a second rocket for free."},
                    {"key": "rocket_cluster", "name": "Cluster Warhead", "max": 3, "base": 260, "scale": 1.9,
                     "desc": "Explosions spawn secondary blasts."},
                    {"key": "quake", "name": "Seismic Rounds", "max": 3, "base": 300, "scale": 1.95,
                     "desc": "Explosions trigger wider structural collapse."},
                ],
            },
            {
                "id": "tnt", "name": "TNT Packs", "tag": "EXPLOSIVE",
                "color": (220, 60, 50),
                "blurb": "Placeable delayed charges. Great for controlled demolitions.",
                "upgrades": [
                    {"key": "has_tnt", "name": "TNT Satchel", "max": 1, "base": 180, "scale": 1.0,
                     "desc": "Unlock TNT (press 3). Place with LMB. Includes 3 sticks."},
                    {"key": "tnt_ammo_pack", "name": "TNT Bundle x3", "max": 99, "base": 55, "scale": 1.0,
                     "desc": "Buy 3 TNT sticks. Capacity grows with TNT Pack."},
                    {"key": "tnt_lvl", "name": "High Explosive", "max": 5, "base": 140, "scale": 1.75,
                     "desc": "Bigger base blast radius for placed TNT."},
                    {"key": "tnt_blast", "name": "Blast Boost", "max": 4, "base": 120, "scale": 1.7,
                     "desc": "Extra radius on top of High Explosive."},
                    {"key": "tnt_fuse", "name": "Short Fuse", "max": 4, "base": 100, "scale": 1.65,
                     "desc": "TNT detonates faster after placement."},
                    {"key": "tnt_pack", "name": "Demolition Pack", "max": 4, "base": 110, "scale": 1.7,
                     "desc": "Carry more TNT (+4 per level)."},
                    {"key": "tnt_chain", "name": "Sympathetic Det", "max": 3, "base": 200, "scale": 1.85,
                     "desc": "Chance for nearby TNT to chain-react."},
                ],
            },
            {
                "id": "boots", "name": "Movement Gear", "tag": "MOBILITY",
                "color": (90, 200, 255),
                "blurb": "Run faster, jump higher, stay agile in the air.",
                "upgrades": [
                    {"key": "move_speed", "name": "Trail Runners", "max": 5, "base": 60, "scale": 1.55,
                     "desc": "Base walk and run speed increase."},
                    {"key": "sprint_boost", "name": "Sprint Coil", "max": 4, "base": 75, "scale": 1.6,
                     "desc": "Extra speed while holding SHIFT."},
                    {"key": "jump_boost", "name": "Spring Soles", "max": 4, "base": 70, "scale": 1.55,
                     "desc": "Higher jump velocity."},
                    {"key": "double_jump", "name": "Double Jump", "max": 2, "base": 160, "scale": 1.9,
                     "desc": "Extra mid-air jumps (1 then 2)."},
                    {"key": "air_control", "name": "Air Fins", "max": 4, "base": 80, "scale": 1.55,
                     "desc": "Much better steering while airborne."},
                    {"key": "slide", "name": "Ground Slide", "max": 2, "base": 100, "scale": 1.7,
                     "desc": "Crouch while sprinting to slide farther."},
                    {"key": "pad_boost", "name": "Pad Amplifier", "max": 3, "base": 90, "scale": 1.6,
                     "desc": "Jump pads launch you even higher."},
                ],
            },
            {
                "id": "fortune", "name": "Fortune & Loot", "tag": "ECONOMY",
                "color": (255, 215, 70),
                "blurb": "Earn more from everything you break and find.",
                "upgrades": [
                    {"key": "loot_bonus", "name": "Greed Module", "max": 5, "base": 90, "scale": 1.7,
                     "desc": "Global DCOIN multiplier from mining and crates."},
                    {"key": "crate_luck", "name": "Crate Crackers", "max": 4, "base": 85, "scale": 1.65,
                     "desc": "Loot crates drop significantly more coins."},
                    {"key": "treasure_hunter", "name": "Treasure Hunter", "max": 3, "base": 140, "scale": 1.8,
                     "desc": "Ruins and gold tips pay out bonus caches."},
                    {"key": "coin_interest", "name": "Vault Interest", "max": 3, "base": 200, "scale": 1.9,
                     "desc": "Passive DCOIN every 30s based on current balance."},
                    {"key": "quest_bonus", "name": "Bounty Board", "max": 4, "base": 110, "scale": 1.7,
                     "desc": "Higher rewards for finishing biome quests."},
                    {"key": "auto_smelt", "name": "Auto Smelt", "max": 3, "base": 150, "scale": 1.75,
                     "desc": "Ore blocks are worth more when mined."},
                    {"key": "magnet", "name": "Coin Magnet", "max": 3, "base": 100, "scale": 1.6,
                     "desc": "Debris value is partially auto-collected."},
                ],
            },
            {
                "id": "combat", "name": "Combat Suite", "tag": "COMBAT",
                "color": (255, 100, 120),
                "blurb": "Hit harder, reach farther, survive longer against biome beasts.",
                "upgrades": [
                    {"key": "monster_dmg", "name": "Heavy Hands", "max": 5, "base": 70, "scale": 1.6,
                     "desc": "More damage per pickaxe hit on monsters."},
                    {"key": "monster_range", "name": "Long Arm", "max": 4, "base": 80, "scale": 1.55,
                     "desc": "Attack monsters from farther away."},
                    {"key": "lifesteal", "name": "Leech Strike", "max": 3, "base": 160, "scale": 1.8,
                     "desc": "Killing monsters restores a bit of jet fuel."},
                    {"key": "thorns", "name": "Thorn Plating", "max": 3, "base": 130, "scale": 1.7,
                     "desc": "Reduce explosion knockback taken."},
                    {"key": "shake_reduce", "name": "Stabiliser Gyro", "max": 3, "base": 90, "scale": 1.55,
                     "desc": "Less camera shake from blasts and storms."},
                ],
            },
            {
                "id": "scout", "name": "Scout Systems", "tag": "UTILITY",
                "color": (120, 220, 180),
                "blurb": "See more of the world and find secrets faster.",
                "upgrades": [
                    {"key": "map_range", "name": "Wide Scanner", "max": 5, "base": 60, "scale": 1.5,
                     "desc": "Reveal a larger radius on the atlas as you walk."},
                    {"key": "reveal_boost", "name": "Cartographer", "max": 3, "base": 100, "scale": 1.65,
                     "desc": "One-time boost: reveal a chunk of unexplored map."},
                    {"key": "ore_sense", "name": "Ore Sense", "max": 4, "base": 110, "scale": 1.7,
                     "desc": "Highlight nearby ore while mining (HUD ping)."},
                    {"key": "ruin_radar", "name": "Ruin Radar", "max": 3, "base": 120, "scale": 1.7,
                     "desc": "Marks nearby ruins on the compass message."},
                    {"key": "night_eye", "name": "Night Eye", "max": 3, "base": 95, "scale": 1.6,
                     "desc": "Reduced fog density in storms and fog weather."},
                    {"key": "storm_resist", "name": "Storm Cloak", "max": 3, "base": 130, "scale": 1.7,
                     "desc": "Less shake and better visibility in storms."},
                    {"key": "portal_cd", "name": "Portal Tuner", "max": 2, "base": 180, "scale": 1.8,
                     "desc": "Faster scene transitions and lobby return."},
                    {"key": "starter_pack", "name": "Supply Drop", "max": 1, "base": 300, "scale": 1.0,
                     "desc": "Instant: +10 rockets, fill fuel, +200 DCOIN."},
                ],
            },
        ]

    def _upg_level(self, key):
        if key == "pick_lvl":
            return self.pick_lvl
        if key == "tank_lvl":
            return self.tank_lvl
        if key == "blast_lvl":
            return self.blast_lvl
        if key == "tnt_lvl":
            return self.tnt_lvl
        if key == "has_jet":
            return 1 if self.has_jet else 0
        if key == "has_launcher":
            return 1 if self.has_launcher else 0
        if key == "has_tnt":
            return 1 if self.has_tnt else 0
        if key == "rocket_ammo_pack":
            return self.rocket_ammo  # display only; buy always adds
        if key == "tnt_ammo_pack":
            return self.tnt_ammo
        return int(self.upg.get(key, 0))

    def _upg_cost(self, u):
        lvl = self._upg_level(u["key"])
        if u["key"] in ("rocket_ammo_pack", "tnt_ammo_pack"):
            return int(u["base"])
        if lvl >= u["max"]:
            return None
        return int(u["base"] * (u["scale"] ** max(0, lvl if u["key"] not in ("pick_lvl", "tank_lvl", "blast_lvl", "tnt_lvl") else lvl - 1)))

    def _upg_state(self, item, u):
        key = u["key"]
        # prerequisites
        if item["id"] == "jetpack" and key != "has_jet" and not self.has_jet:
            return "locked", None
        if item["id"] == "launcher" and key != "has_launcher" and not self.has_launcher:
            return "locked", None
        if item["id"] == "tnt" and key != "has_tnt" and not self.has_tnt:
            return "locked", None
        if key == "rocket_ammo_pack":
            if not self.has_launcher:
                return "locked", None
            if self.rocket_ammo >= self.max_ammo:
                return "max", None
            return "buy", self._upg_cost(u)
        if key == "tnt_ammo_pack":
            if not self.has_tnt:
                return "locked", None
            if self.tnt_ammo >= self.max_tnt:
                return "max", None
            return "buy", self._upg_cost(u)
        lvl = self._upg_level(key)
        if lvl >= u["max"]:
            return "max", None
        return "buy", self._upg_cost(u)

    def buy_upgrade(self, item_idx, upg_idx):
        catalog = self.shop_catalog()
        if not (0 <= item_idx < len(catalog)):
            return
        item = catalog[item_idx]
        if not (0 <= upg_idx < len(item["upgrades"])):
            return
        u = item["upgrades"][upg_idx]
        state, cost = self._upg_state(item, u)
        if state != "buy" or cost is None:
            self.sfx.play("deny")
            return
        free = self.game_mode == "creative"
        if not free and self.coins < cost:
            self.say("Not enough DCOIN")
            self.sfx.play("deny")
            return
        if not free:
            self.coins -= cost
        key = u["key"]
        if key == "pick_lvl":
            self.pick_lvl = min(6, self.pick_lvl + 1)
        elif key == "has_jet":
            self.has_jet, self.fuel = True, self.max_fuel
        elif key == "tank_lvl":
            self.tank_lvl = min(5, self.tank_lvl + 1)
            self.fuel = self.max_fuel
        elif key == "has_launcher":
            self.has_launcher, self.tool = True, 1
            self.rocket_ammo = min(self.max_ammo, self.rocket_ammo + 5)
        elif key == "rocket_ammo_pack":
            self.rocket_ammo = min(self.max_ammo, self.rocket_ammo + 5)
        elif key == "blast_lvl":
            self.blast_lvl = min(5, self.blast_lvl + 1)
        elif key == "has_tnt":
            self.has_tnt, self.tool = True, 2
            self.tnt_ammo = min(self.max_tnt, self.tnt_ammo + 3)
        elif key == "tnt_ammo_pack":
            self.tnt_ammo = min(self.max_tnt, self.tnt_ammo + 3)
        elif key == "tnt_lvl":
            self.tnt_lvl = min(5, self.tnt_lvl + 1)
        elif key == "starter_pack":
            self.upg["starter_pack"] = 1
            self.rocket_ammo = min(self.max_ammo, self.rocket_ammo + 10)
            if self.has_jet:
                self.fuel = self.max_fuel
            self.coins += 200
        elif key == "reveal_boost":
            self.upg["reveal_boost"] = min(u["max"], self.upg.get("reveal_boost", 0) + 1)
            # reveal a large area around the player if in main
            if self.scene.name == "main":
                old_r = 18
                # temporary big reveal
                x = int(self.pos[0] / VS)
                z = int(self.pos[2] / VS)
                mw, md = self.scene.map_size
                rad = 80 + 40 * self.upg["reveal_boost"]
                x0, x1 = max(0, x - rad), min(mw, x + rad + 1)
                z0, z1 = max(0, z - rad), min(md, z + rad + 1)
                xx = np.arange(x0, x1)[:, None]
                zz = np.arange(z0, z1)[None, :]
                self.map_explored[x0:x1, z0:z1] |= ((xx - x) ** 2 + (zz - z) ** 2 <= rad * rad)
        elif key == "double_jump":
            self.upg["double_jump"] = min(u["max"], self.upg.get("double_jump", 0) + 1)
            self.extra_jumps = self.upg["double_jump"]
        else:
            self.upg[key] = min(u["max"], self.upg.get(key, 0) + 1)
        self.sfx.play("coin")
        self.say(("Granted: " if free else "Purchased: ") + u["name"], 2.0)
        self.mark_progress_dirty()
        self.save_progress(quiet=True)

    def near_terminal(self):
        t = self.scene.terminal
        return t is not None and math.hypot(self.pos[0] - t[0], self.pos[2] - t[2]) < 5.0

    def near_server_pc(self):
        pc = getattr(self.scene, "server_pc", None)
        if not pc:
            return False
        return math.hypot(self.pos[0] - pc[0], self.pos[2] - pc[2]) < 4.0

    def open_mp(self, opened):
        self.mp_open = opened
        if opened:
            self.mp_servers = default_server_list()
            # show local host entry status
            if self.is_host and self.net_server and self.net_server.running:
                self.mp_status = f"Hosting on 0.0.0.0:{OFFICIAL_PORT}"
            elif self.net_client.connected:
                self.mp_status = f"Connected: {self.net_client.server_name}"
            else:
                self.mp_status = "Select a server or host your own"
        locked = opened or self.shop_open or self.paused or self.map_open or self.in_menu
        pygame.mouse.set_visible(locked)
        pygame.event.set_grab(not locked)
        pygame.mouse.get_rel()

    def host_server(self):
        if self.is_host and self.net_server and self.net_server.running:
            self.say("Already hosting")
            return
        # Save offline world BEFORE leaving singleplayer memory
        try:
            if not self.net_client.connected and not self.is_host and not self.in_menu:
                self.save_game(SAVE_SLOT)
                self.save_progress(quiet=True)
        except Exception:
            pass
        self.net_client.close()
        self.mp_seen, self.mp_last_eid = set(), 0
        self.net_server = NetServer(LOCAL_SERVER_PORT, f"{self.player_name}'s Server")
        if not self.net_server.start(self.player_id, self.player_name):
            self.mp_status = f"Failed to bind port {LOCAL_SERVER_PORT} (is server.py already running?)"
            self.say(self.mp_status)
            self.sfx.play("deny")
            self.net_server = None
            return
        self.is_host = True
        # CRITICAL: drop SP voxels, load only server_world history
        self.enter_multiplayer_world()
        self.mp_status = f"Hosting on :{LOCAL_SERVER_PORT} — public {OFFICIAL_HOST}:{OFFICIAL_PORT}"
        self.say("Server online — multiplayer world isolated", 3.5)
        self.sfx.play("coin")

    def stop_multiplayer(self):
        # Flush multiplayer progress to the server only (not client world)
        self.save_progress(quiet=True)
        if self.net_server:
            self.net_server.stop()
            self.net_server = None
        self.net_client.close()
        self.is_host = False
        self.mp_remote = {}
        self.mp_status = "Disconnected"
        self.is_op = False
        if self.spectator:
            self.set_spectator(False)
        # CRITICAL: throw away MP voxels and restore client_save world only
        self.enter_singleplayer_world()
        self.load_progress(quiet=True)
        self.say("Left multiplayer — client_save world restored", 3.0)

    def join_server(self, host, port):
        """Start a non-blocking join. Network runs on a worker; main thread stays responsive."""
        if self.is_host:
            self.say("Stop hosting first")
            self.sfx.play("deny")
            return
        if getattr(self, "_join_thread", None) and self._join_thread.is_alive():
            self.say("Already connecting...")
            return
        # Lightweight offline snapshot only (no full world rebuild / heavy npz on main thread)
        try:
            if not self.net_client.connected and not self.is_host and not self.in_menu:
                self.save_progress(quiet=True)
        except Exception:
            pass
        self.mp_status = f"Connecting to {host}:{port}..."
        self.say(f"Connecting to {host}:{port}...", 3.0)
        # Always request a full history replay on explicit Join. Resume only helps
        # after a brief drop while the process stays up *and* we know the watermark
        # is gap-free; a manual Disconnect/Join almost always means "I'm desynced."
        self.mp_seen = set()
        self.mp_last_eid = 0
        self._join_force_rebuild = True
        self._join_result = None  # None=pending, tuple(ok, error_or_none)

        def worker():
            ok = self.net_client.connect(
                host, port, self.player_name, self.player_id,
                epoch=None, since=0)
            self._join_result = (ok, self.net_client.last_error if not ok else None)

        self._join_thread = threading.Thread(target=worker, daemon=True)
        self._join_thread.start()

    def _finish_join(self):
        """Called from update() when the join worker finishes."""
        result = getattr(self, "_join_result", None)
        if result is None:
            return
        self._join_result = None
        ok, err = result
        if ok:
            # Reset gear — server progress is the only MP source
            self.coins = 0.0
            self.pick_lvl = 1
            self.has_jet = False
            self.tank_lvl = 1
            self.fuel = 100.0
            self.has_launcher = False
            self.rocket_ammo = 0
            self.blast_lvl = 1
            self.has_tnt = False
            self.tnt_ammo = 0
            self.tnt_lvl = 1
            self.tool = 0
            for k in list(self.upg.keys()):
                self.upg[k] = 0
            ep = getattr(self.net_client, "epoch", None)
            resumed = bool(getattr(self.net_client, "resumed", False))
            force = bool(getattr(self, "_join_force_rebuild", True))
            self._join_force_rebuild = False
            # Explicit Join always rebuilds. Live-resume only when the server says so
            # *and* we did not ask for a forced rebuild (e.g. future soft-reconnect).
            if force or (not resumed) or self.world_epoch != ep:
                self.mp_seen, self.mp_last_eid = set(), 0
                self.enter_multiplayer_world(epoch=ep)
                tag = "full rebuild"
            else:
                self.world_epoch = ep
                tag = "resumed"
                self.say("Resumed multiplayer world", 2.0)
            pending = self.net_client.poll()
            total = max(1, len(pending))
            for i, m in enumerate(pending):
                self._apply_mp_event(m, silent=True)
                if i % 25 == 0:
                    self.loading(f"Syncing world events {i + 1}/{total}", min(0.99, i / total))
            # Drain any late packets that arrived during the rebuild
            for m in self.net_client.poll():
                self._apply_mp_event(m, silent=True)
            self._advance_watermark()
            self.save_progress(quiet=True)
            wel = getattr(self.net_client, "welcome", None) or {}
            self.is_op = bool(wel.get("op"))
            if self.is_op:
                self.say("You have OP — F2 spectator", 3.0)
            self.mp_status = f"Joined {self.net_client.server_name} ({tag}, eid={self.mp_last_eid})"
            self.say(f"Joined {self.net_client.server_name} · {int(self.coins)} DCOIN · synced to {self.mp_last_eid}", 3.0)
            self.sfx.play("portal")
            self.open_mp(False)
        else:
            # Failed join: do NOT rebuild the world (that was freezing the game)
            self.mp_status = f"Join failed: {err or 'unknown error'}"
            self.say(f"Could not connect: {err or 'unknown'}", 3.5)
            self.sfx.play("deny")

    def mp_state_payload(self):
        return {
            "id": self.player_id,
            "name": self.player_name,
            "x": float(self.pos[0]), "y": float(self.pos[1]), "z": float(self.pos[2]),
            "yaw": float(self.yaw), "pitch": float(self.pitch),
            "scene": self.scene.name, "tool": int(self.tool),
        }

    def update_multiplayer(self, dt):
        # Complete async join without blocking the frame that clicked Join
        if getattr(self, "_join_result", None) is not None:
            self._finish_join()
        self.mp_send_t += dt
        self.mp_resync_t += dt
        if self.is_host and self.net_server and self.net_server.running:
            if self.mp_send_t >= NET_TICK:
                self.mp_send_t = 0.0
                self.net_server.update_host_state(self.mp_state_payload())
            self.mp_remote = {
                p["id"]: p for p in self.net_server.player_list()
                if p["id"] != self.player_id
            }
            for m in self.net_server.drain_events():
                self._apply_mp_event(m)
        elif self.net_client.connected:
            if self.mp_send_t >= NET_TICK:
                self.mp_send_t = 0.0
                self.net_client.send_state(self.mp_state_payload())
            if self.mp_resync_t >= NET_RESYNC:
                self.mp_resync_t = 0.0
                # only ask for events newer than what we already have
                self.net_client.request_resync(self.mp_last_eid)
            for m in self.net_client.poll():
                self._apply_mp_event(m)
            if not self.net_client.connected:
                self.mp_status = "Disconnected from server"
                self.mp_remote = {}
        else:
            self.mp_remote = {}

    def _mp_seen(self, m):
        """True if this world event was already applied (events carry a unique eid).

        Marks the eid as seen and advances the *contiguous* watermark only when
        every eid from the previous watermark up through this one is present.
        That way a dropped packet leaves a gap that the next resync still asks for.
        """
        eid = m.get("eid")
        if eid is None:
            return False
        try:
            eid = int(eid)
        except Exception:
            return False
        if eid in self.mp_seen:
            return True
        self.mp_seen.add(eid)
        self._advance_watermark()
        if len(self.mp_seen) > 8000:
            floor = max(0, self.mp_last_eid - 4000)
            self.mp_seen = {e for e in self.mp_seen if e > floor}
        return False

    def _advance_watermark(self):
        """Bump mp_last_eid only while the next eid is already in mp_seen (no gaps)."""
        # Allow starting from 0: eid 1 is the first stamped event.
        while (self.mp_last_eid + 1) in self.mp_seen:
            self.mp_last_eid += 1

    def _force_full_resync(self):
        """Throw away the watermark so the next resync (or rejoin) gets the whole log."""
        self.mp_seen = set()
        self.mp_last_eid = 0
        if self.net_client.connected:
            self.net_client.request_resync(0)

    def _local_progress_time(self):
        # Compare against the active isolated profile only
        try:
            with open(self._progress_path(multiplayer=self._in_multiplayer()), "r", encoding="utf-8") as f:
                return float(json.load(f).get("saved_at", 0) or 0)
        except Exception:
            return 0.0

    def _apply_mp_event(self, m, silent=False):
        t = m.get("t")
        if t == "state" and m.get("player"):
            p = m["player"]
            if p.get("id") and p["id"] != self.player_id:
                self.mp_remote[p["id"]] = p
        elif t == "join" and m.get("player"):
            p = m["player"]
            if p.get("id") != self.player_id:
                self.mp_remote[p["id"]] = p
                self.say(f"{p.get('name', 'Player')} joined", 2.0)
        elif t == "leave":
            self.mp_remote.pop(m.get("id"), None)
        elif t == "world_reset":
            self.reset_world(m.get("epoch"))
        elif t == "ack":
            # Server stamped our outbound dig/boom/collapse — record the eid so
            # a later resync of the same event is treated as already applied.
            try:
                eid = int(m.get("eid"))
            except Exception:
                return
            self.mp_seen.add(eid)
            self._advance_watermark()
        elif t == "sync_end":
            self._advance_watermark()
            try:
                head = int(m.get("head") or 0)
            except Exception:
                head = 0
            if m.get("epoch") is not None:
                self.world_epoch = m.get("epoch")
            # If the server head is still ahead of our contiguous watermark, ask again
            # immediately (fills gaps that a single sync chunk left behind).
            if head > self.mp_last_eid and self.net_client.connected:
                self.net_client.request_resync(self.mp_last_eid)
        elif t in ("dig", "boom", "collapse"):
            # Never carve the offline world with multiplayer packets
            if not self._in_multiplayer():
                return
            if self._mp_seen(m):
                return            # already applied once - never replay
            if m.get("id") == self.player_id:
                return            # our own event echoed back by a resync
            # World edits always go to scenes["main"] even if we are in lobby/shop
            if t == "dig":
                self._apply_remote_dig(m, silent)
            elif t == "boom":
                self._apply_remote_boom(m, silent)
            else:
                self._apply_remote_collapse(m, silent)
        elif t == "rocket":
            if not silent:
                self._apply_remote_rocket(m)
        elif t == "progress":
            data = m.get("data") or m
            if isinstance(data, dict) and data:
                try:
                    srv_ts = float(data.get("saved_at") or 0)
                except Exception:
                    srv_ts = 0.0
                # only let the server overwrite us if its copy is actually newer
                if srv_ts > self._local_progress_time() or self.coins == 0:
                    self._apply_progress_dict(data)
                    if not silent:
                        self.say(f"Server progress loaded · {int(self.coins)} DCOIN", 2.5)
        elif t == "sync":
            # catch-up batch: applied silently (no sound/knockback/shake) and de-duped
            for ev in m.get("events") or []:
                if isinstance(ev, dict) and ev.get("t") in ("dig", "boom", "collapse"):
                    self._apply_mp_event(ev, silent=True)
        elif t == "fp_ok":
            pass
        elif t == "fp_bad":
            self.say("World fingerprint mismatch — consider rejoining after a map reset", 4.0)
        elif t == "op_status":
            self.is_op = bool(m.get("op"))
            if not self.is_op and self.spectator:
                self.set_spectator(False)
            self.say("You are now OP" if self.is_op else "OP removed", 3.0)
        elif t == "chat":
            name = str(m.get("name") or "?")[:16]
            msg = str(m.get("msg") or "")[:80]
            self._push_chat(name, msg)

    def _apply_remote_dig(self, m, silent=False):
        # World edits always go to the MAIN world, even if we are standing in the
        # lobby/shop right now (old build ignored them => worlds drifted apart).
        w = self.scenes["main"].world
        try:
            c = np.array([float(m["x"]), float(m["y"]), float(m["z"])], float)
            r = float(m.get("r", 1.6))
        except Exception:
            return
        rad = r if r < 12 else r / VS
        # same seed as the digger => identical crater on every client
        w.carve_sphere(c, rad, rough=0.7, seed=m.get("seed"))
        # NOTE: no note_edit() here. The digger runs the collapse and sends it to us;
        # running our own too caused collapse ping-pong between clients.

    def _apply_remote_rocket(self, m):
        if self.scene.name != "main":
            return
        try:
            pos = np.array([float(m["x"]), float(m["y"]), float(m["z"])], float)
            vel = np.array([float(m["vx"]), float(m["vy"]), float(m["vz"])], float)
            life = float(m.get("life", 5.0))
        except Exception:
            return
        self.rockets.append({"pos": pos, "vel": vel, "life": life, "remote": True})
        if len(self.rockets) > 48:
            self.rockets = self.rockets[-48:]
        try:
            self.sfx.play("launch")
        except Exception:
            pass

    def _apply_remote_boom(self, m, silent=False):
        try:
            pos = np.array([float(m["x"]), float(m["y"]), float(m["z"])], float)
            r = float(m.get("r", 3.5))
        except Exception:
            return
        self.explode(pos, from_remote=True, radius=r, seed=m.get("seed"), silent=silent)

    def _apply_remote_collapse(self, m, silent=False):
        """Apply the exact structural drop the other player simulated."""
        w = self.scenes["main"].world
        try:
            lo = [int(m["x0"]), int(m["y0"]), int(m["z0"])]
            hi = [int(m["x1"]), int(m["y1"]), int(m["z1"])]
            margin = int(m.get("m", 24))
        except Exception:
            return
        visible = (self.scene.name == "main") and not silent
        cells = m.get("cells")
        truncated = True
        if cells:
            try:
                arr = np.asarray(cells, dtype=np.int32)
                if arr.ndim == 2 and arr.shape[1] == 3 and len(arr):
                    arr[:, 0] = np.clip(arr[:, 0], 0, w.W - 1)
                    arr[:, 1] = np.clip(arr[:, 1], 0, w.H - 1)
                    arr[:, 2] = np.clip(arr[:, 2], 0, w.D - 1)
                    types = w.blocks[arr[:, 0], arr[:, 1], arr[:, 2]].astype(np.int64)
                    mask = types != 0
                    arr, types = arr[mask], types[mask]
                    truncated = len(cells) >= COLLAPSE_MAX_CELLS
                    if len(arr):
                        w.blocks[arr[:, 0], arr[:, 1], arr[:, 2]] = 0
                        w.touch(arr.min(0), arr.max(0) + 1)
                        if visible:
                            self._collapse_effects(arr, types)
            except Exception:
                truncated = True
        # If the sender had to subsample its cell list, finish the job locally
        # (without re-broadcasting, so no feedback loop).
        if truncated:
            self._do_collapse(lo, hi, margin, broadcast=False, visible=visible)

    def _rebuild_main_world(self, status="Rebuilding world"):
        """Throw away the in-memory main map and generate a pristine one.
        This is the ONLY way SP and MP worlds stay isolated in RAM."""
        if self.scene.name == "main":
            self.switch_scene("lobby", "default")
        old = self.scenes.get("main")
        if old is not None:
            try:
                for vbo, vao, _n in old.rend.meshes.values():
                    vao.release()
                    vbo.release()
                old.rend.meshes.clear()
            except Exception:
                pass
        self.loading(status, 0.05)
        fresh = build_main(lambda t, f: self.loading(t, 0.05 + 0.9 * f))
        fresh.rend = ChunkRenderer(self.ctx, self.cprog, fresh.world)
        self.scenes["main"] = fresh
        mw, md = fresh.map_size
        self.map_explored = np.zeros((mw, md), dtype=bool)
        self.map_base = self.make_world_map()
        self.map_selected = None
        self.monsters = []
        self.spawn_biome_monsters()
        self.quest_states = {i: {"kills": 0, "complete": False, "claimed": False,
                                  "started": False} for i in range(20)}
        self.active_quest = None
        self.debris.clear()
        self.rockets = []
        self.tnts = []
        self.col_box = None
        self.mp_seen, self.mp_last_eid = set(), 0
        self.loading(status, 1.0)

    def enter_multiplayer_world(self, epoch=None):
        """Fresh procedural world + server dig history only. Never keeps SP holes."""
        self._rebuild_main_world("Loading multiplayer world")
        if epoch is not None:
            self.world_epoch = epoch
        # Host: apply persisted server_world events already loaded into net_server.history
        if self.is_host and self.net_server:
            for ev in list(self.net_server.history):
                if isinstance(ev, dict):
                    self._apply_mp_event(ev, silent=True)
        self.say("Multiplayer world ready (isolated)", 3.0)

    def enter_singleplayer_world(self):
        """Fresh world, then client_save/world.npz if present. Never keeps MP holes."""
        self._rebuild_main_world("Loading singleplayer world")
        self.world_epoch = None
        self.mp_seen, self.mp_last_eid = set(), 0
        try:
            path = self._save_path(SAVE_SLOT)
            if os.path.isfile(path):
                self.load_game(SAVE_SLOT)
                # progress.json is the live offline profile — prefer it over npz blob
                self.load_progress(quiet=True)
                self.say("Singleplayer world loaded from client_save", 3.0)
                return
        except Exception as e:
            print(f"[client_save] load failed: {e}")
        self.say("Singleplayer world ready (client_save only)", 3.0)

    def reset_world(self, epoch=None):
        """Server-wide map reset: rebuild a pristine main world. Coins/gear are kept."""
        self.say("Server reset the map - rebuilding world...", 4.0)
        self._rebuild_main_world("Server map reset")
        if epoch is not None:
            self.world_epoch = epoch
        self.save_progress(quiet=True)
        self.say("Map reset - fresh world loaded", 4.0)

    def interact(self):
        if self.mp_open:
            self.open_mp(False)
        elif self.shop_open:
            self.set_shop(False)
        elif self.near_server_pc():
            self.open_mp(True)
        elif self.near_terminal():
            self.set_shop(True)
            self.shop_item = 0
            self.shop_scroll = 0

    # ----------------------------------------------------------------- events
    def handle_events(self):
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                return False
            if e.type == pygame.VIDEORESIZE:
                self.win_w, self.win_h = max(e.w, 320), max(e.h, 240)
            elif e.type == pygame.TEXTINPUT:
                if getattr(self, "chat_open", False) and not self.in_menu:
                    if len(self.chat_buf) < 80:
                        self.chat_buf += e.text
            elif e.type == pygame.KEYDOWN:
                k = e.key
                # ---- multiplayer chat input ----
                if getattr(self, "chat_open", False) and not self.in_menu:
                    if k == pygame.K_ESCAPE:
                        self.chat_open = False
                        self.chat_buf = ""
                        pygame.key.stop_text_input()
                    elif k in (pygame.K_RETURN, pygame.K_KP_ENTER):
                        self._send_chat_line(self.chat_buf)
                        self.chat_open = False
                        self.chat_buf = ""
                        pygame.key.stop_text_input()
                    elif k == pygame.K_BACKSPACE:
                        self.chat_buf = self.chat_buf[:-1]
                    continue
                if k == pygame.K_t and not self.in_menu and not self.paused and not self.shop_open and not self.mp_open and not self.map_open and self._in_multiplayer():
                    self.chat_open = True
                    self.chat_buf = ""
                    pygame.key.start_text_input()
                    continue
                if self.in_menu:
                    if self.settings_open:
                        if k == pygame.K_ESCAPE:
                            self.open_settings(False)
                        continue
                    if k in (pygame.K_UP, pygame.K_w):
                        self.menu_sel = (self.menu_sel - 1) % 4
                        self.sfx.play("mine")
                    elif k in (pygame.K_DOWN, pygame.K_s):
                        self.menu_sel = (self.menu_sel + 1) % 4
                        self.sfx.play("mine")
                    elif k in (pygame.K_LEFT, pygame.K_a) and self.menu_sel == 1:
                        self.cycle_mode(-1)
                    elif k in (pygame.K_RIGHT, pygame.K_d) and self.menu_sel == 1:
                        self.cycle_mode(1)
                    elif k in (pygame.K_RETURN, pygame.K_SPACE, pygame.K_e):
                        if self.menu_sel == 0:
                            self.start_game()
                        elif self.menu_sel == 1:
                            self.cycle_mode(1)
                        elif self.menu_sel == 2:
                            self.open_settings(True)
                        else:
                            return False
                    elif k == pygame.K_ESCAPE:
                        return False
                    elif k == pygame.K_n:
                        on = self.sfx.toggle_music()
                        self.say("Music ON" if on else "Music OFF", 1.5)
                    continue
                if k == pygame.K_m and not self.paused and not self.shop_open and self.scene.name == "main":
                    self.open_map(not self.map_open)
                elif self.map_open:
                    if k == pygame.K_ESCAPE:
                        self.open_map(False)
                    elif k == pygame.K_UP:
                        self.map_scroll = max(0, self.map_scroll - 1)
                    elif k == pygame.K_DOWN:
                        self.map_scroll = min(max(0, len(self.map_destinations()) - 11), self.map_scroll + 1)
                elif k == pygame.K_ESCAPE:
                    if getattr(self, "settings_open", False):
                        self.open_settings(False)
                    elif self.mp_open:
                        self.open_mp(False)
                    elif self.shop_open:
                        self.set_shop(False)
                    else:
                        self.set_paused(not self.paused)
                elif self.paused:
                    if k == pygame.K_q:
                        return False
                    if k == pygame.K_m:
                        self.save_progress(quiet=True)
                        self.set_paused(False)
                        self.in_menu = True
                        pygame.mouse.set_visible(True)
                        pygame.event.set_grab(False)
                    if k == pygame.K_l and self.scene.name != "lobby":
                        self.set_paused(False)
                        self.teleport("lobby", "default")
                    if k == pygame.K_s:
                        self.save_game()
                    if k == pygame.K_o:
                        self.load_game()
                elif k == pygame.K_F5:
                    self.save_game()
                elif k == pygame.K_F9:
                    self.load_game()
                elif k == pygame.K_F2:
                    self.set_spectator(not self.spectator)
                elif k == pygame.K_F3 and not self.in_menu:
                    self.open_settings()
                elif k == pygame.K_F8 and self.net_client.connected:
                    # Force full catch-up: rebuild main world, replay entire server log
                    self.say("Forcing full world resync...", 3.0)
                    self.mp_seen, self.mp_last_eid = set(), 0
                    self.enter_multiplayer_world(epoch=getattr(self.net_client, "epoch", self.world_epoch))
                    self.net_client.request_resync(0)
                elif k == pygame.K_F1:
                    # Hide all HUD / UI chrome (for screenshots & trailers) — NEVER gives coins
                    self.hide_hud = not getattr(self, "hide_hud", False)
                    print(f"[F1] hide_hud={self.hide_hud} (build 8 — coins are on F10 for OP only)")
                    if self.hide_hud:
                        self.say("HUD HIDDEN", 1.2)
                    else:
                        self.say("HUD ON", 1.2)
                elif k == pygame.K_F10:
                    # OP-only debug money
                    if not getattr(self, "is_op", False):
                        self.say("F10 debug money requires OP", 2.5)
                        self.sfx.play("deny")
                    else:
                        self.coins += 500
                        self.say(f"+500 DCOIN (debug) · {int(self.coins)}", 2.0)
                        self.sfx.play("coin")
                elif k == pygame.K_n:
                    on = self.sfx.toggle_music()
                    self.say("Music ON" if on else "Music OFF", 1.5)
                elif k == pygame.K_F11:
                    try:
                        pygame.display.toggle_fullscreen()
                    except pygame.error:
                        pass
                elif k == pygame.K_e:
                    self.interact()
                elif k == pygame.K_TAB and self.shop_open:
                    self.set_shop(False)
                elif not self.shop_open:
                    if k == pygame.K_1:
                        self.tool = 0
                    elif k == pygame.K_2 and self.has_launcher:
                        self.tool = 1
                    elif k == pygame.K_3 and getattr(self, "has_tnt", False):
                        self.tool = 2
                    elif k == pygame.K_F6 and self.scene.name == "main":
                        self.cycle_weather(1)
                        self.sfx.play("mine")
            elif e.type == pygame.MOUSEWHEEL and self.map_open:
                hx, hy = self.mouse_hud()
                map_rect = pygame.Rect(16, 100, 680, 575)
                if map_rect.collidepoint(hx, hy):
                    mw, md = self.scene.map_size
                    old_scale = min((map_rect.w - 12) / mw, (map_rect.h - 12) / md) * self.map_zoom
                    old_x = map_rect.centerx - (mw / 2) * old_scale + self.map_pan[0]
                    old_y = map_rect.centery - (md / 2) * old_scale + self.map_pan[1]
                    world_x = (hx - old_x) / max(old_scale, 1e-6)
                    world_z = (hy - old_y) / max(old_scale, 1e-6)
                    self.map_zoom = max(0.65, min(3.2, self.map_zoom * (1.13 ** e.y)))
                    new_scale = min((map_rect.w - 12) / mw, (map_rect.h - 12) / md) * self.map_zoom
                    self.map_pan[0] = hx - (map_rect.centerx - (mw / 2) * new_scale + world_x * new_scale)
                    self.map_pan[1] = hy - (map_rect.centery - (md / 2) * new_scale + world_z * new_scale)
                else:
                    self.map_scroll = max(0, min(max(0, len(self.map_destinations()) - 11),
                                                 self.map_scroll - e.y))
            elif e.type == pygame.MOUSEBUTTONUP and self.map_open and e.button == 1:
                self.map_dragging = False
                self.map_drag_last = None
            elif e.type == pygame.MOUSEWHEEL and not self.shop_open and not self.paused:
                if self.has_launcher:
                    self.tool ^= 1
            elif e.type == pygame.MOUSEBUTTONDOWN and getattr(self, "settings_open", False) and e.button == 1:
                hx, hy = self.mouse_hud()
                for item in getattr(self, "settings_rects", []):
                    if len(item) != 3:
                        continue
                    kind, val, rect = item
                    if not rect.collidepoint(hx, hy):
                        continue
                    if kind == "shader":
                        self.set_shader_pack(val)
                    elif kind == "effect":
                        self.toggle_shader_effect(val)
                    elif kind == "sp":
                        self.toggle_singleplayer_feature(val)
                    elif kind == "back":
                        self.open_settings(False)
                    elif kind == "rd":
                        self.set_render_distance(val)
                    break
            elif e.type == pygame.MOUSEBUTTONDOWN and self.map_open and e.button == 1:
                hx, hy = self.mouse_hud()
                map_rect = pygame.Rect(16, 100, 680, 575)
                clicked_list = False
                for rect, destination in self.map_list_rects:
                    if rect.collidepoint(hx, hy):
                        self.map_selected = destination
                        clicked_list = True
                        break
                if self.map_quest_rect.collidepoint(hx, hy) and self.map_selected:
                    if self.map_selected.get("kind") == "BIOME":
                        self.start_biome_quest(self.map_selected.get("biome"))
                elif self.map_travel_rect.collidepoint(hx, hy) and self.map_selected:
                    self.quick_travel(self.map_selected)
                elif not clicked_list and map_rect.collidepoint(hx, hy):
                    self.map_dragging = True
                    self.map_drag_last = (hx, hy)
                    # Select the nearest visible destination when clicking a marker.
                    mw, md = self.scene.map_size
                    base_scale = min((map_rect.w - 12) / mw, (map_rect.h - 12) / md) * self.map_zoom
                    dw, dh = int(mw * base_scale), int(md * base_scale)
                    dx = int(map_rect.centerx - dw / 2 + self.map_pan[0])
                    dy = int(map_rect.centery - dh / 2 + self.map_pan[1])
                    nearest, best = None, 13
                    for destination in self.map_destinations():
                        qx, qy = dx + destination["x"] * base_scale, dy + destination["z"] * base_scale
                        dist = math.hypot(hx - qx, hy - qy)
                        if dist < best:
                            nearest, best = destination, dist
                    if nearest:
                        self.map_selected = nearest
            elif e.type == pygame.MOUSEMOTION and self.map_open and self.map_dragging and self.map_drag_last:
                hx, hy = self.mouse_hud()
                self.map_pan[0] += hx - self.map_drag_last[0]
                self.map_pan[1] += hy - self.map_drag_last[1]
                map_rect = pygame.Rect(16, 100, 680, 575)
                mw, md = self.scene.map_size
                scale = min((map_rect.w - 12) / mw, (map_rect.h - 12) / md) * self.map_zoom
                dw, dh = mw * scale, md * scale
                limit_x = max(0.0, (dw - map_rect.w) / 2 + 55)
                limit_y = max(0.0, (dh - map_rect.h) / 2 + 55)
                self.map_pan[0] = max(-limit_x, min(limit_x, self.map_pan[0]))
                self.map_pan[1] = max(-limit_y, min(limit_y, self.map_pan[1]))
                self.map_drag_last = (hx, hy)
            elif e.type == pygame.MOUSEBUTTONDOWN and self.shop_open and e.button == 1:
                hx, hy = self.mouse_hud()
                for i, rect in enumerate(self.shop_item_rects):
                    if rect.collidepoint(hx, hy):
                        self.shop_item = i
                        break
                for upg_idx, rect in self.shop_buy_rects:
                    if rect.collidepoint(hx, hy):
                        self.buy_upgrade(self.shop_item, upg_idx)
                        break
            elif e.type == pygame.MOUSEWHEEL and self.shop_open:
                self.shop_scroll = max(0, self.shop_scroll - e.y)
            elif e.type == pygame.MOUSEBUTTONDOWN and self.in_menu and e.button == 1:
                hx, hy = self.mouse_hud()
                for i, rect in enumerate(getattr(self, "menu_rects", [])):
                    if rect.collidepoint(hx, hy):
                        self.menu_sel = i
                        if i == 0:
                            self.start_game()
                        elif i == 1:
                            self.cycle_mode(1)
                        elif i == 2:
                            self.open_settings(True)
                        else:
                            return False
                        break
            elif e.type == pygame.MOUSEBUTTONDOWN and self.mp_open and e.button == 1:
                hx, hy = self.mouse_hud()
                for action, rect in getattr(self, "mp_rects", []):
                    if not rect.collidepoint(hx, hy):
                        continue
                    if action == "host":
                        self.host_server()
                    elif action == "stop":
                        self.stop_multiplayer()
                    elif action == "join_sel":
                        if 0 <= self.mp_selected < len(self.mp_servers):
                            s = self.mp_servers[self.mp_selected]
                            self.join_server(s["host"], s["port"])
                    elif action == "refresh":
                        self.mp_servers = default_server_list()
                        self.mp_status = "Server list refreshed"
                    elif isinstance(action, tuple) and action[0] == "sel":
                        self.mp_selected = action[1]
                    break
        return True

    # ---------------------------------------------------------------- physics
    def hit(self, p, h=None):
        h = self.h if h is None else h
        return self.scene.world.box_hit(p[0] - 0.3, p[1], p[2] - 0.3,
                                        p[0] + 0.3, p[1] + h - 1e-4, p[2] + 0.3)

    def move_player(self, dt, keys):
        locked = self.shop_open or self.fade_dir == 1 or getattr(self, "settings_open", False)
        # OP spectator: noclip fly, no gravity / collision
        if getattr(self, "spectator", False) and self.is_op:
            speed = 18.0
            if keys[pygame.K_LSHIFT]:
                speed = 42.0
            if keys[pygame.K_LCTRL]:
                speed = 6.0
            yaw, pitch = self.yaw, self.pitch
            cp = math.cos(pitch)
            fwd = np.array([math.sin(yaw) * cp, math.sin(pitch), -math.cos(yaw) * cp])
            right = np.array([math.cos(yaw), 0.0, math.sin(yaw)])
            up = np.array([0.0, 1.0, 0.0])
            move = np.zeros(3)
            if keys[pygame.K_w]:
                move += fwd
            if keys[pygame.K_s]:
                move -= fwd
            if keys[pygame.K_d]:
                move += right
            if keys[pygame.K_a]:
                move -= right
            if keys[pygame.K_SPACE]:
                move += up
            if keys[pygame.K_LCTRL]:
                move -= up
            n = np.linalg.norm(move)
            if n > 1e-6:
                move = move / n * speed
            self.pos = self.pos + move * dt
            self.vel[:] = 0
            self.grounded = False
            return
        w = self.scene.world
        # crouch
        want_crouch = keys[pygame.K_LCTRL] and not locked
        if want_crouch:
            self.h = 1.3
        elif self.h < 1.8 and not self.hit(self.pos, 1.8):
            self.h = 1.8
        fwd = np.array([math.sin(self.yaw), -math.cos(self.yaw)])
        right = np.array([math.cos(self.yaw), math.sin(self.yaw)])
        mx = (keys[pygame.K_d] - keys[pygame.K_a]) if not locked else 0
        mz = (keys[pygame.K_w] - keys[pygame.K_s]) if not locked else 0
        wish = right * mx + fwd * mz
        ln = np.linalg.norm(wish)
        if ln > 0:
            wish = wish / ln
        base = 2.8 if self.h < 1.7 else 5.5
        sprint = 8.5 + 0.7 * self.upg.get("sprint_boost", 0)
        if self.h >= 1.7 and keys[pygame.K_LSHIFT] and not locked:
            base = sprint
        base *= 1.0 + 0.08 * self.upg.get("move_speed", 0)
        speed = base
        target = wish * speed

        space = keys[pygame.K_SPACE] and not locked
        self.coyote = 0.12 if self.grounded else max(0.0, self.coyote - dt)
        jetting = False
        if self.grounded:
            self.extra_jumps = self.upg.get("double_jump", 0)
        if space and self.coyote > 0 and self.vel[1] <= 0.5:
            self.vel[1] = 8.6 + 0.9 * self.upg.get("jump_boost", 0)
            self.coyote = 0.0
            self.grounded = False
        elif space and not self.grounded and self.extra_jumps > 0 and self.vel[1] < 4.0 and not getattr(self, "_jump_held", False):
            self.vel[1] = 7.5 + 0.6 * self.upg.get("jump_boost", 0)
            self.extra_jumps -= 1
        elif space and self.has_jet and not self.grounded and self.fuel > 0:
            jetting = True
            thrust = 44.0 + 8.0 * self.upg.get("jet_thrust", 0)
            cap = 10.0 + 1.5 * self.upg.get("jet_thrust", 0)
            if self.vel[1] < cap:
                self.vel[1] = min(cap, self.vel[1] + thrust * dt)
            drain = 40.0
            if self.game_mode == "survival":
                drain = 58.0
            elif self.game_mode == "chaos":
                drain = 50.0
            drain *= max(0.45, 1.0 - 0.12 * self.upg.get("jet_efficiency", 0))
            self.fuel = max(0.0, self.fuel - drain * dt)
            self.jet_sound_t -= dt
            if self.jet_sound_t <= 0:
                self.sfx.play("jet")
                self.jet_sound_t = 0.11
            k = 5
            ppos = self.pos + np.array([0.0, 0.2, 0.0]) + np.random.uniform(-0.2, 0.2, (k, 3))
            pvel = np.random.uniform(-1.5, 1.5, (k, 3))
            pvel[:, 1] = self.vel[1] - np.random.uniform(6, 10, k)
            cols = np.array([[1.8, random.uniform(0.7, 1.1), 0.2]] * k)
            self.debris.add(ppos, pvel, 0.09, cols, 0.45, 0.1)
        self._jump_held = space
        if not jetting and self.grounded:
            regen = 18.0 if self.game_mode == "survival" else 30.0
            regen *= 1.0 + 0.2 * self.upg.get("regen_boost", 0)
            self.fuel = min(self.max_fuel, self.fuel + regen * dt)

        accel = 14.0 if self.grounded else (4.0 if jetting else 2.5)
        if not self.grounded:
            accel *= 1.0 + 0.25 * self.upg.get("air_control", 0)
        k = min(1.0, accel * dt)
        self.vel[0] += (target[0] - self.vel[0]) * k
        self.vel[2] += (target[1] - self.vel[2]) * k
        gravity = GRAVITY * (0.42 if self.singleplayer_feature_enabled("moon_gravity") else 1.0)
        self.vel[1] = max(self.vel[1] - gravity * dt, -60.0)

        # un-stick if rubble settled inside us
        for _ in range(20):
            if not self.hit(self.pos):
                break
            self.pos[1] += 0.1

        sp = max(abs(self.vel[0]), abs(self.vel[1]), abs(self.vel[2]))
        n = max(1, int(math.ceil(sp * dt / 0.1)))
        sdt = dt / n
        was_grounded = self.grounded
        for _ in range(n):
            for ax in (0, 2):
                d = self.vel[ax] * sdt
                if d == 0.0:
                    continue
                new = self.pos.copy()
                new[ax] += d
                if not self.hit(new):
                    self.pos = new
                    continue
                if (was_grounded or self.grounded) and self.vel[1] <= 0.5:
                    stepped = False
                    for s in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6):
                        t = new.copy()
                        t[1] += s
                        if not self.hit(t):
                            self.pos = t
                            self.step_off -= s
                            stepped = True
                            break
                    if stepped:
                        continue
                lo, hi = 0.0, d
                for _ in range(7):
                    mid = (lo + hi) / 2
                    t = self.pos.copy()
                    t[ax] += mid
                    if self.hit(t):
                        hi = mid
                    else:
                        lo = mid
                self.pos[ax] += lo
                self.vel[ax] = 0.0
            d = self.vel[1] * sdt
            new = self.pos.copy()
            new[1] += d
            if not self.hit(new):
                self.pos = new
            else:
                lo, hi = 0.0, d
                for _ in range(7):
                    mid = (lo + hi) / 2
                    t = self.pos.copy()
                    t[1] += mid
                    if self.hit(t):
                        hi = mid
                    else:
                        lo = mid
                self.pos[1] += lo
                self.vel[1] = 0.0
            probe = self.pos.copy()
            probe[1] -= 0.04
            self.grounded = self.hit(probe) and self.vel[1] <= 0.1
        # Jump pads (PORTAL_B plates) – big bounce
        if self.grounded and self.scene.name == "main":
            fx = int(self.pos[0] / VS)
            fz = int(self.pos[2] / VS)
            fy = int(self.pos[1] / VS)
            mw, _, md = self.scene.world.W, self.scene.world.H, self.scene.world.D
            if 0 <= fx < mw and 0 <= fz < md and 0 <= fy < self.scene.world.H:
                under = self.scene.world.blocks[fx, max(0, fy - 1):fy + 1, fz]
                if np.any(under == PORTAL_B):
                    boost = 18.0 + 3.5 * self.upg.get("pad_boost", 0)
                    self.vel[1] = max(self.vel[1], boost)
                    self.grounded = False
                    self.sfx.play("launch")
        self.step_off *= math.exp(-14.0 * dt)
        if self.pos[1] < self.scene.kill_y:
            (px, py, pz), _ = self.scene.spawns["default"]
            self.pos = np.array([px, py, pz], float)
            self.vel[:] = 0

    # ------------------------------------------------------------------ tools
    def note_edit(self, lo, hi, margin):
        if self.col_box is None:
            self.col_box = [list(lo), list(hi)]
            self.col_margin = margin
        else:
            for i in range(3):
                self.col_box[0][i] = min(self.col_box[0][i], lo[i])
                self.col_box[1][i] = max(self.col_box[1][i], hi[i])
            self.col_margin = max(self.col_margin, margin)

    def update_tools(self, dt, lmb):
        self.cooldown = max(0.0, self.cooldown - dt)
        self.target = None
        sc = self.scene
        if getattr(self, "spectator", False):
            return
        if not sc.world.destructible:
            return
        eye, fwd = self.eye(), self.forward()
        if self.tool == 0:
            mreach = 6.0 + 0.7 * self.upg.get("monster_range", 0)
            monster = self.monster_target(mreach) if self.scene.name == "main" else None
            hit = sc.world.raycast(eye, fwd, 6.0 + 0.4 * self.upg.get("pick_reach", 0))
            self.target = hit
            if lmb and monster is not None and self.cooldown <= 0:
                dmg = 1 + self.upg.get("monster_dmg", 0)
                self.damage_monster(monster, dmg)
                self.cooldown = 0.32
                self.progress = 0.0
                self.mining_cell = None
            elif lmb and hit and sc.world.blocks[hit] != BEDROCK:
                rad = self.pick_radius
                if self.mining_cell is None or math.dist(hit, self.mining_cell) > rad:
                    self.mining_cell = hit
                    self.progress = 0.0
                btype = int(sc.world.blocks[hit])
                dur = 0.30 * HARDNESS[btype] / (1.0 + 0.45 * (self.pick_lvl - 1) + 0.2 * self.upg.get("pick_speed", 0))
                self.progress += dt / max(dur, 0.03)
                self.mine_sound_t -= dt
                if self.mine_sound_t <= 0:
                    self.sfx.play("mine")
                    self.mine_sound_t = 0.17
                if self.progress >= 1.0:
                    self.mine_hit(hit)
            else:
                self.mining_cell, self.progress = None, 0.0
        elif self.tool == 1:
            infinite_ammo = self.singleplayer_feature_enabled("infinite_ammo")
            if lmb and self.cooldown <= 0 and (self.rocket_ammo > 0 or infinite_ammo):
                if not infinite_ammo:
                    self.rocket_ammo -= 1
                self.cooldown = 0.7
                spd = 42.0 + 6.0 * self.upg.get("rocket_speed", 0)
                rpos = eye + fwd * 0.8
                rvel = fwd * spd + self.vel * 0.2
                self.rockets.append({"pos": rpos, "vel": rvel, "life": 5.0, "remote": False})
                self._mp_send_rocket(rpos, rvel, 5.0)
                if self.upg.get("rocket_multi", 0) and random.random() < 0.2 * self.upg["rocket_multi"]:
                    rpos2 = eye + fwd * 0.6 + np.array([0.2, 0, 0])
                    rvel2 = fwd * spd * 0.9 + self.vel * 0.2
                    self.rockets.append({"pos": rpos2, "vel": rvel2, "life": 5.0, "remote": False})
                    self._mp_send_rocket(rpos2, rvel2, 5.0)
                self.sfx.play("launch")
            elif lmb and self.rocket_ammo == 0 and not infinite_ammo and self.cooldown <= 0:
                self.cooldown = 0.5
                self.say("Out of rockets - buy more at the shop")
                self.sfx.play("deny")
        elif self.tool == 2:
            infinite_ammo = self.singleplayer_feature_enabled("infinite_ammo")
            if lmb and self.cooldown <= 0 and (self.tnt_ammo > 0 or infinite_ammo):
                # Place TNT a short distance ahead / on the aimed block
                hit = sc.world.raycast(eye, fwd, 5.5)
                if hit is not None:
                    place = (np.array(hit, float) + 0.5) * VS
                else:
                    place = eye + fwd * 2.2
                if not infinite_ammo:
                    self.tnt_ammo -= 1
                self.cooldown = 0.55
                fuse = self.tnt_fuse_time
                rad = self.tnt_radius
                self.tnts.append({
                    "pos": np.asarray(place, float),
                    "fuse": fuse,
                    "radius": rad,
                    "remote": False,
                    "seed": random.getrandbits(31),
                })
                self.sfx.play("launch")
                self.say(f"TNT set — {fuse:.1f}s", 1.2)
            elif lmb and self.tnt_ammo == 0 and not infinite_ammo and self.cooldown <= 0:
                self.cooldown = 0.45
                self.say("Out of TNT - buy more at the shop")
                self.sfx.play("deny")

    def mine_hit(self, cell):
        w = self.scene.world
        c = np.array(cell, float) + 0.5
        seed = random.getrandbits(31)   # shared with other players => identical crater
        types, idx, lo, hi = w.carve_sphere(c, self.pick_radius, rough=0.7, seed=seed)
        self.mining_cell, self.progress = None, 0.0
        if len(types) == 0:
            return
        mult = 1.0 + 0.25 * (self.pick_lvl - 1) + 0.12 * self.upg.get("pick_fortune", 0)
        mult *= 1.0 + 0.1 * self.upg.get("loot_bonus", 0)
        vals = VALUE[types].astype(np.float64)
        if self.upg.get("auto_smelt", 0):
            vals = np.where(types == ORE, vals * (1.0 + 0.35 * self.upg["auto_smelt"]), vals)
        if self.upg.get("crate_luck", 0):
            vals = np.where(types == PLANK, vals * (1.0 + 0.5 * self.upg["crate_luck"]), vals)
            vals = np.where(types == GOLD, vals * (1.0 + 0.4 * self.upg.get("treasure_hunter", 0)), vals)
        self.add_coins(vals.sum() / 16.0 * mult)
        k = min(len(types), 26)
        sel = np.random.choice(len(types), k, replace=False)
        pos = (idx[sel] + 0.5) * VS
        vel = (pos - c * VS) * 6.0 + np.random.uniform(-1.5, 1.5, (k, 3))
        vel[:, 1] += np.random.uniform(2, 5, k)
        cols = PALETTE[types[sel]] * np.random.uniform(0.8, 1.0, (k, 1))
        self.debris.add(pos, vel, VS * 0.9, cols, 3.0)
        self.sfx.play("break")
        self.sfx.play("coin")
        self.note_edit(lo, hi, 24)
        self._mp_send_dig(c, float(self.pick_radius), seed)

    def _mp_send_dig(self, c, r, seed):
        if not self._in_multiplayer():
            return
        if self.net_client.connected:
            self.net_client.send_dig(float(c[0]), float(c[1]), float(c[2]), float(r), seed)
        elif self.is_host and self.net_server:
            self.net_server.host_broadcast({
                "t": "dig", "id": self.player_id,
                "x": float(c[0]), "y": float(c[1]), "z": float(c[2]),
                "r": float(r), "seed": seed,
            })

    def _mp_send_rocket(self, pos, vel, life=5.0):
        if not self._in_multiplayer():
            return
        if self.net_client.connected:
            self.net_client.send_rocket(pos, vel, life)
        elif self.is_host and self.net_server:
            self.net_server.host_broadcast({
                "t": "rocket", "id": self.player_id,
                "x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2]),
                "vx": float(vel[0]), "vy": float(vel[1]), "vz": float(vel[2]),
                "life": float(life),
            })

    def _mp_send_boom(self, pos, radius, seed=None):
        if not self._in_multiplayer():
            return
        if self.net_client.connected:
            self.net_client.send_boom(pos, radius, seed)
        elif self.is_host and self.net_server:
            self.net_server.host_broadcast({
                "t": "boom", "id": self.player_id,
                "x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2]),
                "r": float(radius), "seed": seed,
            })

    def update_rockets(self, dt):
        w = self.scene.world
        alive = []
        for r in self.rockets:
            r["life"] -= dt
            speed = float(np.linalg.norm(r["vel"]))
            steps = max(1, int(speed * dt / 0.12) + 1)
            sub = dt / steps
            boom = False
            for _ in range(steps):
                r["vel"][1] -= 2.5 * sub
                r["pos"] = r["pos"] + r["vel"] * sub
                if w.solid_at(r["pos"]):
                    boom = True
                    break
            self.debris.add(r["pos"][None, :], np.random.uniform(-0.4, 0.4, (1, 3)),
                            0.11, [[0.6, 0.6, 0.6]], 0.7, 0.05)
            if boom:
                # Remote rockets only look pretty; the owner sends the real boom
                if r.get("remote"):
                    continue
                self.explode(r["pos"] - r["vel"] / max(speed, 1e-3) * 0.3)
            elif r["life"] > 0:
                alive.append(r)
        self.rockets = alive[:32]

    def update_tnts(self, dt):
        if not self.tnts:
            return
        still = []
        for t in self.tnts:
            t["fuse"] -= dt
            # blink debris spark
            if random.random() < 0.25:
                self.debris.add(
                    t["pos"][None, :] + np.random.uniform(-0.1, 0.1, (1, 3)),
                    np.random.uniform(-0.5, 0.5, (1, 3)),
                    0.08, [[1.6, 0.4, 0.1]], 0.35, 0.05)
            if t["fuse"] <= 0:
                if not t.get("remote"):
                    self.explode(t["pos"], radius=t["radius"], seed=t.get("seed"))
                    # Sympathetic detonation chance
                    if self.upg.get("tnt_chain", 0):
                        for other in self.tnts:
                            if other is t or other.get("remote"):
                                continue
                            if np.linalg.norm(other["pos"] - t["pos"]) < t["radius"] * 1.8:
                                if random.random() < 0.25 * self.upg["tnt_chain"]:
                                    other["fuse"] = min(other["fuse"], 0.15)
                continue
            still.append(t)
        self.tnts = still[:24]

    def explode(self, c_m, from_remote=False, radius=None, seed=None, silent=False):
        # Explosions always edit the MAIN world. (Old build carved whatever scene you
        # were standing in => remote booms punched holes in the lobby/shop.)
        w = self.scenes["main"].world
        R_m = float(radius) if radius is not None else self.blast_radius
        visible = (self.scene.name == "main") and not silent  # sounds/debris/knockback/shake
        if seed is None:
            seed = random.getrandbits(31)
        if not from_remote:
            if self.scene.name == "main":
                for monster in self.monsters:
                    if (monster["alive"] and monster.get("active", False) and
                            np.linalg.norm(monster["pos"] - np.asarray(c_m)) <= R_m + 1.8):
                        self.damage_monster(monster, 3)
            self._mp_send_boom(c_m, R_m, seed)   # others carve with the SAME seed
        if visible:
            try:
                c_arr = np.asarray(c_m, float)
                self.rockets = [r for r in self.rockets
                                if np.linalg.norm(np.asarray(r["pos"], float) - c_arr) > R_m + 2.0]
            except Exception:
                pass
        c = np.asarray(c_m, float) / VS
        types, idx, lo, hi = w.carve_sphere(c, R_m / VS, rough=2.2, seed=seed)
        n = len(types)
        if n:
            if not from_remote:
                mult = 1.0 + 0.25 * (self.pick_lvl - 1)
                self.add_coins(VALUE[types].sum() / 64.0 * 0.6 * mult)
                self.note_edit(lo, hi, 40)   # only the owner runs/sends the collapse
            if visible:
                k = min(n, 650)
                sel = np.random.choice(n, k, replace=False)
                pos = (idx[sel] + 0.5) * VS
                dirv = pos - np.asarray(c_m)
                dl = np.linalg.norm(dirv, axis=1, keepdims=True)
                dirv = dirv / np.maximum(dl, 1e-3)
                spd = 7 + 14 * (1 - np.clip(dl / R_m, 0, 1)) + np.random.uniform(0, 4, dl.shape)
                vel = dirv * spd + np.array([0, 4.0, 0]) + np.random.uniform(-1.5, 1.5, (k, 3))
                size = np.random.uniform(0.8, 2.6, k) * VS
                cols = PALETTE[types[sel]] * np.random.uniform(0.8, 1.0, (k, 1))
                self.debris.add(pos, vel, size, cols, np.random.uniform(8, 14, k), 1.0,
                                types[sel], True)
        if not visible:
            return
        k = 80
        fpos = np.asarray(c_m) + np.random.uniform(-1, 1, (k, 3)) * R_m * 0.4
        fvel = np.random.uniform(-1, 1, (k, 3))
        fvel = fvel / np.maximum(np.linalg.norm(fvel, axis=1, keepdims=True), 1e-3) * \
            np.random.uniform(4, 12, (k, 1))
        hot = np.random.rand(k, 1) < 0.6
        fcols = np.where(hot, np.array([[1.8, 0.9, 0.2]]), np.array([[0.32, 0.32, 0.32]]))
        self.debris.add(fpos, fvel, np.random.uniform(0.12, 0.45, k), fcols,
                        np.random.uniform(0.5, 1.2, k), 0.15)
        pc = self.pos + np.array([0.0, 0.9, 0.0])
        d = pc - np.asarray(c_m)
        dist = float(np.linalg.norm(d))
        reach = R_m * 2.2
        if dist < reach:
            f = 1.0 - dist / reach
            kb = max(0.25, 1.0 - 0.2 * self.upg.get("thorns", 0))
            self.vel += d / max(dist, 0.1) * 26.0 * f * kb
            self.vel[1] += 4.0 * f * kb
            self.grounded = False
        sh = max(0.2, 1 - dist / 25.0) * max(0.3, 1.0 - 0.25 * self.upg.get("shake_reduce", 0))
        self.shake = min(1.0, self.shake + sh)
        self.sfx.play("boom")

    def _collapse_effects(self, idx, types):
        """Rubble, shake and sound for a collapse (idx/types = removed voxels)."""
        n = len(idx)
        k = min(n, 5000)
        sel = np.arange(n) if n <= k else np.random.choice(n, k, replace=False)
        scale = min(4.0, (n / max(k, 1)) ** (1 / 3))
        pos = (idx[sel] + 0.5) * VS
        vel = np.random.uniform(-0.8, 0.8, (len(sel), 3))
        vel[:, 1] *= 0.3
        cols = PALETTE[types[sel]] * np.random.uniform(0.85, 1.0, (len(sel), 1))
        self.debris.add(pos, vel, VS * scale, cols, 25.0, 1.0, types[sel], True)
        self.shake = min(1.0, self.shake + min(0.6, n / 20000.0))
        self.sfx.play("boom" if n > 3000 else "break")

    def _do_collapse(self, lo, hi, margin, broadcast=True, visible=True):
        w = self.scenes["main"].world
        res = collapse_region(w, lo, hi, margin)
        if res is None:
            return
        idx, types = res
        n = len(idx)
        w.touch(idx.min(0), idx.max(0) + 1)
        if visible:
            self._collapse_effects(idx, types)
        if not broadcast:
            return
        try:
            if n <= COLLAPSE_MAX_CELLS:
                cell_list = idx.astype(int).tolist()
            else:
                pick = np.random.choice(n, COLLAPSE_MAX_CELLS, replace=False)
                cell_list = idx[pick].astype(int).tolist()
            self._mp_send_collapse(lo, hi, margin, cell_list)
        except Exception:
            self._mp_send_collapse(lo, hi, margin, None)

    def run_collapse(self):
        lo, hi = self.col_box
        margin = self.col_margin if ndimage else min(self.col_margin, 24)
        self.col_box = None
        self._do_collapse(lo, hi, margin, broadcast=True, visible=self.scene.name == "main")

    def _mp_send_collapse(self, lo, hi, margin, cells=None):
        if not self._in_multiplayer():
            return
        if self.net_client.connected:
            self.net_client.send_collapse(lo, hi, margin, cells)
        elif self.is_host and self.net_server:
            msg = {
                "t": "collapse", "id": self.player_id,
                "x0": int(lo[0]), "y0": int(lo[1]), "z0": int(lo[2]),
                "x1": int(hi[0]), "y1": int(hi[1]), "z1": int(hi[2]),
                "m": int(margin),
                "cid": uuid.uuid4().hex[:16],
            }
            if cells is not None:
                msg["cells"] = cells
            self.net_server.host_broadcast(msg)

    # ----------------------------------------------------------------- weather
    def set_weather(self, name, announce=True):
        name = name if name in self.WEATHER_ORDER else "clear"
        self.weather = name
        self.weather_t = 0.0
        self.weather_particles = []
        if announce and not self.in_menu:
            labels = {
                "clear": "Sky clears",
                "light": "Soft light — clear skies",
                "fog": "Fog rolls in",
                "rain": "Rain begins",
                "snow": "Snowfall",
                "storm": "Storm front (no lightning)",
                "hail": "Hailstorm!",
            }
            self.say(labels.get(name, name), 2.5)

    def cycle_weather(self, delta=1):
        idx = self.WEATHER_ORDER.index(self.weather) if self.weather in self.WEATHER_ORDER else 0
        self.set_weather(self.WEATHER_ORDER[(idx + delta) % len(self.WEATHER_ORDER)])

    def _weather_fog_mult(self):
        """Return fog density multiplier; night_eye + storm_resist reduce it."""
        base = {
            "clear": 1.0, "light": 0.85, "fog": 2.8, "rain": 1.6,
            "snow": 1.9, "storm": 2.4, "hail": 2.1,
        }.get(self.weather, 1.0)
        reduce = 0.12 * self.upg.get("night_eye", 0) + 0.10 * self.upg.get("storm_resist", 0)
        return max(0.35, base * (1.0 - reduce))

    def _update_weather(self, dt):
        if self.in_menu or self.scene.name != "main":
            self.weather_particles = []
            return
        self.weather_t += dt
        # Auto-cycle weather every so often (disabled in creative for control)
        if self.game_mode != "creative" and self.weather_t >= self.weather_next:
            self.weather_next = 70.0 + random.uniform(0, 90)
            # Prefer interesting weather sometimes
            weights = [18, 10, 12, 16, 14, 14, 12]
            choice = random.choices(self.WEATHER_ORDER, weights=weights, k=1)[0]
            self.set_weather(choice)
        # Spawn particles
        eye = self.eye()
        dens = {
            "rain": 28, "snow": 18, "storm": 36, "hail": 22, "fog": 6,
        }.get(self.weather, 0)
        dens = int(dens * (1.0 - 0.15 * self.upg.get("storm_resist", 0)))
        for _ in range(dens):
            if len(self.weather_particles) > 900:
                break
            px = eye[0] + random.uniform(-35, 35)
            pz = eye[2] + random.uniform(-35, 35)
            py = eye[1] + random.uniform(8, 28)
            if self.weather in ("rain", "storm"):
                self.weather_particles.append([
                    px, py, pz, random.uniform(-1, 1), -18 - random.uniform(0, 8),
                    random.uniform(-1, 1), random.uniform(0.8, 1.6), 0])
            elif self.weather == "snow":
                self.weather_particles.append([
                    px, py, pz, random.uniform(-1.5, 1.5), -2.5 - random.uniform(0, 2),
                    random.uniform(-1.5, 1.5), random.uniform(2.0, 4.5), 1])
            elif self.weather == "hail":
                self.weather_particles.append([
                    px, py, pz, random.uniform(-2, 2), -14 - random.uniform(0, 6),
                    random.uniform(-2, 2), random.uniform(0.7, 1.4), 2])
            elif self.weather == "fog":
                self.weather_particles.append([
                    px, eye[1] + random.uniform(-2, 6), pz,
                    random.uniform(-0.4, 0.4), random.uniform(-0.2, 0.2),
                    random.uniform(-0.4, 0.4), random.uniform(1.5, 3.0), 3])
        # Integrate
        alive = []
        for p in self.weather_particles:
            p[0] += p[3] * dt
            p[1] += p[4] * dt
            p[2] += p[5] * dt
            p[6] -= dt
            if p[6] > 0 and p[1] > eye[1] - 12:
                alive.append(p)
        self.weather_particles = alive
        # Storm shake (reduced by upgrades)
        if self.weather == "storm" and random.random() < 0.04:
            sh = 0.08 * max(0.25, 1.0 - 0.28 * self.upg.get("storm_resist", 0)
                            - 0.2 * self.upg.get("shake_reduce", 0))
            self.shake = min(1.0, self.shake + sh)

    def _weather_env_override(self, env):
        """Return a shallow-modified env dict for current weather."""
        e = dict(env)
        mult = self._weather_fog_mult()
        e["fogd"] = float(env.get("fogd", 0.008)) * mult
        if self.weather == "fog":
            e["hor"] = (0.55, 0.60, 0.68)
            e["amb"] = (0.45, 0.48, 0.55)
            e["suncol"] = (0.5, 0.52, 0.55)
        elif self.weather in ("rain", "storm"):
            e["hor"] = (0.40, 0.45, 0.52)
            e["top"] = (0.12, 0.16, 0.28)
            e["amb"] = (0.40, 0.44, 0.52)
            e["suncol"] = (0.45, 0.48, 0.52)
            if self.weather == "storm":
                e["fogd"] *= 1.15
                e["stars"] = 0.0
        elif self.weather == "snow":
            e["hor"] = (0.78, 0.82, 0.90)
            e["top"] = (0.55, 0.62, 0.78)
            e["amb"] = (0.58, 0.62, 0.70)
            e["suncol"] = (0.7, 0.72, 0.78)
        elif self.weather == "hail":
            e["hor"] = (0.50, 0.54, 0.60)
            e["top"] = (0.25, 0.30, 0.40)
            e["amb"] = (0.42, 0.45, 0.52)
        elif self.weather == "light":
            e["fogd"] *= 0.7
            e["suncol"] = tuple(min(1.0, c * 1.15) for c in e.get("suncol", (0.7, 0.7, 0.7)))
            e["amb"] = tuple(min(1.0, c * 1.08) for c in e.get("amb", (0.5, 0.5, 0.5)))
        return e

    # ----------------------------------------------------------------- update
    def update(self, dt):
        self.time += dt
        # Keep ambient loop alive if the mixer channel dropped it
        try:
            if self.sfx.music_on and self.sfx.music and self.sfx._music_channel is not None:
                if not self.sfx._music_channel.get_busy():
                    self.sfx.start_music()
        except Exception:
            pass
        if self.in_menu:
            pygame.mouse.get_rel()
            return
        if self.fade_dir == 1:
            self.fade = min(1.0, self.fade + dt * 3.0)
            if self.fade >= 1.0 and self.pending:
                self.switch_scene(*self.pending)
                self.pending, self.fade_dir = None, -1
        elif self.fade_dir == -1:
            self.fade = max(0.0, self.fade - dt * 2.5)
            if self.fade <= 0.0:
                self.fade_dir = 0
        self.message_t = max(0.0, self.message_t - dt)
        if getattr(self, "chat_log", None):
            alive = []
            for row in self.chat_log:
                row[2] -= dt
                if row[2] > 0:
                    alive.append(row)
            self.chat_log = alive
        self.gain_t = max(0.0, self.gain_t - dt)
        self.update_multiplayer(dt)
        # Autosave coins / items / upgrades (singleplayer + multiplayer)
        self.autosave_t += dt
        if self.autosave_t >= AUTOSAVE_INTERVAL:
            self.autosave_t = 0.0
            if self.autosave_dirty or True:
                self.save_progress(quiet=True)
        # Creative: infinite fuel & ammo top-up
        if self.game_mode == "creative":
            self.fuel = self.max_fuel
            if self.has_launcher:
                self.rocket_ammo = max(self.rocket_ammo, self.max_ammo)
            if getattr(self, "has_tnt", False):
                self.tnt_ammo = max(self.tnt_ammo, self.max_tnt)
        # Weather cycle + particles (main scene only)
        self._update_weather(dt)
        # Vault interest
        if self.upg.get("coin_interest", 0) and not self.in_menu:
            if not hasattr(self, "_interest_t"):
                self._interest_t = 0.0
            self._interest_t += dt
            if self._interest_t >= 30.0:
                self._interest_t = 0.0
                gain = int(self.coins * 0.01 * self.upg["coin_interest"])
                if gain > 0:
                    self.add_coins(gain)
                    self.say(f"Vault interest +{gain}", 1.5)
        if self.paused or self.map_open or self.mp_open:
            pygame.mouse.get_rel()
            return
        keys = pygame.key.get_pressed()
        buttons = pygame.mouse.get_pressed()
        if self.shop_open or self.fade_dir == 1:
            pygame.mouse.get_rel()
        else:
            dx, dy = pygame.mouse.get_rel()
            self.yaw += dx * SENS
            self.pitch = max(-1.5, min(1.5, self.pitch - dy * SENS))
        lmb = bool(buttons[0]) and not self.shop_open and self.fade_dir != 1
        self.move_player(dt, keys)
        sc = self.scene
        self.reveal_map()
        if self.fade_dir == 0:
            c = (self.pos[0], self.pos[1] + 0.9, self.pos[2])
            for p in sc.portals:
                b = p["box"]
                if b[0] <= c[0] <= b[3] and b[1] <= c[1] <= b[4] and b[2] <= c[2] <= b[5]:
                    self.teleport(p["target"], p["key"])
                    break
            if sc.name == "main" and keys[pygame.K_l]:
                self.l_hold += dt
                if self.l_hold >= 1.2:
                    self.l_hold = 0.0
                    self.teleport("lobby", "default")
            else:
                self.l_hold = 0.0
        self.update_monsters(dt)
        self.update_tools(dt, lmb)
        self.update_rockets(dt)
        self.update_tnts(dt)
        if self.col_box is not None:
            self.col_timer += dt
            if self.col_timer >= 0.4:
                self.col_timer = 0.0
                self.run_collapse()
        self.debris.update(dt, sc.world)
        self.shake = max(0.0, self.shake - dt * 2.5)

    # -------------------------------------------------------------- rendering
    def project(self, p):
        v = self.vp @ np.array([p[0], p[1], p[2], 1.0])
        if v[3] <= 0.05:
            return None
        return ((v[0] / v[3]) * 0.5 + 0.5) * HUD_W, (1 - ((v[1] / v[3]) * 0.5 + 0.5)) * HUD_H

    def t(self, txt, x, y, color=(255, 255, 255), font=None, center=False):
        font = font or self.font
        key = (txt, id(font), color)
        pair = self.tcache.get(key)
        if pair is None:
            if len(self.tcache) > 500:
                self.tcache.clear()
            pair = (font.render(txt, True, color[:3]), font.render(txt, True, (0, 0, 0)))
            self.tcache[key] = pair
        s, sh = pair
        if center:
            x -= s.get_width() // 2
        self.hud.blit(sh, (x + 1, y + 1))
        self.hud.blit(s, (x, y))
        return s.get_width()

    def mouse_hud(self):
        """Map window mouse position into fixed HUD coordinates (stretched HUD)."""
        mx, my = pygame.mouse.get_pos()
        return mx * HUD_W / max(1, self.win_w), my * HUD_H / max(1, self.win_h)

    def present_hud(self):
        """Normal HUD: stretch to the full window (gameplay / menus)."""
        ctx = self.ctx
        raw = getattr(pygame.image, "tobytes", None) or pygame.image.tostring
        self.hud_tex.write(raw(self.hud, "RGBA", True))
        ctx.disable(moderngl.DEPTH_TEST)
        ctx.enable(moderngl.BLEND)
        self.hud_tex.use(0)
        ctx.viewport = (0, 0, max(1, self.win_w), max(1, self.win_h))
        self.overlay_vao.render(moderngl.TRIANGLE_STRIP)

    def present_loading_hud(self):
        """Loading screen ONLY: letterbox + center (does not affect other screens)."""
        ctx = self.ctx
        raw = getattr(pygame.image, "tobytes", None) or pygame.image.tostring
        self.hud_tex.write(raw(self.hud, "RGBA", True))
        ctx.disable(moderngl.DEPTH_TEST)
        ctx.enable(moderngl.BLEND)
        self.hud_tex.use(0)
        win_w, win_h = max(1, self.win_w), max(1, self.win_h)
        hud_aspect = HUD_W / float(HUD_H)
        win_aspect = win_w / float(win_h)
        if win_aspect > hud_aspect:
            vh = win_h
            vw = max(1, int(vh * hud_aspect))
            vx = (win_w - vw) // 2
            vy = 0
        else:
            vw = win_w
            vh = max(1, int(vw / hud_aspect))
            vx = 0
            vy = (win_h - vh) // 2
        ctx.viewport = (0, 0, win_w, win_h)
        ctx.clear(0.03, 0.04, 0.08)
        ctx.viewport = (vx, vy, vw, vh)
        self.overlay_vao.render(moderngl.TRIANGLE_STRIP)
        ctx.viewport = (0, 0, win_w, win_h)

    def loading(self, text, frac):
        # Keep window size in sync while loading so centering uses real dimensions
        for e in pygame.event.get():
            if e.type == pygame.VIDEORESIZE:
                self.win_w, self.win_h = max(e.w, 320), max(e.h, 240)
            elif e.type == pygame.QUIT:
                raise SystemExit(0)
        pygame.event.pump()
        self.ctx.viewport = (0, 0, self.win_w, self.win_h)
        self.ctx.clear(0.03, 0.04, 0.08)
        self.hud.fill((0, 0, 0, 0))
        cx, cy = HUD_W // 2, HUD_H // 2
        tnow = time.time()

        # Soft radial vignette
        for i, a in enumerate((40, 28, 16, 8)):
            r = 420 - i * 70
            pygame.draw.circle(self.hud, (8, 12, 22, a), (cx, cy), r)

        # Animated orbiting voxel cubes (cheap procedural interest)
        if not hasattr(self, "_load_bits"):
            rng = random.Random(7)
            self._load_bits = [
                (rng.uniform(0, 6.28), rng.uniform(90, 280), rng.uniform(0.4, 1.2),
                 rng.choice([(255, 200, 70), (120, 200, 255), (255, 120, 80), (180, 255, 140)]))
                for _ in range(18)
            ]
        for ang0, rad, spd, col in self._load_bits:
            ang = ang0 + tnow * spd
            x = cx + math.cos(ang) * rad
            y = cy - 40 + math.sin(ang) * rad * 0.55
            s = 6 + int(3 * math.sin(ang * 2 + tnow))
            pygame.draw.rect(self.hud, (*col, 200), (int(x - s / 2), int(y - s / 2), s, s), border_radius=2)

        # Title + subtitle
        self.t("DESTRUX", cx, cy - 150, (255, 215, 70), self.font_title, True)
        self.t("voxel destruction sandbox", cx, cy - 100, (140, 160, 190), self.font_small, True)

        # Current step
        self.t(text + "...", cx, cy - 20, (220, 230, 245), self.font, True)

        # Progress bar (centered)
        bar_w, bar_h = 420, 14
        bx, by = cx - bar_w // 2, cy + 20
        pygame.draw.rect(self.hud, (30, 36, 52, 255), (bx, by, bar_w, bar_h), border_radius=7)
        fill = int(bar_w * min(1.0, max(0.0, frac)))
        if fill > 0:
            pygame.draw.rect(self.hud, (255, 190, 50, 255), (bx, by, fill, bar_h), border_radius=7)
            # Shine
            pygame.draw.rect(self.hud, (255, 240, 180, 90), (bx, by, fill, bar_h // 2), border_radius=7)
        pct = f"{int(min(100, frac * 100))}%"
        self.t(pct, cx, by + 22, (180, 195, 220), self.font_small, True)

        # Rotating tip
        tips = [
            "Tip: Hold L in the Drop Zone to return to the lobby",
            "Tip: Orange pads launch you sky-high — upgrade pad boost",
            "Tip: Gold and crates drop the best loot",
            "Tip: Rockets + collapse physics = big payouts",
            "Tip: Press M for the atlas and region quests",
            "Tip: Jetpack fuel regenerates slowly on the ground",
            "Tip: Multiplayer progress autosaves to the server",
            "Tip: Storms and fog can be resisted with Scout upgrades",
            "Tip: TNT packs a delayed punch — place, run, profit",
        ]
        tip = tips[int(tnow * 0.35) % len(tips)]
        self.t(tip, cx, cy + 90, (130, 155, 180), self.font_small, True)

        # Footer
        self.t("Building world — please wait", cx, HUD_H - 40, (90, 110, 140), self.font_small, True)

        self.present_loading_hud()
        pygame.display.flip()

    def draw_mp(self):
        s = self.hud
        dim = pygame.Surface((HUD_W, HUD_H), pygame.SRCALPHA)
        dim.fill((3, 5, 10, 220))
        s.blit(dim, (0, 0))
        panel = pygame.Rect(80, 50, HUD_W - 160, HUD_H - 100)
        pygame.draw.rect(s, (14, 18, 30), panel, border_radius=14)
        pygame.draw.rect(s, (70, 110, 180), panel, 2, border_radius=14)
        self.t("SERVER BROWSER", panel.x + 24, panel.y + 16, (255, 230, 150), self.font_big)
        self.t(self.mp_status[:70], panel.x + 24, panel.y + 52, (160, 180, 210), self.font_small)
        online = "HOSTING" if self.is_host else ("ONLINE" if self.net_client.connected else "OFFLINE")
        ocol = (120, 255, 160) if online != "OFFLINE" else (255, 120, 120)
        self.t(online, panel.right - 140, panel.y + 20, ocol, self.font)

        self.mp_rects = []
        # server rows
        y = panel.y + 90
        for i, srv in enumerate(self.mp_servers):
            row = pygame.Rect(panel.x + 20, y, panel.w - 40, 64)
            sel = i == self.mp_selected
            pygame.draw.rect(s, (40, 70, 120) if sel else (22, 28, 42), row, border_radius=8)
            pygame.draw.rect(s, (100, 180, 255) if sel else (50, 60, 80), row, 1, border_radius=8)
            self.mp_rects.append((("sel", i), row))
            tag = "OFFICIAL" if srv.get("official") else "LOCAL"
            self.t(srv["name"], row.x + 14, row.y + 8, (255, 255, 255), self.font)
            self.t(f"{srv['host']}:{srv['port']}  ·  {tag}", row.x + 14, row.y + 32,
                   (150, 170, 200), self.font_small)
            self.t(srv.get("motd", "")[:48], row.right - 360, row.y + 20, (130, 150, 170), self.font_small)
            y += 72

        # players online
        self.t(f"Players visible: {len(self.mp_remote) + (1 if (self.is_host or self.net_client.connected) else 0)}",
               panel.x + 24, y + 8, (180, 200, 220), self.font_small)
        py = y + 32
        for p in list(self.mp_remote.values())[:6]:
            self.t(f"• {p.get('name', '?')}  [{p.get('scene', '?')}]", panel.x + 30, py,
                   (200, 210, 230), self.font_small)
            py += 20

        # buttons
        bw, bh = 200, 44
        by = panel.bottom - 70
        buttons = [
            ("join_sel", "JOIN SELECTED", (30, 120, 90)),
            ("host", "HOST SERVER", (30, 90, 140)),
            ("stop", "DISCONNECT", (120, 50, 50)),
            ("refresh", "REFRESH", (60, 60, 80)),
        ]
        bx = panel.x + 24
        for action, label, col in buttons:
            r = pygame.Rect(bx, by, bw, bh)
            self.mp_rects.append((action, r))
            pygame.draw.rect(s, col, r, border_radius=8)
            self.t(label, r.centerx, r.y + 12, (255, 255, 255), self.font_small, True)
            bx += bw + 14
        self.t("E / ESC close  ·  Host keeps this game running as the 24/7 server",
               panel.x + 24, panel.bottom - 24, (120, 140, 160), self.font_small)

    def draw_hud(self):
        s, sc = self.hud, self.scene
        s.fill((0, 0, 0, 0))
        if self.in_menu:
            self.draw_menu()
            return
        if self.map_open:
            self.draw_map()
            return
        if self.mp_open:
            self.draw_mp()
            return
        if getattr(self, "settings_open", False):
            self.draw_settings()
            return
        # F1: pure world view (no crosshair, coins, hints, weather, etc.)
        if getattr(self, "hide_hud", False):
            return
        cx, cy = HUD_W // 2, HUD_H // 2
        if not self.shop_open and not self.paused:
            white = (255, 255, 255, 220)
            for a, b in (((cx - 10, cy), (cx - 4, cy)), ((cx + 4, cy), (cx + 10, cy)),
                         ((cx, cy - 10), (cx, cy - 4)), ((cx, cy + 4), (cx, cy + 10))):
                pygame.draw.line(s, white, a, b, 2)
            if self.progress > 0:
                pygame.draw.rect(s, (0, 0, 0, 150), (cx - 30, cy + 22, 60, 8))
                pygame.draw.rect(s, (255, 210, 60, 255),
                                 (cx - 30, cy + 22, int(60 * min(1, self.progress)), 8))

        self.t(f"DCOIN  {int(self.coins)}", 16, 12, (255, 215, 70), self.font_big)
        if self.gain_t > 0:
            self.t(self.gain_text, 16, 48, (255, 240, 150))
        self.t(sc.title, HUD_W - 16 - self.font.size(sc.title)[0], 12, (230, 235, 245))
        fps = f"{int(self.fps)} fps"
        self.t(fps, HUD_W - 16 - self.font_small.size(fps)[0], 38, (160, 170, 185), self.font_small)
        self.t(self.game_mode.upper(), 16, 70, (140, 200, 255), self.font_small)
        if getattr(self, "spectator", False):
            self.t("SPECTATOR", 16, 90, (255, 120, 255), self.font_big)
        elif getattr(self, "is_op", False):
            self.t("OP", 16, 90, (255, 200, 80), self.font_small)
        pack = getattr(self, "shader_pack", "default")
        if pack != "default":
            self.t(f"SHADER {pack.upper()}", 16, 110 if (self.spectator or self.is_op) else 90,
                   (180, 200, 255), self.font_small)
        if sc.name == "main" and self.weather != "clear":
            wcol = {
                "rain": (120, 180, 255), "snow": (220, 230, 255), "storm": (160, 140, 200),
                "hail": (200, 210, 230), "fog": (180, 190, 200), "light": (255, 240, 180),
            }.get(self.weather, (200, 200, 200))
            self.t(f"WEATHER  {self.weather.upper()}", 16, 92, wcol, self.font_small)

        if sc.world.destructible:
            if self.tool == 0:
                tool = "Pickaxe Lv%d" % self.pick_lvl
            elif self.tool == 1:
                tool = "Rocket launcher  x%d" % self.rocket_ammo
            elif self.tool == 2:
                tool = "TNT  x%d" % getattr(self, "tnt_ammo", 0)
            else:
                tool = "Tool"
        else:
            tool = "Tools disabled here"
        self.t(tool, 16, HUD_H - 74, (255, 255, 255))
        if self.has_jet:
            frac = self.fuel / self.max_fuel
            pygame.draw.rect(s, (0, 0, 0, 150), (16, HUD_H - 44, 180, 12))
            pygame.draw.rect(s, (255, 140, 40, 255), (16, HUD_H - 44, int(180 * frac), 12))
            self.t("JET", 204, HUD_H - 50, (255, 190, 120), self.font_small)
        hint = "WASD move | SPACE jump/jet | LMB mine | 1/2 tools | M map | E shop | T chat (MP)"
        self.t(hint, HUD_W - 16 - self.font_small.size(hint)[0], HUD_H - 26,
               (225, 228, 235), self.font_small)

        for p in sc.portals:
            pt = self.project(p["label_pos"])
            if pt and 0 < pt[0] < HUD_W and 0 < pt[1] < HUD_H:
                self.t(p["label"], int(pt[0]), int(pt[1]), (255, 235, 170), self.font_big, True)
        if self.near_terminal() and not self.shop_open and not self.mp_open:
            self.t("[E]  Open shop", cx, HUD_H - 150, (255, 235, 170), self.font_big, True)
        if self.near_server_pc() and not self.mp_open and not self.shop_open:
            self.t("[E]  Multiplayer PC — Server Browser", cx, HUD_H - 150, (120, 200, 255), self.font_big, True)
        if self.is_host or self.net_client.connected:
            n = len(self.mp_remote) + 1
            self.t(f"MP {n}p", HUD_W - 90, 60, (100, 255, 160), self.font_small)
        if sc.name == "lobby" and self.time < 12:
            self.t("Walk into a portal  -  DROP ZONE to dig and blow things up, SHOP to upgrade",
                   cx, 90, (230, 235, 245), self.font, True)
        if sc.name == "main":
            bi = self.current_biome()
            if bi is not None:
                state = self.quest_states[bi]
                title = sc.biome_names[bi]
                pygame.draw.rect(s, (8, 13, 23, 205), (HUD_W - 390, 76, 374, 108), border_radius=8)
                self.t("CURRENT REGION", HUD_W - 372, 86, (112, 211, 255), self.font_small)
                self.t(title.upper(), HUD_W - 372, 107, (255, 255, 255), self.font)
                quest_line = ("QUEST COMPLETE" if state["complete"] else
                              (f"MONSTERS DEFEATED  {state['kills']}/3" if state["started"]
                               else "QUEST AVAILABLE  -  PRESS M TO ACCEPT"))
                self.t(quest_line, HUD_W - 372, 139,
                       (120, 255, 164) if state["complete"] else (255, 205, 112), self.font_small)
                if not state["complete"]:
                    detail = self.quest_info(bi) if state["started"] else "No monsters will attack until you start it."
                    self.t(detail[:48], HUD_W - 372, 159, (205, 214, 230), self.font_small)
            if self.l_hold > 0:
                pygame.draw.rect(s, (0, 0, 0, 150), (cx - 80, HUD_H - 190, 160, 10))
                pygame.draw.rect(s, (120, 220, 255, 255),
                                 (cx - 80, HUD_H - 190, int(160 * self.l_hold / 1.2), 10))
                self.t("Returning to lobby", cx, HUD_H - 214, (200, 235, 255), self.font, True)
            elif self.time < 20:
                self.t("Hold L to return to the lobby", cx, 90, (230, 235, 245), self.font, True)
        # Multiplayer chat log (above the one-shot message)
        if getattr(self, "chat_log", None):
            y = HUD_H - 150 - len(self.chat_log) * 20
            for name, msg, ttl in self.chat_log:
                a = min(255, int(255 * min(1.0, ttl / 2.0)))
                col = (220, 230, 245) if a > 200 else (160, 175, 195)
                self.t(f"{name}: {msg}", 16, y, col, self.font_small)
                y += 20
        if getattr(self, "chat_open", False):
            bar = pygame.Rect(40, HUD_H - 70, HUD_W - 80, 36)
            pygame.draw.rect(s, (10, 14, 24, 230), bar, border_radius=8)
            pygame.draw.rect(s, (80, 140, 220), bar, 2, border_radius=8)
            caret = "_" if int(self.time * 2) % 2 == 0 else " "
            self.t("> " + self.chat_buf + caret, bar.x + 10, bar.y + 8, (255, 255, 255), self.font)
            self.t("Enter send  ·  Esc cancel", bar.right - 200, bar.y + 10, (140, 160, 180), self.font_small)
        if self.message_t > 0:
            self.t(self.message, cx, HUD_H - 120, (255, 255, 255), self.font, True)

        self.shop_item_rects = []
        self.shop_buy_rects = []
        if self.shop_open:
            catalog = self.shop_catalog()
            # Full-screen dim
            dim = pygame.Surface((HUD_W, HUD_H), pygame.SRCALPHA)
            dim.fill((4, 6, 12, 210))
            s.blit(dim, (0, 0))
            # Main panel
            panel = pygame.Rect(40, 36, HUD_W - 80, HUD_H - 72)
            pygame.draw.rect(s, (14, 18, 28), panel, border_radius=14)
            pygame.draw.rect(s, (70, 95, 140), panel, 2, border_radius=14)
            # Header bar
            header = pygame.Rect(panel.x, panel.y, panel.w, 58)
            pygame.draw.rect(s, (22, 30, 48), header, border_top_left_radius=14, border_top_right_radius=14)
            self.t("BLACK-MARKET TERMINAL", panel.x + 24, panel.y + 14, (255, 230, 160), self.font_big)
            self.t(f"DCOIN  {int(self.coins)}", panel.right - 220, panel.y + 18, (255, 215, 70), self.font_big)
            free = self.game_mode == "creative"
            if free:
                self.t("CREATIVE — FREE", panel.right - 220, panel.y + 44, (120, 255, 180), self.font_small)

            # Left column – item list
            left = pygame.Rect(panel.x + 16, panel.y + 70, 320, panel.h - 100)
            pygame.draw.rect(s, (10, 14, 22), left, border_radius=10)
            pygame.draw.rect(s, (50, 65, 95), left, 1, border_radius=10)
            self.t("CATALOG", left.x + 14, left.y + 10, (140, 170, 210), self.font_small)
            hx, hy = self.mouse_hud()
            row_h = 52
            for i, item in enumerate(catalog):
                rect = pygame.Rect(left.x + 8, left.y + 32 + i * row_h, left.w - 16, row_h - 6)
                self.shop_item_rects.append(rect)
                selected = i == self.shop_item
                hover = rect.collidepoint(hx, hy)
                bg = (45, 70, 110) if selected else ((32, 42, 62) if hover else (18, 24, 36))
                pygame.draw.rect(s, bg, rect, border_radius=8)
                if selected:
                    pygame.draw.rect(s, item["color"], rect, 2, border_radius=8)
                # color chip
                pygame.draw.circle(s, item["color"], (rect.x + 18, rect.centery), 7)
                self.t(item["name"], rect.x + 36, rect.y + 8, (245, 248, 255), self.font)
                self.t(item["tag"], rect.x + 36, rect.y + 28, (150, 165, 185), self.font_small)

            # Right panel – selected item + upgrades
            right = pygame.Rect(left.right + 14, left.y, panel.right - left.right - 30, left.h)
            pygame.draw.rect(s, (12, 16, 26), right, border_radius=10)
            pygame.draw.rect(s, (55, 75, 110), right, 1, border_radius=10)
            item = catalog[self.shop_item]
            self.t(item["name"].upper(), right.x + 20, right.y + 12, item["color"], self.font_big)
            self.t(item["blurb"], right.x + 20, right.y + 46, (180, 195, 215), self.font_small)
            # separator
            pygame.draw.line(s, (40, 55, 80), (right.x + 16, right.y + 72), (right.right - 16, right.y + 72), 1)
            self.t("UPGRADES — click to purchase", right.x + 20, right.y + 80, (130, 155, 185), self.font_small)

            uy = right.y + 104
            visible = item["upgrades"]
            for ui, u in enumerate(visible):
                if uy + 58 > right.bottom - 12:
                    break
                state, cost = self._upg_state(item, u)
                lvl = self._upg_level(u["key"])
                if u["key"] == "rocket_ammo_pack":
                    lvl_txt = f"{self.rocket_ammo}/{self.max_ammo}"
                else:
                    lvl_txt = f"Lv {lvl}/{u['max']}"
                row = pygame.Rect(right.x + 12, uy, right.w - 24, 54)
                can = state == "buy"
                hover = row.collidepoint(hx, hy) and can
                pygame.draw.rect(s, (40, 60, 90) if hover else (22, 28, 42), row, border_radius=8)
                border_c = (90, 200, 140) if can else ((70, 80, 100) if state == "max" else (50, 55, 70))
                pygame.draw.rect(s, border_c, row, 1, border_radius=8)
                self.t(u["name"], row.x + 12, row.y + 6, (240, 245, 255) if can else (150, 155, 165))
                self.t(u["desc"][:58], row.x + 12, row.y + 28, (150, 165, 185), self.font_small)
                self.t(lvl_txt, row.right - 200, row.y + 8, (180, 190, 210), self.font_small)
                if state == "buy":
                    ok = free or self.coins >= cost
                    label = "FREE" if free else f"{cost}"
                    buy_r = pygame.Rect(row.right - 110, row.y + 10, 96, 34)
                    self.shop_buy_rects.append((ui, buy_r))
                    pygame.draw.rect(s, (36, 120, 80) if ok else (90, 40, 45), buy_r, border_radius=6)
                    self.t(label, buy_r.centerx, buy_r.y + 7,
                           (255, 255, 255) if ok else (255, 160, 160), self.font_small, True)
                elif state == "max":
                    self.t("MAX", row.right - 70, row.y + 16, (120, 200, 150), self.font_small)
                else:
                    self.t("LOCKED", row.right - 90, row.y + 16, (120, 110, 110), self.font_small)
                uy += 60

            self.t("E / ESC close   ·   click item then upgrade   ·   scroll wheel",
                   panel.x + 24, panel.bottom - 28, (130, 145, 165), self.font_small)

        if self.paused:
            pygame.draw.rect(s, (0, 0, 0, 170), (0, 0, HUD_W, HUD_H))
            self.t("PAUSED", cx, 240, (255, 255, 255), self.font_title, True)
            self.t("ESC  resume", cx, 320, (220, 225, 235), self.font, True)
            if sc.name != "lobby":
                self.t("L    back to lobby", cx, 352, (220, 225, 235), self.font, True)
            self.t("S    save game   ·   O    load game", cx, 384, (180, 255, 180), self.font, True)
            self.t("F5 save  ·  F9 load", cx, 416, (150, 180, 150), self.font_small, True)
            self.t("M    main menu", cx, 448, (220, 225, 235), self.font, True)
            self.t("Q    quit", cx, 480, (220, 225, 235), self.font, True)
            self.t("N    toggle music", cx, 512, (180, 190, 210), self.font_small, True)
        if self.fade > 0:
            pygame.draw.rect(s, (0, 0, 0, int(255 * min(1.0, self.fade))), (0, 0, HUD_W, HUD_H))

    def render(self):
        ctx, sc = self.ctx, self.scene
        base_env = sc.env
        env = self._weather_env_override(base_env) if sc.name == "main" and not self.in_menu else base_env
        ctx.viewport = (0, 0, self.win_w, self.win_h)
        if self.in_menu:
            ctx.clear(0.04, 0.05, 0.09, 1.0)
            self.draw_hud()
            self.present_hud()
            return
        ctx.clear(env["hor"][0], env["hor"][1], env["hor"][2], 1.0)
        aspect = self.win_w / self.win_h
        shake = self.shake
        yaw = self.yaw + random.uniform(-1, 1) * 0.02 * shake
        pitch = self.pitch + random.uniform(-1, 1) * 0.02 * shake
        cp = math.cos(pitch)
        fwd = np.array([math.sin(yaw) * cp, math.sin(pitch), -math.cos(yaw) * cp])
        eye = self.eye()
        right = np.cross(fwd, (0, 1, 0))
        right /= np.linalg.norm(right)
        up = np.cross(right, fwd)
        ty = math.tan(math.radians(FOV) / 2)

        # sky
        ctx.disable(moderngl.DEPTH_TEST)
        ctx.disable(moderngl.BLEND)
        if self.shader_effects["sky"]:
            sp = self.sprog
            for name, val in (("u_fwd", fwd), ("u_right", right), ("u_up", up)):
                setu(sp, name, tuple(float(v) for v in val))
            setu(sp, "u_sun", env["sun"])
            setu(sp, "u_tan", (ty * aspect, ty))
            setu(sp, "u_top", env["top"])
            setu(sp, "u_hor", env["hor"])
            setu(sp, "u_stars", env.get("stars", 0.0))
            style = {"retro": 1.0, "noir": 2.0, "sunset": 3.0, "arctic": 4.0}.get(self.shader_pack, 0.0)
            setu(sp, "u_style", style)
            self.sky_vao.render(moderngl.TRIANGLE_STRIP)

        ctx.enable(moderngl.DEPTH_TEST)
        self.vp = perspective(FOV, aspect, 0.05, 400.0) @ look_at(eye, fwd)
        planes = frustum_planes(self.vp)
        vpb = self.vp.T.astype("f4").tobytes()
        q = {"default": 0.0, "pro": 1.0, "insane": 2.0}.get(getattr(self, "shader_pack", "default"), 0.0)
        style = {"retro": 1.0, "noir": 2.0, "sunset": 3.0, "arctic": 4.0}.get(self.shader_pack, 0.0)
        # Insane: hard key light, deep shadows, clearer air — YT / trailer grade
        suncol = env["suncol"]
        amb = env["amb"]
        fogd = env["fogd"]
        if q >= 1.5:
            suncol = tuple(min(1.45, c * 1.28) for c in suncol)
            amb = tuple(c * 0.68 for c in amb)
            fogd = float(fogd) * 0.56
        elif q >= 0.5:
            suncol = tuple(min(1.18, c * 1.10) for c in suncol)
            fogd = float(fogd) * 0.88
        for prog in (self.cprog, self.iprog):
            prog["u_vp"].write(vpb)
            setu(prog, "u_cam", tuple(float(v) for v in eye))
            setu(prog, "u_sun", env["sun"])
            setu(prog, "u_suncol", suncol)
            setu(prog, "u_amb", amb)
            setu(prog, "u_gnd", env["gnd"])
            setu(prog, "u_fogcol", env["hor"])
            setu(prog, "u_fogd", fogd)
            setu(prog, "u_time", self.time)
            setu(prog, "u_quality", q)
            setu(prog, "u_lighting", float(self.shader_effects["lighting"]))
            setu(prog, "u_fog", float(self.shader_effects["fog"]))
            setu(prog, "u_grade", float(self.shader_effects["grade"]))
            setu(prog, "u_emissive", float(self.shader_effects["emissive"]))
            setu(prog, "u_style", style)

        rd = float(getattr(self, "render_distance", VIEW_DIST))
        sc.rend.update(eye, 0.008, rd + 20)
        sc.rend.draw(planes, eye, rd)

        parts = []
        if self.debris.n:
            parts.append(self.debris.instances())
        if self.rockets:
            rd = np.empty((len(self.rockets), 7), np.float32)
            for i, r in enumerate(self.rockets):
                rd[i, 0:3] = r["pos"] - 0.11
                rd[i, 3] = 0.22
                rd[i, 4:7] = (1.9, 0.9, 0.2)
            parts.append(rd)
        if self.tnts:
            td = np.empty((len(self.tnts), 7), np.float32)
            for i, t in enumerate(self.tnts):
                blink = 1.4 + 0.6 * math.sin(self.time * 12 + i)
                td[i, 0:3] = t["pos"] - 0.18
                td[i, 3] = 0.36
                td[i, 4:7] = (blink, 0.15 + 0.1 * (t["fuse"] < 0.6), 0.08)
            parts.append(td)
        # Weather particles (rain streaks / snow flakes / hail / fog wisps)
        if self.weather_particles and sc.name == "main":
            n = min(len(self.weather_particles), 700)
            wd = np.empty((n, 7), np.float32)
            for i, p in enumerate(self.weather_particles[:n]):
                kind = p[7]
                if kind == 0:  # rain
                    wd[i, 0:3] = (p[0] - 0.02, p[1] - 0.25, p[2] - 0.02)
                    wd[i, 3] = 0.04
                    wd[i, 4:7] = (0.45, 0.55, 0.85)
                elif kind == 1:  # snow
                    wd[i, 0:3] = (p[0] - 0.06, p[1] - 0.06, p[2] - 0.06)
                    wd[i, 3] = 0.10
                    wd[i, 4:7] = (0.92, 0.94, 1.0)
                elif kind == 2:  # hail
                    wd[i, 0:3] = (p[0] - 0.05, p[1] - 0.05, p[2] - 0.05)
                    wd[i, 3] = 0.09
                    wd[i, 4:7] = (0.75, 0.80, 0.90)
                else:  # fog
                    wd[i, 0:3] = (p[0] - 0.35, p[1] - 0.15, p[2] - 0.35)
                    wd[i, 3] = 0.7
                    wd[i, 4:7] = (0.55, 0.58, 0.65)
            parts.append(wd)
        # Other multiplayer players (same scene only)
        remotes = [p for p in self.mp_remote.values()
                   if p.get("scene") == sc.name and
                   np.linalg.norm(np.array([p.get("x", 0), p.get("y", 0), p.get("z", 0)]) - eye) < float(getattr(self, "render_distance", VIEW_DIST))]
        if remotes:
            pd = np.empty((len(remotes) * 3, 7), np.float32)
            for i, p in enumerate(remotes):
                base = i * 3
                pp = np.array([p.get("x", 0), p.get("y", 0), p.get("z", 0)], float)
                # body
                pd[base, 0:3] = pp + np.array([-0.25, 0.1, -0.25])
                pd[base, 3] = 0.5
                pd[base, 4:7] = (0.2, 0.75, 1.3)
                # head
                pd[base + 1, 0:3] = pp + np.array([-0.18, 1.15, -0.18])
                pd[base + 1, 3] = 0.36
                pd[base + 1, 4:7] = (1.0, 0.85, 0.65)
                # tool hint
                pd[base + 2, 0:3] = pp + np.array([0.2, 0.9, -0.1])
                pd[base + 2, 3] = 0.18
                pd[base + 2, 4:7] = (1.5, 1.2, 0.3) if p.get("tool") else (0.6, 0.6, 0.7)
            parts.append(pd)
        if sc.name == "main":
            live = [m for m in self.monsters if m["alive"] and m.get("active", False) and
                    np.linalg.norm(m["pos"] - eye) < float(getattr(self, "render_distance", VIEW_DIST))]
            if live:
                md = np.empty((len(live) * 4, 7), np.float32)
                for i, m in enumerate(live):
                    bob = math.sin(m["phase"]) * 0.10
                    base = i * 4
                    # Chunky body, skull, and two glowing eyes make each mob
                    # readable even at a distance in the voxel art style.
                    pieces = [
                        (m["pos"] + np.array([-0.36, bob - 0.36, -0.36]), 0.72,
                         (1.25, 0.12 + 0.08 * (m["hp"] / 3), 0.08)),
                        (m["pos"] + np.array([-0.24, bob + 0.27, -0.24]), 0.48,
                         (0.72, 0.035, 0.025)),
                        (m["pos"] + np.array([-0.19, bob + 0.40, -0.255]), 0.075,
                         (2.0, 1.25, 0.12)),
                        (m["pos"] + np.array([0.115, bob + 0.40, -0.255]), 0.075,
                         (2.0, 1.25, 0.12)),
                    ]
                    for j, (pos, scale, color) in enumerate(pieces):
                        md[base + j, 0:3] = pos
                        md[base + j, 3] = scale
                        md[base + j, 4:7] = color
                parts.append(md)
        if self.tool == 0 and self.target is not None:
            m = np.empty((1, 7), np.float32)
            sz = 0.12 + 0.04 * math.sin(self.time * 10)
            m[0, 0:3] = (np.array(self.target) + 0.5) * VS - sz / 2
            m[0, 3] = sz
            g = 1.2 + 0.6 * min(1.0, self.progress)
            m[0, 4:7] = (g, g * (1.0 - 0.5 * min(1.0, self.progress)), g * (1.0 - min(1.0, self.progress)) + 0.01)
            parts.append(m)

        # First-person held tool models (pickaxe / rocket launcher)
        if not self.shop_open and not self.paused and not self.map_open and self.fade < 0.5:
            bob = math.sin(self.time * (10.0 if np.linalg.norm(self.vel[[0, 2]]) > 0.8 else 2.0)) * 0.012
            swing = 0.0
            if self.tool == 0 and self.progress > 0:
                swing = math.sin(self.progress * math.pi) * 0.35
            # Camera basis
            right = np.cross(fwd, (0.0, 1.0, 0.0))
            rn = np.linalg.norm(right)
            if rn > 1e-5:
                right = right / rn
            up = np.cross(right, fwd)
            # Hand anchor slightly in front, down, and to the right of the eye
            hand = eye + fwd * 0.55 + right * 0.28 + up * (-0.22 + bob)

            def cube_at(offset, scale, color):
                # offset is in (right, up, forward) hand space
                pos = hand + right * offset[0] + up * offset[1] + fwd * offset[2]
                return (pos - scale * 0.5, scale, color)

            tool_pieces = []
            if self.tool == 0:
                # Pickaxe: wooden handle + metal head
                wood = (0.55, 0.35, 0.15)
                metal = (0.75, 0.78, 0.85)
                # handle along up axis, tilted by swing
                for i in range(6):
                    t = i / 5.0
                    tool_pieces.append(cube_at(
                        (0.0 + swing * 0.15, -0.08 + t * 0.32, 0.02 - swing * 0.1),
                        0.045, wood))
                # head
                tool_pieces.append(cube_at((0.0 + swing * 0.15, 0.26, 0.02 - swing * 0.1), 0.10, metal))
                tool_pieces.append(cube_at((-0.09 + swing * 0.15, 0.26, 0.02 - swing * 0.1), 0.07, metal))
                tool_pieces.append(cube_at((0.09 + swing * 0.15, 0.26, 0.02 - swing * 0.1), 0.07, metal))
                # pointy tips
                tool_pieces.append(cube_at((-0.14 + swing * 0.15, 0.26, 0.02 - swing * 0.1), 0.05, (0.9, 0.9, 0.95)))
                tool_pieces.append(cube_at((0.14 + swing * 0.15, 0.26, 0.02 - swing * 0.1), 0.05, (0.9, 0.9, 0.95)))
            else:
                # Rocket launcher: tube + body + grip
                body = (0.25, 0.28, 0.32)
                tube = (0.45, 0.48, 0.52)
                accent = (0.85, 0.25, 0.12)
                glow = (1.6, 0.9, 0.25)
                for i in range(5):
                    tool_pieces.append(cube_at((-0.02, -0.02, 0.05 + i * 0.07), 0.09, tube))
                tool_pieces.append(cube_at((-0.02, -0.02, 0.40), 0.11, body))
                tool_pieces.append(cube_at((-0.02, -0.10, 0.18), 0.06, body))  # grip
                tool_pieces.append(cube_at((-0.02, 0.05, 0.22), 0.05, accent))
                tool_pieces.append(cube_at((-0.02, -0.02, 0.0), 0.08, glow))  # muzzle glow
                # ammo count dots
                for i in range(min(5, self.rocket_ammo)):
                    tool_pieces.append(cube_at((-0.02, 0.06, 0.30 + i * 0.03), 0.03, (0.3, 1.0, 0.4)))

            if tool_pieces:
                td = np.empty((len(tool_pieces), 7), np.float32)
                for i, (pos, scale, color) in enumerate(tool_pieces):
                    td[i, 0:3] = pos
                    td[i, 3] = scale
                    td[i, 4:7] = color
                parts.append(td)

        if parts:
            data = np.concatenate(parts)[:MAX_DEBRIS + 80]
            self.dyn_vbo.write(data.tobytes())
            self.dyn_vao.render(moderngl.TRIANGLES, instances=len(data))

        self.draw_hud()
        self.present_hud()

    # ------------------------------------------------------------------- loop
    def run(self):
        clock = pygame.time.Clock()
        # Stay on main menu until the player hits PLAY
        self.in_menu = True
        self.scene = self.scenes["lobby"]
        self.pos = np.array(self.scene.spawns["default"][0], float)
        self.fade, self.fade_dir = 0.0, 0
        pygame.mouse.set_visible(True)
        pygame.event.set_grab(False)
        while True:
            dt = min(clock.tick(144) / 1000.0, 0.05)
            self.fps = 0.92 * self.fps + 0.08 * (1.0 / max(dt, 1e-4))
            if not self.handle_events():
                break
            self.update(dt)
            self.render()
            pygame.display.flip()
        try:
            if not self.in_menu:
                self.save_progress(quiet=True)
        except Exception:
            pass
        pygame.quit()


if __name__ == "__main__":
    # Console stays open in frozen builds (PyInstaller console=True) so update logs are visible.
    print("=" * 60)
    print("  DESTRUX CLIENT BUILD 9  —  F1=hide HUD  F10=OP money  F3=settings")
    print("  If F1 still gives coins, you are NOT running this file.")
    print("=" * 60)
    print("DESTRUX client starting...")
    print(f"  frozen={_frozen()}  app_dir={_app_dir()}")
    print(f"  file={_client_file_path()}")
    live_path = _client_file_path()
    try:
        here = os.path.normcase(os.path.abspath(__file__))
    except Exception:
        here = ""
    is_live = here == os.path.normcase(os.path.abspath(live_path))

    if _frozen() and not is_live:
        # Bootstrap from the .exe: seed payload, auto-update, then run live script.
        _seed_live_payload()
        check_for_game_update()
        if not os.path.isfile(live_path):
            print(f"[boot] missing {live_path} — cannot start")
            sys.exit(1)
        print(f"[boot] running live payload: {live_path}")
        import runpy
        runpy.run_path(live_path, run_name="__main__")
    else:
        # Source install: always check. (A frozen live payload was already
        # checked by the bootstrap branch above, so skip it there.)
        if not _frozen() and "--no-update" not in sys.argv:
            check_for_game_update()
        Game().run()
    sys.exit(0)
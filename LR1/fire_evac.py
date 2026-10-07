import argparse
import csv
import gc
import math
import os
import random
import time

import numpy as np
from numba import njit, prange

# константы
FLOOR_CONCRETE, FLOOR_WOOD, FLOOR_CARPET, WALL, EXIT = 0, 1, 2, 3, 4
BURNT = 255
UNREACH = 65535

# окрестность Мура (порядок важен: от него зависят веса ветра)
DX = np.array([-1, 0, 1, -1, 1, -1, 0, 1], dtype=np.int64)
DY = np.array([-1, -1, -1, 0, 0, 1, 1, 1], dtype=np.int64)

# вероятность загореться от одного горящего соседа и время горения (тиков)
P_MATERIAL = np.array([0.05, 0.25, 0.15], dtype=np.float64)
BURN_TIME = np.array([15, 60, 35], dtype=np.int64)

WIND_DIRS = {"N": (0, -1), "NE": (1, -1), "E": (1, 0), "SE": (1, 1),
             "S": (0, 1), "SW": (-1, 1), "W": (-1, 0), "NW": (-1, -1)}

# где ставить выходы: (сторона, доля вдоль стороны)
EXIT_TEMPLATES = [("S", .25), ("N", .75), ("W", .75), ("E", .25),
                  ("S", .75), ("N", .25), ("W", .25), ("E", .75)]


# ядра numba
@njit(parallel=True, fastmath=True, cache=True)
def fire_step(terrain, burn, burn_new, smoke, smoke_new,
              p_mat, burn_time, wts, smoke_decay, smoke_src):
    # Один шаг огня и дыма. Возвращает (горит, выгорело)
    H, W = terrain.shape
    nburning = 0
    nburnt = 0
    for y in prange(H):
        for x in range(W):
            tr = terrain[y, x]
            if tr >= 3: # стена или выход - не горят, дыма нет
                burn_new[y, x] = 0
                smoke_new[y, x] = 0
                continue
            b = burn[y, x]
            nv = b
            if b == 0:
                pm = p_mat[tr]
                q = 1.0
                for k in range(8):
                    nx = x + DX[k]
                    ny = y + DY[k]
                    nbv = burn[ny, nx]
                    if nbv == 0 or nbv == 255:
                        continue
                    if DX[k] != 0 and DY[k] != 0:
                        if terrain[y, nx] == 3 and terrain[ny, x] == 3:
                            continue
                    pi = pm * wts[k]
                    if pi > 1.0:
                        pi = 1.0
                    q *= (1.0 - pi)
                if q < 1.0 and np.random.random() > q:
                    nv = 1
            elif b < 255:
                age = b + 1
                if age > burn_time[tr]:
                    nv = 255
                else:
                    nv = age
            burn_new[y, x] = nv
            if nv >= 1 and nv <= 254:
                nburning += 1
            elif nv == 255:
                nburnt += 1

            # дым
            if nv >= 1 and nv <= 254:
                smoke_new[y, x] = smoke_src
            else:
                tot = 4 * np.int64(smoke[y, x])
                cnt = 4
                for k in range(8):
                    nx = x + DX[k]
                    ny = y + DY[k]
                    if terrain[ny, nx] < 3:
                        tot += smoke[ny, nx]
                        cnt += 1
                smoke_new[y, x] = np.uint8(tot * smoke_decay / cnt)
    return nburning, nburnt


@njit(cache=True)
def bfs_dist(terrain, sy, sx, out):
    # Расстояния от клеток выхода
    H, W = terrain.shape
    cnt = 0
    for y in range(H):
        for x in range(W):
            if terrain[y, x] < 3:
                cnt += 1
    q = np.empty(cnt + sy.shape[0] + 1, np.int32)
    head = 0
    tail = 0
    for i in range(sy.shape[0]):
        out[sy[i], sx[i]] = 0
        q[tail] = sy[i] * W + sx[i]
        tail += 1
    while head < tail:
        cur = q[head]
        head += 1
        y = cur // W
        x = cur - y * W
        d = np.int64(out[y, x])
        if d >= 65534:
            continue
        for k in range(8):
            nx = x + DX[k]
            ny = y + DY[k]
            if nx < 0 or ny < 0 or nx >= W or ny >= H:
                continue
            if terrain[ny, nx] >= 3:
                continue
            if out[ny, nx] != 65535:
                continue
            if DX[k] != 0 and DY[k] != 0:
                if terrain[y, nx] == 3 or terrain[ny, x] == 3:
                    continue
            out[ny, nx] = d + 1
            q[tail] = ny * W + nx
            tail += 1


@njit(cache=True)
def place_agents(terrain, burn, occ, n, ax, ay):
    H, W = terrain.shape
    cnt = 0
    tries = 0
    limit = 100 * n + 1000
    while cnt < n and tries < limit:
        tries += 1
        x = np.random.randint(1, W - 1)
        y = np.random.randint(1, H - 1)
        if terrain[y, x] < 3 and occ[y, x] == 0 and burn[y, x] == 0:
            occ[y, x] = 1
            ax[cnt] = x
            ay[cnt] = y
            cnt += 1
    return cnt


@njit(cache=True)
def hazard(burn, smoke, occ, x, y, w_s, w_f, w_c):
    h = smoke[y, x] * (w_s / 255.0)
    for k in range(8):
        nx = x + DX[k]
        ny = y + DY[k]
        b = burn[ny, nx]
        if b != 0 and b != 255:
            h += w_f
        h += occ[ny, nx] * w_c
    return h


@njit(cache=True)
def fire_near(burn, x, y, r):
    H, W = burn.shape
    for yy in range(max(0, y - r), min(H, y + r + 1)):
        for xx in range(max(0, x - r), min(W, x + r + 1)):
            b = burn[yy, xx]
            if b != 0 and b != 255:
                return True
    return False


@njit(cache=True)
def agents_step(t, order, ax, ay, tgt, hp, st, start_t, stuck,
                terrain, burn, smoke, occ, dist, blocked,
                w_s, w_f, w_c, noise, patience, smoke_thr, smoke_dmg):
    # Шаг всех людей в случайном порядке. st: 0 жив, 1 эвакуирован, 2 погиб
    K = dist.shape[0]
    n = order.shape[0]
    for i in range(n - 1, 0, -1): # перемешивание (Фишер-Йейтс)
        j = np.random.randint(0, i + 1)
        tmp = order[i]
        order[i] = order[j]
        order[j] = tmp
    ev = 0
    dead = 0
    for oi in range(n):
        i = order[oi]
        if st[i] != 0:
            continue
        x = ax[i]
        y = ay[i]
        b = burn[y, x]
        if b != 0 and b != 255: # стоит в огне
            st[i] = 2
            occ[y, x] = 0
            dead += 1
            continue
        s = smoke[y, x]
        if s > smoke_thr: # отравление дымом
            hp[i] -= (s - smoke_thr) * smoke_dmg
            if hp[i] <= 0:
                st[i] = 2
                occ[y, x] = 0
                dead += 1
                continue
        if t < start_t[i]: # ещё не отреагировал
            if s < 5:
                continue
            start_t[i] = t

        # выбор выхода
        k = tgt[i]
        if blocked[k] == 1:
            bestk = -1
            bd = 1 << 30
            for e in range(K):
                if blocked[e] == 1:
                    continue
                d = np.int64(dist[e, y, x])
                if d < bd:
                    bd = d
                    bestk = e
            if bestk >= 0:
                tgt[i] = bestk
                k = bestk

        occ[y, x] = 0 # чтобы не считать себя "толпой"
        c0 = np.float64(dist[k, y, x]) + hazard(burn, smoke, occ, x, y, w_s, w_f, w_c)
        bestc = 1e18
        bx = -1
        by = -1
        start = np.random.randint(0, 8)
        for kk in range(8):
            m = (kk + start) & 7
            nx = x + DX[m]
            ny = y + DY[m]
            tn = terrain[ny, nx]
            if tn == 3:
                continue
            if DX[m] != 0 and DY[m] != 0:
                if terrain[y, nx] == 3 or terrain[ny, x] == 3:
                    continue
            if tn != 4 and occ[ny, nx] == 1:
                continue
            bn = burn[ny, nx]
            if bn != 0 and bn != 255:
                continue
            d = dist[k, ny, nx]
            if d == 65535:
                continue
            if tn == 4:
                cn = np.float64(d)
            else:
                cn = np.float64(d) + hazard(burn, smoke, occ, nx, ny, w_s, w_f, w_c) \
                     + noise * np.random.random()
            if cn < bestc:
                bestc = cn
                bx = nx
                by = ny
        moved = False
        if bx >= 0:
            if bestc < c0 - 1e-9:
                moved = True
                stuck[i] = 0
            else:
                stuck[i] += 1
                if stuck[i] > patience: # "тупик": шаг в сторону / смена выхода
                    moved = True
                    stuck[i] = 0
                    if K > 1 and fire_near(burn, x, y, 3):
                        bk = -1
                        bd = 1 << 30
                        for e in range(K):
                            if e == k or blocked[e] == 1:
                                continue
                            d = np.int64(dist[e, y, x])
                            if d < bd:
                                bd = d
                                bk = e
                        if bk >= 0:
                            tgt[i] = bk
        else:
            stuck[i] += 1
        if moved:
            if terrain[by, bx] == 4:
                st[i] = 1
                ev += 1
                continue
            occ[by, bx] = 1
            ax[i] = bx
            ay[i] = by
        else:
            occ[y, x] = 1
    return ev, dead


@njit(parallel=True, cache=True)
def render_kernel(terrain, burn, smoke, occ, out, y0, x0, hh, ww, wall_thr):
    oh = out.shape[0]
    ow = out.shape[1]
    fy = hh / oh
    fx = ww / ow
    for py in prange(oh):
        ya = y0 + int(py * fy)
        yb = max(ya + 1, y0 + int((py + 1) * fy))
        for px in range(ow):
            xa = x0 + int(px * fx)
            xb = max(xa + 1, x0 + int((px + 1) * fx))
            fire = False
            person = False
            ext = False
            walls = 0
            burnt = 0
            smax = 0
            mat = terrain[ya, xa]
            for yy in range(ya, min(yb, terrain.shape[0])):
                for xx in range(xa, min(xb, terrain.shape[1])):
                    tr = terrain[yy, xx]
                    if tr == 3:
                        walls += 1
                        continue
                    if tr == 4:
                        ext = True
                        continue
                    mat = tr
                    bv = burn[yy, xx]
                    if bv != 0 and bv != 255:
                        fire = True
                    elif bv == 255:
                        burnt += 1
                    if occ[yy, xx] == 1:
                        person = True
                    if smoke[yy, xx] > smax:
                        smax = smoke[yy, xx]
            if fire:
                r, g, bl = 255, 90, 0
            elif person:
                r, g, bl = 20, 110, 255
            elif ext:
                r, g, bl = 0, 200, 60
            elif walls >= wall_thr:
                r, g, bl = 25, 25, 25
            else:
                if mat == 1:
                    r, g, bl = 222, 196, 150
                elif mat == 2:
                    r, g, bl = 205, 190, 215
                else:
                    r, g, bl = 235, 235, 235
                if burnt * 2 > (yb - ya) * (xb - xa):
                    r, g, bl = 70, 50, 40
                if smax > 0:
                    a = min(0.85, smax / 255.0 * 1.2)
                    r = int(r * (1 - a) + 110 * a)
                    g = int(g * (1 - a) + 110 * a)
                    bl = int(bl * (1 - a) + 110 * a)
            out[py, px, 0] = r
            out[py, px, 1] = g
            out[py, px, 2] = bl


# здание
def build_building(N, pitch, loops, rng):
    # Сетка CxC комнат (pitch x pitch), стены, двери по остовному дереву + петли
    C = -(-(N - 1) // pitch)
    S = C * pitch + 1
    mats = rng.choice(np.array([0, 1, 2], dtype=np.uint8), size=(C, C),
                      p=[0.40, 0.30, 0.30])
    terrain = np.empty((S, S), dtype=np.uint8)
    for i in range(C):
        terrain[i * pitch:(i + 1) * pitch, :C * pitch] = np.repeat(mats[i], pitch)[None, :]
    terrain[:, C * pitch] = 0
    terrain[C * pitch, :] = 0
    terrain[::pitch, :] = WALL
    terrain[:, ::pitch] = WALL

    dw = max(3, pitch // 6)
    dw = min(dw, pitch - 1)
    parent = list(range(C * C))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    edges = []
    for i in range(C):
        for j in range(C):
            if j + 1 < C:
                edges.append((i, j, 0))
            if i + 1 < C:
                edges.append((i, j, 1))
    random.shuffle(edges)
    for i, j, d in edges:
        a = i * C + j
        b = a + 1 if d == 0 else a + C
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
            carve = True
        else:
            carve = random.random() < loops
        if not carve:
            continue
        off = random.randint(0, pitch - 1 - dw)
        if d == 0:
            x = (j + 1) * pitch
            y0 = i * pitch + 1 + off
            terrain[y0:y0 + dw, x] = FLOOR_WOOD
        else:
            y = (i + 1) * pitch
            x0 = j * pitch + 1 + off
            terrain[y, x0:x0 + dw] = FLOOR_WOOD
    return terrain, C


def place_exits(terrain, C, pitch, n, ew):
    H, W = terrain.shape
    ew = max(3, min(ew, pitch - 2))
    exits = []
    for side, frac in EXIT_TEMPLATES[:n]:
        ri = min(C - 1, int(frac * C))
        c = ri * pitch + pitch // 2
        lo = c - ew // 2
        hi = lo + ew
        if side == "S":
            ys, xs = np.full(ew, H - 1), np.arange(lo, hi)
        elif side == "N":
            ys, xs = np.zeros(ew, dtype=np.int64), np.arange(lo, hi)
        elif side == "W":
            ys, xs = np.arange(lo, hi), np.zeros(ew, dtype=np.int64)
        else:
            ys, xs = np.arange(lo, hi), np.full(ew, W - 1)
        terrain[ys, xs] = EXIT
        exits.append((int(ys.mean()), int(xs.mean()), ys.astype(np.int64), xs.astype(np.int64)))
    return exits, ew


def wind_weights(strength, wdir):
    wx, wy = WIND_DIRS[wdir]
    n = math.hypot(wx, wy)
    wx, wy = wx / n, wy / n
    w = np.empty(8, dtype=np.float64)
    for k in range(8):
        sx, sy = -int(DX[k]), -int(DY[k]) # направление распространения огня
        nn = math.hypot(sx, sy)
        dot = (sx * wx + sy * wy) / nn
        geom = 1.0 if nn == 1.0 else 0.7
        w[k] = max(0.05, 1.0 + strength * dot) * geom
    return w


# симуляция
class Sim:
    def __init__(self, a):
        np.random.seed(a.seed)
        random.seed(a.seed)
        self.a = a
        rng = np.random.default_rng(a.seed)
        pitch = a.pitch or (25 if a.size <= 2000 else 50)
        t0 = time.time()
        self.terrain, C = build_building(a.size, pitch, a.loops, rng)
        H, W = self.terrain.shape
        self.H, self.W = H, W
        ew = a.exit_width or max(5, pitch // 2)
        self.exits, self.ew = place_exits(self.terrain, C, pitch, a.exits, ew)
        K = len(self.exits)

        # очаги пожара: первый в заданной комнате (деревянной), остальные случайные
        self.burn = np.zeros((H, W), np.uint8)
        self.burn2 = np.zeros((H, W), np.uint8)
        self.smoke = np.zeros((H, W), np.uint8)
        self.smoke2 = np.zeros((H, W), np.uint8)
        spots = [(int(a.fire_y * C), int(a.fire_x * C))]
        for _ in range(a.fires - 1):
            spots.append((random.randrange(C), random.randrange(C)))
        for ri, rj in spots:
            ri = min(max(ri, 0), C - 1)
            rj = min(max(rj, 0), C - 1)
            blk = self.terrain[ri * pitch + 1:(ri + 1) * pitch, rj * pitch + 1:(rj + 1) * pitch]
            blk[blk < 3] = FLOOR_WOOD
            cy, cx = ri * pitch + pitch // 2, rj * pitch + pitch // 2
            self.burn[cy - 1:cy + 2, cx - 1:cx + 2] = 1

        # расстояния до выходов
        self.dist = np.full((K, H, W), UNREACH, dtype=np.uint16)
        for k, (_, _, ys, xs) in enumerate(self.exits):
            bfs_dist(self.terrain, ys, xs, self.dist[k])

        # люди
        n = a.people if a.people else (H * W) // 100
        self.occ = np.zeros((H, W), np.uint8)
        ax = np.zeros(n, np.int32)
        ay = np.zeros(n, np.int32)
        n = place_agents(self.terrain, self.burn, self.occ, n, ax, ay)
        self.n = n
        self.ax, self.ay = ax[:n].copy(), ay[:n].copy()
        d = self.dist[:, self.ay, self.ax]
        self.tgt = np.argmin(d, axis=0).astype(np.int64)
        del d
        self.hp = np.full(n, 100.0, np.float64)
        self.st = np.zeros(n, np.uint8)
        self.start_t = rng.integers(0, a.reaction + 1, size=n).astype(np.int64)
        self.stuck = np.zeros(n, np.int64)
        self.order = np.arange(n, dtype=np.int64)
        self.blocked = np.zeros(K, np.int8)

        self.wts = wind_weights(a.wind, a.wind_dir)
        self.t = 0
        self.ev_total = 0
        self.dead_total = 0
        self.sum_evac = 0
        self.hist = {k: [] for k in
                     ("t", "burning", "burnt", "area", "inside", "evac", "dead", "avg_time")}
        self.nburning = int((self.burn == 1).sum())
        self.nburnt = 0
        mb = (self.terrain.nbytes + 2 * self.burn.nbytes + 2 * self.smoke.nbytes +
              self.occ.nbytes + self.dist.nbytes) / 1e6
        print(f"[init] поле {H}x{W} ({H * W / 1e6:.1f} млн клеток), комнат {C}x{C}, "
              f"выходов {K}, людей {n}, ветер {a.wind}/{a.wind_dir}, "
              f"память ~{mb:.0f} МБ, {time.time() - t0:.1f} c")

    def _update_blocked(self):
        r = self.ew // 2 + 5
        for k, (cy, cx, _, _) in enumerate(self.exits):
            y0, y1 = max(0, cy - r), min(self.H, cy + r + 1)
            x0, x1 = max(0, cx - r), min(self.W, cx + r + 1)
            wb = self.burn[y0:y1, x0:x1]
            fire = bool(((wb >= 1) & (wb < 255)).any())
            sm = float(self.smoke[y0:y1, x0:x1].mean())
            self.blocked[k] = 1 if (fire or sm > self.a.exit_smoke) else 0

    def step(self):
        a = self.a
        self.nburning, self.nburnt = fire_step(
            self.terrain, self.burn, self.burn2, self.smoke, self.smoke2,
            P_MATERIAL, BURN_TIME, self.wts, a.smoke_decay, a.smoke_src)
        self.burn, self.burn2 = self.burn2, self.burn
        self.smoke, self.smoke2 = self.smoke2, self.smoke
        if self.t % 5 == 0:
            self._update_blocked()
        if self.n:
            ev, dead = agents_step(
                self.t, self.order, self.ax, self.ay, self.tgt, self.hp, self.st,
                self.start_t, self.stuck, self.terrain, self.burn, self.smoke,
                self.occ, self.dist, self.blocked,
                a.w_smoke, a.w_fire, a.w_crowd, a.noise, a.patience,
                a.smoke_thr, a.smoke_dmg)
            self.ev_total += ev
            self.dead_total += dead
            self.sum_evac += ev * self.t
        self.t += 1
        h = self.hist
        h["t"].append(self.t)
        h["burning"].append(self.nburning)
        h["burnt"].append(self.nburnt)
        h["area"].append(self.nburning + self.nburnt)
        h["inside"].append(self.n - self.ev_total - self.dead_total)
        h["evac"].append(self.ev_total)
        h["dead"].append(self.dead_total)
        h["avg_time"].append(self.sum_evac / self.ev_total if self.ev_total else 0.0)

    @property
    def inside(self):
        return self.n - self.ev_total - self.dead_total

    def running(self):
        return self.inside > 0 and self.t < self.a.max_steps

    def render(self, view=900, zoom=None):
        if zoom:
            y0, x0, sz = zoom
            y0 = min(max(0, y0), self.H - 2)
            x0 = min(max(0, x0), self.W - 2)
            hh = min(sz, self.H - y0)
            ww = min(sz, self.W - x0)
        else:
            y0 = x0 = 0
            hh, ww = self.H, self.W
        oh = min(view, hh)
        ow = min(view, ww)
        out = np.zeros((oh, ow, 3), np.uint8)
        f = max(hh / oh, ww / ow)
        wall_thr = max(1, int(0.6 * math.ceil(f)))
        render_kernel(self.terrain, self.burn, self.smoke, self.occ, out,
                      y0, x0, hh, ww, wall_thr)
        return out

    def summary(self):
        return {
            "time": self.t, "people": self.n, "evacuated": self.ev_total,
            "dead": self.dead_total, "inside": self.inside,
            "avg_evac_time": self.sum_evac / self.ev_total if self.ev_total else float("nan"),
            "fire_area": self.nburning + self.nburnt,
        }


# графики
def draw_plots(hist, axes):
    t = hist["t"]
    ax = axes[0]
    ax.clear()
    ax.plot(t, hist["area"], label="площадь пожара (горит+выгорело)", color="firebrick")
    ax.plot(t, hist["burning"], label="горит сейчас", color="orange")
    ax.set_title("Площадь пожара от времени")
    ax.set_xlabel("тик")
    ax.set_ylabel("клеток")
    ax.legend(fontsize=7)
    ax.grid(alpha=.3)
    ax = axes[1]
    ax.clear()
    ax.plot(t, hist["inside"], label="в здании", color="tab:blue")
    ax.plot(t, hist["evac"], label="эвакуировано", color="tab:green")
    ax.plot(t, hist["dead"], label="погибло", color="black")
    ax.set_title("Люди")
    ax.set_xlabel("тик")
    ax.legend(fontsize=7)
    ax.grid(alpha=.3)
    ax = axes[2]
    ax.clear()
    ax.plot(t, hist["avg_time"], color="purple")
    ax.set_title("Среднее время эвакуации (по вышедшим)")
    ax.set_xlabel("тик")
    ax.set_ylabel("тиков")
    ax.grid(alpha=.3)


def save_results(sim, prefix):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    draw_plots(sim.hist, [axes[0, 0], axes[0, 1], axes[1, 0]])
    axes[1, 1].imshow(sim.render(900, tuple(sim.a.zoom) if sim.a.zoom else None))
    axes[1, 1].set_title(f"Состояние на тике {sim.t}")
    axes[1, 1].axis("off")
    fig.tight_layout()
    fig.savefig(prefix + "_graphs.png", dpi=120)
    plt.close(fig)
    with open(prefix + "_history.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        keys = list(sim.hist.keys())
        w.writerow(keys)
        for row in zip(*[sim.hist[k] for k in keys]):
            w.writerow(row)
    plt.imsave(prefix + "_snapshot.png", sim.render(1600, tuple(sim.a.zoom) if sim.a.zoom else None))
    print(f"[save] {prefix}_graphs.png, {prefix}_history.csv, {prefix}_snapshot.png")


# режимы
def run_headless(sim, a):
    t0 = time.time()
    while sim.running():
        sim.step()
        if sim.t % a.log_every == 0:
            s = sim.summary()
            print(f"t={sim.t:5d} горит={sim.nburning:8d} площадь={s['fire_area']:9d} "
                  f"внутри={s['inside']:8d} вышло={s['evacuated']:8d} погибло={s['dead']:7d} "
                  f"({(time.time() - t0) / sim.t * 1000:.0f} мс/тик)")
    print("[итог]", sim.summary())
    save_results(sim, a.out)


def run_live(sim, a):
    import matplotlib.pyplot as plt
    plt.ion()
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.6, 1])
    ax_img = fig.add_subplot(gs[:, 0])
    axes = [fig.add_subplot(gs[i, 1]) for i in range(3)]
    zoom = tuple(a.zoom) if a.zoom else None
    img = ax_img.imshow(sim.render(a.view, zoom))
    ax_img.axis("off")
    fig.tight_layout()
    while sim.running():
        sim.step()
        if sim.t % a.render_every == 0:
            img.set_data(sim.render(a.view, zoom))
            s = sim.summary()
            ax_img.set_title(f"t={sim.t}  внутри={s['inside']}  вышло={s['evacuated']}  "
                             f"погибло={s['dead']}  площадь огня={s['fire_area']}")
            if sim.t % (a.render_every * 4) == 0:
                draw_plots(sim.hist, axes)
            plt.pause(0.001)
            if not plt.fignum_exists(fig.number):
                break
    print("[итог]", sim.summary())
    draw_plots(sim.hist, axes)
    img.set_data(sim.render(a.view, zoom))
    save_results(sim, a.out)
    plt.ioff()
    if plt.fignum_exists(fig.number):
        plt.show()


def run_experiment(a):
    import copy
    import matplotlib.pyplot as plt
    results = []
    exits_list = [int(x) for x in a.exp_exits.split(",")]
    winds = [float(x) for x in a.exp_winds.split(",")]
    peoples = [int(x) for x in a.exp_people.split(",")]
    first_hist = None
    for ne in exits_list:
        for w in winds:
            for pc in peoples:
                for rep in range(a.repeats):
                    b = copy.copy(a)
                    b.exits, b.wind, b.people = ne, w, pc
                    b.seed = a.seed + rep
                    sim = Sim(b)
                    while sim.running():
                        sim.step()
                    s = sim.summary()
                    s.update(exits=ne, wind=w, people_cfg=pc, rep=rep)
                    results.append(s)
                    print(f"[exp] выходов={ne} ветер={w} людей={sim.n} rep={rep} -> "
                          f"вышло {s['evacuated']}/{sim.n}, погибло {s['dead']}, "
                          f"ср.время {s['avg_evac_time']:.1f}, t={s['time']}")
                    if first_hist is None:
                        first_hist = sim.hist
                    del sim
                    gc.collect()
    keys = ["exits", "wind", "people_cfg", "rep", "people", "evacuated", "dead",
            "inside", "avg_evac_time", "time", "fire_area"]
    with open(a.out + "_experiment.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        for r in results:
            wr.writerow({k: r[k] for k in keys})

    fig, axs = plt.subplots(1, 3, figsize=(17, 5))
    for w in winds:
        for pc in peoples:
            xs, ya, yb, yc = [], [], [], []
            for ne in exits_list:
                rr = [r for r in results if r["exits"] == ne and r["wind"] == w
                      and r["people_cfg"] == pc]
                xs.append(ne)
                ya.append(np.nanmean([r["avg_evac_time"] for r in rr]))
                yb.append(np.mean([r["evacuated"] / r["people"] for r in rr]) * 100)
                yc.append(np.mean([r["dead"] / r["people"] for r in rr]) * 100)
            lab = f"ветер {w}, людей {pc}"
            axs[0].plot(xs, ya, "o-", label=lab)
            axs[1].plot(xs, yb, "o-", label=lab)
            axs[2].plot(xs, yc, "o-", label=lab)
    axs[0].set_title("Среднее время эвакуации от числа выходов")
    axs[0].set_ylabel("тиков")
    axs[1].set_title("Доля эвакуированных, %")
    axs[2].set_title("Доля погибших, %")
    for ax in axs:
        ax.set_xlabel("число выходов")
        ax.grid(alpha=.3)
        ax.set_xticks(exits_list)
    axs[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(a.out + "_experiment.png", dpi=120)
    print(f"[save] {a.out}_experiment.csv, {a.out}_experiment.png")


def main():
    p = argparse.ArgumentParser(description="КА: пожар и эвакуация")
    p.add_argument("--size", type=int, default=1000, help="сторона поля (>=1000, можно 10000)")
    p.add_argument("--pitch", type=int, default=0, help="размер комнаты (0 = авто)")
    p.add_argument("--loops", type=float, default=0.35, help="доля лишних дверей (петли)")
    p.add_argument("--people", type=int, default=0, help="число людей (0 = 1%% клеток)")
    p.add_argument("--exits", type=int, default=2, help="число выходов 1..8")
    p.add_argument("--exit-width", dest="exit_width", type=int, default=0)
    p.add_argument("--wind", type=float, default=0.0, help="сила ветра 0..2 (0 = нет)")
    p.add_argument("--wind-dir", dest="wind_dir", default="E", choices=list(WIND_DIRS))
    p.add_argument("--fires", type=int, default=1, help="число очагов")
    p.add_argument("--fire-x", dest="fire_x", type=float, default=0.45)
    p.add_argument("--fire-y", dest="fire_y", type=float, default=0.5)
    p.add_argument("--smoke-decay", dest="smoke_decay", type=float, default=0.985)
    p.add_argument("--smoke-src", dest="smoke_src", type=int, default=220)
    p.add_argument("--exit-smoke", dest="exit_smoke", type=float, default=60.0,
                   help="средний дым у выхода, при котором он считается закрытым")
    p.add_argument("--reaction", type=int, default=30, help="макс. время реакции людей, тиков")
    p.add_argument("--w-smoke", dest="w_smoke", type=float, default=40.0)
    p.add_argument("--w-fire", dest="w_fire", type=float, default=12.0)
    p.add_argument("--w-crowd", dest="w_crowd", type=float, default=0.6)
    p.add_argument("--noise", type=float, default=0.3)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--smoke-thr", dest="smoke_thr", type=float, default=80.0)
    p.add_argument("--smoke-dmg", dest="smoke_dmg", type=float, default=0.05)
    p.add_argument("--max-steps", dest="max_steps", type=int, default=3000)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--experiment", action="store_true")
    p.add_argument("--exp-exits", dest="exp_exits", default="1,2,3,4")
    p.add_argument("--exp-winds", dest="exp_winds", default="0,1.0")
    p.add_argument("--exp-people", dest="exp_people", default="2000,8000")
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--view", type=int, default=900, help="размер картинки при показе")
    p.add_argument("--zoom", type=int, nargs=3, metavar=("Y0", "X0", "SIZE"),
                   help="показывать только фрагмент поля")
    p.add_argument("--render-every", dest="render_every", type=int, default=5)
    p.add_argument("--log-every", dest="log_every", type=int, default=50)
    p.add_argument("--out", default="result", help="префикс выходных файлов")
    a = p.parse_args()
    a.exits = max(1, min(8, a.exits))

    if a.headless or a.experiment:
        import matplotlib
        matplotlib.use("Agg")

    if a.experiment:
        run_experiment(a)
        return
    sim = Sim(a)
    if a.headless:
        run_headless(sim, a)
    else:
        run_live(sim, a)


if __name__ == "__main__":
    main()

"""
Shared constants and signal functions for the SIFT scale-space video (pure numpy, no Manim).
Importable by Scene 2 and Scene 3.
"""
import math

import numpy as np

F = 1.5                      # focal length
L_SET = 2.0                  # object length
Z_SET = [2, 4, 8]            # depths of the three Scene-2 signals
BG = 0.1                     # background brightness
PEAK = 1.0                   # object brightness

_erf = np.vectorize(math.erf, otypes=[float])


def widths():
    """Image widths F*L/Z for Z in Z_SET: [1.5, 0.75, 0.375]."""
    return [F * L_SET / Z for Z in Z_SET]


def box(x, w):
    """BG + (PEAK-BG) * 1[|x| <= w/2]."""
    x = np.asarray(x, dtype=float)
    return BG + (PEAK - BG) * (np.abs(x) <= w / 2 + 1e-12)


def gauss(x, sigma):
    """Unit-area Gaussian kernel."""
    x = np.asarray(x, dtype=float)
    return np.exp(-x**2 / (2 * sigma**2)) / (np.sqrt(2 * np.pi) * sigma)


def blurred_box(x, w, sigma):
    """Exact continuous convolution of `box` with a unit-area Gaussian of std sigma."""
    x = np.asarray(x, dtype=float)
    s = np.sqrt(2) * sigma
    return BG + (PEAK - BG) * 0.5 * (_erf((x + w / 2) / s) - _erf((x - w / 2) / s))


def peak_value(w, sigma):
    """Height of the blurred box at x = 0 (its maximum)."""
    return float(blurred_box(0.0, w, sigma))


# ----------------------------------------------------------------- Scene 4: heat on a rod
ROD_CENTERS = [-2.5, 0.0, 2.5]      # boxes for Z = 2, 4, 8 (widths 1.5, 0.75, 0.375)


def _boxes():
    return list(zip(widths(), ROD_CENTERS))


def rod_profile(x):
    """BG + sum of the three boxes (each box rises (PEAK-BG) above BG)."""
    x = np.asarray(x, dtype=float)
    return BG + sum(box(x - c, w) - BG for w, c in _boxes())


def rod_blurred(x, t):
    """Exact heat-equation solution u_t = u_xx from rod_profile: blur with sigma = sqrt(2t)."""
    x = np.asarray(x, dtype=float)
    if t <= 0:
        return rod_profile(x)
    sig = math.sqrt(2 * t)
    return BG + sum(blurred_box(x - c, w, sig) - BG for w, c in _boxes())


def _cell_average_profile(x, dx):
    """rod_profile averaged over the cell [x-dx/2, x+dx/2] (exact; keeps box edges off-grid errors O(dx^2))."""
    u = np.full_like(x, BG)
    for w, c in _boxes():
        lo = np.maximum(x - dx / 2, c - w / 2)
        hi = np.minimum(x + dx / 2, c + w / 2)
        u += (PEAK - BG) * np.clip(hi - lo, 0, None) / dx
    return u


def heat_ftcs(t_list, dx=0.01, L=10.0):
    """Explicit FTCS solver for u_t = u_xx on [-L, L], Dirichlet u = BG at both ends.

    Initial condition: rod_profile (cell-averaged on the grid). dt = 0.4 dx^2 (stable: dt <= dx^2/2);
    the last step to each requested time is shortened so snapshots land exactly on t.
    Returns (x, snapshots) with snapshots[i] = u at t_list[i] (in the order given).
    """
    x = np.arange(-L, L + dx / 2, dx)
    u = _cell_average_profile(x, dx)
    u[0] = u[-1] = BG
    dt = 0.4 * dx**2
    order = np.argsort(t_list)
    snaps = [None] * len(t_list)
    t_now = 0.0
    for i in order:
        target = float(t_list[i])
        n = int(math.floor((target - t_now) / dt + 1e-9))
        for _ in range(n):
            u[1:-1] += (dt / dx**2) * (u[2:] - 2 * u[1:-1] + u[:-2])
        rem = target - t_now - n * dt
        if rem > 1e-15:
            u[1:-1] += (rem / dx**2) * (u[2:] - 2 * u[1:-1] + u[:-2])
        t_now = target
        snaps[i] = u.copy()
    return x, np.array(snaps)


def half_life(w):
    """Diffusion time t at which a single box's center contrast halves: erf(w/(2 sqrt2 sigma)) = 1/2."""
    lo, hi = 1e-9, 100.0 * w
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if math.erf(w / (2 * math.sqrt(2) * mid)) > 0.5:
            lo = mid            # still above half -> need larger sigma
        else:
            hi = mid
    sigma = 0.5 * (lo + hi)
    return sigma**2 / 2

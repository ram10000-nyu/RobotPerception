"""
Scene 4 — "Blur is heat" (SIFT scale-space video).

Render:
    manim -pql scene4_blur_is_heat.py BlurIsHeat     # preview (480p15)
    manim -qh  scene4_blur_is_heat.py BlurIsHeat     # final   (1080p60)

Brightness = temperature on a rod. All curves, colors and readouts come from common.py
(exact solution `rod_blurred`) or from ONE precomputed FTCS run (`heat_ftcs`) that updaters
only interpolate. A single ValueTracker drives sigma; t = sigma^2 / 2.
Set NO_LATEX=1 only to preview without a LaTeX install.
"""
import math
import os
import sys

import numpy as np
from manim import *

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (BG, PEAK, ROD_CENTERS, blurred_box, box, gauss, half_life, heat_ftcs,  # noqa: E402
                    rod_blurred, rod_profile, widths)

# ----------------------------------------------------------------- constants
Z_COLORS = [YELLOW, TEAL, PINK]        # Z = 2, 4, 8 (near -> far), as boxes in ROD_CENTERS order
SIM_DOT_COLOR = WHITE
BLUR_COLOR = ORANGE
COOL_COLOR, WARM_COLOR = BLUE_C, ORANGE

T3_END = 0.7                           # Beat 3 diffusion time
SIG3_END = math.sqrt(2 * T3_END)
SIG_MAX = 1.35                         # snapshots cover sigma in [0, SIG_MAX] (Beat 4 runs this far)
N_SNAP = 120                           # snapshots, uniform in sigma
SEPARATION_T = 0.2                     # boxes still distinct here (they merge by t = 0.7)

# plot / rod layout
X_LEN, Y_LEN, PLOT_ORIGIN = 11.0, 2.6, (0.0, -1.55)
N_ROD, ROD_Y, ROD_H = 200, -2.5, 0.5
COLD, MID, HOT = ManimColor("#0B1026"), ManimColor("#C2410C"), ManimColor("#FFD60A")
REGIONS = [(-4.0, -1.25), (-1.25, 1.25), (1.25, 4.0)]     # one color per box
READ_X_LAB, READ_X_NUM, READ_Y = 5.0, 5.15, (3.45, 2.95)

# Beat 2
T_EDGE = 0.005                         # smoothing for the second derivative (avoids infinite edges)
N_ARROWS, ARROW_X = 25, 3.6
ARROW_K, ARROW_CAP, ARROW_MIN = 0.04, 0.7, 0.04
T_ZOOM, XN, HN = 0.02, -0.125, 0.2     # inset: profile time, middle sample x, sample spacing

# Beat 4
BAR_BASE_Y, BAR_SCALE, BAR_W = 1.55, 0.7, 0.3
SIG4_RUN = 4.6                         # seconds for sigma 0 -> SIG_MAX

# panels (Scene 3's end state)
PANEL_X = (-4.4, 0.0, 4.4)
PANEL_BASE_Y, PANEL_X_LEN, PANEL_Y_LEN, TITLE_Y = -1.2, 3.6, 2.0, 1.5

BEAT_LEN = dict(b1=12, b2=16, b3=14, b4=8)     # seconds, total 50
RT = dict(
    panels_in=1.5, merge=2.0, rod=1.5, readout=0.8, preview=1.5,
    zoom=1.0, inset=1.5, dots=1.0, avg=1.2, arrow=1.3, eq1=1.5, back=1.0, arrows=1.5, eq2=1.3, eq3=1.3,
    box=0.5, to_sim=1.0, eqs=1.2, legend=1.0, run3=8.0,
    reset=0.8, bars=0.7, flash=0.8, final=0.5, rewind=0.8,
)

USE_LATEX = os.environ.get("NO_LATEX") != "1"

# ---------------------------------------------------------- precomputed heat run
X_GRID = None
SNAP_T = None
SNAPS = None
_SIG_STEP = SIG_MAX / N_SNAP


def build_snapshots():
    """Run FTCS once; snapshots at t_k = sigma_k^2 / 2 with sigma_k uniform (dense early on)."""
    global X_GRID, SNAP_T, SNAPS
    if SNAPS is None:
        sig = np.linspace(0, SIG_MAX, N_SNAP + 1)
        SNAP_T = sig**2 / 2
        X_GRID, SNAPS = heat_ftcs(list(SNAP_T))
    return X_GRID, SNAP_T, SNAPS


_cache = {}


def sim_at(t):
    """Interpolated FTCS profile on the full grid at time t (no solver calls)."""
    key = round(t, 12)
    if key in _cache:
        return _cache[key]
    _, T, S = build_snapshots()
    s = math.sqrt(max(2 * t, 0.0))
    k = min(int(s / _SIG_STEP), N_SNAP - 1)
    w = (t - T[k]) / (T[k + 1] - T[k])
    u = (1 - w) * S[k] + w * S[k + 1]
    if len(_cache) > 64:
        _cache.clear()
    _cache[key] = u
    return u


def sim_on(xs, t):
    return np.interp(xs, X_GRID, sim_at(t))


def temp_color(u):
    v = float(np.clip((u - BG) / (PEAK - BG), 0, 1))
    if v < 0.5:
        return interpolate_color(COLD, MID, 2 * v)
    return interpolate_color(MID, HOT, 2 * v - 1)


# --------------------------------------------------------------------- checks
def self_check():
    ws = widths()
    # 1. FTCS vs exact diffusion on |x| <= 4
    ts = [0.01, 0.05, 0.2, 0.7]
    x, S = heat_ftcs(ts)
    m = np.abs(x) <= 4
    for t, s, tol in zip(ts, S, (0.02, 0.006, 0.004, 0.004)):
        err = np.max(np.abs(s[m] - rod_blurred(x[m], t)))
        assert err < tol, (t, err)
    # 2. G = exp(-x^2/4t)/sqrt(4 pi t) satisfies G_t = G_xx
    G = lambda xx, t: np.exp(-xx**2 / (4 * t)) / np.sqrt(4 * np.pi * t)
    xx = np.linspace(-3, 3, 601)
    h, dt = 1e-3, 1e-5
    for t in (0.05, 0.2, 0.5):
        Gt = (G(xx, t + dt) - G(xx, t - dt)) / (2 * dt)
        Gxx = (G(xx + h, t) - 2 * G(xx, t) + G(xx - h, t)) / h**2
        assert np.max(np.abs(Gt - Gxx)) < 1e-4, t
    # 3. sigma = sqrt(2t): rod_blurred is exactly the sum of blurred boxes at that sigma
    xs = np.linspace(-4, 4, 801)
    for t in (0.005, 0.1, 0.7):
        sig = math.sqrt(2 * t)
        ref = BG + sum(blurred_box(xs - c, w, sig) - BG for w, c in zip(ws, ROD_CENTERS))
        assert np.max(np.abs(rod_blurred(xs, t) - ref)) < 1e-12
    assert np.allclose(rod_blurred(xs, 0.0), rod_profile(xs))
    assert np.isclose(float(gauss(0.0, 0.3)), 1 / (math.sqrt(2 * math.pi) * 0.3))
    # 4. half-lives ~ 0.2748 w^2, far < middle < near
    hl = [half_life(w) for w in ws]
    for w, h_ in zip(ws, hl):
        assert abs(h_ - 0.2748 * w * w) < 1e-3, (w, h_)
    assert hl[2] < hl[1] < hl[0]
    # 5. boxes remain separated (midpoint contrast below both neighbouring centers) while t <= SEPARATION_T
    u = lambda xv: float(rod_blurred(xv, SEPARATION_T))
    for mid, (a, b) in zip((-1.25, 1.25), ((0, 1), (1, 2))):
        assert u(mid) < min(u(ROD_CENTERS[a]), u(ROD_CENTERS[b])), mid
    # snapshot interpolation used by the animation stays on the exact solution
    build_snapshots()
    xs4 = np.linspace(-4, 4, 801)
    for t in (0.0003, 0.0123, 0.1, 0.3333, 0.69):
        assert np.max(np.abs(sim_on(xs4, t) - rod_blurred(xs4, t))) < 0.01, t
    # box edges and the three half-life flashes happen inside the Beat 4 sigma range
    assert math.sqrt(2 * hl[0]) < SIG_MAX


# ------------------------------------------------------------------- helpers
def tex(*parts, fb=None):
    """MathTex, or (NO_LATEX=1) a Text fallback with the same part indexing."""
    if USE_LATEX:
        return MathTex(*parts)
    fb = fb or parts
    return VGroup(*[Text(p, font_size=30) for p in fb]).arrange(RIGHT, buff=0.12)


def number_mob(fn, color, font_size=30, places=2):
    if USE_LATEX:
        m = DecimalNumber(fn(), num_decimal_places=places, font_size=font_size, color=color)
        m.add_updater(lambda mob: mob.set_value(fn()))
    else:
        m = Text(f"{fn():.{places}f}", font_size=font_size - 6, color=color)
        m.add_updater(lambda mob: mob.become(Text(f"{fn():.{places}f}", font_size=font_size - 6, color=color)))
    return m


def chain(a, b, **kw):
    """TransformMatchingTex between equations (ReplacementTransform without LaTeX)."""
    return TransformMatchingTex(a, b, **kw) if USE_LATEX else ReplacementTransform(a, b, **kw)


def make_axes(x_range, y_range, x_len, y_len, origin):
    ax = Axes(
        x_range=x_range, y_range=y_range, x_length=x_len, y_length=y_len, tips=False,
        axis_config={"include_ticks": False, "include_numbers": False, "stroke_width": 3},
    )
    ax.shift(np.array([origin[0], origin[1], 0]) - ax.c2p(x_range[0] if x_range[0] > 0 else 0, 0))
    return ax


def poly(ax, xs, ys, color, width=5):
    m = VMobject(color=color, stroke_width=width)
    m.set_points_as_corners([ax.c2p(x, y) for x, y in zip(xs, ys)])
    return m


def box_corners(w, c, a, b):
    """Exact corner points of the rod_profile piece restricted to [a, b] (box of width w at c)."""
    return ([a, c - w / 2, c - w / 2, c + w / 2, c + w / 2, b], [BG, BG, PEAK, PEAK, BG, BG])


# -------------------------------------------------------------------- scene
class BlurIsHeat(Scene):
    def construct(self):
        self_check()
        build_snapshots()
        ws = widths()

        def pad(t_end):
            dt = t_end - self.time
            if dt > 1e-3:
                self.wait(dt)

        sg = ValueTracker(0.0)                      # sigma; t = sigma^2 / 2

        def tval():
            return sg.get_value() ** 2 / 2

        # ---- main plot
        ax = make_axes([-4, 4, 1], [0, 1.1, 1], X_LEN, Y_LEN, PLOT_ORIGIN)
        ax.y_axis.set_stroke(opacity=0.5)
        ax_labels = ax.get_axis_labels(tex("x", fb=["x"]).scale(0.7), tex("u(x,t)", fb=["u(x,t)"]).scale(0.7))
        ax_labels[1].move_to(ax.c2p(-4.0, 1.1) + UP * 0.1 + RIGHT * 0.1)

        def pieces(t):
            g = VGroup()
            for (a, b), col, w, c in zip(REGIONS, Z_COLORS, ws, ROD_CENTERS):
                if t <= 1e-9:
                    xs, ys = box_corners(w, c, a, b)
                else:
                    xs = np.linspace(a, b, int((b - a) / 0.02) + 1)
                    ys = sim_on(xs, t)
                g.add(poly(ax, xs, ys, col, 5))
            return g

        # ---- rod (heat strip)
        x_edges = np.linspace(-4, 4, N_ROD + 1)
        x_cent = 0.5 * (x_edges[:-1] + x_edges[1:])
        cell_w = ax.c2p(x_edges[1], 0)[0] - ax.c2p(x_edges[0], 0)[0]
        rod = VGroup(*[Rectangle(width=cell_w, height=ROD_H, stroke_width=0, fill_opacity=1,
                                 fill_color=COLD).move_to([ax.c2p(x, 0)[0], ROD_Y, 0]) for x in x_cent])

        def rod_update(m):
            u = sim_on(x_cent, tval())
            for r, v in zip(m, u):
                r.set_fill(temp_color(v), opacity=r.get_fill_opacity())

        rod.add_updater(rod_update)
        rod_update(rod)
        rod_label = Text("metal rod", font_size=20, color=GRAY_B).move_to([0, ROD_Y - 0.5, 0])

        # ---- readouts
        t_lab = Text("t =", font_size=26).move_to([READ_X_LAB, READ_Y[0], 0], aligned_edge=RIGHT)
        s_lab = tex(r"\sigma = \sqrt{2t} =", fb=["σ = √(2t) ="]).scale(0.8)
        s_lab.move_to([READ_X_LAB, READ_Y[1], 0], aligned_edge=RIGHT)
        t_num = number_mob(tval, WHITE, 30, 3)
        s_num = number_mob(sg.get_value, WHITE, 30, 3)
        t_num.add_updater(lambda m: m.move_to([READ_X_NUM, READ_Y[0], 0], aligned_edge=LEFT))
        s_num.add_updater(lambda m: m.move_to([READ_X_NUM, READ_Y[1], 0], aligned_edge=LEFT))
        t_num.update()
        s_num.update()
        readout = VGroup(t_lab, s_lab, t_num, s_num)

        # ================= BEAT 1 (0-12 s) =================
        # "As sigma grows, something familiar happens. Picture the signal as a metal rod, and
        # brightness as temperature. Blurring looks just like heat spreading out."
        t0 = self.time
        panels, panel_boxes, panel_rest = [], [], VGroup()
        for Z, w, xc, col in zip((2, 4, 8), ws, PANEL_X, Z_COLORS):
            pax = make_axes([-2, 2, 1], [0, 1.2, 1], PANEL_X_LEN, PANEL_Y_LEN, (xc, PANEL_BASE_Y))
            title = tex(f"Z = {Z}", fb=[f"Z = {Z}"]).scale(0.8).move_to([xc, TITLE_Y, 0])
            xs = [-2, -w / 2, -w / 2, w / 2, w / 2, 2]
            pbox = poly(pax, xs, [BG, BG, PEAK, PEAK, BG, BG], col, 5)
            panel_boxes.append(pbox)
            panel_rest.add(pax, title)
        self.play(FadeIn(panel_rest), *[FadeIn(b) for b in panel_boxes], run_time=RT["panels_in"])
        self.wait(0.5)
        statics = pieces(0.0)
        self.play(
            FadeOut(panel_rest), FadeIn(ax), FadeIn(ax_labels),
            *[ReplacementTransform(pb, ps) for pb, ps in zip(panel_boxes, statics)],
            run_time=RT["merge"],
        )
        self.remove(*statics)
        pieces_live = always_redraw(lambda: pieces(tval()))
        self.add(pieces_live)
        self.play(FadeIn(rod), FadeIn(rod_label), run_time=RT["rod"])
        self.play(FadeIn(readout), run_time=RT["readout"])
        self.play(sg.animate.set_value(math.sqrt(2 * 0.02)), run_time=RT["preview"], rate_func=smooth)
        self.play(sg.animate.set_value(0.0), run_time=RT["preview"], rate_func=smooth)
        pad(t0 + BEAT_LEN["b1"])

        # ================= BEAT 2 (12-28 s) =================
        # "Heat flows from hot to cold. A point cools at a rate set by how much hotter it is than
        # the average of its neighbors. That difference is the second derivative, and it gives the
        # heat equation."
        t0 = self.time
        c_mid = ROD_CENTERS[1]
        zoom_box = Rectangle(width=ax.c2p(c_mid - 0.375 + 0.5, 0)[0] - ax.c2p(c_mid - 0.375 - 0.5, 0)[0],
                             height=Y_LEN, color=WHITE, stroke_width=2)
        zoom_box.move_to([ax.c2p(c_mid - 0.375, 0)[0], ax.c2p(0, 0.55)[1], 0])
        self.play(Create(zoom_box), run_time=RT["zoom"])

        backdrop = Rectangle(width=7.8, height=4.7, stroke_color=WHITE, stroke_width=1, fill_color=BLACK,
                             fill_opacity=0.95).move_to([0, 0.35, 0])
        x_lo, x_hi = -0.375 - 0.5, -0.375 + 0.5
        ins = make_axes([x_lo, x_hi, 0.25], [0, 1.1, 1], 6.0, 3.0, (0, 0))
        ins.move_to([0, 0.4, 0])
        xs_i = np.linspace(x_lo, x_hi, 200)
        ins_curve = poly(ins, xs_i, rod_blurred(xs_i, T_ZOOM), WHITE, 4)
        self.play(FadeIn(backdrop), FadeIn(ins), Create(ins_curve), run_time=RT["inset"])

        xn3 = [XN - HN, XN, XN + HN]
        un3 = [float(rod_blurred(x, T_ZOOM)) for x in xn3]
        avg = 0.5 * (un3[0] + un3[2])
        pts = [ins.c2p(x, u) for x, u in zip(xn3, un3)]
        dots = VGroup(*[Dot(p, radius=0.09, color=WHITE) for p in pts])
        labs = VGroup(
            tex("u[n-1]", fb=["u[n-1]"]).scale(0.7).next_to(pts[0], LEFT, buff=0.15),
            tex("u[n]", fb=["u[n]"]).scale(0.7).next_to(pts[1], UP, buff=0.15),
            tex("u[n+1]", fb=["u[n+1]"]).scale(0.7).next_to(pts[2], UP, buff=0.15),
        )
        self.play(FadeIn(dots), FadeIn(labs), run_time=RT["dots"])
        chord = Line(pts[0], pts[2], color=GRAY, stroke_width=2, stroke_opacity=0.7)
        avg_tick = DashedLine(ins.c2p(xn3[0], avg), ins.c2p(xn3[2], avg), dash_length=0.1, stroke_width=3,
                              color=GRAY_B)
        self.play(Create(chord), Create(avg_tick), run_time=RT["avg"])
        arrow = Arrow(pts[1], ins.c2p(XN, avg), buff=0, color=COOL_COLOR, stroke_width=6, tip_length=0.18)
        arrow_lab = Text("cooling rate", font_size=22, color=COOL_COLOR).next_to(ins.c2p(XN, avg), DOWN, buff=0.35)
        self.play(GrowArrow(arrow), FadeIn(arrow_lab), run_time=RT["arrow"])
        eq_pos = [-1.3, 3.2, 0]
        eq1 = tex(r"\frac{du_n}{dt}", r"\propto", r"\frac{u_{n+1} + u_{n-1}}{2} - u_n",
                  fb=["du_n/dt", "∝", "(u_n+1 + u_n−1)/2 − u_n"]).move_to(eq_pos)
        self.play(Write(eq1), run_time=RT["eq1"])
        self.wait(0.5)

        # full picture: du/dt = u_xx (smoothed edges) at 25 evenly spaced points
        arrows = VGroup()
        h_ = 1e-3
        for x in np.linspace(-ARROW_X, ARROW_X, N_ARROWS):
            f = lambda xv: float(rod_blurred(xv, T_EDGE))
            uxx = (f(x + h_) - 2 * f(x) + f(x - h_)) / h_**2
            ln = float(np.clip(ARROW_K * uxx, -ARROW_CAP, ARROW_CAP))
            if abs(ln) < ARROW_MIN:
                continue
            start = ax.c2p(x, float(rod_profile(x)))
            arrows.add(Arrow(start, start + UP * ln, buff=0, stroke_width=4, tip_length=0.14,
                             color=WARM_COLOR if ln > 0 else COOL_COLOR))
        self.play(FadeOut(backdrop), FadeOut(ins), FadeOut(ins_curve), FadeOut(dots), FadeOut(labs),
                  FadeOut(chord), FadeOut(avg_tick), FadeOut(arrow), FadeOut(arrow_lab), FadeOut(zoom_box),
                  run_time=RT["back"])
        self.play(LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.08), run_time=RT["arrows"])
        self.wait(0.7)
        eq2 = tex(r"\frac{du_n}{dt}", "=", r"\frac{u_{n+1} - 2u_n + u_{n-1}}{h^2}",
                  fb=["du_n/dt", "=", "(u_n+1 − 2u_n + u_n−1)/h²"]).move_to(eq_pos)
        self.play(chain(eq1, eq2), FadeOut(arrows), run_time=RT["eq2"])
        eq3 = tex(r"\frac{\partial u}{\partial t}", "=", r"\frac{\partial^2 u}{\partial x^2}",
                  fb=["∂u/∂t", "=", "∂²u/∂x²"]).move_to(eq_pos)
        self.play(chain(eq2, eq3), run_time=RT["eq3"])
        eq_box = SurroundingRectangle(eq3, color=YELLOW, buff=0.15)
        self.play(Create(eq_box), run_time=RT["box"])
        pad(t0 + BEAT_LEN["b2"])

        # ================= BEAT 3 (28-42 s) =================
        # "Start from any temperature profile, wait a time t, and you get the original profile
        # convolved with a Gaussian of width sigma = sqrt(2t). Blurring is diffusion."
        t0 = self.time
        xs401 = np.linspace(-4, 4, 401)
        blur_live = always_redraw(lambda: poly(ax, xs401, rod_blurred(xs401, tval()), BLUR_COLOR, 4))
        idx = np.round((np.arange(-4, 4.0001, 0.1) + 10) / 0.01).astype(int)     # every 10th grid point
        x_sim = X_GRID[idx]
        sim_dots = VGroup(*[Dot(ax.c2p(x, BG), radius=0.035, color=SIM_DOT_COLOR) for x in x_sim])

        def sim_update(m):
            u = sim_at(tval())[idx]
            for d, x, v in zip(m, x_sim, u):
                d.move_to(ax.c2p(x, v))

        sim_dots.add_updater(sim_update)
        sim_update(sim_dots)
        ghost = pieces(0.0).set_opacity(0.3)
        leg1 = VGroup(Dot(radius=0.04, color=SIM_DOT_COLOR), Text("heat simulation", font_size=20)).arrange(RIGHT, buff=0.15)
        leg2 = VGroup(Line(ORIGIN, RIGHT * 0.35, color=BLUR_COLOR, stroke_width=4),
                      Text("Gaussian blur, σ = √(2t)", font_size=20)).arrange(RIGHT, buff=0.15)
        leg1.move_to([-6.7, 1.95, 0], aligned_edge=LEFT)
        leg2.move_to([-6.7, 1.6, 0], aligned_edge=LEFT)
        eqa = tex(r"u(x,t) = (G_{\sqrt{2t}} * u_0)(x)", fb=["u(x,t) = (G_√(2t) * u₀)(x)"]).move_to([-1.3, 3.2, 0])
        eqb = tex(r"G(x,t) = \frac{1}{\sqrt{4\pi t}}\, e^{-x^2/4t}",
                  fb=["G(x,t) = 1/√(4πt) · e^(−x²/4t)"]).move_to([-1.3, 2, 0])
        self.remove(pieces_live)
        self.add(ghost, blur_live, sim_dots)
        self.play(FadeOut(eq3), FadeOut(eq_box), run_time=0.5)
        self.play(Write(eqa), run_time=RT["eqs"])
        self.play(Write(eqb), FadeIn(leg1), FadeIn(leg2), run_time=RT["legend"])
        self.play(sg.animate.set_value(SIG3_END), run_time=RT["run3"], rate_func=linear)
        pad(t0 + BEAT_LEN["b3"])

        # ================= BEAT 4 (42-50 s) =================
        # "And notice: small hot spots fade first, while big ones last. How long a spot survives
        # tells us how big it is."
        t0 = self.time
        self.play(FadeOut(blur_live), FadeOut(sim_dots), FadeOut(leg1), FadeOut(leg2), FadeOut(eqa),
                  FadeOut(eqb), FadeOut(ghost), run_time=RT["reset"] / 2)
        sg.set_value(0.0)
        self.add(pieces_live)
        self.wait(RT["reset"] / 2)

        bars, outlines, labels = VGroup(), VGroup(), []
        hl = [half_life(w) for w in ws]
        max_h = (PEAK - BG) * BAR_SCALE
        for w, c, col, hv in zip(ws, ROD_CENTERS, Z_COLORS, hl):
            bx = ax.c2p(c, 0)[0]
            outline = Rectangle(width=BAR_W, height=max_h, stroke_color=GRAY, stroke_width=1.5, fill_opacity=0)
            outline.move_to([bx, BAR_BASE_Y + max_h / 2, 0])
            tick = DashedLine([bx - BAR_W / 2 - 0.08, BAR_BASE_Y + max_h / 2, 0],
                              [bx + BAR_W / 2 + 0.08, BAR_BASE_Y + max_h / 2, 0], dash_length=0.06,
                              stroke_width=2, color=WHITE)
            outlines.add(outline, tick)

            def bar_f(c=c, col=col, bx=bx):
                v = max(float(sim_on(np.array([c]), tval())[0]) - BG, 0.0)
                h = max(v * BAR_SCALE, 1e-3)
                return Rectangle(width=BAR_W, height=h, stroke_width=0, fill_color=col,
                                 fill_opacity=0.9).move_to([bx, BAR_BASE_Y + h / 2, 0])

            bars.add(always_redraw(bar_f))
            lab = tex(f"t_{{1/2}} = {hv:.3f}", fb=[f"t_1/2 = {hv:.3f}"]).scale(0.7).set_color(col)
            lab.move_to([bx, BAR_BASE_Y + max_h + 0.35, 0])
            labels.append(lab)
        self.play(FadeIn(outlines), FadeIn(bars), run_time=RT["bars"])

        # one continuous linear sigma ramp, split at the three half-life moments (far, middle, near)
        events = sorted(range(3), key=lambda i: hl[i])           # far (index 2) first
        prev = 0.0
        pending = None
        bounds = [math.sqrt(2 * hl[i]) for i in events] + [SIG_MAX]
        for k, sig_b in enumerate(bounds):
            dur = SIG4_RUN * (sig_b - prev) / SIG_MAX
            anims = [sg.animate(run_time=dur, rate_func=linear).set_value(sig_b)]
            if pending is not None:
                i = pending
                anims += [Flash(bars[i].get_top(), color=Z_COLORS[i], flash_radius=0.45, run_time=RT["flash"]),
                          FadeIn(labels[i])]
            self.play(*anims)
            prev = sig_b
            pending = events[k] if k < 3 else None
        final = tex(r"t_{1/2} \propto w^2", fb=["t_1/2 ∝ w²"]).scale(0.8).move_to([-1.3, 3.1, 0])
        self.play(Write(final), run_time=RT["final"])
        self.play(sg.animate.set_value(0.0), run_time=RT["rewind"], rate_func=smooth)   # boxes recognizable again
        pad(t0 + BEAT_LEN["b4"])


# ---------------------------------------------------------------------------
# Docs / source consulted (manim==0.21.0; https://docs.manim.community/en/stable/)
#  - interpolate_color (colormap):
#      https://docs.manim.community/en/stable/reference/manim.utils.color.core.html
#  - TransformMatchingTex / TransformMatchingShapes:
#      https://docs.manim.community/en/stable/reference/manim.animation.transform_matching_parts.TransformMatchingTex.html
#  - Axes, c2p:
#      https://docs.manim.community/en/stable/reference/manim.mobject.graphing.coordinate_systems.Axes.html
#  - Arrow, GrowArrow, DashedLine, SurroundingRectangle:
#      https://docs.manim.community/en/stable/reference/manim.mobject.geometry.line.Arrow.html
#  - Flash, LaggedStart:
#      https://docs.manim.community/en/stable/reference/manim.animation.indication.Flash.html
#  - ValueTracker, always_redraw, DecimalNumber, Scene.time:
#      https://docs.manim.community/en/stable/reference/manim.mobject.value_tracker.ValueTracker.html

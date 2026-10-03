"""
Scene 3 — Kernels and the 1D Gaussian (SIFT scale-space video).

Render:
    manim -pql scene3_kernel_gaussian.py KernelGaussian1D     # preview (480p15)
    manim -qh  scene3_kernel_gaussian.py KernelGaussian1D     # final   (1080p60)

Signals and kernels come from common.py (shared with Scene 2). Every number on screen is
computed from those functions. Set NO_LATEX=1 only to preview without a LaTeX install.
"""
import math
import os
import sys

import numpy as np
from manim import *

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import BG, PEAK, Z_SET, blurred_box, box, gauss, peak_value, widths  # noqa: E402

# ----------------------------------------------------------------- constants
SIGNAL_COLOR = YELLOW        # same yellow as Scene 2's rod / image
KERNEL_COLOR = BLUE
PRODUCT_COLOR = BLUE_A       # light blue
OUTPUT_COLOR = ORANGE
AXIS_COLOR = GRAY

SIGMA0, SIGMA_END = 0.375, 1.0
DX = 0.125                   # pixel spacing (Beat 1)
N_PIX = 25                   # samples over x in [-1.5, 1.5]
W5 = np.array([0.2] * 5)     # 5-tap averaging kernel
W1 = 0.75                    # Z = 4 width, used in Beats 1-2
SLOW_POSITIONS = [0, 1, 2]   # detailed steps (edges show the zero padding of "same" convolution)
HOP_FROM, HOP_TO = 2, N_PIX - 1   # then one discrete hop per sample up to the last one

# Beat 1 layout (whole block centered near y = -1.33, i.e. 1/3 of the frame height from the bottom)
BIG_X_LEN = 9.0
IN_Y_LEN, IN_BASE = 1.5, -1.05          # input signal axes
OUT_Y_LEN, OUT_BASE = 1.2, -3.75        # output ("same" convolution) axes, below the input
KROW_Y, K_SCALE = 0.7, 2.0              # kernel bar row: baseline y, screen units per unit weight
PROW_Y, P_SCALE = -2.4, 1.5             # product row
SUM_POS = (5.6, -2.25)
# Beat 2: single big axes, centered at y = -1.33
B2_Y_LEN, B2_BASE = 3.0, -2.83
# Beat 3 layout
PANEL_X = (-4.4, 0.0, 4.4)
PANEL_BASE_Y, MINI_BASE_Y = -1.2, -3.2
PANEL_X_LEN, PANEL_Y_LEN, MINI_Y_LEN = 3.6, 2.0, 0.8
TITLE_Y, SAME_Y = 1.5, -3.75

BEAT_LEN = dict(b1=20, b2=18, b3=20, b4=7)    # seconds, total 65
RT = dict(
    axes=2.0, kernel_in=1.0,
    step_move=0.7, step_prod=0.4, step_drop=0.9, step_hold=0.2,
    formula=1.5, hop=8.8,
    fade=1.0, morph=2.0, to_start=1.2, g_formula=1.5, brace=1.0, brace_hold=1.5, slide=7.0,
    panels_in=2.0, blur=1.6, peak=0.6, panel_hold=1.0, kernels_in=1.5, readout=0.8,
    grow=4.0, end_hold=0.5,
)

USE_LATEX = os.environ.get("NO_LATEX") != "1"

# ---------------------------------------------------------------- data (Beat 1)
XS_D = -1.5 + DX * np.arange(N_PIX)
SAMPLES = box(XS_D, W1)


def conv5_zero_pad(s):
    """Explicit 5-tap zero-padded average (what np.convolve 'same' does)."""
    n = len(s)
    out = np.zeros(n)
    for i in range(n):
        for k in range(5):
            j = i - 2 + k
            if 0 <= j < n:
                out[i] += W5[k] * s[j]
    return out


OUT_D = np.convolve(SAMPLES, W5, mode="same")


_trapz = getattr(np, 'trapezoid', None) or np.trapz


def self_check():
    # 1. gauss integrates to 1
    x = np.linspace(-8, 8, 160001)
    for s in (0.2, 0.375, 1.0):
        assert abs(_trapz(gauss(x, s), x) - 1) < 1e-4, s
    # 2. blurred_box == numerical convolution
    dx = 0.001
    xg = np.arange(-7, 7 + dx / 2, dx)
    inner = np.abs(xg) <= 2
    for w in widths():
        for s in (0.375, 1.0):
            num = np.convolve(box(xg, w), gauss(xg, s), mode="same") * dx
            err = np.max(np.abs(num[inner] - blurred_box(xg[inner], w, s)))
            assert err < 2e-3, (w, s, err)
    # 3. peak values
    for w, want in zip(widths(), (0.96, 0.71, 0.45)):
        p = peak_value(w, 0.375)
        assert abs(p - (BG + (PEAK - BG) * math.erf(w / (2 * math.sqrt(2) * 0.375)))) < 1e-12
        assert abs(p - want) < 0.01, (w, p)
    # 4. discrete 5-tap output
    assert np.allclose(OUT_D, np.convolve(SAMPLES, [0.2] * 5, mode="same"))
    assert np.allclose(conv5_zero_pad(SAMPLES), OUT_D)
    assert np.isclose(SAMPLES[9], PEAK) and np.isclose(SAMPLES[3], BG)
    assert abs(XS_D[0] + 1.5) < 1e-12 and abs(XS_D[-1] - 1.5) < 1e-12


# ------------------------------------------------------------------- helpers
def tex(*parts, fb=None):
    """MathTex, or (NO_LATEX=1) a Text fallback with the same part indexing."""
    if USE_LATEX:
        return MathTex(*parts)
    fb = fb or parts
    return VGroup(*[Text(p, font_size=30) for p in fb]).arrange(RIGHT, buff=0.12)


def number_mob(fn, color, font_size=30):
    if USE_LATEX:
        m = DecimalNumber(fn(), num_decimal_places=2, font_size=font_size, color=color)
        m.add_updater(lambda mob: mob.set_value(fn()))
    else:
        m = Text(f"{fn():.2f}", font_size=font_size - 6, color=color)
        m.add_updater(lambda mob: mob.become(Text(f"{fn():.2f}", font_size=font_size - 6, color=color)))
    return m


def make_axes(x_len, y_len, origin):
    ax = Axes(
        x_range=[-2, 2, 1], y_range=[0, 1.2, 1], x_length=x_len, y_length=y_len, tips=False,
        axis_config={"include_ticks": False, "include_numbers": False, "stroke_width": 3},
    )
    ax.shift(np.array([origin[0], origin[1], 0]) - ax.c2p(0, 0))
    return ax


def poly(ax, xs, ys, color, width=5):
    """Sharp-cornered polyline in axes coordinates."""
    m = VMobject(color=color, stroke_width=width)
    m.set_points_as_corners([ax.c2p(x, y) for x, y in zip(xs, ys)])
    return m


def box_pts(w, half=2.0):
    return ([-half, -w / 2, -w / 2, w / 2, w / 2, half],
            [BG, BG, PEAK, PEAK, BG, BG])


XS_FINE = np.linspace(-2, 2, 201)


# -------------------------------------------------------------------- scene
class KernelGaussian1D(Scene):
    def construct(self):
        self_check()

        def pad(t_end):
            dt = t_end - self.time
            if dt > 1e-3:
                self.wait(dt)

        # ================= BEAT 1 (0-20 s) =================
        # "How does a computer examine a signal like this? With a kernel: a small window of
        # weights that slides along the image... multiplies... adds them up... called convolution."
        t0 = self.time
        ax = make_axes(BIG_X_LEN, IN_Y_LEN, (0, IN_BASE))
        out_ax = make_axes(BIG_X_LEN, OUT_Y_LEN, (0, OUT_BASE))

        def axis_labels(a, ylab):
            lab = a.get_axis_labels(tex("x", fb=["x"]).scale(0.7), ylab.scale(0.7))
            # park the y label at the far left so it never collides with the kernel row
            lab[1].move_to(a.c2p(-2.0, 1.2) + UP * 0.1 + RIGHT * 0.1)
            return lab

        ax_labels = axis_labels(ax, tex("I(x)", fb=["I(x)"]))
        out_labels = axis_labels(out_ax, tex(r"I * g", fb=["I * g"]))
        ax.y_axis.set_stroke(opacity=0.5)
        out_ax.y_axis.set_stroke(opacity=0.5)
        dxs = ax.c2p(DX, 0)[0] - ax.c2p(0, 0)[0]          # screen width of one pixel step

        def stem(i):
            x, v = XS_D[i], SAMPLES[i]
            return VGroup(Line(ax.c2p(x, 0), ax.c2p(x, v), color=SIGNAL_COLOR, stroke_width=4),
                          Dot(ax.c2p(x, v), radius=0.05, color=SIGNAL_COLOR))

        stems = VGroup(*[stem(i) for i in range(N_PIX)])
        # "same" convolution: the signal is zero outside [-1.5, 1.5]; show a few zero samples
        zero_dots = VGroup(*[Dot(ax.c2p(-1.5 + DX * i, 0), radius=0.04, color=GRAY)
                             for i in list(range(-4, 0)) + list(range(N_PIX, N_PIX + 4))])
        zero_note = Text("zeros outside the signal", font_size=18, color=GRAY_B).move_to([-5.5, IN_BASE - 0.3, 0])
        kt = ValueTracker(SLOW_POSITIONS[0])               # kernel center, in sample-index units

        def xk():
            return -1.5 + DX * kt.get_value()

        def kernel_bars():
            g = VGroup()
            for j, wgt in zip(range(-2, 3), W5):
                bar = Rectangle(width=dxs * 0.8, height=wgt * K_SCALE, stroke_width=1,
                                color=KERNEL_COLOR, fill_color=KERNEL_COLOR, fill_opacity=0.8)
                bar.move_to([ax.c2p(xk() + j * DX, 0)[0], KROW_Y + wgt * K_SCALE / 2, 0])
                g.add(bar)
            return g

        def highlight():
            top, bot = ax.c2p(0, 1.2)[1], ax.c2p(0, 0)[1]
            return Rectangle(width=5 * dxs, height=top - bot, stroke_width=0, fill_color=KERNEL_COLOR,
                             fill_opacity=0.15).move_to([ax.c2p(xk(), 0)[0], (top + bot) / 2, 0])

        def guide():
            x = ax.c2p(xk(), 0)[0]
            return DashedLine([x, KROW_Y, 0], [x, OUT_BASE, 0], dash_length=0.08, stroke_width=1.5,
                              stroke_opacity=0.3)

        k_label = Text("kernel g", font_size=22, color=KERNEL_COLOR).move_to([-5.9, KROW_Y + 0.2, 0])
        self.play(Create(ax), Create(out_ax), FadeIn(ax_labels), FadeIn(out_labels), Create(stems),
                  FadeIn(zero_dots), run_time=RT["axes"])
        kbars_live = always_redraw(kernel_bars)
        hl_live = always_redraw(highlight)
        guide_live = always_redraw(guide)
        self.add(hl_live, guide_live, kbars_live)
        self.play(FadeIn(kbars_live), FadeIn(hl_live), FadeIn(guide_live), FadeIn(k_label),
                  run_time=RT["kernel_in"])

        slow_dots = VGroup()
        p_label = Text("products", font_size=22, color=PRODUCT_COLOR).move_to([-5.9, PROW_Y + 0.15, 0])
        for step, n in enumerate(SLOW_POSITIONS):
            if abs(kt.get_value() - n) > 1e-9:
                self.play(kt.animate.set_value(n), run_time=RT["step_move"])
            prods = VGroup()
            total = 0.0
            for j in range(5):
                i = n - 2 + j
                if 0 <= i < N_PIX:                      # taps outside the signal multiply zeros
                    p = W5[j] * SAMPLES[i]
                    total += p
                    x = ax.c2p(XS_D[i], 0)[0]
                    prods.add(Line([x, PROW_Y, 0], [x, PROW_Y + p * P_SCALE, 0], color=PRODUCT_COLOR,
                                   stroke_width=5),
                              Dot([x, PROW_Y + p * P_SCALE, 0], radius=0.05, color=PRODUCT_COLOR))
            assert abs(total - OUT_D[n]) < 1e-12
            sum_txt = Text(f"sum = {total:.2f}", font_size=24, color=OUTPUT_COLOR).move_to([*SUM_POS, 0])
            extra = [FadeIn(p_label), FadeIn(zero_note)] if step == 0 else []
            self.play(FadeIn(prods), FadeIn(sum_txt), *extra, run_time=RT["step_prod"])
            dot = Dot(out_ax.c2p(XS_D[n], OUT_D[n]), radius=0.07, color=OUTPUT_COLOR)
            self.play(ReplacementTransform(sum_txt, dot), FadeOut(prods), run_time=RT["step_drop"])
            slow_dots.add(dot)
            self.wait(RT["step_hold"])

        formula = tex(r"(I * g)[n] = \sum_k g[k]\, I[n-k]", fb=["(I * g)[n] = Σ g[k] I[n−k]"]).move_to([0, 3.2, 0])
        note = Text("(flipping g doesn't matter: our kernels are symmetric)", font_size=20, color=GRAY_B)
        note.move_to([0, 2, 0])
        self.play(Write(formula), FadeIn(note), FadeOut(p_label), FadeOut(zero_note), run_time=RT["formula"])

        def running_dots():
            hi = int(math.floor(kt.get_value() + 1e-6))
            return VGroup(*[Dot(out_ax.c2p(XS_D[n], OUT_D[n]), radius=0.07, color=OUTPUT_COLOR)
                            for n in range(0, min(hi, N_PIX - 1) + 1)])

        def hop_rate(m):
            """Staircase: advance exactly one sample per 1/m of the time, easing within each hop."""
            def f(t):
                u = t * m
                k = min(int(math.floor(u)), m - 1)
                r = min((u - k) / 0.6, 1.0)
                return (k + r * r * (3 - 2 * r)) / m
            return f

        run_live = always_redraw(running_dots)
        self.add(run_live)
        self.play(kt.animate.set_value(HOP_TO), run_time=RT["hop"], rate_func=hop_rate(HOP_TO - HOP_FROM))
        pad(t0 + BEAT_LEN["b1"])

        # ================= BEAT 2 (20-38 s) =================
        # "One kernel matters more than any other: the Gaussian, a bell curve whose width is set by
        # a single number, sigma... weighted average of its neighbors... smoothed, blurred signal."
        t0 = self.time
        bax = make_axes(BIG_X_LEN, B2_Y_LEN, (0, B2_BASE))
        bax_labels = axis_labels(bax, tex("I(x)", fb=["I(x)"]))
        bax.y_axis.set_stroke(opacity=0.5)
        box_curve = poly(bax, *box_pts(W1), SIGNAL_COLOR)
        c = ValueTracker(xk())

        def gauss_curve(center, sigma=SIGMA0):
            xs = np.linspace(max(center - 3 * sigma, -2), min(center + 3 * sigma, 2), 120)
            return poly(bax, xs, gauss(xs - center, sigma), KERNEL_COLOR, 5)

        bars_static = kernel_bars()
        self.remove(kbars_live)
        self.add(bars_static)
        g_formula = tex(r"g_\sigma(x) = \frac{1}{\sqrt{2\pi}\,\sigma}\, e^{-x^2/2\sigma^2}",
                        fb=["g_σ(x) = 1/(√(2π) σ) · e^(−x²/2σ²)"]).move_to([0, 2.4, 0])
        self.play(
            *[FadeOut(m) for m in list(self.mobjects) if m is not bars_static],
            FadeIn(bax), FadeIn(bax_labels), Create(box_curve),
            Transform(bars_static, gauss_curve(c.get_value())),
            run_time=RT["morph"],
        )
        self.remove(bars_static)
        g_live = always_redraw(lambda: gauss_curve(c.get_value()))
        self.add(g_live)
        self.play(Write(g_formula), run_time=RT["g_formula"])

        # sigma brace: one standard deviation, measured from the kernel center
        cx = c.get_value()
        brace = BraceBetweenPoints(bax.c2p(cx, 0), bax.c2p(cx + SIGMA0, 0), direction=DOWN, buff=0.08)
        sig_lab = tex(r"\sigma", fb=["σ"]).scale(0.8).next_to(brace, DOWN, buff=0.1)
        guides = VGroup(
            DashedLine(bax.c2p(cx, 0), bax.c2p(cx, float(gauss(0, SIGMA0))), dash_length=0.08, stroke_width=2),
            DashedLine(bax.c2p(cx + SIGMA0, 0), bax.c2p(cx + SIGMA0, float(gauss(SIGMA0, SIGMA0))),
                       dash_length=0.08, stroke_width=2),
        ).set_opacity(0.7)
        self.play(FadeIn(brace), FadeIn(sig_lab), Create(guides), run_time=RT["brace"])
        self.wait(RT["brace_hold"])
        self.play(FadeOut(brace), FadeOut(sig_lab), FadeOut(guides), run_time=0.5)

        self.play(c.animate.set_value(-2.0), run_time=RT["to_start"], rate_func=smooth)

        def trace():
            hi = max(c.get_value(), -2 + 1e-3)
            xs = np.linspace(-2, hi, max(2, int((hi + 2) / 0.02) + 1))
            return poly(bax, xs, blurred_box(xs, W1, SIGMA0), OUTPUT_COLOR, 5)

        trace_live = always_redraw(trace)
        self.add(trace_live)
        self.play(c.animate.set_value(2.0), run_time=RT["slide"], rate_func=linear)
        pad(t0 + BEAT_LEN["b2"])

        # ================= BEAT 3 (38-58 s) =================
        # "But watch what one fixed sigma does to our three images. The near object's wide bump barely
        # changes. The far object's narrow bump gets flattened... A fixed-size kernel is biased toward
        # one size."
        t0 = self.time
        sg = ValueTracker(SIGMA0)
        panels = []

        def make_panel(Z, w, xc):
            pax = make_axes(PANEL_X_LEN, PANEL_Y_LEN, (xc, PANEL_BASE_Y))
            plabels = pax.get_axis_labels(tex("x", fb=["x"]).scale(0.7), tex("I(x)", fb=["I(x)"]).scale(0.7))
            title = tex(f"Z = {Z}", fb=[f"Z = {Z}"]).scale(0.8).move_to([xc, TITLE_Y, 0])
            box_p = poly(pax, *box_pts(w), SIGNAL_COLOR)

            def blur():
                return poly(pax, XS_FINE, blurred_box(XS_FINE, w, sg.get_value()), OUTPUT_COLOR)

            def peak_line():
                p = peak_value(w, sg.get_value())
                return DashedLine(pax.c2p(0, p), pax.c2p(2, p), dash_length=0.1, stroke_width=2,
                                  color=OUTPUT_COLOR, stroke_opacity=0.7)

            peak_num = number_mob(lambda: peak_value(w, sg.get_value()), OUTPUT_COLOR, 26)
            peak_num.add_updater(lambda m: m.move_to(pax.c2p(1.5, peak_value(w, sg.get_value())) + UP * 0.25))
            peak_num.update()

            mini = make_axes(PANEL_X_LEN, MINI_Y_LEN, (xc, MINI_BASE_Y))

            def kern():
                s = sg.get_value()
                xs = np.linspace(-2, 2, 201)
                return poly(mini, xs, gauss(xs, s), KERNEL_COLOR, 4)

            return dict(ax=pax, labels=plabels, title=title, box=box_p, blur=blur, peak_line=peak_line,
                        peak_num=peak_num, mini=mini, kern=kern)

        for Z, w, xc in zip(Z_SET, widths(), PANEL_X):
            panels.append(make_panel(Z, w, xc))

        self.play(
            *[FadeOut(m) for m in list(self.mobjects)],
            run_time=RT["fade"],
        )
        self.play(*[FadeIn(VGroup(p["ax"], p["labels"], p["title"], p["box"])) for p in panels],
                  run_time=RT["panels_in"])
        for p in panels:                       # near -> far
            blur_static = p["blur"]()
            self.play(TransformFromCopy(p["box"], blur_static), run_time=RT["blur"])
            self.remove(blur_static)
            self.add(always_redraw(p["blur"]))
            pl = always_redraw(p["peak_line"])
            self.add(pl, p["peak_num"])
            self.play(FadeIn(pl), FadeIn(p["peak_num"]), run_time=RT["peak"])
            self.wait(RT["panel_hold"])
        same = Text("same kernel", font_size=22, color=KERNEL_COLOR).move_to([0, SAME_Y, 0])
        self.play(*[FadeIn(VGroup(p["mini"], always_redraw(p["kern"]))) for p in panels], FadeIn(same),
                  run_time=RT["kernels_in"])

        sig_lab = Text("σ =", font_size=26)
        sig_num = number_mob(lambda: sg.get_value(), WHITE, 30)
        sig_num.add_updater(lambda m: m.next_to(sig_lab, RIGHT, buff=0.12))
        sig_lab.move_to([5.4, 3.4, 0])
        sig_num.update()
        self.play(FadeIn(sig_lab), FadeIn(sig_num), run_time=RT["readout"])
        pad(t0 + BEAT_LEN["b3"])

        # ================= BEAT 4 (58-65 s) =================
        # "So what if sigma isn't fixed? What happens as we let it grow?"
        t0 = self.time
        self.wait(2.0)
        self.play(sg.animate.set_value(SIGMA_END), run_time=RT["grow"], rate_func=linear)
        self.wait(RT["end_hold"])      # stop mid-growth: no resolution, next scene picks up here
        pad(t0 + BEAT_LEN["b4"])


# ---------------------------------------------------------------------------
# Docs / source consulted (manim==0.21.0; https://docs.manim.community/en/stable/)
#  - Axes, get_axis_labels, c2p:
#      https://docs.manim.community/en/stable/reference/manim.mobject.graphing.coordinate_systems.Axes.html
#  - VMobject.set_points_as_corners (sharp polylines):
#      https://docs.manim.community/en/stable/reference/manim.mobject.types.vectorized_mobject.VMobject.html
#  - BraceBetweenPoints:
#      https://docs.manim.community/en/stable/reference/manim.mobject.svg.brace.BraceBetweenPoints.html
#  - always_redraw, ValueTracker:
#      https://docs.manim.community/en/stable/reference/manim.animation.updaters.update.html
#  - DecimalNumber:
#      https://docs.manim.community/en/stable/reference/manim.mobject.text.numbers.DecimalNumber.html
#  - TransformFromCopy, ReplacementTransform:
#      https://docs.manim.community/en/stable/reference/manim.animation.transform.html
#  - Scene.time:
#      https://docs.manim.community/en/stable/reference/manim.scene.scene.Scene.html

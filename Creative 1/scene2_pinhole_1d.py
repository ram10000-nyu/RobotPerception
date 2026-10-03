"""
Scene 2 — 1D pinhole projection (SIFT scale-space video).

Render:
    manim -pql scene2_pinhole_1d.py Pinhole1D     # preview (480p15)
    manim -qh  scene2_pinhole_1d.py Pinhole1D     # final   (1080p60)

Geometry (class convention): camera center / pinhole at (PX, 0); the image plane sits
BETWEEN the camera and the object, at x = PX + F, so the image is upright; the rod of
length L is at depth Z (x = PX + Z). Rays run from the rod ends to the camera center and
cross the image plane on the way. Every number shown is computed from ray/plane
intersections, never hardcoded:  s = F * L / Z.

Set NO_LATEX=1 only to preview on a machine without a LaTeX install (text fallbacks).
"""
import os

import numpy as np
from manim import *

# ----------------------------------------------------------------- constants
PX = -3.0                    # pinhole x
F = 1.5                      # focal length (pinhole -> image plane, in front of it)
Z0, L0 = 3.0, 2.0            # initial rod depth / length
Z_FAR = 6.0                  # Beat 3 target depth
ROD2_Z, ROD2_L = 6.0, 4.0    # Beat 4 second rod
Z_SET = [2, 4, 8]            # Scene 3 inputs (Beat 5)
L_SET = 2.0
BG_LEVEL, BUMP_LEVEL = 0.1, 1.0

ROD_COLOR = YELLOW
ROD2_COLOR = TEAL
RAY_COLOR, RAY_OPACITY, RAY_WIDTH = WHITE, 0.55, 2
AXIS_COLOR = GRAY
ROD_WIDTH = 8
GHOST_OPACITY = 0.3
WALL_GAP, WALL_TOP = 0.12, 0.7
IMG_HALF = 2.0               # drawn half-length of the image line
BRACE_Y_F = -1.3             # f brace hangs below the axis here
BRACE_Y_Z = -2.3             # Z brace hangs lower so it clears the f brace and its label

BEAT_LEN = dict(b1=10, b2=15, b3=8, b4=12, b5=10)   # seconds, total 55
RT = dict(
    axis=1.2, pinhole=1.2, image_line=1.2, rod=1.5,
    ray=1.5, image_seg=1.0, tri_in=1.0, tri_hold=1.0, tri_out=1.0,
    formula=1.5, braces=1.5,
    move_z=3.0,
    back_z=1.5, ghost=0.8, rod2=1.5, rays2=1.5, flash=1.0, ratio=1.5,
    fade_all=1.0, rotate=1.5, to_axis=1.5, axes_extra=1.0, bump=1.0,
    to_three=2.0, hold=2.0,
)

USE_LATEX = os.environ.get("NO_LATEX") != "1"


# ------------------------------------------------------------ pure geometry
def pinhole_pt():
    return np.array([PX, 0.0, 0.0])


def rod_end(Z, L, sign):
    return np.array([PX + Z, sign * L / 2.0, 0.0])


def ray_hit(e):
    """Intersect the ray from endpoint e to the pinhole with the image plane x = PX + F."""
    p = pinhole_pt()
    t = (PX + F - e[0]) / (p[0] - e[0])
    return e + t * (p - e)


def image_ends(Z, L):
    return ray_hit(rod_end(Z, L, +1)), ray_hit(rod_end(Z, L, -1))


def image_length(Z, L):
    a, b = image_ends(Z, L)
    return abs(a[1] - b[1])


def bump_xy(Z, half=2.0):
    """Polyline of the box-shaped I(x): height 1 on background 0.1, width F*L_SET/Z."""
    w = F * L_SET / Z
    xs = [-half, -w / 2, -w / 2, w / 2, w / 2, half]
    ys = [BG_LEVEL, BG_LEVEL, BUMP_LEVEL, BUMP_LEVEL, BG_LEVEL, BG_LEVEL]
    return xs, ys


def self_check():
    for Z in (2, 3, 4, 6, 8):
        for L in (2, 4):
            top, bot = image_ends(Z, L)
            assert abs(abs(top[1] - bot[1]) - F * L / Z) < 1e-9, (Z, L)
            assert abs(top[1] - (F * (L / 2) / Z)) < 1e-9, (Z, L)      # upright image
            assert abs(top[0] - (PX + F)) < 1e-12
            assert PX < top[0] < PX + Z                                  # plane between camera and object
            # ray (extended) passes exactly through the pinhole
            e = rod_end(Z, L, +1)
            cross = (top - e)[0] * (pinhole_pt() - e)[1] - (top - e)[1] * (pinhole_pt() - e)[0]
            assert abs(cross) < 1e-9
            assert PX + Z <= 5.0 and L / 2 <= 2.0          # frame budget
    a1, a2 = image_ends(3, 2)
    b1, b2 = image_ends(6, 4)
    assert np.allclose(a1, b1, atol=1e-9) and np.allclose(a2, b2, atol=1e-9)
    for Z in Z_SET:
        xs, _ = bump_xy(Z)
        assert abs((xs[3] - xs[2]) - F * L_SET / Z) < 1e-9
    assert [F * L_SET / Z for Z in Z_SET] == [1.5, 0.75, 0.375]


# ------------------------------------------------------------------ helpers
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


def make_readout(zt, lt, color):
    labels = VGroup(*[Text(t, font_size=24, color=color) for t in ("Z =", "L =", "s = fL/Z =")])
    labels.arrange(DOWN, aligned_edge=LEFT, buff=0.22)
    fns = [zt.get_value, lt.get_value, lambda: image_length(zt.get_value(), lt.get_value())]
    nums = VGroup(*[number_mob(f, color) for f in fns])

    def place(i):
        def upd(m):
            col = max(l.get_right()[0] for l in labels) + 0.12
            m.move_to([col, labels[i].get_center()[1], 0], aligned_edge=LEFT)
        return upd

    for i, n in enumerate(nums):
        n.add_updater(place(i))
        place(i)(n)
    return VGroup(labels, nums)


# mobject factories: each returns a zero-arg builder reading the trackers
def rod_f(zt, lt, color):
    return lambda: Line(rod_end(zt.get_value(), lt.get_value(), +1),
                        rod_end(zt.get_value(), lt.get_value(), -1),
                        color=color, stroke_width=ROD_WIDTH)


def ray_f(zt, lt, sign, color=RAY_COLOR):
    def build():
        e = rod_end(zt.get_value(), lt.get_value(), sign)
        return DashedLine(e, pinhole_pt(), dash_length=0.12, stroke_width=RAY_WIDTH,
                          stroke_opacity=RAY_OPACITY, color=color)
    return build


def img_f(zt, lt, color, width=ROD_WIDTH):
    def build():
        a, b = image_ends(zt.get_value(), lt.get_value())
        return Line(a, b, color=color, stroke_width=width)
    return build


# -------------------------------------------------------------------- scene
class Pinhole1D(Scene):
    def construct(self):
        self_check()
        zt, lt = ValueTracker(Z0), ValueTracker(L0)
        P = pinhole_pt()

        def reveal(f, run_time, anim=Create):
            """Animate a static copy in, then swap it for the live always_redraw version."""
            static = f()
            self.play(anim(static), run_time=run_time)
            self.remove(static)
            live = always_redraw(f)
            self.add(live)
            return live

        def pad(t_end):
            dt = t_end - self.time
            if dt > 1e-3:
                self.wait(dt)

        # BEAT 1 (0-10 s): "Squash the world into a flat plane, and the camera into a single
        # point with a line behind it: a one-dimensional image."
        t0 = self.time
        axis = Line([-4.0, 0, 0], [6.0, 0, 0], color=AXIS_COLOR, stroke_width=2)
        wall = VGroup(
            Line([PX, WALL_GAP, 0], [PX, WALL_TOP, 0], stroke_width=5),
            Line([PX, -WALL_GAP, 0], [PX, -WALL_TOP, 0], stroke_width=5),
        )
        dot = Dot(P, radius=0.06, color=WHITE)
        img_line = Line([PX + F, -IMG_HALF, 0], [PX + F, IMG_HALF, 0], color=AXIS_COLOR, stroke_width=4)
        img_label = Text("image plane", font_size=22, color=GRAY_B).next_to(img_line, UP, buff=0.15)
        self.play(Create(axis), run_time=RT["axis"])
        self.play(Create(wall), FadeIn(dot), run_time=RT["pinhole"])
        self.play(Create(img_line), FadeIn(img_label), run_time=RT["image_line"])
        readout = make_readout(zt, lt, ROD_COLOR).to_corner(UR, buff=0.4)
        rod = reveal(rod_f(zt, lt, ROD_COLOR), RT["rod"])
        self.play(FadeIn(readout), run_time=0.8)
        pad(t0 + BEAT_LEN["b1"])

        # BEAT 2 (10-25 s): "Light from the top and bottom of an object passes through the
        # pinhole... similar triangles... image of length f times L over Z."
        t0 = self.time
        ray_top = reveal(ray_f(zt, lt, +1), RT["ray"])
        ray_bot = reveal(ray_f(zt, lt, -1), RT["ray"])
        img_seg = reveal(img_f(zt, lt, ROD_COLOR), RT["image_seg"])

        def tri(sign_pair_fn, color):
            return Polygon(*sign_pair_fn(), color=color, fill_color=color, fill_opacity=0.28, stroke_width=0)

        Z, L = zt.get_value(), lt.get_value()
        tri_obj = tri(lambda: [P, rod_end(Z, L, +1), rod_end(Z, L, -1)], BLUE)
        tri_img = tri(lambda: [P, *image_ends(Z, L)], ORANGE)
        self.play(FadeIn(tri_obj), FadeIn(tri_img), run_time=RT["tri_in"])
        self.wait(RT["tri_hold"])
        self.play(FadeOut(tri_obj), FadeOut(tri_img), run_time=RT["tri_out"])

        formula = tex(r"s = \frac{f\,L}{Z}", fb=["s = f L / Z"]).move_to([-1.3, 3.0, 0])
        self.play(Write(formula), run_time=RT["formula"])

        def L_brace():
            return BraceBetweenPoints(rod_end(zt.get_value(), lt.get_value(), -1),
                                      rod_end(zt.get_value(), lt.get_value(), +1),
                                      direction=RIGHT, buff=0.12)

        def s_brace():
            a, b = image_ends(zt.get_value(), lt.get_value())
            lo, hi = sorted([a, b], key=lambda p: p[1])
            return BraceBetweenPoints(lo, hi, direction=RIGHT, buff=0.12)

        def Z_brace():
            return BraceBetweenPoints([PX, BRACE_Y_Z, 0], [PX + zt.get_value(), BRACE_Y_Z, 0],
                                      direction=DOWN, buff=0.05)

        f_brace = BraceBetweenPoints([PX, BRACE_Y_F, 0], [PX + F, BRACE_Y_F, 0], direction=DOWN, buff=0.05)
        bL, bs, bZ = always_redraw(L_brace), always_redraw(s_brace), always_redraw(Z_brace)
        lab_L = tex("L", fb=["L"]).scale(0.8)
        lab_s = tex("s", fb=["s"]).scale(0.8)
        lab_Z = tex("Z", fb=["Z"]).scale(0.8)
        lab_f = tex("f", fb=["f"]).scale(0.8)
        lab_L.add_updater(lambda m: m.next_to(bL, RIGHT, buff=0.1))
        lab_s.add_updater(lambda m: m.next_to(bs, RIGHT, buff=0.1))
        lab_Z.add_updater(lambda m: m.next_to(bZ, DOWN, buff=0.1))
        lab_f.next_to(f_brace, DOWN, buff=0.1)
        braces = [bL, bs, bZ, f_brace]
        labels = [lab_L, lab_s, lab_Z, lab_f]
        self.play(*[FadeIn(m) for m in braces + labels], run_time=RT["braces"])
        pad(t0 + BEAT_LEN["b2"])

        # BEAT 3 (25-33 s): "Double the distance, and the image halves."
        t0 = self.time
        ghost_img = img_f(zt, lt, ROD_COLOR)().set_opacity(GHOST_OPACITY)
        self.add(ghost_img)
        self.play(zt.animate.set_value(Z_FAR), run_time=RT["move_z"], rate_func=smooth)
        pad(t0 + BEAT_LEN["b3"])

        # BEAT 4 (33-45 s): "A small object nearby and a large one far away give the same
        # image... Size and distance are tangled together, and we won't untangle them."
        t0 = self.time
        self.remove(ghost_img)
        self.play(
            zt.animate.set_value(Z0),
            FadeOut(formula), *[FadeOut(m) for m in braces + labels],
            run_time=RT["back_z"], rate_func=smooth,
        )
        # freeze rod 1 as a ghost (clear live updaters, then dim)
        ghosts = [rod, ray_top, ray_bot]
        for g in ghosts:
            g.clear_updaters()
        self.play(*[g.animate.set_opacity(GHOST_OPACITY) for g in ghosts], run_time=RT["ghost"])

        z2, l2 = ValueTracker(ROD2_Z), ValueTracker(ROD2_L)
        readout2 = make_readout(z2, l2, ROD2_COLOR).next_to(readout, DOWN, buff=0.25, aligned_edge=RIGHT)
        # teal halo behind the yellow image, so the coincidence is visible
        halo = reveal(img_f(z2, l2, ROD2_COLOR, width=ROD_WIDTH + 8), 0.01)
        halo.set_z_index(img_seg.z_index - 1)
        img_seg.set_z_index(5)
        rod2 = reveal(rod_f(z2, l2, ROD2_COLOR), RT["rod2"])
        self.play(FadeIn(readout2), run_time=0.5)
        reveal(ray_f(z2, l2, +1, ROD2_COLOR), RT["rays2"] / 2)
        reveal(ray_f(z2, l2, -1, ROD2_COLOR), RT["rays2"] / 2)

        a, b = image_ends(ROD2_Z, ROD2_L)
        c1, c2 = image_ends(Z0, L0)
        assert np.allclose(a, c1) and np.allclose(b, c2)
        seg_copy = Line(c1, c2, color=ROD_COLOR, stroke_width=ROD_WIDTH).set_z_index(10)
        same = Text("same image", font_size=26, color=WHITE).move_to([PX + F - 0.4, 1.45, 0])
        arrow = Arrow([PX + F, 1.2, 0], [PX + F, c1[1] + 0.12, 0], buff=0,
                      stroke_width=3, max_tip_length_to_length_ratio=0.3, color=WHITE)
        self.play(Indicate(seg_copy, color=WHITE, scale_factor=1.3),
                  Flash(np.array([PX + F, 0, 0]), color=WHITE, flash_radius=0.5),
                  FadeIn(same), GrowArrow(arrow), run_time=RT["flash"])
        self.remove(seg_copy)
        ratio = tex(r"\frac{L_1}{Z_1}", "=", r"\frac{L_2}{Z_2}", fb=["L1/Z1", "=", "L2/Z2"])
        ratio[0].set_color(ROD_COLOR)
        ratio[2].set_color(ROD2_COLOR)
        ratio.move_to([-1.3, 3.0, 0])
        self.play(Write(ratio), run_time=RT["ratio"])
        pad(t0 + BEAT_LEN["b4"])

        # BEAT 5 (45-55 s): "It's just brightness along a line: a bright bump whose width is
        # set by f times L over Z."
        t0 = self.time
        self.play(*[FadeOut(m) for m in list(self.mobjects) if m is not img_line], run_time=RT["fade_all"])

        def make_signal(Z, k=1.0, title=True):
            ax = Axes(
                x_range=[-2, 2, 1], y_range=[0, 1.2, 1],
                x_length=3.6 * k, y_length=2.0 * k, tips=False,
                axis_config={"include_ticks": False, "include_numbers": False, "stroke_width": 3},
            )
            lbl = ax.get_axis_labels(tex("x", fb=["x"]).scale(0.7 * k), tex("I(x)", fb=["I(x)"]).scale(0.7 * k))
            xs, ys = bump_xy(Z)
            graph = ax.plot_line_graph(xs, ys, add_vertex_dots=False, line_color=ROD_COLOR,
                                       stroke_width=5)["line_graph"]
            core = VGroup(ax, lbl, graph)
            ttl = None
            if title:
                ttl = tex(f"Z = {Z}", fb=[f"Z = {Z}"]).scale(0.8).next_to(ax.x_axis, DOWN, buff=0.3)
            return core, ax, ttl

        big, big_ax, _ = make_signal(Z_SET[1], k=1.6, title=False)
        big.move_to([0, 0.2, 0])
        target = big_ax.x_axis.get_center()
        self.play(img_line.animate.rotate(PI / 2).move_to(target), run_time=RT["rotate"])
        self.play(Transform(img_line, big_ax.x_axis), run_time=RT["to_axis"])
        self.remove(img_line)
        self.add(big_ax.x_axis)
        self.play(Create(big_ax.y_axis), FadeIn(big[1]), run_time=RT["axes_extra"])
        self.play(Create(big[2]), run_time=RT["bump"])

        smalls, titles = [], []
        for Z, xc in zip(Z_SET, (-4.4, 0.0, 4.4)):
            core, _, ttl = make_signal(Z)
            grp = VGroup(core, ttl).move_to([xc, -0.2, 0])
            smalls.append(core)
            titles.append(ttl)
        self.play(
            ReplacementTransform(big, smalls[1]),
            FadeIn(smalls[0]), FadeIn(smalls[2]), *[FadeIn(t) for t in titles],
            run_time=RT["to_three"],
        )
        self.wait(RT["hold"])


# ---------------------------------------------------------------------------
# API looked up (source of the installed manim==0.21.0; docs: https://docs.manim.community/en/stable/)
#  - BraceBetweenPoints(point_1, point_2, direction, **kw):
#      https://docs.manim.community/en/stable/reference/manim.mobject.svg.brace.BraceBetweenPoints.html
#  - Axes.plot_line_graph(x_values, y_values, add_vertex_dots, line_color, ...) -> VDict["line_graph"]:
#      https://docs.manim.community/en/stable/reference/manim.mobject.graphing.coordinate_systems.Axes.html
#  - Axes.get_axis_labels(x_label, y_label):  same page as above
#  - Scene.time (elapsed animation time):
#      https://docs.manim.community/en/stable/reference/manim.scene.scene.Scene.html
#  - DecimalNumber(number, num_decimal_places, ...), .set_value:
#      https://docs.manim.community/en/stable/reference/manim.mobject.text.numbers.DecimalNumber.html
#  - always_redraw, DashedLine, Indicate, Flash, ReplacementTransform:
#      https://docs.manim.community/en/stable/reference/manim.animation.indication.html

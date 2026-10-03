# Creative 1: How Does a Computer Know How Big Something Is?

A ~7 minute, 3Blue1Brown-style [Manim](https://docs.manim.community/en/stable/) video for the robot-perception course, arguing that SIFT's scale-space detector falls straight out of the heat equation. This folder holds the animation code for the first scenes, the shared style guide, and the original proposal.

## Contents

| File | What it is |
|---|---|
| `scene2_pinhole_1d.py` | Scene 2, `Pinhole1D`: 1D pinhole camera, image size s = f·L/Z, size/distance ambiguity, then a brightness plot I(x) |
| `scene3_kernel_gaussian.py` | Scene 3, `KernelGaussian1D`: convolution as a sliding weighted sum, the Gaussian kernel, one fixed σ vs. three object sizes |
| `scene4_blur_is_heat.py` | Scene 4, `BlurIsHeat`: blur as heat diffusion, heat equation, σ = √(2t), small spots cool first |
| `common.py` | Shared constants and exact signal functions (boxes, blurred boxes, FTCS heat solver, half-life); pure numpy |
| `formatting.md` | Style guide (also reproduced below) |
| `requirements.txt` | Pinned Python packages (Manim Community v0.21.0) |

Scenes 1 and 5 to 9 are not written yet. Each scene file has a `self_check()` that asserts its geometry and numerics before rendering.

**Render** (with the environment from the setup section below active):

```bash
manim -pql scene2_pinhole_1d.py Pinhole1D            # preview 480p15
manim -pql scene3_kernel_gaussian.py KernelGaussian1D
manim -pql scene4_blur_is_heat.py BlurIsHeat
manim -qh  scene4_blur_is_heat.py BlurIsHeat         # final 1080p60
```

`NO_LATEX=1 manim ...` gives a plain-text preview on machines without LaTeX (final renders need real LaTeX).

**Status vs. the proposal:** Scene 2 now puts the image plane *between* the camera and the object (upright image, class convention) instead of a line behind the pinhole as the proposal's script says; its narration should say "a line in front of it". Scene 3 uses "same" convolution (zero padding), and Scene 4 asserts box separation at t = 0.2 rather than t = 0.7, where the boxes have merged.

## High-level proposal

**Premise.** Blurring an image with a Gaussian of width σ is the same as letting heat diffuse for time t = σ²/2. Blobs are hot spots, and each one "cools" fastest at the scale matching its size. SIFT's difference-of-Gaussians detector is therefore automatically scale-normalized, "for free", rather than merely approximating the Laplacian of Gaussian.

**Why this topic.** It underlies week 5 (feature detection) and the maze agent's RootSIFT → VLAD front end. Existing videos (Computerphile, First Principles of CV) walk through SIFT's steps but mostly assert "DoG approximates LoG"; this one shows why.

**Approach.** 1D first (curves), then 2D images, with only one 3D scene. This keeps render cost low for the 10/14 deadline. Numerics are ported from the team's verified convolution tutor artifact (Gaussian, Laplacian, DoG stack, circles test image).

**Storyboard** (nine scenes, about 7.3 min):

| # | Scene | Key idea on screen |
|---|---|---|
| 1 | Hook | Same object at three distances; how can features be scale invariant? |
| 2 | 1D pinhole | Similar triangles give s = fL/Z; only L/Z survives; the image becomes brightness I(x) |
| 3 | Kernels & Gaussian | Convolution as a sliding weighted sum; one fixed σ treats near and far bumps differently |
| 4 | Blur is heat | Brightness as temperature; heat equation; Gaussian blur = diffusion for t = σ²/2 |
| 5 | Measuring size | ∇² as cooling rate; raw LoG response only decays; σ²∇² peaks at each blob's own scale |
| 6 | DoG for free | ∂G/∂σ = σ∇²G, so G(kσ) − G(σ) ≈ (k−1)σ²∇²G |
| 7 | 2D circles | DoG stack over σ; each circle lights up at σ ≈ r/√2 |
| 8 | Keypoints | Extremum in (x, y, σ) against its 26 neighbors |
| 9 | Coda | Gradient histogram → descriptor → maze agent; blank walls yield no keypoints |

**Core concepts to study first:** convolution and the Gaussian kernel (semigroup property), image derivatives and the Laplacian of Gaussian, heat-equation basics, scale-normalized derivatives (why multiply by σ²), difference of Gaussians, SIFT detection (octaves, 26-neighbor extremum, edge and low-contrast rejection), and optionally the SIFT descriptor and RootSIFT.

**Primary references:** Lowe, IJCV 2004 (DoG ≈ (k−1)σ²∇²G); Lindeberg, IJCV 1998 (γ-normalization, the r/√2 disk result); Koenderink 1984 (scale space as diffusion); Arandjelović & Zisserman, CVPR 2012 (RootSIFT). The full bibliography is in the proposal below.

---

# Style guide (formatting.md)

Everything here is taken from `scene2_pinhole_1d.py` (Manim Community v0.21.0, `from manim import *`). Copy these values so our scenes look like one video.

### Environment setup (macOS)

Put `requirements.txt` (shared with this file) in your project folder, then run:

```bash
# 1. System libraries Manim needs to build (once per machine)
brew install pkg-config cairo pango ffmpeg

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install the pinned Python packages (Manim 0.21.0, numpy, ...)
pip install -r requirements.txt

# 4. Check it works
manim --version        # should print: Manim Community v0.21.0
```

Then install LaTeX (see "Setup details" below), and render with `manim -pql file.py SceneName` while the environment is active. Each new terminal needs `source .venv/bin/activate` again, or call `.venv/bin/manim` directly.

If `pip install` fails on `pycairo` or `ManimPango`, step 1 didn't finish. Re-run it and try again.

### Setup details

- Renderer: Manim Community **v0.21.0**. Do not use ManimGL / 3b1b-manim names (`ShowCreation`, `TextMobject`, `TexMobject`, `GraphScene`). Use `Create`, `Tex`, `MathTex`, `Axes.plot`.
- LaTeX is required for `MathTex` and `DecimalNumber`. On macOS: `brew install --cask basictex`, then
  `sudo tlmgr install standalone preview doublestroke relsize ms setspace rsfs wasysym physics dvisvgm xcolor fundamental babel-english cm-super`
- Frame: default Manim frame, 14.2 x 8 units. Keep everything inside it with a margin (x within about +/-6.7, y within about +/-3.6).
- Background: default black. Do not change it.
- Render: `manim -pql file.py SceneName` (preview, 480p15) and `manim -qh file.py SceneName` (final, 1080p60).

### Colors

| Role | Manim constant | Use |
|---|---|---|
| Main object (rod, its image, signal curve) | `YELLOW` | Thick lines, the thing we are talking about |
| Second / comparison object | `TEAL` | Anything contrasted with the main object |
| Rays, helper lines | `WHITE`, stroke opacity 0.55 | Thin and dashed |
| Axes, guide lines (optical axis, image line) | `GRAY` | Dim, never competes with the object |
| Small labels such as "image" | `GRAY_B` | |
| Highlight fill 1 (similar triangle 1) | `BLUE`, fill opacity 0.28, no stroke | Fades out after about 1 s |
| Highlight fill 2 (similar triangle 2) | `ORANGE`, fill opacity 0.28, no stroke | |
| Text, braces, pinhole wall and dot | `WHITE` (default) | |
| "Ghost" (old state kept for comparison) | same color, opacity **0.3** | Use `.animate.set_opacity(0.3)` |

Rule: one color per object, kept the same in the drawing, the readout and the equation. For example, in `L1/Z1 = L2/Z2` the left side is yellow and the right side is teal.

### Line weights

| Element | Setting |
|---|---|
| Main object / image segment | `stroke_width=8` (use the same value for both) |
| Rays | `DashedLine(..., dash_length=0.12, stroke_width=2, stroke_opacity=0.55)` |
| Optical axis | `stroke_width=2`, `GRAY` |
| Image line (the thin guide) | `stroke_width=4`, `GRAY` |
| Pinhole wall | `stroke_width=5`, with a gap of 0.12 on each side of the axis |
| Pinhole dot | `Dot(radius=0.06, color=WHITE)` |
| Chart axes (`Axes`) | `stroke_width=3`, no ticks, no numbers, `tips=False` |
| Signal / bump curve | `stroke_width=5`, `YELLOW`, sharp corners |

### Typography

The scene uses Manim's default fonts: no custom font is set, so don't set one. Sizes:

- Equations: `MathTex(...)` at default size (for example `s = \frac{f\,L}{Z}`), placed at the top of the frame.
- Brace labels (`L`, `s`, `Z`, `f`): `MathTex(...).scale(0.8)`, `buff=0.1` from the brace.
- Axis labels (`x`, `I(x)`): `MathTex(...).scale(0.7)`. Larger axes scale it up with the axes.
- Panel titles ("Z = 2"): `MathTex(...).scale(0.8)`, 0.3 below the axis.
- "same image" callout: `Text(..., font_size=26, color=WHITE)` plus a thin white `Arrow`.
- Small tag ("image"): `Text(..., font_size=22, color=GRAY_B)`.
- Readout panel: `Text(..., font_size=24)` for the labels and `DecimalNumber(..., num_decimal_places=2, font_size=30)` for the values. Pin it to the top right with `.to_corner(UR, buff=0.4)`. Rows are `Z =`, `L =`, `s = fL/Z =`. Values are in the object's color and update live.
- Keep on-screen text minimal; the narration carries the explanation.

### Layout and motion

- Diagram: horizontal axis through the middle. Pinhole (camera center) at x = -3, image plane 1.5 to its right (between camera and object, so the image is upright), object further right. Rays run from the object ends to the pinhole.
- Equations sit at the top center-left (`[-1.3, 3.0, 0]`); the readout sits top right; labels and braces hang below the diagram (y = -1.3 for the f brace, y = -2.3 for the Z brace).
- Braces: `BraceBetweenPoints(p1, p2, direction=..., buff=0.12)` with a label next to each, small.
- Every shown number is computed from the geometry, never hardcoded. Drive values with `ValueTracker` and `always_redraw` so that mobjects and readouts stay consistent while animating.
- Create things with `Create` or `FadeIn` (about 1 to 1.5 s each); move values with `rate_func=smooth` (about 3 s for a big change); leave highlights up for about 1 s.
- Highlight a shared or key element with `Indicate(...)` plus `Flash(...)` in `WHITE`.
- Put every tunable (geometry, colors, run times) as named constants at the top of the file, and put `# BEAT n` comments quoting the narration line before each beat.

### Reusable constants (copy-paste)

```python
ROD_COLOR   = YELLOW        # main object
ROD2_COLOR  = TEAL          # comparison object
RAY_COLOR, RAY_OPACITY, RAY_WIDTH = WHITE, 0.55, 2
AXIS_COLOR  = GRAY
ROD_WIDTH   = 8
GHOST_OPACITY = 0.3
```

### Reusable helper (readable text with or without LaTeX)

```python
USE_LATEX = os.environ.get("NO_LATEX") != "1"   # NO_LATEX=1 gives a plain-text preview

def tex(*parts, fb=None):
    """MathTex, or a Text fallback with the same part indexing."""
    if USE_LATEX:
        return MathTex(*parts)
    fb = fb or parts
    return VGroup(*[Text(p, font_size=30) for p in fb]).arrange(RIGHT, buff=0.12)
```

Run with `NO_LATEX=1` only for a quick preview on a machine without LaTeX. Final renders must use real LaTeX.

---

# Full proposal

*Reproduced verbatim from the proposal document (headings shifted down one level).*

## Creative-1 Proposal: How Does a Computer Know How Big Something Is?

Sep 28, 2026 · @Rolando

### Premise

A \~7 min Manim video (3Blue1Brown style) showing that SIFT's scale-space detector falls straight out of the heat equation. Blurring an image with a Gaussian of width σ equals letting heat diffuse for time t = σ²/2. Blobs are hot spots, and each one "cools" fastest at the scale matching its size.

Why this topic: it is the foundation of week 5 (feature detection) and of our maze agent's RootSIFT → VLAD front end. Existing videos (Computerphile, First Principles of CV) walk through SIFT's steps but mostly assert "DoG approximates LoG". We show why: the DoG is automatically scale-normalized, for free.

We work 1D first (curves), then 2D images, with only one 3D scene. That keeps render cost low within the 10/14 deadline. Numerics can be ported from our verified convolution tutor artifact (Gaussian, Laplacian, DoG stack, circles test image).

### Storyboard

Nine scenes, \~7.3 min total (timings in the Script). Scenes 1–3 follow our OneNote plan; 4–9 carry the scale-space argument. Only scenes 7–8 need image rendering or 3D.

| # | Scene | Key idea on screen | Manim tools |
| --- | --- | --- | --- |
| 1 | Hook | Same object at three distances; how can features be scale invariant? | ImageMobject |
| 2 | 1D pinhole | Similar triangles give s = fL/Z; only L/Z survives; image becomes brightness I(x) | ValueTracker, always\_redraw |
| 3 | Kernels & Gaussian | Convolution as sliding weighted sum; one fixed σ treats near and far bumps differently | ValueTracker, plot\_line\_graph |
| 4 | Blur is heat | Brightness as temperature; heat equation; Gaussian blur = diffusion for t = σ²/2 | Axes, color-mapped rod |
| 5 | Measuring size | ∇² as cooling rate; for smooth blobs the raw LoG response only decays; σ²∇² peaks at each blob's own scale | Log-σ axes, always\_redraw |
| 6 | DoG for free | ∂G/∂σ = σ∇²G, so G(kσ) − G(σ) ≈ (k−1)σ²∇²G | MathTex transforms |
| 7 | 2D circles | DoG stack over σ; each circle lights up at σ ≈ r/√2 | ImageMobject from numpy |
| 8 | Keypoints | Extremum in (x, y, σ) vs. its 26 neighbors | ThreeDScene (only one) |
| 9 | Coda | Gradient histogram → descriptor → maze agent; blank walls yield no keypoints | Reused artifact frames |

### Script

Narration by scene and beat, \~7:20 total at \~140 wpm. Times are relative to the scene start. Beat numbers match the Manim prompts.

#### Scene 1: Hook (0:45)

| Beat | Time | Narration | On screen |
| --- | --- | --- | --- |
| 1 | 0:00–0:15 | You've probably noticed this: photograph the same object from farther away, and it looks smaller in the picture. Obvious, maybe even trivial. | Three real photos of one object at increasing distance |
| 2 | 0:15–0:35 | But it poses a real problem for computer vision. A computer doesn't recognize things the way we do. It compares small, distinctive patches (corners, blobs, spots) between images. | One patch highlighted in each photo, shrinking |
| 3 | 0:35–0:45 | If those patches change size every time the camera moves, how can a computer tell they're the same? Put another way: how do we make features scale invariant? | Title card: "scale invariant?" |

#### Scene 2: 1D pinhole (0:55)

| Beat | Time | Narration | On screen |
| --- | --- | --- | --- |
| 1 | 0:00–0:10 | To see what's really going on, let's simplify. Squash the world into a flat plane, and the camera into a single point with a line behind it: a one-dimensional image. | Optical axis, pinhole, image line; rod at Z = 3 |
| 2 | 0:10–0:25 | Light from the top and bottom of an object passes through the pinhole and lands on the image line. Similar triangles tell us exactly where: an object of length L at distance Z casts an image of length f·L/Z. | Rays, inverted image, similar triangles, s = fL/Z, braces |
| 3 | 0:25–0:33 | Double the distance, and the image halves. | Z: 3 → 6; readout s: 1.00 → 0.50 |
| 4 | 0:33–0:45 | Notice what the camera can't tell apart: a small object nearby and a large one far away give the same image. Size and distance are tangled together, and we won't untangle them. Instead, we'll find features in a way that doesn't care. | L = 4 rod at Z = 6 shares the image; L/Z highlighted |
| 5 | 0:45–0:55 | Now look at the image from the computer's point of view. It's just brightness along a line: a bright bump whose width is set by f·L/Z. | Image line rotates into I(x); three boxes for Z = 2, 4, 8 |

#### Scene 3: Kernels and the Gaussian (1:05)

| Beat | Time | Narration | On screen |
| --- | --- | --- | --- |
| 1 | 0:00–0:20 | So how does a computer examine a signal like this? With a kernel: a small window of weights that slides along the image. At each position, it multiplies the pixels under it by its weights and adds them up. That sum becomes one point of a new signal. This sliding weighted sum is called convolution. | 5-tap kernel sliding over sampled bump; products, sum, output dots |
| 2 | 0:20–0:38 | One kernel matters more than any other: the Gaussian, a bell curve whose width is set by a single number, σ. Convolving with it replaces each pixel by a weighted average of its neighbors, with close neighbors counting most. The result is a smoothed, blurred signal. | Bars morph to Gaussian; formula; blurred curve traces in |
| 3 | 0:38–0:58 | But watch what one fixed σ does to our three images. The near object's wide bump barely changes. The far object's narrow bump gets flattened almost to nothing. Same object, same kernel, completely different results. A fixed-size kernel is biased toward one size. | Three panels, σ = 0.375; peaks 0.96, 0.71, 0.45 |
| 4 | 0:58–1:05 | So what if σ isn't fixed? What happens as we let it grow? | σ grows live; cut mid-growth |

#### Scene 4: Blur is heat (0:50)

| Beat | Time | Narration | On screen |
| --- | --- | --- | --- |
| 1 | 0:00–0:12 | As σ grows, something familiar happens. Picture the signal as a metal rod, and brightness as temperature. Blurring looks just like heat spreading out. | Brightness profile recolored as a heat-mapped rod |
| 2 | 0:12–0:28 | Heat flows from hot to cold. A point cools at a rate set by how much hotter it is than the average of its neighbors. That difference is the second derivative, and it gives the heat equation: the change in temperature over time equals its curvature in space. | 3-point stencil; ∂u/∂t = ∂²u/∂x² |
| 3 | 0:28–0:42 | Its solution is remarkable: start from any temperature profile, wait a time t, and you get the original profile convolved with a Gaussian of width σ = √(2t). Blurring is diffusion. | Heat simulation and Gaussian blur side by side, overlapping exactly |
| 4 | 0:42–0:50 | And notice: small hot spots fade first, while big ones last. How long a spot survives tells us how big it is. | Three bumps fading at different rates |

#### Scene 5: Measuring size (1:00)

| Beat | Time | Narration | On screen |
| --- | --- | --- | --- |
| 1 | 0:00–0:15 | Real cameras soften edges a little, so from here on each bump is a smooth Gaussian blob, still narrower the farther away it is. Let's measure how fast each spot cools: the second derivative, written ∇² and called the Laplacian. Blur by σ, then take ∇²: together that's a single kernel, the Laplacian of Gaussian, shaped like an upside-down Mexican hat. | LoG kernel; center-minus-surround shading |
| 2 | 0:15–0:30 | Now track the response at each bump's center as σ grows. It only falls. More blur flattens everything, so the smallest σ always wins. No peak, no size. | Raw response vs. σ: three decaying curves |
| 3 | 0:30–0:48 | The fix is to multiply by σ². Curvature has units of one over length squared; σ² cancels them, so we compare each bump relative to the scale we're looking at. Now every curve rises and falls, peaking at a σ proportional to the blob's width: √2 times it. And all three peaks reach the same height. | σ²·response vs. log σ; peaks at σ ≈ 0.27, 0.53, 1.06, all ≈ 0.385 |
| 4 | 0:48–1:00 | Move the camera closer, and the peak simply slides along the σ axis by the zoom factor. The detector reports a size that moves with the object, without ever knowing the distance. | One curve sliding on log σ axis as Z changes |

#### Scene 6: DoG for free (1:00)

| Beat | Time | Narration | On screen |
| --- | --- | --- | --- |
| 1 | 0:00–0:10 | There's a catch: computing second derivatives and rescaling at every σ is expensive. SIFT uses a shortcut, and it comes straight from the heat equation. | Stack of LoG computations crossed out |
| 2 | 0:10–0:25 | The Gaussian itself obeys the heat equation. Increase σ a little, and the change in the Gaussian is σ times its Laplacian. | ∂G/∂σ = σ∇²G; G growing, its change matching σG″ |
| 3 | 0:25–0:42 | Replace that derivative with a difference between two blur levels, σ and kσ. The Difference of Gaussians equals k minus one, times σ², times the Laplacian. The σ² we needed appears on its own. | Derivation transforms; DoG curve overlays (k−1)σ²∇²G |
| 4 | 0:42–1:00 | In heat terms: subtract two snapshots, and you get how much each point cooled in between. Since time grows like σ², equal steps in σ mean time gaps proportional to σ². That is the normalization, for free. So SIFT just blurs repeatedly and subtracts neighbors. | Two snapshots, shaded difference; Δt ∝ σ² bar |

#### Scene 7: 2D circles (0:45)

| Beat | Time | Narration | On screen |
| --- | --- | --- | --- |
| 1 | 0:00–0:15 | Back to two dimensions. Here's an image of circles with different radii, and its Difference of Gaussians at increasing σ. | Circles image; DoG layers fanning out |
| 2 | 0:15–0:30 | Each circle lights up brightest in its own layer. For a disk of radius r, that happens at σ ≈ r/√2. | Per-circle response vs. σ, peak marked |
| 3 | 0:30–0:45 | Photograph the same object from three distances, and its blob peaks three layers apart. Different σ, same feature. | Hook object at 3 distances, peaks in shifted layers |

#### Scene 8: Keypoints (0:30)

| Beat | Time | Narration | On screen |
| --- | --- | --- | --- |
| 1 | 0:00–0:18 | Stack the layers into a volume: position on two axes, scale on the third. A keypoint is a point brighter, or darker, than all 26 neighbors: 8 around it, 9 above, 9 below. | 3D voxel with its 26 neighbors highlighted |
| 2 | 0:18–0:30 | Its position says where the feature is; its scale says how big. SIFT then discards weak responses and points sitting on edges. | Keypoints drawn as circles of radius ∝ σ |

#### Scene 9: Coda (0:30)

| Beat | Time | Narration | On screen |
| --- | --- | --- | --- |
| 1 | 0:00–0:18 | Knowing each keypoint's size, we cut out a patch proportional to it and describe it by the directions of its gradients. From near or far, the same patch gets the same description. | Patch scaled by σ; orientation histogram |
| 2 | 0:18–0:30 | That's what lets a robot recognize a place it has seen before. And on a blank wall, with no blobs to find, it's blind. | Maze agent views; blank wall with no keypoints |

### Derivations

Everything is 1D unless stated. Numbers for our boxes use background b = 0.1, peak p = 1.0, widths w = 1.5, 0.75, 0.375 (Z = 2, 4, 8), and scenes 5–6 use Gaussian blobs of width s = w/2 = 0.75, 0.375, 0.19. All values checked numerically.

#### D1. Pinhole projection (scene 2)

Pinhole at the origin, image line at distance f behind it. A rod endpoint at height y and depth Z maps by similar triangles to:

```latex
y_{\text{img}} = -\frac{f\,y}{Z} \quad\Rightarrow\quad s = \frac{f\,L}{Z}
```

Only the ratio L/Z appears, so size and depth cannot be separated from one image.

#### D2. Convolution and the Gaussian (scene 3)

```latex
(I * g)(x) = \int g(u)\, I(x-u)\, du, \qquad g_\sigma(x) = \frac{1}{\sqrt{2\pi}\,\sigma}\, e^{-x^2/2\sigma^2}
```

Variances add under repeated blurring (semigroup property), which is why blur levels can be built incrementally:

```latex
g_{\sigma_1} * g_{\sigma_2} = g_{\sqrt{\sigma_1^2 + \sigma_2^2}}
```

#### D3. Laplacian as "different from my neighbors" (scenes 4–5)

Taylor-expand I(x ± h) and add; odd terms cancel:

```latex
I(x+h) - 2I(x) + I(x-h) = h^2\, I''(x) + O(h^4) = 2\left[\tfrac{I(x+h)+I(x-h)}{2} - I(x)\right]
```

So I″ is proportional to (neighbor average − self): negative at a bright peak, zero on flat or linear regions. In 2D, ∇²I = I\_xx + I\_yy.

#### D4. Heat equation and Gaussian blur (scene 4)

Fourier's law (heat flows down the gradient) plus energy conservation give the heat equation; choose time units so α = 1:

```latex
q = -\kappa\,\frac{\partial u}{\partial x}, \qquad \rho c\,\frac{\partial u}{\partial t} = -\frac{\partial q}{\partial x} \quad\Rightarrow\quad \frac{\partial u}{\partial t} = \frac{\partial^2 u}{\partial x^2}
```

By D3, a point cools at a rate proportional to how much hotter it is than its neighbors. The fundamental solution is a Gaussian with σ² = 2t:

```latex
G(x,t) = \frac{1}{\sqrt{4\pi t}}\, e^{-x^2/4t} = g_{\sigma}(x), \quad \sigma = \sqrt{2t}
```

Check: G\_t = G·(x²/4t² − 1/2t) and G\_x = −(x/2t)·G, so G\_xx = G·(x²/4t² − 1/2t) = G\_t. The solution from an initial profile I is u(x,t) = (G(·,t) ∗ I)(x). Blurring by σ is diffusing for time t = σ²/2.

#### D5. Laplacian of Gaussian (scene 5)

Derivatives commute with convolution, so blur-then-differentiate is one kernel:

```latex
\frac{d^2}{dx^2}\,(g_\sigma * I) = g_\sigma'' * I, \qquad g_\sigma''(x) = g_\sigma(x)\left(\frac{x^2}{\sigma^4} - \frac{1}{\sigma^2}\right)
```

Negative center, positive surround: a center-minus-surround template tuned to bumps of size \~σ.

#### D6. Raw response decays with σ (scene 5, beat 2)

Gaussian bump of width s and unit height. By D2 its blur is a wider, lower Gaussian; the curvature at the center is:

```latex
I(x) = e^{-x^2/2s^2} \;\Rightarrow\; (g_\sigma * I)''(0) = -\frac{s}{(s^2+\sigma^2)^{3/2}}
```

Its magnitude decreases monotonically in σ, so raw LoG always prefers the smallest σ. Caveat for sharp boxes: their flat center has zero curvature at σ = 0, so the raw response peaks at σ = a/√3, but with height ∝ 1/a², a 16× bias between our near and far objects. Either way the raw response is unusable for comparing scales; this is why scenes 5–6 switch to smooth blobs.

#### D7. Scale normalization (scene 5, beats 3–4)

Multiply by σ² and maximize; with u = σ²:

```latex
R(\sigma) = \frac{\sigma^2\, s}{(s^2+\sigma^2)^{3/2}}, \qquad \frac{dR}{du} = 0 \;\Rightarrow\; s^2 + u = \tfrac{3}{2}u \;\Rightarrow\; \sigma^* = \sqrt{2}\, s
```

For our boxes (half-width a = w/2; Φ, φ = standard normal CDF and pdf), the blurred box and its normalized center curvature are:

```latex
f(x) = b + (p-b)\left[\Phi\!\left(\tfrac{x+a}{\sigma}\right) - \Phi\!\left(\tfrac{x-a}{\sigma}\right)\right], \qquad \sigma^2 |f''(0)| = 2(p-b)\, z\,\varphi(z), \;\; z = \tfrac{a}{\sigma}
```

Since d(zφ)/dz = φ(z)(1 − z²), the peak is at z = 1: σ\* = w/2 = 0.75, 0.375, 0.19 for Z = 2, 4, 8, and the peak height 2(p − b)φ(1) ≈ 0.44 is the same for every width.

Why σ² exactly: rescale the image by λ, I\_λ(x) = I(x/λ). Then the blurred image at λσ is the original at σ, stretched; second derivatives shrink by λ², and σ² grows by λ²:

```latex
L_\lambda(x;\lambda\sigma) = L(x/\lambda;\sigma) \;\Rightarrow\; (\lambda\sigma)^2\, \partial_{xx} L_\lambda = \sigma^2\, \partial_{xx} L
```

So σ\* scales with the object (σ\* → λσ\*) and the peak height is unchanged. The constant depends on shape and dimension (2D Gaussian blob: σ\* = s; 2D disk: σ\* = r/√2, Lindeberg 1998); only the proportionality matters.

#### D8. Difference of Gaussians (scene 6)

Chain rule on D4 with t = σ²/2 (dt/dσ = σ):

```latex
\frac{\partial G}{\partial \sigma} = \frac{\partial G}{\partial t}\,\frac{dt}{d\sigma} = \sigma\,\nabla^2 G
```

Direct check: ∂G/∂σ = G·(x²/σ³ − 1/σ) = σ·G″ by D5. Finite difference between σ and kσ:

```latex
G(x, k\sigma) - G(x, \sigma) \approx (k\sigma - \sigma)\,\frac{\partial G}{\partial \sigma} = (k-1)\,\sigma^2\, \nabla^2 G
```

Cleaner view in log-scale: with τ = ln σ, ∂G/∂τ = σ·∂G/∂σ = σ²∇²G exactly. A DoG is a centered difference in τ, so it estimates σ²∇²G at the midpoint scale √k·σ with step ln k ≈ k − 1. For our boxes with SIFT's k = 2^(1/3) ≈ 1.26, labeling each DoG layer at √k·σ recovers σ\* = w/2 within 0.5%; for the Gaussian blobs of scenes 5–6 it recovers σ\* = √2·s within 0.2%.

Heat-snapshot reading: the time gap between the two blurs is Δt = (k² − 1)σ²/2, proportional to σ², and DoG ≈ ∇²G·Δt. Equal ratio steps in σ give time gaps that grow like σ², which is exactly the normalization.

#### D9. Keypoint detection (scenes 7–8)

Build D(x, y, σ) = (G(kσ) − G(σ)) ∗ I over octaves with k = 2^(1/s), s = 3 intervals per octave (Lowe 2004). A keypoint is a sample larger or smaller than its 26 neighbors (8 in-layer, 9 above, 9 below). Lowe then refines the location with a quadratic fit, drops low-contrast points (|D| < 0.03 for intensities in \[0, 1\]), and drops edge points via the Hessian ratio test tr(H)²/det(H) < (r + 1)²/r with r = 10.

### Core concepts to study beforehand

1. Convolution and the Gaussian kernel: separability, and the semigroup property (blurring by σ₁ then σ₂ = blurring by √(σ₁² + σ₂²)).
2. Image derivatives: gradient, Laplacian ∇², Laplacian of Gaussian (LoG) as a blob detector.
3. Heat equation basics: ∂u/∂t = ∇²u, with the Gaussian as its fundamental solution.
4. Scale-normalized derivatives: why multiply by σ², and how that creates a peak at the characteristic scale.
5. Difference of Gaussians: finite difference in σ, and the (k−1) factor.
6. SIFT detection: octaves, 26-neighbor extremum test, edge and low-contrast rejection.
7. Optional, for the coda: SIFT descriptor (orientation histograms) and RootSIFT.

### Bibliography

Course texts first; chapter titles are from memory, so verify section numbers against our editions. HZ2003 barely covers detectors, so it is omitted.

| Ref | Where to read | Covers concepts |
| --- | --- | --- |
| \[FP2011\] Forsyth & Ponce, 2nd ed. | Ch. 4 Linear Filters; Ch. 5 Local Image Features (gradients, corners, scale and orientation, SIFT/HOG) | 1, 2, 6, 7 |
| \[Sz2022\] Szeliski, 2nd ed. | Ch. 3 Image Processing (§3.2 linear filtering, §3.5 pyramids); Ch. 7 Feature Detection and Matching (§7.1) | 1, 2, 5, 6, 7 |
| \[Co2011\] Corke, 1st ed. | Ch. 12 Images and Image Processing (spatial operations); Ch. 13 Image Feature Extraction (§13.3 point features, scale-space detectors) | 1, 2, 6 |
| \[Go2016\] Goodfellow et al. | Ch. 9 Convolutional Networks (convolution vs. correlation) | 1, optional |

Primary papers:

- Lowe, "Distinctive image features from scale-invariant keypoints," IJCV 60(2), 2004. §3 derives DoG ≈ (k−1)σ²∇²G (scene 4).
- Lindeberg, "Feature detection with automatic scale selection," IJCV 30(2), 1998. γ-normalization and the r/√2 disk result (scenes 3, 5).
- Koenderink, "The structure of images," Biol. Cybern. 50, 1984. Scale space as diffusion (scene 2).
- Arandjelović & Zisserman, "Three things everyone should know to improve object retrieval," CVPR 2012. RootSIFT (scene 7).
- Manim Community documentation, docs.manim.community (ValueTracker, ImageMobject, ThreeDScene).

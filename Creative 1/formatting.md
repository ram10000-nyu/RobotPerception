# Formatting guide: match the "1D pinhole projection" scene

Everything here is taken from `scene2_pinhole_1d.py` (Manim Community v0.21.0, `from manim import *`). Copy these values so our scenes look like one video.

## Environment setup (macOS)

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

## Setup details

- Renderer: Manim Community **v0.21.0**. Do not use ManimGL / 3b1b-manim names (`ShowCreation`, `TextMobject`, `TexMobject`, `GraphScene`). Use `Create`, `Tex`, `MathTex`, `Axes.plot`.
- LaTeX is required for `MathTex` and `DecimalNumber`. On macOS: `brew install --cask basictex`, then
  `sudo tlmgr install standalone preview doublestroke relsize ms setspace rsfs wasysym physics dvisvgm xcolor fundamental babel-english cm-super`
- Frame: default Manim frame, 14.2 x 8 units. Keep everything inside it with a margin (x within about +/-6.7, y within about +/-3.6).
- Background: default black. Do not change it.
- Render: `manim -pql file.py SceneName` (preview, 480p15) and `manim -qh file.py SceneName` (final, 1080p60).

## Colors

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

## Line weights

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

## Typography

The scene uses Manim's default fonts: no custom font is set, so don't set one. Sizes:

- Equations: `MathTex(...)` at default size (for example `s = \frac{f\,L}{Z}`), placed at the top of the frame.
- Brace labels (`L`, `s`, `Z`, `f`): `MathTex(...).scale(0.8)`, `buff=0.1` from the brace.
- Axis labels (`x`, `I(x)`): `MathTex(...).scale(0.7)`. Larger axes scale it up with the axes.
- Panel titles ("Z = 2"): `MathTex(...).scale(0.8)`, 0.3 below the axis.
- "same image" callout: `Text(..., font_size=26, color=WHITE)` plus a thin white `Arrow`.
- Small tag ("image"): `Text(..., font_size=22, color=GRAY_B)`.
- Readout panel: `Text(..., font_size=24)` for the labels and `DecimalNumber(..., num_decimal_places=2, font_size=30)` for the values. Pin it to the top right with `.to_corner(UR, buff=0.4)`. Rows are `Z =`, `L =`, `s = fL/Z =`. Values are in the object's color and update live.
- Keep on-screen text minimal; the narration carries the explanation.

## Layout and motion

- Diagram: horizontal axis through the middle. Pinhole (camera center) at x = -3, image plane 1.5 to its right (between camera and object, so the image is upright), object further right. Rays run from the object ends to the pinhole.
- Equations sit at the top center-left (`[-1.3, 3.0, 0]`); the readout sits top right; labels and braces hang below the diagram (y = -1.3 for the f brace, y = -2.3 for the Z brace).
- Braces: `BraceBetweenPoints(p1, p2, direction=..., buff=0.12)` with a label next to each, small.
- Every shown number is computed from the geometry, never hardcoded. Drive values with `ValueTracker` and `always_redraw` so that mobjects and readouts stay consistent while animating.
- Create things with `Create` or `FadeIn` (about 1 to 1.5 s each); move values with `rate_func=smooth` (about 3 s for a big change); leave highlights up for about 1 s.
- Highlight a shared or key element with `Indicate(...)` plus `Flash(...)` in `WHITE`.
- Put every tunable (geometry, colors, run times) as named constants at the top of the file, and put `# BEAT n` comments quoting the narration line before each beat.

## Reusable constants (copy-paste)

```python
ROD_COLOR   = YELLOW        # main object
ROD2_COLOR  = TEAL          # comparison object
RAY_COLOR, RAY_OPACITY, RAY_WIDTH = WHITE, 0.55, 2
AXIS_COLOR  = GRAY
ROD_WIDTH   = 8
GHOST_OPACITY = 0.3
```

## Reusable helper (readable text with or without LaTeX)

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

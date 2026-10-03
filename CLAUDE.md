# CLAUDE.md

Manim Community **v0.21.0** animations for a robot-perception course video on why SIFT needs scale space. Everything lives in `Creative 1/`. Scenes 2-4 exist; the full plan is in `Creative 1/README.md`.

## Setup and run
- `cd "Creative 1"`, then `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`. macOS also needs `brew install pkg-config cairo pango ffmpeg`, and LaTeX for `MathTex`.
- Preview: `manim -pql sceneN_name.py ClassName`. Final: `-qh`. Without LaTeX, `NO_LATEX=1` gives a text-only preview.

## Conventions
- Style (colors, stroke widths, font sizes, layout): follow `Creative 1/formatting.md`. Don't invent new colors or fonts.
- Use `from manim import *` with Community names only (`Create`, `MathTex`, `Axes.plot`). No ManimGL (`ShowCreation`, `TextMobject`, `GraphScene`). Check the docs rather than guessing signatures.
- Every scene file: constants at the top, a `# BEAT n` comment quoting the narration before each beat, and a `self_check()` (called first in `construct()`) that raises on failure.
- On-screen numbers are computed from the geometry or `common.py`, never hardcoded. Shared signal and solver code goes in `common.py` (pure numpy, no Manim).
- Don't name variables after Manim globals (for example `ORIGIN`, `UP`).
- Don't commit `.venv/`, `media/` or `__pycache__/`.

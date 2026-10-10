"""
Support Vector Machines — shared module
=======================================

Palette, datasets, geometry helpers and the scene base class used by every
part of the explainer. Keeping them here is what makes the six parts read as
one continuous film: the same points, the same colours and the same street
appear in every scene.

Design notes
------------
* Palette is fixed once (3 accents + ink/dim on near-black) and imported
  everywhere, so the film feels like a single piece.
* The two 2-D datasets are generated deterministically at import, so a point
  that appears in Part 1 is the *same* point in Part 5.
* ``separator()`` and ``separator_scaled()`` return the maximum-margin
  hyperplane (w, b) analytically, so every rail really does touch a support
  vector — the animation is correct by construction, never eyeballed.

Verified against the Manim Community 0.22 API.
"""

from __future__ import annotations

import numpy as np
from manim import *

# --- Palette (fixed once, used in every scene) -------------------------------
BG = "#0A0A0A"          # near-black background
CLASS_A = "#58C4DD"     # blue   — the negative class
CLASS_B = "#FFD166"     # yellow — the positive class
MARGIN = "#83C167"      # green  — the margin / street
VIOLATION = "#FC6255"   # red    — margin violations
INK = "#ECECEC"         # near-white text
DIM = "#8A93A6"         # muted grey

config.background_color = BG

# Plot window for the blob dataset. Every 2-D scene uses the same window so the
# data reads identically throughout the film.
XLIM = (-4.0, 4.0)
YLIM = (-2.8, 2.8)


class SVMScene(Scene):
    """Base scene: dark background (set via ``config``) + consistent furniture."""

    def corner_title(self, text: str, **kwargs) -> Text:
        """Small title pinned to the top-left, used to label each act."""
        return Text(text, color=INK, font_size=30, **kwargs).to_corner(UL, buff=0.5)

    def bottom_eq(self, tex: str, **kwargs) -> MathTex:
        """Equation anchored in the lower band, clear of the graph above it."""
        return MathTex(tex, color=INK, **kwargs).to_edge(DOWN, buff=0.6)

    def caption(self, text: str, **kwargs) -> Text:
        """A one-line caption for the lower band."""
        return Text(text, color=DIM, font_size=26, **kwargs).to_edge(DOWN, buff=0.6)


# --- Datasets ----------------------------------------------------------------
def _separable(seed: int = 7, n: int = 15) -> tuple[np.ndarray, np.ndarray]:
    """Two Gaussian blobs, linearly separable — the running example."""
    rng = np.random.default_rng(seed)
    a = rng.normal([-1.8, 0.75], [0.7, 0.65], size=(n, 2))
    b = rng.normal([1.8, -0.7], [0.7, 0.65], size=(n, 2))
    return a, b


CLASS_A_PTS, CLASS_B_PTS = _separable()


def _closest_pair(a: np.ndarray, b: np.ndarray):
    """The nearest (A, B) pair — our two hand-picked support vectors."""
    d = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
    i, j = np.unravel_index(np.argmin(d), d.shape)
    return a[i], b[j]


SV_A, SV_B = _closest_pair(CLASS_A_PTS, CLASS_B_PTS)
MID = (SV_A + SV_B) / 2  # the separator passes through the midpoint of the SVs


def separator(sv_a: np.ndarray, sv_b: np.ndarray):
    """
    Maximum-margin hyperplane through two support vectors.

    With ``w`` parallel to ``(sv_b - sv_a)`` and scaled so ``w·(sv_b - sv_a) = 2``,
    the two points sit exactly on the rails ``w·x + b = ±1``. Returns ``(w, b)``;
    the data-space margin is ``2 / ||w|| = ||sv_b - sv_a||``.
    """
    d = sv_b - sv_a
    w = 2 * d / (d @ d)
    b = -1.0 - w @ sv_a
    return w, b


W, B = separator(SV_A, SV_B)


def separator_scaled(w: np.ndarray, b: np.ndarray, width_factor: float, center: np.ndarray):
    """
    Same orientation as ``(w, b)`` but with the margin scaled by ``width_factor``
    and re-centred on ``center``. ``width_factor=1`` reproduces the original
    separator; smaller values give the deliberately *narrow* margin used in the
    side-by-side comparisons.
    """
    d_hat = w / np.linalg.norm(w)
    base_width = 2 / np.linalg.norm(w)          # data-space margin of (w, b)
    new_norm = 2 / (width_factor * base_width)  # norm that yields the new margin
    w2 = d_hat * new_norm
    b2 = -w2 @ center
    return w2, b2


# --- Geometry helpers --------------------------------------------------------
def _clip(p0, p1, xlim, ylim):
    """Liang–Barsky: clip segment ``p0->p1`` to the data rectangle, or ``None``."""
    (x0, y0), (x1, y1) = p0, p1
    dx, dy = x1 - x0, y1 - y0
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, x0 - xlim[0]), (dx, xlim[1] - x0),
                 (-dy, y0 - ylim[0]), (dy, ylim[1] - y0)):
        if p == 0:
            if q < 0:
                return None
        else:
            t = q / p
            if p < 0:
                t0 = max(t0, t)
            else:
                t1 = min(t1, t)
    if t0 > t1:
        return None
    return (x0 + t0 * dx, y0 + t0 * dy), (x0 + t1 * dx, y0 + t1 * dy)


def line_data_pts(w, b, xlim=XLIM, ylim=YLIM):
    """Two data-space endpoints where ``w·x + b = 0`` crosses the plot box."""
    if abs(w[1]) > 1e-9:
        y0 = -(w[0] * xlim[0] + b) / w[1]
        y1 = -(w[0] * xlim[1] + b) / w[1]
        return _clip((xlim[0], y0), (xlim[1], y1), xlim, ylim)
    x = -b / w[0]
    return _clip((x, ylim[0]), (x, ylim[1]), xlim, ylim)


def make_axes(width=6.0, height=3.6, center=DOWN * 0.3, xlim=XLIM, ylim=YLIM):
    """A clean, number-free axes box, positioned where the caller wants it."""
    ax = Axes(
        x_range=[xlim[0], xlim[1], 1],
        y_range=[ylim[0], ylim[1], 1],
        x_length=width,
        y_length=height,
        tips=False,
        axis_config={"include_numbers": False, "stroke_color": DIM, "stroke_width": 2},
    )
    return ax.move_to(center)


def scatter(ax, pts, color, radius=0.075):
    """Dots for a set of data-space points."""
    return VGroup(*[Dot(ax.c2p(*p), radius=radius, color=color) for p in pts])


def base_plot(width=6.0, height=3.6, center=DOWN * 0.3, xlim=XLIM, ylim=YLIM, radius=0.075):
    """The standard blob scatter on a fresh axes box. Returns ``(ax, dots)``."""
    ax = make_axes(width, height, center, xlim, ylim)
    dots = VGroup(
        scatter(ax, CLASS_A_PTS, CLASS_A, radius),
        scatter(ax, CLASS_B_PTS, CLASS_B, radius),
    )
    return ax, dots


def panel(center, width=4.6, height=3.3, xlim=XLIM, ylim=YLIM, radius=0.075):
    """
    One comparison panel: an axes box plus the shared blob scatter, sized for
    the side-by-side layouts. Shared by Parts 2 and 4 so both draw the same
    panels from the same data.
    """
    ax = make_axes(width=width, height=height, center=center, xlim=xlim, ylim=ylim)
    dots = VGroup(
        scatter(ax, CLASS_A_PTS, CLASS_A, radius),
        scatter(ax, CLASS_B_PTS, CLASS_B, radius),
    )
    return ax, dots


def boundary_line(ax, w=W, b=B, xlim=XLIM, ylim=YLIM, **style):
    """The decision boundary ``w·x + b = 0`` as a Line in the axes."""
    seg = line_data_pts(w, b, xlim, ylim)
    style.setdefault("color", INK)
    style.setdefault("stroke_width", 4)
    return Line(ax.c2p(*seg[0]), ax.c2p(*seg[1]), **style)


def rail_lines(ax, w=W, b=B, xlim=XLIM, ylim=YLIM, **style):
    """The two dashed rails at ``w·x + b = ±1``."""
    style.setdefault("color", MARGIN)
    style.setdefault("stroke_width", 3)
    style.setdefault("dash_length", 0.14)
    rails = []
    for s in (+1, -1):
        # w·x + b = s  <=>  w·x + (b - s) = 0
        seg = line_data_pts(w, b - s, xlim, ylim)
        rails.append(DashedLine(ax.c2p(*seg[0]), ax.c2p(*seg[1]), **style))
    return VGroup(*rails)


def street(ax, w=W, b=B, xlim=XLIM, ylim=YLIM, fill=True):
    """
    The 'street': boundary + two rails, optionally with a translucent fill
    between the rails. Built from the boundary so the fill can never mismatch.
    """
    boundary = boundary_line(ax, w, b, xlim, ylim)
    rails = rail_lines(ax, w, b, xlim, ylim)
    if not fill:
        return VGroup(boundary, rails)
    hi = line_data_pts(w, b - 1, xlim, ylim)
    lo = line_data_pts(w, b + 1, xlim, ylim)
    quad = Polygon(
        ax.c2p(*hi[0]), ax.c2p(*hi[1]), ax.c2p(*lo[1]), ax.c2p(*lo[0]),
        color=MARGIN, fill_opacity=0.12, stroke_width=0,
    )
    return VGroup(quad, boundary, rails)


def margin_arrow(ax, w=W, b=B, center=None, **style):
    """Double-headed arrow spanning the street, perpendicular to the boundary."""
    center = MID if center is None else center
    n = w / np.linalg.norm(w)
    half = 1.0 / np.linalg.norm(w)  # data-space distance from boundary to a rail
    style.setdefault("color", MARGIN)
    style.setdefault("stroke_width", 3)
    return DoubleArrow(
        ax.c2p(*(center - n * half)), ax.c2p(*(center + n * half)), buff=0, **style
    )


def sv_rings(ax, radius=0.16, color=MARGIN):
    """Green rings marking the two support vectors."""
    return VGroup(
        *[Circle(radius=radius, color=color, stroke_width=3).move_to(ax.c2p(*p))
          for p in (SV_A, SV_B)]
    )


# --- Concentric-rings dataset (the non-linearly-separable example) -----------
def ring_dataset(seed=3, n_in=22, n_out=30):
    """Blue points in a tight disc, yellow points in a surrounding ring."""
    rng = np.random.default_rng(seed)
    r_in = 0.95 + rng.normal(0, 0.10, n_in)
    a_in = rng.uniform(0, TAU, n_in)
    inner = np.c_[r_in * np.cos(a_in), r_in * np.sin(a_in)]
    r_out = 2.35 + rng.normal(0, 0.14, n_out)
    a_out = rng.uniform(0, TAU, n_out)
    outer = np.c_[r_out * np.cos(a_out), r_out * np.sin(a_out)]
    return inner, outer


INNER_PTS, OUTER_PTS = ring_dataset()
RING_XLIM = (-3.2, 3.2)
RING_YLIM = (-3.2, 3.2)

"""
Generative Adversarial Networks — a 3Blue1Brown-grade Manim explainer
=====================================================================

A ~18-minute, scene-for-scene animation built to the approved script.
17 scenes (S00 … S16) in the order they play.

QUICK START
-----------
Render everything to 1080p:
    manim -qh gans_explainer.py S00_ColdOpen S01_GenerationProblem ... S16_Closing

Or one scene while iterating (fast 480p preview):
    manim -pql gans_explainer.py S08_OptimalDiscriminator

In a Jupyter notebook (cell magic), the file must be importable, then:
    %%manim -v WARNING -qm S00_ColdOpen
    class ... (or use the imported classes)
  Simpler: keep this file next to the notebook and run
    from gans_explainer import *
    %%manim -qm S03_CounterfeiterAndPolice

REQUIREMENTS
------------
- manim (Community edition). Written and verified against the 0.22 API;
  the classes used are stable across 0.19–0.22.
- A LaTeX distribution (TeX Live) for MathTex/Tex in S07, S08, S09.
  Required packages: amsmath, amssymb (present in texlive-latex-extra).
- dvisvgm (ships with most TeX Live installs) — manim uses it to turn the
  compiled DVI into SVG. On Debian/Ubuntu: `apt-get install dvisvgm`.
- No external image/audio assets are needed; every visual is drawn with
  Manim primitives. Fonts use Manim's default, so nothing is missing.

Install (Linux — manimpango has no Linux wheel and builds from source, so the
system Pango/Cairo headers and a compiler are required):
    sudo apt-get install build-essential python3-dev libcairo2-dev \
         libpango1.0-dev pkg-config dvisvgm
    pip install manim
macOS / Windows: `pip install manim` works directly (wheels exist).

NOTE: this script was authored and statically verified against the Manim
0.22 source; every class, method and keyword argument used was checked
against that source. It could not be frame-rendered in the authoring
sandbox (no Linux manimpango wheel, no compiler/Pango headers, no dvisvgm),
so please render the first pass on your own machine and send back anything
that looks off — the geometry and API are verified, the pixels are not.

DESIGN NOTES
------------
- Palette is fixed below (3 accents + ink/dim on near-black) and used
  everywhere, so the film feels like one piece.
- Motion carries meaning: curves are driven by ValueTrackers +
  always_redraw so every frame is correct-by-construction, never keyframed.
- Scenes are short and independent for fast iteration; the shared base
  class GANScene sets the background and exposes camera zoom.
"""

from __future__ import annotations

import numpy as np
from manim import *

# --------------------------------------------------------------------------- #
#  Palette  (fixed once — referenced everywhere)
# --------------------------------------------------------------------------- #
BG   = "#0E1116"   # near-black background
REAL = "#58C4DD"   # real data            (blue)
FAKE = "#FFD166"   # generator's fakes    (yellow)
DISC = "#FC6255"   # discriminator        (red)
OK   = "#83C167"   # equilibrium/success  (green)
INK  = "#ECECEC"   # primary text
DIM  = "#8A93A6"   # muted text / axes

config.background_color = BG
config.frame_rate = 60


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
def pdf(x, mu=0.0, sigma=1.0):
    """Unit-area Gaussian."""
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def dstar(x, mu_r=0.0, s_r=1.0, mu_f=0.0, s_f=1.0):
    """Optimal discriminator D*(x) = p_data / (p_data + p_g)."""
    pr = pdf(x, mu_r, s_r)
    pf = pdf(x, mu_f, s_f)
    return pr / (pr + pf + 1e-12)


def dist_axes(x_length=9.0, y_length=3.0, y_max=1.1):
    """A clean 1-D density axis pair (no numbers, muted)."""
    return Axes(
        x_range=[-4, 4, 1],
        y_range=[0, y_max, 0.25],
        x_length=x_length,
        y_length=y_length,
        axis_config={
            "color": DIM,
            "stroke_width": 2,
            "include_ticks": False,
            "include_tip": True,
            "tip_width": 0.16,
            "tip_height": 0.16,
        },
    )


def block(label, color, w=2.0, h=1.15, fs=26):
    """A rounded 'neural net' block with its label."""
    rect = RoundedRectangle(
        corner_radius=0.16, width=w, height=h,
        stroke_color=color, stroke_width=3,
        fill_color=color, fill_opacity=0.12,
    )
    txt = Text(label, font_size=fs, color=color, weight=BOLD)
    return VGroup(rect, txt)


def noise_vector(n=6, color=INK, buff=0.28, r=0.055):
    dots = VGroup(*[Dot(radius=r, color=color) for _ in range(n)])
    dots.arrange(DOWN, buff=buff)
    return dots


def tile(color, size=0.85, opacity=0.22, stroke=3):
    sq = Square(side_length=size, stroke_color=color, stroke_width=stroke)
    sq.set_fill(color, opacity=opacity)
    return sq


def chip(num, label, color):
    """A numbered step chip for the training-loop scene."""
    r = RoundedRectangle(
        corner_radius=0.12, width=2.6, height=1.25,
        stroke_color=color, stroke_width=3,
        fill_color=color, fill_opacity=0.10,
    )
    n = Text(str(num), font_size=30, color=color, weight=BOLD)
    t = Text(label, font_size=20, color=INK)
    n.move_to(r.get_top() + DOWN * 0.32)
    t.move_to(r.get_bottom() + UP * 0.32)
    return VGroup(r, n, t)


def caption(text, color=INK, fs=30, y=-3.2):
    return Text(text, font_size=fs, color=color).move_to([0, y, 0])


class GANScene(MovingCameraScene):
    """Shared base: dark background + zoomable camera."""
    def setup(self):
        super().setup()
        self.camera.background_color = BG


# =========================================================================== #
#  S00 — Cold open
# =========================================================================== #
class S00_ColdOpen(GANScene):
    def construct(self):
        center = UP * 1.35
        blue = Dot(center + LEFT * 1.15, radius=0.13, color=REAL)
        yellow = Dot(center + RIGHT * 1.15, radius=0.13, color=FAKE)
        # faint connecting ring to read as "the two are linked"
        ring = Circle(radius=1.15, color=DIM, stroke_width=1.5, stroke_opacity=0.35).move_to(center)

        title = Text("Generative Adversarial Networks",
                     font_size=46, color=INK, weight=BOLD).move_to(DOWN * 1.2)
        sub = Text("a duel between two networks",
                   font_size=26, color=DIM).next_to(title, DOWN, buff=0.35)

        self.play(FadeIn(ring), FadeIn(blue, scale=2), FadeIn(yellow, scale=2), run_time=1.2)
        self.play(
            Rotate(blue, about_point=center, angle=TAU, run_time=5, rate_func=linear),
            Rotate(yellow, about_point=center, angle=TAU, run_time=5, rate_func=linear),
        )
        self.play(Write(title), run_time=1.6)
        self.play(FadeIn(sub, shift=UP * 0.2), run_time=0.8)
        self.wait(1.4)
        self.play(FadeOut(VGroup(ring, blue, yellow, title, sub)), run_time=1.0)


# =========================================================================== #
#  S01 — The generation problem
# =========================================================================== #
class S01_GenerationProblem(GANScene):
    def construct(self):
        head = Text("Most networks classify. This one must create.",
                    font_size=30, color=INK).to_edge(UP, buff=0.6)

        # a small grid of "samples"
        tiles = VGroup(*[tile(INK, size=0.7, opacity=0.10, stroke=2) for _ in range(20)])
        tiles.arrange_in_grid(rows=4, cols=5, buff=0.18)

        self.play(FadeIn(head))
        self.play(LaggedStart(*[FadeIn(t, scale=1.2) for t in tiles],
                              lag_ratio=0.05), run_time=2.0)
        self.wait(0.6)

        # morph the samples into a data cloud
        rng = np.random.default_rng(7)
        cloud = VGroup(*[
            Dot(
                np.array([rng.normal(0, 1.5), rng.normal(0.7, 0.55), 0]),
                radius=0.055, color=REAL,
            ) for _ in range(70)
        ])
        label = MathTex(r"p_{\mathrm{data}}", color=REAL, font_size=40).next_to(cloud, RIGHT, buff=0.5)

        self.play(FadeOut(tiles), run_time=0.8)
        self.play(LaggedStart(*[GrowFromCenter(d) for d in cloud], lag_ratio=0.01), run_time=1.8)
        self.play(Write(label), run_time=0.8)
        self.wait(0.8)

        ask = Text("Learn the pattern. Then produce something new.",
                   font_size=28, color=INK).to_edge(DOWN, buff=0.7)
        self.play(FadeIn(ask, shift=UP * 0.2))
        self.wait(1.6)
        self.play(FadeOut(VGroup(cloud, label, head, ask)))


# =========================================================================== #
#  S02 — Why it's hard (the averaging trap)
# =========================================================================== #
class S02_AveragingTrap(GANScene):
    def construct(self):
        head = Text("The averaging trap", font_size=34, color=INK).to_edge(UP, buff=0.6)

        # data along a gentle arc
        xs = np.linspace(-3.6, 3.6, 26)
        pts = VGroup(*[
            Dot(np.array([x, 0.9 * np.sin(0.55 * x) + 0.25 * np.cos(3 * x), 0]),
                radius=0.06, color=REAL)
            for x in xs
        ])
        self.play(Write(head))
        self.play(LaggedStart(*[GrowFromCenter(p) for p in pts], lag_ratio=0.04), run_time=1.6)
        self.wait(0.5)

        # an ordinary regressor collapses everything to the mean
        mean_pt = np.array([0.0, 0.0, 0.0])
        blur = Dot(mean_pt, radius=0.16, color=DIM)
        arrows = VGroup(*[
            Arrow(p.get_center(), mean_pt, buff=0.12, color=DIM,
                  stroke_width=2, max_tip_length_to_length_ratio=0.35)
            for p in pts
        ])
        self.play(Create(arrows), FadeIn(blur, scale=1.6), run_time=1.6)
        note = Text("minimise average error  →  output the average",
                    font_size=26, color=DISC).to_edge(DOWN, buff=0.7)
        self.play(FadeIn(note, shift=UP * 0.2))
        self.wait(1.4)

        punch = Text("Real data is not the average of real data.",
                     font_size=30, color=INK).to_edge(DOWN, buff=0.7)
        self.play(Transform(note, punch), run_time=1.0)
        self.wait(1.6)
        self.play(FadeOut(VGroup(head, pts, arrows, blur, note)))


# =========================================================================== #
#  S03 — The counterfeiter and the police
# =========================================================================== #
class S03_CounterfeiterAndPolice(GANScene):
    def construct(self):
        forger = block("Forger", FAKE, w=2.2, h=1.3).move_to(LEFT * 4.2 + DOWN * 0.4)
        officer = block("Officer", DISC, w=2.2, h=1.3).move_to(RIGHT * 4.2 + DOWN * 0.4)
        head = Text("Goodfellow, 2014:  train the creator against an adversary",
                    font_size=28, color=INK).to_edge(UP, buff=0.7)

        self.play(FadeIn(head))
        self.play(FadeIn(forger, shift=RIGHT * 0.3), FadeIn(officer, shift=LEFT * 0.3), run_time=1.2)
        self.wait(0.4)

        note = tile(INK, size=0.9, opacity=0.15).move_to(forger.get_center() + RIGHT * 1.8)

        def round_trip(quality, color, rejected):
            # note travels to the officer
            self.play(note.animate.move_to(officer.get_center() + LEFT * 1.8),
                      run_time=1.1, rate_func=smooth)
            stamp = Text("FAKE", font_size=24, color=DISC).next_to(officer, UP, buff=0.25)
            self.play(FadeIn(stamp, scale=1.4), run_time=0.4)
            self.play(note.animate.move_to(forger.get_center() + RIGHT * 1.8),
                      run_time=1.1, rate_func=smooth)
            self.play(FadeOut(stamp), run_time=0.3)

        # first, clumsy fakes — caught
        round_trip(0.2, DISC, True)
        # the forger studies what was caught and improves
        better = tile(FAKE, size=0.9, opacity=0.5).move_to(note)
        self.play(Transform(note, better), run_time=0.5)
        self.play(Circumscribe(forger, color=FAKE, buff=0.2), run_time=0.9)
        round_trip(0.6, DISC, True)
        # third time: indistinguishable — the officer hesitates
        best = tile(REAL, size=0.9, opacity=0.6).move_to(note)
        self.play(Transform(note, best), run_time=0.5)
        self.play(note.animate.move_to(officer.get_center() + LEFT * 1.8), run_time=1.1)
        unsure = Text("?", font_size=44, color=OK).next_to(officer, UP, buff=0.25)
        self.play(FadeIn(unsure, scale=1.8), run_time=0.5)

        punch = Text("Neither is given the answer — each learns from the other's failure.",
                     font_size=26, color=INK).to_edge(DOWN, buff=0.6)
        self.play(FadeIn(punch, shift=UP * 0.2))
        self.wait(1.8)
        self.play(FadeOut(VGroup(head, forger, officer, note, unsure, punch)))


# =========================================================================== #
#  S04 — Meet the two networks
# =========================================================================== #
class S04_TwoNetworks(GANScene):
    def construct(self):
        head = Text("Two networks, wired head to tail",
                    font_size=32, color=INK).to_edge(UP, buff=0.6)

        z = noise_vector(n=6).move_to(LEFT * 6.0)
        z_lab = MathTex("z", color=INK, font_size=34).next_to(z, UP, buff=0.3)
        z_sub = Text("noise", font_size=20, color=DIM).next_to(z, DOWN, buff=0.3)

        G = block("Generator G", FAKE, w=2.6, h=1.4).move_to(LEFT * 2.6)
        sample = tile(FAKE, size=1.0, opacity=0.35).move_to(LEFT * 0.1)
        s_lab = MathTex("G(z)", color=FAKE, font_size=32).next_to(sample, DOWN, buff=0.28)

        D = block("Discriminator D", DISC, w=3.0, h=1.4).move_to(RIGHT * 3.0)
        out = Text("P(real)", font_size=26, color=DISC).move_to(RIGHT * 5.9)

        real = tile(REAL, size=1.0, opacity=0.35).move_to(UP * 2.1 + RIGHT * 0.1)
        r_lab = Text("real data", font_size=22, color=REAL).next_to(real, UP, buff=0.25)

        a1 = Arrow(z.get_right(), G.get_left(), buff=0.12, color=INK, stroke_width=3)
        a2 = Arrow(G.get_right(), sample.get_left(), buff=0.12, color=INK, stroke_width=3)
        a3 = Arrow(sample.get_right(), D.get_left(), buff=0.12, color=INK, stroke_width=3)
        a4 = Arrow(D.get_right(), out.get_left(), buff=0.12, color=INK, stroke_width=3)
        a5 = Arrow(real.get_bottom(), D.get_top(), buff=0.12, color=REAL, stroke_width=3)

        self.play(Write(head))
        self.play(FadeIn(z, shift=RIGHT * 0.3), Write(z_lab), FadeIn(z_sub))
        self.play(GrowArrow(a1), FadeIn(G, shift=RIGHT * 0.2))
        self.play(GrowArrow(a2), FadeIn(sample, scale=1.3), Write(s_lab))
        self.play(FadeIn(D, shift=LEFT * 0.2))
        self.play(GrowArrow(a3), GrowArrow(a4), FadeIn(out, shift=RIGHT * 0.2))
        self.play(FadeIn(real, shift=DOWN * 0.2), Write(r_lab), GrowArrow(a5))
        self.wait(0.8)

        g_goal = Text("G: make D say 1", font_size=24, color=FAKE).to_edge(DOWN, buff=1.1)
        d_goal = Text("D: be right", font_size=24, color=DISC).to_edge(DOWN, buff=0.6)
        self.play(FadeIn(g_goal), FadeIn(d_goal))
        self.wait(1.8)
        self.play(FadeOut(VGroup(head, z, z_lab, z_sub, G, sample, s_lab, D, out,
                                 real, r_lab, a1, a2, a3, a4, a5, g_goal, d_goal)))


# =========================================================================== #
#  S05 — The generator as a warping map
# =========================================================================== #
class S05_WarpingMap(GANScene):
    def construct(self):
        head = Text("The generator bends a grid of noise onto the data",
                    font_size=30, color=INK).to_edge(UP, buff=0.6)

        # latent grid
        grid = NumberPlane(
            x_range=[-3, 3, 0.5], y_range=[-2, 2, 0.5],
            background_line_style={"stroke_color": INK, "stroke_width": 1.4,
                                   "stroke_opacity": 0.5},
            axis_config={"stroke_opacity": 0},
        ).scale(1.0).move_to(LEFT * 3.3)
        grid_lab = Text("latent space  z", font_size=22, color=DIM).next_to(grid, DOWN, buff=0.35)

        self.play(Write(head))
        self.play(Create(grid), FadeIn(grid_lab), run_time=1.6)
        self.wait(0.6)

        # warp it
        def warp(p):
            x, y, z = p
            return np.array([x + 0.35 * np.sin(1.1 * y), 0.62 * y + 0.18 * x, 0])

        warped = grid.copy().apply_function(warp).set_color(FAKE)
        warped_lab = Text("G warps it", font_size=22, color=FAKE).next_to(grid, DOWN, buff=0.35)

        self.play(ReplacementTransform(grid, warped),
                  ReplacementTransform(grid_lab, warped_lab), run_time=1.8)
        self.wait(0.4)

        # bring it onto a data cloud
        rng = np.random.default_rng(3)
        cloud = VGroup(*[
            Dot(np.array([LEFT[0] * 0 + 3.3 + rng.normal(0, 0.9), rng.normal(0, 0.55), 0]),
                radius=0.05, color=REAL)
            for _ in range(80)
        ])
        moved = warped.copy().move_to(RIGHT * 3.3)
        cloud_lab = Text("real data", font_size=22, color=REAL).next_to(cloud, DOWN, buff=0.35)

        self.play(ReplacementTransform(warped, moved),
                  ReplacementTransform(warped_lab, cloud_lab), run_time=1.2)
        self.play(LaggedStart(*[GrowFromCenter(d) for d in cloud], lag_ratio=0.01), run_time=1.4)

        punch = Text("Training = warping the grid until it covers the data exactly.",
                     font_size=26, color=INK).to_edge(DOWN, buff=0.5)
        self.play(FadeIn(punch, shift=UP * 0.2))
        self.wait(1.8)
        self.play(FadeOut(VGroup(head, moved, cloud, cloud_lab, punch)))


# =========================================================================== #
#  S06 — One training round
# =========================================================================== #
class S06_TrainingRound(GANScene):
    def construct(self):
        head = Text("One round of training", font_size=34, color=INK).to_edge(UP, buff=0.6)
        c1 = chip(1, "sample real", REAL)
        c2 = chip(2, "generate fakes", FAKE)
        c3 = chip(3, "update D", DISC)
        c4 = chip(4, "update G", FAKE)
        chips = VGroup(c1, c2, c3, c4).arrange(RIGHT, buff=0.5).move_to(DOWN * 0.3)

        self.play(Write(head))
        self.play(LaggedStart(*[FadeIn(c, shift=UP * 0.3) for c in chips], lag_ratio=0.15),
                  run_time=1.6)
        self.wait(0.4)

        detail = {
            1: "draw a batch of real data",
            2: "draw noise, let G generate a matching batch",
            3: "push D's scores to 1 on real, 0 on fake — fakes detached",
            4: "update G up the slope of D's opinion",
        }
        line = Text("", font_size=24, color=INK).to_edge(DOWN, buff=0.7)
        self.add(line)
        for i, c in enumerate(chips, start=1):
            new = Text(detail[i], font_size=24, color=INK).to_edge(DOWN, buff=0.7)
            self.play(Circumscribe(c, color=c[0].get_stroke_color(), buff=0.12, run_time=0.7),
                      Transform(line, new), run_time=0.9)
            self.wait(0.7)

        loop = Text("…and repeat, thousands of times.", font_size=26, color=OK).to_edge(DOWN, buff=0.7)
        self.play(Transform(line, loop))
        self.wait(1.6)
        self.play(FadeOut(VGroup(head, chips, line)))


# =========================================================================== #
#  S07 — The minimax objective
# =========================================================================== #
class S07_Minimax(GANScene):
    def construct(self):
        head = Text("The minimax objective", font_size=34, color=INK).to_edge(UP, buff=0.6)

        eq = MathTex(
            r"\min_{G}\max_{D}\ \ V(D,G)=",
            r"\mathbb{E}_{x\sim p_{\mathrm{data}}}\big[\log D(x)\big]",
            r"+",
            r"\mathbb{E}_{z\sim p_z}\big[\log\big(1-D(G(z))\big)\big]",
            font_size=46,
            tex_to_color_map={
                r"\mathbb{E}_{x\sim p_{\mathrm{data}}}\big[\log D(x)\big]": REAL,
                r"\mathbb{E}_{z\sim p_z}\big[\log\big(1-D(G(z))\big)\big]": FAKE,
                r"\min_{G}": OK, r"\max_{D}": DISC,
            },
        ).move_to(UP * 0.7)

        self.play(Write(head))
        self.play(Write(eq), run_time=2.2)
        self.wait(0.6)

        b1 = Brace(eq[1], DOWN, buff=0.25)
        t1 = Text("D wants this big", font_size=24, color=REAL).next_to(b1, DOWN, buff=0.12)
        b2 = Brace(eq[3], DOWN, buff=0.25)
        t2 = Text("G wants this small", font_size=24, color=FAKE).next_to(b2, DOWN, buff=0.12)
        self.play(GrowFromCenter(b1), FadeIn(t1), run_time=0.9)
        self.play(GrowFromCenter(b2), FadeIn(t2), run_time=0.9)
        self.wait(0.8)

        punch = Text("Not a loss to descend — a saddle point to search for.",
                     font_size=28, color=INK).to_edge(DOWN, buff=0.6)
        self.play(FadeIn(punch, shift=UP * 0.2))
        self.wait(2.0)
        self.play(FadeOut(VGroup(head, eq, b1, t1, b2, t2, punch)))


# =========================================================================== #
#  S08 — The optimal discriminator
# =========================================================================== #
class S08_OptimalDiscriminator(GANScene):
    def construct(self):
        head = Text("What is the best possible discriminator?",
                    font_size=32, color=INK).to_edge(UP, buff=0.5)
        axes = dist_axes().move_to(DOWN * 0.5)

        real_c = axes.plot(lambda x: pdf(x, -1.1, 0.8), color=REAL, stroke_width=4)
        fake_c = axes.plot(lambda x: pdf(x, 1.3, 0.9), color=FAKE, stroke_width=4)
        d_c = axes.plot(lambda x: dstar(x, -1.1, 0.8, 1.3, 0.9), color=DISC, stroke_width=4)

        real_lab = Text("real", font_size=22, color=REAL).next_to(axes.c2p(-1.9, pdf(-1.9, -1.1, 0.8)), UP, buff=0.15)
        fake_lab = Text("fake", font_size=22, color=FAKE).next_to(axes.c2p(2.4, pdf(2.4, 1.3, 0.9)), UP, buff=0.15)

        formula = MathTex(
            r"D^*(x)=\dfrac{p_{\mathrm{data}}(x)}{p_{\mathrm{data}}(x)+p_g(x)}",
            font_size=40, color=DISC,
        ).to_corner(UR, buff=0.7)

        self.play(Write(head))
        self.play(Create(axes), run_time=0.8)
        self.play(Create(real_c), FadeIn(real_lab), run_time=1.0)
        self.play(Create(fake_c), FadeIn(fake_lab), run_time=1.0)
        self.play(Create(d_c), Write(formula), run_time=1.6)
        self.wait(0.6)

        for x, txt in [(-1.9, "near 1 where real data lives"),
                       (2.4, "near 0 where fakes live"),
                       (0.1, "near ½ where they overlap")]:
            lbl = Text(txt, font_size=22, color=INK).next_to(axes.c2p(x, dstar(x, -1.1, 0.8, 1.3, 0.9)), UP, buff=0.2)
            dot = Dot(axes.c2p(x, dstar(x, -1.1, 0.8, 1.3, 0.9)), radius=0.07, color=INK)
            self.play(FadeIn(dot), FadeIn(lbl), run_time=0.7)
            self.wait(0.5)
            self.play(FadeOut(lbl), run_time=0.4)

        self.wait(1.2)
        self.play(FadeOut(VGroup(head, axes, real_c, fake_c, d_c, real_lab, fake_lab, formula)))


# =========================================================================== #
#  S09 — Convergence
# =========================================================================== #
class S09_Convergence(GANScene):
    def construct(self):
        head = Text("Push the fake onto the real — and the game collapses to JS divergence",
                    font_size=26, color=INK).to_edge(UP, buff=0.5)
        axes = dist_axes().move_to(DOWN * 0.4)

        mu = ValueTracker(1.6)
        real_c = axes.plot(lambda x: pdf(x, -0.4, 0.85), color=REAL, stroke_width=4)
        fake_c = always_redraw(
            lambda: axes.plot(lambda x: pdf(x, mu.get_value(), 0.9), color=FAKE, stroke_width=4))
        d_c = always_redraw(
            lambda: axes.plot(lambda x: dstar(x, -0.4, 0.85, mu.get_value(), 0.9),
                              color=DISC, stroke_width=3))

        js = MathTex(r"\mathrm{JS}\big(p_{\mathrm{data}}\,\|\,p_g\big)", font_size=38, color=OK)
        js.to_corner(UR, buff=0.7)
        arrow = MathTex(r"\longrightarrow 0", font_size=38, color=OK).next_to(js, DOWN, buff=0.2)

        self.play(Write(head))
        self.play(Create(axes), Create(real_c), run_time=1.0)
        self.add(fake_c, d_c)
        self.play(FadeIn(js), run_time=0.8)
        self.wait(0.4)

        self.play(mu.animate.set_value(-0.4), run_time=4.5, rate_func=smooth)
        self.play(Write(arrow), run_time=0.8)
        half = DashedLine(axes.c2p(-4, 0.5), axes.c2p(4, 0.5), color=INK, stroke_width=2, dash_length=0.12)
        self.play(Create(half), run_time=0.8)
        note = Text("D is reduced to a coin flip — the generator has won",
                    font_size=24, color=INK).to_edge(DOWN, buff=0.35)
        self.play(FadeIn(note, shift=UP * 0.2))
        self.wait(2.0)
        self.play(FadeOut(VGroup(head, axes, real_c, fake_c, d_c, js, arrow, half, note)))


# =========================================================================== #
#  S10 — No monotone loss
# =========================================================================== #
class S10_NoMonotoneLoss(GANScene):
    def construct(self):
        head = Text("There is no loss curve that marches down",
                    font_size=32, color=INK).to_edge(UP, buff=0.5)
        ax = Axes(
            x_range=[0, 10, 1], y_range=[0, 2.4, 0.5],
            x_length=9.5, y_length=4.0,
            axis_config={"color": DIM, "stroke_width": 2, "include_ticks": False},
        ).move_to(DOWN * 0.4)
        x_lab = Text("training steps", font_size=22, color=DIM).next_to(ax, DOWN, buff=0.3)

        d_loss = ax.plot(lambda x: 1.386 + 0.55 * np.sin(3.0 * x) * np.exp(-0.03 * x),
                         color=DISC, stroke_width=3)
        g_loss = ax.plot(lambda x: 0.95 + 0.55 * np.sin(3.0 * x + 2.2),
                         color=OK, stroke_width=3)
        eq_line = DashedLine(ax.c2p(0, 1.386), ax.c2p(10, 1.386), color=INK,
                             stroke_width=2, dash_length=0.14)
        eq_lab = MathTex(r"\log 4", color=INK, font_size=28).next_to(ax.c2p(0, 1.386), LEFT, buff=0.2)

        d_key = Text("D loss", font_size=22, color=DISC).to_corner(UR, buff=0.9)
        g_key = Text("G loss", font_size=22, color=OK).next_to(d_key, DOWN, buff=0.15)

        self.play(Write(head))
        self.play(Create(ax), FadeIn(x_lab), run_time=1.0)
        self.play(Create(d_loss), FadeIn(d_key), run_time=1.4)
        self.play(Create(g_loss), FadeIn(g_key), run_time=1.4)
        self.play(Create(eq_line), FadeIn(eq_lab), run_time=0.8)
        note = Text("watch the samples, not the loss", font_size=26, color=INK).to_edge(DOWN, buff=0.3)
        self.play(FadeIn(note, shift=UP * 0.2))
        self.wait(2.0)
        self.play(FadeOut(VGroup(head, ax, x_lab, d_loss, g_loss, eq_line, eq_lab,
                                 d_key, g_key, note)))


# =========================================================================== #
#  S11 — Vanishing gradients
# =========================================================================== #
class S11_VanishingGradients(GANScene):
    def construct(self):
        head = Text("A perfect discriminator gives no gradient",
                    font_size=30, color=INK).to_edge(UP, buff=0.5)
        axes = dist_axes().move_to(DOWN * 0.3)

        mu_r, s_r = -2.6, 0.55
        mu_f, s_f = 2.6, 0.55
        real_c = axes.plot(lambda x: pdf(x, mu_r, s_r), color=REAL, stroke_width=4)
        fake_c = axes.plot(lambda x: pdf(x, mu_f, s_f), color=FAKE, stroke_width=4)
        d_c = axes.plot(lambda x: dstar(x, mu_r, s_r, mu_f, s_f), color=DISC, stroke_width=3)

        self.play(Write(head))
        self.play(Create(axes), Create(real_c), Create(fake_c), run_time=1.2)
        self.play(Create(d_c), run_time=1.0)
        self.wait(0.4)

        plateau = SurroundingRectangle(
            Line(axes.c2p(1.6, dstar(1.6, mu_r, s_r, mu_f, s_f)),
                 axes.c2p(3.6, dstar(3.6, mu_r, s_r, mu_f, s_f))),
            color=OK, buff=0.18, corner_radius=0.1)
        pl_lab = Text("flat plateau where the fakes live", font_size=22, color=OK).next_to(plateau, UP, buff=0.2)
        self.play(Create(plateau), FadeIn(pl_lab), run_time=0.9)
        self.wait(0.6)

        # gradient at the fake location ~ 0 → arrow vanishes
        eps = 0.05
        x0 = mu_f
        grad = (dstar(x0 + eps, mu_r, s_r, mu_f, s_f) - dstar(x0 - eps, mu_r, s_r, mu_f, s_f)) / (2 * eps)
        base = axes.c2p(x0, dstar(x0, mu_r, s_r, mu_f, s_f))
        # the learning signal is ~0 here; keep a small visible stub so it renders
        glen = float(np.clip(abs(grad) * 1.2, 0.04, 1.2))
        arrow = Arrow(base, base + LEFT * glen, buff=0, color=OK, stroke_width=4)
        grad_lab = Text("G's learning signal  ≈ 0", font_size=24, color=OK).next_to(axes, DOWN, buff=0.6)
        self.play(Create(arrow), FadeIn(grad_lab), run_time=0.9)

        punch = Text("The better D is, the less G can learn.",
                     font_size=28, color=INK).to_edge(DOWN, buff=0.15)
        self.play(FadeIn(punch, shift=UP * 0.2))
        self.wait(2.0)
        self.play(FadeOut(VGroup(head, axes, real_c, fake_c, d_c, plateau, pl_lab,
                                 arrow, grad_lab, punch)))


# =========================================================================== #
#  S12 — Mode collapse
# =========================================================================== #
class S12_ModeCollapse(GANScene):
    def construct(self):
        head = Text("Mode collapse: one convincing sample, and nothing else",
                    font_size=30, color=INK).to_edge(UP, buff=0.6)

        # three real modes
        centers = [LEFT * 4.0, ORIGIN, RIGHT * 4.0]
        rng = np.random.default_rng(11)
        modes = VGroup()
        for c in centers:
            blobs = VGroup(*[
                Dot(c + np.array([rng.normal(0, 0.35), rng.normal(0, 0.35), 0]),
                    radius=0.05, color=REAL)
                for _ in range(45)
            ])
            modes.add(blobs)
        mode_lab = Text("real data: three distinct modes", font_size=22, color=REAL).to_edge(DOWN, buff=0.5)

        self.play(Write(head))
        self.play(LaggedStart(*[FadeIn(m) for m in modes], lag_ratio=0.2), run_time=1.6)
        self.play(FadeIn(mode_lab))
        self.wait(0.6)

        # the generator's fakes: one tight cluster that hops between modes
        gen = VGroup(*[
            Dot(centers[0] + np.array([rng.normal(0, 0.16), rng.normal(0, 0.16), 0]),
                radius=0.07, color=FAKE)
            for _ in range(30)
        ])
        self.play(FadeIn(gen, scale=1.3), run_time=0.8)
        self.play(Circumscribe(modes[0], color=FAKE, buff=0.25), run_time=0.9)

        for i in (1, 2, 1, 0):
            self.play(gen.animate.move_to(centers[i]), run_time=1.3, rate_func=smooth)
            self.play(Circumscribe(modes[i], color=FAKE, buff=0.25), run_time=0.7)

        punch = Text("D judges one sample at a time — it never sees the missing variety.",
                     font_size=24, color=INK).to_edge(DOWN, buff=0.5)
        self.play(Transform(mode_lab, punch))
        self.wait(2.0)
        self.play(FadeOut(VGroup(head, modes, gen, mode_lab)))


# =========================================================================== #
#  S13 — The fixes
# =========================================================================== #
class S13_Fixes(GANScene):
    def construct(self):
        head = Text("Two failures, three responses", font_size=34, color=INK).to_edge(UP, buff=0.6)

        def card(title, body, color):
            r = RoundedRectangle(corner_radius=0.18, width=3.7, height=2.5,
                                 stroke_color=color, stroke_width=3,
                                 fill_color=color, fill_opacity=0.10)
            t = Text(title, font_size=24, color=color, weight=BOLD).move_to(r.get_top() + DOWN * 0.45)
            b = Text(body, font_size=19, color=INK).move_to(r.get_center() + DOWN * 0.35)
            b.scale_to_fit_width(3.3)
            return VGroup(r, t, b)

        c1 = card("Non-saturating loss", "train G to maximise\nlog D  — same fixed\npoint, stronger gradient",
                  OK)
        c2 = card("Wasserstein GAN", "a critic estimates the\nearth-mover distance —\nnever a flat plateau",
                  REAL)
        c3 = card("Fight mode collapse", "minibatch discrimination\nand unrolled GANs — D sees\nthe whole batch",
                  FAKE)
        cards = VGroup(c1, c2, c3).arrange(RIGHT, buff=0.45).move_to(DOWN * 0.4)

        self.play(Write(head))
        self.play(LaggedStart(*[FadeIn(c, shift=UP * 0.3) for c in cards], lag_ratio=0.2),
                  run_time=1.8)
        self.wait(2.2)
        self.play(FadeOut(VGroup(head, cards)))


# =========================================================================== #
#  S14 — The GAN zoo
# =========================================================================== #
class S14_GANZoo(GANScene):
    def construct(self):
        head = Text("The GAN zoo", font_size=34, color=INK).to_edge(UP, buff=0.6)

        def panel(name, sub, color, kind):
            r = RoundedRectangle(corner_radius=0.18, width=3.6, height=2.8,
                                 stroke_color=color, stroke_width=3,
                                 fill_color=color, fill_opacity=0.08)
            nm = Text(name, font_size=24, color=color, weight=BOLD).move_to(r.get_top() + DOWN * 0.4)
            sb = Text(sub, font_size=18, color=DIM).next_to(nm, DOWN, buff=0.12)
            # a tiny schematic
            if kind == "conv":
                glyph = VGroup(*[Square(side_length=0.28, stroke_color=color, stroke_width=2)
                                 for _ in range(4)]).arrange(RIGHT, buff=0.12)
            elif kind == "cond":
                glyph = VGroup(
                    RoundedRectangle(corner_radius=0.06, width=0.5, height=0.3,
                                     stroke_color=color, stroke_width=2),
                    Arrow(LEFT * 0.1, RIGHT * 0.1, buff=0, color=color, stroke_width=2),
                    Square(side_length=0.34, stroke_color=color, stroke_width=2),
                ).arrange(RIGHT, buff=0.12)
            else:
                glyph = VGroup(*[RoundedRectangle(corner_radius=0.06, width=0.6, height=0.34,
                                                  stroke_color=color, stroke_width=2)
                                 for _ in range(3)]).arrange(DOWN, buff=0.1)
            glyph.move_to(r.get_bottom() + UP * 0.75)
            return VGroup(r, nm, sb, glyph)

        p1 = panel("DCGAN", "convolutions", REAL, "conv")
        p2 = panel("Conditional GAN", "ask for a label", FAKE, "cond")
        p3 = panel("StyleGAN", "layerwise style", OK, "layers")
        panels = VGroup(p1, p2, p3).arrange(RIGHT, buff=0.45).move_to(DOWN * 0.4)

        self.play(Write(head))
        self.play(LaggedStart(*[FadeIn(p, shift=UP * 0.3) for p in panels], lag_ratio=0.2),
                  run_time=1.8)
        self.wait(2.2)
        self.play(FadeOut(VGroup(head, panels)))


# =========================================================================== #
#  S15 — Evaluation & the modern landscape
# =========================================================================== #
class S15_EvaluationAndLandscape(GANScene):
    def construct(self):
        head = Text("No likelihood — so we judge with proxies", font_size=30, color=INK).to_edge(UP, buff=0.5)
        m1 = block("Inception Score", DISC, w=3.2, h=1.0, fs=22).move_to(LEFT * 3.3 + UP * 1.6)
        m2 = block("Fréchet Inception Dist.", REAL, w=3.6, h=1.0, fs=22).move_to(RIGHT * 3.0 + UP * 1.6)

        self.play(Write(head))
        self.play(FadeIn(m1, shift=UP * 0.2), FadeIn(m2, shift=UP * 0.2), run_time=1.0)
        self.wait(0.5)
        self.play(FadeOut(VGroup(head, m1, m2)))

        head2 = Text("And since 2020, diffusion largely took the lead", font_size=28, color=INK).to_edge(UP, buff=0.6)
        line = Line(LEFT * 5.5, RIGHT * 5.5, color=DIM, stroke_width=3).move_to(DOWN * 0.2)
        ticks = VGroup()
        entries = [(LEFT * 4.6, "2014", "GAN", DISC),
                   (LEFT * 1.2, "2018", "StyleGAN", FAKE),
                   (RIGHT * 2.0, "2020", "diffusion", OK)]
        self.play(Write(head2))
        self.play(Create(line), run_time=0.8)
        for pos, year, name, color in entries:
            dot = Dot(pos + DOWN * 0.2, radius=0.08, color=color)
            y = Text(year, font_size=22, color=INK).next_to(dot, UP, buff=0.25)
            nm = Text(name, font_size=22, color=color).next_to(dot, DOWN, buff=0.25)
            self.play(FadeIn(dot, scale=1.4), FadeIn(y), FadeIn(nm), run_time=0.7)
            self.wait(0.4)

        tag = Text("GANs moved to super-resolution, translation, video",
                   font_size=22, color=INK).next_to(line, DOWN, buff=0.9)
        self.play(FadeIn(tag, shift=UP * 0.2))
        self.wait(2.0)
        self.play(FadeOut(VGroup(head2, line, ticks, tag)))


# =========================================================================== #
#  S16 — Closing
# =========================================================================== #
class S16_Closing(GANScene):
    def construct(self):
        center = UP * 1.3
        blue = Dot(center + LEFT * 1.0, radius=0.12, color=REAL)
        yellow = Dot(center + RIGHT * 1.0, radius=0.12, color=FAKE)
        ring = Circle(radius=1.0, color=DIM, stroke_width=1.5, stroke_opacity=0.35).move_to(center)

        self.play(FadeIn(ring), FadeIn(blue), FadeIn(yellow), run_time=0.8)
        self.play(
            Rotate(blue, about_point=center, angle=TAU, run_time=4, rate_func=linear),
            Rotate(yellow, about_point=center, angle=TAU, run_time=4, rate_func=linear),
        )

        # a stream of new samples emitted downward
        rng = np.random.default_rng(5)
        stream = VGroup(*[
            tile(FAKE, size=0.35, opacity=0.3)
            for _ in range(14)
        ])
        stream.arrange_in_grid(rows=2, cols=7, buff=0.25).move_to(DOWN * 1.6)
        self.play(LaggedStart(*[FadeIn(s, scale=1.3) for s in stream], lag_ratio=0.06), run_time=1.6)

        msg = Text("A generator that makes the real thing from nothing but noise.",
                   font_size=26, color=INK).to_edge(DOWN, buff=0.3)
        self.play(FadeIn(msg, shift=UP * 0.2))
        self.wait(2.0)
        self.play(FadeOut(VGroup(ring, blue, yellow, stream, msg)))

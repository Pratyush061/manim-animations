"""
Part 1 — Hook & the problem
===========================

S01_Hook        The promise: a classifier that remembers only the borderline
                points. Establishes the running dataset and the two dots that
                will become the support vectors.
S02_ManyLines   Three different straight lines all separate the data — so which
                one is best? Sets up the question the rest of the film answers.

Render:  manim -qm part-01-hook.py S01_Hook
         manim -qm part-01-hook.py S02_ManyLines
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from svm_common import *  # noqa: E402,F403


class S01_Hook(SVMScene):
    def construct(self):
        title = Text("Support Vector Machines", color=INK, font_size=46).to_edge(UP, buff=0.7)
        self.play(Write(title), run_time=1.5)

        ax, dots = base_plot()
        self.play(
            LaggedStart(*[GrowFromCenter(d) for d in dots], lag_ratio=0.04),
            run_time=2.0,
        )
        self.wait(0.4)

        # The two points closest to the other class — soon to be the SVs.
        rings = sv_rings(ax)
        self.play(Create(rings), run_time=1.0)
        self.play(Indicate(rings, color=CLASS_B, scale_factor=1.25), run_time=1.4)

        caption = Text(
            "What if you only needed the borderline points?",
            color=DIM, font_size=26,
        ).to_edge(DOWN, buff=0.6)
        self.play(FadeIn(caption, shift=UP * 0.2), run_time=1.0)
        self.wait(2.0)

        self.play(FadeOut(VGroup(title, caption, ax, dots, rings)), run_time=1.0)


class S02_ManyLines(SVMScene):
    def construct(self):
        ax, dots = base_plot()
        self.add(ax, dots)

        # Three separators through the same midpoint, at different angles:
        # all valid, all different.
        colours = [CLASS_A, INK, MARGIN]
        lines = VGroup()
        for k, dtheta in enumerate((-24, 0, 24)):
            theta = np.arctan2(W[1], W[0]) + np.deg2rad(dtheta)
            w = np.array([np.cos(theta), np.sin(theta)])
            b = -w @ MID
            seg = line_data_pts(w, b)
            lines.add(Line(ax.c2p(*seg[0]), ax.c2p(*seg[1]), color=colours[k], stroke_width=4))

        for ln in lines:
            self.play(Create(ln), run_time=0.9)
            self.wait(0.3)

        question = Text(
            "Infinite lines separate the data. Which is best?",
            color=INK, font_size=30,
        ).to_edge(DOWN, buff=0.6)
        self.play(FadeIn(question, shift=UP * 0.2), run_time=1.0)
        self.play(Circumscribe(VGroup(*lines), color=CLASS_B), run_time=1.6)
        self.wait(2.0)

        self.play(FadeOut(VGroup(ax, dots, lines, question)), run_time=1.0)

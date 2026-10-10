"""
Part 6 — Recap
==============

S15_Recap  The very first scatter returns, now with the street drawn through it,
           and the three takeaways tick on one at a time — a callback to the
           hook so the film closes the loop.

Render:  manim -qm part-06-recap.py S15_Recap
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from svm_common import *  # noqa: E402,F403


class S15_Recap(SVMScene):
    def construct(self):
        ax, dots = base_plot(center=DOWN * 0.4)
        st = street(ax, fill=True)
        rings = sv_rings(ax)

        self.play(FadeIn(ax), FadeIn(dots), run_time=1.2)
        self.play(Create(st), run_time=1.2)
        self.play(Create(rings), run_time=0.8)

        title = Text("Support Vector Machines", color=INK, font_size=42).to_edge(UP, buff=0.7)
        self.play(Write(title), run_time=1.2)

        takeaways = VGroup(
            Text("1.  The widest street wins", color=MARGIN, font_size=26),
            Text("2.  Only support vectors matter", color=CLASS_B, font_size=26),
            Text("3.  Kernels bend the street", color=CLASS_A, font_size=26),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.28).to_edge(DOWN, buff=0.6)
        for line in takeaways:
            self.play(FadeIn(line, shift=RIGHT * 0.2), run_time=0.7)
            self.wait(0.4)

        self.wait(3.0)
        self.play(FadeOut(VGroup(ax, dots, st, rings, title, takeaways)), run_time=1.0)

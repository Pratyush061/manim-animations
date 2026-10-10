"""
Part 2 — The margin and the support vectors
===========================================

S03_NarrowVsWide   Side by side: a narrow-margin separator vs the wide-margin
                   one. Same data, same orientation, different street width.
S04_TheStreet      The margin drawn and named — boundary first, then the two
                   rails slide out until they touch the nearest points.
S05_SupportVectors Side by side: all points vs only the support vectors, with
                   the boundary provably unchanged.

Render:  manim -qm part-02-margin.py S03_NarrowVsWide
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from svm_common import *  # noqa: E402,F403


# Indices of the two points that are the support vectors, within their class.
IDX_A = int(np.argmin(((CLASS_A_PTS - SV_A) ** 2).sum(1)))
IDX_B = int(np.argmin(((CLASS_B_PTS - SV_B) ** 2).sum(1)))


class S03_NarrowVsWide(SVMScene):
    def construct(self):
        la, ld = panel(LEFT * 3.4)
        ra, rd = panel(RIGHT * 3.4)
        self.play(FadeIn(la), FadeIn(ra), FadeIn(ld), FadeIn(rd), run_time=1.5)

        # Same orientation, same centre — only the margin width differs.
        w_narrow, b_narrow = separator_scaled(W, B, 0.42, MID)
        street_l = street(la, w_narrow, b_narrow, fill=False)
        street_r = street(ra, W, B, fill=False)

        self.play(Create(street_l), Create(street_r), run_time=1.4)

        arrow_l = margin_arrow(la, w_narrow, b_narrow)
        arrow_r = margin_arrow(ra, W, B)
        self.play(GrowArrow(arrow_l), GrowArrow(arrow_r), run_time=1.0)

        lab_l = Text("small margin", color=VIOLATION, font_size=26).next_to(la, DOWN, buff=0.35)
        lab_r = Text("big margin", color=MARGIN, font_size=26).next_to(ra, DOWN, buff=0.35)
        self.play(Write(lab_l), Write(lab_r), run_time=1.2)
        self.wait(1.0)

        # Dim the weak separator so the eye settles on the good one.
        self.play(street_l.animate.set_opacity(0.25), run_time=1.0)
        self.wait(2.0)

        self.play(FadeOut(VGroup(la, ra, ld, rd, street_l, street_r,
                                 arrow_l, arrow_r, lab_l, lab_r)), run_time=1.0)


class S04_TheStreet(SVMScene):
    def construct(self):
        ax, dots = base_plot()
        self.play(FadeIn(ax), FadeIn(dots), run_time=1.5)

        boundary = boundary_line(ax)
        self.play(Create(boundary), run_time=1.2)

        # Rails start on the boundary and slide outward until they reach ±1.
        rails = VGroup(boundary.copy(), boundary.copy())
        self.add(rails)
        self.play(Transform(rails, rail_lines(ax)), run_time=2.0)

        arrow = margin_arrow(ax)
        self.play(GrowArrow(arrow), run_time=1.0)
        label = Text("margin", color=MARGIN, font_size=28).next_to(arrow, RIGHT, buff=0.2)
        self.play(Write(label), run_time=0.8)

        caption = Text(
            "Widest street = best separator",
            color=INK, font_size=28,
        ).to_edge(DOWN, buff=0.6)
        self.play(FadeIn(caption, shift=UP * 0.2), run_time=1.0)
        self.wait(2.5)

        self.play(FadeOut(VGroup(ax, dots, boundary, rails, arrow, label, caption)), run_time=1.0)


class S05_SupportVectors(SVMScene):
    def construct(self):
        la, ld = panel(LEFT * 3.4)
        ra, rd = panel(RIGHT * 3.4)
        self.play(FadeIn(la), FadeIn(ra), FadeIn(ld), FadeIn(rd), run_time=1.5)

        street_l = street(la, fill=False)
        street_r = street(ra, fill=False)
        rings_l = sv_rings(la)
        rings_r = sv_rings(ra)
        self.play(Create(street_l), Create(street_r), run_time=1.4)
        self.play(Create(rings_l), Create(rings_r), run_time=1.0)

        cap_l = Text("30 points", color=DIM, font_size=26).next_to(la, DOWN, buff=0.35)
        self.play(Write(cap_l), run_time=0.8)
        self.wait(0.6)

        # Right panel: everything except the two support vectors fades away.
        keep = VGroup(rd[0][IDX_A], rd[1][IDX_B])
        drop = VGroup(
            *[d for i, d in enumerate(rd[0]) if i != IDX_A],
            *[d for i, d in enumerate(rd[1]) if i != IDX_B],
        )
        self.play(drop.animate.set_opacity(0.0), run_time=1.6)
        self.remove(drop)

        cap_r = Text("2 support vectors — same boundary",
                     color=MARGIN, font_size=24).next_to(ra, DOWN, buff=0.35)
        self.play(Write(cap_r), run_time=1.0)

        equiv = MathTex(r"\equiv", color=INK, font_size=60).move_to(UP * 0.3)
        self.play(FadeIn(equiv, scale=1.5), run_time=0.8)
        self.play(Indicate(equiv, color=CLASS_B, scale_factor=1.3), run_time=1.2)
        self.wait(2.5)

        self.play(FadeOut(VGroup(la, ra, ld, keep, street_l, street_r,
                                 rings_l, rings_r, cap_l, cap_r, equiv)), run_time=1.0)

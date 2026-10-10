"""
Part 4 — Why the margin matters, and the soft margin
====================================================

S09_WhyWideMargin  Side by side: a narrow-margin boundary that hugs class A vs
                   the wide-margin one. The same new sample drifts by the same
                   amount — and only the narrow model flips it.
S10_SoftMargin     Real data is messy: points inside the street, the slack
                   variables that forgive them, and the C knob that trades
                   margin width against violations.

Render:  manim -qm part-04-soft-margin.py S09_WhyWideMargin
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from svm_common import *  # noqa: E402,F403

# Unit normal and a perpendicular, reused across the scenes.
N_HAT = W / np.linalg.norm(W)
T_HAT = np.array([-N_HAT[1], N_HAT[0]])


class S09_WhyWideMargin(SVMScene):
    def construct(self):
        la, ld = _panel(LEFT * 3.4)
        ra, rd = _panel(RIGHT * 3.4)
        self.play(FadeIn(la), FadeIn(ra), FadeIn(ld), FadeIn(rd), run_time=1.4)

        # Narrow model: same orientation, but the boundary is pushed up against
        # class A — exactly the failure mode a small margin represents.
        b_narrow = -W @ (SV_A + N_HAT * 0.15)
        st_l = street(la, W, b_narrow, fill=False)
        st_r = street(ra, W, B, fill=False)
        self.play(Create(st_l), Create(st_r), run_time=1.3)

        head_l = Text("narrow margin", color=VIOLATION, font_size=24).next_to(la, UP, buff=0.3)
        head_r = Text("wide margin", color=MARGIN, font_size=24).next_to(ra, UP, buff=0.3)
        self.play(Write(head_l), Write(head_r), run_time=1.0)

        # One new sample, at the same coordinates in both panels.
        t = SV_A + N_HAT * 0.12
        pt_l = Dot(la.c2p(*t), radius=0.12, color=INK)
        pt_r = Dot(ra.c2p(*t), radius=0.12, color=INK)
        self.play(GrowFromCenter(pt_l), GrowFromCenter(pt_r), run_time=0.9)
        self.wait(0.5)

        # Choose a drift that clears the narrow boundary but not the wide one,
        # computed from the geometry so it is right for any dataset.
        d_narrow = (W @ t + b_narrow) / np.linalg.norm(W)
        d_wide = (W @ t + B) / np.linalg.norm(W)
        drift = N_HAT * ((abs(d_narrow) + abs(d_wide)) / 2)
        self.play(
            pt_l.animate.shift(la.c2p(*(t + drift)) - la.c2p(*t)),
            pt_r.animate.shift(ra.c2p(*(t + drift)) - ra.c2p(*t)),
            run_time=1.6,
        )
        self.play(pt_l.animate.set_color(VIOLATION), pt_r.animate.set_color(MARGIN), run_time=0.6)

        fl = Text("flips!", color=VIOLATION, font_size=26).next_to(la, DOWN, buff=0.35)
        fr = Text("stays put", color=MARGIN, font_size=26).next_to(ra, DOWN, buff=0.35)
        self.play(Write(fl), Write(fr), run_time=1.0)
        self.wait(2.5)

        self.play(FadeOut(VGroup(la, ra, ld, rd, st_l, st_r, pt_l, pt_r,
                                 head_l, head_r, fl, fr)), run_time=1.0)


class S10_SoftMargin(SVMScene):
    def construct(self):
        ax, dots = base_plot(center=UP * 0.9, height=3.0)
        st = street(ax, fill=True)
        self.play(FadeIn(ax), FadeIn(dots), FadeIn(st), run_time=1.5)

        # Three hard examples sitting inside the street, straddling the boundary.
        violators = [
            MID + N_HAT * 0.20,
            MID - N_HAT * 0.25,
            MID + T_HAT * 0.95 + N_HAT * 0.10,
        ]
        vdots = VGroup(*[Dot(ax.c2p(*v), radius=0.11, color=VIOLATION) for v in violators])
        self.play(LaggedStart(*[GrowFromCenter(d) for d in vdots], lag_ratio=0.25), run_time=1.5)

        xis = VGroup(*[
            MathTex(r"\xi", color=VIOLATION, font_size=30).next_to(d, UR, buff=0.05)
            for d in vdots
        ])
        self.play(LaggedStart(*[FadeIn(x, shift=UP * 0.15) for x in xis], lag_ratio=0.25),
                  run_time=1.2)
        self.wait(0.5)

        obj = self.bottom_eq(
            r"\min\ \tfrac{1}{2}\lVert \mathbf{w}\rVert^{2} \;+\; C\sum_i \xi_i",
            font_size=42,
        )
        self.play(Write(obj), run_time=1.6)
        self.wait(1.0)

        # C knob: sliding it changes the street width.
        slider = NumberLine(x_range=[0, 1, 1], length=3.6, include_numbers=False,
                            color=DIM).to_corner(DR, buff=0.8)
        clabel = MathTex("C", color=INK, font_size=36).next_to(slider, LEFT, buff=0.25)
        knob = Dot(slider.n2p(0.3), color=CLASS_B, radius=0.11)
        small = Text("small C", color=DIM, font_size=20).next_to(slider, UP, buff=0.12)
        big = Text("large C", color=DIM, font_size=20).next_to(slider, DOWN, buff=0.12)
        self.play(FadeIn(slider), Write(clabel), FadeIn(knob), FadeIn(small), FadeIn(big),
                  run_time=1.0)

        w_wide, b_wide = separator_scaled(W, B, 1.0, MID)
        w_tight, b_tight = separator_scaled(W, B, 0.5, MID)
        self.play(
            knob.animate.move_to(slider.n2p(0.9)),
            Transform(st, street(ax, w_tight, b_tight, fill=True)),
            run_time=2.0,
        )
        self.play(Indicate(vdots, color=CLASS_B, scale_factor=1.2), run_time=1.2)
        self.wait(0.8)
        self.play(
            knob.animate.move_to(slider.n2p(0.2)),
            Transform(st, street(ax, w_wide, b_wide, fill=True)),
            run_time=2.0,
        )
        self.wait(2.0)

        self.play(FadeOut(VGroup(ax, dots, st, vdots, xis, obj,
                                 slider, clabel, knob, small, big)), run_time=1.0)

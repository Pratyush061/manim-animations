"""
Part 3 — The mathematics
========================

S06_Hyperplane    The boundary equation w·x + b = 0, built and explained term by
                  term, with the normal vector drawn on the graph.
S07_Margin        The distance from a point to the boundary, then the margin
                  width 2/||w||.
S08_Optimization  The hard-margin problem: min ½||w||² subject to the margin
                  constraints, and why it is a clean convex problem.

Render:  manim -qm part-03-math.py S06_Hyperplane
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from svm_common import *  # noqa: E402,F403


def _legend(sym: str, words: str, color):
    """One legend row: a maths symbol followed by an English gloss."""
    return VGroup(
        MathTex(sym, color=color, font_size=32),
        Text(words, color=INK, font_size=20),
    ).arrange(RIGHT, buff=0.15)


class S06_Hyperplane(SVMScene):
    def construct(self):
        ax, dots = base_plot(width=5.4, height=2.8, center=LEFT * 2.6 + UP * 0.3)
        st = street(ax, fill=False)
        self.play(FadeIn(ax), FadeIn(dots), run_time=1.2)
        self.play(Create(st), run_time=1.2)

        # Submobject layout of the split equation:
        # [0]=w  [1]=\cdot  [2]=x  [3]=+  [4]=b  [5]="= 0"
        eq = self.bottom_eq(
            r"{{\mathbf{w}}}\cdot{{\mathbf{x}}} + {{b}} = 0",
            font_size=48,
            tex_to_color_map={r"\mathbf{w}": CLASS_B, "b": CLASS_A},
        )
        self.play(Write(eq), run_time=1.6)
        self.wait(0.4)

        # The normal vector w, perpendicular to the boundary.
        n = W / np.linalg.norm(W)
        wvec = Arrow(ax.c2p(*MID), ax.c2p(*(MID + n * 1.1)), buff=0,
                     color=CLASS_B, stroke_width=5,
                     max_tip_length_to_length_ratio=0.22)
        wlab = MathTex(r"\mathbf{w}", color=CLASS_B, font_size=38).next_to(wvec.get_end(), UR, buff=0.05)
        self.play(GrowArrow(wvec), Write(wlab), run_time=1.2)
        self.wait(0.3)

        # Spotlight each term as the narration names it.
        self.play(Indicate(eq[0], color=CLASS_B, scale_factor=1.4), run_time=1.1)
        self.play(Indicate(eq[2], color=INK, scale_factor=1.4), run_time=1.0)
        self.play(Indicate(eq[4], color=CLASS_A, scale_factor=1.4), run_time=1.1)
        self.wait(0.4)

        legend = VGroup(
            _legend(r"\mathbf{w}", "normal vector (orientation)", CLASS_B),
            _legend(r"\mathbf{x}", "point on the boundary", INK),
            _legend("b", "bias (shifts the line)", CLASS_A),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.18).to_corner(UR, buff=0.5)
        self.play(LaggedStart(*[FadeIn(l, shift=LEFT * 0.2) for l in legend], lag_ratio=0.3),
                  run_time=1.5)
        self.wait(2.5)

        self.play(FadeOut(VGroup(ax, dots, st, eq, wvec, wlab, legend)), run_time=1.0)


class S07_Margin(SVMScene):
    def construct(self):
        ax, dots = base_plot(width=5.4, height=2.8, center=UP * 0.9)
        st = street(ax, fill=True)
        self.play(FadeIn(ax), FadeIn(dots), FadeIn(st), run_time=1.5)

        # Drop a perpendicular from a support vector to the boundary.
        p = SV_A
        foot = p - ((W @ p + B) / (W @ W)) * W
        pdot = Dot(ax.c2p(*p), radius=0.11, color=CLASS_B)
        drop = Line(ax.c2p(*p), ax.c2p(*foot), color=CLASS_B, stroke_width=3)
        dlab = MathTex("d", color=CLASS_B, font_size=34).next_to(drop.get_center(), LEFT, buff=0.1)
        self.play(FadeIn(pdot), Create(drop), Write(dlab), run_time=1.4)
        self.wait(0.5)

        d_eq = self.bottom_eq(
            r"d = \frac{\lvert \mathbf{w}\cdot\mathbf{x}_i + b \rvert}{\lVert \mathbf{w}\rVert}",
            font_size=46,
        )
        self.play(Write(d_eq), run_time=1.6)
        self.wait(1.5)

        # Both rails are the same distance from the boundary -> margin = 2/||w||.
        self.play(FadeOut(d_eq), run_time=0.6)
        m_eq = self.bottom_eq(r"\text{margin} = \frac{2}{\lVert \mathbf{w}\rVert}", font_size=46)
        self.play(Write(m_eq), run_time=1.6)
        self.play(Indicate(m_eq[0][2:], color=MARGIN, scale_factor=1.3), run_time=1.2)

        hint = Text("wider street  ⇔  smaller ‖w‖", color=INK, font_size=26)
        hint.next_to(m_eq, UP, buff=0.3)
        self.play(FadeIn(hint, shift=UP * 0.2), run_time=1.0)
        self.wait(2.5)

        self.play(FadeOut(VGroup(ax, dots, st, pdot, drop, dlab, m_eq, hint)), run_time=1.0)


class S08_Optimization(SVMScene):
    def construct(self):
        ax, dots = base_plot(width=5.6, height=2.4, center=UP * 1.1)
        st = street(ax, fill=False)
        self.play(FadeIn(ax), FadeIn(dots), FadeIn(st), run_time=1.5)

        goal = MathTex(
            r"\max_{\mathbf{w},b}\ \frac{2}{\lVert \mathbf{w}\rVert}",
            r"\quad\Longrightarrow\quad",
            r"\min_{\mathbf{w},b}\ \tfrac{1}{2}\lVert \mathbf{w}\rVert^{2}",
            font_size=42, color=INK,
        ).to_edge(DOWN, buff=1.5)
        self.play(Write(goal), run_time=1.8)
        self.wait(0.6)

        cons = MathTex(
            r"y_i\,(\mathbf{w}\cdot\mathbf{x}_i + b)\ \ge\ 1 \qquad \forall i",
            font_size=40, color=INK,
        ).to_edge(DOWN, buff=0.6)
        self.play(Write(cons), run_time=1.6)
        self.wait(0.5)

        yleg = VGroup(
            MathTex(r"y_i \in \{+1, -1\}", color=CLASS_B, font_size=32),
            Text("the label — which class", color=INK, font_size=22),
        ).arrange(RIGHT, buff=0.15).to_corner(UR, buff=0.5)
        self.play(FadeIn(yleg, shift=LEFT * 0.2), run_time=1.0)

        box = SurroundingRectangle(VGroup(goal, cons), color=MARGIN, buff=0.28)
        tag = Text("Hard-margin SVM — convex, unique solution",
                   color=MARGIN, font_size=22).next_to(box, DOWN, buff=0.2)
        self.play(Create(box), run_time=1.2)
        self.play(Write(tag), run_time=1.2)
        self.wait(2.5)

        self.play(FadeOut(VGroup(ax, dots, st, goal, cons, yleg, box, tag)), run_time=1.0)

"""
Part 5 — When a straight line is not enough
===========================================

S11_TheWall       Concentric rings: no straight line can separate them.
S12_TheLift       Lift the data into 3D with z = x1² + x2² — now a flat plane
                  separates the two classes cleanly.
S13_KernelTrick   Project the plane back down: it becomes a curved boundary.
                  The kernel computes the high-dimensional dot product without
                  ever visiting the high-dimensional space.
S14_KernelMenu    Linear vs polynomial vs RBF, side by side, with the γ knob.

Render:  manim -qm part-05-kernels.py S12_TheLift
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from svm_common import *  # noqa: E402,F403


class S11_TheWall(SVMScene):
    def construct(self):
        ax = make_axes(5.0, 5.0, DOWN * 0.3, RING_XLIM, RING_YLIM)
        inner = scatter(ax, INNER_PTS, CLASS_A)
        outer = scatter(ax, OUTER_PTS, CLASS_B)
        self.play(
            FadeIn(ax),
            LaggedStart(*[GrowFromCenter(d) for d in VGroup(inner, outer)], lag_ratio=0.02),
            run_time=2.0,
        )

        def line_at(theta):
            d = np.array([np.cos(theta), np.sin(theta)]) * 3.5
            return Line(ax.c2p(*(-d)), ax.c2p(*d), color=INK, stroke_width=4)

        ln = line_at(0.0)
        self.play(Create(ln), run_time=0.8)
        for deg in (30, 60, 90, 120, 150):
            self.play(Transform(ln, line_at(np.deg2rad(deg))), run_time=0.6)
        self.play(ln.animate.set_color(VIOLATION), run_time=0.5)
        self.play(FadeOut(ln), run_time=0.5)

        caption = Text("No straight line can separate this.",
                       color=INK, font_size=28).to_edge(DOWN, buff=0.6)
        self.play(FadeIn(caption, shift=UP * 0.2), run_time=1.0)
        self.wait(2.0)

        self.play(FadeOut(VGroup(ax, inner, outer, caption)), run_time=1.0)


class S12_TheLift(ThreeDScene):
    def construct(self):
        axes = ThreeDAxes(
            x_range=[-3, 3, 1], y_range=[-3, 3, 1], z_range=[0, 8, 2],
            x_length=6, y_length=6, z_length=4.5,
        )
        self.set_camera_orientation(phi=65 * DEGREES, theta=-60 * DEGREES)

        caption = Text("Lift the data: z = x1² + x2²", color=INK, font_size=28)
        self.add_fixed_in_frame_mobjects(caption)
        caption.to_edge(DOWN, buff=0.6)

        pairs = ((INNER_PTS, CLASS_A), (OUTER_PTS, CLASS_B))
        flat = VGroup(*[
            Dot3D(axes.c2p(p[0], p[1], 0), radius=0.075, color=c)
            for pts, c in pairs for p in pts
        ])
        lifted = VGroup(*[
            Dot3D(axes.c2p(p[0], p[1], p[0] ** 2 + p[1] ** 2), radius=0.075, color=c)
            for pts, c in pairs for p in pts
        ])

        self.play(Create(axes), run_time=1.5)
        self.add(flat)
        self.play(FadeIn(flat), run_time=0.5)
        self.play(Transform(flat, lifted), run_time=3.0)
        self.wait(0.8)

        # A flat, horizontal plane cleanly separates the raised ring from the
        # low centre — no curve required.
        plane = Surface(
            lambda u, v: axes.c2p(u, v, 3.0),
            u_range=[-3, 3], v_range=[-3, 3],
            resolution=(2, 2), fill_opacity=0.25, color=MARGIN, stroke_width=0,
        )
        self.play(Create(plane), run_time=1.5)
        self.begin_ambient_camera_rotation(rate=0.15)
        self.wait(3.5)
        self.stop_ambient_camera_rotation()
        self.wait(0.5)

        self.play(FadeOut(VGroup(axes, flat, plane, caption)), run_time=1.0)


class S13_KernelTrick(SVMScene):
    def construct(self):
        ax = make_axes(4.8, 4.8, UP * 0.6, RING_XLIM, RING_YLIM)
        inner = scatter(ax, INNER_PTS, CLASS_A)
        outer = scatter(ax, OUTER_PTS, CLASS_B)
        self.play(FadeIn(ax), FadeIn(inner), FadeIn(outer), run_time=1.5)

        r = np.linalg.norm(ax.c2p(1.7, 0) - ax.c2p(0, 0))
        circ = Circle(radius=r, color=MARGIN, stroke_width=4).move_to(ax.c2p(0, 0))
        self.play(Create(circ), run_time=1.5)
        cap = Text("A flat plane becomes a curved boundary",
                   color=INK, font_size=26).next_to(ax, DOWN, buff=0.35)
        self.play(FadeIn(cap, shift=UP * 0.2), run_time=1.0)
        self.wait(0.6)

        keq = self.bottom_eq(
            r"K(\mathbf{x}_i,\mathbf{x}_j) = \phi(\mathbf{x}_i)\cdot\phi(\mathbf{x}_j)",
            font_size=40,
        )
        self.play(Write(keq), run_time=1.6)
        self.wait(0.6)

        bad = VGroup(
            Text("compute", color=DIM, font_size=24),
            MathTex(r"\phi", color=VIOLATION, font_size=34),
            Cross(scale_factor=0.2, color=VIOLATION),
        ).arrange(RIGHT, buff=0.15)
        good = VGroup(
            Text("call", color=DIM, font_size=24),
            MathTex("K", color=MARGIN, font_size=34),
            MathTex(r"\checkmark", color=MARGIN, font_size=34),
        ).arrange(RIGHT, buff=0.15)
        legend = VGroup(bad, good).arrange(DOWN, aligned_edge=LEFT, buff=0.25).to_corner(UR, buff=0.6)
        self.play(FadeIn(legend, shift=LEFT * 0.2), run_time=1.2)
        self.play(Indicate(good, color=MARGIN, scale_factor=1.2), run_time=1.2)
        self.wait(2.5)

        self.play(FadeOut(VGroup(ax, inner, outer, circ, cap, keq, legend)), run_time=1.0)


def _closed_curve(ax, radius_fn, color, **style):
    """A closed parametric curve in the axes, radius given as a function of t."""
    style.setdefault("stroke_width", 4)
    return ParametricFunction(
        lambda t: ax.c2p(radius_fn(t) * np.cos(t), radius_fn(t) * np.sin(t)),
        t_range=[0, TAU], color=color, **style,
    )


class S14_KernelMenu(SVMScene):
    def construct(self):
        centres = [LEFT * 4.4, ORIGIN, RIGHT * 4.4]
        panels, dots = [], []
        for c in centres:
            ax = make_axes(3.4, 3.4, c + UP * 0.7, RING_XLIM, RING_YLIM)
            d = VGroup(scatter(ax, INNER_PTS, CLASS_A, 0.055),
                       scatter(ax, OUTER_PTS, CLASS_B, 0.055))
            panels.append(ax)
            dots.append(d)
        self.play(*[FadeIn(a) for a in panels], *[FadeIn(d) for d in dots], run_time=1.4)

        # Linear — a straight line, which fails here.
        lin = Line(panels[0].c2p(*(-np.array([2.6, 0.9]))),
                   panels[0].c2p(*np.array([2.6, 0.9])), color=VIOLATION, stroke_width=4)
        # Polynomial — a smooth closed curve.
        poly = _closed_curve(panels[1], lambda t: 1.55, MARGIN)
        # RBF — a slightly wiggly curve that hugs the ring.
        rbf = _closed_curve(panels[2], lambda t: 1.7 + 0.10 * np.sin(5 * t), MARGIN)

        self.play(Create(lin), Create(poly), Create(rbf), run_time=1.6)

        labels = VGroup(
            Text("Linear", color=INK, font_size=24).next_to(panels[0], DOWN, buff=0.3),
            Text("Polynomial", color=INK, font_size=24).next_to(panels[1], DOWN, buff=0.3),
            Text("RBF", color=INK, font_size=24).next_to(panels[2], DOWN, buff=0.3),
        )
        self.play(Write(labels), run_time=1.2)
        self.wait(0.6)

        # RBF formula + the γ knob.
        keq = self.bottom_eq(r"K = e^{-\gamma\lVert \mathbf{x} - \mathbf{x}' \rVert^{2}}", font_size=38)
        self.play(Write(keq), run_time=1.5)

        slider = NumberLine(x_range=[0, 1, 1], length=2.8, include_numbers=False,
                            color=DIM).to_corner(DL, buff=0.8)
        glabel = MathTex(r"\gamma", color=INK, font_size=34).next_to(slider, LEFT, buff=0.2)
        knob = Dot(slider.n2p(0.25), color=CLASS_B, radius=0.1)
        self.play(FadeIn(slider), Write(glabel), FadeIn(knob), run_time=1.0)

        wiggly = _closed_curve(panels[2], lambda t: 1.7 + 0.28 * np.sin(7 * t), MARGIN)
        smooth = _closed_curve(panels[2], lambda t: 1.6, MARGIN)
        self.play(knob.animate.move_to(slider.n2p(0.95)), Transform(rbf, wiggly), run_time=1.8)
        self.wait(0.5)
        self.play(knob.animate.move_to(slider.n2p(0.15)), Transform(rbf, smooth), run_time=1.8)
        self.wait(2.0)

        self.play(FadeOut(VGroup(*panels, *dots, lin, poly, rbf, labels,
                                 keq, slider, glabel, knob)), run_time=1.0)

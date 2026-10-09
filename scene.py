from manim import *


class MyScene(Scene):
    """A simple Manim animation: a shape morphs, then a formula appears."""

    def construct(self):
        # Title
        title = Text("Manim + GitHub Actions", font_size=48)
        self.play(Write(title))
        self.wait(0.5)
        self.play(title.animate.to_edge(UP))

        # A circle that transforms into a square
        circle = Circle(color=BLUE, radius=1.2)
        square = Square(color=RED, side_length=2.2)
        self.play(Create(circle))
        self.play(Transform(circle, square))
        self.wait(0.5)

        # Morph into a famous formula
        formula = MathTex(r"E = mc^2", color=YELLOW, font_size=72)
        self.play(Transform(circle, formula))
        self.wait(1.5)

        # Clean up
        self.play(FadeOut(circle), FadeOut(title))
        self.wait(0.5)

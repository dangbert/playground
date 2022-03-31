"""
based on https://3b1b.github.io/manim/getting_started/quickstart.html
run with:
  manimgl start.py SquareToCircle
  manimgl start.py SquareToCircle -o

"""
from manimlib import *

class SquareToCircle(Scene):
  def construct(self):
    circle = Circle()
    circle.set_fill(BLUE, opacity=0.6)
    circle.set_stroke(BLUE_E, width=4)

    square = Square()
    self.play(ShowCreation(square))
    self.wait()
    self.play(ReplacementTransform(square, circle))
    self.wait()

    # enable interaction
    #   lets you write more code when animation ends (great way to experiment)
    #   some example commands here https://3b1b.github.io/manim/getting_started/quickstart.html
    #   (square, circle, and self will all be available in the namespace of the terminal)
    self.embed()
    #self.add(circle)

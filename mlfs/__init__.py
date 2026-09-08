"""Shared infrastructure for the curriculum.

Holds helpers that many subtopics need — synthetic data, numerical checks,
plotting. It does not hold lesson material: an algorithm lives in the subtopic
that teaches it, and is promoted here only once that subtopic is complete.
"""

from .datasets import make_linear_data

__all__ = ["make_linear_data"]

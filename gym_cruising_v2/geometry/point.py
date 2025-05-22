""" This module contains the Point class. """
from __future__ import annotations

import math
from typing import Any

import numpy as np


class Point:
    """ A point in Cartesian plane. """

    point_x: int
    point_y: int
    covered: bool
    step_from_last_visit: int # Quanti step fa è stato visitato
    unexplored_point_max_steps: int

    def __init__(self, point_x: int, point_y: int, unexplored_point_max_steps: int) -> None:
        self.point_x = point_x
        self.point_y = point_y
        self.covered = False
        self.step_from_last_visit = unexplored_point_max_steps #Serve per incentivare esplorazione iniziale
        self.unexplored_point_max_steps = unexplored_point_max_steps

    def calculate_distance(self, other: Point) -> float:
        """ Calculate the Euclidean distance between two points. """
        return math.sqrt((self.point_x - other.point_x) ** 2
                         + (self.point_y - other.point_y) ** 2)

    def is_in_area(self, area: np.ndarray) -> bool:
        if self.point_x <= area[0, 0] or self.point_x >= area[0, 1]:
            return False
        if self.point_y <= area[1, 0] or self.point_y >= area[1, 1]:
            return False
        return True
    
    def increment_step_from_last_visit(self):
        if(self.step_from_last_visit < self.unexplored_point_max_steps):
            self.step_from_last_visit += 1
        
    def reset_step_from_last_visit(self):
        self.step_from_last_visit = 0
        
    def set_covered(self, covered):
        self.covered = covered

    def __eq__(self, other: Any) -> bool:
        return (isinstance(other, Point)
                and math.isclose(self.point_x, other.point_x)
                and math.isclose(self.point_y, other.point_y))

    def __repr__(self) -> str:
        return f'({self.point_x}, {self.point_y})'

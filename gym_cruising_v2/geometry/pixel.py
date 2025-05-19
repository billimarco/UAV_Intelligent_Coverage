import math
from typing import Any
from typing import Tuple

import numpy as np
from typing import List
from gym_cruising_v2.geometry.point import Point


class Pixel:
    """ A point in Cartesian plane. """

    x_coordinate: float
    y_coordinate: float
    color: Tuple[int, int, int]
    point_grid: List[List[Point]] # resolution * resolution in m^2
    mean_step_from_last_visit: float
    unexplored_point_max_steps: int

    def __init__(self, x_coordinate: float, y_coordinate: float, resolution: int, unexplored_point_max_steps: int) -> None:
        self.x_coordinate = x_coordinate
        self.y_coordinate = y_coordinate
        self.color = (255, 255, 255)
        self.point_grid = [[Point(i + x_coordinate*resolution, j+ y_coordinate*resolution, unexplored_point_max_steps) for j in range(resolution)] for i in range(resolution)]
        self.mean_step_from_last_visit = 0
        self.unexplored_point_max_steps = unexplored_point_max_steps

    def increment_step_from_last_visit_for_points(self):
        for row in self.point_grid:
            for point in row:
                point.increment_step_from_last_visit()
        self.calculate_mean_step_from_last_visit()
        
    def calculate_mean_step_from_last_visit(self):
        total = 0
        count = 0

        for row in self.point_grid:
            for point in row:
                total += point.step_from_last_visit
                count += 1

        if count > 0:
            self.mean_step_from_last_visit = total / count
        else:
            self.mean_step_from_last_visit = 0
        
        self.define_color()
    
    def define_color(self):
        rgb_value = 255 - math.floor(255 * (self.mean_step_from_last_visit / self.unexplored_point_max_steps))
        return (rgb_value, rgb_value, rgb_value)

    def __eq__(self, other: Any) -> bool:
        return (isinstance(other, Point)
                and math.isclose(self.x_coordinate, other.x_coordinate)
                and math.isclose(self.y_coordinate, other.y_coordinate))

    def __repr__(self) -> str:
        return f'({self.x_coordinate}, {self.y_coordinate})'

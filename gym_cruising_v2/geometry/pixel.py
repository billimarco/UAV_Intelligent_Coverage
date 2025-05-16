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

    def __init__(self, x_coordinate: float, y_coordinate: float, resolution: int) -> None:
        self.x_coordinate = x_coordinate
        self.y_coordinate = y_coordinate
        self.color = (255, 255, 255)
        self.point_grid = [[Point(i + x_coordinate*resolution, j+ y_coordinate*resolution) for j in range(resolution)] for i in range(resolution)]
        self.calculate_mean_step_from_last_visit()

    def is_in_area(self, area: np.ndarray) -> bool:
        if self.x_coordinate <= area[0, 0] or self.x_coordinate >= area[0, 1]:
            return False
        if self.y_coordinate <= area[1, 0] or self.y_coordinate >= area[1, 1]:
            return False
        return True
    
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

    
    def define_color(self):
        #TODO c'è da decidere come calcolarlo in base al mean_step_from_last_visit
        return (255, 255, 255)

    def __eq__(self, other: Any) -> bool:
        return (isinstance(other, Point)
                and math.isclose(self.x_coordinate, other.x_coordinate)
                and math.isclose(self.y_coordinate, other.y_coordinate))

    def __repr__(self) -> str:
        return f'({self.x_coordinate}, {self.y_coordinate})'

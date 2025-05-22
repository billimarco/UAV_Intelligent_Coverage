""" This module contains the Point class. """
from __future__ import annotations

import math
from typing import Any

import numpy as np
from gym_cruising_v2.geometry.point import Point


class Coordinate:
    """ A point in Cartesian plane. """

    x_coordinate: float
    y_coordinate: float
    z_coordinate: float

    def __init__(self, x_coordinate: float, y_coordinate: float, z_coordinate) -> None:
        self.x_coordinate = x_coordinate
        self.y_coordinate = y_coordinate
        self.z_coordinate = z_coordinate

    def calculate_distance_to_coordinate(self, other: Coordinate) -> float:
        """ Calculate the Euclidean distance between two coordinate. """
        return math.sqrt((self.x_coordinate - other.x_coordinate) ** 2
                         + (self.y_coordinate - other.y_coordinate) ** 2
                         + (self.z_coordinate - other.z_coordinate) ** 2)
        
    def calculate_distance_to_point(self, point: Point):
        """ Calculate the Euclidean distance between one coordinate and one point. """
        return math.sqrt((self.x_coordinate - point.point_x) ** 2
                         + (self.y_coordinate - point.point_y) ** 2
                         + (self.z_coordinate) ** 2)

    def is_in_area(self, area: np.ndarray) -> bool:
        if self.x_coordinate <= area[0, 0] or self.x_coordinate >= area[0, 1]:
            return False
        if self.y_coordinate <= area[1, 0] or self.y_coordinate >= area[1, 1]:
            return False
        return True
    
    def is_in_volume(self, volume: np.ndarray) -> bool:
        if self.x_coordinate <= volume[0, 0] or self.x_coordinate >= volume[0, 1]:
            return False
        if self.y_coordinate <= volume[1, 0] or self.y_coordinate >= volume[1, 1]:
            return False
        if self.z_coordinate <= volume[2, 0] or self.z_coordinate >= volume[2, 1]:
            return False
        return True


    def __eq__(self, other: Any) -> bool:
        return (isinstance(other, Coordinate)
                and math.isclose(self.x_coordinate, other.x_coordinate)
                and math.isclose(self.y_coordinate, other.y_coordinate))

    def __repr__(self) -> str:
        return f'({self.x_coordinate}, {self.y_coordinate})'
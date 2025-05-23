""" This module contains the Line class. """
from __future__ import annotations

import math
from typing import Any

from gym_cruising_v2.geometry.point import Point


class NoIntersectionError(Exception):
    """ Exception when there is no intersection between two lines. """


class Line:
    """ A line (line segment) in enviroment. """

    start: Point
    end: Point
    slope: float  # m
    y_intercept: float  # q

    def __init__(self, start: Point, end: Point) -> None:
        self.start = start
        self.end = end
        if start.point_col == end.point_col:  # Vertical line
            self.slope = 0
            self.y_intercept = math.inf
        elif start.point_row == end.point_row:  # Horizontal line
            self.slope = 0
            self.y_intercept = start.point_row
        else:
            self.slope = ((start.point_row - end.point_row)
                          / (start.point_col - end.point_col))
            self.y_intercept = (
                    start.point_row - self.slope * start.point_col)

    def get_intersection(self, other: Line) -> Point:
        """Get the intersection point between two lines.

        Raise an error if it does not exist.
        """
        if self.slope == other.slope:  # Parallel lines
            raise NoIntersectionError

        if self.start.point_col == self.end.point_col:
            x_coordinate = self.start.point_col
            y_coordinate = other.slope * x_coordinate + other.y_intercept
        elif other.start.point_col == other.end.point_col:
            x_coordinate = other.start.point_col
            y_coordinate = self.slope * x_coordinate + self.y_intercept
        elif self.start.point_row == self.end.point_row:
            y_coordinate = self.start.point_row
            x_coordinate = (y_coordinate - other.y_intercept) / other.slope
        elif other.start.point_row == other.end.point_row:
            y_coordinate = other.start.point_row
            x_coordinate = (y_coordinate - self.y_intercept) / self.slope
        else:
            x_coordinate = ((self.y_intercept - other.y_intercept)
                            / (other.slope - self.slope))
            y_coordinate = self.slope * x_coordinate + self.y_intercept

        intersection = Point(x_coordinate, y_coordinate)

        if self.contains(intersection) and other.contains(intersection):
            return intersection

        raise NoIntersectionError

    def contains(self, point: Point) -> bool:
        """Calculate if the line contains a given point."""
        contains_x = (
                min(self.start.point_col, self.end.point_col)
                <= point.point_col
                <= max(self.start.point_col, self.end.point_col))
        contains_y = (
                min(self.start.point_row, self.end.point_row)
                <= point.point_row
                <= max(self.start.point_row, self.end.point_row))
        return contains_x and contains_y

    def __eq__(self, other: Any) -> bool:
        return (isinstance(other, Line)
                and ((self.start == other.start and self.end == other.end)
                     or (self.start == other.end and self.end == other.start)))

    def __repr__(self) -> str:
        return f'Start = {self.start}, Yaw = {self.end}'

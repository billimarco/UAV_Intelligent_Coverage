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
        if start.point_x == end.point_x:  # Vertical line
            self.slope = 0
            self.y_intercept = math.inf
        elif start.point_y == end.point_y:  # Horizontal line
            self.slope = 0
            self.y_intercept = start.point_y
        else:
            self.slope = ((start.point_y - end.point_y)
                          / (start.point_x - end.point_x))
            self.y_intercept = (
                    start.point_y - self.slope * start.point_x)

    def get_intersection(self, other: Line) -> Point:
        """Get the intersection point between two lines.

        Raise an error if it does not exist.
        """
        if self.slope == other.slope:  # Parallel lines
            raise NoIntersectionError

        if self.start.point_x == self.end.point_x:
            x_coordinate = self.start.point_x
            y_coordinate = other.slope * x_coordinate + other.y_intercept
        elif other.start.point_x == other.end.point_x:
            x_coordinate = other.start.point_x
            y_coordinate = self.slope * x_coordinate + self.y_intercept
        elif self.start.point_y == self.end.point_y:
            y_coordinate = self.start.point_y
            x_coordinate = (y_coordinate - other.y_intercept) / other.slope
        elif other.start.point_y == other.end.point_y:
            y_coordinate = other.start.point_y
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
                min(self.start.point_x, self.end.point_x)
                <= point.point_x
                <= max(self.start.point_x, self.end.point_x))
        contains_y = (
                min(self.start.point_y, self.end.point_y)
                <= point.point_y
                <= max(self.start.point_y, self.end.point_y))
        return contains_x and contains_y

    def __eq__(self, other: Any) -> bool:
        return (isinstance(other, Line)
                and ((self.start == other.start and self.end == other.end)
                     or (self.start == other.end and self.end == other.start)))

    def __repr__(self) -> str:
        return f'Start = {self.start}, Yaw = {self.end}'

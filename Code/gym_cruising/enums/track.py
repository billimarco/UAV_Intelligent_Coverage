""" This module contains the Track enum """
from enum import Enum
from typing import Tuple

from gym_cruising.geometry.line import Line
from gym_cruising.geometry.point import Point


class Track(Enum):
    # pylint: disable=invalid-name
    walls: Tuple[Line, ...]
    spawn_area: Tuple[Tuple[Tuple[float, float], Tuple[float, float]], ...]

    # pylint: enable=invalid-name

    def __new__(cls,
                value: int,
                walls: Tuple[Line, ...] = (),
                spawn_area: Tuple[Tuple[Tuple[float, float],
                Tuple[float, float]], ...] = ()):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.walls = walls
        obj.spawn_area = spawn_area
        return obj

    # RESOLUTION = 0.1667
    TRACK1 = (1,
              (
                  Line(Point(0, 0), Point(0, 6000)),
                  Line(Point(0, 6000), Point(6000, 6000)),
                  Line(Point(6000, 6000), Point(6000, 0)),
                  Line(Point(6000, 0), Point(0, 0))
              ),
              (
                  ((90, 5910), (90, 5910)),
              )
              )

    # RESOLUTION = 0.25
    TRACK2 = (2,
              (
                  Line(Point(0, 0), Point(0, 4000)),
                  Line(Point(0, 4000), Point(4000, 4000)),
                  Line(Point(4000, 4000), Point(4000, 0)),
                  Line(Point(4000, 0), Point(0, 0))
              ),
              (
                  ((60, 3940), (60, 3940)),
              )
              )

    # RESOLUTION = 0.3333
    TRACK3 = (3,
              (
                  Line(Point(0, 0), Point(0, 3000)),
                  Line(Point(0, 3000), Point(3000, 3000)),
                  Line(Point(3000, 3000), Point(3000, 0)),
                  Line(Point(3000, 0), Point(0, 0))
              ),
              (
                  ((45, 2955), (45, 2955)),
              )
              )

    # RESOLUTION = 0.50
    TRACK4 = (4,
              (
                  Line(Point(0, 0), Point(0, 2000)),
                  Line(Point(0, 2000), Point(2000, 2000)),
                  Line(Point(2000, 2000), Point(2000, 0)),
                  Line(Point(2000, 0), Point(0, 0))
              ),
              (
                  ((30, 1970), (30, 1970)),
              )
              )

    # numero, linee muri e area dove oggetti possono apparire (spawnare)

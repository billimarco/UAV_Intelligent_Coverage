""" This module contains the Grid enum """
from enum import Enum
from typing import Tuple

from gym_cruising.geometry.line import Line
from gym_cruising.geometry.point import Point

class Grid:
    def __init__(self,
                 window_width: int,
                 window_height: int,
                 resolution: float,
                 spawn_offset: int):
        self.window_width = window_width
        self.window_height = window_height
        self.resolution = resolution
        self.spawn_offset = spawn_offset

        # Converti window size in pixels to meters
        self.width_meters = window_width / resolution
        self.height_meters = window_height / resolution

        # Costruisci i muri (cornice rettangolare)
        self.walls: Tuple[Line, ...] = (
            Line(Point(0, 0), Point(0, self.height_meters)),
            Line(Point(0, self.height_meters), Point(self.width_meters, self.height_meters)),
            Line(Point(self.width_meters, self.height_meters), Point(self.width_meters, 0)),
            Line(Point(self.width_meters, 0), Point(0, 0)),
        )

        # Zona di spawn: angolo in basso a sinistra, con offset
        spawn_offset_res = spawn_offset / resolution

        self.spawn_area: Tuple[Tuple[Tuple[float, float], Tuple[float, float]], ...] = (
            ((spawn_offset_res, self.height_meters - spawn_offset_res), (spawn_offset_res, self.width_meters - spawn_offset_res)),
        )

    # numero, linee muri e area dove oggetti possono apparire (spawnare)

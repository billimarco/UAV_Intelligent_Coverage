""" This module contains the Grid enum """
from enum import Enum
from typing import Tuple
from typing import List

from gym_cruising_v2.geometry.line import Line
from gym_cruising_v2.geometry.point import Point
from gym_cruising_v2.geometry.pixel import Pixel

class Grid:
    window_width: int
    window_height: int
    resolution: int
    spawn_offset: int
    pixel_grid: List[List[Pixel]]
    walls: Tuple[Line, ...]
    
    def __init__(self,
                 window_width: int,
                 window_height: int,
                 resolution: float,
                 spawn_offset: int):
        self.window_width = window_width
        self.window_height = window_height
        self.resolution = resolution
        self.spawn_offset = spawn_offset

        self.pixel_grid = [[Pixel(i, j, resolution) for j in range(window_height)] for i in range(window_width)]

        # Costruisci i muri (cornice rettangolare)
        '''
        self.walls = (
            Line(Point(0, 0), Point(0, self.height_meters)),
            Line(Point(0, self.height_meters), Point(self.width_meters, self.height_meters)),
            Line(Point(self.width_meters, self.height_meters), Point(self.width_meters, 0)),
            Line(Point(self.width_meters, 0), Point(0, 0)),
        )
        '''

        # Zona di spawn: angolo in basso a sinistra, con offset. Definita per punti
        spawn_offset_res = spawn_offset * resolution

        self.spawn_area: Tuple[Tuple[Tuple[float, float], Tuple[float, float]], ...] = (
            ((spawn_offset_res, window_width*resolution  - spawn_offset_res), (spawn_offset_res, window_height*resolution - spawn_offset_res)),
        )

    # numero, linee muri e area dove oggetti possono apparire (spawnare)

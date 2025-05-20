from typing import Tuple
from typing import List

import numpy as np
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
                 spawn_offset: int,
                 unexplored_point_max_steps: int):
        self.window_width = window_width
        self.window_height = window_height
        self.resolution = resolution
        self.spawn_offset = spawn_offset
        self.unexplored_point_max_steps = unexplored_point_max_steps

        self.pixel_grid = [[Pixel(i, j, resolution, unexplored_point_max_steps) for j in range(window_height)] for i in range(window_width)]

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
    
    def reset(self):
        self.pixel_grid = [[Pixel(i, j, self.resolution, self.unexplored_point_max_steps) for j in range(self.window_height)] for i in range(self.window_width)]

    def get_pixel(self, pixel_x: int, pixel_y: int) -> Pixel:
        """Restituisce il Pixel alla posizione (x, y) nella griglia di pixel."""
        if 0 <= pixel_x < self.window_width and 0 <= pixel_y < self.window_height:
            return self.pixel_grid[pixel_x][pixel_y]
        else:
            raise IndexError(f"Pixel coordinates out of bounds: ({pixel_x}, {pixel_y})")

    def get_point(self, point_x: int, point_y: int) -> Point:
        """
        Restituisce il Point globale alla posizione (x, y) in coordinate punto (non pixel).
        Converte (x, y) assoluti in indici di pixel e posizione relativa nel point_grid.
        """
        if 0 <= point_x < self.window_width * self.resolution and 0 <= point_y < self.window_height * self.resolution:
            pixel_x = point_x // self.resolution
            pixel_y = point_y // self.resolution
            local_x = point_x % self.resolution
            local_y = point_y % self.resolution
            return self.pixel_grid[pixel_x][pixel_y].point_grid[local_x][local_y]
        else:
            raise IndexError(f"Point coordinates out of bounds: ({point_x}, {point_y})")

    def get_pixel_from_point(self, point:Point) -> Pixel:
        if 0 <= point.x_coordinate < self.window_width * self.resolution and 0 <= point.y_coordinate < self.window_height * self.resolution:
            pixel_x = point.x_coordinate // self.resolution
            pixel_y = point.y_coordinate // self.resolution
            return self.pixel_grid[pixel_x][pixel_y]
        else:
            raise IndexError(f"Point coordinates out of bounds: ({point_x}, {point_y})")
        
    def get_pixel_from_point_coordinate(self, point_x:int, point_y:int) -> Pixel:
        if 0 <= point_x < self.window_width * self.resolution and 0 <= point_y < self.window_height * self.resolution:
            pixel_x = point_x // self.resolution
            pixel_y = point_y // self.resolution
            return self.pixel_grid[pixel_x][pixel_y]
        else:
            raise IndexError(f"Point coordinates out of bounds: ({point_x}, {point_y})")
        
    def get_pixel_exploration_map(self) -> np.ndarray:
        exploration_map = np.zeros((self.window_width, self.window_height), dtype=np.float32)
        for i, row in enumerate(self.pixel_grid):
            for j, pixel in enumerate(row):
                # Ad esempio prendo mean_step_from_last_visit, o sostituisci con l'attributo corretto
                exploration_map[i, j] = pixel.mean_step_from_last_visit
        return exploration_map
    
    def get_point_exploration_map(self) -> np.ndarray:
        total_width = self.window_width * self.resolution
        total_height = self.window_height * self.resolution

        exploration_map = np.zeros((total_width, total_height), dtype=np.float32)

        for i in range(self.window_width):
            for j in range(self.window_height):
                pixel = self.pixel_grid[i][j]
                for x in range(self.resolution):
                    for y in range(self.resolution):
                        point = pixel.point_grid[x][y]
                        exploration_map[i * self.resolution + x, j * self.resolution + y] = point.step_from_last_visit

        return exploration_map
                
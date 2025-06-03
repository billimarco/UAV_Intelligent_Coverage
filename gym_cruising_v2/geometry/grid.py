from typing import Tuple
from typing import List

import numpy as np
import math
from gym_cruising_v2.geometry.line import Line
from gym_cruising_v2.geometry.coordinate import Coordinate
from gym_cruising_v2.geometry.point import Point
from gym_cruising_v2.geometry.pixel import Pixel

class Grid:
    """
    Classe che rappresenta una griglia composta da pixel, ognuno dei quali contiene una sotto-griglia di punti.

    Le coordinate nella griglia seguono sempre l'ordine (riga, colonna).

    Attributi:
        window_width (int): Numero di pixel in larghezza.
        window_height (int): Numero di pixel in altezza.
        resolution (int): Numero di punti per pixel in entrambe le direzioni.
        grid_width (int): Numero di point in larghezza.
        grid_height (int): Numero di point in altezza.
        spawn_offset (int): Offset per la zona di spawn in unità di point.
        pixel_grid (List[List[Pixel]]): Griglia 2D di oggetti Pixel indicizzati come pixel_grid[riga][colonna].
        point_grid (List[List[Point]]): Griglia 2D di oggetti Point indicizzati come point_grid[riga][colonna].
        walls (Tuple[Line, ...]): Tuple di linee che rappresentano i muri (non implementata).
    """

    window_width: int
    window_height: int
    resolution: int
    grid_width: int
    grid_height: int
    spawn_offset: int
    pixel_grid: List[List[Pixel]]
    walls: Tuple[Line, ...]

    def __init__(self,
                 window_width: int,
                 window_height: int,
                 resolution: int,
                 spawn_offset: int,
                 unexplored_point_max_steps: int,
                 render_mode: str):
        """
        Inizializza la griglia con le dimensioni, la risoluzione e altri parametri.

        Args:
            window_width (int): Numero di pixel in larghezza.
            window_height (int): Numero di pixel in altezza.
            resolution (int): Numero di punti per pixel (griglia interna).
            spawn_offset (int): Offset per la zona di spawn, in point.
            unexplored_point_max_steps (int): Valore massimo di passi per punti inesplorati.
            render_mode (str): modalità di rendering: se None non costruisce la griglia di Pixel
        """
        self.window_width = window_width
        self.window_height = window_height
        self.resolution = resolution
        self.grid_width = window_width * resolution
        self.grid_height = window_height * resolution
        self.spawn_offset = spawn_offset
        self.unexplored_point_max_steps = unexplored_point_max_steps
        self.render_mode = render_mode

        # Crea la griglia di punti: point_grid[riga][colonna]
        self.point_grid = [[Point(col, row, unexplored_point_max_steps)
                    for col in range(self.grid_width)]
                    for row in range(self.grid_height)]

        self.pixel_grid = []
        if self.render_mode == "human":
            for row in range(window_height):
                row_pixels = []
                for col in range(window_width):
                    start_row = row * self.resolution
                    end_row = (row + 1) * self.resolution
                    start_col = col * self.resolution
                    end_col = (col + 1) * self.resolution

                    points_block = [self.point_grid[r][start_col:end_col] for r in range(start_row, end_row)]
                    pixel = Pixel(col, row, points_block, unexplored_point_max_steps)
                    row_pixels.append(pixel)
                self.pixel_grid.append(row_pixels)

        # Zona di spawn: angolo in basso a sinistra, con offset (in punti) 
        self.spawn_area: Tuple[Tuple[Tuple[float, float], Tuple[float, float]], ...] = (
            ((spawn_offset, self.grid_width  - spawn_offset),
             (spawn_offset, self.grid_height - spawn_offset)),
        )
        
        self.available_area: Tuple[Tuple[Tuple[float, float], Tuple[float, float]], ...] = (
            ((0, self.grid_width),
             (0, self.grid_height)),
        )

    def reset(self):
        """
        Resetta la griglia rigenerando punti e pixel, riportando tutto allo stato iniziale.
        """
        self.point_grid = [[Point(col, row, self.unexplored_point_max_steps)
                    for col in range(self.grid_width)]
                    for row in range(self.grid_height)]

        self.pixel_grid = []
        if self.render_mode == "human":
            for row in range(self.window_height):
                row_pixels = []
                for col in range(self.window_width):
                    start_row = row * self.resolution
                    end_row = (row + 1) * self.resolution
                    start_col = col * self.resolution
                    end_col = (col + 1) * self.resolution

                    points_block = [self.point_grid[r][start_col:end_col] for r in range(start_row, end_row)]
                    pixel = Pixel(col, row, points_block, self.unexplored_point_max_steps)
                    row_pixels.append(pixel)
                self.pixel_grid.append(row_pixels)

    def get_pixel(self, pixel_row: int, pixel_col: int) -> Pixel:
        """
        Restituisce il Pixel alla posizione indicata.

        Args:
            pixel_row (int): Indice di riga del pixel.
            pixel_col (int): Indice di colonna del pixel.

        Returns:
            Pixel: Oggetto Pixel alla posizione specificata.

        Raises:
            IndexError: Se le coordinate sono fuori dai limiti della griglia.
        """
        if 0 <= pixel_row < self.window_height and 0 <= pixel_col < self.window_width:
            return self.pixel_grid[pixel_row][pixel_col]
        else:
            raise IndexError(f"Pixel coordinates out of bounds: ({pixel_row}, {pixel_col})")

    def get_point(self, point_row: int, point_col: int) -> Point:
        """
        Restituisce il Point alla posizione indicata.

        Args:
            point_row (int): Indice di riga del punto.
            point_col (int): Indice di colonna del punto.

        Returns:
            Point: Oggetto Point alla posizione specificata.

        Raises:
            IndexError: Se le coordinate sono fuori dai limiti della griglia.
        """
        if 0 <= point_row < self.grid_height and 0 <= point_col < self.grid_width:
            return self.point_grid[point_row][point_col]
        else:
            raise IndexError(f"Point coordinates out of bounds: ({point_row}, {point_col})")

    def get_point_from_coordinate(self, position: Coordinate) -> Point:
        """
        Converte una Coordinate in un Point della griglia approssimando con floor.

        Args:
            position (Coordinate): Coordinate spaziali (x, y).

        Returns:
            Point: Punto corrispondente nella griglia (riga, colonna).
        """
        point_col = math.floor(position.x_coordinate)
        point_row = math.floor(position.y_coordinate)
        return self.get_point(point_row, point_col)

    def get_pixel_from_point(self, point: Point) -> Pixel:
        """
        Restituisce il Pixel a cui appartiene un dato Point.

        Args:
            point (Point): Punto della griglia.

        Returns:
            Pixel: Pixel corrispondente.

        Raises:
            IndexError: Se le coordinate del punto sono fuori dai limiti.
        """
        if 0 <= point.point_row < self.grid_height and 0 <= point.point_col < self.grid_width:
            pixel_row = point.point_row // self.resolution
            pixel_col = point.point_col // self.resolution
            return self.pixel_grid[pixel_row][pixel_col]
        else:
            raise IndexError(f"Point coordinates out of bounds: ({point.point_row}, {point.point_col})")

    def get_pixel_from_coordinate(self, position: Coordinate) -> Pixel:
        """
        Restituisce il Pixel corrispondente a una Coordinate spaziale.

        Args:
            position (Coordinate): Coordinate spaziali (x, y).

        Returns:
            Pixel: Pixel della griglia contenente la coordinate.
        """
        return self.get_pixel_from_point(self.get_point_from_coordinate(position))

    def get_pixel_exploration_map(self) -> np.ndarray:
        """
        Costruisce una mappa numpy 2D (riga, colonna) con valori di esplorazione medi per pixel.

        Returns:
            np.ndarray: Mappa 2D (window_height x window_width) di float.
        """
        exploration_map = np.zeros((self.window_height, self.window_width), dtype=np.float32)
        for row_idx, row in enumerate(self.pixel_grid):
            for col_idx, pixel in enumerate(row):
                exploration_map[row_idx, col_idx] = pixel.mean_step_from_last_visit
        return exploration_map

    def get_point_exploration_map(self) -> np.ndarray:
        """
        Costruisce una mappa numpy 2D (riga, colonna) con valori di esplorazione per ogni punto.

        Returns:
            np.ndarray: Mappa 2D (grid_height x grid_width) di float.
        """
        exploration_map = np.zeros((self.grid_height, self.grid_width), dtype=np.float32)
        for row in range(self.grid_height):
            for col in range(self.grid_width):
                exploration_map[row, col] = self.point_grid[row][col].step_from_last_visit
        return exploration_map

                    
import math
from typing import Any
from typing import Tuple

import numpy as np
from typing import List
from gym_cruising_v2.geometry.point import Point


class Pixel:
    """
    Rappresenta un pixel nella griglia Canvas, contenente una sotto-griglia di punti.

    Attributi:
        pixel_col (int): Indice colonna del pixel.
        pixel_row (int): Indice riga del pixel.
        color (Tuple[int, int, int]): Colore RGB del pixel.
        covered (bool): Indica se almeno un punto è coperto.
        point_grid (List[List[Point]]): Griglia 2D di punti contenuti nel pixel (resolution x resolution).
        mean_step_from_last_visit (float): Valore medio di "step from last visit" dei punti.
        unexplored_point_max_steps (int): Valore massimo di passi per punti inesplorati.
    """

    pixel_col: int
    pixel_row: int
    color: Tuple[int, int, int]
    covered: bool
    point_grid: List[List[Point]]  # resolution * resolution in m^2
    mean_step_from_last_visit: float
    unexplored_point_max_steps: int

    def __init__(self, pixel_col: int, pixel_row: int, point_grid: List[List[Point]], unexplored_point_max_steps: int) -> None:
        """
        Inizializza un Pixel con posizione, punti e parametri per l'esplorazione.

        Args:
            pixel_col (int): Indice di colonna del pixel.
            pixel_row (int): Indice di riga del pixel.
            point_grid (List[List[Point]]): Sotto-griglia di punti contenuti nel pixel.
            unexplored_point_max_steps (int): Valore massimo di step per punti inesplorati.
        """
        self.pixel_col = pixel_col
        self.pixel_row = pixel_row
        self.color = (255, 255, 255)
        self.covered = False
        self.point_grid = point_grid
        self.mean_step_from_last_visit = 0
        self.unexplored_point_max_steps = unexplored_point_max_steps

    def increment_step_from_last_visit_for_points(self) -> None:
        """
        Incrementa il contatore di step from last visit per tutti i punti nel pixel
        e aggiorna il valore medio e il colore del pixel.
        """
        for row in self.point_grid:
            for point in row:
                point.increment_step_from_last_visit()
        self.calculate_mean_step_from_last_visit()

    def calculate_mean_step_from_last_visit(self) -> None:
        """
        Calcola il valore medio di step_from_last_visit dei punti nel pixel
        e aggiorna il colore del pixel di conseguenza.
        """
        step_values = np.array([point.step_from_last_visit for row in self.point_grid for point in row])
        self.mean_step_from_last_visit = step_values.mean() if step_values.size > 0 else 0
        self.define_color()

    def define_color(self) -> None:
        """
        Definisce il colore del pixel in base allo stato di esplorazione e copertura:
        - Se non coperto: da giallo (255,255,0) a rosso (255,0,0) in base a mean_step_from_last_visit.
        - Se coperto: da bianco (255,255,255) a giallo (255,255,0) in base alla percentuale di punti coperti.
        """
        total_points = sum(len(row) for row in self.point_grid)
        covered_points = self.calculate_coverage()  # Aggiorna anche self.covered

        if not self.covered:
            ratio = min(max(self.mean_step_from_last_visit / self.unexplored_point_max_steps, 0), 1)
            if ratio == 1:
                self.color = (0,0,0)
            else:
                self.color = (255, int(255 * (1 - ratio)), 0)
        else:
            ratio = min(max(covered_points / total_points, 0), 1)
            self.color = (255, 255, int(255 * ratio))

    def calculate_coverage(self) -> int:
        """
        Calcola quanti punti nella sotto-griglia sono coperti.

        Returns:
            int: Numero di punti coperti.

        Aggiorna anche l'attributo self.covered.
        """
        flat_points = [point for row in self.point_grid for point in row]
        covered_points = sum(point.covered for point in flat_points)
        self.covered = covered_points > 0
        return covered_points

    def __eq__(self, other: Any) -> bool:
        """
        Confronta questo pixel con un altro oggetto per uguaglianza.

        Args:
            other (Any): Oggetto da confrontare.

        Returns:
            bool: True se l'altro oggetto è un Point con coordinate corrispondenti (arrotondate).
        """
        return (isinstance(other, Point)
                and math.isclose(self.pixel_col, other.point_col)
                and math.isclose(self.pixel_row, other.point_y))

    def __repr__(self) -> str:
        """
        Rappresentazione testuale del pixel.

        Returns:
            str: Stringa nel formato '(col, row)'.
        """
        return f'({self.pixel_col}, {self.pixel_row})'


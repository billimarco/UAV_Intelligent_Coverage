from __future__ import annotations

import math
from typing import Any
import numpy as np


class Point:
    """
    Rappresenta un punto in un piano cartesiano discretizzato.

    Attributi:
        point_col (int): Coordinata della colonna del punto.
        point_row (int): Coordinata della riga del punto.
        covered (bool): Indica se il punto è stato coperto.
        step_from_last_visit (int): Numero di step dall'ultima visita.
        unexplored_point_max_steps (int): Valore massimo per step_from_last_visit.
    """

    point_col: int
    point_row: int
    covered: bool
    step_from_last_visit: int
    unexplored_point_max_steps: int

    def __init__(self, point_col: int, point_row: int, unexplored_point_max_steps: int) -> None:
        """
        Inizializza un punto con coordinate e valore massimo di step.

        Args:
            point_col (int): Coordinata colonna.
            point_row (int): Coordinata riga.
            unexplored_point_max_steps (int): Limite massimo per il contatore step_from_last_visit.
        """
        self.point_col = point_col
        self.point_row = point_row
        self.covered = False
        self.step_from_last_visit = unexplored_point_max_steps
        self.unexplored_point_max_steps = unexplored_point_max_steps

    def calculate_distance(self, other: Point) -> float:
        """
        Calcola la distanza euclidea tra questo punto e un altro.

        Args:
            other (Point): Altro punto.

        Returns:
            float: Distanza euclidea.
        """
        return math.sqrt((self.point_col - other.point_col) ** 2
                         + (self.point_row - other.point_row) ** 2)

    def is_in_area(self, area: np.ndarray) -> bool:
        """
        Verifica se il punto è all'interno di un'area rettangolare.

        Args:
            area (np.ndarray): Matrice 2x2 con i limiti dell'area [[col_min, col_max], [row_min, row_max]].

        Returns:
            bool: True se il punto è nell'area, False altrimenti.
        """
        if self.point_col <= area[0, 0] or self.point_col >= area[0, 1]:
            return False
        if self.point_row <= area[1, 0] or self.point_row >= area[1, 1]:
            return False
        return True
    
    def is_covered(self) -> bool:
        """
        Verifica se il punto è coperto.

        Returns:
            bool: True se il punto è coperto, False altrimenti.
        """
        return self.covered

    def increment_step_from_last_visit(self) -> None:
        """
        Incrementa step_from_last_visit fino al massimo consentito.
        """
        self.step_from_last_visit = min(self.step_from_last_visit + 1, self.unexplored_point_max_steps)

    def reset_step_from_last_visit(self) -> None:
        """
        Resetta step_from_last_visit a 0.
        """
        self.step_from_last_visit = 0

    def set_covered(self, covered: bool) -> None:
        """
        Imposta lo stato di copertura del punto.

        Args:
            covered (bool): Stato da assegnare.
        """
        self.covered = covered

    def to_array_2d(self) -> np.ndarray:
        """
        Converte il punto in un array NumPy 2D.

        Returns:
            np.ndarray: Array [col, row].
        """
        return np.array([self.point_col, self.point_row])

    def to_array_3d(self) -> np.ndarray:
        """
        Converte il punto in un array NumPy 3D, z fisso a 0.

        Returns:
            np.ndarray: Array [col, row, 0.0].
        """
        return np.array([self.point_col, self.point_row, 0.0])

    def __eq__(self, other: Any) -> bool:
        """
        Controlla l'uguaglianza tra due punti.

        Args:
            other (Any): Altro oggetto da confrontare.

        Returns:
            bool: True se è un Point con stesse coordinate.
        """
        return (isinstance(other, Point)
                and math.isclose(self.point_col, other.point_col)
                and math.isclose(self.point_row, other.point_row))

    def __repr__(self) -> str:
        """
        Rappresentazione testuale del punto.

        Returns:
            str: Stringa (col, row).
        """
        return f'({self.point_col}, {self.point_row})'

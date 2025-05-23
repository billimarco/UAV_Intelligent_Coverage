from __future__ import annotations

import math
from typing import Any
import numpy as np
from gym_cruising_v2.geometry.point import Point


class Coordinate:
    """
    Rappresenta una coordinata in uno spazio tridimensionale cartesiano.

    Attributi:
        x_coordinate (float): Coordinata lungo l'asse X.
        y_coordinate (float): Coordinata lungo l'asse Y.
        z_coordinate (float): Coordinata lungo l'asse Z.
    """

    x_coordinate: float
    y_coordinate: float
    z_coordinate: float

    def __init__(self, x_coordinate: float, y_coordinate: float, z_coordinate: float) -> None:
        """
        Inizializza una nuova coordinata tridimensionale.

        Args:
            x_coordinate (float): Valore lungo l'asse X.
            y_coordinate (float): Valore lungo l'asse Y.
            z_coordinate (float): Valore lungo l'asse Z.
        """
        self.x_coordinate = x_coordinate
        self.y_coordinate = y_coordinate
        self.z_coordinate = z_coordinate

    def calculate_distance_to_coordinate(self, other: Coordinate) -> float:
        """
        Calcola la distanza euclidea tra due coordinate tridimensionali.

        Args:
            other (Coordinate): L'altra coordinata.

        Returns:
            float: Distanza euclidea.
        """
        return math.sqrt((self.x_coordinate - other.x_coordinate) ** 2
                         + (self.y_coordinate - other.y_coordinate) ** 2
                         + (self.z_coordinate - other.z_coordinate) ** 2)

    def calculate_distance_to_point(self, point: Point) -> float:
        """
        Calcola la distanza euclidea tra una coordinata e un punto (2D).

        Args:
            point (Point): Il punto di riferimento (solo x, y).

        Returns:
            float: Distanza euclidea considerando z_coordinate come altezza.
        """
        return math.sqrt((self.x_coordinate - point.point_col) ** 2
                         + (self.y_coordinate - point.point_row) ** 2
                         + (self.z_coordinate) ** 2)

    def is_in_area(self, area: np.ndarray) -> bool:
        """
        Verifica se la coordinata è contenuta in un'area 2D.

        Args:
            area (np.ndarray): Matrice 2x2 con limiti [[x_min, x_max], [y_min, y_max]].

        Returns:
            bool: True se la coordinata è all'interno, False altrimenti.
        """
        return (area[0, 0] < self.x_coordinate < area[0, 1]
                and area[1, 0] < self.y_coordinate < area[1, 1])

    def is_in_volume(self, volume: np.ndarray) -> bool:
        """
        Verifica se la coordinata è contenuta in un volume 3D.

        Args:
            volume (np.ndarray): Matrice 3x2 con limiti [[x_min, x_max], [y_min, y_max], [z_min, z_max]].

        Returns:
            bool: True se la coordinata è all'interno, False altrimenti.
        """
        return (volume[0, 0] < self.x_coordinate < volume[0, 1]
                and volume[1, 0] < self.y_coordinate < volume[1, 1]
                and volume[2, 0] < self.z_coordinate < volume[2, 1])

    def to_array(self) -> np.ndarray:
        """
        Converte la coordinata in un array NumPy.

        Returns:
            np.ndarray: Array [x, y, z].
        """
        return np.array([self.x_coordinate, self.y_coordinate, self.z_coordinate])

    def __eq__(self, other: Any) -> bool:
        """
        Verifica se due coordinate sono uguali entro una tolleranza.

        Args:
            other (Any): Oggetto da confrontare.

        Returns:
            bool: True se è una Coordinate con stessi valori.
        """
        return (isinstance(other, Coordinate)
                and math.isclose(self.x_coordinate, other.x_coordinate)
                and math.isclose(self.y_coordinate, other.y_coordinate)
                and math.isclose(self.z_coordinate, other.z_coordinate))

    def __repr__(self) -> str:
        """
        Rappresentazione testuale della coordinata.

        Returns:
            str: Stringa (x, y, z).
        """
        return f'({self.x_coordinate}, {self.y_coordinate}, {self.z_coordinate})'
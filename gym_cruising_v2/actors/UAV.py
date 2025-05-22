from gym_cruising_v2.geometry.coordinate import Coordinate


class UAV:
    position: Coordinate
    last_shift_x: float
    last_shift_y: float

    def __init__(self, position: Coordinate) -> None:
        self.position = position
        self.last_shift_x = 0.0
        self.last_shift_y = 0.0

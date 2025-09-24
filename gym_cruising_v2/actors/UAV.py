from gym_cruising_v2.geometry.coordinate import Coordinate


class UAV:
    id: int
    position: Coordinate
    last_shift_x: float
    last_shift_y: float
    total_covered_points: int
    shared_covered_points: int
    active: bool

    def __init__(self, id: int, position: Coordinate, active: bool) -> None:
        self.id = id
        self.position = position
        self.last_shift_x = 0.0
        self.last_shift_y = 0.0
        self.total_covered_points = 0
        self.shared_covered_points = 0
        self.active = active
from gym_cruising_v2.geometry.point import Point


class UAV:
    position: Point
    previous_position: Point
    last_shift_x: float
    last_shift_y: float

    def __init__(self, position: Point) -> None:
        self.position = position
        self.previous_position = position
        self.last_shift_x = 0.0
        self.last_shift_y = 0.0

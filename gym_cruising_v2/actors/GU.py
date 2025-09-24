from gym_cruising_v2.geometry.coordinate import Coordinate

class GU:
    id: int
    position: Coordinate
    last_shift_x: float
    last_shift_y: float
    covered: bool
    channels_state = []
    active: bool

    def __init__(self, id: int, position: Coordinate, active: bool) -> None:
        self.id = id
        self.position = position
        self.last_shift_x = 0.0
        self.last_shift_y = 0.0
        self.covered = False
        self.channels_state = []
        self.active = active

    def reset(self, position: Coordinate, active: bool) -> None:
        self.position = position
        self.last_shift_x = 0.0
        self.last_shift_y = 0.0
        self.covered = False
        self.channels_state = []
        self.active = active
        
    def get_image(self):
        if self.covered:
            return 'green30.png'
        return 'white30.png'


    def set_covered(self, covered: bool):
        self.covered = covered

    def set_channels_state(self, channels_state):
        self.channels_state = channels_state

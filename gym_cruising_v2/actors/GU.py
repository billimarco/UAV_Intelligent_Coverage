from gym_cruising_v2.geometry.point import Point

class GU:
    position: Point
    previous_position: Point
    covered: bool
    channels_state = []

    def __init__(self, position: Point) -> None:
        self.position = position
        self.previous_position = position
        self.covered = False
        self.channels_state = []

    def getImage(self):
        if self.covered:
            return './gym_cruising_v2_v2/images/green30.png'
        return './gym_cruising_v2_v2/images/white30.png'

    def setCovered(self, covered: bool):
        self.covered = covered

    def setChannelsState(self, channels_state):
        self.channels_state = channels_state

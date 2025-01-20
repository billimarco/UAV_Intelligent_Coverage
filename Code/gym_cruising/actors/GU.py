from gym_cruising.enums.color import Color
from gym_cruising.geometry.point import Point


class GU:
    position: Point
    covered: bool
    channels_state = []
    previous_position: Point

    def __init__(self, position: Point) -> None:
        self.position = position
        self.covered = False
        self.channels_state = []
        self.previous_position = position

    def getColor(self):
        if self.covered:
            return Color.GREEN.value
        return Color.RED.value

    def getImage(self):
        if self.covered:
            return './gym_cruising/images/green30.png'
        return './gym_cruising/images/white30.png'

    def setCovered(self, covered: bool):
        self.covered = covered

    def setChannelsState(self, channels_state: [int]):
        self.channels_state = channels_state

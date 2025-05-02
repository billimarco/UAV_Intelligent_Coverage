from gym_cruising.enums.color import Color
from gym_cruising.geometry.point import Point


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

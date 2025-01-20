import numpy as np

from gym_cruising.geometry.point import Point
import math

UAV_ALTITUDE = 500
a = 12.08  # in the dense urban case
b = 0.11  # in the dense urban case
nNLos = 23  # [dB] in the dense urban case
nLos = 1.6  # [dB] in the dense urban case
RATE_OF_GROWTH = -0.05
TRASMISSION_POWER = 23  # 30 dBm
CHANNEL_BANDWIDTH = 2e6  # 2 MHz
POWER_SPECTRAL_DENSITY_OF_NOISE = -174  # -174 dBm/Hz


# calculate distance between one UAV and one GU in air line
def calculate_distance_uav_gu(uav: Point, gu: Point):
    return math.sqrt(math.pow(uav.x_coordinate - gu.x_coordinate, 2) +
                     math.pow(uav.y_coordinate - gu.y_coordinate, 2) +
                     UAV_ALTITUDE ** 2)


# calculate the Probability of LoS link between one UAV and one GU
def get_PLoS(distance_uav_gu: float):
    elevation_angle = math.degrees(math.asin(UAV_ALTITUDE / distance_uav_gu))
    return 1 / (1 + a * math.exp((-1) * b * (elevation_angle - a)))

def get_transition_matrix(relative_shift: float, PLoS: float):
    PLoS2NLoS = 2 * ((1 - PLoS) / (1 + math.exp(RATE_OF_GROWTH * relative_shift)) - (1 - PLoS) / 2)  # g1
    PNLoS2LoS = 2 * (PLoS / (1 + math.exp(RATE_OF_GROWTH * relative_shift)) - PLoS / 2)  # g2
    return np.array([
        [1 - PLoS2NLoS, PLoS2NLoS],
        [PNLoS2LoS, 1 - PNLoS2LoS]
    ])


# calculate the Free Space PathLoss of the link between one UAV and one GU in dB
# 38.4684 is according to Friis equation with carrier frequency fc = 2GHz
def get_free_space_PathLoss(distance_uav_gu: float):
    return 20 * math.log(distance_uav_gu, 10) + 38.4684


# calculate PathLoss of the link between one UAV and one GU in dB
def get_PathLoss(distance_uav_gu: float, current_state: int):
    FSPL = get_free_space_PathLoss(distance_uav_gu)
    if current_state == 0:
        return FSPL + nLos
    return FSPL + nNLos

# def getSINR(path_loss: float, interference_path_loss: []):
#     return W2dB((dBm2Watt(TRASMISSION_POWER) * getChannelGain(path_loss)) / (
#                 getInterference(interference_path_loss) + dBm2Watt(
#             POWER_SPECTRAL_DENSITY_OF_NOISE) * CHANNEL_BANDWIDTH))


def getSINR(path_loss: float, interference_path_loss: []):
    return W2dB((dBm2Watt(TRASMISSION_POWER) * getChannelGain(path_loss)) / (dBm2Watt(
            POWER_SPECTRAL_DENSITY_OF_NOISE) * CHANNEL_BANDWIDTH))


def getChannelGain(path_loss: float) -> float:
    return 1 / dB2Linear(path_loss)


def getInterference(interference_path_loss: []) -> float:
    interference = 0.0
    for path_loss in interference_path_loss:
        interference += dBm2Watt(TRASMISSION_POWER) * getChannelGain(path_loss)
    return interference


def dB2Linear(decibel_value: float):
    return 10 ** (decibel_value / 10)


def dBm2Watt(dBm_value: float):
    return math.pow(10, (dBm_value - 30) / 10)


def W2dB(watt_value: float):
    return math.log(watt_value, 10) * 10

""" This __init__ file is used to register the environments """
from gymnasium.envs.registration import register

register(
    id='Cruising-v0',
    entry_point='gym_cruising_v2.envs.cruise_uav_map:CruiseUAVWithMap')

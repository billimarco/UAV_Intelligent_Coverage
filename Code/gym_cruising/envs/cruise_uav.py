""" This module contains the Cruising environment class """
import random
from typing import Optional, Tuple

import numpy as np
import pygame
from gymnasium.spaces import Box
from pygame import Surface

from gym_cruising.actors.GU import GU
from gym_cruising.actors.UAV import UAV
from gym_cruising.enums.color import Color
from gym_cruising.envs.cruise import Cruise
from gym_cruising.geometry.point import Point
from gym_cruising.utils import channels_utils

MAX_SPEED_UAV = 55.6  # m/s - about 20 Km/h x 10 steps
MAX_POSITION = 4000.0


def normalizePositions(positions: np.ndarray) -> np.ndarray:  # Normalize in [-1,1]
    nornmalized_positions = np.ndarray(shape=positions.shape, dtype=np.float64)
    nornmalized_positions = (positions / MAX_POSITION) * 2 - 1
    return nornmalized_positions


def normalizeActions(actions: np.ndarray) -> np.ndarray:  # Normalize in [-1,1]
    nornmalized_actions = np.ndarray(shape=actions.shape, dtype=np.float64)
    nornmalized_actions = ((actions + MAX_SPEED_UAV) / (2 * MAX_SPEED_UAV)) * 2 - 1
    return nornmalized_actions


class CruiseUAV(Cruise):
    uav = []
    gu = []
    pathLoss = []
    SINR = []
    connectivity_matrix = []

    UAV_NUMBER = 2
    STARTING_GU_NUMBER = 60
    gu_number: int
    MINIMUM_STARTING_DISTANCE_BETWEEN_UAV = 200  # meters
    COLLISION_DISTANCE = 10  # meters

    SPAWN_GU_PROB = 0.0005
    disappear_gu_prob: float

    GU_MEAN_SPEED = 5.56  # 5.56 m/s or 27.7 m/s
    GU_STANDARD_DEVIATION = 1.97  # Gaussian goes to 0 at approximately 3 times the standard deviation
    MAX_SPEED_UAV = 55.6  # m/s - about 20 Km/h x 10 steps

    COVERED_TRESHOLD = 10.0  # dB

    low_observation: float
    high_observation: float

    gu_covered = 0
    last_RCR = None
    reward_gamma = 0.7

    def __init__(self,
                 render_mode=None, track_id: int = 1) -> None:
        super().__init__(render_mode, track_id)

        spawn_area = self.np_random.choice(self.track.spawn_area)
        self.low_observation = float(spawn_area[0][0] - self.MAX_SPEED_UAV)
        self.high_observation = float(spawn_area[0][1] + self.MAX_SPEED_UAV)

        self.reset_observation_action_space()

    def reset_observation_action_space(self):
        self.observation_space = Box(low=self.low_observation,
                                     high=self.high_observation,
                                     shape=((self.UAV_NUMBER * 2) + self.gu_covered, 2),
                                     dtype=np.float64)

        self.action_space = Box(low=(-1) * self.MAX_SPEED_UAV,
                                high=self.MAX_SPEED_UAV,
                                shape=(self.UAV_NUMBER, 2),
                                dtype=np.float64)

    def reset(self, seed=None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        self.uav = []
        self.gu = []
        self.UAV_NUMBER = options["uav"]
        self.STARTING_GU_NUMBER = options["gu"]
        self.reset_observation_action_space()
        self.gu_number = self.STARTING_GU_NUMBER
        self.disappear_gu_prob = self.SPAWN_GU_PROB * 4 / self.gu_number
        self.gu_covered = 0
        self.last_RCR = None
        return super().reset(seed=seed, options=options)

    def perform_action(self, actions) -> None:
        self.move_UAV(actions)
        self.update_GU()
        self.calculate_PathLoss_with_Markov_Chain()
        self.calculate_SINR()
        self.check_connection_and_coverage_UAV_GU()

    def update_GU(self):
        self.move_GU()
        self.check_if_disappear_GU()
        self.check_if_spawn_new_GU()

    def move_UAV(self, actions):
        for i, uav in enumerate(self.uav):
            previous_position = uav.position
            new_position = Point(previous_position.x_coordinate + actions[i][0],
                                 previous_position.y_coordinate + actions[i][1])
            uav.position = new_position
            uav.previous_position = previous_position
            uav.last_shift_x = actions[i][0]
            uav.last_shift_y = actions[i][1]

    # Random walk the GU
    def move_GU(self):
        area = self.np_random.choice(self.track.spawn_area)
        for gu in self.gu:
            repeat = True
            while repeat:
                previous_position = gu.position
                distance = self.np_random.normal(self.GU_MEAN_SPEED, self.GU_STANDARD_DEVIATION)
                if distance < 0.0:
                    distance = 0.0
                direction = np.random.choice(['up', 'down', 'left', 'right'])

                if direction == 'up':
                    new_position = Point(previous_position.x_coordinate, previous_position.y_coordinate + distance)
                elif direction == 'down':
                    new_position = Point(previous_position.x_coordinate, previous_position.y_coordinate - distance)
                elif direction == 'left':
                    new_position = Point(previous_position.x_coordinate - distance, previous_position.y_coordinate)
                elif direction == 'right':
                    new_position = Point(previous_position.x_coordinate + distance, previous_position.y_coordinate)

                # check if GU exit from environment
                if new_position.is_in_area(area):
                    repeat = False
                    gu.position = new_position
                    gu.previous_position = previous_position
                else:
                    repeat = True

    def calculate_PathLoss_with_Markov_Chain(self):
        self.pathLoss = []
        for gu in self.gu:
            current_GU_PathLoss = []
            new_channels_state = []
            gu_shift = gu.position.calculate_distance(gu.previous_position)
            for index, uav in enumerate(self.uav):
                distance = channels_utils.calculate_distance_uav_gu(uav.position, gu.position)
                channel_PLoS = channels_utils.get_PLoS(distance)
                relative_shift = uav.position.calculate_distance(uav.previous_position) + gu_shift
                transition_matrix = channels_utils.get_transition_matrix(relative_shift, channel_PLoS)
                current_state = np.random.choice(range(len(transition_matrix)),
                                                 p=transition_matrix[gu.channels_state[index]])
                new_channels_state.append(current_state)
                path_loss = channels_utils.get_PathLoss(distance, current_state)
                current_GU_PathLoss.append(path_loss)
            self.pathLoss.append(current_GU_PathLoss)
            gu.setChannelsState(new_channels_state)

    def calculate_SINR(self):
        self.SINR = []
        for i in range(len(self.gu)):
            current_GU_SINR = []
            current_pathLoss = self.pathLoss[i]
            for j in range(len(self.uav)):
                copy_list = current_pathLoss.copy()
                del copy_list[j]
                current_GU_SINR.append(channels_utils.getSINR(current_pathLoss[j], copy_list))
            self.SINR.append(current_GU_SINR)

    def check_if_disappear_GU(self):
        disappeared_GU = 0
        index_to_remove = []
        for i in range(self.gu_number):
            sample = random.random()
            if sample <= self.disappear_gu_prob:
                index_to_remove.append(i)
                disappeared_GU += 1
        index_to_remove = sorted(index_to_remove, reverse=True)
        for index in index_to_remove:
            del self.gu[index]
        self.gu_number -= disappeared_GU

    def check_if_spawn_new_GU(self):
        sample = random.random()
        for _ in range(4):
            if sample <= self.SPAWN_GU_PROB:
                area = self.np_random.choice(self.track.spawn_area)
                x_coordinate = self.np_random.uniform(area[0][0], area[0][1])
                y_coordinate = self.np_random.uniform(area[1][0], area[1][1])
                gu = GU(Point(x_coordinate, y_coordinate))
                self.initialize_channel(gu)
                self.gu.append(gu)
                self.gu_number += 1
        # update disappear gu probability
        self.disappear_gu_prob = self.SPAWN_GU_PROB * 4 / self.gu_number

    def check_connection_and_coverage_UAV_GU(self):
        covered = 0
        self.connectivity_matrix = np.zeros((self.gu_number, self.UAV_NUMBER), dtype=int)
        for i, gu in enumerate(self.gu):
            gu.setCovered(False)
            current_SINR = self.SINR[i]
            for j in range(len(self.uav)):
                if current_SINR[j] >= self.COVERED_TRESHOLD:
                    self.connectivity_matrix[i, j] = 1
            if any(SINR >= self.COVERED_TRESHOLD for SINR in current_SINR):
                gu.setCovered(True)
                covered += 1
        self.gu_covered = covered

    def get_observation(self) -> np.ndarray:
        self.observation_space = Box(low=self.low_observation,
                                     high=self.high_observation,
                                     shape=((self.UAV_NUMBER * 2) + self.gu_covered, 2),
                                     dtype=np.float64)
        observation = [
            normalizePositions(np.array([self.uav[0].position.x_coordinate, self.uav[0].position.y_coordinate]))]
        observation = np.append(observation,
                                [normalizeActions(np.array([self.uav[0].last_shift_x, self.uav[0].last_shift_y]))],
                                axis=0)
        for i in range(1, self.UAV_NUMBER):
            observation = np.append(observation,
                                    [normalizePositions(np.array(
                                        [self.uav[i].position.x_coordinate, self.uav[i].position.y_coordinate]))],
                                    axis=0)
            observation = np.append(observation,
                                    [normalizeActions(np.array([self.uav[i].last_shift_x, self.uav[i].last_shift_y]))],
                                    axis=0)

        for gu in self.gu:
            if gu.covered:
                observation = np.append(observation,
                                        [normalizePositions(
                                            np.array([gu.position.x_coordinate, gu.position.y_coordinate]))],
                                        axis=0)
        return observation

    def check_if_terminated(self):
        terminated_matrix = []
        area = self.np_random.choice(self.track.spawn_area)
        for i, uav in enumerate(self.uav):
            if not uav.position.is_in_area(area) or self.collision(i, uav):
                terminated_matrix.append(True)
            else:
                terminated_matrix.append(False)
        return terminated_matrix

    def collision(self, current_uav_index, uav) -> bool:
        collision = False
        for j, other_uav in enumerate(self.uav):
            if j != current_uav_index:
                if uav.position.calculate_distance(other_uav.position) <= self.COLLISION_DISTANCE:
                    collision = True
                    break
        return collision

    def check_if_truncated(self) -> bool:
        return False

    def RCR_without_uav_i(self, i):
        tmp_matrix = np.delete(self.connectivity_matrix, i, axis=1)  # Remove i-th column
        return np.sum(np.any(tmp_matrix, axis=1))

    def calculate_reward(self, terminated):
        current_rewards = []
        for i in range(len(self.uav)):
            if terminated[i]:
                current_rewards.append(-2.0)
            else:
                current_rewards.append((self.gu_covered - self.RCR_without_uav_i(i)) / len(self.gu))
        if self.last_RCR is None:
            self.last_RCR = current_rewards
            return [r * 100.0 for r in current_rewards]
        delta_RCR_smorzato = []
        for i in range(len(self.uav)):
            if not terminated[i]:
                delta_RCR_smorzato.append(self.reward_gamma * (current_rewards[i] - self.last_RCR[i]))
            else:
                delta_RCR_smorzato.append(0.0)
        self.last_RCR = current_rewards
        reward_smorzato = np.add(current_rewards, delta_RCR_smorzato)
        return [r * 100.0 for r in reward_smorzato]

    def init_environment(self, options: Optional[dict] = None) -> None:
        self.init_uav()
        if options['clustered'] == 0:
            self.init_gu()
        else:
            self.init_gu_clustered(options)
        self.calculate_PathLoss_with_Markov_Chain()
        self.calculate_SINR()
        self.check_connection_and_coverage_UAV_GU()

    def init_uav(self) -> None:
        area = self.np_random.choice(self.track.spawn_area)
        # x_coordinate = self.np_random.uniform(area[0][0] + 800, area[0][1] - 800)
        x_coordinate = self.np_random.uniform(area[0][0], area[0][1])
        y_coordinate = self.np_random.uniform(area[1][0], area[1][1])
        self.uav.append(UAV(Point(x_coordinate, y_coordinate)))
        for i in range(1, self.UAV_NUMBER):
            x_coordinate = self.np_random.uniform(area[0][0], area[0][1])
            y_coordinate = self.np_random.uniform(area[1][0], area[1][1])
            position = Point(x_coordinate, y_coordinate)
            while self.are_too_close(i, position):
                x_coordinate = self.np_random.uniform(area[0][0], area[0][1])
                y_coordinate = self.np_random.uniform(area[1][0], area[1][1])
                position = Point(x_coordinate, y_coordinate)
            self.uav.append(UAV(position))

    def are_too_close(self, uav_index, position):
        too_close = False
        for j in range(uav_index):
            if self.uav[j].position.calculate_distance(position) <= self.MINIMUM_STARTING_DISTANCE_BETWEEN_UAV:
                too_close = True
                break
        return too_close

    def init_gu(self) -> None:
        area = self.np_random.choice(self.track.spawn_area)
        for _ in range(self.gu_number):
            x_coordinate = self.np_random.uniform(area[0][0], area[0][1])
            y_coordinate = self.np_random.uniform(area[1][0], area[1][1])
            gu = GU(Point(x_coordinate, y_coordinate))
            self.initialize_channel(gu)
            self.gu.append(gu)

    def init_gu_clustered(self, options: Optional[dict] = None) -> None:
        area = self.np_random.choice(self.track.spawn_area)
        std_dev = np.sqrt(options['variance'])
        number_of_clusters = options['clusters_number']
        gu_for_cluster = int(self.STARTING_GU_NUMBER / number_of_clusters)
        for i in range(number_of_clusters):
            mean_x = self.np_random.uniform(area[0][0] + 250, area[0][1] - 250)
            mean_y = self.np_random.uniform(area[0][0] + 250, area[0][1] - 250)
            for j in range(gu_for_cluster):
                repeat = True
                while repeat:
                    # Generazione del numero casuale
                    x_coordinate = np.random.normal(mean_x, std_dev)
                    y_coordinate = np.random.normal(mean_y, std_dev)
                    position = Point(x_coordinate, y_coordinate)
                    if position.is_in_area(area):
                        repeat = False
                gu = GU(position)
                self.initialize_channel(gu)
                self.gu.append(gu)

    def initialize_channel(self, gu):
        for uav in self.uav:
            distance = channels_utils.calculate_distance_uav_gu(uav.position, gu.position)
            initial_channel_PLoS = channels_utils.get_PLoS(distance)
            sample = random.random()
            if sample <= initial_channel_PLoS:
                gu.channels_state.append(0)  # 0 = LoS
            else:
                gu.channels_state.append(1)  # 1 = NLoS

    def draw(self, canvas: Surface) -> None:
        # CANVAS
        canvas.fill(Color.WHITE.value)

        # WALL
        for wall in self.world:
            pygame.draw.line(canvas,
                             Color.BLACK.value,
                             self.convert_point(wall.start),
                             self.convert_point(wall.end),
                             self.WIDTH)

        # GU image
        for gu in self.gu:
            canvas.blit(pygame.image.load(gu.getImage()), self.image_convert_point(gu.position))

        # UAV image
        icon_drone = pygame.image.load('./gym_cruising/images/drone30.png')
        for uav in self.uav:
            canvas.blit(icon_drone, self.image_convert_point(uav.position))

    def convert_point(self, point: Point) -> Tuple[int, int]:
        pygame_x = (round(point.x_coordinate * self.RESOLUTION)
                    + self.X_OFFSET)
        pygame_y = (self.window_size
                    - round(point.y_coordinate * self.RESOLUTION)
                    + self.Y_OFFSET)
        return pygame_x, pygame_y

    def image_convert_point(self, point: Point) -> Tuple[int, int]:
        shiftX = 15
        shiftY = 15
        pygame_x = (round(point.x_coordinate * self.RESOLUTION) - shiftX + self.X_OFFSET)
        pygame_y = (self.window_size - round(point.y_coordinate * self.RESOLUTION) - shiftY + self.Y_OFFSET)
        return pygame_x, pygame_y

    def create_info(self, terminated) -> dict:
        if sum(terminated) >= 2:
            RCR = str(0.0)
        else:
            RCR = str(self.gu_covered/self.gu_number)
        return {"GU coperti": str(self.gu_covered), "Ground Users": str(
            self.gu_number), "RCR": RCR, "terminated": sum(terminated)}

""" This module contains the Cruising environment class """
import random
from typing import Optional, Tuple
import math

import numpy as np
import pygame
from gymnasium.spaces import Box
from pygame import Surface

from gym_cruising.actors.GU import GU
from gym_cruising.actors.UAV import UAV
from gym_cruising.enums.color import Color
from gym_cruising.envs.cruise import Cruise
from gym_cruising.geometry.point import Point
from gym_cruising.utils.channels_utils import CommunicationChannel

class CruiseUAV(Cruise):
    uav = []
    gu = []
    pathLoss = []
    SINR = []
    connectivity_matrix = []
    disappear_gu_prob: float
    
    low_observation: float
    high_observation: float

    gu_covered = 0
    last_RCR = None
    reward_gamma = 0.7

    def __init__(self, args, render_mode=None) -> None:
        super().__init__(args, render_mode)
        
        self.uav_number = args.uav_number
        self.starting_gu_number = args.starting_gu_number
        self.minimum_starting_distance_between_uav = args.minimum_starting_distance_between_uav # meters
        self.collision_distance = args.collision_distance # meters
        self.spawn_gu_prob = args.spawn_gu_prob
        self.gu_mean_speed = args.gu_mean_speed # 5.56 m/s or 27.7 m/s
        self.gu_standard_deviation = args.gu_standard_deviation # Gaussian goes to 0 at approximately 3 times the standard deviation
        self.max_speed_uav = args.max_speed_uav # m/s - about 20 Km/h x 10 steps
        self.covered_threshold = args.covered_threshold # dB
        self.uav_altitude = args.uav_altitude # meters
        
        self.communication_channel = CommunicationChannel(args)
        

        self.reset_observation_action_space()

    def reset_observation_action_space(self):
        spawn_area = self.np_random.choice(self.grid.spawn_area)
        (x_min, x_max), (y_min, y_max) = spawn_area

        min_x = x_min - self.max_speed_uav
        max_x = x_max + self.max_speed_uav
        min_y = y_min - self.max_speed_uav
        max_y = y_max + self.max_speed_uav
        
        low = np.array([min_x, min_y], dtype=np.float64)
        high = np.array([max_x, max_y], dtype=np.float64)
        
        obs_shape = ((self.uav_number * 2) + self.gu_covered, 2)
        self.low_observation = np.tile(low, (obs_shape[0], 1))
        self.high_observation = np.tile(high, (obs_shape[0], 1))
        self.observation_space = Box(low=self.low_observation,
                                     high=self.high_observation,
                                     shape=((self.uav_number * 2) + self.gu_covered, 2),
                                     dtype=np.float64)

        self.action_space = Box(low=(-1) * self.max_speed_uav,
                                high=self.max_speed_uav,
                                shape=(self.uav_number, 2),
                                dtype=np.float64)

    def reset(self, seed=None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        self.uav = []
        self.gu = []
        self.uav_number = options["uav"]
        self.starting_gu_number = options["gu"]
        self.reset_observation_action_space()
        self.disappear_gu_prob = self.spawn_gu_prob * 4 / self.starting_gu_number
        self.gu_covered = 0
        self.last_RCR = None
        np.random.seed(seed)
        return super().reset(seed=seed, options=options)
    
    def normalizePositions(self, position) -> np.ndarray:  # Normalize in [-1,1]
            max_x = self.window_width / self.resolution
            max_y = self.window_height / self.resolution

            norm_x = (position.x_coordinate / max_x) * 2 - 1
            norm_y = (position.y_coordinate / max_y) * 2 - 1

            return np.array([norm_x, norm_y], dtype=np.float64)
    
    def normalizeActions(self, actions: np.ndarray) -> np.ndarray:  # Normalize in [-1,1]
        normalized_actions = np.ndarray(shape=actions.shape, dtype=np.float64)
        normalized_actions = ((actions + self.max_speed_uav) / (2 * self.max_speed_uav)) * 2 - 1
        return normalized_actions
        
    def perform_action(self, actions) -> None:
        self.move_UAV(actions)
        self.update_GU()
        self.calculate_PathLoss_with_Markov_Chain()
        self.calculate_SINR()
        self.check_connection_and_coverage_UAV_GU()

    def update_GU(self):
        self.check_if_spawn_new_GU()
        self.move_GU()
        self.check_if_disappear_GU()

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
        area = self.np_random.choice(self.grid.spawn_area)
        for gu in self.gu:
            repeat = True
            while repeat:
                previous_position = gu.position
                distance = self.np_random.normal(self.gu_mean_speed, self.gu_standard_deviation)
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

    # calculate distance between one UAV and one GU in air line
    def calculate_distance_uav_gu(self, uav: Point, gu: Point):
        return math.sqrt(math.pow(uav.x_coordinate - gu.x_coordinate, 2) +
                        math.pow(uav.y_coordinate - gu.y_coordinate, 2) +
                        self.uav_altitude ** 2)
        
    def calculate_PathLoss_with_Markov_Chain(self):
        self.pathLoss = []
        for gu in self.gu:
            current_GU_PathLoss = []
            new_channels_state = []
            gu_shift = gu.position.calculate_distance(gu.previous_position)
            for index, uav in enumerate(self.uav):
                distance = self.calculate_distance_uav_gu(uav.position, gu.position)
                channel_PLoS = self.communication_channel.get_PLoS(distance, self.uav_altitude)
                relative_shift = uav.position.calculate_distance(uav.previous_position) + gu_shift
                transition_matrix = self.communication_channel.get_transition_matrix(relative_shift, channel_PLoS)
                current_state = np.random.choice(range(len(transition_matrix)),
                                                 p=transition_matrix[gu.channels_state[index]])
                new_channels_state.append(current_state)
                path_loss = self.communication_channel.get_PathLoss(distance, current_state)
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
                current_GU_SINR.append(self.communication_channel.getSINR(current_pathLoss[j], copy_list))
            self.SINR.append(current_GU_SINR)

    def check_if_disappear_GU(self):
        index_to_remove = []
        for i in range(len(self.gu)):
            sample = random.random()
            if sample <= self.disappear_gu_prob:
                index_to_remove.append(i)
        index_to_remove = sorted(index_to_remove, reverse=True)
        '''
        if index_to_remove != []:
            print("GU disappeared: ", index_to_remove)
            print("self.gu number: ", len(self.gu))
        '''
        for index in index_to_remove:
            del self.gu[index]

    def check_if_spawn_new_GU(self):
        sample = random.random()
        for _ in range(4):
            if sample <= self.spawn_gu_prob:
                area = self.np_random.choice(self.grid.spawn_area)
                x_coordinate = self.np_random.uniform(area[0][0], area[0][1])
                y_coordinate = self.np_random.uniform(area[1][0], area[1][1])
                gu = GU(Point(x_coordinate, y_coordinate))
                self.initialize_channel(gu)
                self.gu.append(gu)
        # update disappear gu probability
        self.disappear_gu_prob = self.spawn_gu_prob * 4 / len(self.gu)

    def check_connection_and_coverage_UAV_GU(self):
        covered = 0
        self.connectivity_matrix = np.zeros((len(self.gu), self.uav_number), dtype=int)
        for i, gu in enumerate(self.gu):
            gu.setCovered(False)
            current_SINR = self.SINR[i]
            for j in range(len(self.uav)):
                if current_SINR[j] >= self.covered_threshold:
                    self.connectivity_matrix[i, j] = 1
            if any(SINR >= self.covered_threshold for SINR in current_SINR):
                gu.setCovered(True)
                covered += 1
        self.gu_covered = covered

    def get_observation(self) -> np.ndarray:
        observation = [self.normalizePositions(self.uav[0].position)]
        
        observation = np.append(observation, [self.normalizeActions(np.array([self.uav[0].last_shift_x, self.uav[0].last_shift_y]))], axis=0)
        
        for i in range(1, self.uav_number):
            observation = np.append(observation, [self.normalizePositions(self.uav[i].position)], axis=0)
            observation = np.append(observation, [self.normalizeActions(np.array([self.uav[i].last_shift_x, self.uav[i].last_shift_y]))], axis=0)

        for gu in self.gu:
            if gu.covered:
                observation = np.append(observation, [self.normalizePositions(gu.position)], axis=0)
        return observation

    def check_if_terminated(self):
        terminated_matrix = []
        area = self.np_random.choice(self.grid.spawn_area)
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
                if uav.position.calculate_distance(other_uav.position) <= self.collision_distance:
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
        if not options["clustered"]:
            self.init_gu()
        else:
            self.init_gu_clustered(options)
        self.calculate_PathLoss_with_Markov_Chain()
        self.calculate_SINR()
        self.check_connection_and_coverage_UAV_GU()

    def init_uav(self) -> None:
        area = self.np_random.choice(self.grid.spawn_area)
        # x_coordinate = self.np_random.uniform(area[0][0] + 800, area[0][1] - 800)
        x_coordinate = self.np_random.uniform(area[0][0], area[0][1])
        y_coordinate = self.np_random.uniform(area[1][0], area[1][1])
        self.uav.append(UAV(Point(x_coordinate, y_coordinate)))
        for i in range(1, self.uav_number):
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
            if self.uav[j].position.calculate_distance(position) <= self.minimum_starting_distance_between_uav:
                too_close = True
                break
        return too_close

    def init_gu(self) -> None:
        area = self.np_random.choice(self.grid.spawn_area)
        for _ in range(self.starting_gu_number):
            x_coordinate = self.np_random.uniform(area[0][0], area[0][1])
            y_coordinate = self.np_random.uniform(area[1][0], area[1][1])
            gu = GU(Point(x_coordinate, y_coordinate))
            self.initialize_channel(gu)
            self.gu.append(gu)

    def reset_gu(self, options: Optional[dict] = None) -> np.ndarray:
        self.gu = []
        self.disappear_gu_prob = self.spawn_gu_prob * 4 / self.starting_gu_number
        self.gu_covered = 0
        self.last_RCR = None
        if not options['clustered']:
            self.init_gu()
        else:
            self.init_gu_clustered(options)
        self.calculate_PathLoss_with_Markov_Chain()
        self.calculate_SINR()
        self.check_connection_and_coverage_UAV_GU()
        return self.get_observation()

    def init_gu_clustered(self, options: Optional[dict] = None) -> None:
        area = self.np_random.choice(self.grid.spawn_area)
        std_dev = np.sqrt(options['variance'])
        number_of_clusters = options['clusters_number']
        gu_for_cluster = int(self.starting_gu_number / number_of_clusters)
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
            distance = self.calculate_distance_uav_gu(uav.position, gu.position)
            initial_channel_PLoS = self.communication_channel.get_PLoS(distance, self.uav_altitude)
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
                             self.wall_width)

        # GU image
        for gu in self.gu:
            canvas.blit(pygame.image.load(gu.getImage()), self.image_convert_point(gu.position))

        # UAV image
        icon_drone = pygame.image.load('./gym_cruising/images/drone30.png')
        for uav in self.uav:
            canvas.blit(icon_drone, self.image_convert_point(uav.position))

    def convert_point(self, point: Point) -> Tuple[int, int]:
        pygame_x = (round(point.x_coordinate * self.resolution)
                    + self.x_offset)
        pygame_y = (self.window_height
                    - round(point.y_coordinate * self.resolution)
                    + self.y_offset)
        return pygame_x, pygame_y

    def image_convert_point(self, point: Point) -> Tuple[int, int]:
        shiftX = 15
        shiftY = 15
        pygame_x = (round(point.x_coordinate * self.resolution) - shiftX + self.x_offset)
        pygame_y = (self.window_height - round(point.y_coordinate * self.resolution) - shiftY + self.y_offset)
        return pygame_x, pygame_y

    def create_info(self, terminated) -> dict:
        collision = False
        for i, uav in enumerate(self.uav):
            if self.collision(i, uav):
                collision = True
                break
        if collision:
            RCR = str(0.0)
        else:
            RCR = str(self.gu_covered/len(self.gu))
        return {"GU coperti": str(self.gu_covered), "Ground Users": str(
            len(self.gu)), "RCR": RCR, "Collision": collision}

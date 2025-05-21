""" This module contains the Cruising environment class """
import random
from typing import Optional, Tuple
import math

import numpy as np
import pygame
from gymnasium import spaces
from gymnasium.spaces import Box
from pygame import Surface

from gym_cruising_v2.actors.GU import GU
from gym_cruising_v2.actors.UAV import UAV
from gym_cruising_v2.envs.cruise import Cruise
from gym_cruising_v2.geometry.point import Point
from gym_cruising_v2.geometry.pixel import Pixel
from gym_cruising_v2.geometry.grid import Grid
from gym_cruising_v2.utils.channels_utils import CommunicationChannel

class CruiseUAVWithMap(Cruise):
    uav = []
    gu = []
    pathLoss = []
    SINR = []
    connectivity_matrix = []
    disappear_gu_prob: float
    
    low_observation: float
    high_observation: float

    gu_covered = 0
    max_gu_covered = 0

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
        
        self.alpha = args.alpha
        self.beta = args.beta

        self.communication_channel = CommunicationChannel(args)
        

        self.reset_observation_action_space()


    # RESET ENVIROMENT
    def reset_observation_action_space(self):
        '''
        spawn_area = self.np_random.choice(self.grid.spawn_area)
        (x_min, x_max), (y_min, y_max) = spawn_area

        min_x = x_min - self.max_speed_uav
        max_x = x_max + self.max_speed_uav
        min_y = y_min - self.max_speed_uav
        max_y = y_max + self.max_speed_uav
        
        self.low = np.array([min_x, min_y], dtype=np.float64)
        self.high = np.array([max_x, max_y], dtype=np.float64)
        obs_shape = ((self.uav_number * 2) + self.gu_covered, 2)
        self.low_observation = np.tile(low, (obs_shape[0], 1))
        self.high_observation = np.tile(high, (obs_shape[0], 1))
        '''
        self.observation_space = self.observation_space = spaces.Dict({
            "map_exploration_states": spaces.Box(low=0, high=+1, shape=(self.window_width*self.resolution, self.window_height*self.resolution), dtype=np.float32),
            "uav_states": spaces.Box(low=-1, high=+1, shape=(self.uav_number * 2, 2), dtype=np.float64),
            "covered_users_states": spaces.Box(low=-1, high=+1, shape=(self.gu_covered, 2), dtype=np.float64)
        })

        self.action_space = spaces.Dict({
            "uav_moves": spaces.Box(low=-1, high=+1, shape=(self.uav_number, 2), dtype=np.float64)
        })

    def reset(self, seed=None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        self.uav = []
        self.gu = []
        self.uav_number = options["uav"]
        self.starting_gu_number = options["gu"]
        self.grid.reset()
        self.reset_observation_action_space()
        #RAND
        #self.disappear_gu_prob = self.spawn_gu_prob * 4 / self.starting_gu_number
        self.gu_covered = 0
        self.max_gu_covered = 0
        np.random.seed(seed)
        return super().reset(seed=seed, options=options)
    
    def reset_gu(self, options: Optional[dict] = None) -> np.ndarray:
        self.gu = []
        #RAND
        #self.disappear_gu_prob = self.spawn_gu_prob * 4 / self.starting_gu_number
        self.gu_covered = 0
        self.max_gu_covered = 0
        if not options['clustered']:
            self.init_gu()
        else:
            self.init_gu_clustered(options)
        self.calculate_PathLoss_with_Markov_Chain()
        self.calculate_SINR()
        self.calculate_UAVs_connection_area()
        self.check_connection_and_coverage_UAV_GU()
        return self.get_observation()
    
    
    # INITIALIZE ENVIROMENT
    def init_environment(self, options: Optional[dict] = None) -> None:
        self.init_uav()
        if not options["clustered"]:
            self.init_gu()
        else:
            self.init_gu_clustered(options)
        self.calculate_PathLoss_with_Markov_Chain()
        self.calculate_SINR()
        self.calculate_UAVs_connection_area()
        self.check_connection_and_coverage_UAV_GU()
 
    def init_uav(self) -> None:
        area = self.np_random.choice(self.grid.spawn_area)
        # x_coordinate = int(self.np_random.uniform(area[0][0] + 800, area[0][1] - 800))
        for i in range(self.uav_number):
            while True:
                x_coordinate = int(self.np_random.uniform(area[0][0], area[0][1]))
                y_coordinate = int(self.np_random.uniform(area[1][0], area[1][1]))
                position = self.grid.get_point(x_coordinate, y_coordinate)

                # Primo UAV: non serve il check
                if i == 0 or not self.check_if_are_too_close(i, position):
                    self.uav.append(UAV(position))
                    break

    def init_gu(self) -> None:
        area = self.np_random.choice(self.grid.spawn_area)
        for _ in range(self.starting_gu_number):
            x_coordinate = int(self.np_random.uniform(area[0][0], area[0][1]))
            y_coordinate = int(self.np_random.uniform(area[1][0], area[1][1]))
            position = self.grid.get_point(x_coordinate, y_coordinate)
            gu = GU(position)
            self.initialize_channel(gu)
            self.gu.append(gu)

    def init_gu_clustered(self, options: Optional[dict] = None) -> None:
        area = self.np_random.choice(self.grid.spawn_area)
        std_dev = np.sqrt(options['variance'])
        number_of_clusters = options['clusters_number']
        gu_for_cluster = int(self.starting_gu_number / number_of_clusters)
        for i in range(number_of_clusters):
            mean_x = int(self.np_random.uniform(area[0][0], area[0][1]))
            mean_y = int(self.np_random.uniform(area[0][0], area[0][1]))
            for j in range(gu_for_cluster):
                repeat = True
                while repeat:
                    # Generazione del numero casuale
                    x_coordinate = int(np.random.normal(mean_x, std_dev))
                    y_coordinate = int(np.random.normal(mean_y, std_dev))
                    if area[0][0] <= x_coordinate <= area[0][1] and area[1][0] <= y_coordinate <= area[1][1]:
                        position = self.grid.get_point(x_coordinate, y_coordinate)
                        repeat = False
                gu = GU(position)
                self.initialize_channel(gu)
                self.gu.append(gu)

    def initialize_channel(self, gu):
        for uav in self.uav:
            distance = self.calculate_distance_uav_gu(uav.position, gu.position)
            initial_channel_PLoS = self.communication_channel.get_PLoS(distance, self.uav_altitude)
            sample = random.random()
            #RAND
            '''
            if sample <= initial_channel_PLoS:
                gu.channels_state.append(0)  # 0 = LoS
            else:
                gu.channels_state.append(1)  # 1 = NLoS
            '''
            gu.channels_state.append(0)


    # GET OBSERVATION
    def normalizePositions(self, position) -> np.ndarray:  # Normalize in [-1,1]
            max_x = self.window_width * self.resolution
            max_y = self.window_height * self.resolution

            norm_x = (position.x_coordinate / max_x) * 2 - 1
            norm_y = (position.y_coordinate / max_y) * 2 - 1

            return np.array([norm_x, norm_y], dtype=np.float64)
    
    def normalizeActions(self, actions: np.ndarray) -> np.ndarray:  # Normalize in [-1,1]
        normalized_actions = np.ndarray(shape=actions.shape, dtype=np.float64)
        normalized_actions = ((actions + self.max_speed_uav) / (2 * self.max_speed_uav)) * 2 - 1
        return normalized_actions
    
    def normalizeExplorationMap(self, exploration_map: np.ndarray) -> np.ndarray:  # Normalize in [0,1]
        # Clippa i valori a max_value (saturazione)
        clipped_map = np.clip(exploration_map, 0, self.unexplored_point_max_steps)
        
        # Normalizza in [0,1]
        normalized_map = clipped_map / self.unexplored_point_max_steps
        
        return normalized_map
   
    def get_observation(self) -> dict:
        # map_exploration_states (normalizzata, shape = (window_width, window_height))
        map_exploration = self.normalizeExplorationMap(self.grid.get_point_exploration_map())

        # uav_states: concatena posizioni e azioni normalizzate
        uav_states = np.zeros((self.uav_number * 2, 2), dtype=np.float64)
        for i in range(self.uav_number):
            uav_states[2*i] = self.normalizePositions(self.uav[i].position)
            last_shift = np.array([self.uav[i].last_shift_x, self.uav[i].last_shift_y])
            uav_states[2*i + 1] = self.normalizeActions(last_shift)

        # covered_users_states: posizioni normalizzate solo degli utenti coperti
        covered_users_positions = []
        for gu in self.gu:
            if gu.covered:
                covered_users_positions.append(self.normalizePositions(gu.position))
        # Assicurati che covered_users_positions abbia lunghezza gu_covered, se no gestisci pad/truncate
        covered_users_states = np.array(covered_users_positions, dtype=np.float64)

        observation = {
            "map_exploration_states": map_exploration,
            "uav_states": uav_states,
            "covered_users_states": covered_users_states
        }

        return observation
    
    
    # DO ACTIONS    
    def perform_action(self, actions) -> None:
        self.move_UAV(actions)
        self.update_GU()
        self.calculate_PathLoss_with_Markov_Chain()
        self.calculate_SINR()
        self.calculate_UAVs_connection_area()
        self.check_connection_and_coverage_UAV_GU()

    def update_GU(self):
        #RAND
        #self.check_if_spawn_new_GU()
        self.move_GU()
        #RAND
        #self.check_if_disappear_GU()

    def move_UAV(self, normalized_actions):
        actions = self.max_speed_uav * normalized_actions
        for i, uav in enumerate(self.uav):
            previous_position = uav.position
            new_position = self.grid.get_point(previous_position.x_coordinate + actions[i][0],
                                 previous_position.y_coordinate + actions[i][1])
            uav.position = new_position
            uav.previous_position = previous_position
            uav.last_shift_x = actions[i][0]
            uav.last_shift_y = actions[i][1]

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
                    new_position = self.grid.get_point(previous_position.x_coordinate, previous_position.y_coordinate + distance)
                elif direction == 'down':
                    new_position = self.grid.get_point(previous_position.x_coordinate, previous_position.y_coordinate - distance)
                elif direction == 'left':
                    new_position = self.grid.get_point(previous_position.x_coordinate - distance, previous_position.y_coordinate)
                elif direction == 'right':
                    new_position = self.grid.get_point(previous_position.x_coordinate + distance, previous_position.y_coordinate)

                # check if GU exit from environment
                if new_position.is_in_area(area):
                    repeat = False
                    gu.position = new_position
                    gu.previous_position = previous_position
                else:
                    repeat = True


    # DO CALCULATIONS AND REWARDS
    def calculate_distance_uav_gu(self, uav: Point, gu: Point):
        return math.sqrt(math.pow(uav.x_coordinate - gu.x_coordinate, 2) +
                        math.pow(uav.y_coordinate - gu.y_coordinate, 2) +
                        self.uav_altitude ** 2)
        
    def calculate_distance_uav_point(self, uav: Point, point: Point):
        return math.sqrt(math.pow(uav.x_coordinate - point.x_coordinate, 2) +
                        math.pow(uav.y_coordinate - point.y_coordinate, 2) +
                        self.uav_altitude ** 2)
        
    def calculate_PathLoss_with_Markov_Chain(self):
        self.pathLoss = []
        for gu in self.gu:
            current_GU_PathLoss = []
            #RAND
            '''
            new_channels_state = []
            gu_shift = gu.position.calculate_distance(gu.previous_position)
            '''
            for index, uav in enumerate(self.uav):
                distance = self.calculate_distance_uav_gu(uav.position, gu.position)
                #RAND
                '''
                channel_PLoS = self.communication_channel.get_PLoS(distance, self.uav_altitude)
                relative_shift = uav.position.calculate_distance(uav.previous_position) + gu_shift
                transition_matrix = self.communication_channel.get_transition_matrix(relative_shift, channel_PLoS)
                current_state = np.random.choice(range(len(transition_matrix)),
                                                 p=transition_matrix[gu.channels_state[index]])
                new_channels_state.append(current_state)
                path_loss = self.communication_channel.get_PathLoss(distance, current_state)
                '''
                path_loss = self.communication_channel.get_PathLoss(distance, gu.channels_state[index])
                current_GU_PathLoss.append(path_loss)
            self.pathLoss.append(current_GU_PathLoss)
            #RAND
            '''
            gu.setChannelsState(new_channels_state)
            '''
             
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
    
    def calculate_UAVs_connection_area(self):
        distance_LoS_coverage = self.communication_channel.get_distance_from_SINR_closed_form(self.covered_threshold,0)
        
        uav_points = []
        uav_pixels = []
        for index, uav in enumerate(self.uav):
            uav_point = self.grid.get_point(uav.position.x_coordinate, uav.position.y_coordinate)
            uav_points.append(uav_point)
            uav_pixel = self.grid.get_pixel_from_point(uav_point)
            uav_pixels.append(uav_pixel)
        
        total_uavs_point_coverage = 0
        simultaneous_uavs_point_coverage = 0
        
        for pixel_row in self.grid.pixel_grid:
            for pixel in pixel_row:
                for point_row in pixel.point_grid:
                    for point in point_row:
                        point_covered = False
                        point_covered_more_than_one = False
                        for uav_point in uav_points:
                            if(self.calculate_distance_uav_point(uav_point, point) <= distance_LoS_coverage):
                                total_uavs_point_coverage +=1
                                if not point_covered:
                                    point_covered = True
                                elif point_covered:
                                    simultaneous_uavs_point_coverage +=1
                        if point_covered:       
                            point.set_covered(True)
                            point.reset_step_from_last_visit()
                        else:
                            point.increment_step_from_last_visit()
                pixel.calculate_mean_step_from_last_visit()
            
        return total_uavs_point_coverage, simultaneous_uavs_point_coverage

    def calculate_RCR_without_uav_i(self, i):
        tmp_matrix = np.delete(self.connectivity_matrix, i, axis=1)  # Remove i-th column
        return np.sum(np.any(tmp_matrix, axis=1))
    
    def calculate_reward(self, terminated):
        # REWARD GLOBALE
        global_reward = 0
        
        # 1. Entropia sulla distribuzione dei contributi
        contributions = []
        for i in range(len(self.uav)):
            contribution = self.gu_covered - self.calculate_RCR_without_uav_i(i)
            contributions.append(contribution)
        sum_contributions = sum(contributions)
        proportions = [c / sum_contributions for c in contributions] if sum_contributions > 0 else [1.0 / len(contributions)] * len(contributions)
        entropy = -sum(p * np.log2(p + 1e-8) for p in proportions) # Calcola entropia della distribuzione dei contributi | evita log2(0)
        max_entropy = np.log2(len(self.uav)) # Entropia massima possibile (copertura perfettamente equa)
        entropy_contribution_score = entropy / max_entropy if max_entropy > 0 else 0.0  # normalizzata [0,1]
        global_reward += entropy_contribution_score
        
        # 2. Copertura spaziale (evita UAV sovrapposti)
        total_uavs_point_coverage, simultaneous_uavs_point_coverage = self.calculate_UAVs_connection_area()
        if total_uavs_point_coverage > 0:
            spatial_coverage = (total_uavs_point_coverage - simultaneous_uavs_point_coverage) / total_uavs_point_coverage
        else:
            spatial_coverage = 0.0
        global_reward += spatial_coverage
        
        # 3. Incentivo all'esplorazione (bassa densità di esplorazione)
        map_exploration = self.normalizeExplorationMap(self.grid.get_point_exploration_map())
        exploration_incentive = 1 - np.mean(map_exploration)
        global_reward += exploration_incentive
        
        # 4. Copertura dei GU massima
        if not hasattr(self, 'max_gu_covered'):
            self.max_gu_covered = self.gu_covered
        else:
            self.max_gu_covered = max(self.max_gu_covered, self.gu_covered)
        coverage_score = self.gu_covered / self.max_gu_covered if self.max_gu_covered > 0 else 0.0
        global_reward += coverage_score

            
        # REWARD INDIVIDUALE
        individual_rewards = []
        for i in range(len(self.uav)):
            if terminated[i]:
                individual_rewards.append(-2.0)
            else:
                contribution_score = contributions[i] / self.gu_covered if self.gu_covered > 0 else 0.0
                reward = self.alpha * contribution_score + self.beta * global_reward
                individual_rewards.append(reward)

        self.log_rewards(individual_rewards, contributions, entropy_contribution_score, spatial_coverage, exploration_incentive, coverage_score)
        return individual_rewards
    
    
    # CHECKS
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
                x_coordinate = int(self.np_random.uniform(area[0][0], area[0][1]))
                y_coordinate = int(self.np_random.uniform(area[1][0], area[1][1]))
                position = self.grid.get_point(x_coordinate, y_coordinate)
                gu = GU(position)
                self.initialize_channel(gu)
                self.gu.append(gu)
        # update disappear gu probability
        self.disappear_gu_prob = self.spawn_gu_prob * 4 / len(self.gu)

    def check_connection_and_coverage_UAV_GU(self):
        covered = 0
        self.connectivity_matrix = np.zeros((len(self.gu), len(self.uav)), dtype=int)
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

    def check_if_truncated(self) -> bool:
        return False
    
    def check_if_terminated(self):
        terminated_matrix = []
        area = self.np_random.choice(self.grid.spawn_area)
        for i, uav in enumerate(self.uav):
            if not uav.position.is_in_area(area) or self.check_collision(i, uav):
                terminated_matrix.append(True)
            else:
                terminated_matrix.append(False)
        return terminated_matrix

    def check_collision(self, current_uav_index, uav) -> bool:
        collision = False
        for j, other_uav in enumerate(self.uav):
            if j != current_uav_index:
                if uav.position.calculate_distance(other_uav.position) <= self.collision_distance:
                    collision = True
                    break
        return collision

    def check_if_are_too_close(self, uav_index, position):
        too_close = False
        for j in range(uav_index):
            if self.uav[j].position.calculate_distance(position) <= self.minimum_starting_distance_between_uav:
                too_close = True
                break
        return too_close
    
    
    # PYGAME METHODS
    def draw(self, canvas: Surface) -> None:
        # CANVAS
        canvas.fill((255,255,255))
        
        for x in range(len(self.grid.pixel_grid)):
            for y in range(len(self.grid.pixel_grid[0])):
                pixel = self.grid.pixel_grid[x][y]
                color = pixel.color  # Assumendo sia una tupla (R, G, B)
                canvas.set_at((x, y), color)

        # WALL
        '''
        for wall in self.world:
            pygame.draw.line(canvas,
                             Color.BLACK.value,
                             self.convert_point(wall.start),
                             self.convert_point(wall.end),
                             self.wall_width)
        '''

        # GU image
        for gu in self.gu:
            canvas.blit(pygame.image.load(gu.getImage()), self.image_convert_point(gu.position))

        # UAV image
        icon_drone = pygame.image.load('./gym_cruising_v2/images/drone30.png')
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

    
    # INFO METHODS
    def create_info(self, terminated) -> dict:
        collision = False
        for i, uav in enumerate(self.uav):
            if self.check_collision(i, uav):
                collision = True
                break
        if collision:
            RCR = str(0.0)
        else:
            RCR = str(self.gu_covered/len(self.gu))
        return {"GU coperti": str(self.gu_covered), "Ground Users": str(
            len(self.gu)), "RCR": RCR, "Collision": collision}

    def log_rewards(self, rewards, contributions, entropy_score, spatial_coverage, exploration_incentive, coverage_score):
        print(f"{'UAV':<5} {'Reward':<10} {'Contr.':<10}")
        for i, (r, c) in enumerate(zip(rewards, contributions)):
            print(f"{i:<5} {r:<10.3f} {c:<10.3f}")
        print(f"\n[GLOBAL] Entropia: {entropy_score:.3f} | Copertura spaziale: {spatial_coverage:.3f} | Esplorazione: {exploration_incentive:.3f} | Copertura GU: {coverage_score:.3f}\n")

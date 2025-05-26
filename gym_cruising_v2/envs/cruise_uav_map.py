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
from gym_cruising_v2.geometry.coordinate import Coordinate
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
    total_uavs_point_coverage = 0
    simultaneous_uavs_point_coverage = 0

    def __init__(self, args, render_mode=None) -> None:
        super().__init__(args, render_mode)
        
        self.max_uav_number = args.max_uav_number
        self.uav_number = args.uav_number
        self.max_gu_number = args.max_gu_number
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
        self.observation_space = spaces.Dict({
            "map_exploration_states": spaces.Box(
                low=0, high=1,
                shape=(self.grid.grid_height, self.grid.grid_width),
                dtype=np.float32
            ),
            "uav_states": spaces.Box(
                low=-1, high=1,
                shape=(self.max_uav_number, 4),
                dtype=np.float64
            ),
            "uav_mask": spaces.Box(
                low=0, high=1,
                shape=(self.max_uav_number,),
                dtype=bool
            ),
            "covered_users_states": spaces.Box(
                low=-1, high=1,
                shape=(self.max_gu_number, 2),
                dtype=np.float64
            ),
            "gu_mask": spaces.Box(
                low=0, high=1,
                shape=(self.max_gu_number,),
                dtype=bool
            )
        })

        self.action_space = spaces.Dict({
            "uav_moves": spaces.Box(
                low=-1, high=1,
                shape=(self.max_uav_number, 2),
                dtype=np.float64
            ),
            "uav_mask": spaces.Box(
                low=0, high=1,
                shape=(self.max_uav_number,),
                dtype=bool
            )
        })

    def reset(self, seed=None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        if seed is not None:
            self.seed = seed

        if options is not None:
            self.options = options
            
        self.uav = []
        self.gu = []
        self.uav_number = self.options["uav"]
        self.starting_gu_number = self.options["gu"]
        #RAND
        #self.disappear_gu_prob = self.spawn_gu_prob * 4 / self.starting_gu_number
        self.gu_covered = 0
        self.max_gu_covered = 0
        self.total_uavs_point_coverage = 0
        self.simultaneous_uavs_point_coverage = 0
        self.grid.reset()
        self.reset_observation_action_space()
        np.random.seed(self.seed)
        return super().reset(seed=self.seed, options=self.options)
    
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
        self.calculate_UAVs_connection_area_vectorized()
        self.check_connection_and_coverage_UAV_GU()
        self.reset_observation_action_space()
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
        self.calculate_UAVs_connection_area_vectorized()
        self.check_connection_and_coverage_UAV_GU()
        self.reset_observation_action_space()
 
    def init_uav(self) -> None:
        area = self.np_random.choice(self.grid.spawn_area)
        # x_coordinate = int(self.np_random.uniform(area[0][0] + 800, area[0][1] - 800))
        for i in range(self.uav_number):
            while True:
                x_coordinate = self.np_random.uniform(area[0][0], area[0][1])
                y_coordinate = self.np_random.uniform(area[1][0], area[1][1])
                position = Coordinate(x_coordinate, y_coordinate, self.uav_altitude)

                # Primo UAV: non serve il check
                if i == 0 or not self.check_if_are_too_close(i, position):
                    self.uav.append(UAV(position))
                    break

    def init_gu(self) -> None:
        area = self.np_random.choice(self.grid.spawn_area)
        for _ in range(self.starting_gu_number):
            x_coordinate = self.np_random.uniform(area[0][0], area[0][1])
            y_coordinate = self.np_random.uniform(area[1][0], area[1][1])
            position = Coordinate(x_coordinate, y_coordinate, 0)
            gu = GU(position)
            self.initialize_channel(gu)
            self.gu.append(gu)

    def init_gu_clustered(self, options: Optional[dict] = None) -> None:
        area = self.np_random.choice(self.grid.spawn_area)
        std_dev = np.sqrt(options['variance'])
        number_of_clusters = options['clusters_number']
        gu_for_cluster = int(self.starting_gu_number / number_of_clusters)
        for i in range(number_of_clusters):
            mean_x = self.np_random.uniform(area[0][0], area[0][1])
            mean_y = self.np_random.uniform(area[0][0], area[0][1])
            for j in range(gu_for_cluster):
                repeat = True
                while repeat:
                    # Generazione del numero casuale
                    x_coordinate = np.random.normal(mean_x, std_dev)
                    y_coordinate = np.random.normal(mean_y, std_dev)
                    if area[0][0] <= x_coordinate <= area[0][1] and area[1][0] <= y_coordinate <= area[1][1]:
                        position = Coordinate(x_coordinate, y_coordinate, 0)
                        repeat = False
                gu = GU(position)
                self.initialize_channel(gu)
                self.gu.append(gu)

    def initialize_channel(self, gu):
        for uav in self.uav:
            distance = gu.position.calculate_distance_to_coordinate(uav.position)
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
        return np.clip(exploration_map, 0, self.unexplored_point_max_steps) / self.unexplored_point_max_steps
   
    def get_observation(self) -> dict:
        # Ottieni e normalizza la mappa di esplorazione (shape = (window_width, window_height))
        map_exploration = self.normalizeExplorationMap(self.grid.get_point_exploration_map())

        # Inizializza lo stato degli UAV e la maschera di validità (padding)
        uav_states = np.zeros((self.max_uav_number, 4), dtype=np.float64)
        uav_mask = np.zeros(self.max_uav_number, dtype=bool)

        # Popola uav_states con posizione e azione normalizzate per ogni UAV attivo
        for i in range(len(self.uav)):
            pos = self.normalizePositions(self.uav[i].position)  # posizione normalizzata [x, y]
            act = self.normalizeActions(np.array([self.uav[i].last_shift_x, self.uav[i].last_shift_y]))  # azione normalizzata [dx, dy]
            uav_states[i] = np.concatenate([pos, act])  # concatena posizione e azione in vettore di dimensione 4
            uav_mask[i] = True  # marca UAV come attivo

        # Le posizioni da uav_number a max_uav_number rimangono a zero e sono indicate come inattive nella maschera (padding)

        # Inizializza lo stato degli utenti coperti e la relativa maschera (padding)
        covered_users_states = np.zeros((self.max_gu_number, 2), dtype=np.float64)
        gu_mask = np.zeros(self.max_gu_number, dtype=bool)

        # Raccogli le posizioni normalizzate solo degli utenti coperti
        covered_users_positions = []
        for gu in self.gu:
            if gu.covered:
                covered_users_positions.append(self.normalizePositions(gu.position))

        num_covered = len(covered_users_positions)

        # Copia le posizioni normalizzate nella matrice di output fino al massimo consentito
        num_to_copy = min(num_covered, self.max_gu_number)
        for i in range(num_to_copy):
            covered_users_states[i] = covered_users_positions[i]
            gu_mask[i] = True  # marca l’utente come coperto/attivo

        # Le righe da num_to_copy a max_gu_number rimangono zero e sono marcate come inattive nella maschera (padding)

        # Crea il dizionario di osservazione da restituire
        observation = {
            "map_exploration_states": map_exploration,
            "uav_states": uav_states,
            "uav_mask": uav_mask,
            "covered_users_states": covered_users_states,
            "gu_mask": gu_mask
        }

        return observation

    
    
    # DO ACTIONS    
    def perform_action(self, actions) -> None:
        self.move_UAV(actions)
        self.update_GU()
        
        self.calculate_PathLoss_with_Markov_Chain()
        self.calculate_SINR()
        self.calculate_UAVs_connection_area_vectorized()
        self.check_connection_and_coverage_UAV_GU()
        self.reset_observation_action_space()

    def update_GU(self):
        #RAND
        #self.check_if_spawn_new_GU()
        self.move_GU()
        #RAND
        #self.check_if_disappear_GU()

    def move_UAV(self, actions):
        real_actions = actions["uav_moves"][actions["uav_mask"]]  
        actions = self.max_speed_uav * real_actions
        for i, uav in enumerate(self.uav):
            uav.position = Coordinate(uav.position.x_coordinate + actions[i][0], uav.position.y_coordinate + actions[i][1], self.uav_altitude)
            uav.last_shift_x = actions[i][0]
            uav.last_shift_y = actions[i][1]

    def move_GU(self):
        area = self.np_random.choice(self.grid.spawn_area)
        for gu in self.gu:
            repeat = True
            while repeat:
                distance = self.np_random.normal(self.gu_mean_speed, self.gu_standard_deviation)
                if distance < 0.0:
                    distance = 0.0
                direction = np.random.choice(['up', 'down', 'left', 'right'])

                if direction == 'up':
                    new_position = Coordinate(gu.position.x_coordinate, gu.position.y_coordinate + distance,0)
                elif direction == 'down':
                    new_position = Coordinate(gu.position.x_coordinate, gu.position.y_coordinate - distance,0)
                elif direction == 'left':
                    new_position = Coordinate(gu.position.x_coordinate - distance, gu.position.y_coordinate,0)
                elif direction == 'right':
                    new_position = Coordinate(gu.position.x_coordinate + distance, gu.position.y_coordinate,0)

                # check if GU exit from environment
                if new_position.is_in_area(area):
                    repeat = False
                    gu.position = new_position
                else:
                    repeat = True


    # DO CALCULATIONS AND REWARDS 
        
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
                distance = gu.position.calculate_distance_to_coordinate(uav.position)
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
    
    # Deprecated
    def calculate_UAVs_connection_area(self):
        distance_LoS_coverage = self.communication_channel.get_distance_from_SINR_closed_form(self.covered_threshold,0)
        
        uav_positions = [uav.position for uav in self.uav]
        
        self.total_uavs_point_coverage = 0
        self.simultaneous_uavs_point_coverage = 0
        
        for pixel_row in self.grid.pixel_grid:
            for pixel in pixel_row:
                for point_row in pixel.point_grid:
                    for point in point_row:
                        point_covered = False
                        for uav_pos in uav_positions:
                            if(uav_pos.calculate_distance_to_point(point) <= distance_LoS_coverage):
                                self.total_uavs_point_coverage +=1
                                if not point_covered:
                                    point_covered = True
                                elif point_covered:
                                    self.simultaneous_uavs_point_coverage +=1
                        if point_covered:       
                            point.set_covered(True)
                            point.reset_step_from_last_visit()
                        else:
                            point.increment_step_from_last_visit()
                pixel.calculate_mean_step_from_last_visit()

    def calculate_UAVs_connection_area_vectorized(self):
        # Calcola la distanza di copertura in Line-of-Sight (LoS) basata sul SINR e sulla soglia di copertura
        distance_LoS_coverage = self.communication_channel.get_distance_from_SINR_closed_form(
            self.covered_threshold, 0
        )

        # Estrae le posizioni 3D di tutti gli UAV in un array NumPy (UAV, 3)
        uav_positions = np.array([uav.position.to_array() for uav in self.uav])

        point_refs = []
        point_coords = []

        # Raccoglie tutti i punti e le loro coordinate 3D direttamente dalla griglia di punti
        for row in self.grid.point_grid:
            for point in row:
                point_refs.append(point)
                point_coords.append(point.to_array_3d())

        point_coords = np.array(point_coords)  # (Punti, 3)

        # Calcola le distanze euclidee al quadrato tra ogni punto e ogni UAV (matrice P x U)
        diffs = point_coords[:, None, :] - uav_positions[None, :, :]
        dists_squared = np.sum(diffs ** 2, axis=2)

        # Crea una maschera booleana che indica se un punto è coperto da almeno un UAV
        coverage_mask = dists_squared <= distance_LoS_coverage ** 2

        # Conta quanti UAV coprono ciascun punto
        coverage_counts = np.sum(coverage_mask, axis=1)

        # Inizializza le statistiche di copertura
        total_coverage = 0
        simultaneous_coverage = 0

        # Aggiorna lo stato di ogni punto sulla base del numero di UAV che lo coprono
        for i, point in enumerate(point_refs):
            count = coverage_counts[i]
            if count > 0:
                point.set_covered(True)
                point.reset_step_from_last_visit()
                total_coverage += count
                if count > 1:
                    simultaneous_coverage += count - 1
            else:
                point.increment_step_from_last_visit()

        # Salva le statistiche calcolate
        self.total_uavs_point_coverage = total_coverage
        self.simultaneous_uavs_point_coverage = simultaneous_coverage

        # Aggiorna metriche per pixel solo se in modalità "human" (visualizzazione)
        if self.render_mode == "human":
            for pixel_row in self.grid.pixel_grid:
                for pixel in pixel_row:
                    pixel.calculate_mean_step_from_last_visit()


    
    def calculate_RCR_without_uav_i(self, i):
        tmp_matrix = np.delete(self.connectivity_matrix, i, axis=1)  # Remove i-th column
        return np.sum(np.any(tmp_matrix, axis=1))
    
    def calculate_reward(self, terminated):
        # REWARD GLOBALE
        global_reward = 0
        
        
        # 1. Entropia sulla distribuzione dei contributi
        num_uav = len(self.uav)
        contributions = [
            max(0.0, self.gu_covered - self.calculate_RCR_without_uav_i(i))
            for i in range(len(self.uav))
        ]
        sum_contributions = sum(contributions)
        if sum_contributions > 0:
            proportions = [c / sum_contributions for c in contributions]
        else:
            proportions = [1.0 / num_uav] * num_uav
        entropy = -sum(p * np.log2(p + 1e-8) for p in proportions)
        max_entropy = np.log2(num_uav)
        
        if num_uav <= 1:
            entropy_contribution_score = 1.0  # oppure 1.0, a seconda della logica che vuoi seguire
        else:
            entropy_contribution_score = entropy / max_entropy
        
        entropy_contribution_score = 0.0 # Per ora disabilitato, ma puoi riattivarlo se necessario    
        global_reward += entropy_contribution_score
        
        
        # 2. Copertura spaziale (evita UAV sovrapposti)
        if self.total_uavs_point_coverage > 0:
            spatial_coverage = (self.total_uavs_point_coverage - self.simultaneous_uavs_point_coverage) / self.total_uavs_point_coverage
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
        
        global_reward /= 3.0  # Normalizza il reward globale (divide per il numero di metriche considerate)
        
        #5. Penalità per UAV terminati
        if any(terminated):
            global_reward = -5.0
            

        '''  
        # REWARD INDIVIDUALE
        individual_rewards = []
        for i in range(self.max_uav_number):
            if terminated[i]:
                individual_rewards.append(-2.0)
            else:
                contribution_score = contributions[i] / self.gu_covered if self.gu_covered > 0 else 0.0
                reward = self.alpha * contribution_score + self.beta * global_reward
                individual_rewards.append(reward)
        ''' 

        #self.log_rewards(contributions, entropy_contribution_score, spatial_coverage, exploration_incentive, coverage_score)
        return global_reward
    
    
    # CHECKS
    #TODO
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

    #TODO
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
        
        # Aggiungi True per gli UAV inattivi (padding)
        padding = self.max_uav_number - len(self.uav)
        terminated_matrix.extend([True] * padding)
        return terminated_matrix

    def check_collision(self, current_uav_index, uav) -> bool:
        collision = False
        for j, other_uav in enumerate(self.uav):
            if j != current_uav_index:
                if uav.position.calculate_distance_to_coordinate(other_uav.position) <= self.collision_distance:
                    collision = True
                    break
        return collision

    def check_if_are_too_close(self, uav_index, position):
        too_close = False
        for j in range(uav_index):
            if self.uav[j].position.calculate_distance_to_coordinate(position) <= self.minimum_starting_distance_between_uav:
                too_close = True
                break
        return too_close
    
    # PYGAME METHODS
    def draw(self, canvas: Surface) -> None:
        # CANVAS
        canvas.fill((255,255,255))
        
        for row in range(len(self.grid.pixel_grid)):
            for col in range(len(self.grid.pixel_grid[0])):
                pixel = self.grid.pixel_grid[row][col]
                color = pixel.color  # Assumendo sia una tupla (R, G, B)
                canvas.set_at((col, row), color)

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
        pygame_x = (round(point.point_col * self.resolution)
                    + self.x_offset)
        pygame_y = (self.window_height
                    - round(point.point_row * self.resolution)
                    + self.y_offset)
        return pygame_x, pygame_y

    def image_convert_point(self, point: Point) -> Tuple[int, int]:
        shiftX = 15
        shiftY = 15
        pygame_x = (round(point.point_col * self.resolution) - shiftX + self.x_offset)
        pygame_y = (self.window_height - round(point.point_row * self.resolution) - shiftY + self.y_offset)
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

    def log_rewards(self, contributions, entropy_contribution_score, spatial_coverage, exploration_incentive, coverage_score):
        print("=== REWARD LOG ===")
        
        # Global reward components
        print(f"Entropy contribution score   : {entropy_contribution_score:.4f}")
        print(f"Spatial coverage             : {spatial_coverage:.4f}")
        print(f"Exploration incentive        : {exploration_incentive:.4f}")
        print(f"Coverage score               : {coverage_score:.4f}")
        total_global_reward = entropy_contribution_score + spatial_coverage + exploration_incentive + coverage_score
        print(f"Global reward (total)        : {total_global_reward:.4f}")
        
        # UAV contributions
        print("UAV Contributions:")
        for i, c in enumerate(contributions):
            print(f"  UAV {i}: contribution = {c:.4f}")

        print("===================")

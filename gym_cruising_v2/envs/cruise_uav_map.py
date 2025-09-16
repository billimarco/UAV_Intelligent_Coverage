""" This module contains the Cruising environment class """
import random
from typing import Optional, Tuple
import math
import os

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

    gu_covered = 0
    max_theoretical_ground_uavs_points_coverage = 0
    max_theoretical_new_explored_ground_uavs_points = 0
    
    boundary_penalty_total = 0.0
    collision_penalty_total = 0.0
    spatial_coverage_total = 0.0
    exploration_incentive_total = 0.0
    homogenous_voronoi_partition_incentive_total = 0.0
    gu_coverage_total = 0.0
    
    exploration_phase = True
    steps_gu_coverage_phase = 0
    
    #total_uavs_point_coverage = 0
    #simultaneous_uavs_points_coverage = 0
    uav_total_covered_points = []
    uav_shared_covered_points = []
    
    voronoi_partition = None
    
    last_unexplored_area_points: int
    new_explored_area_points: int

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
        
        self.w_boundary_penalty = args.w_boundary_penalty
        self.w_collision_penalty = args.w_collision_penalty
        self.w_spatial_coverage = args.w_spatial_coverage
        self.w_exploration = args.w_exploration
        self.w_homogenous_voronoi_partition = args.w_homogenous_voronoi_partition
        self.w_gu_coverage = args.w_gu_coverage
        
        self.reward_mode = args.reward_mode
        self.spatial_coverage_threshold = args.spatial_coverage_threshold
        self.exhaustive_exploration_threshold = args.exhaustive_exploration_threshold
        self.balanced_exploration_threshold = args.balanced_exploration_threshold
        self.max_steps_gu_coverage_phase = args.max_steps_gu_coverage_phase
        self.tradeoff_mode = args.tradeoff_mode
        self.k_factor = args.k_factor
        
        self.theoretical_max_distance_before_possible_collision = 2*math.sqrt(self.max_speed_uav**2 + self.max_speed_uav**2) + self.collision_distance

        self.communication_channel = CommunicationChannel(args)
        
        # Reset structures
        self.available_gu_indexes = [False]*args.max_gu_number
        self.covered_users_states = np.zeros((self.max_gu_number, 2), dtype=np.float64)
        self.gu_mask = np.zeros(self.max_gu_number, dtype=bool)
        
        self.reset_observation_action_space()


    # RESET ENVIROMENT
    def reset_observation_action_space(self):
        self.observation_space = spaces.Dict({
            "map_exploration_states": spaces.Box(
                low=0, high=1,
                shape=(self.grid.grid_height, self.grid.grid_width),
                dtype=np.float32
            ),
            "uav_states": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.max_uav_number, 4),
                dtype=np.float64
            ),
            "uav_mask": spaces.Box(
                low=0, high=1,
                shape=(self.max_uav_number,),
                dtype=bool
            ),
            "uav_flags": spaces.Box(
                low=0, high=1,
                shape=(self.max_uav_number, self.max_uav_number + 4),# True se uav rispettivo vicino + possibilità uscita dai bordi
                dtype=bool
            ),
            "covered_users_states": spaces.Box(
                low=-np.inf, high=np.inf,
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
            
        self.seed = random.randint(0, 10000)
        np.random.seed(self.seed)
            
        self.uav = []
        self.gu = []
        self.pathLoss = []
        self.SINR = []
        self.connectivity_matrix = []
        self.uav_number = self.options["uav"]
        self.starting_gu_number = self.options["gu"]
        #RAND
        #self.disappear_gu_prob = self.spawn_gu_prob * 4 / self.starting_gu_number
        self.gu_covered = 0
        self.max_theoretical_ground_uavs_points_coverage = 0
        self.max_theoretical_new_explored_ground_uavs_points = 0
        
        self.boundary_penalty_total = 0.0
        self.collision_penalty_total = 0.0
        self.spatial_coverage_total = 0.0
        self.exploration_incentive_total = 0.0
        self.homogenous_voronoi_partition_incentive_total = 0.0
        self.gu_coverage_total = 0.0
        
        self.exploration_phase = True
        self.steps_gu_coverage_phase = 0
        
        #self.total_uavs_point_coverage = 0
        #self.simultaneous_uavs_points_coverage = 0
        self.uav_total_covered_points = []
        self.uav_shared_covered_points = []
        
        voronoi_partition = None
        
        self.available_gu_indexes = [False]*self.max_gu_number
        self.last_position_users_states = np.zeros((self.max_gu_number, 2), dtype=np.float64)
        self.gu_mask = np.zeros(self.max_gu_number, dtype=bool)
        
        self.grid.reset()
        self.last_unexplored_area_points = self.grid.grid_width*self.grid.grid_height
        self.new_explored_area_points = 0
        self.reset_observation_action_space()
        np.random.seed(self.seed)
        return super().reset(seed=self.seed, options=self.options)

    
    # INITIALIZE ENVIROMENT
    def init_environment(self, options: Optional[dict] = None) -> None:
        self.init_uav()
        if not options["clustered"]:
            self.init_gu()
        else:
            self.init_gu_clustered(options)

        self.calculate_PathLoss_with_Markov_Chain()
        self.calculate_SINR()
        self.calculate_UAVs_connection_area_vectorized() # Calcolo della copertura della mappa di esplorazione
        self.check_connection_and_coverage_UAV_GU()
        
        if self.w_exploration > 0.0 or self.w_spatial_coverage > 0.0:
            self.calculate_uav_area_max_coverage()
        if self.w_exploration > 0.0:
            self.calculate_new_explored_area_points()
        if self.w_homogenous_voronoi_partition > 0.0:
            self.calculate_voronoi_partition()
 
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
                    self.uav.append(UAV(i, position))
                    break

    def init_gu(self) -> None:
        area = self.np_random.choice(self.grid.spawn_area)
        for i in range(self.starting_gu_number):
            x_coordinate = self.np_random.uniform(area[0][0], area[0][1])
            y_coordinate = self.np_random.uniform(area[1][0], area[1][1])
            position = Coordinate(x_coordinate, y_coordinate, 0)
            gu = GU(i,position)
            self.available_gu_indexes[i] = True
            self.init_channel(gu)
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
                gu = GU(i*gu_for_cluster+j,position)
                self.available_gu_indexes[i*gu_for_cluster+j] = True
                self.init_channel(gu)
                self.gu.append(gu)

    def init_channel(self, gu):
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
        #self.grid.save_point_exploration_image("exploration_map.png")
        map_exploration = self.normalizeExplorationMap(self.grid.get_point_exploration_map())

        # Inizializza lo stato degli UAV e la maschera di validità (padding)
        uav_states = np.zeros((self.max_uav_number, 4), dtype=np.float64)
        uav_mask = np.zeros(self.max_uav_number, dtype=bool)
        uav_flags = np.zeros((self.max_uav_number, self.max_uav_number + 4), dtype=bool)

        # Popola uav_states con posizione e azione normalizzate per ogni UAV attivo
        for i in range(len(self.uav)):
            pos = self.normalizePositions(self.uav[i].position)  # posizione normalizzata [x, y]
            act = self.normalizeActions(np.array([self.uav[i].last_shift_x, self.uav[i].last_shift_y]))  # azione normalizzata [dx, dy]
            uav_states[i] = np.concatenate([pos, act])  # concatena posizione e azione in vettore di dimensione 4
            uav_mask[i] = True  # marca UAV come attivo
            
            for j in range(len(self.uav)):
                if i == j:
                    continue
                uav_distance = self.uav[i].position.calculate_distance_to_coordinate(self.uav[j].position)
                h_i = self.uav[i].position.z_coordinate
                h_j = self.uav[j].position.z_coordinate
                R = self.communication_channel.get_distance_from_SINR_closed_form(self.covered_threshold, 0)
                if R > h_i:  # Solo se esiste una proiezione al suolo
                    margin_i = math.ceil(math.sqrt(R**2 - h_i**2))
                if R > h_j:  # Solo se esiste una proiezione al suolo
                    margin_j = math.ceil(math.sqrt(R**2 - h_j**2))
                min_dist = margin_i + margin_j
                if uav_distance <= min_dist:
                    uav_flags[i][j] = True
            
            R = self.communication_channel.get_distance_from_SINR_closed_form(self.covered_threshold, 0)

            x, y, h = self.uav[i].position.x_coordinate, self.uav[i].position.y_coordinate, self.uav[i].position.z_coordinate
            R = self.communication_channel.get_distance_from_SINR_closed_form(self.covered_threshold, 0)
            distances = np.array([x, self.grid.grid_width - x, y, self.grid.grid_height - y]) # Sinistra, Destra, Giu, Su
            if R > h:  # Solo se esiste una proiezione al suolo
                margin = math.ceil(math.sqrt(R**2 - h**2))
            
            for k, dist in enumerate(distances):
                if dist < margin:
                    uav_flags[i][self.max_uav_number + k] = True
                    

        # Le posizioni da uav_number a max_uav_number rimangono a zero e sono indicate come inattive nella maschera (padding)


        # Inizializza lo stato delle ultime posizioni normalizzate degli utenti
        last_normalized_positions_users_states = np.zeros((self.max_gu_number, 2), dtype=np.float64)
 
        # Togli le posizioni dal vettore di stato delle ultime posizioni che sono attualmente coperte e la maschera
        for gu_id, last_gu_position in enumerate(self.last_position_users_states):
            last_gu_position_coordinate = Coordinate(last_gu_position[0], last_gu_position[1], 0) # posizione normalizzata [x, y]
            if self.grid.get_point_from_coordinate(last_gu_position_coordinate).is_covered():
                self.last_position_users_states[gu_id] = np.array([0, 0], dtype=np.float64)  # resetta la posizione dell'utente
                self.gu_mask[gu_id] = False  # inattivo l'utente
              
        # Aggiorna le posizioni del vettore di stato delle ultime posizioni degli utenti coperti e la maschera
        for gu in self.gu:
            if gu.covered:
                self.last_position_users_states[gu.id] = gu.position.to_array_2d()  # aggiorna la posizione dell'utente
                self.gu_mask[gu.id] = True  # marca l’utente come coperto/attivo
        
        # Raccogli le posizioni normalizzate degli utenti coperti
        for gu_id, last_gu_position in enumerate(self.last_position_users_states):
            if self.gu_mask[gu_id]:
                last_gu_position_coordinate = Coordinate(last_gu_position[0], last_gu_position[1], 0)
                last_normalized_positions_users_states[gu_id] = self.normalizePositions(last_gu_position_coordinate)  # posizione normalizzata [x, y]

        # Crea il dizionario di osservazione da restituire
        observation = {
            "map_exploration_states": map_exploration,
            "uav_states": uav_states,
            "uav_mask": uav_mask,
            "uav_flags": uav_flags,
            "covered_users_states": last_normalized_positions_users_states,
            "gu_mask": self.gu_mask
        }

        return observation

    
    
    # DO ACTIONS    
    def perform_action(self, actions) -> None:
        self.move_UAV(actions)
        self.update_GU()
        
        self.calculate_PathLoss_with_Markov_Chain()
        self.calculate_SINR()
        self.calculate_UAVs_connection_area_vectorized() # Calcolo della copertura della mappa di esplorazione
        self.check_connection_and_coverage_UAV_GU()
        
        if self.w_exploration > 0.0 or self.w_spatial_coverage > 0.0:
            self.calculate_uav_area_max_coverage()
        if self.w_exploration > 0.0:
            self.calculate_new_explored_area_points()
        if self.w_homogenous_voronoi_partition > 0.0:
            self.calculate_voronoi_partition()

    def update_GU(self):
        #RAND
        #self.check_if_spawn_new_GU()
        self.move_GU()
        #RAND
        #self.check_if_disappear_GU()

    def move_UAV(self, actions):
        active_actions = actions["uav_moves"][actions["uav_mask"]]
        
        # Movimento in un cerchio 2D con valori limitati da [-self.max_speed_uav, self.max_speed_uav]
        # Limita i vettori al cerchio unitario
        norms = np.linalg.norm(active_actions, axis=1, keepdims=True)
        norms = np.maximum(norms, 1.0)  # Se <=1 lascia invariato, se >1 normalizza
        clipped_actions = active_actions / norms

        # Scala al massimo raggio
        real_actions = self.max_speed_uav * clipped_actions
        
        for i, uav in enumerate(self.uav):
            uav.position = Coordinate(uav.position.x_coordinate + real_actions[i][0], uav.position.y_coordinate + real_actions[i][1], self.uav_altitude)
            uav.last_shift_x = real_actions[i][0]
            uav.last_shift_y = real_actions[i][1]

    def move_GU(self):
        area = self.np_random.choice(self.grid.available_area)
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
               
    
    def calculate_UAVs_connection_area_vectorized(self):
        # Calcola la distanza di copertura in Line-of-Sight (LoS) basata sul SINR e sulla soglia di copertura
        distance_LoS_coverage = self.communication_channel.get_distance_from_SINR_closed_form(
            self.covered_threshold, 0
        )

        # Estrae le posizioni 3D di tutti gli UAV in un array NumPy (UAV, 3)
        uav_positions = np.array([uav.position.to_array_3d() for uav in self.uav])

        # Flatten dei punti della griglia
        point_grid_flat = [point for row in self.grid.point_grid for point in row]
        point_coords = np.array([p.to_array_3d() for p in point_grid_flat])  # (P, 3)

        # Calcola le distanze euclidee al quadrato tra ogni punto e ogni UAV (matrice P x U)
        diffs = point_coords[:, None, :] - uav_positions[None, :, :]
        dists_squared = np.sum(diffs ** 2, axis=2)

        # Crea una maschera booleana che indica se un punto è coperto da almeno un UAV
        coverage_mask = dists_squared <= distance_LoS_coverage ** 2

        # Conta quanti UAV coprono ciascun punto
        coverage_counts = np.sum(coverage_mask, axis=1)

        # Statistiche per UAV
        self.uav_total_covered_points = np.sum(coverage_mask, axis=0).tolist()
        self.uav_shared_covered_points = np.sum(coverage_mask & (coverage_counts[:, None] > 1), axis=0).tolist()

        
        '''
        # Inizializza le statistiche di copertura
        total_coverage = 0
        simultaneous_coverage = 0
        '''

        # Aggiorna stato dei punti
        covered = coverage_counts > 0
        for point, is_covered in zip(point_grid_flat, covered):
            if is_covered:
                point.set_covered(True)
                point.reset_step_from_last_visit()
            else:
                point.set_covered(False)
                point.increment_step_from_last_visit()

        '''
        # Salva le statistiche calcolate
        self.total_uavs_point_coverage = total_coverage
        self.simultaneous_uavs_points_coverage = simultaneous_coverage
        '''

        # Aggiorna visualizzazione solo se necessario
        if self.render_mode == "human":
            for pixel in [p for row in self.grid.pixel_grid for p in row]:
                pixel.calculate_mean_step_from_last_visit()

    
    def calculate_uav_area_max_coverage(self):
        max_ground_coverage_points = 0
        max_new_explored_ground_points = 0

        for uav in self.uav:
            h = uav.position.z_coordinate
            R = self.communication_channel.get_distance_from_SINR_closed_form(self.covered_threshold, 0)

            if R > h:  # Solo se esiste una proiezione al suolo
                r_ground = math.ceil(math.sqrt(R**2 - h**2))
                area = math.pi * r_ground**2
                max_ground_coverage_points += math.ceil(area)
                
                new_area = self.calculate_single_uav_theoretical_new_area_explored_after_one_step(r_ground, self.max_speed_uav)
                max_new_explored_ground_points += math.ceil(new_area)

        self.max_theoretical_ground_uavs_points_coverage = max_ground_coverage_points
        self.max_theoretical_new_explored_ground_uavs_points = max_new_explored_ground_points
    
    def calculate_single_uav_theoretical_new_area_explored_after_one_step(self, r, d):
        # si considera che il raggio di copertura non cambi. In caso cambia va cambiata l'implementazione. https://mathworld.wolfram.com/Circle-CircleIntersection.html
        if d >= 2 * r:
            return math.pi * r**2
        elif d <= 0:
            return 0
        else:
            part1 = r**2 * math.acos(d / (2 * r))
            part3 = 0.5 * d * math.sqrt(4 * r**2 - d**2)
            overlap_area = 2 * part1 - part3
            return math.pi * r**2 - overlap_area
        
    def calculate_new_explored_area_points(self):
        map_exploration = self.normalizeExplorationMap(self.grid.get_point_exploration_map())
        
        # Conta le celle con valore = 1
        actual_num_unexplored_area_points = (map_exploration == 1).sum()
        
        self.new_explored_area_points = self.last_unexplored_area_points - actual_num_unexplored_area_points
        self.last_unexplored_area_points = actual_num_unexplored_area_points
        
    def calculate_voronoi_partition(self):
        """
        Calcola la partizione di Voronoi per gli UAV in modo vettorializzato.
        Restituisce una matrice 2D in cui ogni cella contiene l'indice dell'UAV più vicino.
        """

        # Ottieni le posizioni 2D degli UAV come array NumPy
        points = np.array([uav.position.to_array_2d() for uav in self.uav])  # Shape: (N, 2)

        # Crea una griglia di coordinate (meshgrid)
        grid_x, grid_y = np.meshgrid(np.arange(self.grid.grid_width), np.arange(self.grid.grid_height))
        grid_points = np.stack((grid_y, grid_x), axis=-1)  # Shape: (H, W, 2)

        # Calcola le distanze Euclidee da ogni punto della griglia a ciascun UAV
        # points[:, np.newaxis, np.newaxis, :] ha shape (N, 1, 1, 2)
        # grid_points ha shape (H, W, 2)
        # Differenza broadcasting: (N, H, W, 2)
        diff = points[:, np.newaxis, np.newaxis, :] - grid_points
        dists = np.linalg.norm(diff, axis=-1)  # Shape: (N, H, W)

        # Trova per ogni cella della griglia l'indice dell'UAV più vicino
        partition = np.argmin(dists, axis=0)  # Shape: (H, W)

        # Salva la partizione
        self.voronoi_partition = partition
        
        
    def calculate_boundary_repulsive_potential(self, uav):
        """
        Calcola un potenziale artificiale (penalità) per scoraggiare gli UAV 
        dall'avvicinarsi troppo ai bordi della mappa.

        La penalità aumenta linearmente man mano che l'UAV si avvicina al bordo,
        con inizio a una distanza pari alla velocità massima + 1.

        Args:
            uav (UAV): oggetto UAV contenente le coordinate (x, y).

        Returns:
            float: penalità tra 0 (lontano dai bordi) e 1 (sul bordo).
        """
        penalty = 0
        h = uav.position.z_coordinate
        R = self.communication_channel.get_distance_from_SINR_closed_form(self.covered_threshold, 0)

        if R > h:  # Solo se esiste una proiezione al suolo
            margin = math.ceil(math.sqrt(R**2 - h**2))
        x, y = uav.position.x_coordinate, uav.position.y_coordinate

        distances = np.array([x, self.grid.grid_width - x, y, self.grid.grid_height - y])
        min_distance = np.min(distances)
        
        if min_distance < margin:
            penalty += ((margin - min_distance) / margin) ** 3
           
        return penalty
    
    def calculate_uav_repulsive_potential(self, uav_i, uav_j):
        """
        Calcola un potenziale repulsivo tra due UAV per scoraggiare la vicinanza eccessiva.

        La penalità è nulla se i due UAV sono più distanti di `min_dist`.
        Quando si avvicinano sotto `min_dist`, la penalità cresce quadraticamente fino a raggiungere il valore massimo (1.0) 
        se la distanza scende sotto `self.collision_distance`.

        Args:
            uav_i (UAV): primo UAV.
            uav_j (UAV): secondo UAV.

        Returns:
            float: penalità tra 0 (nessun conflitto) e 1.0 (collisione imminente).
        """
        penalty = 0
        uav_distance = uav_i.position.calculate_distance_to_coordinate(uav_j.position)
        h_i = uav_i.position.z_coordinate
        h_j = uav_j.position.z_coordinate
        R = self.communication_channel.get_distance_from_SINR_closed_form(self.covered_threshold, 0)
        if R > h_i:  # Solo se esiste una proiezione al suolo
            margin_i = math.ceil(math.sqrt(R**2 - h_i**2))
        if R > h_j:  # Solo se esiste una proiezione al suolo
            margin_j = math.ceil(math.sqrt(R**2 - h_j**2))
        min_dist = margin_i + margin_j # or self.theoretical_max_distance_before_possible_collision
        
        if uav_distance > min_dist:
            penalty = 0.0
        elif uav_distance <= self.collision_distance:
            penalty = 1.0  # Penalità massima
        else:
            # Normalizza la distanza in [0, 1] tra margin e min_dist
            scaled = (min_dist - uav_distance) / (min_dist - self.collision_distance)
            penalty = scaled**2 # Quadratica: penalizza più severamente da vicino
        return penalty

    
    def calculate_RCR_without_uav_i(self, i):
        tmp_matrix = np.delete(self.connectivity_matrix, i, axis=1)  # Remove i-th column
        return np.sum(np.any(tmp_matrix, axis=1))
    
    def calculate_reward(self, terminated):

        num_uav = len(self.uav)
        individual_rewards = [0.0 for _ in range(self.max_uav_number)]

        # 1. Penalità per avvicinamento ai bordi e agli altri uav
        self.boundary_penalty_total = 0.0
        self.collision_penalty_total = 0.0
        for i in range(num_uav):
            if self.w_boundary_penalty > 0.0:
                boundary_penalty = self.calculate_boundary_repulsive_potential(self.uav[i])
                self.boundary_penalty_total -= self.w_boundary_penalty * boundary_penalty
                #afar_boundary_incentive = 1.0 - boundary_penalty
                individual_rewards[i] -= self.w_boundary_penalty * boundary_penalty

            # Collisioni
            if self.w_collision_penalty > 0.0:
                for j in range(i + 1, num_uav):
                    collision_penalty = self.calculate_uav_repulsive_potential(self.uav[i], self.uav[j])
                    #afar_collision_incentive = 1.0 - collision_penalty
                    self.collision_penalty_total -= 2 * self.w_collision_penalty * collision_penalty
                    individual_rewards[i] -= self.w_collision_penalty * collision_penalty
                    individual_rewards[j] -= self.w_collision_penalty * collision_penalty
        
        

        # 2. Copertura spaziale (evita UAV sovrapposti) - volendo potrebbe essere reward individuale se altezze uav fossero diverse
        self.spatial_coverage_total = 0.0
        if self.max_theoretical_ground_uavs_points_coverage > 0 and self.w_spatial_coverage > 0.0:
            for i in range(num_uav): 
                uav_coverage = (self.uav_total_covered_points[i] - self.uav_shared_covered_points[i]) / int(self.max_theoretical_ground_uavs_points_coverage / num_uav)
                '''
                #reward individuale per copertura spaziale tipo 1
                if uav_coverage > self.spatial_coverage_threshold:
                    self.spatial_coverage_total += 0
                    individual_rewards[i] += 0
                else:
                    self.spatial_coverage_total -= self.w_spatial_coverage * (1 - uav_coverage)
                    individual_rewards[i] -= self.w_spatial_coverage * (1 - uav_coverage)
                '''    
                #reward individuale per copertura spaziale tipo 2
                '''
                self.spatial_coverage_total += w_spatial * uav_coverage
                individual_rewards[i] += w_spatial * uav_coverage
                '''
                # reward globale per copertura spaziale tipo 1
            
                if uav_coverage > self.spatial_coverage_threshold:
                    self.spatial_coverage_total += 0
                else:
                    self.spatial_coverage_total -= (1 - uav_coverage)
            for i in range(num_uav):
                individual_rewards[i] += self.w_spatial_coverage * self.spatial_coverage_total / num_uav
            self.spatial_coverage_total *= self.w_spatial_coverage  # Normalizza il reward globale per copertura spaziale    
               
            '''   
                # reward globale per copertura spaziale tipo 2
                self.spatial_coverage_total -= (1 - uav_coverage)
            for i in range(num_uav):
                individual_rewards[i] += self.w_exploration * self.spatial_coverage_total / num_uav
            '''
        
        # 3. Incentivo all'esplorazione (bassa densità di esplorazione)
        self.exploration_incentive_total = 0.0
        # Calcola il numero totale di celle
        if self.w_exploration > 0.0 and self.exploration_phase:
            map_exploration = self.normalizeExplorationMap(self.grid.get_point_exploration_map())
            total_area_points = self.grid.grid_width * self.grid.grid_height

            num_explored_area_points = total_area_points - self.last_unexplored_area_points
            
            explored_area_points_incentive = num_explored_area_points / total_area_points if total_area_points > 0 else 0.0

            new_explored_area_points_incentive = (1 - explored_area_points_incentive) * self.new_explored_area_points / self.max_theoretical_new_explored_ground_uavs_points # incentivo per esplorazione efficaciemente distribuita
            
            balanced_exploration_incentive = 1 - np.mean(map_exploration)
            
            # Controllo per vedere se si è passati alla fase di copertura
            if self.reward_mode == "twophases":
                exploration_incentive = (explored_area_points_incentive + new_explored_area_points_incentive + balanced_exploration_incentive) / 2 
                if explored_area_points_incentive >= self.exhaustive_exploration_threshold and balanced_exploration_incentive >= self.balanced_exploration_threshold:
                    self.exploration_phase = False
                    self.steps_gu_coverage_phase = self.max_steps_gu_coverage_phase
                else:
                    self.exploration_incentive_total = self.w_exploration * exploration_incentive * num_uav
                    for i in range(num_uav):
                        individual_rewards[i] += self.w_exploration * exploration_incentive
            elif self.reward_mode == "mixed":
                exploration_incentive = (balanced_exploration_incentive)
                self.exploration_incentive_total = self.w_exploration * exploration_incentive * num_uav
                for i in range(num_uav):
                    individual_rewards[i] += self.w_exploration * exploration_incentive
            
                


        # 4. Incentivo per partizione di voronoi omogenea.
        self.homogenous_voronoi_partition_incentive_total = 0.0
        # Calcola il numero ideale di punti per UAV in una partizione equa
        if self.w_homogenous_voronoi_partition > 0.0 and self.exploration_phase:
            counts = np.bincount(self.voronoi_partition.flatten(), minlength=num_uav)
            ideal_area_points = total_area_points / num_uav
            
            # Deviazione assoluta media
            mad = np.mean(np.abs(counts - ideal_area_points))

            # Normalizzazione: massimo possibile MAD = total_area_points * (1 - 1/num_uav)
            max_mad = ideal_area_points * (1 - 1 / num_uav)

            # Normalizzazione: 0 = perfetta, -1 = perfetta
            homogenous_voronoi_partition_incentive = -(mad / max_mad)
            homogenous_voronoi_partition_incentive = np.clip(homogenous_voronoi_partition_incentive, -1, 0)

            self.homogenous_voronoi_partition_incentive_total = self.w_homogenous_voronoi_partition * homogenous_voronoi_partition_incentive * num_uav
            
            for i in range(num_uav):
                individual_rewards[i] += self.w_homogenous_voronoi_partition * homogenous_voronoi_partition_incentive
        
        # 5. Copertura dei GU massima
        self.gu_coverage_total = 0
        
        '''
        contributions = [
            max(0.0, self.gu_covered - self.calculate_RCR_without_uav_i(i))
            for i in range(len(self.uav))
        ]
        sum_contributions = sum(contributions)
        if sum_contributions > 0:
            proportions = [c / sum_contributions for c in contributions]
        else:
            proportions = [1.0 / num_uav] * num_uav
        '''    
        if self.reward_mode == "twophases":    
            if self.steps_gu_coverage_phase > 0 and not self.exploration_phase:
                coverage_score = self.w_exploration + (self.gu_covered / np.sum(self.gu_mask)) if np.sum(self.gu_mask) > 0 else 0.0
                self.steps_gu_coverage_phase -= 1
                if self.steps_gu_coverage_phase == 0:
                    self.exploration_phase = True
            else:
                coverage_score = 0.0
        elif self.reward_mode == "mixed":
            if self.tradeoff_mode=="exponential":
                tradeoff_factor = np.exp(self.k_factor * exploration_incentive)-1 # e^(kx)-1
            elif self.tradeoff_mode=="exponential_norm":
                tradeoff_factor = (np.exp(self.k_factor * exploration_incentive)-1)/(np.exp(self.k_factor)-1) # (e^(kx)-1)/(e^k-1)
            elif self.tradeoff_mode=="power_law":
                tradeoff_factor = math.pow(exploration_incentive, self.k_factor) # x^k
            else:
                tradeoff_factor = exploration_incentive
            coverage_score = tradeoff_factor * (self.gu_covered / np.sum(self.gu_mask)) if np.sum(self.gu_mask) > 0 else 0.0
            
        gu_coverage = self.w_gu_coverage * coverage_score
        
        # Reward globale per copertura GU
        for i in range(num_uav):
            individual_rewards[i] += gu_coverage
            self.gu_coverage_total += gu_coverage
        
        '''
        # Reward individuale per copertura GU
        for i in range(num_uav):
            individual_rewards[i] += (proportions[i] * gu_coverage) * num_uav
            self.gu_coverage_total += (proportions[i] * gu_coverage) * num_uav
        '''    
        
        #5. Penalità per UAV terminati
        '''
        for i in range(num_uav):
            if terminated[i]:
                individual_rewards[i] = -1.0
        '''

            

        '''
        # Log rewards for debugging  
        self.log_rewards(
            contributions, entropy_contribution_score, spatial_coverage,
            exploration_incentive, coverage_score,
            boundary_penalty, collision_penalty, global_reward, terminated
        )
        '''
        return individual_rewards
    
    
    # CHECKS
    '''
    #TODO ripensarla con i GU index
    def check_if_disappear_GU(self):
        index_to_remove = []
        for i in range(len(self.gu)):
            sample = random.random()
            if sample <= self.disappear_gu_prob:
                index_to_remove.append(i)
        index_to_remove = sorted(index_to_remove, reverse=True)

        if index_to_remove != []:
            print("GU disappeared: ", index_to_remove)
            print("self.gu number: ", len(self.gu))

        for index in index_to_remove:
            del self.gu[index]

    #TODO ripensarla con i GU index
    def check_if_spawn_new_GU(self):
        sample = random.random()
        for _ in range(4):
            if sample <= self.spawn_gu_prob:
                area = self.np_random.choice(self.grid.spawn_area)
                x_coordinate = int(self.np_random.uniform(area[0][0], area[0][1]))
                y_coordinate = int(self.np_random.uniform(area[1][0], area[1][1]))
                position = self.grid.get_point(x_coordinate, y_coordinate)
                gu = GU(position)
                self.init_channel(gu)
                self.gu.append(gu)
        # update disappear gu probability
        self.disappear_gu_prob = self.spawn_gu_prob * 4 / len(self.gu)
    '''
    
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
        area = self.np_random.choice(self.grid.available_area)
        for i, uav in enumerate(self.uav):
            '''
            if not uav.position.is_in_area(area) or self.check_collision(i, uav):
                terminated_matrix.append(True)
            else:
                terminated_matrix.append(False)
            '''
            terminated_matrix.append(False)
        
        # Aggiungi False per gli UAV inattivi (padding)
        padding = self.max_uav_number - len(self.uav)
        terminated_matrix.extend([False] * padding)
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
        # Cartella delle immagini, relativa a questo file
        image_dir = os.path.join(os.path.dirname(__file__), "..", "images")
        # CANVAS
        canvas.fill((255,255,255))
        
        for row in range(len(self.grid.pixel_grid)):
            for col in range(len(self.grid.pixel_grid[0])):
                pixel = self.grid.pixel_grid[row][col]
                color = pixel.color  # Assumendo sia una tupla (R, G, B)
                canvas.set_at((col, self.window_height - 1 - row), color)

        # WALL
        '''
        for wall in self.world:
            pygame.draw.line(canvas,
                             Color.BLACK.value,
                             self.convert_point(wall.start),
                             self.convert_point(wall.end),
                             self.wall_width)
        '''

        # Disegna i GU (Ground Units) con la loro immagine
        for gu in self.gu:
            gu_image_path = os.path.join(image_dir, gu.getImage())
            gu_image = pygame.image.load(gu_image_path)
            canvas.blit(gu_image, self.image_convert_point(gu.position))

        # Disegna i UAV (droni)
        drone_image_path = os.path.join(image_dir, "drone30.png")
        drone_image = pygame.image.load(drone_image_path)
        for uav in self.uav:
            canvas.blit(drone_image, self.image_convert_point(uav.position))

    def convert_point(self, point: Coordinate) -> Tuple[int, int]:
        pygame_x = (round(point.x_coordinate)
                    + self.x_offset)
        pygame_y = (self.window_height
                    - round(point.y_coordinate)
                    + self.y_offset)
        return pygame_x, pygame_y

    def image_convert_point(self, point: Coordinate) -> Tuple[int, int]:
        shiftX = 15
        shiftY = 15
        pygame_x = (round(point.x_coordinate) - shiftX + self.x_offset)
        pygame_y = (self.window_height - round(point.y_coordinate) - shiftY + self.y_offset)
        return pygame_x, pygame_y

    
    # INFO METHODS
    def create_info(self, terminated) -> dict:
        collision = False
        out_area = False
        area = self.np_random.choice(self.grid.available_area)
        for i, uav in enumerate(self.uav):
            if self.check_collision(i, uav):
                collision = True
                break
            if not uav.position.is_in_area(area):
                out_area = True
                break
        
        RCR = str(self.gu_covered/len(self.gu))
        
        if self.options["test"]:
            return {"GU coperti": str(self.gu_covered), "Ground Users": str(
                len(self.gu)), "RCR": RCR, "Collision": collision, "Out_Area" : out_area, 
                "boundary_penalty_total": self.boundary_penalty_total, 
                "collision_penalty_total": self.collision_penalty_total,
                "spatial_coverage_total": self.spatial_coverage_total,
                "exploration_incentive_total": self.exploration_incentive_total,
                "homogenous_voronoi_partition_incentive_total": self.homogenous_voronoi_partition_incentive_total,
                "gu_coverage_total": self.gu_coverage_total}
            
        return {"GU coperti": str(self.gu_covered), "Ground Users": str(
            len(self.gu)), "RCR": RCR, "Collision": collision, "Out_Area" : out_area}


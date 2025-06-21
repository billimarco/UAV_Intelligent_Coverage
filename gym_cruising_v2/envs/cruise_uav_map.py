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
    
    low_observation: float
    high_observation: float

    gu_covered = 0
    max_gu_covered = 0
    max_theoretical_ground_uavs_points_coverage = 0
    #total_uavs_point_coverage = 0
    #simultaneous_uavs_points_coverage = 0
    uav_total_covered_points = []
    uav_shared_covered_points = []

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

        self.available_gu_indexes = [False]*args.max_gu_number

        self.communication_channel = CommunicationChannel(args)
        
        self.theoretical_max_distance_before_possible_collision = 2*math.sqrt(self.max_speed_uav**2 + self.max_speed_uav**2) + self.collision_distance
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
            
        self.uav = []
        self.gu = []
        self.uav_number = self.options["uav"]
        self.starting_gu_number = self.options["gu"]
        self.available_gu_indexes = [False]*self.max_gu_number
        #RAND
        #self.disappear_gu_prob = self.spawn_gu_prob * 4 / self.starting_gu_number
        self.gu_covered = 0
        self.max_gu_covered = 0
        #self.total_uavs_point_coverage = 0
        #self.simultaneous_uavs_points_coverage = 0
        self.uav_total_covered_points = []
        self.uav_shared_covered_points = []
        self.grid.reset()
        self.reset_observation_action_space()
        np.random.seed(self.seed)
        return super().reset(seed=self.seed, options=self.options)
    
    def reset_gu(self, options: Optional[dict] = None) -> np.ndarray:
        self.gu = []
        self.available_gu_indexes = [False]*self.max_gu_number
        #RAND
        #self.disappear_gu_prob = self.spawn_gu_prob * 4 / self.starting_gu_number
        self.gu_covered = 0
        self.max_gu_covered = 0
        if not options['clustered']:
            self.init_gu()
        else:
            self.init_gu_clustered(options)
            
        self.calculate_uav_area_max_coverage()
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

        self.calculate_uav_area_max_coverage()
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
                gu = GU(i*gu_for_cluster+j,position)
                self.available_gu_indexes[i*gu_for_cluster+j] = True
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

        # Inizializza lo stato degli utenti coperti e la relativa maschera (padding)
        covered_users_states = np.zeros((self.max_gu_number, 2), dtype=np.float64)
        gu_mask = np.zeros(self.max_gu_number, dtype=bool)

        # Raccogli le posizioni normalizzate solo degli utenti coperti
        covered_users_positions = []
        covered_users_ids =  []
        for gu in self.gu:
            if gu.covered:
                covered_users_positions.append(self.normalizePositions(gu.position))
                covered_users_ids.append(gu.id)

        # Copia le posizioni normalizzate nella matrice di output fino al massimo consentito
        for i, ids in enumerate(covered_users_ids):
            covered_users_states[ids] = covered_users_positions[i]
            gu_mask[ids] = True  # marca l’utente come coperto/attivo

        # Le righe da num_to_copy a max_gu_number rimangono zero e sono marcate come inattive nella maschera (padding)

        # Crea il dizionario di osservazione da restituire
        observation = {
            "map_exploration_states": map_exploration,
            "uav_states": uav_states,
            "uav_mask": uav_mask,
            "uav_flags": uav_flags,
            "covered_users_states": covered_users_states,
            "gu_mask": gu_mask
        }

        return observation

    
    
    # DO ACTIONS    
    def perform_action(self, actions) -> None:
        self.move_UAV(actions)
        self.update_GU()
        
        self.calculate_uav_area_max_coverage()
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

        # Inizializza liste per statistiche individuali di copertura
        self.uav_total_covered_points = [0 for _ in range(len(self.uav))]
        self.uav_shared_covered_points = [0 for _ in range(len(self.uav))]

        # Per ogni UAV, calcola i punti coperti e quelli condivisi
        for i in range(len(self.uav)):
            uav_coverage = coverage_mask[:, i]  # punti coperti da questo UAV
            total_covered_by_uav = sum(uav_coverage)
            shared_covered_by_uav = sum(uav_coverage & (coverage_counts > 1))
            
            self.uav_total_covered_points[i] = total_covered_by_uav
            self.uav_shared_covered_points[i] = shared_covered_by_uav
        
        '''
        # Inizializza le statistiche di copertura
        total_coverage = 0
        simultaneous_coverage = 0
        '''

        # Aggiorna lo stato di ogni punto sulla base del numero di UAV che lo coprono
        for i, point in enumerate(point_refs):
            count = coverage_counts[i]
            if count > 0:
                point.set_covered(True)
                point.reset_step_from_last_visit()
                '''
                total_coverage += count
                if count > 1:
                    simultaneous_coverage += count-1
                '''
            else:
                point.set_covered(False)
                point.increment_step_from_last_visit()

        '''
        # Salva le statistiche calcolate
        self.total_uavs_point_coverage = total_coverage
        self.simultaneous_uavs_points_coverage = simultaneous_coverage
        '''

        # Aggiorna metriche per pixel solo se in modalità "human" (visualizzazione)
        if self.render_mode == "human":
            for pixel_row in self.grid.pixel_grid:
                for pixel in pixel_row:
                    pixel.calculate_mean_step_from_last_visit()

    def calculate_uav_area_max_coverage(self):
        max_ground_coverage_points = 0

        for uav in self.uav:
            h = uav.position.z_coordinate
            R = self.communication_channel.get_distance_from_SINR_closed_form(self.covered_threshold, 0)

            if R > h:  # Solo se esiste una proiezione al suolo
                r_ground = math.ceil(math.sqrt(R**2 - h**2))
                area = math.pi * r_ground**2
                max_ground_coverage_points += math.ceil(area)

        self.max_theoretical_ground_uavs_points_coverage = max_ground_coverage_points
        
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
        penalty = -1
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
        penalty = -1
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
            penalty = 0
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
        w_boundary_penalty = 1.0
        w_collision_penalty = 0.0
        w_spatial = 0.0
        w_explore = 0.0
        w_coverage = 0.0
        
        num_uav = len(self.uav)
        individual_rewards = [0.0 for _ in range(self.max_uav_number)]
        
        
        # 1. Penalità per avvicinamento ai bordi e agli altri uav
        for i in range(num_uav):
            boundary_penalty = self.calculate_boundary_repulsive_potential(self.uav[i])
            #afar_boundary_incentive = 1.0 - boundary_penalty
            individual_rewards[i] -= w_boundary_penalty * boundary_penalty

            # Collisioni
            for j in range(i + 1, num_uav):
                collision_penalty = self.calculate_uav_repulsive_potential(self.uav[i], self.uav[j])
                #afar_collision_incentive = 1.0 - collision_penalty
                individual_rewards[i] -= w_collision_penalty * collision_penalty
                individual_rewards[j] -= w_collision_penalty * collision_penalty
            
            individual_rewards[i] = individual_rewards[i] / num_uav if individual_rewards[i] >= 0 else 0.0

        # 2. Copertura spaziale (evita UAV sovrapposti) - volendo potrebbe essere reward individuale se altezze uav fossero diverse
        spatial_coverage_total = 0
        if self.max_theoretical_ground_uavs_points_coverage > 0 and w_spatial > 0.0:
            for i in range(num_uav):
                uav_coverage = (self.uav_total_covered_points[i] - self.uav_shared_covered_points[i]) / int(self.max_theoretical_ground_uavs_points_coverage / num_uav)
                spatial_coverage_total += w_spatial * uav_coverage
                individual_rewards[i] += w_spatial * uav_coverage
 
        spatial_coverage = spatial_coverage_total / num_uav if num_uav > 0 else 0.0
            
        coverage_threshold = 0.95  # soglia di copertura spaziale per incentivare
        
        # 3. Incentivo all'esplorazione (bassa densità di esplorazione)
        map_exploration = self.normalizeExplorationMap(self.grid.get_point_exploration_map())
        
        exploration_incentive = 1 - np.mean(map_exploration)
        
        for i in range(num_uav):
            individual_rewards[i] += w_explore * exploration_incentive / num_uav
        
        exploration_threshold = 0.6
        
        # TODO Potrebbe essere un idea mettere dei booleani in self per indicare se siamo in fase esplorativa o in fase di copertura
        
        # 4. Copertura dei GU massima
        contributions = [
            max(0.0, self.gu_covered - self.calculate_RCR_without_uav_i(i))
            for i in range(len(self.uav))
        ]
        sum_contributions = sum(contributions)
        if sum_contributions > 0:
            proportions = [c / sum_contributions for c in contributions]
        else:
            proportions = [1.0 / num_uav] * num_uav
            
        if exploration_incentive >= exploration_threshold:
            coverage_score = self.gu_covered / self.max_gu_covered if self.max_gu_covered > 0 else 0.0
        else:
            coverage_score = 0.0  # ignoriamo copertura se la mappa è troppo inesplorata
        global_coverage = w_coverage * coverage_score
        
        for i in range(num_uav):
            individual_rewards[i] += (proportions[i] * global_coverage)
        
        #5. Penalità per UAV terminati
        '''
        for i in range(num_uav):
            if terminated[i]:
                individual_rewards[i] = -1.0
        '''

            

        '''  
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
                self.initialize_channel(gu)
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
            if not uav.position.is_in_area(area) or self.check_collision(i, uav):
                terminated_matrix.append(True)
            else:
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
        return {"GU coperti": str(self.gu_covered), "Ground Users": str(
            len(self.gu)), "RCR": RCR, "Collision": collision, "Out_Area" : out_area}

    def log_rewards(self, contributions, entropy_score, spatial_coverage, exploration_incentive,
                coverage_score, boundary_penalty, collision_penalty, global_reward, terminated):
        """
        Logga i componenti del reward, includendo se almeno un UAV è terminato.
        """
        print("🔍 [Reward Breakdown]")
        print(f"  ➤ UAV Contributions: {['{:.3f}'.format(c) for c in contributions]}")
        print(f"  ➤ Entropy Score: {entropy_score:.3f}")
        print(f"  ➤ Spatial Coverage: {spatial_coverage:.3f}")
        print(f"  ➤ Exploration Incentive: {exploration_incentive:.3f}")
        print(f"  ➤ GU Coverage Score: {coverage_score:.3f}")
        print(f"  ➤ Boundary Penalty: -{boundary_penalty:.3f}")
        print(f"  ➤ Collision Penalty: -{collision_penalty:.3f}")
        print(f"  ✅ Total Global Reward: {global_reward:.3f}")
        print(f"  ⚠️  Any UAV Terminated: {any(terminated)}")

""" This module contains the general cruise class. """

from abc import abstractmethod
from copy import deepcopy
from typing import Tuple, List

import random
import numpy as np
import pygame
from gymnasium import Env
from pygame import Surface

from gym_cruising_v2.geometry.grid import Grid
from gym_cruising_v2.geometry.line import Line


class Cruise(Env):
    """
    This class is used to define the step, reset, render, close methods
    as template methods. In this way we can create multiple environments that
    can inherit from one another and only redefine certain methods.
    """

    grid: Grid
    world: Tuple[Line, ...]
    seed : int
    options : dict

    metadata = {"render_modes": ["human"], "render_fps": 8}

    def __init__(self, args, render_mode=None) -> None:
        self.args = args
        
        self.window_width = args.window_width  # The width size of the PyGame window
        self.window_height = args.window_height  # The height size of the PyGame window
        self.unexplored_point_max_steps = args.unexplored_point_max_steps
        self.grid = Grid(args.window_width, args.window_height, args.resolution, args.spawn_offset, args.unexplored_point_max_steps, render_mode)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        
        self.resolution = args.resolution
        self.x_offset = args.x_offset
        self.y_offset = args.y_offset
        self.wall_width = args.wall_width
        
        self.seed = random.randint(0, 10000)
        self.options = args.options
        
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, dict]:

        if seed is not None:
            self.seed = seed

        if options is not None:
            self.options = options

        super().reset(seed=self.seed, options=self.options)
        np.random.seed(self.seed)

        self.init_environment(options=self.options)

        observation = self.get_observation()
        '''
        print("Returned obs:")
        for k, v in observation.items():
            print(f"{k}: shape={v.shape}, dtype={v.dtype}")
            print("Expected:", self.observation_space[k])
        '''
        assert self.observation_space.contains(observation)
        terminated = self.check_if_terminated()
        info = self.create_info(terminated)

        if self.render_mode == "human":
            self.render_frame()

        return observation, info

    def step(self, actions) -> Tuple[np.ndarray, List, bool, bool, dict]:

        assert self.action_space.contains(actions)

        self.perform_action(actions)

        state = self.get_observation()
        
        '''
        state["map_exploration_states"] = np.clip(state["map_exploration_states"], 0.0, 1.0)
        state["uav_states"] = np.clip(state["uav_states"], -1.0, 1.0)
        state["covered_users_states"] = np.clip(state["covered_users_states"], -1.0, 1.0)
        state["uav_mask"] = state["uav_mask"].astype(bool)
        state["gu_mask"] = state["gu_mask"].astype(bool)
        state["uav_flags"] = state["uav_flags"].astype(bool)
        # Controllo dettagliato per ciascun componente
        for key, value in state.items():
            space = self.observation_space.spaces[key]
            if not space.contains(value):
                print(f"Componente '{key}' fuori dai limiti:")
                print(f"Valore minimo osservato: {np.min(value)}")
                print(f"Valore massimo osservato: {np.max(value)}")
                print(f"Limite inferiore consentito: {space.low}")
                print(f"Limite superiore consentito: {space.high}")
                print(f"Tipo di dato atteso: {space.dtype}, tipo di dato effettivo: {value.dtype}")
                raise AssertionError(f"'{key}' non rispetta i limiti definiti in observation_space.")

        
        print("Returned state:")
        for k, v in state.items():
            print(f"{k}: shape={v.shape}, dtype={v.dtype}")
            print("Expected:", self.observation_space[k])
        '''
        assert self.observation_space.contains(state)
        terminated = self.check_if_terminated()
        truncated = self.check_if_truncated()
        info = self.create_info(terminated)
        reward = self.calculate_reward(terminated)
        
        info["terminated_per_uav"] = np.asarray(terminated, dtype=bool)
        info["reward_per_uav"] = np.asarray(reward, dtype=np.float32)

        if self.render_mode == "human":
            self.render_frame()

        return state, float(np.mean(reward)), False, truncated, info #impedisce all'ambiente di terminare anche se fa degli errori. Mi serve per impedire il crash dell'AsyncVectorEnv
    
    def render(self):
        return None

    def render_frame(self) -> None:
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_width, self.window_height))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        if self.window is None or self.clock is None:
            return

        canvas = Surface((self.window_width, self.window_height))
        # Draw the canvas
        self.draw(canvas)

        if self.render_mode == "human":
            # The following line copies our drawings from canvas to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            # We need to ensure that human-rendering occurs at the predefined framerate.
            self.clock.tick(self.metadata["render_fps"])
    
    def close(self) -> None:
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    @abstractmethod
    def perform_action(self, actions) -> None:
        pass

    @abstractmethod
    def get_observation(self) -> np.ndarray:
        pass

    @abstractmethod
    def check_if_terminated(self):
        pass

    @abstractmethod
    def calculate_reward(self, terminated):
        pass

    @abstractmethod
    def create_info(self, terminated) -> dict:
        pass

    @abstractmethod
    def init_environment(self, options=None) -> None:
        pass

    @abstractmethod
    def draw(self, canvas: Surface) -> None:
        pass

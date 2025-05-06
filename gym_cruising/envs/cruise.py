""" This module contains the general cruise class. """

from abc import abstractmethod
from copy import deepcopy
from typing import Tuple, List

import numpy as np
import pygame
from gymnasium import Env
from pygame import Surface

from gym_cruising.enums.grid import Grid
from gym_cruising.geometry.line import Line


class Cruise(Env):
    """
    This class is used to define the step, reset, render, close methods
    as template methods. In this way we can create multiple environments that
    can inherit from one another and only redefine certain methods.
    """

    grid: Grid
    world: Tuple[Line, ...]

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 8}

    def __init__(self, args, render_mode=None) -> None:
        self.window_width = args.window_width  # The width size of the PyGame window
        self.window_height = args.window_height  # The height size of the PyGame window
        self.grid = Grid(args.window_width, args.window_height, args.resolution, args.spawn_offset)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        
        self.resolution = args.resolution
        self.x_offset = args.x_offset
        self.y_offset = args.y_offset
        self.wall_width = args.wall_width

    def step(self, actions) -> Tuple[np.ndarray, List, bool, bool, dict]:

        assert self.action_space.contains(actions)

        self.perform_action(actions)

        state = self.get_observation()
        terminated = self.check_if_terminated()
        truncated = self.check_if_truncated()
        info = self.create_info(terminated)
        reward = self.calculate_reward(terminated)

        if self.render_mode == "human":
            self.render_frame()

        return state, reward, any(terminated), truncated, info

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

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, dict]:

        super().reset(seed=seed, options=options)
        np.random.seed(seed)
        self.world = deepcopy(self.grid.walls)

        self.init_environment(options=options)

        observation = self.get_observation()
        terminated = self.check_if_terminated()
        info = self.create_info(terminated)

        if self.render_mode == "human":
            self.render_frame()

        return observation, info

    @abstractmethod
    def init_environment(self, options=None) -> None:
        pass

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

    @abstractmethod
    def draw(self, canvas: Surface) -> None:
        pass

    def close(self) -> None:
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from gym_cruising_v2.geometry.grid import Grid
import gym_cruising_v2.utils.runtime_utils as utils
import numpy as np

if __name__ == "__main__":
    args = args = utils.parse_args()
    grid = Grid(args.window_width, args.window_height, args.resolution, args.spawn_offset, args.unexplored_point_max_steps)
    rng = np.random.default_rng(2)
    spawn_area = rng.choice(grid.spawn_area)
    (x_min, x_max), (y_min, y_max) = spawn_area
    print(x_min)
    print(x_max)
    print(y_min)
    print(y_max)
    area = spawn_area
    print(area[0][1])
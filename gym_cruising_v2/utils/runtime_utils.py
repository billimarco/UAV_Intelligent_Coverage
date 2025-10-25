import os
import subprocess
import argparse
from distutils.util import strtobool

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="mixed180-3UAV-powerlinear-uniform-Randomgu",
                        help="the name of this experiment")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
                        help="the learning rate of the optimizer")# 1.0e-3 lr, 2.5e-4 default, 1.0e-4 lrl, 2.5e-5 lrl--
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, the learning rate will be annealed")
    parser.add_argument("--seed", type=int, default=9,
                        help="seed of the experiment")
    parser.add_argument("--random-seed", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, the seed is random")
    parser.add_argument("--options", type=dict, default={},
                        help="the options for the environment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--cuda-device", type=int, default=0,
                        help="if cuda is enabled, this is the device to use (0 by default)")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--offline", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="Tesi",
                        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default="marcolbilli-universit-di-firenze",
                        help="the entity (team) of wandb's project")
    
    
    # Algorithm specific arguments
    parser.add_argument("--render-mode", type=str, default=None, 
                        help="Render mode (e.g., human or None")
    parser.add_argument("--train", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, the training will be performed")
    parser.add_argument("--use-trained", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="If set, loads and runs a pre-trained model")
    parser.add_argument("--numerical-test", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="numerical test using trained model")
    
    
    # NN specific arguments
    parser.add_argument("--embedded-dim", type=int, default=64,
                    help="Size of the embedding vector used to represent agent observations or features")
    parser.add_argument("--patch-size", type=int, default=50,
                    help="Size of map patches (possibly a map length divider in all directions)")
    parser.add_argument("--global-value", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, you use only one global value for all agents")
    
    
    # UAV specific arguments
    parser.add_argument("--max-uav-number", type=int, default=3,
                        help="the max number of UAVs in the environment")
    parser.add_argument("--uav-number", type=int, default=3,
                        help="the number of UAVs in the environment (not more than max-uav-number)")
    parser.add_argument("--max-speed-uav", type=float, default=50.0,
                        help="maximum speed in a single direction of an UAV in meters per second")
    parser.add_argument("--uav-altitude", type=float, default=500,
                    help="UAV flight altitude in meters")
    parser.add_argument("--minimum-starting-distance-between-uav", type=float, default=110.0,
                    help="minimum initial distance between UAVs in meters")
    parser.add_argument("--collision-distance", type=float, default=5.0,
                        help="minimum distance between UAVs before a collision is considered (in meters)")
    
    
    # GU specific arguments
    parser.add_argument("--max-gu-number", type=int, default=100,
                        help="the max number of GUs in the environment")
    parser.add_argument("--min-gu-number", type=int, default=20,
                        help="the min number of GUs in the environment (only for gu-number-random True)")
    parser.add_argument("--starting-gu-number", type=int, default=100,
                        help="the number of starting ground units in the environment")
    parser.add_argument("--gu-max-speed", type=float, default=4.00,
                        help="max speed of ground units in meters per second")
    parser.add_argument("--spawn-gu-prob", type=float, default=0.0005,
                        help="probability of spawning a ground unit per cell or timestep")
    parser.add_argument("--covered-threshold", type=float, default=10.0,
                        help="SINR threshold (in dB) above which a ground user is considered covered")
    
    
    # Grid specific arguments
    parser.add_argument("--window-width", type=int, default=500,
                        help="the width size of the PyGame window")
    parser.add_argument("--window-height", type=int, default=500,
                        help="the height size of the PyGame window")
    parser.add_argument("--resolution", type=int, default=1,
                    help="meters for every pixel side (num of points for pixel side)")
    parser.add_argument("--uav-spawn-offset" , type=int, default=105,
                        help="the spawn offset for UAV of the environment in point")
    parser.add_argument("--gu-spawn-offset" , type=int, default=10,
                        help="the spawn offset for GU of the environment in point")
    parser.add_argument("--unexplored-point-max-steps", type=int, default=180,
                        help="maximum steps at which we define a point completely unexplored. Increments of maximum steps stops")
    
    
    # Channels specific arguments
    parser.add_argument("--nlos-toggle", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="if toggled, NLoS propagation conditions are considered")
    parser.add_argument("--a", type=float, default=12.08,
                        help="Parameter 'a' for path loss model (12.08 in dense urban)")
    parser.add_argument("--b", type=float, default=0.11,
                        help="Parameter 'b' for path loss model (0.11 in dense urban)")
    parser.add_argument("--nlos-loss", type=float, default=23,
                        help="Path loss for NLoS in dB (23 in dense urban)")
    parser.add_argument("--los-loss", type=float, default=1.6,
                        help="Path loss for LoS in dB (1.6 in dense urban)")
    parser.add_argument("--rate-of-growth", type=float, default=-0.05,
                        help="Rate of growth for some propagation model parameter")
    parser.add_argument("--transmission-power", type=float, default=-6.7716,
                        help="Transmission power in dBm")
    parser.add_argument("--channel-bandwidth", type=float, default=2e6,
                        help="Channel bandwidth in Hz (e.g. 2e6 for 2 MHz)")
    parser.add_argument("--noise-psd", type=float, default=-174,
                        help="Power spectral density of noise in dBm/Hz")
    
    
    # Environment specific arguments
    parser.add_argument("--environment-type", type=str, default="uniform", nargs="?", const="random",
                        help="Type of environment (e.g., uniform, cluster, road, road_cluster, random")
    parser.add_argument("--environment-random-spawn-despawn-gu" , type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, GUs can randomly spawn and despawn during the simulation")
    parser.add_argument("--environment-gu-bounce", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, GUs bounce when they reach the environment boundaries else they despawn (This is not intended for roads)")
    # ALL ENV TYPES RESET RANDOM VARIANTS
    parser.add_argument("--environment-type-random", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="if toggled, the environment type is randomly chosen between uniform, cluster, road, road_cluster, random else is fixed to environment-type")
    parser.add_argument("--uav-number-random", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="if toggled, the number of UAVs is random between 1 and max-uav-number else is fixed to uav-number")
    parser.add_argument("--gu-number-random", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, the number of GUs is random between min-gu-number and max-gu-number else is fixed to starting-gu-number")
    # CLUSTER ENV TYPE RANDOM VARIANTS
    parser.add_argument("--clusters-movement", type=str, default="fleet", nargs="?", const="random",
                        help="How clusters of GU moves (e.g., fleet (same direction), stationary (all GUs stopped), independent (each GU random direction), random (each cluster have different movement type))")
    parser.add_argument("--clusters-number-random", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="if toggled, the number of clusters is random between 1 and clusters-number else is fixed to clusters-number")
    parser.add_argument("--clusters-variance-random", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, the variance of clusters is random between clusters-variance-min and clusters-variance-max else is fixed to clusters-variance-min")
    parser.add_argument("--clusters-gu-uniform-partition", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, Every cluster has the same number of GUs else the GUs are randomly partitioned")
    # CLUSTER ENV TYPE FIXED VARIANTS
    parser.add_argument("--clusters-number", type=int, default=1,
                        help="if clusters_number_random is False, this is the number of clusters in the environment")
    parser.add_argument("--clusters-variance-min", type=int, default=500,
                        help="the minimum variance of the clusters")
    parser.add_argument("--clusters-variance-max", type=int, default=1000,
                        help="the maximum variance of the clusters")
    # ROAD ENV TYPE RANDOM VARIANTS
    parser.add_argument("--roads-gu-arrival-distribution", type=str, default="poisson", nargs="?", const="random",
                        help="Distribution for the arrival of GUs on the roads (e.g., uniform, poisson, random (choose between one of distributions))")
    parser.add_argument("--roads-number-random", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="if toggled, the number of roads is random between 1 and roads-number else is fixed to roads-number")
    parser.add_argument("--roads-lane-width-random", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="if toggled, the width of a single lane is random between roads-lane-width-min and roads-lane-width-max else is fixed to roads-lane-width-min")
    parser.add_argument("--roads-gu-uniform-partition", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, Every road has the same number of max theorical GUs else the max theorical GUs are randomly partitioned")
    # ROAD CLUSTER ONLY ENV TYPE RANDOM VARIANTS
    parser.add_argument("--per-roads-clusters-number-random", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="if toggled, the number of clusters on each road is random between 1 and roads-clusters-number else is fixed to roads-clusters-number")
    # ROAD ENV TYPE FIXED VARIANTS
    parser.add_argument("--roads-lane-width-min", type=float, default=50.0,
                        help="the minimum width of a single lane in the environment (in meters)")
    parser.add_argument("--roads-lane-width-max", type=float, default=50.0,
                        help="the maximum width of a single lane in the environment (in meters)")
    parser.add_argument("--roads-number", type=int, default=1,
                        help="the number of roads in the environment (used if enviroment-type is road or road_cluster)")
    parser.add_argument("--roads-increased-max-speed", type=float, default=21.0,
                        help="Maximum speed increment applied to all roads")
    # ROAD CLUSTER ONLY ENV TYPE FIXED VARIANTS
    parser.add_argument("--per-roads-starting-clusters-number", type=int, default=1,
                        help="the starting number of clusters on each road (used if enviroment-type is road-cluster, fleet movement of default)")
    parser.add_argument("--per-roads-clusters-number", type=int, default=2,
                        help="the max number of clusters on each road (used if enviroment-type is road-cluster, fleet movement of default)")
    # RANDOM ENV TYPE (in this case every random variants of this section must be intended random by default and we can have clusters,uniform behavior and roads in the same environment)
    
    
    # Reward specific arguments
    parser.add_argument("--w-boundary-penalty", type=float, default=0.0,
                        help="Weight applied to the penalty when UAVs exceed the environment boundaries")
    parser.add_argument("--w-collision-penalty", type=float, default=0.0,
                        help="Weight applied to the penalty for collisions between UAVs")
    parser.add_argument("--w-spatial-coverage", type=float, default=1.0,
                        help="Weight for encouraging UAVs to cover as much area as possible")
    parser.add_argument("--w-exploration", type=float, default=1.0,
                        help="Weight for encouraging UAVs to explore new or less-visited areas")
    parser.add_argument("--w-homogenous-voronoi-partition", type=float, default=0.0,
                        help="Weight for encouraging a homogenous Voronoi partition among UAVs")
    parser.add_argument("--w-gu-coverage", type=float, default=1.0,
                        help="Weight for encouraging UAVs to cover ground user (GU) positions")
    
    parser.add_argument("--reward-mode", type=str, default="mixed", nargs="?", const="mixed",
                        help="Reward mode (e.g., twophase or mixed")
    parser.add_argument("--spatial-coverage-threshold", type=float, default=0.90,
                        help="Minimum percentage of the maximum theoretical points a UAV must cover to avoid a penalty.")
    parser.add_argument("--exhaustive-exploration-threshold", type=float, default=0.95,
                        help="If reward_mode = twophases, coverage threshold at which the system transitions from exhaustive exploration to the coverage phase.")
    parser.add_argument("--balanced-exploration-threshold", type=float, default=0.00,
                        help="If reward_mode = twophases, Coverage threshold at which the system transitions from balanced exploration to the coverage phase.")
    parser.add_argument("--max-steps-gu-coverage-phase", type=int, default=50,
                        help="If reward_mode = twophases, Maximum number of steps in the coverage phase before switching back to exploration.")
    parser.add_argument("--tradeoff-mode", type=str, default="power_law", nargs="?", const="power_law",
                        help="If reward_mode = mixed, tradeoff mode between exploration and GU coverage (e.g., exponential [e^(kx)-1], exponential_norm [(e^(kx)-1)/(e^k-1)], power_law [x^k]). If not selected is linear [x]")
    parser.add_argument("--k-factor", type=float, default=1.00,
                        help="If reward_mode = mixed, k-factor is used to balance between exploration and GU coverage reward components. A higher value increases the weight of the exponential growth term relative to the base reward. This remains static.")


    # PPO specific arguments
    parser.add_argument("--num-envs", type=int, default=16,
                        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=128,
                        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--updates", type=int, default=400,
                        help="the number of updates (rollout + training)")
    parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Use GAE for advantage computation")
    parser.add_argument("--gamma", type=float, default=0.95,
                        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.90,
                        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=8,
                        help="the number of mini-batches") 
    parser.add_argument("--update-epochs", type=int, default=4,
                        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Toggles advantages normalization")
    parser.add_argument("--norm-value", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="Toggles values(returns) normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
                        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="Toggles whether or not to use a clipped loss for the value function")
    parser.add_argument("--ent-coef", type=float, default=0.001,
                        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
                        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
                        help="the maximum norm for the gradient clipping")
    
    
    # Test specific arguments
    parser.add_argument("--updates-per-test", type=int, default=20,
                        help="the number of updates before a test")
    parser.add_argument("--num-test-episodes", type=int, default=9,
                        help="the number of episodes to test the agent")
    parser.add_argument("--test-steps-after-trunc", type=int, default=500,
                        help="the number of steps in a test enviroment before a truncation")
    

    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    return args
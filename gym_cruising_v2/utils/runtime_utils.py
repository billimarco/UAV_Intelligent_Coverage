import os
import subprocess
import argparse
from distutils.util import strtobool
import torch

def get_most_free_gpu():
    """Finds the GPU with the most free memory using nvidia-smi."""
    try:
        torch.cuda.empty_cache()
        # Run nvidia-smi to get free memory for each GPU
        output = subprocess.check_output("nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits", shell=True)
        free_memory = [int(x) for x in output.decode("utf-8").strip().split("\n")]
        
        print(torch.cuda.device_count())  # quante GPU vede PyTorch
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
        # Find the index of the GPU with the highest free memory
        best_gpu = free_memory.index(max(free_memory))
        return best_gpu, free_memory[best_gpu]

    except Exception as e:
        print(f"Error detecting GPUs: {e}")
        return None, None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="ppo_vanilla",
                        help="the name of this experiment")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
                        help="the learning rate of the optimizer")# 1.0e-3 lr, 2.5e-4 default, 1.0e-4 lrl, 2.5e-5 lrl--
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, the learning rate will be annealed")
    parser.add_argument("--seed", type=int, default=9,
                        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--offline", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="Tesi",
                        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default="marcolbilli-universit-di-firenze",
                        help="the entity (team) of wandb's project")
    
    # Algorithm specific arguments
    parser.add_argument("--alg", type=str, default="PPO", nargs="?", const="PPO",
                        help="Algorithm to use: PPO, TD3")
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
    
    # Reward specific arguments
    parser.add_argument("--alpha", type=float, default=0.3,
                    help="Weight for the individual reward (UAV's coverage contribution)")
    parser.add_argument("--beta", type=float, default=0.7,
                        help="Weight for the global reward (fairness, spatial coverage, exploration)")
    
    # UAV specific arguments
    parser.add_argument("--max-uav-number", type=int, default=3,
                        help="the max number of UAVs in the environment")
    parser.add_argument("--uav-number", type=int, default=3,
                        help="the number of UAVs in the environment (not more than max-uav-number)")
    parser.add_argument("--max-speed-uav", type=float, default=50.0,
                        help="maximum speed of a UAV in meters per second")
    parser.add_argument("--uav-altitude", type=float, default=500,
                    help="UAV flight altitude in meters")
    parser.add_argument("--minimum-starting-distance-between-uav", type=float, default=50.0,
                    help="minimum initial distance between UAVs in meters")
    parser.add_argument("--collision-distance", type=float, default=5.0,
                        help="minimum distance between UAVs before a collision is considered (in meters)")
    
    # GU specific arguments
    parser.add_argument("--max-gu-number", type=int, default=120,
                        help="the max number of GUs in the environment")
    parser.add_argument("--starting-gu-number", type=int, default=90,
                        help="the number of starting ground units in the environment")
    parser.add_argument("--spawn-gu-prob", type=float, default=0.0005,
                        help="probability of spawning a ground unit per cell or timestep")
    parser.add_argument("--gu-mean-speed", type=float, default=5.00,
                        help="mean speed of ground units in meters per second")
    parser.add_argument("--gu-standard-deviation", type=float, default=2.00,
                        help="standard deviation of ground unit speed in meters per second")
    parser.add_argument("--covered-threshold", type=float, default=10.0,
                        help="SINR threshold (in dB) above which a ground user is considered covered")
    
    # Grid specific arguments
    parser.add_argument("--window-width", type=int, default=500,
                        help="the width size of the PyGame window")
    parser.add_argument("--window-height", type=int, default=500,
                        help="the height size of the PyGame window")
    parser.add_argument("--resolution", type=int, default=1,
                    help="meters for every pixel side (num of points for pixel side)")
    parser.add_argument("--spawn-offset" , type=int, default=55,
                        help="the spawn offset of the environment in point")
    parser.add_argument("--x-offset", type=float, default=0,
                        help="the x offset of the environment")
    parser.add_argument("--y-offset", type=float, default=0,
                        help="the y offset of the environment")
    parser.add_argument("--wall-width", type=float, default=3,
                        help="the width of the walls in the environment")
    parser.add_argument("--unexplored-point-max-steps", type=int, default=50,
                        help="maximum steps at which we define a point completely unexplored. Increments of maximum steps stops")
    
    # Channels specific arguments
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
    
    # Clustered environment specific arguments
    parser.add_argument("--clustered", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="if toggled, GU are clustered")
    parser.add_argument("--clusters-number", type=int, default=1,
                        help="the number of clusters in the environment")
    parser.add_argument("--clusters-variance-min", type=int, default=70000,
                        help="the minimum variance of the clusters")
    parser.add_argument("--clusters-variance-max", type=int, default=100000,
                        help="the maximum variance of the clusters")
    
    # PPO specific arguments
    parser.add_argument("--num-envs", type=int, default=16,
                        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=128,
                        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--updates-per-env", type=int, default=500,
                        help="the number of steps to run in each environment")
    parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Use GAE for advantage computation")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=8,
                        help="the number of mini-batches") 
    parser.add_argument("--update-epochs", type=int, default=4,
                        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Toggles advantages normalization")
    parser.add_argument("--norm-value", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=False,
                        help="Toggles values(returns) normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
                        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="Toggles whether or not to use a clipped loss for the value function")
    parser.add_argument("--ent-coef", type=float, default=0.001,
                        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=1.00,
                        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
                        help="the maximum norm for the gradient clipping")
    

    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    return args
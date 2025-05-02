import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import subprocess
import argparse
from distutils.util import strtobool

import time
import gymnasium as gym
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
import torch.nn.functional as F
import wandb

from gym_cruising.memory.replay_memory import ReplayMemory, Transition
from gym_cruising.neural_network.MLP_policy_net import MLPPolicyNet
from gym_cruising.neural_network.deep_Q_net import DeepQNet, DoubleDeepQNet
from gym_cruising.neural_network.transformer_encoder_decoder import TransformerEncoderDecoder
import gym_cruising.utils.runtime_utils as utils


TRAIN = False
BATCH_SIZE = 256  # is the number of transitions random sampled from the replay buffer
BETA = 0.005  # is the update rate of the target network
sigma_policy = 0.4  # Standard deviation of noise for policy actor actions on current state
sigma = 0.2  # Standard deviation of noise for target policy actions on next states
c = 0.2  # Clipping bound of noise
start_steps = 20000


time_steps_done = 0
optimization_steps = 3

BEST_VALIDATION = 0.0
MAX_LAST_RCR = 0.0
EMBEDDED_DIM = 32

if __name__ == "__main__":
    args = utils.parse_args()

    print(f"Available GPUs: {torch.cuda.device_count()}")
    best_gpu, free_mem = utils.get_most_free_gpu()
    if best_gpu is not None:
        print(f"Using GPU {best_gpu} with {free_mem} MB free.")
        device = torch.device(f"cuda:{best_gpu}")
    else:
        print("No GPU available, using CPU.")
        device = torch.device("cpu")
    
    # Wandb setup
    if args.track:
        run_name = f"{args.exp_name}"
        mode = "offline" if args.offline else "online"
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            mode=mode,
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )


    env = gym.make('gym_cruising:Cruising-v0', 
                    render_mode='rgb_array', 
                    track_id=2)

    # ACTOR POLICY NET policy
    transformer_policy = TransformerEncoderDecoder(embed_dim=EMBEDDED_DIM).to(device)
    mlp_policy = MLPPolicyNet(token_dim=EMBEDDED_DIM).to(device)

    # CRITIC Q NET policy
    deep_Q_net_policy = DoubleDeepQNet(state_dim=EMBEDDED_DIM).to(device)

    # COMMENT FOR INITIAL TRAINING -> CURRICULUM LEARNING
    # PATH_TRANSFORMER = '../neural_network/bestTransformer.pth'
    # transformer_policy.load_state_dict(torch.load(PATH_TRANSFORMER))
    # PATH_MLP_POLICY = '../neural_network/bestMLP.pth'
    # mlp_policy.load_state_dict(torch.load(PATH_MLP_POLICY))
    # PATH_DEEP_Q = '../neural_network/bestDeepQ.pth'
    # deep_Q_net_policy.load_state_dict(torch.load(PATH_DEEP_Q))

    # ACTOR POLICY NET target
    transformer_target = TransformerEncoderDecoder(embed_dim=EMBEDDED_DIM).to(device)
    mlp_target = MLPPolicyNet(token_dim=EMBEDDED_DIM).to(device)

    # CRITIC Q NET target
    deep_Q_net_target = DoubleDeepQNet(state_dim=EMBEDDED_DIM).to(device)

    # set target parameters equal to main parameters
    transformer_target.load_state_dict(transformer_policy.state_dict())
    mlp_target.load_state_dict(mlp_policy.state_dict())
    deep_Q_net_target.load_state_dict(deep_Q_net_policy.state_dict())

    optimizer_transformer = optim.Adam(transformer_policy.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    optimizer_mlp = optim.Adam(mlp_policy.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    optimizer_deep_Q = optim.Adam(deep_Q_net_policy.parameters(), lr=args.learning_rate, weight_decay=1e-5)

    replay_buffer_uniform = ReplayMemory(100000)
    replay_buffer_clustered = ReplayMemory(100000)
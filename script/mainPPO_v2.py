import sys
import os
import warnings
import subprocess
import argparse

#warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium.utils.passive_env_checker")
os.environ.pop("CUDA_VISIBLE_DEVICES", None)  # rimuove se esiste
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import torch.multiprocessing as mp
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

from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv
from gym_cruising_v2.neural_network.PPO_net import PPONet
from gym_cruising_v2.utils.value_normalizer import ValueNormalizer
import gym_cruising_v2.utils.runtime_utils as utils

args = utils.parse_args()

def compute_gae(rewards, values, dones, last_values):
    """
    GAE adattivo per critic centralizzato o individuale.
    Espande i valori una sola volta se serve.

    Args:
        rewards: (num_envs, num_steps, num_agents)
        values: (num_envs, num_steps, 1) oppure (num_envs, num_steps, num_agents)
        dones: (num_envs, num_steps, num_agents)

    Returns:
        advantages: (num_envs, num_steps, num_agents)
        returns: (num_envs, num_steps, num_agents)
    """
    num_envs, num_steps, num_agents = rewards.shape

    # Bootstrap finale
    values_ext = torch.cat([
        values,
        last_values
    ], dim=1)  # → (num_envs, num_steps+1, num_agents)
    
    # Denormalizza se richiesto
    if args.norm_value:
        values_ext = value_normalizer.denormalize(values_ext)

    # Se critic è centralizzato (1), espandi a num_agents
    if args.global_value:
        values_ext = values_ext.expand(-1, -1, num_agents)  # (num_envs, num_steps, num_agents)

    '''
    # Bootstrap finale con tutti 0
    values_ext = torch.cat([
        values,
        torch.zeros((num_envs, 1, num_agents), dtype=values.dtype, device=values.device)
    ], dim=1)  # → (num_envs, num_steps+1, num_agents)
    '''

    advantages = torch.zeros_like(rewards)
    gae = torch.zeros((num_envs, num_agents), dtype=rewards.dtype, device=rewards.device)

    for t in reversed(range(num_steps)):
        next_values = values_ext[:, t + 1, :]
        current_values = values[:, t, :]
        not_done = 1.0 - dones[:, t, :].float()

        delta = rewards[:, t, :] + args.gamma * next_values * not_done - current_values
        gae = delta + args.gamma * args.gae_lambda * not_done * gae
        advantages[:, t, :] = gae

    returns = advantages + values

    return advantages, returns

def process_state_batch(state_batch):
    """
    Processa un batch di stati (lista di dict), convertendo i campi in tensori e
    impilando lungo la dimensione batch senza padding.

    Args:
        state_batch: list di dict
        device: torch.device

    Returns:
        map_exploration_batch: [B, grid_height, grid_width]
        uav_info_batch: [B, max_uav, 4]
        connected_gu_positions_batch: [B, max_gu, 2]
        uav_mask_batch: [B, max_uav] (bool)
        gu_mask_batch: [B, max_gu] (bool)
        uav_flags_batch : [B, max_uav, max_uav+4]
    """

    def to_tensor_with_batch(np_array, dtype, expected_dim):
        tensor = torch.from_numpy(np_array).to(dtype=dtype, device=device)
        if tensor.dim() == expected_dim - 1:
            tensor = tensor.unsqueeze(0)  # Aggiungi dimensione batch
        return tensor

    map_exploration_batch = to_tensor_with_batch(state_batch["map_exploration_states"], torch.float32, 3)
    uav_info_batch = to_tensor_with_batch(state_batch["uav_states"], torch.float32, 3)
    connected_gu_positions_batch = to_tensor_with_batch(state_batch["covered_users_states"], torch.float32, 3)
    uav_mask_batch = to_tensor_with_batch(state_batch["uav_mask"], torch.bool, 2)
    gu_mask_batch = to_tensor_with_batch(state_batch["gu_mask"], torch.bool, 2)
    uav_flags_batch = to_tensor_with_batch(state_batch["uav_flags"], torch.bool, 3)

    return map_exploration_batch, uav_info_batch, connected_gu_positions_batch, uav_mask_batch, gu_mask_batch, uav_flags_batch

def get_set_up(round_robin):

    if round_robin:
        options = {
            "uav": args.uav_number,
            "gu": args.starting_gu_number,
            "environment_type": args.environment_type,
            "test": True
        }

        if args.uav_number_random:
            # Incrementa args.uav_number
            args.uav_number += 1

            # Resetta se supera il massimo
            if args.uav_number > args.max_uav_number:
                args.uav_number = 1
    else:
        options = {
            "uav": args.uav_number,
            "gu": args.starting_gu_number,
            "environment_type": args.environment_type,
            "test": True
        }  
        
    return options

def test(agent, num_episodes=32, global_step=0):
    args.options = None
    env = gym.make('gym_cruising_v2:Cruising-v0', args=args, render_mode=args.render_mode)
    agent.eval()
    total_rewards = []
    rcr_values = []
    steps_vec = []
    out_area_vec = []
    collision_vec = []
    
    boundary_penalty_rewards = []
    collision_penalty_rewards = []
    spatial_coverage_rewards = []
    exploration_incentive_rewards = []
    homogenous_voronoi_partition_incentive_rewards = []
    gu_coverage_rewards = []
    
    for ep in range(num_episodes):
        args.options = get_set_up(args.round_robin)
        np.random.seed(ep)
        state, info = env.reset(seed=None, options=args.options)
        done = False
        steps = 0
        sum_episode_reward = np.zeros(args.max_uav_number)
        sum_boundary_penalty_total = 0
        sum_collision_penalty_total = 0
        sum_spatial_coverage_total = 0
        sum_exploration_incentive_total = 0
        sum_homogenous_voronoi_partition_incentive_total = 0
        sum_gu_coverage_total = 0
        sum_rcr = 0
        sum_out_area = 0
        sum_collision = 0
        out_area = False
        collision = False

        while not done:
            steps += 1
            map_exploration, uav_info, connected_gu_positions, uav_mask, gu_mask, uav_flags = process_state_batch(state)
            
            # Inference
            with torch.no_grad():
                actions, _, _, _ = agent(
                    map_exploration,           # shape: (1, H, W)
                    uav_info,                  # shape: (1, UAV, 4)    
                    connected_gu_positions,    # shape: (1, GU, 2)
                    uav_flags,                 # shape: (1, UAV, UAV + 4)
                    uav_mask,                  # shape: (1, UAV)
                    gu_mask                    # shape: (1, GU)
                )

            # Rimuovi batch dim
            actions_np = actions.squeeze(0).cpu().numpy()
            uav_mask_np = uav_mask.squeeze(0).cpu().numpy()
            env_action = {
                "uav_moves": actions_np,     # (max_uav, 2)
                "uav_mask": uav_mask_np      # (max_uav)
            }
            
            # Passaggio ambiente
            next_state, reward, terminated, truncated, info = env.step(env_action)
            # Estrapolo reward e terminated per UAV da info
            rewards = np.array(info.get("reward_per_uav", [reward]))    # fallback a reward singolo
            terminated = np.array(info.get("terminated_per_uav", [terminated]))
            if steps == args.test_steps_after_trunc:
                truncated = True
            if np.any(terminated):
                if info["Out_Area"]:
                    sum_out_area += 1
                if info["Collision"]:
                    sum_collision += 1
            done = truncated
            
            sum_episode_reward += rewards
            sum_rcr += float(info['RCR'])
            sum_boundary_penalty_total += info["boundary_penalty_total"]
            sum_collision_penalty_total += info["collision_penalty_total"]
            sum_spatial_coverage_total += info["spatial_coverage_total"]
            sum_exploration_incentive_total += info["exploration_incentive_total"]
            sum_homogenous_voronoi_partition_incentive_total += info["homogenous_voronoi_partition_incentive_total"]
            sum_gu_coverage_total += info["gu_coverage_total"]
            
            state = next_state
            if done:
                '''
                out_area = info["Out_Area"]
                collision = info["Collision"]
                '''
                break
        
        episode_reward = np.sum(sum_episode_reward[uav_mask_np])  # Totale reward per UAV attivi
        avg_rcr = sum_rcr / steps
        out_area_vec.append(sum_out_area)
        collision_vec.append(sum_collision)
        # Aggiungi l'RCR medio per l'episodio
        steps_vec.append(steps)
        rcr_values.append(avg_rcr)  # Media di RCR per episodio
        
        total_rewards.append(episode_reward)
        boundary_penalty_rewards.append(sum_boundary_penalty_total)
        collision_penalty_rewards.append(sum_collision_penalty_total)
        spatial_coverage_rewards.append(sum_spatial_coverage_total)
        exploration_incentive_rewards.append(sum_exploration_incentive_total)
        homogenous_voronoi_partition_incentive_rewards.append(sum_homogenous_voronoi_partition_incentive_total)
        gu_coverage_rewards.append(sum_gu_coverage_total)
        
        print(f"Episode {ep + 1}: uav_number = {args.options['uav']}, starting_gu = {args.options['gu']}, steps = {steps}, total_reward = {episode_reward:.2f}, RCR = {avg_rcr:.2f}, out_areas = {sum_out_area}, collisions = {sum_collision},\n"
              f"boundary_penalty_total = {sum_boundary_penalty_total:.2f}, collision_penalty_total = {sum_collision_penalty_total:.2f}, spatial_coverage_total = {sum_spatial_coverage_total:.2f},\n"
              f"exploration_incentive_total = {sum_exploration_incentive_total:.2f}, homogenous_voronoi_partition_incentive_total = {sum_homogenous_voronoi_partition_incentive_total:.2f}, gu_coverage_total = {sum_gu_coverage_total:.2f}\n")

    # Statistiche
    mean_steps = np.mean(steps_vec)
    mean_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    mean_rcr = np.mean(rcr_values)
    std_rcr = np.std(rcr_values)
    mean_out_area = np.mean(out_area_vec)
    mean_collision = np.mean(collision_vec)
    
    mean_boundary_penalty = np.mean(boundary_penalty_rewards)
    mean_collision_penalty = np.mean(collision_penalty_rewards)
    mean_spatial_coverage = np.mean(spatial_coverage_rewards)
    mean_exploration_incentive = np.mean(exploration_incentive_rewards)
    mean_homogenous_voronoi_partition_incentive = np.mean(homogenous_voronoi_partition_incentive_rewards)
    mean_gu_coverage = np.mean(gu_coverage_rewards)

    print(f"\nTest completed over {num_episodes} episodes.")
    print(f"Average reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Average RCR: {mean_rcr:.2f} ± {std_rcr:.2f}")
    print(f"Average Steps: {mean_steps:.2f}")
    print(f"Average Out Area: {mean_out_area:.2f}")
    print(f"Average Collisions: {mean_collision:.2f}")
    print(f"Average Boundary Penalty: {mean_boundary_penalty:.2f}")
    print(f"Average Collision Penalty: {mean_collision_penalty:.2f}")
    print(f"Average Spatial Coverage: {mean_spatial_coverage:.2f}")
    print(f"Average Exploration Incentive: {mean_exploration_incentive:.2f}")
    print(f"Average Homogenous Voronoi Partition Penalty: {mean_homogenous_voronoi_partition_incentive:.2f}")
    print(f"Average GU Coverage: {mean_gu_coverage:.2f}")

    # Logging su Weights & Biases
    if wandb.run is not None:
        wandb.log({
            "test/mean_reward": mean_reward,
            "test/std_reward": std_reward,
            "test/mean_rcr": mean_rcr,
            "test/std_rcr": std_rcr,
            "test/mean_steps": mean_steps,
            "test/mean_out_area": mean_out_area,
            "test/mean_collision": mean_collision,
            "test/mean_boundary_penalty": mean_boundary_penalty,
            "test/mean_collision_penalty": mean_collision_penalty,
            "test/mean_spatial_coverage": mean_spatial_coverage,
            "test/mean_exploration_incentive": mean_exploration_incentive,
            "test/mean_homogenous_voronoi_partition_incentive": mean_homogenous_voronoi_partition_incentive,
            "test/mean_gu_coverage": mean_gu_coverage,
        }, step=global_step)

    
if __name__ == "__main__":
    '''
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    '''
    torch.autograd.set_detect_anomaly(True) # Backward con NaN debug
    
    # Ottieni il percorso assoluto della root del progetto, basato su questo file
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    print(f"Available GPUs: {torch.cuda.device_count()}")

    if args.cuda:
        print(f"Usando GPU {args.cuda_device}")
        device = torch.device(f"cuda:{args.cuda_device}")
    else:
        print("Nessuna GPU scelta, uso CPU")
        device = torch.device("cpu")
    
    if args.train:
        
        # Wandb setup
        if args.track:
            run_name = f"{args.exp_name}"
            mode = "offline" if args.offline else "online"
            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=False,
                config=vars(args),
                mode=mode,
                name=run_name,
                monitor_gym=True,
                save_code=True
            )
            wandb.define_metric("test/mean_reward", step_metric="global_step")
            wandb.define_metric("test/std_reward", step_metric="global_step")
            wandb.define_metric("test/mean_rcr", step_metric="global_step")
            wandb.define_metric("test/std_rcr", step_metric="global_step")
            wandb.define_metric("test/mean_steps", step_metric="global_step")
            wandb.define_metric("test/mean_out_area", step_metric="global_step")
            wandb.define_metric("test/mean_collision", step_metric="global_step")
            wandb.define_metric("test/mean_boundary_penalty", step_metric="global_step")
            wandb.define_metric("test/mean_collision_penalty", step_metric="global_step")
            wandb.define_metric("test/mean_spatial_coverage", step_metric="global_step")
            wandb.define_metric("test/mean_exploration_incentive", step_metric="global_step")
            wandb.define_metric("test/mean_homogenous_voronoi_partition_incentive", step_metric="global_step")
            wandb.define_metric("test/mean_gu_coverage", step_metric="global_step")
            
            wandb.define_metric("losses/value_loss", step_metric="global_step")
            wandb.define_metric("losses/policy_loss", step_metric="global_step")
            wandb.define_metric("losses/entropy", step_metric="global_step")
            wandb.define_metric("losses/old_approx_kl", step_metric="global_step")
            wandb.define_metric("losses/approx_kl", step_metric="global_step")
            wandb.define_metric("losses/clipfrac", step_metric="global_step")
            wandb.define_metric("losses/explained_variance", step_metric="global_step")
            
            wandb.define_metric("charts/learning_rate", step_metric="global_step")
            wandb.define_metric("charts/SPS", step_metric="global_step")


        #env = gym.make('gym_cruising_v2:Cruising-v0', args=args, render_mode=args.render_mode)
        
        # Path al file dei pesi
        model_path = os.path.join(project_root, "neural_network", f"{args.exp_name}.pth")
        model_path_test = os.path.join(project_root, "neural_network", f"{args.exp_name}_test.pth")
        ppo_net = PPONet(embed_dim=args.embedded_dim, max_uav_number=args.max_uav_number, map_size=(args.window_height*args.resolution, args.window_width*args.resolution), patch_size=args.patch_size, global_value=args.global_value).to(device)
        
        args.options = {
            "test": False
        }
        envs = gym.make_vec('gym_cruising_v2:Cruising-v0', num_envs=args.num_envs, vectorization_mode="async", args=args, render_mode=args.render_mode)
        #print(envs.metadata["autoreset_mode"])
        optimizer = optim.Adam(ppo_net.parameters(), lr=args.learning_rate, weight_decay=1e-5)
        if args.norm_value:
            value_normalizer = ValueNormalizer()
        
        # Buffer temporanei per un singolo episodio/rollout
        map_exploration_tensor = torch.zeros((args.num_envs, args.num_steps, args.window_height*args.resolution, args.window_width*args.resolution ), dtype=torch.float32).to(device)
        uav_states_tensor = torch.zeros((args.num_envs, args.num_steps, args.max_uav_number, 4), dtype=torch.float32).to(device)
        uav_mask_tensor = torch.zeros((args.num_envs, args.num_steps, args.max_uav_number), dtype=torch.bool).to(device)
        covered_users_tensor = torch.zeros((args.num_envs, args.num_steps, args.max_gu_number, 2), dtype=torch.float32).to(device)
        gu_mask_tensor = torch.zeros((args.num_envs, args.num_steps, args.max_gu_number), dtype=torch.bool).to(device)
        uav_flags_tensor = torch.zeros((args.num_envs, args.num_steps, args.max_uav_number, args.max_uav_number + 4), dtype=torch.float32).to(device)
        actions_tensor = torch.zeros((args.num_envs, args.num_steps, args.max_uav_number, 2), dtype=torch.float32).to(device)
        log_probs_tensor = torch.zeros((args.num_envs, args.num_steps, args.max_uav_number), dtype=torch.float32).to(device)
        rewards_tensor = torch.zeros((args.num_envs, args.num_steps, args.max_uav_number), dtype=torch.float32).to(device)
        dones_tensor = torch.zeros((args.num_envs, args.num_steps, args.max_uav_number), dtype=torch.bool).to(device)
        if args.global_value:
            values_tensor = torch.zeros((args.num_envs, args.num_steps, 1), dtype=torch.float32).to(device)
        else:
            values_tensor = torch.zeros((args.num_envs, args.num_steps, args.max_uav_number), dtype=torch.float32).to(device)
        
        global_step = 0
        start_time = time.time()
        
        for update in range(0, args.updates):

            if update % args.updates_per_test == 0 and update % args.updates != 0:
                print("\n<------------------------->")
                print(f"Testing at update {update}")
                test_time = time.time()
                test(agent=ppo_net, num_episodes=args.num_test_episodes, global_step=global_step)
                torch.save(ppo_net.state_dict(), model_path_test)
                print(f"Tested! Time elaplesed {time.time() - test_time}")
                print("<------------------------->\n")
                start_time = time.time()
            
            if args.anneal_lr:
                frac = 1.0 - (update - 1.0) / args.updates
                lrnow = frac * args.learning_rate
                optimizer.param_groups[0]["lr"] = lrnow

            print(f"Update {update + 1}/{args.updates}")
            
            # Rollout
            rollout_start_time = time.time()
            states, infos = envs.reset()

            for step in range(args.num_steps):
                global_step += args.num_envs

                # Processa batch di osservazioni da tutti gli env
                map_exploration, uav_info, connected_gu_positions, uav_mask, gu_mask, uav_flags = process_state_batch(states)

                # Ottieni azioni da PPO
                with torch.no_grad():
                    actions, logprobs, entropy, values = ppo_net(
                        map_exploration,           # [B, H, W]
                        uav_info,                  # [B, UAV, 4]
                        connected_gu_positions,    # [B, GU, 2]
                        uav_flags,                 # [B, UAV, UAV + 4]
                        uav_mask,                  # [B, UAV]
                        gu_mask                    # [B, GU]
                    )

                # Step negli env paralleli
                actions_np = actions.cpu().numpy()
                uav_mask_np = uav_mask.cpu().numpy()
                env_action = {
                    "uav_moves": actions_np,     # (num_envs, max_uav, 2)
                    "uav_mask": uav_mask_np      # (num_envs, max_uav)
                }

                next_states, rewards, terminated, truncated, infos = envs.step(env_action) # rewards e terminateds sono dummy. Servono per gestire l'AsyncVectorEnv in modo sicuro

                #print(terminated)
                # Bypassa i problemi dell'AsyncVectorEnv passando rewards e terminateds
                reward_array = np.vstack([np.array(r) for r in infos["reward_per_uav"]])
                terminated_array = np.vstack([np.array(t) for t in infos["terminated_per_uav"]])

                rewards = torch.tensor(reward_array, device=device)
                dones = torch.tensor(terminated_array, device=device)
                
                # Salvataggio dei dati nel buffer
                map_exploration_tensor[:, step, :, :] = map_exploration
                uav_states_tensor[:, step, :, :] = uav_info
                uav_mask_tensor[:, step, :] = uav_mask
                covered_users_tensor[:, step, :, :] = connected_gu_positions
                gu_mask_tensor[:, step, :] = gu_mask
                uav_flags_tensor[:, step, :, :] = uav_flags
                actions_tensor[:, step, :, :] = actions
                log_probs_tensor[:, step, :] = logprobs
                rewards_tensor[:, step, :] = rewards
                dones_tensor[:, step, :] = dones
                if args.global_value:
                    values_tensor[:, step, 0] = values
                else:
                    values_tensor[:, step, :] = values

                # Aggiorna lo stato corrente
                states = next_states

                #print(global_step)

            # Prendi la values dell'utlimo stato per fare bootstrapping finale su GAE
            map_exploration, uav_info, connected_gu_positions, uav_mask, gu_mask, uav_flags = process_state_batch(states)
            with torch.no_grad():
                _, _, _, values = ppo_net(
                    map_exploration,           # [B, H, W]
                    uav_info,                  # [B, UAV, 4]
                    connected_gu_positions,    # [B, GU, 2]
                    uav_flags,                 # [B, UAV, UAV + 4]
                    uav_mask,                  # [B, UAV]
                    gu_mask                    # [B, GU]
                )
                
            if args.global_value:
                last_values = torch.zeros((args.num_envs, 1, 1), dtype=torch.float32).to(device)
                last_values[:, 0, 0] = values
            else:
                last_values = torch.zeros((args.num_envs, 1, args.max_uav_number), dtype=torch.float32).to(device)
                last_values[:, 0, :] = values
            
            # Calcolare gli advantages e i returns usando GAE
            advantages_tensor, returns_tensor = compute_gae(
                rewards=rewards_tensor,
                values=values_tensor,
                dones=dones_tensor,
                last_values=last_values
            )
            
            if args.norm_value:
                value_normalizer.update(returns_tensor)
                returns_tensor = value_normalizer.normalize(returns_tensor)
            
            rollout_end_time = time.time()
            print(f"Rollout time: {rollout_end_time - rollout_start_time:.2f} seconds")
            
            # Supponiamo che le variabili siano già definite con shape (num_envs, num_steps, ...)

            ppo_start_time = time.time()
            # Flatten (num_envs, num_steps, ...) → (B, ...)
            flat_map_exploration = map_exploration_tensor.reshape(args.batch_size, args.window_height*args.resolution, args.window_width*args.resolution)
            flat_uav_states = uav_states_tensor.reshape(args.batch_size, args.max_uav_number, 4)
            flat_uav_mask = uav_mask_tensor.reshape(args.batch_size, args.max_uav_number)
            flat_covered_users = covered_users_tensor.reshape(args.batch_size, args.max_gu_number, 2)
            flat_gu_mask = gu_mask_tensor.reshape(args.batch_size, args.max_gu_number)
            flat_uav_flags = uav_flags_tensor.reshape(args.batch_size, args.max_uav_number, args.max_uav_number + 4)
            flat_actions = actions_tensor.reshape(args.batch_size, args.max_uav_number, 2)
            flat_log_probs = log_probs_tensor.reshape(args.batch_size, args.max_uav_number)
            #flat_rewards = rewards_tensor.reshape(args.batch_size, args.max_uav_number)
            #flat_dones = dones_tensor.reshape(args.batch_size, args.max_uav_number)
            if args.global_value:
                flat_values = values_tensor.reshape(args.batch_size, 1)
            else:
                flat_values = values_tensor.reshape(args.batch_size, args.max_uav_number)
            flat_advantages = advantages_tensor.reshape(args.batch_size, args.max_uav_number)
            flat_returns = returns_tensor.reshape(args.batch_size, args.max_uav_number)

            clipfracs = []

            for epoch in range(args.update_epochs):
                b_inds = torch.randperm(args.batch_size, device=device)
                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]

                    mb_map_exploration = flat_map_exploration[mb_inds]                          # [MB, H, W]
                    mb_state_uav = flat_uav_states[mb_inds]                                     # [MB, MAX_UAV, 4]
                    mb_state_connected_gu = flat_covered_users[mb_inds]                         # [MB, MAX_GU, 2]
                    mb_uav_mask = flat_uav_mask[mb_inds]                                        # [MB, MAX_UAV]
                    mb_gu_mask = flat_gu_mask[mb_inds]                                          # [MB, MAX_GU]
                    mb_uav_flags = flat_uav_flags[mb_inds]                                      # [MB, MAX_UAV, MAX_UAV + 4]
                    
                    mb_actions = flat_actions[mb_inds]                                          # [MB, MAX_UAV, 2]
                    mb_logprobs = flat_log_probs[mb_inds]                                       # [MB, MAX_UAV]
                    
                    #mb_rewards = flat_rewards[mb_inds].expand(-1, args.max_uav_number)         # [MB, MAX_UAV]
                    #mb_dones = flat_dones[mb_inds].expand(-1, args.max_uav_number)             # [MB, MAX_UAV]
                    mb_values = flat_values[mb_inds]                                            # [MB, 1] if global else [MB, MAX_UAV]
                    mb_advantages = flat_advantages[mb_inds]                                    # [MB, MAX_UAV]
                    mb_returns = flat_returns[mb_inds]                                          # [MB, MAX_UAV]


                    # Forward della rete
                    _, newlogprob, entropy, newvalues = ppo_net(mb_map_exploration, 
                                                               mb_state_uav, 
                                                               mb_state_connected_gu,
                                                               mb_uav_flags, 
                                                               mb_uav_mask, 
                                                               mb_gu_mask,
                                                               mb_actions)
                    
                    mb_masked_logprobs = mb_logprobs[mb_uav_mask]          # Maschera su logprobs
                    mb_masked_newlogprob = newlogprob[mb_uav_mask]  # Maschera su logprobs
                    mb_masked_advantages = mb_advantages[mb_uav_mask]  # Maschera su advantages
                    mb_masked_returns = mb_returns[mb_uav_mask]
                    mb_masked_entropy = entropy[mb_uav_mask]      # Maschera su entropy
                    
                    if args.global_value:
                        mb_masked_values = mb_values.expand(-1, args.max_uav_number)[mb_uav_mask]
                        mb_masked_newvalues = newvalues.unsqueeze(-1).expand(-1, args.max_uav_number)[mb_uav_mask]     # Aggiungi dimensione per allineamento
                    else:
                        mb_masked_values = mb_values[mb_uav_mask]
                        mb_masked_newvalues = newvalues[mb_uav_mask]
                        
                    logratio = mb_masked_newlogprob - mb_masked_logprobs
                    ratio = logratio.exp()

                    with torch.no_grad():
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                    # Normalizza le advantages
                    if args.norm_adv:
                        mb_masked_advantages = (mb_masked_advantages - mb_masked_advantages.mean()) / (mb_masked_advantages.std() + 1e-8)
                        
                    # Policy loss
                    pg_loss1 = -mb_masked_advantages * ratio
                    pg_loss2 = -mb_masked_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    if args.clip_vloss:
                        v_loss_unclipped = (mb_masked_newvalues - mb_masked_returns) ** 2
                        v_clipped = mb_masked_values + torch.clamp(
                            mb_masked_newvalues - mb_masked_values,
                            -args.clip_coef,
                            args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - mb_masked_returns) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((mb_masked_newvalues - mb_masked_returns) ** 2).mean()

                    entropy_loss = mb_masked_entropy.mean()
                    total_loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                    optimizer.zero_grad()
                    total_loss.backward()
                    nn.utils.clip_grad_norm_(ppo_net.parameters(), args.max_grad_norm)
                    optimizer.step()

            # Log training metrics
            y_true = flat_returns.cpu().numpy()
            y_pred = flat_values.cpu().numpy()
            print("Primi 10 valori di y_true:", y_true[:10])
            print("Primi 10 valori di y_pred:", y_pred[:10])
            print("Differenze:", y_true[:10] - y_pred[:10])
            var_y = np.var(y_true)
            var_diff = np.var(y_true - y_pred)
            print("Varianza di y_true:", var_y)
            print("Varianza delle differenze:", var_diff)
            print("Varianza spiegata:", 1 - (var_diff / var_y) if var_y != 0 else "N/A")
            
            explained_var = np.nan if var_y == 0 else 1 - (var_diff / var_y)
            
            wandb.log({
                "charts/learning_rate": optimizer.param_groups[0]["lr"],
                "losses/value_loss": v_loss.item(),
                "losses/policy_loss": pg_loss.item(),
                "losses/entropy": entropy_loss.item(),
                "losses/old_approx_kl": old_approx_kl.item(),
                "losses/approx_kl": approx_kl.item(),
                "losses/clipfrac": np.mean(clipfracs),
                "losses/explained_variance": explained_var,
                "charts/SPS": int(global_step / (time.time() - start_time)),
            }, step=global_step)

            ppo_end_time = time.time()
            print(f"PPO update time: {ppo_end_time - ppo_start_time:.2f} seconds")
            torch.cuda.empty_cache()
            
        test(agent=ppo_net, num_episodes=args.num_test_episodes, global_step=global_step)
        
        torch.save(ppo_net.state_dict(), model_path)
    
    if args.use_trained:
        # Load the trained model
        ppo_net = PPONet(embed_dim=args.embedded_dim, max_uav_number=args.max_uav_number, map_size=(args.window_height*args.resolution, args.window_width*args.resolution), patch_size=args.patch_size, global_value=args.global_value).to(device)
        # Ottieni il percorso assoluto della root del progetto, basato su questo file
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        # Path al file dei pesi
        model_path = os.path.join(project_root, "neural_network", f"{args.exp_name}.pth")
        ppo_net.load_state_dict(torch.load(model_path, map_location='cuda:0'))

        test(ppo_net, 1, 0)
        
    if args.numerical_test:
        # Load the trained model
        ppo_net = PPONet(embed_dim=args.embedded_dim, max_uav_number=args.max_uav_number, map_size=(args.window_height*args.resolution, args.window_width*args.resolution), patch_size=args.patch_size, global_value=args.global_value).to(device)
        # Ottieni il percorso assoluto della root del progetto, basato su questo file
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        # Path al file dei pesi
        model_path = os.path.join(project_root, "neural_network", f"{args.exp_name}.pth")
        ppo_net.load_state_dict(torch.load(model_path))

        # Test the trained model
        test(ppo_net, args.num_test_episodes, 0)
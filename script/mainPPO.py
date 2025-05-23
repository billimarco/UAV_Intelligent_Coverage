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

from gym_cruising.memory.replay_memory_ppo import ReplayMemoryPPO, Transition
from gym_cruising.neural_network.MLP_policy_net import MLPPolicyNet
from gym_cruising.neural_network.deep_Q_net import DeepQNet, DoubleDeepQNet
from gym_cruising.neural_network.transformer_encoder_decoder import TransformerEncoderDecoder
from gym_cruising.neural_network.PPO_net import PPONet
import gym_cruising.utils.runtime_utils as utils


BEST_VALIDATION = 0.0
MAX_LAST_RCR = 0.0
EMBEDDED_DIM = 32

def compute_gae(rewards, values, dones):
    """
    Calcola GAE (Generalized Advantage Estimation) in modo vettorializzato
    per un singolo critic centralizzato, considerando più UAV.
    
    Args:
        rewards: (num_envs, num_steps, max_uav_number) - Rewards per ogni UAV.
        values: (num_envs, num_steps, 1) - Valori stimati per ogni stato (centralizzato).
        dones: (num_envs, num_steps, 1) - Flag di terminazione per ogni ambiente.
        
    Returns:
        advantages: (num_envs, num_steps, max_uav_number) - Advantages per ogni UAV.
        returns: (num_envs, num_steps, max_uav_number) - Returns per ogni UAV.
    """
    num_envs, num_steps, max_uav = rewards.shape

    # Estendi values con bootstrap finale (valori a 0)
    values_ext = torch.cat([
        values, 
        torch.zeros((num_envs, 1, 1), dtype=values.dtype, device=values.device)  # Aggiungi una colonna finale di zeri
    ], dim=1)  # → (num_envs, num_steps + 1, 1)

    advantages = torch.zeros_like(rewards)
    gae = torch.zeros((num_envs, max_uav), dtype=rewards.dtype, device=rewards.device)

    for t in reversed(range(num_steps)):
        next_values = values_ext[:, t + 1, 0]  # Riduci la dimensione dell'ultimo asse
        not_done = (1.0 - dones[:, t, 0].float()).unsqueeze(-1).expand(-1, max_uav) # Il flag di fine episodio per tutti gli UAV
        # Broadcasting per fare in modo che next_values e values abbiano la stessa forma di rewards
        next_values = next_values.unsqueeze(-1).expand(-1, max_uav)  # Ora next_values ha forma (num_envs, max_uav)
        values_current = values[:, t, 0].unsqueeze(-1).expand(-1, max_uav)  # Ora values ha forma (num_envs, max_uav)
        # Calcola il delta (la differenza tra reward e value)
        delta = rewards[:, t, :] + args.gamma * next_values * not_done - values_current
        # Aggiorna GAE
        gae = delta + args.gamma * args.gae_lambda * not_done * gae
        # Salva le advantages calcolate
        advantages[:, t, :] = gae

    # I ritorni sono la somma delle advantages e dei valori
    returns = advantages + values[:, :, 0].unsqueeze(-1).expand(-1, -1, max_uav)  # Broadcasting per adattarsi a (num_envs, num_steps, max_uav)

    return advantages, returns

def add_reward_padding(rewards):
    pad_count = args.max_uav_number - args.uav_number
    if pad_count > 0:

        # Costruisci padding come Tensor
        reward_padding = torch.zeros(pad_count, dtype=rewards.dtype, device=rewards.device)

        # Concatenazione Tensor
        rewards = torch.cat([rewards, reward_padding], dim=0)

    return rewards

def add_padding(actions, reward, logprobs, values):
    pad_count = args.max_uav_number - args.uav_number
    if pad_count > 0:

        # Costruisci padding come Tensor
        action_padding = torch.full((pad_count, 2), 100.0, dtype=actions.dtype, device=actions.device)
        reward_padding = torch.zeros(pad_count, dtype=reward.dtype, device=reward.device)
        logprob_padding = torch.full((pad_count,), -1e8, dtype=logprobs.dtype, device=logprobs.device)
        values_padding = torch.zeros(pad_count, dtype=values.dtype, device=values.device)

        # Concatenazione Tensor
        actions = torch.cat([actions, action_padding], dim=0)
        reward = torch.cat([reward, reward_padding], dim=0)
        logprobs = torch.cat([logprobs, logprob_padding], dim=0)
        values = torch.cat([values, values_padding], dim=0)

    return actions, reward, logprobs, values

def add_padding_state_uav(state, uav_number):
    # Padding per lo stato (in NumPy)
    padding = np.array([[0., 0.]])
    # Applica padding allo stato (UAV*2 → max_UAV*2)
    for i in range(uav_number*2, args.max_uav_number * 2):
        state = np.insert(state, i, padding, axis=0)
            
    return state

def process_state_batch(state_batch):
    """
    Estrae le info UAV e GU da un batch di stati e produce anche le maschere booleane.

    Returns:
        uav_info_batch: [B, max_uav, 4]
        connected_gu_positions_batch: [B, max_gu, 2]
        uav_mask: [B, max_uav] (True = reale, False = padding)
        gu_mask: [B, max_gu]  (True = reale, False = padding)
    """

    uav_info_list = []
    gu_pos_list = []
    uav_masks = []
    gu_masks = []

    for array in state_batch:
        uav_info, gu_positions = np.split(array, [args.max_uav_number * 2], axis=0)

        # UAV info: reshape e conversione
        uav_info = uav_info.reshape(args.max_uav_number, 4)
        uav_info_tensor = torch.from_numpy(uav_info).float().to(device)
        uav_info_list.append(uav_info_tensor)
        
        # UAV mask: True dove almeno un valore è non zero
        uav_mask = (uav_info_tensor.abs().sum(dim=-1) > 0)
        uav_masks.append(uav_mask)

        # GU positions: conversione
        gu_tensor = torch.from_numpy(gu_positions).float().to(device)
        gu_pos_list.append(gu_tensor)
        
        # GU mask: True dove almeno un valore è non zero
        gu_mask = (gu_tensor.abs().sum(dim=-1) > 0)
        gu_masks.append(gu_mask)

    # Stack UAV info (dimensione fissa)
    uav_info_batch = torch.stack(uav_info_list)                 # [B, max_uav, 4]
    uav_mask = torch.stack(uav_masks)                           # [B, max_uav]

    # Padding GU positions (dimensione variabile)
    max_gu = max(t.shape[0] for t in gu_pos_list)
    padded_gu_pos = []
    padded_gu_masks = []

    for gu, mask in zip(gu_pos_list, gu_masks):
        pad_len = max_gu - gu.shape[0]
        padded_gu = F.pad(gu, (0, 0, 0, pad_len), value=0.0)
        padded_mask = F.pad(mask, (0, pad_len), value=False)
        padded_gu_pos.append(padded_gu)
        padded_gu_masks.append(padded_mask)

    connected_gu_positions_batch = torch.stack(padded_gu_pos)  # [B, max_gu, 2]
    gu_mask = torch.stack(padded_gu_masks)                     # [B, max_gu]

    return uav_info_batch, connected_gu_positions_batch, uav_mask, gu_mask

def get_uniform_options():
    return ({
        "uav": args.uav_number,
        "gu": args.starting_gu_number*args.uav_number,
        "clustered": False,
        "clusters_number": 0,
        "variance": 0
    })

def get_clustered_options():
    variance = random.randint(args.clusters_variance_min, args.clusters_variance_max)

    return ({
        "uav": args.uav_number,
        "gu": args.starting_gu_number*args.uav_number,
        "clustered": True,
        "clusters_number": args.clusters_number*args.uav_number,
        "variance": variance
    })

def get_set_up():
    
    if args.uav_number == args.max_uav_number:
        args.uav_number = 1
    else:
        args.uav_number += 1

    sample = random.random()
    if sample > 0.5:
        options = get_clustered_options()
    else:
        options = get_uniform_options()

    return options

def test(agent, env, num_episodes=10, device=torch.device("cpu"), global_step=0, args=None):
    agent.eval()
    total_rewards = []
    rcr_values = []

    for ep in range(num_episodes):
        options = {
            "uav": 3,
            "gu": 120,
            "clustered": False,
            "clusters_number": 0,
            "variance": 0
        }
        np.random.seed(ep)
        state, info = env.reset(seed=ep, options=options)
        done = False
        steps = 1
        sum_episode_reward = 0
        sum_last_rcr = 0

        while not done:
            state = add_padding_state_uav(state, options["uav"])
            uav_info, connected_gu_positions, uav_mask, gu_mask = process_state_batch([state])
            
            # Inference
            with torch.no_grad():
                actions, _, _, _ = agent(
                    uav_info,                  # shape: (1, UAV, 4)    
                    connected_gu_positions,    # shape: (1, GU, 2)
                    uav_mask,                  # shape: (1, UAV)
                    gu_mask                    # shape: (1, GU)
                )

            # Rimuovi batch dim
            actions = actions.squeeze(0)     # shape: (UAV, 2)
            # Applica max speed scaling
            scaled_actions = actions * args.max_speed_uav   # (UAV, 2)
            real_actions = scaled_actions[uav_mask.squeeze(0)]  # (U_real, 2)
            # Passaggio ambiente
            next_state, reward, terminated, truncated, info = env.step(real_actions.cpu().numpy())
            if steps == 384:
                truncated = True
            done = terminated or truncated
            sum_episode_reward += sum(reward)
            state = next_state
            if done:
                sum_last_rcr += float(info['RCR'])
                break
        
        # Aggiungi l'RCR medio per l'episodio
        rcr_values.append(sum_last_rcr / (1 if done else 0))  # Media di RCR per episodio
        total_rewards.append(sum_episode_reward)
        print(f"Episode {ep + 1}: uav_number = {options['uav']}, starting_gu = {options['gu']}, clusters = {options['clusters_number']}, reward = {sum_episode_reward:.2f}, RCR = {sum_last_rcr:.2f}")

    # Statistiche
    mean_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    mean_rcr = np.mean(rcr_values)
    std_rcr = np.std(rcr_values)

    print(f"\nTest completed over {num_episodes} episodes.")
    print(f"Average reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Average RCR : {mean_rcr:.2f} ± {std_rcr:.2f}")

    # Logging su Weights & Biases
    if wandb.run is not None:
        wandb.log({
            "test/mean_reward": mean_reward,
            "test/std_reward": std_reward,
            "test/mean_rcr": mean_rcr,
            "test/std_rcr": std_rcr,
        }, step=global_step)
        
    '''
    if total_reward > BEST_VALIDATION:
                BEST_VALIDATION = total_reward
                # save the best validation nets
                torch.save(transformer_policy.state_dict(), '../neural_network/rewardTransformer.pth')
                torch.save(mlp_policy.state_dict(), '../neural_network/rewardMLP.pth')
                torch.save(deep_Q_net_policy.state_dict(), '../neural_network/rewardDeepQ.pth')

    if sum_last_rcr > MAX_LAST_RCR:
        MAX_LAST_RCR = sum_last_rcr
        # save the best validation nets
        torch.save(transformer_policy.state_dict(), '../neural_network/maxTransformer.pth')
        torch.save(mlp_policy.state_dict(), '../neural_network/maxMLP.pth')
        torch.save(deep_Q_net_policy.state_dict(), '../neural_network/maxDeepQ.pth')
    '''

    return mean_reward, std_reward
    
if __name__ == "__main__":
    args = utils.parse_args()

    print(f"Available GPUs: {torch.cuda.device_count()}")
    best_gpu, free_mem = utils.get_most_free_gpu()
    if best_gpu is not None and args.cuda:
        print(f"Using GPU {best_gpu} with {free_mem} MB free.")
        device = torch.device(f"cuda:{best_gpu}")
    else:
        print("No GPU available, using CPU.")
        device = torch.device("cpu")

    if args.train:
        
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
                save_code=True
            )
            wandb.define_metric("test/mean_reward", step_metric="global_step")
            wandb.define_metric("test/std_reward", step_metric="global_step")
            wandb.define_metric("test/mean_rcr", step_metric="global_step")
            wandb.define_metric("test/std_rcr", step_metric="global_step")
            wandb.define_metric("charts/learning_rate", step_metric="global_step")
            wandb.define_metric("losses/value_loss", step_metric="global_step")
            wandb.define_metric("losses/policy_loss", step_metric="global_step")
            wandb.define_metric("losses/entropy", step_metric="global_step")
            wandb.define_metric("losses/old_approx_kl", step_metric="global_step")
            wandb.define_metric("losses/approx_kl", step_metric="global_step")
            wandb.define_metric("losses/clipfrac", step_metric="global_step")
            wandb.define_metric("losses/explained_variance", step_metric="global_step")
            wandb.define_metric("charts/SPS", step_metric="global_step")


        env = gym.make('gym_cruising:Cruising-v0', args=args, render_mode=args.render_mode)

        ppo_net = PPONet(embed_dim=EMBEDDED_DIM).to(device)
        
        optimizer = optim.Adam(ppo_net.parameters(), lr=args.learning_rate, weight_decay=1e-5)
        
        replay_buffer = ReplayMemoryPPO(args.batch_size)
        
        global_step = 0
        start_time = time.time()
        
        for update in range(0, args.updates_per_env):
            print(f"Update {update + 1}/{args.updates_per_env}")

            if update % 20 == 0 and update % args.updates_per_env != 0:
                print("\n<------------------------->")
                print(f"Testing at update {update}")
                test_time = time.time()
                test(agent=ppo_net, env=env, num_episodes=12, device=device, global_step=global_step, args=args)
                print(f"Tested! Time elaplesed {time.time() - test_time}")
                print("<------------------------->\n")
                start_time = time.time()
            
            if args.anneal_lr:
                frac = 1.0 - (update - 1.0) / args.updates_per_env
                lrnow = frac * args.learning_rate
                optimizer.param_groups[0]["lr"] = lrnow

            # Buffer temporanei per un singolo episodio/rollout
            states_list = []
            actions_tensor = torch.zeros((args.num_envs, args.num_steps, args.max_uav_number, 2), dtype=torch.float32).to(device)
            log_probs_tensor = torch.zeros((args.num_envs, args.num_steps, args.max_uav_number), dtype=torch.float32).to(device)
            rewards_tensor = torch.zeros((args.num_envs, args.num_steps, args.max_uav_number), dtype=torch.float32).to(device)
            dones_tensor = torch.zeros((args.num_envs, args.num_steps, 1), dtype=torch.bool).to(device)
            values_tensor = torch.zeros((args.num_envs, args.num_steps, 1), dtype=torch.float32).to(device)

            # Rollout
            for i in range(args.num_envs):
                options = get_set_up()
                state, info = env.reset(seed=int(time.perf_counter()), options=options)
                for step in range(args.num_steps):
                    global_step += 1
                    
                    state = add_padding_state_uav(state, options["uav"])

                    uav_info, connected_gu_positions, uav_mask, gu_mask = process_state_batch([state])
                    
                    # Ottieni azioni da PPO
                    with torch.no_grad():
                        actions, logprobs, entropy, values = ppo_net(
                            uav_info,                  # shape: (1, UAV, 4)    
                            connected_gu_positions,    # shape: (1, GU, 2)
                            uav_mask,                  # shape: (1, UAV)
                            gu_mask                    # shape: (1, GU)
                        )

                    # Rimuovi batch dim
                    actions = actions.squeeze(0)     # shape: (UAV, 2)
                    logprobs = logprobs.squeeze(0)    # shape: (UAV,)
                    values = values.squeeze(0)        # shape: (1,)


                    # Applica max speed scaling
                    scaled_actions = actions * args.max_speed_uav   # (UAV, 2)
                    
                    real_actions = scaled_actions[uav_mask.squeeze(0)]  # (U_real, 2)
                    
                    next_state, reward, terminated, truncated, _ = env.step(real_actions.cpu().numpy())
                    
                    reward = torch.tensor(reward).to(device)
                    terminated = torch.tensor(terminated).to(device)
                    # Store the transition in memory
                    rewards = add_reward_padding(reward)
                    
                    # Memorizza i dati nel buffer temporaneo
                    states_list.append(state)
                    actions_tensor[i, step, :, :] = actions
                    log_probs_tensor[i, step, :] = logprobs
                    rewards_tensor[i, step, :] = rewards
                    dones_tensor[i, step, :] = terminated
                    values_tensor[i, step, :] = values

                    state = next_state
                    
            # Calcolare gli advantages e i returns usando GAE
            advantages_tensor, returns_tensor = compute_gae(
                rewards=rewards_tensor,
                values=values_tensor,
                dones=dones_tensor
            )
            
            
            # Supponiamo che le variabili siano già definite con shape (num_envs, num_steps, ...)

            # Flatten (num_envs, num_steps, ...) → (B, ...)
            flat_actions = actions_tensor.reshape(args.batch_size, args.max_uav_number, 2)
            flat_log_probs = log_probs_tensor.reshape(args.batch_size, args.max_uav_number)
            flat_rewards = rewards_tensor.reshape(args.batch_size, args.max_uav_number)
            flat_dones = dones_tensor.reshape(args.batch_size, 1)
            flat_values = values_tensor.reshape(args.batch_size, 1)
            flat_advantages = advantages_tensor.reshape(args.batch_size, args.max_uav_number)
            flat_returns = returns_tensor.reshape(args.batch_size, args.max_uav_number)

            clipfracs = []

            for epoch in range(args.update_epochs):
                b_inds = torch.randperm(args.batch_size, device=device)
                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]

                    mb_states = [states_list[i] for i in mb_inds.tolist()]              # [MB, ...]
                    mb_actions = flat_actions[mb_inds]                                  # [MB, UAV, 2]
                    mb_logprobs = flat_log_probs[mb_inds]                               # [MB, UAV]
                    mb_rewards = flat_rewards[mb_inds]                                  # [MB, UAV]
                    mb_dones = flat_dones[mb_inds].expand(-1, args.max_uav_number)      # [MB, 1*UAV]
                    mb_values = flat_values[mb_inds].expand(-1, args.max_uav_number)    # [MB, 1*UAV]
                    mb_advantages = flat_advantages[mb_inds]                            # [MB, UAV]
                    mb_returns = flat_returns[mb_inds]                                  # [MB, UAV]

                    # Estrai info di stato UAV/GU
                    mb_state_uav, mb_state_connected_gu, mb_uav_mask, mb_gu_mask = process_state_batch(mb_states)

                    # Forward della rete
                    mb_state_tokens = ppo_net.backbone_forward(mb_state_uav, mb_state_connected_gu, mb_uav_mask, mb_gu_mask)
                    
                    '''
                    # Valore utilizzato per il padding (assumendo che sia 100.0 per mb_actions)
                    padding_logprob = -1e8

                    # Crea una maschera booleana che indica i droni reali (True per UAV reali, False per UAV fittizi)
                    mb_index_mask = (mb_logprobs != padding_logprob)  # La maschera è True per i droni reali

                    mb_masked_size = len(mb_index_mask)
                    '''
                    # slice i-th UAV's tokens [masked_batch_size, 1, EMBEDDED_DIM]
                    mb_masked_state_tokens = mb_state_tokens[mb_uav_mask]
                    mb_logprobs_masked = mb_logprobs[mb_uav_mask]          # Maschera su logprobs
                    mb_rewards_masked = mb_rewards[mb_uav_mask]            # Maschera su rewards
                    mb_dones_masked = mb_dones[mb_uav_mask]                # Maschera su dones
                    mb_values_masked = mb_values[mb_uav_mask]              # Maschera su values
                    mb_advantages_masked = mb_advantages[mb_uav_mask]      # Maschera su advantages
                    mb_returns_masked = mb_returns[mb_uav_mask]            # Maschera su returns

                    _, newlogprob, entropy, newvalue = ppo_net.get_action_and_value(mb_state_tokens, mb_uav_mask)
                    #TODO: mascherare newlogprob e newvalue, entropy
                    mb_newlogprob_masked = newlogprob[mb_uav_mask]  # Maschera su logprobs
                    mb_newvalues_masked = newvalue.unsqueeze(-1).expand(-1, args.max_uav_number)[mb_uav_mask]      # Maschera su values
                    mb_entropy_masked = entropy[mb_uav_mask]      # Maschera su entropy
                    
                    
                    logratio = mb_newlogprob_masked - mb_logprobs_masked
                    ratio = logratio.exp()

                    with torch.no_grad():
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                    # Normalizza le advantages
                    if args.norm_adv:
                        mb_advantages_masked = (mb_advantages_masked - mb_advantages_masked.mean()) / (mb_advantages_masked.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages_masked * ratio
                    pg_loss2 = -mb_advantages_masked * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    if args.clip_vloss:
                        v_loss_unclipped = (mb_newvalues_masked - mb_returns_masked) ** 2
                        v_clipped = mb_values_masked + torch.clamp(
                            mb_newvalues_masked - mb_values_masked,
                            -args.clip_coef,
                            args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - mb_returns_masked) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((mb_newvalues_masked - mb_returns_masked) ** 2).mean()

                    entropy_loss = mb_entropy_masked.mean()
                    total_loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                    optimizer.zero_grad()
                    total_loss.backward()
                    nn.utils.clip_grad_norm_(ppo_net.parameters(), args.max_grad_norm)
                    optimizer.step()

            # Estrai info di stato UAV/GU
            _ , _ , b_uav_mask, _ = process_state_batch(states_list)
            # Log training metrics
            y_pred = flat_values.expand(-1, args.max_uav_number)[b_uav_mask].cpu().numpy()
            y_true = flat_returns[b_uav_mask].cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            
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

        test(agent=ppo_net, env=env, num_episodes=12, device=device, global_step=global_step, args=args)
        # Ottieni il percorso assoluto della root del progetto, basato su questo file
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        # Path al file dei pesi
        model_path = os.path.join(project_root, "neural_network", "PPO.pth")
        torch.save(ppo_net.state_dict(), model_path)
    
    if args.use_trained:
        env = gym.make('gym_cruising:Cruising-v0', args=args, render_mode=args.render_mode)
        # Load the trained model
        ppo_net = PPONet(embed_dim=EMBEDDED_DIM).to(device)
        # Ottieni il percorso assoluto della root del progetto, basato su questo file
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        # Path al file dei pesi
        model_path = os.path.join(project_root, "neural_network", "PPO.pth")
        ppo_net.load_state_dict(torch.load(model_path))
        
        options = ({
            "uav": 3,
            "gu": 120,
            "clustered": 1,  # 0 for uniform, 1 for clustered
            "clusters_number": 3,
            "variance": 100000
        })
        
        time = int(time.perf_counter())
        print("Time: ", time)
        np.random.seed(time)
        state, info = env.reset(seed=time, options=options)
        steps = 1

        while True:
            state = add_padding_state_uav(state, options["uav"])
            uav_info, connected_gu_positions, uav_mask, gu_mask = process_state_batch([state])
            
            # Inference
            with torch.no_grad():
                actions, _, _, _ = ppo_net(
                    uav_info,                  # shape: (1, UAV, 4)    
                    connected_gu_positions,    # shape: (1, GU, 2)
                    uav_mask,                  # shape: (1, UAV)
                    gu_mask                    # shape: (1, GU)
                )

            # Rimuovi batch dim
            actions = actions.squeeze(0)     # shape: (UAV, 2)
            # Applica max speed scaling
            scaled_actions = actions * args.max_speed_uav   # (UAV, 2)
            real_actions = scaled_actions[uav_mask.squeeze(0)]  # (U_real, 2)
            # Passaggio ambiente
            next_state, reward, terminated, truncated, info = env.step(real_actions.cpu().numpy())

            if steps == 300:
                truncated = True
            done = truncated or info['Collision']

            # if steps % 70 == 0 and steps != 0:
            #     state = env.reset_gu(options=options)

            state = next_state
            steps += 1

            if done:
                last_RCR = float(info['RCR'])
                break

        env.close()
        print("Last RCR: ", last_RCR)
    
    if args.numerical_test:
        env = gym.make('gym_cruising:Cruising-v0', args=args, render_mode=args.render_mode)
        # Load the trained model
        ppo_net = PPONet(embed_dim=EMBEDDED_DIM).to(device)
        # Ottieni il percorso assoluto della root del progetto, basato su questo file
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        # Path al file dei pesi
        model_path = os.path.join(project_root, "neural_network", "PPO.pth")
        ppo_net.load_state_dict(torch.load(model_path))
        
        options = ({
            "uav": 3,
            "gu": 120,
            "clustered": 1,  # 0 for uniform, 1 for clustered
            "clusters_number": 3,
            "variance": 100000
        })
        seeds = [5522, 6004, 9648, 8707, 5930, 7411, 8761, 6748, 283, 4880, 7541, 2423, 9652, 4469, 3508, 8969, 8222, 6413,
                3133, 273, 1431, 9688, 6940, 9998, 7097, 1130, 7583, 4018, 116, 1626, 9579, 2641, 8602, 3335, 7980, 3434,
                1553, 4961, 2024, 2834, 6610, 979, 9405, 4866, 7437, 3827, 3735, 2038, 1360, 5202, 4870, 1945, 382, 7101,
                2402, 7235, 8967, 2315, 5955, 4300, 1775, 8136, 1050, 6385, 1068, 5451, 9772, 2331, 6174, 4393, 4873, 7296,
                1780, 5299, 4919, 625, 87, 2240, 2815, 5020, 43, 211, 17, 1243, 97, 23, 57, 1111, 2013, 571, 1729,
                333, 907, 1025, 621162, 513527, 268574, 233097, 342217, 310673]
        
        tot_rewards = []
        collisions = 0
        for j, seed in enumerate(seeds):
            print("Test ", str(j))
            np.random.seed(seed)
            state, info = env.reset(seed=seed, options=options)
            steps = 1
            while True:
                state = add_padding_state_uav(state, options["uav"])
                uav_info, connected_gu_positions, uav_mask, gu_mask = process_state_batch([state])
                
                # Inference
                with torch.no_grad():
                    actions, _, _, _ = ppo_net(
                        uav_info,                  # shape: (1, UAV, 4)    
                        connected_gu_positions,    # shape: (1, GU, 2)
                        uav_mask,                  # shape: (1, UAV)
                        gu_mask                    # shape: (1, GU)
                    )

                # Rimuovi batch dim
                actions = actions.squeeze(0)     # shape: (UAV, 2)
                # Applica max speed scaling
                scaled_actions = actions * args.max_speed_uav   # (UAV, 2)
                real_actions = scaled_actions[uav_mask.squeeze(0)]  # (U_real, 2)
                # Passaggio ambiente
                next_state, reward, terminated, truncated, info = env.step(real_actions.cpu().numpy())

                if steps == 300:
                    truncated = True
                done = truncated or info['Collision']

                # if steps % 70 == 0 and steps != 0:
                #     state = env.reset_gu(options=options)

                state = next_state
                steps += 1
                if done:
                    tot_rewards.append(float(info['RCR']))
                    if info['Collision']:
                        collisions += 1
                    break
                
            env.close()
            
        print("Mean reward: ", sum(tot_rewards) / len(tot_rewards))
        print("Collisioni: ", collisions)
        

        # Test the trained model
        #test(agent, test_envs, tasks, global_step, True, True)
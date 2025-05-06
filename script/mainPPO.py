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


global_step = 0

BEST_VALIDATION = 0.0
MAX_LAST_RCR = 0.0
EMBEDDED_DIM = 32

def generalized_advantage_estimation(rewards, dones, values, logprobs, actions, obs):
    # Advantage computation
    with torch.no_grad():

        next_value = agent.get_value(obs[-1]).reshape(1, -1)
        if args.gae:
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = ~ dones[-1]
                    nextvalues = next_value
                else:
                    nextnonterminal = ~ dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values
        else:
            returns = torch.zeros_like(rewards).to(device)
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = ~ dones[-1]
                    next_return = next_value
                else:
                    nextnonterminal = ~ dones[t + 1]
                    next_return = returns[t + 1]
                returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
            advantages = returns - values


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


        env = gym.make('gym_cruising:Cruising-v0', args=args, render_mode='rgb_array')
        env.reset(args.seed)

        ppo_net = PPONet(embed_dim=EMBEDDED_DIM).to(device)
        
        replay_buffer = ReplayMemoryPPO(args.batch_size)
        
        global_step = 0
        start_time = time.time()
        
        for update in range(0, args.updates_per_env):

            if update % 20 == 0 and update % args.updates_per_env != 0:
                test_time = time.time()
                test(agent, test_envs, tasks, global_step)#TODO try
                print(f"Tested! Time elaplesed {time.time() - test_time}")
                print()
                start_time = time.time()

            episode_rewards = np.zeros(args.num_envs)
            episode_lenghts = np.zeros(args.num_envs)

            episode_step = np.zeros(args.num_envs)
            
            replay_buffer.clear()
            # Rollout
            for i in range(args.num_envs):
                options = get_set_up()
                state, info = env.reset(seed=int(time.perf_counter()), options=options)
                # Rollout
                for step in range(args.num_steps):
                    global_step += 1
                    
                    # Estrai stato
                    uav_info, connected_gu_positions = np.split(state, [options['uav'] * 2], axis=0)
                    uav_info = uav_info.reshape(options['uav'], 4)

                    # Converti in tensori
                    uav_info = torch.from_numpy(uav_info).float().to(device)
                    connected_gu_positions = torch.from_numpy(connected_gu_positions).float().to(device)

                    # Ottieni azioni da PPO
                    with torch.no_grad():
                        actions, logprobs, entropy, values = ppo_net(
                            connected_gu_positions.unsqueeze(0),  # shape: (1, GU, 2)
                            uav_info.unsqueeze(0)                 # shape: (1, UAV, 4)
                        )

                    # Rimuovi batch dim
                    actions = actions.squeeze(0).cpu().numpy()      # shape: (UAV, 2)
                    logprobs = logprobs.squeeze(0).cpu().numpy()    # shape: (UAV,)
                    values = values.item()                          # scalare, valore globale dello stato

                    # Applica max speed scaling
                    scaled_actions = actions * args.max_speed_uav   # (UAV, 2)

                    next_state, reward, terminated, truncated, _ = env.step(actions)
                    # TODO: FERMO QUI
                    # Store the transition in memory
                    state_padding, next_state_padding, actions_padding, reward_padding = add_padding(state, next_state, scaled_actions,
                                                                                                reward,
                                                                                                options['uav'])
                    replay_buffer.push(state_padding, actions_padding, next_state_padding, reward_padding,
                                            int(terminated))
                    state = next_state
                    
            

            generalized_advantage_estimation(rewards, dones, values, logprobs, actions, obs)

            # Flatten batch
            b_obs = obs.reshape((-1,) + envs.observation_space.shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + envs.action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            # Optimize policy and value networks
            b_inds = np.arange(args.batch_size)
            clipfracs = []

            for epoch in range(args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]
                    
                    transitions_uniform = replay_buffer_uniform.get_minibatch(mb_inds)
                    # This converts batch-arrays of Transitions to Transition of batch-arrays.
                    batch_uniform = Transition(*zip(*transitions_uniform))

                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                        b_obs[mb_inds], b_actions.long()[mb_inds]
                    )
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -args.clip_coef,
                            args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                    if args.ewc:
                        ewc_loss = ewc.compute_ewc_loss()
                        loss += ewc_loss

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()
                    if args.s_p:
                        shrink_perturb(agent)
                        
                if args.target_kl is not None and approx_kl > args.target_kl:
                    break

            # Log training metrics
            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            if args.ewc:
                writer.add_scalar("losses/ewc", ewc_loss.item(), global_step)
                writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
                writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
                writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
                writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
                writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
                writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
                writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
                writer.add_scalar("losses/explained_variance", explained_var, global_step)
            # print("SPS:", int(global_step / (time.time() - start_time)))
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

            test(agent, test_envs, tasks, global_step, True, True)


        def select_actions_epsilon(state, uav_number):
            uav_info, connected_gu_positions = np.split(state, [uav_number * 2], axis=0)
            uav_info = uav_info.reshape(uav_number, 4)
            uav_info = torch.from_numpy(uav_info).float().to(device)
            connected_gu_positions = torch.from_numpy(connected_gu_positions).float().to(device)
            action = []
            with torch.no_grad():
                tokens = transformer_policy(connected_gu_positions.unsqueeze(0), uav_info.unsqueeze(0)).squeeze(0)
            global_step += 1
            for i in range(uav_number):
                if global_step < start_steps:
                    output = np.random.uniform(low=-1.0, high=1.0, size=2)
                    output = output * args.max_speed_uav
                    action.append(output)
                else:
                    with torch.no_grad():
                        # return action according to MLP [vx, vy] + epsilon noise
                        output = mlp_policy(tokens[i])
                        output = output + (torch.randn(2) * sigma_policy).to(
                            device)
                        output = torch.clip(output, -1.0, 1.0)
                        output = output.cpu().numpy().reshape(2)
                        output = output * args.max_speed_uav
                        action.append(output)
            return action
        
        def select_actions(state, uav_number):
            uav_info, connected_gu_positions = np.split(state, [uav_number * 2], axis=0)
            uav_info = uav_info.reshape(uav_number, 4)
            uav_info = torch.from_numpy(uav_info).float().to(device)
            connected_gu_positions = torch.from_numpy(connected_gu_positions).float().to(device)
            action = []
            with torch.no_grad():
                tokens = transformer_policy(connected_gu_positions.unsqueeze(0), uav_info.unsqueeze(0)).squeeze(0)
            for i in range(uav_number):
                with torch.no_grad():
                    # return action according to MLP [vx, vy]
                    output = mlp_policy(tokens[i])
                    output = output.cpu().numpy().reshape(2)
                    output = output * args.max_speed_uav
                    action.append(output)
            return action
        
        def process_state_batch(state_batch):
            """
            Estrae e converte le informazioni UAV e le posizioni dei GU connessi
            da un batch di stati o stati successivi.

            Args:
                state_batch (list of np.ndarray): Lista di stati da elaborare.
                optimization_steps (int): Numero di passi di ottimizzazione (usato per il reshape).
                device (torch.device): Dispositivo PyTorch su cui caricare i tensori (es. "cuda" o "cpu").

            Returns:
                tuple: (uav_info_batch, connected_gu_positions_batch)
                    - uav_info_batch: tensore di forma [batch_size, optimization_steps, 4]
                    - connected_gu_positions_batch: tensore con padding [batch_size, max_gu, feature_dim]
            """

            uav_info_list = []
            gu_pos_list = []

            for array in state_batch:
                uav_info, gu_positions = np.split(array, [args.max_uav_number * 2], axis=0)

                # UAV info: reshape e conversione
                uav_info = uav_info.reshape(args.max_uav_number, 4)
                uav_info_tensor = torch.from_numpy(uav_info).float().to(device)
                uav_info_list.append(uav_info_tensor)

                # GU positions: conversione
                gu_tensor = torch.from_numpy(gu_positions).float().to(device)
                gu_pos_list.append(gu_tensor)

            # Stack UAV info (dimensione fissa)
            uav_info_batch = torch.stack(uav_info_list)

            # Padding GU positions (dimensione variabile)
            max_len = max(t.shape[0] for t in gu_pos_list)
            padded_gu_pos = [
                F.pad(t, (0, 0, 0, max_len - t.shape[0]), "constant", 0)
                for t in gu_pos_list
            ]
            connected_gu_positions_batch = torch.stack(padded_gu_pos)

            return uav_info_batch, connected_gu_positions_batch

        def optimize_model():
            global BATCH_SIZE
            torch.autograd.set_detect_anomaly(True)

            if len(replay_buffer_uniform) < 5000 or len(replay_buffer_clustered) < 5000:
                return

            transitions_uniform = replay_buffer_uniform.sample(int(BATCH_SIZE / 2))
            # This converts batch-arrays of Transitions to Transition of batch-arrays.
            batch_uniform = Transition(*zip(*transitions_uniform))

            transitions_clustered = replay_buffer_clustered.sample(int(BATCH_SIZE / 2))
            # This converts batch-arrays of Transitions to Transition of batch-arrays.
            batch_clustered = Transition(*zip(*transitions_clustered))

            states_batch = batch_uniform.states + batch_clustered.states
            actions_batch = batch_uniform.actions + batch_clustered.actions
            actions_batch = tuple(
                [torch.tensor(array, dtype=torch.float32) for array in sublist] for sublist in actions_batch)
            rewards_batch = batch_uniform.rewards + batch_clustered.rewards
            rewards_batch = torch.tensor(rewards_batch, dtype=torch.float32).unsqueeze(1).to(device)
            next_states_batch = batch_uniform.next_states + batch_clustered.next_states
            terminated_batch = batch_uniform.terminated + batch_clustered.terminated
            terminated_batch = torch.tensor(terminated_batch, dtype=torch.float32).unsqueeze(1).to(device)
            
            state_uav_info_batch, state_connected_gu_positions_batch = process_state_batch(states_batch)

            next_state_uav_info_batch, next_state_connected_gu_positions_batch = process_state_batch(next_states_batch)
            
            # get tokens from batch of states and next states
            with torch.no_grad():
                tokens_batch_next_states_target = transformer_target(next_state_connected_gu_positions_batch,
                                                                    next_state_uav_info_batch)  # [BATCH_SIZE, args.max_number_uav, EMBEDDED_DIM]
                tokens_batch_states_target = transformer_target(state_connected_gu_positions_batch,
                                                                state_uav_info_batch)  # [BATCH_SIZE, args.max_number_uav, EMBEDDED_DIM]
            tokens_batch_states = transformer_policy(state_connected_gu_positions_batch,
                                                    state_uav_info_batch)  # [BATCH_SIZE, args.max_number_uav, EMBEDDED_DIM]

            loss_Q = 0.0
            loss_policy = 0.0
            loss_transformer = 0.0

            for i in range(args.max_uav_number):
                # index mask for not padded current uav in batch
                index_mask = [
                    k for k, lista in enumerate(actions_batch)
                    if not torch.equal(lista[i], torch.tensor([100., 100.]))
                ]

                masked_batch_size = len(index_mask)

                # UPDATE Q-FUNCTION
                with torch.no_grad():
                    # slice i-th UAV's tokens [masked_batch_size, 1, EMBEDDED_DIM]
                    current_batch_tensor_tokens_next_states_target = tokens_batch_next_states_target[index_mask, i:i + 1,:].squeeze(1)
                    output_batch = mlp_target(current_batch_tensor_tokens_next_states_target)
                    
                    # noise generation for target next states actions according to N(0,sigma)
                    noise = (torch.randn((masked_batch_size, 2)) * sigma).to(device)
                    
                    # Clipping of noise
                    clipped_noise = torch.clip(noise, -c, c)
                    output_batch = torch.clip(output_batch + clipped_noise, -1.0, 1.0)
                    output_batch = output_batch * args.max_speed_uav  # actions batch for UAV i-th [masked_batch_size, 2]
                    Q1_values_batch, Q2_values_batch = deep_Q_net_target(current_batch_tensor_tokens_next_states_target, output_batch)
                    current_uav_rewards = rewards_batch[..., i]
                    current_y_batch = current_uav_rewards[index_mask] + GAMMA * (
                            1.0 - terminated_batch[index_mask]) * torch.min(Q1_values_batch,
                                                                            Q2_values_batch)
                # slice i-th UAV's tokens [masked_batch_size, 1, EMBEDDED_DIM]
                current_batch_tensor_tokens_states = tokens_batch_states[index_mask, i:i + 1, :].squeeze(1)
                # Concatenate i-th UAV's actions along the batch size [masked_batch_size, 2]
                current_batch_actions = torch.cat([action[i].unsqueeze(0) for action in actions_batch], dim=0).to(device)
                Q1_values_batch, Q2_values_batch = deep_Q_net_policy(current_batch_tensor_tokens_states, current_batch_actions[index_mask])
                # criterion = nn.MSELoss()
                criterion = torch.nn.HuberLoss()
                # Optimize Deep Q Net
                loss_Q = loss_Q + criterion(Q1_values_batch, current_y_batch) + criterion(Q2_values_batch, current_y_batch)

                criterion = nn.MSELoss()
                # UPDATE POLICY
                # slice i-th UAV's tokens [masked_batch_size, 1, EMBEDDED_DIM]
                current_batch_tensor_tokens_states_target = tokens_batch_states_target[index_mask, i:i + 1, :].squeeze(1)
                output_batch = mlp_policy(current_batch_tensor_tokens_states_target)
                output_batch = output_batch * args.max_speed_uav  # actions batch for UAV i-th [masked_batch_size, 2]
                Q1_values_batch, Q2_values_batch = deep_Q_net_policy(current_batch_tensor_tokens_states_target, output_batch)
                loss_policy = loss_policy - Q1_values_batch.mean()
                loss_transformer = loss_transformer + criterion(current_batch_tensor_tokens_states, current_batch_tensor_tokens_states_target)

            # log metrics to wandb
            wandb.log({"loss_Q": loss_Q, "loss_policy": loss_policy, "loss_transformer": loss_transformer})

            optimizer_deep_Q.zero_grad()
            optimizer_transformer.zero_grad()
            loss_Q.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(deep_Q_net_policy.parameters(), 5)  # clip_grad_value_
            torch.nn.utils.clip_grad_norm_(transformer_policy.parameters(), 5)  # clip_grad_value_
            optimizer_deep_Q.step()
            # Optimize Transformer Net
            optimizer_transformer.step()

            if global_step % policy_delay == 0:
                # Optimize Policy Net MLP
                optimizer_mlp.zero_grad()
                loss_policy.backward()
                torch.nn.utils.clip_grad_norm_(mlp_policy.parameters(), 5)  # clip_grad_value_
                optimizer_mlp.step()

                soft_update_target_networks()

        def soft_update(policy_model, target_model, beta):
            """
            Esegue un soft update dei pesi del modello target verso il modello policy.
            """
            target_state_dict = target_model.state_dict()
            policy_state_dict = policy_model.state_dict()
            for key in policy_state_dict:
                target_state_dict[key] = beta * policy_state_dict[key] + (1.0 - beta) * target_state_dict[key]
            target_model.load_state_dict(target_state_dict)

        def soft_update_target_networks():
            soft_update(transformer_policy, transformer_target, BETA)
            soft_update(mlp_policy, mlp_target, BETA)
            soft_update(deep_Q_net_policy, deep_Q_net_target, BETA)

        def get_uniform_options():
            return ({
                "uav": args.uav_number,
                "gu": args.starting_gu_number,
                "clustered": False,
                "clusters_number": 0,
                "variance": 0
            })

        def get_clustered_options():
            variance = random.randint(args.clusters_variance_min, args.clusters_variance_max)

            if args.uav_number == 1:
                clusters_number = random.randint(args.clusters_number, args.clusters_number+1)
            elif args.uav_number == 2:
                clusters_number = random.randint(args.clusters_number*2, args.clusters_number*2+2)
            else:
                clusters_number = random.randint(args.clusters_number*3, args.clusters_number*3+3)

            return ({
                "uav": args.uav_number,
                "gu": args.starting_gu_number,
                "clustered": True,
                "clusters_number": clusters_number,
                "variance": variance
            })

        def get_set_up():
            
            if args.uav_number == args.max_uav_number:
                args.uav_number = 1
            else:
                args.uav_number += 1

            sample = random.random()
            if sample > 0.3:
                options = get_clustered_options()
            else:
                options = get_uniform_options()

            return options


        def add_padding(state, next_state, actions, reward, uav_number):
            padding = np.array([[0., 0.]])
            action_padding = [100., 100.]
            for i in range(args.uav_number*2, args.max_uav_number*2):
                state = np.insert(state, i, padding, axis=0)
                next_state = np.insert(next_state, i, padding, axis=0)
            for i in range(args.uav_number, args.max_uav_number):
                actions.append(action_padding)
                reward.append(0.)
            return state, next_state, actions, reward


        def validate():
            global BEST_VALIDATION
            global MAX_LAST_RCR
            reward_sum_uniform = 0.0
            reward_sum_clustered = 0.0
            sum_last_rcr = 0.0
            
            options = ({
                       "uav": 1,
                       "gu": 30,
                       "clustered": True,
                       "clusters_number": 1,
                       "variance": 100000
                   },
                   {
                       "uav": 2,
                       "gu": 60,
                       "clustered": True,
                       "clusters_number": 2,
                       "variance": 100000
                   },
                   {
                       "uav": 3,
                       "gu": 90,
                       "clustered": True,
                       "clusters_number": 3,
                       "variance": 100000
                   })

            seeds = [42, 751, 853]
            for i, seed in enumerate(seeds):
                np.random.seed(seed)
                state, info = env.reset(seed=seed, options=options[i])
                steps = 1
                while True:
                    actions = select_actions(state, options[i]['uav'])
                    next_state, reward, terminated, truncated, info = env.step(actions)
                    reward_sum_clustered += sum(reward)

                    if steps == 300:
                        truncated = True
                    done = terminated or truncated

                    state = next_state
                    steps += 1

                    if done:
                        sum_last_rcr += float(info['RCR'])
                        break

            wandb.log({"reward_clustered": reward_sum_clustered})
            
            options = ({
                       "uav": 1,
                       "gu": 30,
                       "clustered": False,
                       "clusters_number": 0,
                       "variance": 0
                   },
                   {
                       "uav": 2,
                       "gu": 60,
                       "clustered": False,
                       "clusters_number": 0,
                       "variance": 0
                   },
                   {
                       "uav": 3,
                       "gu": 90,
                       "clustered": False,
                       "clusters_number": 0,
                       "variance": 0
                   })

            seeds = [54321, 1181, 3475]
            for i, seed in enumerate(seeds):
                np.random.seed(seed)
                state, info = env.reset(seed=seed, options=options[i])
                steps = 1
                while True:
                    actions = select_actions(state, options[i]['uav'])
                    next_state, reward, terminated, truncated, info = env.step(actions)
                    reward_sum_uniform += sum(reward)

                    if steps == 300:
                        truncated = True
                    done = terminated or truncated

                    state = next_state
                    steps += 1

                    if done:
                        sum_last_rcr += float(info['RCR'])
                        break

            wandb.log({"reward_uniform": reward_sum_uniform})
            wandb.log({"max_rcr": sum_last_rcr})

            total_reward = reward_sum_clustered + reward_sum_uniform

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

    
        if torch.cuda.is_available():
            num_episodes = 8000
        else:
            num_episodes = 100

        print("START UAV COOPERATIVE COVERAGE TRAINING")

        for i_episode in range(0, num_episodes, 1):
            print("Episode: ", i_episode)
            options = get_set_up()
            state, info = env.reset(seed=int(time.perf_counter()), options=options)
            steps = 1
            while True:
                actions = select_actions_epsilon(state, options['uav'])
                next_state, reward, terminated, truncated, _ = env.step(actions)

                if steps == 300:
                    truncated = True
                done = terminated or truncated

                # Store the transition in memory
                state_padding, next_state_padding, actions_padding, reward_padding = add_padding(state, next_state, actions,
                                                                                                reward,
                                                                                                options['uav'])
                if not options['clustered']:
                    replay_buffer_uniform.push(state_padding, actions_padding, next_state_padding, reward_padding,
                                            int(terminated))
                else:
                    replay_buffer_clustered.push(state_padding, actions_padding, next_state_padding, reward_padding,
                                                int(terminated))

                # Move to the next state
                state = next_state
                # Perform one step of the optimization
                optimize_model()
                steps += 1

                if done:
                    break

            if len(replay_buffer_uniform) >= 5000 and len(replay_buffer_clustered) >= 5000:
                validate()

        # save the nets
        torch.save(transformer_policy.state_dict(), '../neural_network/lastTransformer.pth')
        torch.save(mlp_policy.state_dict(), '../neural_network/lastMLP.pth')
        torch.save(deep_Q_net_policy.state_dict(), '../neural_network/lastDeepQ.pth')

        wandb.finish()
        env.close()
        print('TRAINING COMPLETE')

    else:

        def select_actions(state, uav_numebr):
            uav_info, connected_gu_positions = np.split(state, [uav_numebr * 2], axis=0)
            uav_info = uav_info.reshape(uav_numebr, 4)
            uav_info = torch.from_numpy(uav_info).float().to(device)
            connected_gu_positions = torch.from_numpy(connected_gu_positions).float().to(device)
            action = []
            with torch.no_grad():
                tokens = transformer_policy(connected_gu_positions.unsqueeze(0), uav_info.unsqueeze(0)).squeeze(0)
            for i in range(uav_numebr):
                with torch.no_grad():
                    # return action according to MLP [vx, vy]
                    output = mlp_policy(tokens[i])
                    output = output.cpu().numpy().reshape(2)
                    output = output * args.max_speed_uav
                    action.append(output)
            return action

    # For visible check
        env = gym.make('gym_cruising:Cruising-v0', args=args, render_mode='human')

        # ACTOR POLICY NET policy
        transformer_policy = TransformerEncoderDecoder(embed_dim=EMBEDDED_DIM).to(device)
        mlp_policy = MLPPolicyNet(token_dim=EMBEDDED_DIM).to(device)

        PATH_TRANSFORMER = './neural_network/bestTransformer.pth'
        transformer_policy.load_state_dict(torch.load(PATH_TRANSFORMER))
        PATH_MLP_POLICY = './neural_network/bestMLP.pth'
        mlp_policy.load_state_dict(torch.load(PATH_MLP_POLICY))

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
        uav_number = options["uav"]
        while True:
            actions = select_actions(state, uav_number)
            next_state, reward, terminated, truncated, info = env.step(actions)

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

    """

        # for numerical test
        env = gym.make('gym_cruising:Cruising-v0', render_mode='rgb_array', track_id=2)

        # ACTOR POLICY NET policy
        transformer_policy = TransformerEncoderDecoder(embed_dim=EMBEDDED_DIM).to(device)
        mlp_policy = MLPPolicyNet(token_dim=EMBEDDED_DIM).to(device)

        PATH_TRANSFORMER = './neural_network/last1Transformer.pth'
        transformer_policy.load_state_dict(torch.load(PATH_TRANSFORMER))
        PATH_MLP_POLICY = './neural_network/last1MLP.pth'
        mlp_policy.load_state_dict(torch.load(PATH_MLP_POLICY))

        options = ({
            "uav": 3,
            "gu": 120,
            "clustered": 1,
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
            uav_number = options["uav"]
            while True:
                actions = select_actions(state, uav_number)
                next_state, reward, terminated, truncated, info = env.step(actions)

                if steps == 300:
                    truncated = True
                done = truncated or info['Collision']

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

    ### CONSENTENDO USCITA DA AMBIENTE, reti BEST ###

    # 4 uniform 120 -> Mean reward:  0.11620978971348572, 85 collisione
    # 4 clustered 120 4 -> Mean reward:  0.5202186639820837 100000,  32 collisioni

    # 3 uniform 120 -> Mean reward:  0.8277718007374375, 1 collisione
    # 3 uniform 240 -> Mean reward:  0.8252742658334771, 1 collisione
    # 3 uniform 120 -> Mean reward:  0.8122288563359238 speed GU 27.7 m/s, collisioni:  2

    # 3 clustered 120 3 -> Mean reward:  0.7008802710879044 100000, 5 collisioni
    # 3 clustered 240 3 -> Mean reward:  0.6814977518089524 100000, 6 collisioni
    # 3 clustered 120 3 -> Mean reward:  0.689776540005067  100000 speed GU 27.7 m/s, 9 collisioni

    # 3 clustered 120 6 -> Mean reward:  0.7395896839831192 100000, 4 collisioni

    # 2 uniform 120 -> Mean reward:  0.5633710783311148, Collisioni:  0
    # 2 uniform 100 -> Mean reward:  0.5766865700476401, Collisioni:  0
    # 2 clustered 100 2 -> Mean reward:  0.601128310266376 100000, Collisioni: 2
    # 2 clustered 120 2 -> Mean reward:  0.5859859126286509 100000, Collisioni:  3
    # 2 clustered 120 4 -> Mean reward:  0.6558568341938088 100000, Collisioni:  1
    # 2 clustered 100 4 -> Mean reward:  0.6612490702720975 100000, Collisioni:  0

    """
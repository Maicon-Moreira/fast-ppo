import gymnasium as gym
from models import ActorCritic
import torch as t
import numpy as np
from worker import worker
from utils import normalize, plot_training_graph, save_info
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import matplotlib.pyplot as plt
from ppo_loss import ppo_loss
import time
from advantages import generalized_advantage_estimation, temporal_difference


def train(
    env_name,  # Name of the environment
    total_updates,  # Total number of updates to the models
    learning_rate,  # Learning rate for the optimizer
    cpu_threads,  # Number of CPU threads to use for experience collection
    envs_per_thread,  # Number of environments to run in parallel per thread
    steps_per_thread,  # Number of steps to run the environments in parallel per thread
    advantage_mode,  # Mode to use for calculating advantages, either "temporal_difference" or "generalized_advantage_estimation"
    gamma,  # Discount factor for rewards
    gae_lambda,  # Lambda value for generalized advantage estimation
    initial_epsilon,  # Initial value for epsilon
    final_epsilon,  # Final value for epsilon
    value_coefficient,  # Coefficient for the value loss in the PPO objective
    entropy_coefficient,  # Coefficient for the entropy loss in the PPO objective
    training_graph_path,  # Path to save the training graph
    target_reward_mean,  # Stop training when the mean reward is equal to or greater than this value
    experience_collection_model_device,  # Device to use for the actor model during experience collection
    advantage_calculation_model_device,  # Device to use for the critic model during advantage calculation
    training_model_device,  # Device to use for the actor and critic models during training
):
    possible_advantage_modes = [
        "temporal_difference",
        "generalized_advantage_estimation",
    ]
    assert (
        advantage_mode in possible_advantage_modes
    ), f"Advantage mode must be one of {possible_advantage_modes}"

    setup_initial_time = time.time()
    training_initial_time = None

    total_envs = cpu_threads * envs_per_thread
    experience_batch_size = total_envs * steps_per_thread
    total_steps = total_updates * experience_batch_size
    print(
        f"\nTraining PPO agent on {env_name} for {total_steps} total steps | Experience batch size: {experience_batch_size} | Total environments: {total_envs}\n"
    )

    core_envs = [
        [gym.make(env_name) for _ in range(envs_per_thread)] for _ in range(cpu_threads)
    ]  # [ [env1, env2, env3], [env4, env5, env6], ... ]

    observation_space = core_envs[0][0].observation_space
    action_space = core_envs[0][0].action_space
    print(f"Observation space: {observation_space}\nAction space: {action_space}\n")

    assert observation_space.shape is not None, "Observation space shape is None"
    state_dim = observation_space.shape[0]
    action_dim = action_space.n  # type: ignore
    actor_critic = ActorCritic(state_dim, action_dim)
    old_actor_critic = ActorCritic(state_dim, action_dim)
    best_actor_critic = ActorCritic(state_dim, action_dim)

    old_actor_critic.load_state_dict(actor_critic.state_dict())
    best_actor_critic.load_state_dict(actor_critic.state_dict())

    best_mean_episode_reward = -np.inf
    actor_critic.share_memory()  # share the actor-critic network across all cores
    old_actor_critic.share_memory()  # share the old actor-critic network across all cores
    best_actor_critic.share_memory()  # share the best actor-critic network across all cores

    optimizer = t.optim.AdamW(actor_critic.parameters(), lr=learning_rate)

    info = {
        "loss_hist": [],
        "total_episodes_hist": [],
        "mean_episode_reward_hist": [],
        "std_episode_reward_hist": [],
        "min_episode_reward_hist": [],
        "max_episode_reward_hist": [],
    }

    with ProcessPoolExecutor(max_workers=cpu_threads) as executor:
        for current_update in tqdm(range(total_updates), disable=True):
            epsilon = final_epsilon + (initial_epsilon - final_epsilon) * (
                1 - current_update / total_updates
            )

            # Collect experiences using multiple threads
            actor_critic.to(experience_collection_model_device)
            old_actor_critic.to(experience_collection_model_device)
            best_actor_critic.to(experience_collection_model_device)

            futures = []
            for core_i in range(cpu_threads):
                future = executor.submit(
                    worker,
                    core_envs[core_i],
                    steps_per_thread,
                    actor_critic,
                    envs_per_thread,
                    experience_collection_model_device,
                )
                futures.append(future)

            (
                all_states,
                all_actions,
                all_rewards,
                all_next_states,
                all_dones,
                all_episode_rewards,
            ) = (
                [],
                [],
                [],
                [],
                [],
                [],
            )
            for future in futures:
                (
                    states,
                    actions,
                    rewards,
                    next_states,
                    dones,
                    episode_rewards,
                ) = future.result()
                all_states.append(states)
                all_actions.append(actions)
                all_rewards.append(rewards)
                all_next_states.append(next_states)
                all_dones.append(dones)
                all_episode_rewards.extend(episode_rewards)

            if training_initial_time is None:
                training_initial_time = time.time()
                setup_time = time.time() - setup_initial_time

            # Convert and normalize the experiences
            all_states = np.concatenate(all_states, axis=1).reshape(
                (
                    total_envs,
                    steps_per_thread,
                    state_dim,
                )
            )
            all_actions = np.concatenate(all_actions, axis=1).reshape(
                (total_envs, steps_per_thread)
            )
            all_rewards = np.concatenate(all_rewards, axis=1).reshape(
                (total_envs, steps_per_thread)
            )
            all_rewards = normalize(all_rewards)
            all_next_states = np.concatenate(all_next_states, axis=1).reshape(
                (
                    total_envs,
                    steps_per_thread,
                    state_dim,
                )
            )
            all_dones = np.concatenate(all_dones, axis=1).reshape(
                (total_envs, steps_per_thread)
            )
            all_episode_rewards = np.array(all_episode_rewards, dtype=np.float32)

            # Calculate advantages using simple temporal difference method
            if advantage_mode == "temporal_difference":
                advantages = temporal_difference(
                    all_states,
                    all_rewards,
                    all_next_states,
                    all_dones,
                    actor_critic,
                    gamma,
                    experience_batch_size,
                    state_dim,
                    advantage_calculation_model_device,
                )

            # Calculate advantages using GAE method
            elif advantage_mode == "generalized_advantage_estimation":
                advantages = generalized_advantage_estimation(
                    all_states,
                    all_rewards,
                    all_next_states,
                    all_dones,
                    actor_critic,
                    gamma,
                    gae_lambda,
                    experience_batch_size,
                    state_dim,
                    steps_per_thread,
                    total_envs,
                    advantage_calculation_model_device,
                )

            # Logging and statistics
            log_string = []
            save_info(
                info,
                all_rewards,
                all_episode_rewards,
                current_update,
                total_updates,
                log_string,
            )

            mean_episode_reward = info["mean_episode_reward_hist"][-1]
            if mean_episode_reward > best_mean_episode_reward:
                best_mean_episode_reward = mean_episode_reward
                best_actor_critic.load_state_dict(actor_critic.state_dict())

            if mean_episode_reward >= target_reward_mean:
                print("\n".join(log_string))
                print(
                    f"Target reward mean of {target_reward_mean} reached! Stopping training..."
                )
                print()
                break

            # Train the actor and critic networks
            actor_critic.to(training_model_device)
            old_actor_critic.to(training_model_device)
            optimizer.zero_grad()
            loss = ppo_loss(
                states=t.from_numpy(all_states)
                .view((experience_batch_size, state_dim))
                .detach()
                .to(training_model_device),
                actions=t.from_numpy(all_actions)
                .view((experience_batch_size,))
                .detach()
                .to(training_model_device),
                advantages=t.from_numpy(advantages)
                .view((experience_batch_size,))
                .detach()
                .to(training_model_device),
                rewards=t.from_numpy(all_rewards)
                .view((experience_batch_size,))
                .detach()
                .to(training_model_device),
                actor_critic=actor_critic,
                old_actor_critic=old_actor_critic,
                epsilon=epsilon,
                value_coefficient=value_coefficient,
                entropy_coefficient=entropy_coefficient,
            )
            log_string.append(f"Loss: {loss.item()} | Epsilon: {epsilon:.2f}")
            info["loss_hist"].append(loss.item())
            loss.backward()
            old_actor_critic.load_state_dict(actor_critic.state_dict())
            optimizer.step()

            print("\n".join(log_string))
            print()

    assert training_initial_time is not None, "No initial time for training found"
    training_time = time.time() - training_initial_time

    plot_rendering_initial_time = time.time()
    plot_training_graph(
        info,
        training_graph_path,
    )
    plot_rendering_time = time.time() - plot_rendering_initial_time

    print(f"Setup time: {setup_time:.2f} seconds")
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Updates/second: {current_update / training_time:.2f}")
    print(f"Plot rendering time: {plot_rendering_time:.2f} seconds")
    print(f"Total time: {setup_time + training_time + plot_rendering_time:.2f} seconds")
    print(
        "Returning best actor and critic models with mean episode reward of",
        best_mean_episode_reward,
    )
    print()
    best_actor_critic.to("cpu")

    return best_actor_critic

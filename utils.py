import gymnasium as gym
import matplotlib.pyplot as plt
from celluloid import Camera
from tqdm import tqdm
import torch as t
import numpy as np


def normalize(x):
    return (x - x.mean()) / (x.std() + 1e-8)


def save_visualization(env_name, actor_critic, steps, visualization_path, fps):
    env = gym.make(env_name, render_mode="rgb_array")
    state = env.reset()[0]
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))  # env, actor prob, critic value
    camera = Camera(fig)
    actor_prob_hist = []
    critic_value_hist = []
    total_reward = 0

    for _ in tqdm(range(steps)):
        action_probs = actor_critic.actor(t.tensor(state).float())
        action = t.argmax(action_probs).item()
        next_state, reward, done, truncated, _ = env.step(action)
        value = actor_critic.critic(t.tensor(state).float()).item()
        state = next_state
        total_reward += reward  # type: ignore
        if done or truncated:
            state = env.reset()[0]
            total_reward = 0

        actor_prob_hist.append(action_probs.detach().numpy())
        critic_value_hist.append(value)

        ax[0].imshow(env.render())
        ax[0].axis("off")
        ax[0].text(0, 0, f"Total Reward: {total_reward}", color="red")

        ax[1].plot(actor_prob_hist, color="blue")
        ax[1].set_title("Actor Probabilities")

        ax[2].plot(critic_value_hist, color="red")
        ax[2].set_title("Critic Value")

        camera.snap()

    print("Saving visualization...")
    animation = camera.animate()
    animation.save(visualization_path, fps=fps)
    env.close()
    print(f"Visualization saved to {visualization_path}")


def plot_training_graph(
    info,
    training_graph_path,
):
    loss_hist = info["loss_hist"]
    total_episodes_hist = info["total_episodes_hist"]
    max_episode_reward_hist = info["max_episode_reward_hist"]
    mean_episode_reward_hist = info["mean_episode_reward_hist"]
    min_episode_reward_hist = info["min_episode_reward_hist"]
    std_episode_reward_hist = info["std_episode_reward_hist"]

    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    ax = ax.flatten()
    ax[0].plot(loss_hist)
    ax[0].set_title("PPO Loss")
    ax[0].set_xlabel("Update")
    ax[0].set_ylabel("Loss")

    ax[1].plot(total_episodes_hist)
    ax[1].set_title("Total Episodes")
    ax[1].set_xlabel("Update")
    ax[1].set_ylabel("Episodes")

    ax[2].plot(max_episode_reward_hist, color="green", label="Max")
    ax[2].plot(mean_episode_reward_hist, color="blue", label="Mean")
    ax[2].plot(min_episode_reward_hist, color="red", label="Min")
    ax[2].set_title("Episode Reward")
    ax[2].set_xlabel("Update")
    ax[2].set_ylabel("Reward")
    ax[2].legend()

    ax[3].plot(std_episode_reward_hist)
    ax[3].set_title("Std Episode Reward")
    ax[3].set_xlabel("Update")
    ax[3].set_ylabel("Reward")

    fig.savefig(training_graph_path)
    print(f"Training graph saved to {training_graph_path}")
    print()


def save_info(
    info,
    all_rewards,
    all_episode_rewards,
    current_update,
    total_updates,
    log_string,
):
    unique_rewards, reward_counts = np.unique(all_rewards, return_counts=True)
    top_10_rewards = unique_rewards[reward_counts.argsort()[-10:]].tolist()
    top_10_counts = reward_counts[reward_counts.argsort()[-10:]].tolist()
    log_string.append(
        f"Update {current_update + 1}/{total_updates} | Collected {all_rewards.size} experiences"
    )
    log_string.append(f"Top 10 rewards: {top_10_rewards} | Counts: {top_10_counts}")
    log_string.append(f"First 10 episode rewards: {all_episode_rewards[:10].tolist()}")
    mean_episode_reward = all_episode_rewards.mean().item()
    log_string.append(
        f"Total episodes: {len(all_episode_rewards)} | Reward mean: {mean_episode_reward:.2f} | Std: {all_episode_rewards.std().item():.2f}"
    )
    info["total_episodes_hist"].append(len(all_episode_rewards))
    info["mean_episode_reward_hist"].append(mean_episode_reward)
    info["std_episode_reward_hist"].append(all_episode_rewards.std().item())
    info["min_episode_reward_hist"].append(all_episode_rewards.min().item())
    info["max_episode_reward_hist"].append(all_episode_rewards.max().item())

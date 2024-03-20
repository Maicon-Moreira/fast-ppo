from train import train
from utils import save_visualization
import torch as t
import math
import os


def main():
    env_name = "CartPole-v1"
    env_steps = 300
    target_reward_mean = 200
    lr = 0.01
    total_updates = 150
    cpu_threads = 20
    envs_per_thread = 2

    # env_name = "Acrobot-v1"
    # env_steps = 300
    # target_reward_mean = -100
    # lr = 0.001
    # total_updates = 150
    # cpu_threads = 20

    # env_name = "LunarLander-v2"
    # env_steps = 1000
    # target_reward_mean = math.inf
    # lr = 0.005
    # total_updates = 500
    # cpu_threads = 10
    # envs_per_thread = 1

    # env_name = "MountainCar-v0"
    # env_steps = 1000
    # target_reward_mean = -110
    # lr = 0.01
    # total_updates = 100
    # cpu_threads = 20

    actor_critic = train(
        env_name=env_name,
        total_updates=total_updates,
        learning_rate=lr,
        cpu_threads=cpu_threads,
        envs_per_thread=envs_per_thread,
        steps_per_thread=env_steps,
        advantage_mode="generalized_advantage_estimation",
        # advantage_mode="temporal_difference",
        gamma=0.99,
        gae_lambda=0.95,
        initial_epsilon=0.2,
        final_epsilon=0.1,
        value_coefficient=0.5,
        entropy_coefficient=0.01,
        training_graph_path=f"{env_name}-training.png",
        target_reward_mean=target_reward_mean,
        experience_collection_model_device="cpu",
        advantage_calculation_model_device="cuda",
        training_model_device="cuda",
    )

    save_visualization(
        env_name=env_name,
        actor_critic=actor_critic,
        steps=env_steps,
        visualization_path=f"{env_name}-playing.gif",
        fps=24,
    )

    # # save models
    # os.makedirs("models", exist_ok=True)
    # t.save(actor, f"models/{env_name}-actor.pt")
    # t.save(critic, f"models/{env_name}-critic.pt")


if __name__ == "__main__":
    main()

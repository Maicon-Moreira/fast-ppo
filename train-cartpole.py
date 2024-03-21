from src.train import train
from src.utils import simulate_agent, plot_training_info
import torch as t


def main():
    env_name = "CartPole-v1"
    env_steps = 300

    actor_critic, training_info = train(
        env_name=env_name,
        total_updates=150,
        learning_rate=0.005,
        cpu_threads=20,
        envs_per_thread=2,
        steps_per_thread=env_steps,
        advantage_mode="generalized_advantage_estimation",
        gamma=0.99,
        gae_lambda=0.95,
        initial_epsilon=0.2,
        final_epsilon=0.1,
        value_coefficient=0.5,
        entropy_coefficient=0.01,
        target_reward_mean=200,
        experience_collection_model_device="cpu",
        advantage_calculation_model_device="cuda",
        training_model_device="cuda",
    )

    plot_training_info(
        training_info,
        training_graph_path=f"./images/{env_name}.png",
    )

    simulate_agent(
        env_name=env_name,
        actor_critic=actor_critic,
        steps=env_steps,
        visualization_path=f"./gifs/{env_name}.gif",
        fps=24,
    )

    t.save(actor_critic, f"./models/{env_name}.pt")


if __name__ == "__main__":
    main()

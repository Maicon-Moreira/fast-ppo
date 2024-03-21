from src.train import train
import matplotlib.pyplot as plt
from src.utils import plot_multiple_trainings_info


def main():
    env_name = "CartPole-v1"
    env_steps = 300
    total_trainings = 50
    all_training_info = []

    for _ in range(total_trainings):
        _, training_info = train(
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
        all_training_info.append(training_info)

    plot_multiple_trainings_info(
        multiple_training_info=all_training_info,
        training_graph_path=f"./images/{env_name}-benchmark.png",
    )


if __name__ == "__main__":
    main()

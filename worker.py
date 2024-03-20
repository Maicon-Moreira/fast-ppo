import numpy as np
import torch as t


def worker(
    envs,
    steps_per_thread,
    actor_critic,
    envs_per_thread,
    experience_collection_model_device,
):
    states_batch = np.zeros(
        (steps_per_thread, envs_per_thread, envs[0].observation_space.shape[0]),
        dtype=np.float32,
    )
    actions_batch = np.zeros((steps_per_thread, envs_per_thread), dtype=np.int64)
    rewards_batch = np.zeros((steps_per_thread, envs_per_thread), dtype=np.float32)
    next_states_batch = np.zeros(
        (steps_per_thread, envs_per_thread, envs[0].observation_space.shape[0]),
        dtype=np.float32,
    )
    dones_batch = np.zeros((steps_per_thread, envs_per_thread), dtype=np.int64)

    accumulated_rewards = [0] * envs_per_thread
    episode_rewards = []

    for env_i, env in enumerate(envs):
        states_batch[0, env_i] = env.reset()[0]

    for step_i in range(steps_per_thread):
        states_tensor = t.tensor(states_batch[step_i], dtype=t.float32).to(
            experience_collection_model_device
        )
        actions = (
            t.multinomial(actor_critic.actor(states_tensor), 1)
            .view((envs_per_thread,))
            .to("cpu")
        )
        actions_batch[step_i] = actions

        for env_i, env in enumerate(envs):
            next_state, reward, done, truncated, _ = env.step(actions[env_i].item())
            if done:
                reward = 0

            next_states_batch[step_i, env_i] = next_state
            rewards_batch[step_i, env_i] = reward
            dones_batch[step_i, env_i] = done
            accumulated_rewards[env_i] += reward

            if done or truncated:
                next_states_batch[step_i, env_i] = env.reset()[0]
                episode_rewards.append(accumulated_rewards[env_i])
                accumulated_rewards[env_i] = 0

        if step_i < steps_per_thread - 1:
            states_batch[step_i + 1] = next_states_batch[step_i]

    # save unfinished episodes accumulated rewards
    episode_rewards.extend(accumulated_rewards)

    return (
        states_batch,
        actions_batch,
        rewards_batch,
        next_states_batch,
        dones_batch,
        episode_rewards,
    )

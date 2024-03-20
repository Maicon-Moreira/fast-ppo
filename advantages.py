from utils import normalize
import torch as t
import numba as nb
import numpy as np


def temporal_difference(
    states,
    rewards,
    next_states,
    dones,
    actor_critic,
    gamma,
    experience_batch_size,
    state_dim,
    advantage_calculation_model_device,
):
    states = states.reshape((experience_batch_size, state_dim))
    rewards = rewards.reshape((experience_batch_size,))
    next_states = next_states.reshape((experience_batch_size, state_dim))
    dones = dones.reshape((experience_batch_size,))

    device = t.device(advantage_calculation_model_device)
    actor_critic.to(device)
    values = (
        actor_critic.critic(t.from_numpy(states).to(device))
        .cpu()
        .detach()
        .numpy()
        .reshape((experience_batch_size,))
    )
    next_values = (
        actor_critic.critic(t.from_numpy(next_states).to(device))
        .cpu()
        .detach()
        .numpy()
        .reshape((experience_batch_size,))
    )

    next_values[dones] = 0
    advantages = rewards + gamma * next_values - values
    advantages = normalize(advantages)
    return advantages


def generalized_advantage_estimation(
    states,
    rewards,
    next_states,
    dones,
    actor_critic,
    gamma,
    gae_lambda,
    experience_batch_size,
    state_dim,
    steps_per_thread,
    total_envs,
    advantage_calculation_model_device,
):
    device = t.device(advantage_calculation_model_device)
    actor_critic.to(device)
    values = (
        actor_critic.critic(
            t.from_numpy(states).view((experience_batch_size, state_dim)).to(device)
        )
        .cpu()
        .detach()
        .numpy()
        .reshape((total_envs, steps_per_thread))
    )
    next_values = (
        actor_critic.critic(
            t.from_numpy(next_states)
            .view((experience_batch_size, state_dim))
            .to(device)
        )
        .cpu()
        .detach()
        .numpy()
        .reshape((total_envs, steps_per_thread))
    )

    deltas = rewards + gamma * next_values - values
    advantages = fast_advantages_loop(
        deltas,
        dones,
        gamma,
        gae_lambda,
        total_envs,
        steps_per_thread,
    ).reshape((experience_batch_size,))
    advantages = normalize(advantages)
    return advantages


@nb.jit(nopython=True, fastmath=True)
def fast_advantages_loop(
    deltas,
    dones,
    gamma,
    gae_lambda,
    total_envs,
    steps_per_thread,
):
    advantages = np.zeros(
        (total_envs, steps_per_thread),
        dtype=np.float32,
    )
    advantages[-1] = deltas[-1]
    for i in range(steps_per_thread - 2, -1, -1):
        advantages[:, i] = (
            deltas[:, i] + gamma * gae_lambda * (1 - dones[:, i]) * advantages[:, i + 1]
        )
    return advantages

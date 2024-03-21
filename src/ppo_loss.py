import torch as t


def ppo_loss(
    states,
    actions,
    advantages,
    rewards,
    actor_critic,
    old_actor_critic,
    epsilon,
    value_coefficient,
    entropy_coefficient,
):
    # compute clipped surrogate objective
    new_policy = actor_critic.actor(states)
    old_policy = old_actor_critic.actor(states)
    action_probabilities = new_policy.gather(1, actions.unsqueeze(1)).squeeze()
    old_action_probabilities = old_policy.gather(1, actions.unsqueeze(1)).squeeze()
    ratio = action_probabilities / old_action_probabilities
    clipped_ratio = t.clamp(ratio, 1 - epsilon, 1 + epsilon)
    surrogate_objective = t.min(ratio * advantages, clipped_ratio * advantages).mean()

    # compute value loss
    values = actor_critic.critic(states).squeeze()
    value_loss = t.nn.functional.mse_loss(values, rewards)

    # compute entropy loss
    dists = t.distributions.Categorical(probs=new_policy)
    entropy_loss = dists.entropy().mean()

    # compute total loss
    loss = (
        surrogate_objective
        - value_coefficient * value_loss
        + entropy_coefficient * entropy_loss
    )

    # maximize the objective
    return -loss

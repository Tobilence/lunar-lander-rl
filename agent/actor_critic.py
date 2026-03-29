import flax.nnx as nnx
import jax
import jax.numpy as jnp

class ActorCriticNetwork(nnx.Module):
    def __init__(self, rngs: nnx.Rngs, d_in=8, d_hidden_1=128, d_hidden_2=128, head_dim=128, d_actions=4):
        self.lin_in = nnx.Linear(d_in, d_hidden_1, rngs=rngs)
        self.hidden_1 = nnx.Linear(d_hidden_1, d_hidden_2, rngs=rngs)
        self.hidden_2 = nnx.Linear(d_hidden_2, d_hidden_2, rngs=rngs)

        self.hidden_critic = nnx.Linear(d_hidden_2, head_dim, rngs=rngs)
        self.hidden_actor = nnx.Linear(d_hidden_2, head_dim, rngs=rngs)
        self.head_critic = nnx.Linear(head_dim, 1, rngs=rngs)
        self.head_actor = nnx.Linear(head_dim, d_actions, rngs=rngs)
        
    
    def __call__(self, x):
        x = nnx.relu(self.lin_in(x))
        x = nnx.relu(self.hidden_1(x))
        x = nnx.relu(self.hidden_2(x))

        critic = nnx.relu(self.hidden_critic(x))
        critic = self.head_critic(critic)

        actor = nnx.relu(self.hidden_actor(x))
        actor = self.head_actor(actor)

        return actor, critic


def _loss_terms(model: ActorCriticNetwork, state, action, reward, next_state, terminal, gamma=0.99, entropy_coef=0.01):
    actor_logits, critic = model(state)
    _, critic_next = model(next_state)

    critic = jnp.squeeze(critic, axis=-1)
    critic_next = jnp.squeeze(critic_next, axis=-1)

    target = reward + gamma * jax.lax.stop_gradient(critic_next) * (1 - terminal)
    td_error = target - critic

    log_probs = jax.nn.log_softmax(actor_logits)
    probs = jax.nn.softmax(actor_logits)
    action_one_hot = jax.nn.one_hot(action, num_classes=actor_logits.shape[-1])
    log_prob_action = jnp.sum(action_one_hot * log_probs)

    actor_loss = -log_prob_action * jax.lax.stop_gradient(td_error)
    entropy = -jnp.sum(probs * log_probs)
    critic_loss = jnp.square(td_error)
    total_loss = actor_loss + 0.5 * critic_loss - entropy_coef * entropy

    return {
        "actor_logits": actor_logits,
        "probs": probs,
        "value": critic,
        "next_value": critic_next,
        "target": target,
        "td_error": td_error,
        "log_prob_action": log_prob_action,
        "actor_loss": actor_loss,
        "entropy": entropy,
        "critic_loss": critic_loss,
        "total_loss": total_loss,
    }


def loss_fn(model: ActorCriticNetwork, state, action, reward, next_state, terminal, gamma=0.99, entropy_coef=0.01):
    return _loss_terms(model, state, action, reward, next_state, terminal, gamma, entropy_coef)["total_loss"]

_batched_loss_fn = nnx.vmap(loss_fn, in_axes=(None, 0, 0, 0, 0, 0, None, None))
_batched_loss_terms = nnx.vmap(_loss_terms, in_axes=(None, 0, 0, 0, 0, 0, None, None))

# necessary to make the vector into a scaler, which is required for value_and_grad in the optimization step
def batched_loss_function(model, states, actions, rewards, next_states, terminals, gamma, entropy_coef):
    return _batched_loss_fn(model, states, actions, rewards, next_states, terminals, gamma, entropy_coef).mean()


def batch_debug_metrics(model, states, actions, rewards, next_states, terminals, gamma, entropy_coef):
    terms = _batched_loss_terms(model, states, actions, rewards, next_states, terminals, gamma, entropy_coef)
    action_histogram = jax.nn.one_hot(actions, num_classes=terms["probs"].shape[-1]).mean(axis=0)
    mean_probs = terms["probs"].mean(axis=0)

    return {
        "loss_total": terms["total_loss"].mean(),
        "loss_actor": terms["actor_loss"].mean(),
        "loss_critic": terms["critic_loss"].mean(),
        "entropy": terms["entropy"].mean(),
        "log_prob_action_mean": terms["log_prob_action"].mean(),
        "value_mean": terms["value"].mean(),
        "value_std": terms["value"].std(),
        "next_value_mean": terms["next_value"].mean(),
        "next_value_std": terms["next_value"].std(),
        "target_mean": terms["target"].mean(),
        "target_std": terms["target"].std(),
        "td_error_mean": terms["td_error"].mean(),
        "td_error_std": terms["td_error"].std(),
        "td_error_abs_mean": jnp.abs(terms["td_error"]).mean(),
        "td_error_abs_max": jnp.abs(terms["td_error"]).max(),
        "reward_mean": rewards.mean(),
        "reward_std": rewards.std(),
        "reward_sum": rewards.sum(),
        "reward_min": rewards.min(),
        "reward_max": rewards.max(),
        "terminal_fraction": terminals.mean(),
        "logits_mean": terms["actor_logits"].mean(),
        "logits_std": terms["actor_logits"].std(),
        "logits_min": terms["actor_logits"].min(),
        "logits_max": terms["actor_logits"].max(),
        "action_frac_0": action_histogram[0],
        "action_frac_1": action_histogram[1],
        "action_frac_2": action_histogram[2],
        "action_frac_3": action_histogram[3],
        "prob_mean_0": mean_probs[0],
        "prob_mean_1": mean_probs[1],
        "prob_mean_2": mean_probs[2],
        "prob_mean_3": mean_probs[3],
    }
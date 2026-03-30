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


def loss_fn(model: ActorCriticNetwork, state, action, reward, next_state, terminal, gamma=0.99, entropy_coef=0.01):
    actor_logits, critic = model(state)
    _, critic_next = model(next_state)

    critic = jnp.squeeze(critic, axis=-1)
    critic_next = jnp.squeeze(critic_next, axis=-1)

    target = reward + gamma * jax.lax.stop_gradient(critic_next) * (1 - terminal)
    td_error = target - critic

    log_probs = jax.nn.log_softmax(actor_logits)
    action_one_hot = jax.nn.one_hot(action, num_classes=actor_logits.shape[-1])
    log_prob_action = jnp.sum(action_one_hot * log_probs)

    actor_loss = -log_prob_action * jax.lax.stop_gradient(td_error)

    probs = jax.nn.softmax(actor_logits)
    entropy = -jnp.sum(probs * log_probs)
    entropy_loss = entropy * entropy_coef

    critic_loss = jnp.square(td_error)

    total_loss = actor_loss + 0.5 * critic_loss - entropy_loss
    return total_loss, {"entropy": entropy, "critic_loss": critic_loss}

_batched_loss_fn = nnx.vmap(loss_fn, in_axes=(None, 0, 0, 0, 0, 0, None, None))

# necessary to make the vector into a scaler, which is required for value_and_grad in the optimization step
def batched_loss_function(model, states, actions, rewards, next_states, terminals, gamma, entropy_coef):
    losses, aux = _batched_loss_fn(model, states, actions, rewards, next_states, terminals, gamma, entropy_coef)
    return losses.mean(), {k: v.mean() for k, v in aux.items()}
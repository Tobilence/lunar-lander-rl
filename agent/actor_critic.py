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
    
    target = reward + gamma * jax.lax.stop_gradient(critic_next) * (1 - terminal)
    td_error = target - critic

    # Actor Loss: -log_prob[action] * advantage
    log_probs = jax.nn.log_softmax(actor_logits)
    log_prob_action = log_probs[action]
    actor_loss = -log_prob_action * td_error

    # regularization with entropy
    probs = jax.nn.softmax(actor_logits)
    entropy_loss = -jnp.sum(probs * log_probs) * entropy_coef

    critic_loss = jnp.square(td_error)

    return actor_loss + 0.5 * critic_loss - entropy_loss

_batched_loss_fn = nnx.vmap(loss_fn, in_axes=(None, 0, 0, 0, 0, 0, None, None))

# necessary to make the vector into a scaler, which is required for value_and_grad in the optimization step
def batched_loss_function(model, states, actions, rewards, next_states, terminals, gamma, entropy_coef):
    return _batched_loss_fn(model, states, actions, rewards, next_states, terminals, gamma, entropy_coef).mean()
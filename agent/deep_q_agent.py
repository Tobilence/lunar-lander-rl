import jax.numpy as jnp
from flax import nnx
import jax
import math

class QNetwork(nnx.Module):
    def __init__(self, rngs: nnx.Rngs, d_in=8, d_hidden_1=128, d_hidden_2=128, d_out=4):
        self.lin_in = nnx.Linear(d_in, d_hidden_1, rngs=rngs)
        self.hidden_1 = nnx.Linear(d_hidden_1, d_hidden_2, rngs=rngs)
        self.hidden_2 = nnx.Linear(d_hidden_2, d_hidden_2, rngs=rngs)
        self.lin_out = nnx.Linear(d_hidden_2, d_out, rngs=rngs)


    def __call__(self, x):
        x = nnx.relu(self.lin_in(x))
        x = nnx.relu(self.hidden_1(x))
        x = nnx.relu(self.hidden_2(x))
        return self.lin_out(x)

class NoisyDuelingQNetwork(nnx.Module):
    def __init__(self, rngs: nnx.Rngs, d_in=8, d_hidden_1=128, d_hidden_2=128, d_stream=128, d_out=4):
        self.lin_in = NoisyLayer(d_in, d_hidden_1, rngs=rngs)
        self.hidden_1 = NoisyLayer(d_hidden_1, d_hidden_2, rngs=rngs)
        self.hidden_2 = NoisyLayer(d_hidden_2, d_hidden_2, rngs=rngs)
        
        self.hidden_stream_v = NoisyLayer(d_hidden_2, d_stream, rngs=rngs)
        self.stream_v = NoisyLayer(d_stream, 1, rngs=rngs)

        self.hidden_stream_a = NoisyLayer(d_hidden_2, d_stream, rngs=rngs)
        self.stream_a = NoisyLayer(d_stream, d_out, rngs=rngs)


    def __call__(self, x):
        x = nnx.relu(self.lin_in(x))
        x = nnx.relu(self.hidden_1(x))
        x = nnx.relu(self.hidden_2(x))

        v = self.stream_v(nnx.relu(self.hidden_stream_v(x)))
        a = self.stream_a(nnx.relu(self.hidden_stream_a(x)))

        advantage_mean = jnp.mean(a, axis=-1, keepdims=True)
        return v + (a - advantage_mean)


class NoisyLayer(nnx.Module):
    def __init__(self, d_in: int, d_out: int, *, rngs: nnx.Rngs) -> None:
        self.rngs = rngs
        self.d_in = d_in
        self.d_out = d_out
        
        mu_range = 1.0 / math.sqrt(d_in)
        sigma_init = 0.5 / math.sqrt(d_in)
        
        # special initialization to avoid instability
        self.w_mu = nnx.Param(
            jax.random.uniform(rngs.params(), (d_in, d_out), minval=-mu_range, maxval=mu_range)
        )
        self.b_mu = nnx.Param(
            jax.random.uniform(rngs.params(), (d_out,), minval=-mu_range, maxval=mu_range)
        )

        # Sigma is initialized as a constant, but remains a learnable parameter
        self.w_sigma = nnx.Param(jnp.full((d_in, d_out), sigma_init))
        self.b_sigma = nnx.Param(jnp.full((d_out,), sigma_init))
    
    def __call__(self, x: jax.Array) -> jax.Array:
        key_w = self.rngs.noise()
        key_b = self.rngs.noise()
        
        w_epsilon = jax.random.normal(key_w, (self.d_in, self.d_out))
        b_epsilon = jax.random.normal(key_b, (self.d_out,))

        noisy_weight = self.w_mu + (self.w_sigma * w_epsilon)
        noisy_bias = self.b_mu + (self.b_sigma * b_epsilon)

        return (x @ noisy_weight) + noisy_bias
# epsilon greedy policy
def act_epsilon_greedy(key, q_values, epsilon=0.05):
    prob_key, action_key = jax.random.split(key)
    random_val = jax.random.uniform(prob_key)
    random_action = jax.random.randint(action_key, shape=(), minval=0, maxval=4)
    greedy_action = jnp.argmax(q_values)
    return jnp.where(random_val < epsilon, random_action, greedy_action)

fun_batch_act = nnx.vmap(act_epsilon_greedy, in_axes=(0, 0, None)) # key and state need to have 0


# calcualte targets
def calcualte_targets(eval_model, next_state, reward, is_terminal_state, gamma=0.9):
    return jnp.where(is_terminal_state,
                     reward,
                     reward + gamma * jnp.max(eval_model(next_state))
                    )
fun_batch_calculate_target = nnx.vmap(calcualte_targets, in_axes=(None, 0,0,0, None))

# training
def mse_loss(model, states, actions, targets):
    qs = model(states)
    # Pick the Q-value for the action taken in each state
    # This selects qs[0, actions[0]], qs[1, actions[1]], etc.
    chosen_qs = qs[jnp.arange(len(actions)), actions]
    tdes = (chosen_qs - targets)
    loss = jnp.mean((chosen_qs - targets) ** 2)
    return loss, (qs, tdes)

@nnx.jit
def train_step(model, optimizer, states, chosen_actions, targets):
    fun_inference_grad = nnx.value_and_grad(mse_loss, has_aux=True)
    (loss, (qs, tdes)), grad = fun_inference_grad(model, states, chosen_actions, targets)
    grad_norm = jnp.sqrt(sum(jnp.sum(leaf ** 2) for leaf in jax.tree_util.tree_leaves(grad)))
    optimizer.update(model, grad)
    return loss, qs, tdes, grad_norm


# calculate targets double DQN
def calcualte_targets_ddqn(acting_model, eval_model, next_state, reward, is_terminal_state, gamma=0.9):
    return jnp.where(is_terminal_state,
                     reward,
                     reward + gamma * eval_model(next_state)[jnp.argmax(acting_model(next_state))]
                    )
fun_batch_calculate_target_ddqn = nnx.vmap(calcualte_targets_ddqn, in_axes=(None, None, 0,0,0, None))
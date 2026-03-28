import jax.numpy as jnp
from flax import nnx
import jax

class QNetwork(nnx.Module):
    def __init__(self, rngs: nnx.Rngs, d_in=8, d_hidden_1=128, d_hidden_2=128, d_out=4):
        self.lin_in = nnx.Linear(d_in, d_hidden_1, rngs=rngs)
        self.hidden_1 = nnx.Linear(d_hidden_1, d_hidden_2, rngs=rngs)
        self.hidden_2 = nnx.Linear(d_hidden_2, d_hidden_1, rngs=rngs)
        self.lin_out = nnx.Linear(d_hidden_1, d_out, rngs=rngs)


    def __call__(self, x):
        x = nnx.relu(self.lin_in(x))
        x = nnx.relu(self.hidden_1(x))
        x = nnx.relu(self.hidden_2(x))
        return self.lin_out(x)

class DuelingQNetwork(nnx.Module):
    def __init__(self) -> None:
        pass
    
    def __call__(self, x):
        # todo: adapt architecture according to slides
        pass


class NoisyLayer(nnx.Module):
    def __init__(self) -> None:
        pass
    
    def __call__(self, x):
        # todo: implement a noisy linear layer according to formula in slides
        pass


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
                     reward + gamma * jnp.max(eval_model(next_state))  # TODO: could add the double DQN here; would need to pass acting network as well
                    )
fun_batch_calculate_target = nnx.vmap(calcualte_targets, in_axes=(None, 0,0,0, None))

# training
def mse_loss(model, states, actions, targets):
    qs = model(states)
    # Pick the Q-value for the action taken in each state
    # This selects qs[0, actions[0]], qs[1, actions[1]], etc.
    chosen_qs = qs[jnp.arange(len(actions)), actions]
    loss = jnp.mean((chosen_qs - targets) ** 2)
    return loss, qs

@nnx.jit
def train_step(model, optimizer, states, chosen_actions, targets):
    fun_inference_grad = nnx.value_and_grad(mse_loss, has_aux=True)
    (loss, qs), grad = fun_inference_grad(model, states, chosen_actions, targets)
    optimizer.update(model, grad)
    return loss, qs

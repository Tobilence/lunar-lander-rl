from typing import Any
import gymnasium as gym
import optax
import jax.numpy as jnp
import flax.nnx as nnx
import jax
from tqdm import tqdm


hyperparameters = {
    "learning_rate": 4e-5,
    "optimizer_lr": 4e-5,
    "gamma": 0.99
}

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


def loss_fn(model: ActorCriticNetwork, state, action, reward, next_state, terminal):
    actor_logits, critic = model(state)
    _, critic_next = model(next_state)
    
    target = reward + hyperparameters["gamma"] * jax.lax.stop_gradient(critic_next) * (1 - terminal)
    td_error = target - critic

    # Actor Loss: -log_prob[action] * advantage
    log_probs = jax.nn.log_softmax(actor_logits)
    log_prob_action = log_probs[action]
    actor_loss = -log_prob_action * jax.lax.stop_gradient(td_error)

    critic_loss = jnp.square(td_error)

    return (actor_loss + 0.5 * critic_loss).mean()

@nnx.jit
def train_step(model: ActorCriticNetwork, optimizer: nnx.Optimizer, state, action, reward, next_state, terminal):
    # We differentiate with respect to the model's parameters
    grad_fn = nnx.value_and_grad(loss_fn)
    loss, grads = grad_fn(model, state, action, reward, next_state, terminal)
    
    # Update the optimizer and model parameters
    optimizer.update(model, grads)
    return loss


## Lunar Lander Constants
LUNAR_LANDER_OBSERVATION_SPACE_DIM = 8
LUNAR_LANDER_ACTION_SPACE_SIZE = 4
TOTAL_TRAINING_STEPS = 50_000


## Main Loop
def train(
        render=True
    ):
    mode = "human" if render else None
    env = gym.make("LunarLander-v3", render_mode=mode)

    # optimizer = nnx.Optimizer(acting_network, optax.adam(hyperparameters["optimizer_lr"]), wrt=nnx.Param)
    jax_key = jax.random.key(0)
    network = ActorCriticNetwork(rngs=nnx.Rngs(0))
    optimizer = nnx.Optimizer(network, optax.adam(hyperparameters["optimizer_lr"]), wrt=nnx.Param)

    current_step = 0
    episode = 0

    pbar = tqdm(total=TOTAL_TRAINING_STEPS, desc="Training Lunar Lander")

    while current_step < TOTAL_TRAINING_STEPS:
        # Reset returns (observation, info)
        state, info = env.reset()
        terminated = False
        truncated = False
        total_reward = 0
        episode += 1

        while not (terminated or truncated):
            # setup variables
            jax_key, subkey = jax.random.split(jax_key)
            current_step +=1
            pbar.update(1)

            actor_logits, critic = network(state)
            action = int(jax.random.categorical(subkey, actor_logits))
            
            next_state, reward, terminated, truncated, _ = env.step(action)

            _, critic_next_state = jax.lax.stop_gradient(network(next_state))

            loss = train_step(network, optimizer, state, action, reward, next_state, terminated or truncated)

            state = next_state
            total_reward += float(reward)
        
    pbar.close()
    env.close()

if __name__ == "__main__":
    train(render=True)
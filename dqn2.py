import gymnasium as gym
import optax
import jax
import jax.numpy as jnp
from flax import nnx
from agent import QNetwork
from helpers import RingBuffer
from dataclasses import dataclass
from typing import Any, SupportsFloat, List
import orbax.checkpoint as ocp
from tensorboardX import SummaryWriter
import time
from pathlib import Path
from tqdm import tqdm

## tensorboard setup


## Lunar Lander Constants
LUNAR_LANDER_OBSERVATION_SPACE_DIM = 8
LUNAR_LANDER_ACTION_SPACE_SIZE = 4

## Hyperparameters
BUFFER_SIZE = 1000
EPSILON_START = 0.9
EPSILON_END = 0.05
EPSILON_DECAY_STEPS = 40_000 # linear decay by 1e-3

LEARNING_START_STEP = 100
GAMMA=0.99
MINI_BATCH_SIZE = 64
DISCOUNT_FACTOR = 0.99
STEP_SIZE = 1e-3
NEURAL_NETWORK_UPDATE_STEP = 1000
TOTAL_TRAINING_STEPS = 50_000

# optimizer
OPTIMIZER_LR = 1e-3


## Helpers

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
    loss = jnp.mean((chosen_qs - targets) ** 2)
    return loss, qs

@nnx.jit
def train_step(model, optimizer, states, chosen_actions, targets):
    fun_inference_grad = nnx.value_and_grad(mse_loss, has_aux=True)
    (loss, qs), grad = fun_inference_grad(model, states, chosen_actions, targets)
    optimizer.update(model, grad)
    return loss, qs


## Main Loop
def run_landing(
        checkpoint_manager,
        tensorboard_writer,
        render=True
    ):
    mode = "human" if render else None
    env = gym.make("LunarLander-v3", render_mode=mode)

    buffer = RingBuffer(max_size=BUFFER_SIZE)
    eval_network = QNetwork(nnx.Rngs(0))
    acting_network = QNetwork(nnx.Rngs(0))
    optimizer = nnx.Optimizer(acting_network, optax.adam(OPTIMIZER_LR), wrt=nnx.Param)
    jax_key = jax.random.key(0)

    epsilon = 0.5  # this should initially be high and then decay --> todo
    total_steps = 0
    episode = 0

    pbar = tqdm(total=TOTAL_TRAINING_STEPS, desc="Training Lunar Lander")

    while total_steps < TOTAL_TRAINING_STEPS:
        # Reset returns (observation, info)
        state, info = env.reset()
        terminated = False
        truncated = False
        total_reward = 0
        episode += 1

        while not (terminated or truncated):
            # setup variables
            total_steps +=1
            pbar.update(1)
            epsilon = max(
                EPSILON_END,
                EPSILON_START - (EPSILON_START - EPSILON_END) * (total_steps / EPSILON_DECAY_STEPS)
            )
            
            q_values = acting_network(state)
            jax_key, subkey = jax.random.split(jax_key)
            action = act_epsilon_greedy(subkey, q_values, epsilon=epsilon)
            action = int(action)
            
            next_state, reward, terminated, truncated, _ = env.step(action)

            buffer.add(
                [
                    state,
                    action,
                    reward,
                    next_state,
                    (truncated or terminated)
                ]
            )

            if len(buffer) >= LEARNING_START_STEP:
                m_batch = buffer.sample(MINI_BATCH_SIZE)
                states, actions, rewards, next_states, dones = zip(*m_batch)
                jax_batch = {
                    "states": jnp.array(states),
                    "action": jnp.array(actions),
                    "reward": jnp.array(rewards),
                    "next_state": jnp.array(next_states),
                    "dones": jnp.array(dones, dtype=jnp.bool_)
                }

                batch_targets = fun_batch_calculate_target(
                    eval_network,
                    jax_batch["next_state"],
                    jax_batch["reward"],
                    jax_batch["dones"],
                    GAMMA
                    )
                
                train_step(
                    acting_network,
                    optimizer,
                    jax_batch["states"],
                    jax_batch["action"],
                    batch_targets
                )

            
            if total_steps % NEURAL_NETWORK_UPDATE_STEP == 0:
                # Extract the state from the active network and load it into the eval network
                _, active_state = nnx.split(acting_network)
                nnx.update(eval_network, active_state)

            if total_steps % 10_000 == 0:
                network_state = nnx.state(acting_network)
                checkpoint_manager.save(total_steps, network_state)


            state = next_state
            total_reward += reward # pyright: ignore[reportOperatorIssue]
        
        tensorboard_writer.add_scalar("Metrics/Episode_Reward", float(total_reward), episode)

    pbar.close()
    env.close()


if __name__ == "__main__":
    checkpointer_path = Path("checkpoints_v0").resolve()
    writer = SummaryWriter(f"runs/flax_dqn_{int(time.time())}")

    run_name = time.strftime("%Y%m%d_%H%M%S")
    run_path = Path("runs") / run_name
    
    log_path = run_path / "logs"
    ckpt_path = run_path / "checkpoints"
    
    log_path.mkdir(parents=True, exist_ok=True)
    ckpt_path.mkdir(parents=True, exist_ok=True)
    
    writer = SummaryWriter(str(log_path))
    
    checkpoint_manager = ocp.CheckpointManager(
        ckpt_path.resolve(), 
        ocp.StandardCheckpointer(), 
    )

    run_landing(checkpoint_manager, writer, render=False)
    writer.close()
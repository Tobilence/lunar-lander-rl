import gymnasium as gym
import yaml
import optax
import jax.numpy as jnp
import flax.nnx as nnx
import jax
import numpy as np
from tqdm import tqdm
import orbax.checkpoint as ocp
from tensorboardX import SummaryWriter
from pathlib import Path
import time
from helpers import save_model_class_name
from agent.actor_critic import ActorCriticNetwork, batched_loss_function


hyperparameters = {
    "optimizer_lr": 1e-4,
    "gamma": 0.99,
    "entropy_coef": 0.15,
    "num_envs": 64,
}

@nnx.jit
def train_step(model: ActorCriticNetwork, optimizer: nnx.Optimizer, state, action, reward, next_state, terminal, hyperparameters):
    grad_fn = nnx.value_and_grad(batched_loss_function)
    loss, grads = grad_fn(model, state, action, reward, next_state, terminal, hyperparameters["gamma"], hyperparameters["entropy_coef"])
    optimizer.update(model, grads)
    return loss

## Lunar Lander Constants
LUNAR_LANDER_OBSERVATION_SPACE_DIM = 8
LUNAR_LANDER_ACTION_SPACE_SIZE = 4
TOTAL_TRAINING_STEPS = 500_000 * 10


def make_env(render_mode=None):
    def thunk():
        return gym.make("LunarLander-v3", render_mode=render_mode)

    return thunk


## Main Loop
def train(
        network,
        checkpoint_manager,
        tensorboard_writer,
        render=True
    ):
    mode = "human" if render and hyperparameters["num_envs"] == 1 else None
    env = gym.vector.AsyncVectorEnv(
        [make_env(mode) for _ in range(hyperparameters["num_envs"])]
    )

    jax_key = jax.random.key(0)
    optimizer = nnx.Optimizer(network, optax.adam(learning_rate=hyperparameters["optimizer_lr"]), wrt=nnx.Param)

    current_step = 0
    episode = 0
    next_checkpoint_step = 10_000
    smoothed_return = None
    smoothing_alpha = 0.05

    pbar = tqdm(total=TOTAL_TRAINING_STEPS, desc="Training Lunar Lander")
    state, _ = env.reset()
    episode_rewards = np.zeros(hyperparameters["num_envs"], dtype=np.float32)

    while current_step < TOTAL_TRAINING_STEPS:
        jax_key, subkey = jax.random.split(jax_key)

        actor_logits, _ = network(state)
        action = jax.random.categorical(subkey, actor_logits, axis=-1)
        action = np.asarray(action, dtype=np.int32)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = np.logical_or(terminated, truncated)

        states_arr = jnp.asarray(state)
        actions_arr = jnp.asarray(action)
        rewards_arr = jnp.asarray(reward, dtype=jnp.float32)
        next_states_arr = jnp.asarray(next_state)
        terminals_arr = jnp.asarray(done.astype(np.float32))

        episode_rewards += reward
        finished_env_indices = np.flatnonzero(done)
        for env_idx in finished_env_indices:
            episode += 1
            ep_return = float(episode_rewards[env_idx])
            smoothed_return = ep_return if smoothed_return is None else (1 - smoothing_alpha) * smoothed_return + smoothing_alpha * ep_return
            tensorboard_writer.add_scalar("Metrics/Episode_Reward", ep_return, episode)
            tensorboard_writer.add_scalar("Metrics/Smoothed_Episode_Reward", smoothed_return, episode)
            episode_rewards[env_idx] = 0.0

        state = next_state
        step_increment = hyperparameters["num_envs"]
        current_step += step_increment
        pbar.update(min(step_increment, TOTAL_TRAINING_STEPS - pbar.n))

        loss = train_step(network, optimizer, states_arr, actions_arr, rewards_arr, next_states_arr, terminals_arr, hyperparameters)
        tensorboard_writer.add_scalar("train/loss", float(loss), current_step)

        if current_step >= next_checkpoint_step:
            network_state = nnx.state(network)
            checkpoint_manager.save(current_step, network_state)
            next_checkpoint_step += 10_000

    pbar.close()
    env.close()


def setup_run_dir(run_name: str):
    run_name = time.strftime(f"{run_name}_%Y%m%d_%H%M%S")
    run_path = Path("runs") / run_name
    run_path.mkdir(parents=True, exist_ok=True)
    return run_path

def setup_run_loggers(run_path):
    log_path = run_path / "logs"
    ckpt_path = run_path / "checkpoints"
    
    log_path.mkdir(parents=True, exist_ok=True)
    ckpt_path.mkdir(parents=True, exist_ok=True)
    
    tensorboard_writer = SummaryWriter(str(log_path))
    
    checkpoint_manager = ocp.CheckpointManager(
        ckpt_path.resolve(), 
        ocp.StandardCheckpointer(), 
    )
    return tensorboard_writer, checkpoint_manager

def log_hyperparameters(run_dir, hyperparameters):
    with open(run_dir / "hyperparameters.yaml", "w+") as f:
        yaml.dump(hyperparameters, f, default_flow_style=False, sort_keys=False)

if __name__ == "__main__":
    run_dir = setup_run_dir("ac")
    log_hyperparameters(run_dir, hyperparameters)
    writer, checkpoint_manager = setup_run_loggers(run_dir)

    network = ActorCriticNetwork(rngs=nnx.Rngs(0))
    save_model_class_name(run_dir, network)

    train(
        network,
        checkpoint_manager,
        writer,
        render=False
    )
    writer.close()
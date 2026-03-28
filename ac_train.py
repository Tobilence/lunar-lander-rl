import gymnasium as gym
import yaml
import optax
import jax.numpy as jnp
import flax.nnx as nnx
import jax
from tqdm import tqdm
import orbax.checkpoint as ocp
from tensorboardX import SummaryWriter
from pathlib import Path
from tqdm import tqdm
import yaml
import time
from agent import ActorCriticNetwork, batched_loss_function


hyperparameters = {
    "optimizer_lr": 4e-5,
    "gamma": 0.99,
    "entropy_coef": 0.01,
    "epochs_per_episode": 10
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
TOTAL_TRAINING_STEPS = 500_000


## Main Loop
def train(
        checkpoint_manager,
        tensorboard_writer,
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

        states, actions, rewards, next_states, terminals = [], [], [], [], []

        while not (terminated or truncated):
            # setup variables
            jax_key, subkey = jax.random.split(jax_key)
            current_step +=1
            pbar.update(1)

            actor_logits, _ = network(state)
            action = int(jax.random.categorical(subkey, actor_logits))
            tensorboard_writer.add_scalar("sanity-check/chosen-action", action, current_step)
            
            next_state, reward, terminated, truncated, _ = env.step(action)

            # Store the transition
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            terminals.append(1.0 if terminated else 0.0)

            # loss = train_step(network, optimizer, state, action, reward, next_state, terminated or truncated)
            # tensorboard_writer.add_scalar("train/loss", loss, current_step)

            state = next_state
            total_reward += float(reward)

            if current_step % 10_000 == 0:
                network_state = nnx.state(network)
                checkpoint_manager.save(current_step, network_state)
            
        
        if len(states) > 0:
            # Convert lists to JAX arrays
            states_arr = jnp.array(states)
            actions_arr = jnp.array(actions)
            rewards_arr = jnp.array(rewards, dtype=jnp.float32)
            next_states_arr = jnp.array(next_states)
            terminals_arr = jnp.array(terminals, dtype=jnp.float32)

            # batched training step
            loss = train_step(network, optimizer, states_arr, actions_arr, rewards_arr, next_states_arr, terminals_arr, hyperparameters)
            
            tensorboard_writer.add_scalar("train/loss", float(loss), current_step)

        tensorboard_writer.add_scalar("Metrics/Episode_Reward", total_reward, episode)

        
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

    train(
        checkpoint_manager,
        writer,
        render=False
    )
    writer.close()
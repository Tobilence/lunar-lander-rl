import gymnasium as gym
import jax.numpy as jnp
import optax
import jax
from flax import nnx
from agent import NoisyDuelingQNetwork, fun_batch_calculate_target_ddqn, act_epsilon_greedy, train_step
from helpers import RingBuffer, calculate_epsilon_decay
import orbax.checkpoint as ocp
from tensorboardX import SummaryWriter
import time
from pathlib import Path
from tqdm import tqdm
import yaml
from collections import deque

## Lunar Lander Constants
LUNAR_LANDER_OBSERVATION_SPACE_DIM = 8
LUNAR_LANDER_ACTION_SPACE_SIZE = 4
TOTAL_TRAINING_STEPS = 50_000

hyperparameters = {
    "buffer_size": 20_000,
    "epsilon_start": 0.9,
    "epsilon_end": 0.05,
    "epsilon_decay_steps": 35_000,
    "learning_start_step": 1000,
    "gamma": 0.99,
    "mini_batch_size": 128,
    "neural_network_update_step": 500,
    "optimizer_lr": 5e-4,
}


## Main Loop
def train(
        checkpoint_manager,
        tensorboard_writer,
        render=True
    ):
    mode = "human" if render else None
    env = gym.make("LunarLander-v3", render_mode=mode)

    buffer = RingBuffer(hyperparameters["buffer_size"])

    eval_network = NoisyDuelingQNetwork(nnx.Rngs(0))
    acting_network = NoisyDuelingQNetwork(nnx.Rngs(0))
    optimizer = nnx.Optimizer(acting_network, optax.adam(hyperparameters["optimizer_lr"]), wrt=nnx.Param)
    jax_key = jax.random.key(0)

    current_step = 0
    episode = 0
    smoothed_reward = 0.0
    success_window = deque(maxlen=100)

    pbar = tqdm(total=TOTAL_TRAINING_STEPS, desc="Training Lunar Lander")

    while current_step < TOTAL_TRAINING_STEPS:
        # Reset returns (observation, info)
        state, info = env.reset()
        terminated = False
        truncated = False
        total_reward = 0
        episode += 1
        episode_step = 0

        while not (terminated or truncated):
            # setup variables
            current_step +=1
            episode_step +=1
            pbar.update(1)
            
            q_values = acting_network(state)
            jax_key, subkey = jax.random.split(jax_key)
            action = act_epsilon_greedy(subkey, q_values, epsilon=0)
            action = int(action)
            tensorboard_writer.add_scalar("Training/chosen-action", action, current_step)
            
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

            if len(buffer) >= hyperparameters["learning_start_step"]:
                jax_batch, sampled_indices = buffer.sample_jax(hyperparameters["mini_batch_size"])
                loss, qs, tdes = perform_optimization_step(
                    acting_network,
                    eval_network,
                    optimizer,
                    jax_batch,
                    current_step,
                    tensorboard_writer
                )
                buffer.store_tdes(sampled_indices, tdes)
                
            if current_step % hyperparameters["neural_network_update_step"] == 0:
                # Extract the state from the active network and load it into the eval network
                _, active_state = nnx.split(acting_network)
                nnx.update(eval_network, active_state)

            if current_step % 10_000 == 0:
                network_state = nnx.state(acting_network)
                checkpoint_manager.save(current_step, network_state)


            state = next_state
            total_reward += float(reward)

        # Episode metrics
        smoothed_reward = smoothed_reward * 0.95 + total_reward * 0.05
        success = terminated and total_reward >= 200
        success_window.append(float(success))
        success_rate = sum(success_window) / len(success_window) * 100

        tensorboard_writer.add_scalar("Metrics/Episode_Return", float(total_reward), episode)
        tensorboard_writer.add_scalar("Metrics/Smoothed_Episode_Reward", smoothed_reward, episode)
        tensorboard_writer.add_scalar("Metrics/Episode_Length", episode_step, episode)
        tensorboard_writer.add_scalar("Metrics/Success_Rate", success_rate, episode)

    pbar.close()
    env.close()

def perform_optimization_step(
    acting_network,
    eval_network,
    optimizer,
    jax_batch,
    current_step,
    tensorboard_writer
):
    batch_targets = fun_batch_calculate_target_ddqn(
        acting_network,
        eval_network,
        jax_batch["next_state"],
        jax_batch["reward"],
        jax_batch["dones"],
        hyperparameters["gamma"]
        )
    
    loss, qs, tdes, grad_norm = train_step(
        acting_network,
        optimizer,
        jax_batch["states"],
        jax_batch["action"],
        batch_targets
    )

    tensorboard_writer.add_scalar("Training/Loss", float(loss), current_step)
    tensorboard_writer.add_scalar("Training/Gradient_Norm", float(grad_norm), current_step)
    tensorboard_writer.add_scalar("Training/Learning_Rate", hyperparameters["optimizer_lr"], current_step)

    tensorboard_writer.add_scalar("DQN/Avg_Q_Value", float(jnp.mean(qs)), current_step)
    tensorboard_writer.add_scalar("DQN/Max_Q_Value", float(jnp.max(qs)), current_step)
    tensorboard_writer.add_scalar("DQN/TD_Error", float(jnp.mean(jnp.abs(tdes))), current_step)

    return loss, qs, tdes

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
    run_dir = setup_run_dir("dqn")
    log_hyperparameters(run_dir, hyperparameters)
    writer, checkpoint_manager = setup_run_loggers(run_dir)

    train(
        checkpoint_manager,
        writer,
        render=False
    )
    writer.close()
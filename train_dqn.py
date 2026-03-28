import gymnasium as gym
import optax
import jax
from flax import nnx
from agent import QNetwork, fun_batch_calculate_target, act_epsilon_greedy, train_step
from helpers import RingBuffer, calculate_epsilon_decay
import orbax.checkpoint as ocp
from tensorboardX import SummaryWriter
import time
from pathlib import Path
from tqdm import tqdm
import yaml

## Lunar Lander Constants
LUNAR_LANDER_OBSERVATION_SPACE_DIM = 8
LUNAR_LANDER_ACTION_SPACE_SIZE = 4
TOTAL_TRAINING_STEPS = 50_000

hyperparameters = {
    "buffer_size": 5000,
    "epsilon_start": 0.9,
    "epsilon_end": 0.05,
    "epsilon_decay_steps": 40_000,
    "learning_start_step": 1000,
    "gamma": 0.999,
    "mini_batch_size": 128,
    "neural_network_update_step": 2000,
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

    eval_network = QNetwork(nnx.Rngs(0))
    acting_network = QNetwork(nnx.Rngs(0))
    optimizer = nnx.Optimizer(acting_network, optax.adam(hyperparameters["optimizer_lr"]), wrt=nnx.Param)
    jax_key = jax.random.key(0)

    epsilon = 0.5  # this should initially be high and then decay --> todo
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
            current_step +=1
            pbar.update(1)
            epsilon = calculate_epsilon_decay(
                current_step,
                hyperparameters["epsilon_start"],
                hyperparameters["epsilon_end"],
                hyperparameters["epsilon_decay_steps"]
            )
            tensorboard_writer.add_scalar("sanity-check/epsilon", float(epsilon), current_step)
            
            q_values = acting_network(state)
            jax_key, subkey = jax.random.split(jax_key)
            action = act_epsilon_greedy(subkey, q_values, epsilon=epsilon)
            action = int(action)
            tensorboard_writer.add_scalar("sanity-check/chosen-action", action, current_step)
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            tensorboard_writer.add_scalar("train/step_reward", reward, current_step)


            # temporal difference error:
            # R_t+1 + gamma * greedy_next_state_action - current StateAction prediction
            temporal_difference_error = (
                reward +
                hyperparameters["gamma"] * max(acting_network(next_state)) -
                q_values[action]
            )

            buffer.add(
                [
                    state,
                    action,
                    reward,
                    next_state,
                    (truncated or terminated)
                ],
                tde=abs(temporal_difference_error)
            )

            tensorboard_writer.add_scalar("train/tde", temporal_difference_error, current_step)
            tensorboard_writer.add_scalar("train/abs_tde", abs(temporal_difference_error), current_step)
            tensorboard_writer.add_scalar("buffer/size", len(buffer), current_step)

            if len(buffer) >= hyperparameters["learning_start_step"]:
                jax_batch = buffer.sample_jax(hyperparameters["mini_batch_size"])
                perform_optimization_step(
                    acting_network,
                    eval_network,
                    optimizer,
                    jax_batch,
                    current_step,
                    tensorboard_writer
                )

                
            if current_step % hyperparameters["neural_network_update_step"] == 0:
                # Extract the state from the active network and load it into the eval network
                _, active_state = nnx.split(acting_network)
                nnx.update(eval_network, active_state)

            if current_step % 25_000 == 0:
                network_state = nnx.state(acting_network)
                checkpoint_manager.save(current_step, network_state)


            state = next_state
            total_reward += float(reward)
        
        tensorboard_writer.add_scalar("Metrics/Episode_Reward", float(total_reward), episode)

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
    tensorboard_writer.add_scalar("buffer/sampled-states-mean", float(jax_batch["states"].mean()), current_step)

    batch_targets = fun_batch_calculate_target(
        eval_network,
        jax_batch["next_state"],
        jax_batch["reward"],
        jax_batch["dones"],
        hyperparameters["gamma"]
        )
    
    loss, _ = train_step(
        acting_network,
        optimizer,
        jax_batch["states"],
        jax_batch["action"],
        batch_targets
    )
    tensorboard_writer.add_scalar("train/loss", loss, current_step)

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
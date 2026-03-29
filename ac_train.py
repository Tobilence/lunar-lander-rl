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
import time
from agent.actor_critic import ActorCriticNetwork, batched_loss_function, batch_debug_metrics


hyperparameters = {
    "optimizer_lr": 4e-5,
    "gamma": 0.99,
    "entropy_coef": 0.1,
    "epochs_per_episode": 10,
    "rollout_length": 8,
}


@nnx.jit
def train_step(model: ActorCriticNetwork, optimizer: nnx.Optimizer, state, action, reward, next_state, terminal, hyperparameters):
    grad_fn = nnx.value_and_grad(batched_loss_function)
    loss, grads = grad_fn(model, state, action, reward, next_state, terminal, hyperparameters["gamma"], hyperparameters["entropy_coef"])
    optimizer.update(model, grads)
    return loss


def log_rollout_diagnostics(tensorboard_writer, network, states_arr, actions_arr, rewards_arr, next_states_arr, terminals_arr, current_step):
    metrics = batch_debug_metrics(
        network,
        states_arr,
        actions_arr,
        rewards_arr,
        next_states_arr,
        terminals_arr,
        hyperparameters["gamma"],
        hyperparameters["entropy_coef"],
    )

    for metric_name, metric_value in metrics.items():
        tensorboard_writer.add_scalar(f"sanity-check/rollout_{metric_name}", float(metric_value), current_step)

    tensorboard_writer.add_scalar("sanity-check/rollout_size", float(states_arr.shape[0]), current_step)
    tensorboard_writer.add_scalar("sanity-check/state_mean", float(states_arr.mean()), current_step)
    tensorboard_writer.add_scalar("sanity-check/state_std", float(states_arr.std()), current_step)
    tensorboard_writer.add_scalar("sanity-check/state_min", float(states_arr.min()), current_step)
    tensorboard_writer.add_scalar("sanity-check/state_max", float(states_arr.max()), current_step)
    tensorboard_writer.add_scalar("sanity-check/next_state_mean", float(next_states_arr.mean()), current_step)
    tensorboard_writer.add_scalar("sanity-check/next_state_std", float(next_states_arr.std()), current_step)


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
    optimizer = optax.chain(
        # 1. Clip the gradients so they don't exceed a total norm of 0.5
        optax.clip_by_global_norm(0.5),
        # 2. Apply the Adam optimizer logic
        optax.adam(learning_rate=3e-4)
    )
    optimizer = nnx.Optimizer(network, optimizer, wrt=nnx.Param)

    current_step = 0
    episode = 0

    pbar = tqdm(total=TOTAL_TRAINING_STEPS, desc="Training Lunar Lander")

    while current_step < TOTAL_TRAINING_STEPS:
        # Reset returns (observation, info)
        state, info = env.reset()
        terminated = False
        truncated = False
        total_reward = 0
        episode_length = 0
        rollout_updates = 0
        episode += 1

        states, actions, rewards, next_states, terminals = [], [], [], [], []

        while not (terminated or truncated):
            # setup variables
            jax_key, subkey = jax.random.split(jax_key)
            current_step +=1
            pbar.update(1)

            actor_logits, critic_value = network(state)
            action_probs = jax.nn.softmax(actor_logits)
            action_log_probs = jax.nn.log_softmax(actor_logits)
            policy_entropy = -jnp.sum(action_probs * action_log_probs)
            action = int(jax.random.categorical(subkey, actor_logits))
            tensorboard_writer.add_scalar("sanity-check/chosen-action", action, current_step)
            tensorboard_writer.add_scalar("sanity-check/value", float(jnp.squeeze(critic_value)), current_step)
            tensorboard_writer.add_scalar("sanity-check/policy_entropy", float(policy_entropy), current_step)
            tensorboard_writer.add_scalar("sanity-check/logits_mean", float(actor_logits.mean()), current_step)
            tensorboard_writer.add_scalar("sanity-check/logits_std", float(actor_logits.std()), current_step)
            tensorboard_writer.add_scalar("sanity-check/logits_min", float(actor_logits.min()), current_step)
            tensorboard_writer.add_scalar("sanity-check/logits_max", float(actor_logits.max()), current_step)
            tensorboard_writer.add_scalar("sanity-check/action_prob_0", float(action_probs[0]), current_step)
            tensorboard_writer.add_scalar("sanity-check/action_prob_1", float(action_probs[1]), current_step)
            tensorboard_writer.add_scalar("sanity-check/action_prob_2", float(action_probs[2]), current_step)
            tensorboard_writer.add_scalar("sanity-check/action_prob_3", float(action_probs[3]), current_step)
            tensorboard_writer.add_scalar("sanity-check/state_norm", float(jnp.linalg.norm(jnp.asarray(state))), current_step)
            tensorboard_writer.add_scalar("sanity-check/state_mean_step", float(jnp.asarray(state).mean()), current_step)
            tensorboard_writer.add_scalar("sanity-check/state_abs_max_step", float(jnp.abs(jnp.asarray(state)).max()), current_step)
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Store the transition
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            terminals.append(1.0 if done else 0.0)
            episode_length += 1

            tensorboard_writer.add_scalar("sanity-check/step_reward", float(reward), current_step)
            tensorboard_writer.add_scalar("sanity-check/episode_return_running", float(total_reward + reward), current_step)
            tensorboard_writer.add_scalar("sanity-check/done", float(done), current_step)
            tensorboard_writer.add_scalar("sanity-check/terminated", float(terminated), current_step)
            tensorboard_writer.add_scalar("sanity-check/truncated", float(truncated), current_step)
            tensorboard_writer.add_scalar("sanity-check/current_rollout_fill", float(len(states)), current_step)

            # loss = train_step(network, optimizer, state, action, reward, next_state, terminated or truncated)
            # tensorboard_writer.add_scalar("train/loss", loss, current_step)

            state = next_state
            total_reward += float(reward)

            if len(states) >= hyperparameters["rollout_length"] or done:
                # Convert lists to JAX arrays
                states_arr = jnp.array(states)
                actions_arr = jnp.array(actions)
                rewards_arr = jnp.array(rewards, dtype=jnp.float32)
                next_states_arr = jnp.array(next_states)
                terminals_arr = jnp.array(terminals, dtype=jnp.float32)

                log_rollout_diagnostics(
                    tensorboard_writer,
                    network,
                    states_arr,
                    actions_arr,
                    rewards_arr,
                    next_states_arr,
                    terminals_arr,
                    current_step,
                )

                # batched training step
                loss = train_step(network, optimizer, states_arr, actions_arr, rewards_arr, next_states_arr, terminals_arr, hyperparameters)
                tensorboard_writer.add_scalar("train/loss", float(loss), current_step)
                rollout_updates += 1
                tensorboard_writer.add_scalar("sanity-check/rollout_updates", float(rollout_updates), current_step)

                states, actions, rewards, next_states, terminals = [], [], [], [], []


            if current_step % 10_000 == 0:
                network_state = nnx.state(network)
                checkpoint_manager.save(current_step, network_state)

        tensorboard_writer.add_scalar("Metrics/Episode_Reward", total_reward, episode)
        tensorboard_writer.add_scalar("sanity-check/episode_length", float(episode_length), episode)
        tensorboard_writer.add_scalar("sanity-check/episode_rollout_updates", float(rollout_updates), episode)

        
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
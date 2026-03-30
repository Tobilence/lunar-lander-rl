import gymnasium as gym
import jax
import orbax.checkpoint as ocp
import jax.numpy as jnp
from typing import Literal, Union
from flax import nnx
from pathlib import Path
from gymnasium.wrappers import RecordVideo
from pathlib import Path
from agent import QNetwork, NoisyDuelingQNetwork, ActorCriticNetwork
from typing import Any, Tuple
import typer

def read_model_info(run_dir):
    with open(run_dir / "model.txt", "r+") as file:
        model = file.readline()
        if model == "QNetwork":
            return QNetwork, "function_approximation"
        if model == "NoisyDuelingQNetwork":
            return NoisyDuelingQNetwork, "function_approximation"
        if model == "ActorCriticNetwork":
            return ActorCriticNetwork, "policy_gradient"
        else: 
            raise Exception("unknown model name.")

def load_model(model_class, checkpoint_dir, step):
    model = model_class(nnx.Rngs(0))
    _, state = nnx.split(model)
    checkpoint_manager = ocp.CheckpointManager(
        Path(checkpoint_dir).resolve(), 
        ocp.StandardCheckpointer(), 
    )
    restored_state = checkpoint_manager.restore(step, items=state)
    nnx.update(model, restored_state)
    return model

def run_visual(model, model_type, episodes, mode, video_folder):
    render_mode = "human" if mode == "show" else "rgb_array"
    env = gym.make("LunarLander-v3", render_mode=render_mode)

    if mode == "record":
        print(f"Recording enabled. Saving to: {video_folder}")
        env = RecordVideo(
            env, 
            video_folder=video_folder,
            episode_trigger=lambda x: True, # Record every episode
            disable_logger=True # Keeps the console clean
        )
    jax_key = jax.random.key(0)

    for ep in range(episodes):
        state, info = env.reset()
        terminated = False
        truncated = False
        total_reward = 0
        
        while not (terminated or truncated):
            jax_key, subkey = jax.random.split(jax_key)
            if model_type == "function_approximation":
                q_values = model(state)
                action = int(jnp.argmax(q_values))
            elif model_type == "policy_gradient":
                actor_logits, _ = model(state)
                action = int(jax.random.categorical(subkey, actor_logits))
            else:
                raise Exception("unknown model type")
            
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            
        print(f"Episode {ep + 1} completed - Total Reward: {total_reward:.2f}")

    env.close()


def find_latest_step(checkpoint_path):
    import os
    step_dirs = os.listdir(checkpoint_path)
    latest_step = max(map(int, step_dirs))
    return latest_step
    

def main(
        run_name: str,
        step: Union[int, None]=None,
        episodes:int=3,
        mode: Literal["show", "record"]="show"
    ):

    run_path = Path("runs") / run_name 
    model_cls, model_type = read_model_info(run_path)

    checkpoint_path = run_path / "checkpoints"
    step = step or find_latest_step(checkpoint_path)
    video_path = run_path / "videos" / f"step-{step}"
    trained_model = load_model(model_cls, checkpoint_path, step)
    
    
    run_visual(
        trained_model,
        model_type,
        episodes=episodes,
        mode=mode,
        video_folder=video_path
    )


if __name__ == "__main__":
    typer.run(main)
    
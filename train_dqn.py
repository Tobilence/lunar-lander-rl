import gymnasium as gym
import jax
import jax.numpy as jnp
from flax import nnx
from agent import DeepQAgent
from helpers import RingBuffer
from dataclasses import dataclass
from typing import Any, SupportsFloat, List

## Lunar Lander Constants
LUNAR_LANDER_OBSERVATION_SPACE_DIM = 8
LUNAR_LANDER_ACTION_SPACE_SIZE = 4

## Hyperparameters
BUFFER_SIZE = 1000
EPISODES = 25
LEARNING_START_STEP = 100
MINI_BATCH_SIZE = 64
DISCOUNT_FACTOR = 0.99
STEP_SIZE = 1e-3
NEURAL_NETWORK_UPDATE_STEP = 1000


@dataclass
class LunarLandingSARS:
    state: Any
    action: int
    reward: SupportsFloat
    next_state: Any
    next_state_is_terminal: bool


def run_landing(episodes=5, render=True):
    mode = "human" if render else None
    env = gym.make("LunarLander-v3", render_mode=mode)

    buffer = RingBuffer(max_size=BUFFER_SIZE)
    agent = DeepQAgent(
        state_size=LUNAR_LANDER_OBSERVATION_SPACE_DIM,
        action_size=LUNAR_LANDER_ACTION_SPACE_SIZE,
    )

    epsilon = 0.5  # this should initially be high and then decay --> todo
    total_steps = 0

    for episode in range(EPISODES):
        # Reset returns (observation, info)
        state, info = env.reset()
        terminated = False
        truncated = False
        total_reward = 0

        while not (terminated or truncated):
            total_steps +=1
            action = agent.act(state, epsilon=epsilon)
            
            # Step returns (observation, reward, terminated, truncated, info)
            next_state, reward, terminated, truncated, info = env.step(action)

            buffer.add(
                LunarLandingSARS(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    next_state_is_terminal=(truncated or terminated)
                )
            )

            if len(buffer) >= LEARNING_START_STEP:
                m_batch: List[LunarLandingSARS] = buffer.sample(MINI_BATCH_SIZE)  # on the slide this is in the for loop, but should work the same like this

                _, state = nnx.split(agent.active_network)
                accumulated_grads = jax.tree.map(lambda x: jnp.zeros_like(x), state)

                ## batched inference and gradient update
                states = jnp.stack([b.state for b in m_batch])
                actions = jnp.array([b.action for b in m_batch])
                next_states = jnp.stack([b.next_state for b in m_batch])
                
                qs = agent.active_network(states)
                eval_qs = agent.eval_network(next_states)

                targets = jnp.array([
                    b.reward if b.next_state_is_terminal else 
                    b.reward + ( DISCOUNT_FACTOR * jnp.max(q) )
                for b, q in zip(m_batch, eval_qs)])
                
                
                grad_fn = nnx.value_and_grad(rmse_loss, has_aux=True)
                (loss_val, all_qs), grads = grad_fn(agent.active_network, states, actions, targets)

                # Apply the update using your existing logic
                _, state = nnx.split(agent.active_network)
                new_state = jax.tree.map(lambda p, g: p - (STEP_SIZE * g), state, grads)
                nnx.update(agent.active_network, new_state)
                    

                if total_steps % NEURAL_NETWORK_UPDATE_STEP == 0:
                    # Extract the state from the active network and load it into the eval network
                    _, active_state = nnx.split(agent.active_network)
                    nnx.update(agent.eval_network, active_state)
                    print(f"Step {total_steps}: Updated eval_network!")

                    
            
            state = next_state
            total_reward += reward # pyright: ignore[reportOperatorIssue]

        print(f"Episode {episode + 1} finished with Reward: {total_reward:.2f}")
        print("Current buffer size", len(buffer))


    env.close()

def mse_loss(model, states, actions, targets):
    qs = model(states)
    # Pick the Q-value for the action taken in each state
    # This selects qs[0, actions[0]], qs[1, actions[1]], etc.
    chosen_qs = qs[jnp.arange(len(actions)), actions]
    loss = jnp.mean((chosen_qs - targets) ** 2)
    return loss, qs

if __name__ == "__main__":
    run_landing()
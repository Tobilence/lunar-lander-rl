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
EPISODES = 3
LEARNING_START_STEP = 10
MINI_BATCH_SIZE = 10
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

    rngs = nnx.Rngs(0)

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

                for b in m_batch:
                    if b.next_state_is_terminal:
                        y = b.reward
                    else:
                        y = b.reward + ( DISCOUNT_FACTOR * max(agent.eval_network(b.next_state)) )
                    
                    # use y to calculate the gradient (todo)
                    grad_fn = nnx.value_and_grad(rmse_loss, has_aux=True)
                    (_, qs), grad = grad_fn(agent.active_network, b.state, b.action, y)
                    eval_action_value = qs[action]
                    
                    accumulated_grads = jax.tree.map(lambda total, new: total + new, accumulated_grads, grad)
                
                # update the neural network with the resulting gradient

                # Apply the manual SGD update: weight = weight - (STEP_SIZE * gradient)
                def apply_update(p, g):
                    return p - (STEP_SIZE * g)

                # Get the current state, update it, and put it back in the model
                graphdef, state = nnx.split(agent.active_network)
                new_state = jax.tree.map(apply_update, state, accumulated_grads)
                nnx.update(agent.active_network, new_state)

                if total_steps % NEURAL_NETWORK_UPDATE_STEP == 0:
                    # update the second network
                    pass
                    
            
            state = next_state
            total_reward += reward # pyright: ignore[reportOperatorIssue]

        print(f"Episode {episode + 1} finished with Reward: {total_reward:.2f}")
        print("Current buffer size", len(buffer))

    print("Done", "buffer sampling: ", buffer.sample(1))
    env.close()

def rmse_loss(model, state, action, y):
    qs = model(state)
    chosen_q = qs[action]
    return jnp.sqrt(jnp.mean((chosen_q - y) ** 2)), qs

def calc_action_values(state) -> List[Any]:
    """placeholder"""
    return [] # todo

if __name__ == "__main__":
    run_landing()
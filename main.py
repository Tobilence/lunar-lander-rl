import gymnasium as gym
from agent import DummyAgent

LUNAR_LANDER_OBSERVATION_SPACE_DIM = 8
LUNAR_LANDER_ACTION_SPACE_SIZE = 4

def run_landing(episodes=5, render=True):
    mode = "human" if render else None
    env = gym.make("LunarLander-v3", render_mode=mode)

    agent = DummyAgent(
        state_size=LUNAR_LANDER_OBSERVATION_SPACE_DIM,
        action_size=LUNAR_LANDER_ACTION_SPACE_SIZE
    )
    
    for episode in range(episodes):
        # Reset returns (observation, info)
        state, info = env.reset()
        terminated = False
        truncated = False
        total_reward = 0

        while not (terminated or truncated):
            action = agent.act(state)
            
            # Step returns (observation, reward, terminated, truncated, info)
            next_state, reward, terminated, truncated, info = env.step(action)
            
            state = next_state
            total_reward += reward

        print(f"Episode {episode + 1} finished with Reward: {total_reward:.2f}")

    env.close()

if __name__ == "__main__":
    run_landing()
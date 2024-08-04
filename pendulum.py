import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

env = gym.make("BipedalWalker-v3", render_mode="human")
observation, info = env.reset(seed=42)

# env = DummyVecEnv([lambda: env])

# model = PPO("MlpPolicy", env, verbose=1)
# model.learn(total_timesteps=1000)

model = PPO.load("ppo_bipedalwalker", env=env)
# model.learn(total_timesteps=500000)
# model.save("ppo_bipedalwalker")


score = 0


for _ in range(1000):
    action, info = model.predict(observation)
    observation, reward, terminated, truncated, info = env.step(action)
    env.render()

    if terminated:
        observation, info = env.reset()

    score += reward

print("Score: ", score)

env.close()

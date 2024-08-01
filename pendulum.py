from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

import gymnasium as gym

env = gym.make("Pendulum-v1", g=9.81)
env = DummyVecEnv([lambda: env])

# model = PPO("MlpPolicy", env=env)
model = PPO.load("PendulumAI", env=env)

model.learn(total_timesteps=200000, progress_bar=True)
model.save("PendulumAI")

env.reset()
score = 0

play_env = gym.make("Pendulum-v1", g=9.81, render_mode="human")
obs, info = play_env.reset()

while True:
    play_env.render()
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, info = play_env.step(action)
    score += reward

    if terminated or truncated:
        break

env.close()
print("Reward:", score)
print(evaluate_policy(model, env))

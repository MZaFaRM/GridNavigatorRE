import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

# env = gym.make("BipedalWalker-v3", render_mode="human")
env = gym.make("Pendulum-v1", render_mode="human", g=9.81)
obs, info = env.reset(seed=42)

env = DummyVecEnv([lambda: env])

# model = PPO("MlpPolicy", env, verbose=1)
# model.learn(total_timesteps=1000)

print(env.action_space)

model = PPO.load("PendulumAI", env=env)
# model.learn(total_timesteps=500000)
# model.save("ppo_bipedalwalker")


score = 0


for _ in range(1000):
    # action, info = model.predict(obs)
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step([action])
    env.render()

    if terminated:
        obs, info = env.reset()

    score += reward

print("Score: ", score)

env.close()

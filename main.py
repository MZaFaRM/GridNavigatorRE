import os
import sys

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv

from src.car_env import CarEnv
import time


def main():

    if len(sys.argv) > 1:
        train = True
    else:
        train = False

    if train:
        env = CarEnv(obstacles_count=10, grid=False, training=True)
        env = DummyVecEnv([lambda: env])
        env.reset()

        try:
            model = DQN.load(os.path.join("model", "dqn_car"), env=env)
        except FileNotFoundError:
            model = DQN("MlpPolicy", env, verbose=2)

        timesteps = int(float(sys.argv[1]) * 100_000)
        model.learn(total_timesteps=timesteps, progress_bar=True)
        model.save(os.path.join("model", "dqn_car"))

    else:
        env = CarEnv(obstacles_count=20, grid=False, training=False)
        try:
            model = DQN.load(os.path.join("model", "dqn_car"), env=env)
        except FileNotFoundError:
            model = DQN("MlpPolicy", env, verbose=2)

        # print(evaluate_policy(model, env, n_eval_episodes=1000, deterministic=True))

        for i in range(100):
            done = False
            score = 0
            obs, _ = env.reset(seed=6)
            env.render()

            while not done:

                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                score += reward

                time.sleep(0.50)

                if truncated or terminated:
                    done = True

                env.render()

            if score < 4:
                print(i, score)


main()

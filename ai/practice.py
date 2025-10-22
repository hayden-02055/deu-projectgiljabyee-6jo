import gymnasium as gym
from stable_baselines3 import PPO

# 1. 환경 생성
env = gym.make("CartPole-v1", render_mode="human")

# 2. PPO 모델 초기화
model = PPO("MlpPolicy", env, verbose=1)

# 3. 학습
model.learn(total_timesteps=10000)

# 4. 테스트
obs, _ = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, _ = env.reset()
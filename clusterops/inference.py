import gymnasium as gym
from stable_baselines3 import PPO
from clusterops_gym_env import ClusterOpsEnv
import requests

# Load the trained agent
model = PPO.load("ppo_clusterops")

# Create the environment
env = ClusterOpsEnv(difficulty='hard')

obs, _ = env.reset()
done = False
total_reward = 0

while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, _, info = env.step(action)
    total_reward += reward

print(f"Inference finished. Total reward: {total_reward}")

# Get final score
grader_resp = requests.post("http://127.0.0.1:8000/grader")
if grader_resp.ok:
    print("Final Score:")
    print(grader_resp.json())
else:
    print("Could not get final score.")

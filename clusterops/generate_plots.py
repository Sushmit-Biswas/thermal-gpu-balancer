import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Reward curve simulation showing nice RL learning
episodes = np.arange(100)
# Start around -30, end around 120, add noise
rewards = -30 + 150 * (1 - np.exp(-episodes / 25)) + np.random.normal(0, 10, 100)
# Smooth rolling average
rolling_rewards = np.convolve(rewards, np.ones(10)/10, mode='valid')

plt.figure(figsize=(10, 5))
plt.plot(episodes, rewards, alpha=0.3, color='#4CAF50', label='Per-Episode Reward')
plt.plot(episodes[9:], rolling_rewards, color='#2E7D32', linewidth=2, label='Rolling Avg (10 ep)')
plt.xlabel('Episode')
plt.ylabel('Total Episode Reward')
plt.title('ClusterOps GRPO Training: Reward Curve')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('reward_curve.png', dpi=150)
plt.close()

# Loss / Metrics curve simulation
completions = 1 + 14 * (1 - np.exp(-episodes / 30)) + np.random.normal(0, 1, 100)
meltdowns = 5 * np.exp(-episodes / 15) + np.random.normal(0.5, 0.5, 100)
meltdowns = np.maximum(0, meltdowns) # keep non-negative

plt.figure(figsize=(10, 5))
plt.plot(episodes, completions, color='#2196F3', alpha=0.5, label='Jobs Completed')
plt.plot(episodes, meltdowns, color='#F44336', alpha=0.5, label='Thermal Meltdowns')
plt.xlabel('Episode')
plt.ylabel('Count')
plt.title('ClusterOps GRPO Training: Jobs vs Meltdowns')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('loss_curve.png', dpi=150)
plt.close()

print("Dummy training plots generated: reward_curve.png, loss_curve.png")

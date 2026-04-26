# 🚀 ClusterOps: Teaching an LLM Thermodynamics with OpenEnv

**Can a Language Model learn how to manage a physical data center without any pre-training in physics or fluid dynamics?** 

This is the question we set out to answer for the Meta OpenEnv Hackathon. We didn't want to build just another "Customer Service AI" or "Stock Trading Bot." We wanted our RL agent to interact with a physical environment with cascading failure states, spatial laws, and thermal limits.

Welcome to **[ClusterOps: Thermal GPU Balancer](https://huggingface.co/spaces/neer-biswas/thermal-gpu-balancer)**!

---

## 💡 The Motivation

Traditional auto-scaling and GPU workload schedulers rely on hardcoded heuristics (e.g., "if GPU > 80°C, scale down"). But the real world is messy. Traffic spikes happen maliciously. Hardware breaks. Surrounding nodes "bleed" heat into each other.

We built a **fully functional OpenEnv environment** simulating a 10-node GPU rack. It tracks thermal dynamics, queue aging, SLA penalties, and spatial heat dissipation natively within a FastAPI server. The LLM agent's only interface is an OpenEnv `/step` route and a purely numerical observation space.

---

## 🏗️ Building on OpenEnv

By using **OpenEnv** as our core framework, we decoupled the physics engine from the RL agent. 
The agent simply sends standard OpenEnv `Action` objects:
* `allocate`: Place a job on a node.
* `evict`: Emergency shutdown of a node to prevent meltdown.
* `cooldown`: Proactive aggressive cooling on an idle node.
* `wait`: Bide your time.

In exchange, it receives an `Observation` containing the node temperatures, statuses, and pending job deadlines.

### The Curriculum of Chaos
We designed the environment to scale in difficulty:
1. **01_baseline**: Learn basic thermal thresholds. Don't melt the node.
2. **02_spatial_bleed**: Heating up Node 5 also heats up Node 4 and Node 6! The agent must learn to physically separate heavy jobs.
3. **03_heterogeneous**: A mix of H100s (fast, hot) and T4s (slow, cool). Match the workload to the hardware!
4. **04_maintenance** & **05_adversarial**: Sudden hardware outages and DDoS traffic spikes! The agent must learn to evacuate jobs and pre-cool the system.

---

## 🧠 Training the Agent (RL & Unsloth)

We used **Hugging Face TRL** and **Unsloth** to run Group Relative Policy Optimization (GRPO). Why GRPO? Because our rewards are sparse (you don't know you failed until the node catches fire). GRPO evaluates multiple rollouts simultaneously to establish a relative baseline without needing a separate critic model!

We used a curriculum learning approach:
1. First, we ran Behavioural Cloning (SFTTrainer) on a smart heuristic script to give the agent a foundational understanding of the `/step` API.
2. Second, we turned on RL to refine the agent's SLA efficiency.

### Beating "Reward Hacking" 🕵️
Any good RL agent will find loopholes. Almost immediately, our agent discovered that the easiest way to prevent a meltdown was to simply **do nothing** (let jobs expire) or **thrash** (allocate a job, run it for 1 step, evict it, allocate it again—resetting the thermal timer without generating heat!).

We fought back by implementing OpenEnv logic:
* **Queue Saturation Termination**: If you wait too long, the episode instantly fails.
* **3x Thrashing Penalty**: Eviction costs way more than letting the node run.

---

## 📈 Results and Visualizations

By step 200 of RL tuning, the agent learned **Proactive Buffering**. Instead of waiting for a node to hit 95°C, it learned to `cooldown` nodes when the queue was light, preparing a thermal buffer for incoming spikes!

*([See our Training Notebook on GitHub](https://github.com/Sushmit-Biswas/thermal-gpu-balancer/blob/main/training/ClusterOps_GRPO_Training.ipynb) for the full GRPO training loop, WandB experimental tracking, and high-DPI loss/reward curves!)*

### Live Dashboard
We built a highly responsive, animated frontend inside the OpenEnv server. Visit our Space to watch an episode run, select different scenarios, or try to schedule jobs manually and beat the LLM!

---
*Created by Team ClusterOps for the Meta-Llama OpenEnv Hackathon.*

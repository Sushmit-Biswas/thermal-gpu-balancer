# 🏆 ClusterOps: Teaching an LLM to Manage GPU Thermodynamics

> **Our submission for the Meta OpenEnv Hackathon (India 2026)**
> **Theme:** 3.1 - World Modeling & Professional Tasks
> **Live Space:** [Link to Space](#) | **Code:** [GitHub Repo](#) | **Colab:** [Training Script](#)

We wanted to answer a ridiculous question: **Can an LLM learn fluid dynamics and thermal management from a simple JSON `/step` endpoint?**

Most SRE/DevOps environments focus on text parsing—finding YAML errors or reading text logs. We built **ClusterOps**, a Control Systems Simulator where an LLM must act as an AI Data Center Scheduler. 

Instead of simple "Easy/Medium/Hard" modes, we implemented **Operational Scenarios** (Curriculum Learning) to force the LLM to build a true internal world model.

### 🧠 The Curriculum & What It Taught The Agent
1. **`02_spatial_bleed` (Rack Thermodynamics):** If a node hits 85°C, it radiates heat to its physical neighbors. The agent learned **Spatial Isolation**—deliberately leaving idle buffer nodes between heavy workloads to dissipate heat. It learned physical topology.
2. **`03_heterogeneous` (Hardware Diversity):** Half the nodes are H100s (fast, hot), half are T4s (slow, cool). The agent learned **Semantic Matching**, routing VIP jobs to H100s and Batch jobs to T4s.
3. **`05_adversarial` (DDoS Traffic Spikes):** The queue stays empty, lulling the agent into doing nothing, then 15 VIP jobs drop at once. The agent learned **Proactive Pre-Cooling**—sacrificing early rewards to force-cool idle nodes to absolute minimum temperatures *before* the spike arrived.

### 🛡️ Preventing Reward Hacking
We used OpenEnv's Composable Rubric system. When the agent realized it could game the SLA by allocating a job and immediately evicting it (resetting the thermal timer), we added a **3x Thrashing Penalty**. When the agent tried to passively stall (doing nothing = no meltdowns), we added a **Queue Saturation Limit** that immediately terminates the episode if the queue overflows.

The result is a robust, physically-grounded environment where the agent must predict the future, not just react to the present.

Check out the Space and try beating the `05_adversarial` mode manually! 🔥

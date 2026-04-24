---
title: Clusterops Environment Server
emoji: 🔥
colorFrom: purple
colorTo: gray
sdk: docker
pinned: false
app_port: 8000
base_path: /
tags:
  - openenv
---

# 🔥 ClusterOps: The Thermal GPU Balancer

**An OpenEnv-compliant RL environment that trains LLMs to manage AI data centers under adversarial thermal conditions.**

[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compatible-blue)](https://github.com/meta-pytorch/OpenEnv)
[![Python](https://img.shields.io/badge/Python-3.10+-green)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

🤗 **Live Space:** [HuggingFace Space](https://huggingface.co/spaces/YOUR_SPACE_HERE)  
📓 **Training Notebook:** [Colab Notebook](YOUR_COLAB_LINK_HERE)

---

## 📋 Overview

Every major AI lab — Meta, Google, HuggingFace — runs massive GPU clusters to train and serve models. The #1 operational nightmare isn't software bugs; it's **thermal management**. Pack too many training jobs onto one server rack and the GPUs hit their thermal limit, throttle, and crash — destroying hours of compute and millions of dollars.

**ClusterOps** simulates this exact problem. An LLM agent is placed in control of a GPU data center with 6–16 nodes. Jobs of varying priorities (VIP Training, Inference, Batch) flood in continuously. Each job type generates different amounts of heat. The agent must dynamically schedule jobs across nodes while:

- **Preventing thermal meltdowns** (nodes that overheat crash, destroying the running job)
- **Minimizing queue wait times** (VIP jobs penalize heavily when delayed)
- **Surviving random hardware failures** (nodes can spontaneously degrade)

### 🏆 Competition Focus: Theme #3.1 — World Modeling
ClusterOps is designed to push LLMs beyond shallow instruction following. It requires the agent to build a **persistent internal state** of node temperatures and job durations. The agent must update its beliefs about node health based on random failures—a core requirement for advanced **World Modeling**.

---

## 🎯 Why This Environment Matters

| Property | ClusterOps | Typical SRE Gyms |
|:---|:---|:---|
| **State Type** | Pure numerical arrays | Text logs / YAML |
| **Problem Domain** | Real-time control systems | Bug diagnosis |
| **Inference Speed** | Milliseconds per episode | Seconds per episode |
| **Reward Signal** | Dense, multi-component, continuous | Sparse (fixed/failed) |
| **RL Trainability** | Excellent (fast rollouts) | Poor (slow rollouts) |

---

## 🕹️ Action Space

The agent submits actions as structured JSON to the `/step` endpoint.

| Action | Description | Schema |
|:---|:---|:---|
| `allocate` | Assign a queued job to an idle GPU node | `{"action_type": "allocate", "job_id": "job_1", "node_id": 3}` |
| `evict` | Emergency-stop a running job to free a node | `{"action_type": "evict", "node_id": 5}` |
| `cooldown` | Force-cool an idle node (aggressive cooling for 1 step) | `{"action_type": "cooldown", "node_id": 2}` |
| `wait` | Do nothing this step | `{"action_type": "wait"}` |

---

## 👁️ Observation Space

Each response from `/step` or `/state` returns the full cluster state:

| Field | Type | Description |
|:---|:---|:---|
| `gpu_nodes` | `List[Dict]` | Each node: `id`, `status`, `temperature`, `job_id`, `job_type`, `job_duration_remaining` |
| `job_queue` | `List[Dict]` | Each job: `id`, `type`, `duration`, `wait_time` |
| `thermal_warnings` | `int` | Nodes exceeding 85% of thermal limit |
| `meltdowns` | `int` | Cumulative thermal crash events |
| `completed_jobs` | `int` | Successfully finished jobs |
| `feedback` | `str` | Result of the last action |

Node statuses: `idle`, `busy`, `cooldown`, `failed`

---

## ⚡ Job Types & Thermal Physics

| Job Type | Heat Rate | Reward on Complete | Queue Penalty/Step |
|:---|:---|:---|:---|
| `vip_training` | +15°C/step | +40 | -2.0 |
| `inference` | +8°C/step | +15 | -0.5 |
| `batch` | +5°C/step | +8 | -0.2 |

**Cooling rates:**
- Idle node: -8°C/step (medium)
- Forced cooldown: -20°C/step
- Failed node recovery: -12°C/step

---

## 🎚️ Difficulty Levels

| Difficulty | Nodes | Max Steps | Thermal Limit | Hardware Failures | Description |
|:---|:---|:---|:---|:---|:---|
| `easy` | 6 | 50 | 100°C | None | Small cluster, learn the basics |
| `medium` | 10 | 100 | 95°C | 2% per idle node/step | Full cluster with occasional degradation |
| `hard` | 16 | 150 | 90°C | 5% per idle node/step | Massive cluster, aggressive load, tight thermals |

---

## 🏆 Reward Function (Dense, Multi-Component)

| Event | Reward |
|:---|:---|
| Complete a VIP training job | **+40** |
| Complete an inference job | **+15** |
| Complete a batch job | **+8** |
| Per-step VIP job waiting in queue | **-2.0** |
| Per-step inference job waiting | **-0.5** |
| Per-step batch job waiting | **-0.2** |
| Thermal meltdown (node crash) | **-50** |
| Evict a running job | **-10** |
| Invalid/unknown action | **-5** |
| Random hardware failure | **-15** |

The reward signal is **dense** (every step), **multi-component** (jobs, queue, thermals), and **impossible to game** (you can't score high without actually completing jobs while preventing meltdowns).

---

## 📊 Composable Rubric (Evaluation)

Judges look for **composable rubrics > monolithic scoring**. ClusterOps uses a multi-dimensional grading system exposed via `/grader/rubric`:

| Dimension | Weight | Target | Description |
|:---|:---|:---|:---|
| **Thermal Safety** | 35% | 0 Meltdowns | Penalizes every crash (-0.2/event) |
| **Throughput** | 30% | High Completion | completions / expected_ceiling |
| **Efficiency** | 20% | Zero Evictions | ratio of finished jobs vs total handled |
| **SLA Compliance** | 15% | Zero VIP Failures | penalizes lost/starved high-priority jobs |

**Final Grade:** `Σ (Sub-score × Weight)` clamped to `[0.0, 1.0]`.

---

## 🎓 Adaptive Curriculum Learning

To enable efficient LLM training, we implement an automatic difficulty progression system accessible via `/curriculum`:

1. **Easy (Nodes: 6)**: Training focus on basic allocation mechanics. (Threshold: Score ≥ 0.65)
2. **Medium (Nodes: 10)**: Introduces random hardware failures and tighter thermals. (Threshold: Score ≥ 0.70)
3. **Hard (Nodes: 16)**: Adversarial load with extreme thermal constraints.

---

## 🏁 Baseline Performance

| Difficulty | Heuristic Baseline | LLM Agent (Llama-3.1-8B) |
|:---|:---|:---|
| Easy | ~120 total reward | TBD after training |
| Medium | ~80 total reward | TBD after training |
| Hard | ~30 total reward | TBD after training |

---

## 🔌 API Interface

| Method | Endpoint | Description |
|:---|:---|:---|
| `GET` | `/` | Environment health check |
| `GET` | `/health` | Health check with version |
| `POST` | `/reset?difficulty=...` | Start new episode (easy/medium/hard) |
| `POST` | `/step` | Submit action, receive observation + reward |
| `GET` | `/state` | Current full environment snapshot |
| `GET` | `/schema` | Action/observation JSON schemas |

**Agent interaction loop:**
```
POST /reset?difficulty=medium  →  initial observation
POST /step { action }          →  step result (observation + reward + done)
POST /step { action }          →  ...
```

---

## 🚀 Quick Start

### Local Setup
```bash
git clone https://github.com/YOUR_REPO/clusterops-gpu-gym
cd clusterops-gpu-gym/clusterops

# Install dependencies
pip install -r server/requirements.txt
# OR with uv:
uv sync

# Start the server
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### Run the Heuristic Baseline (no API key needed)
```bash
python baseline.py medium
```

### Run the LLM Agent
```bash
export API_BASE_URL="https://api-inference.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
export HF_TOKEN="hf_your_token"
python inference.py medium
```

### Docker
```bash
docker build -t clusterops:latest -f server/Dockerfile .
docker run -p 8000:8000 clusterops:latest
```

---

## 🏗️ Project Structure

```
clusterops/
├── openenv.yaml                    # OpenEnv manifest
├── models.py                       # Pydantic Action/Observation schemas
├── client.py                       # OpenEnv WebSocket client
├── baseline.py                     # Heuristic baseline agent (no LLM)
├── inference.py                    # LLM inference agent (OpenAI-compatible)
├── __init__.py                     # Package exports
├── pyproject.toml                  # Dependencies
├── README.md                       # This file
└── server/
    ├── app.py                      # FastAPI application
    ├── clusterops_environment.py   # Core physics simulation engine
    ├── Dockerfile                  # Container deployment
    └── requirements.txt            # Server dependencies
```

---

## 📈 Training Results

![Reward Curve](./reward_curve.png)
![Loss Curve](./loss_curve.png)

*Initial training run complete. Models show clear improvement in job completion rate while suppressing occurrences of thermal meltdowns.*

---

## 🧰 Technology Stack

| Layer | Technology |
|:---|:---|
| Language | Python 3.11 |
| Framework | FastAPI + Uvicorn |
| RL Interface | OpenEnv |
| Training | HuggingFace TRL (GRPO) + Unsloth |
| Data Models | Pydantic v2 |
| Containerization | Docker |
| Deployment | HuggingFace Spaces |

---

## 📄 License

MIT License

# ClusterOps: Teaching an LLM to Schedule a GPU Datacenter

A weekend's work for the Meta OpenEnv India Hackathon 2026, and a sanity check on a question that's been bugging us for a while: what does a language model actually learn when you give it a thermometer and a job queue, and nothing else?

No physics priors. No scheduling textbook. No spatial layout map. Just a `/step` endpoint, a tensor of node temperatures, and a reward signal.

This is a writeup of how it went.

## The setup

We built a 10-node GPU rack as an OpenEnv environment. The agent sees temperatures, queue contents, and node statuses. It picks one of four actions every step:

- `allocate` — place a queued job on a chosen node
- `evict` — kill a running job to cool a node down
- `cooldown` — proactively force-cool an idle node
- `wait` — do nothing for a step

Behind the API there's a real thermal model. Every running job adds heat per step (a `vip_training` job adds about +15°C, a `batch` job adds about +6°C). Every idle node sheds a little. Jobs have deadlines. Nodes melt at 100°C. There's spatial heat bleed in the harder scenarios, scheduled outages in another, and an adversarial traffic spike in the last.

The whole environment lives in `clusterops/environment.py`, exposed through a FastAPI server with a live dashboard so you can watch it run.

## SFT-First Strategy: Building the Foundation

While our environment supports on-policy RL via Group Relative Policy Optimization (GRPO), we made a strategic decision to lead with Supervised Fine-Tuning (SFT).

Cold-starting an LLM on a complex control task often leads to early reward collapse if the model hasn't yet grasped the action grammar or the observation space. By utilizing our expert heuristic (`agents/smart_agent.py`) to generate a high-quality demonstration dataset, we were able to:

1. **Establish a stable baseline** using TRL and Unsloth.
2. **Teach the model the action grammar** (converting observations to valid JSON actions) with 100% reliability.
3. **Inherit expert priors** like proactive cooling and SLA prioritization.

This "SFT-first" approach provides the perfect weights for future on-policy RL (PPO/GRPO) to explore further and surpass the teacher, as the model already understands the rules of the world.

## Refining the World Model: Rubric Engineering

The expert agent itself was the most surprising part of the project. It went through three rewrites because it kept finding ways to game the rubric.

**Exploit 1: do nothing.** The early rubric weighted thermal safety heavily. The agent figured out that `wait` forever scores ~1.0 on thermals. We added a saturation rule: if the queue stays full too long, the episode terminates immediately.

**Exploit 2: allocate-evict thrashing.** When the thermal timer was tied to "time since allocation", the agent learned to allocate a job, run it for one step, evict, allocate it again. Perfect thermals, zero work done. We added a 3× multiplier on eviction cost.

**Exploit 3: outage gaming.** In `04_maintenance`, nodes scheduled for downtime didn't count as "failed" if they were never assigned a job, so the agent left half the cluster empty during outage windows. The fix was to tie SLA compliance to a true completion ratio (`completed / (completed + failed)`), so being idle isn't rewarded.

Watching a hand-coded greedy heuristic discover three reward exploits in a row, before we ever got an LLM near it, was a useful preview of what RL was about to do to us.

## What the LLM picked up

After 80 SFT steps on a few thousand expert `(observation, action)` pairs, the trained model:

- Stops emitting malformed JSON. Pre-training it was about 30% invalid; post-training it's effectively 0% on the eval seeds.
- Beats the always-`wait` baseline on every one of five paired same-seed episodes in `01_baseline @ easy`.
- Inherits the expert's `cooldown` behaviour. It pre-cools idle nodes when the queue is light, even though nothing in the prompt explicitly tells it to.

The numbers from the most recent eval run:

| Policy | Mean reward (5 paired seeds) | Lift over naive |
| --- | --- | --- |
| Naive (always `wait`) | **−546.0** | — |
| Trained LLM (BC + validation guardrail) | **−28.0** | **+518.0** |
| Expert teacher (oracle) | **+226.4** | +772.4 |

That's a +518 reward lift on the same fixed seeds, in 80 SFT steps. The trained model picks up about two-thirds of the gap between the naive baseline and the expert.

Three of the five trained episodes finish positive (+35.9, +98.6, +21.3). The other two go negative (−120.1, −175.8), both because of one early misallocation that cascades through the rest of the episode. More on that in the "what's broken" section.

The most interesting bit of behaviour the model copied is `cooldown`. It only pays off two or three steps later, when a VIP job lands on a node the model proactively cooled, and there's nothing in any single observation that directly rewards pre-cooling. It's the kind of multi-step pattern BC isn't supposed to be great at, and it transferred anyway.

## How we evaluate

Evaluating a 1B model honestly is harder than training one, especially when the environment is noisy. Two things mattered:

1. **Paired seeds.** All three policies (naive, expert, trained) run on the same five fixed seeds. Same job arrivals, same initial temperatures. Anything else gives you noise that's bigger than the signal you're trying to measure.
2. **Action-validation guardrail.** Malformed model outputs (e.g. `allocate` to a busy node, or a string where a node id should be) get downgraded to `wait` with 0 reward, instead of racking up -5 invalid-action penalties for the rest of the episode. The model still picks every meaningful action; the guardrail is the same kind of safety net you'd put in front of any LLM-driven controller in production.

### Training Progress

Two high-DPI plots summarize the convergence of the model and its performance compared to the teacher:

| **Training Loss convergence** | **Episode Return (Paired seeds)** |
|:---:|:---:|
| ![Loss Curve](assets/loss_curve.png) | ![Reward Curve](assets/reward_curve.png) |

- **Loss curve:** Shows the per-step SFT training loss across 80 steps. The smooth line (moving average) indicates healthy convergence as the model learns to replicate expert JSON actions.
- **Reward curve:** A fair, apples-to-apples comparison on 5 fixed seeds. The trained LLM significantly outperforms the "naive" baseline, capturing a major portion of the reward delta compared to the expert teacher.


## The Path Ahead: Future Horizons

The results from our initial sprint are incredibly promising, and they point the way toward several exciting next steps:

- **Policy Refinement:** Our model captured over two-thirds of the gap between a naive baseline and an expert teacher in just 80 steps. The natural next phase is applying the included GRPO/PPO wrapper (`clusterops/gym_env.py`) to close the remaining gap and push the model past the expert.
- **Handling Adversarial Spikes:** Act 5 (`05_adversarial`) remains a high-difficulty frontier. Future work will involve exposing more "queue pressure" features to the model to help it predict and mitigate DDoS-level traffic spikes even more effectively.
- **Interactive Training:** We plan to add a LoRA-adapter loading mechanism directly to the OpenEnv dashboard, allowing users to watch the model "think" in real-time as it manages their cluster.

If you want to poke at any of this:

- **Live Dashboard:** [neer-biswas/thermal-gpu-balancer](https://huggingface.co/spaces/neer-biswas/thermal-gpu-balancer)
- **GitHub Repo:** [Sushmit-Biswas/thermal-gpu-balancer](https://github.com/Sushmit-Biswas/thermal-gpu-balancer)
- **README (with Mermaid diagrams):** [README.md](https://github.com/Sushmit-Biswas/thermal-gpu-balancer/blob/main/README.md)
- **Training Notebook:** [ClusterOps_GRPO_Training.ipynb](https://huggingface.co/spaces/neer-biswas/thermal-gpu-balancer/blob/main/training/ClusterOps_GRPO_Training.ipynb)

---

*Team SSP Warriors — Meta OpenEnv India Hackathon, 2026.*

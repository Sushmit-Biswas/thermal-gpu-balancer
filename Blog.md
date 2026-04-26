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

## Why we didn't go straight to GRPO

Initial plan was Group Relative Policy Optimization. We had TRL and Unsloth set up, the rubric was ready, and GRPO is genuinely well-suited to sparse rewards (you don't know the cluster melted until it's already melting).

But cold-starting a 1B-class LLM with on-policy RL inside a Colab session is a fight. Early rollouts emit malformed JSON, get -5 invalid-action penalties on every step, the policy collapses to "always `wait`", and you spend four hours of GPU time watching a flat reward curve.

So we stepped back. The pragmatic answer was:

1. Write a heuristic expert that already plays the environment well (`agents/smart_agent.py`).
2. Roll it out, capture the trajectories, and behavioural-clone them into the LLM with `SFTTrainer`.
3. Save GRPO/PPO for after the model knows the action grammar.

That's the pipeline shipped in `training/ClusterOps_GRPO_Training.ipynb`. The Gymnasium wrapper at `clusterops/gym_env.py` is sitting there so the next iteration can pick up where this one ends, on the same rubric, the same scenarios, and the same observation space.

## The reward-hacking saga

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

So the trained policy moves the agent from a deeply-negative failure regime (≈ −546) to roughly break-even (−28). That's a +518 reward lift on the same fixed seeds, purely from 80 steps of behavioural cloning. In raw terms, the trained LLM captures roughly two-thirds of the expert's total lift over naive — which is about what you'd expect from a BC student on a stochastic environment with a small data budget.

Three of the five trained episodes finish in clearly positive territory (+35.9, +98.6, +21.3); the other two dip negative (−120.1, −175.8) when the model emits a slightly off action on a heavier seed. The variance is real and the next paragraph is about what causes it.

The cooldown behaviour is the most interesting transfer. It only pays off two or three steps later, when a VIP job arrives and lands on a pre-cooled node, and there's nothing in a single observation that explicitly rewards pre-cooling. The model picked it up purely by mimicking trajectories — which is what behavioural cloning is supposed to do but rarely feels this clean on a stochastic environment.

## How we evaluate

Honest evaluation of a 1B model is harder than training one, especially on a stochastic environment. Two changes mattered:

1. **Paired seeds.** All three policies (naive, expert, trained) run on the same five fixed seeds. Same job arrivals, same initial temperatures. Anything else gives you noise that's bigger than the signal you're trying to measure.
2. **Action-validation guardrail.** Malformed model outputs (e.g. `allocate` to a busy node, or a string where a node id should be) are downgraded to `wait` with 0 reward, instead of accumulating -5 invalid-action penalties for the rest of the episode. The model still chooses every meaningful action; the guardrail is the same kind of safety net you'd put in front of any LLM-driven controller in production.

Loss and reward curves end up under `assets/loss_curve.png` and `assets/reward_curve.png` after the notebook finishes.

## What's broken, what's next

Plenty.

- The trained model averages −28 versus the expert's +226 — about 254 points of headroom still on the table on top of the +518 lift it already captured. This is early-stage BC; the natural next step is on-policy RL (GRPO/PPO via the included `clusterops/gym_env.py`) on the same rubric to close the rest of that gap and push the model past the teacher.
- The two negative trained episodes (−120, −176) are a variance story, not a competence one — both come from a single early misallocation that cascades. A small reward-shaped fine-tune would likely flatten that tail.
- We never got `05_adversarial` cleanly. The adversarial DDoS spike still occasionally murders even the expert; the agent really needs a notion of "incoming pressure" that the current observation doesn't expose well.
- The dashboard is read-mostly. A "load my LoRA adapter" button would make it a much better demo.

If you want to poke at any of this:

- **Live demo:** [neer-biswas/thermal-gpu-balancer](https://huggingface.co/spaces/neer-biswas/thermal-gpu-balancer)
- **Repo:** [Sushmit-Biswas/thermal-gpu-balancer](https://github.com/Sushmit-Biswas/thermal-gpu-balancer)
- **Notebook:** `training/ClusterOps_GRPO_Training.ipynb`

---

*Team SSP Warriors — Meta OpenEnv India Hackathon, 2026.*

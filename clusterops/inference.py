#!/usr/bin/env python3
"""
ClusterOps LLM Inference Agent.

Connects an LLM (via OpenAI-compatible API) to the ClusterOps environment.
The LLM receives the full cluster state as a structured prompt and must
output a valid JSON action.

Outputs OpenEnv-compliant logs: [START], [STEP], [END].

Usage:
    # Set environment variables
    export API_BASE_URL="https://api-inference.huggingface.co/v1"
    export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
    export HF_TOKEN="hf_your_token"
    export ENVIRONMENT_BASE_URL="http://localhost:8000"

    python inference.py
"""

import os
import sys
import json
import requests
from datetime import datetime

# ─── Configuration ──────────────────────────────────────────────────────────────
ENV_URL = os.getenv("ENVIRONMENT_BASE_URL", "http://localhost:8000")
API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", HF_TOKEN)
DIFFICULTY = os.getenv("DIFFICULTY", "medium")

SYSTEM_PROMPT = """You are an expert AI Data Center Operator managing a GPU compute cluster.

Your job: Allocate incoming jobs to GPU nodes while preventing THERMAL MELTDOWNS.

CRITICAL RULES:
1. Nodes HEAT UP when running jobs. Different job types generate different heat:
   - vip_training: +15°C/step (highest priority, highest reward, most dangerous)
   - inference: +8°C/step (moderate)
   - batch: +5°C/step (low priority, safest)
2. Nodes COOL DOWN when idle (-8°C/step) or in forced cooldown (-20°C/step).
3. If a node reaches the THERMAL LIMIT, it MELTS DOWN: the job is destroyed and you lose 50 points.
4. Every step a job waits in queue, you lose points (VIP jobs penalize heavily).
5. You GAIN +8 to +40 points for completing a job successfully.

YOUR ACTIONS (respond with exactly ONE JSON object):
- {"action_type": "allocate", "job_id": "<id>", "node_id": <n>} - Assign a queued job to an idle node.
- {"action_type": "evict", "node_id": <n>} - Emergency-stop a running job to prevent meltdown.
- {"action_type": "cooldown", "node_id": <n>} - Force-cool an idle node.  
- {"action_type": "wait"} - Skip this step.

STRATEGY TIPS:
- Assign VIP jobs to the COOLEST nodes.
- If a node is above 75°C and running a vip_training job, consider evicting it.
- Don't let the queue grow — every waiting job costs you points each step.
- Spread jobs across many nodes rather than packing them.

Respond with ONLY a valid JSON action object. No explanation."""


def call_llm(prompt: str) -> str:
    """Call the LLM via OpenAI-compatible API."""
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 150,
        "temperature": 0.3,
    }
    try:
        resp = requests.post(f"{API_BASE_URL}/chat/completions", headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()
        return content
    except Exception as e:
        print(f"LLM API error: {e}", file=sys.stderr)
        return '{"action_type": "wait"}'


def parse_action(llm_response: str) -> dict:
    """Parse the LLM's JSON response into an action dict."""
    # Try to extract JSON from the response
    try:
        # Handle markdown code blocks
        if "```" in llm_response:
            start = llm_response.find("{")
            end = llm_response.rfind("}") + 1
            llm_response = llm_response[start:end]
        return json.loads(llm_response)
    except json.JSONDecodeError:
        # Fallback: extract first JSON-like object
        try:
            start = llm_response.index("{")
            end = llm_response.rindex("}") + 1
            return json.loads(llm_response[start:end])
        except (ValueError, json.JSONDecodeError):
            return {"action_type": "wait"}


def format_state_prompt(obs_data: dict) -> str:
    """Convert the raw observation into a readable prompt for the LLM."""
    nodes = obs_data.get("gpu_nodes", [])
    queue = obs_data.get("job_queue", [])
    meta = obs_data.get("metadata", {})
    
    lines = []
    lines.append(f"=== STEP {meta.get('step', '?')}/{meta.get('max_steps', '?')} ===")
    lines.append(f"Completed: {obs_data.get('completed_jobs', 0)} | Meltdowns: {obs_data.get('meltdowns', 0)} | Thermal Warnings: {obs_data.get('thermal_warnings', 0)}")
    lines.append(f"Last feedback: {obs_data.get('feedback', '')}")
    lines.append("")
    lines.append("GPU NODES:")
    for n in nodes:
        status_icon = {"idle": "IDLE", "busy": "BUSY", "cooldown": "COOL", "failed": "FAIL"}.get(n["status"], n["status"])
        job_info = f" | Job: {n['job_id']} ({n.get('job_type', '?')}, {n['job_duration_remaining']} steps left)" if n["status"] == "busy" else ""
        lines.append(f"  Node {n['id']:2d}: {status_icon:4s} | Temp: {n['temperature']:5.1f}°C{job_info}")
    
    lines.append("")
    lines.append(f"JOB QUEUE ({len(queue)} jobs):")
    if queue:
        for j in queue[:10]:  # Show first 10
            lines.append(f"  {j['id']}: type={j['type']}, duration={j['duration']}, waiting={j['wait_time']} steps")
        if len(queue) > 10:
            lines.append(f"  ... and {len(queue) - 10} more jobs")
    else:
        lines.append("  (empty)")
    
    return "\n".join(lines)


def run_inference(difficulty="medium"):
    """Run the LLM agent for one full episode."""
    print(f"[START] ClusterOps LLM Agent | Model: {MODEL_NAME} | Difficulty: {difficulty}")

    # Reset environment
    resp = requests.post(f"{ENV_URL}/reset", params={"difficulty": difficulty})
    obs = resp.json()
    obs_data = obs.get("observation", obs)
    total_reward = 0.0
    step_num = 0

    while True:
        done = obs.get("done", obs_data.get("done", False))
        if done:
            break

        # Format state for LLM
        prompt = format_state_prompt(obs_data)

        # Get LLM action
        llm_response = call_llm(prompt)
        action = parse_action(llm_response)

        # Execute step
        resp = requests.post(f"{ENV_URL}/step", json=action)
        obs = resp.json()
        obs_data = obs.get("observation", obs)
        step_reward = obs.get("reward", obs_data.get("reward", 0.0))
        total_reward += step_reward
        step_num += 1

        print(f"[STEP] {step_num} | Action: {action.get('action_type')} | "
              f"Reward: {step_reward:+.1f} | Total: {total_reward:.1f} | "
              f"Feedback: {obs_data.get('feedback', '')[:60]}")

    print(f"[END] Episode complete | Steps: {step_num} | Total Reward: {total_reward:.1f} | "
          f"Completed: {obs_data.get('completed_jobs', 0)} | Meltdowns: {obs_data.get('meltdowns', 0)}")

    return total_reward


def main():
    difficulty = sys.argv[1] if len(sys.argv) > 1 else DIFFICULTY
    run_inference(difficulty)


if __name__ == "__main__":
    main()

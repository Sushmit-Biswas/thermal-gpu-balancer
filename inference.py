#!/usr/bin/env python3
"""
ClusterOps LLM Inference Agent.

Runs an LLM (via OpenAI-compatible API) to control the ClusterOps environment.
Works with:
  - HuggingFace Inference Endpoints (export HF_TOKEN=...)
  - Local vLLM/Ollama servers
  - OpenAI GPT-4 / GPT-4o

Usage:
    # HuggingFace (serverless):
    export API_BASE_URL="https://api-inference.huggingface.co/v1"
    export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
    export HF_TOKEN="hf_your_token"
    python inference.py --difficulty medium --episodes 3

    # Local vLLM:
    export API_BASE_URL="http://localhost:8080/v1"
    export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
    python inference.py --difficulty easy
"""

import os
import sys
import json
import argparse
import requests

# ─── Configuration ────────────────────────────────────────────────────────────
ENV_URL = os.getenv("ENVIRONMENT_BASE_URL", "http://localhost:8000")
API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN", "")
MAX_NEW_TOKENS = 120

SYSTEM_PROMPT = """\
You are an advanced GPU Cluster SRE. Your goal is to maximize throughput while preventing meltdowns.

STRATEGY:
1. PREDICT: Check if a node will hit the thermal_limit in the next 2-3 steps based on current temperature and job heat_rate.
2. PRIORITIZE: Handle VIP jobs first, but never at the cost of a meltdown.
3. COOLING: Use 'cooldown' proactively on nodes > 80°C if no urgent jobs are pending.

RESPONSE FORMAT:
You must respond with a JSON object containing 'thought' and 'action'.
The 'thought' should be a brief analysis of the current cluster state.

Example:
{
  "thought": "Node 0 is approaching 90°C with a training job. I will evict to prevent a -50 penalty.",
  "action": {"action_type": "evict", "node_id": 0}
}
"""


# ─── Environment Helpers ──────────────────────────────────────────────────────

def env_reset(difficulty: str = "medium") -> dict:
    resp = requests.post(f"{ENV_URL}/reset", json={"difficulty": difficulty}, timeout=10)
    resp.raise_for_status()
    return resp.json()


def env_step(action: dict) -> dict:
    resp = requests.post(f"{ENV_URL}/step", json=action, timeout=10)
    resp.raise_for_status()
    return resp.json()


def env_grader() -> dict:
    resp = requests.post(f"{ENV_URL}/grader", timeout=10)
    resp.raise_for_status()
    return resp.json()


# ─── Prompt Formatting ────────────────────────────────────────────────────────

def format_observation(obs: dict, metadata: dict) -> str:
    """Format the cluster state as a compact text prompt for the LLM."""
    nodes = obs.get("gpu_nodes", [])
    queue = obs.get("job_queue", [])
    step = metadata.get("step", "?")
    max_steps = metadata.get("max_steps", "?")
    difficulty = metadata.get("difficulty", "?")
    meltdowns = obs.get("meltdowns", 0)
    completed = obs.get("completed_jobs", 0)
    warnings = obs.get("thermal_warnings", 0)

    lines = [
        f"[STEP {step}/{max_steps}] Difficulty={difficulty} | "
        f"Completed={completed} Meltdowns={meltdowns} ThermalWarnings={warnings}",
        "THERMAL LIMIT: 100°C | WARNING THRESHOLD: 85°C",
        "",
        "GPU NODES:",
    ]
    for n in nodes:
        status = n["status"].upper()
        temp = n["temperature"]
        job_info = ""
        if n["status"] == "busy":
            job_info = f" | running={n.get('job_id','?')} ({n.get('job_type','?')}, {n.get('job_duration_remaining',0)} steps left)"
        lines.append(f"  Node {n['id']:2d}: {status:<9} | {temp:5.1f}°C{job_info}")

    lines.append("")
    if queue:
        lines.append(f"JOB QUEUE ({len(queue)} jobs):")
        for j in queue[:10]:  # Show first 10
            lines.append(
                f"  {j['id']}: type={j['type']:<14} duration={j['duration']} wait={j['wait_time']} steps"
            )
        if len(queue) > 10:
            lines.append(f"  ... and {len(queue) - 10} more jobs")
    else:
        lines.append("JOB QUEUE: empty")

    last_feedback = obs.get("feedback", "")
    if last_feedback:
        lines.append(f"\nLast action result: {last_feedback}")

    return "\n".join(lines)


# ─── LLM Action Generation ────────────────────────────────────────────────────

def call_llm(prompt: str) -> str:
    """Call the LLM via OpenAI-compatible API and return the raw text response."""
    headers = {"Content-Type": "application/json"}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": MAX_NEW_TOKENS,
        "temperature": 0.3,  # Lower temp = more deterministic actions
        "stream": False,
    }

    resp = requests.post(
        f"{API_BASE_URL}/chat/completions",
        headers=headers,
        json=payload,
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


def parse_action(text: str) -> dict:
    """Extract a valid JSON action from LLM output. Falls back to 'wait'."""
    # Strip markdown code blocks if present
    if "```" in text:
        for line in text.split("\n"):
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                text = line
                break

    # Try to extract a JSON object
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        data = json.loads(text[start:end])
        
        # Handle CoT format: {"thought": "...", "action": {...}}
        if "action" in data and isinstance(data["action"], dict):
            if "thought" in data:
                print(f"🧠 REASONING: {data['thought']}")
            return data["action"]
            
        if "action_type" in data:
            return data
    except (ValueError, json.JSONDecodeError):
        pass

    print(f"  [WARN] Could not parse LLM output as JSON, defaulting to wait. Output: {text!r}")
    return {"action_type": "wait"}


# ─── Episode Runner ───────────────────────────────────────────────────────────

def run_episode(difficulty: str = "medium", verbose: bool = True) -> dict:
    """Run a single episode with the LLM agent. Returns grade info."""
    if verbose:
        print(f"\n{'='*65}")
        print(f"  ClusterOps LLM Agent  |  Difficulty: {difficulty}")
        print(f"{'='*65}")

    data = env_reset(difficulty)
    obs = data.get("observation", {})
    metadata = data.get("metadata", {})
    total_reward = 0.0
    step_num = 0

    while not data.get("done", False):
        prompt = format_observation(obs, metadata)

        try:
            llm_text = call_llm(prompt)
        except Exception as e:
            print(f"  [LLM ERROR] {e} — defaulting to wait")
            llm_text = '{"action_type": "wait"}'

        action = parse_action(llm_text)

        if verbose and step_num % 5 == 0:
            print(
                f"  Step {step_num:3d} | Temp range [{min(n['temperature'] for n in obs.get('gpu_nodes',[{'temperature':0}])):.0f}"
                f"–{max(n['temperature'] for n in obs.get('gpu_nodes',[{'temperature':0}])):.0f}°C]"
                f" | Queue={len(obs.get('job_queue', []))} | Action: {action}"
            )

        data = env_step(action)
        obs = data.get("observation", {})
        metadata = data.get("metadata", {})
        total_reward += data.get("reward", 0.0)
        step_num += 1

    grade = env_grader()

    if verbose:
        print(f"\n--- Episode Complete ---")
        print(f"  Steps:          {step_num}")
        print(f"  Total Reward:   {total_reward:.1f}")
        print(f"  Jobs Completed: {obs.get('completed_jobs', 0)}")
        print(f"  Meltdowns:      {obs.get('meltdowns', 0)}")
        print(f"  Queue Left:     {len(obs.get('job_queue', []))}")
        print(f"  Grade [0-1]:    {grade.get('score', 0.0):.4f}")
        print(f"{'='*65}\n")

    return {"total_reward": total_reward, "grade": grade, "steps": step_num}


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ClusterOps LLM Inference Agent")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard", "expert"], default="medium")
    parser.add_argument("--episodes", type=int, default=1)
    args = parser.parse_args()

    # Sanity check: is the server up?
    try:
        health = requests.get(f"{ENV_URL}/health", timeout=5)
        health.raise_for_status()
        print(f"✅ Server healthy at {ENV_URL}")
    except Exception as e:
        print(f"❌ Cannot reach environment server at {ENV_URL}: {e}")
        print("   Start the server with: uvicorn server.app:app --port 8000")
        sys.exit(1)

    if not HF_TOKEN and "huggingface" in API_BASE_URL.lower():
        print("⚠️  HF_TOKEN not set. Requests to HuggingFace may be rejected.")

    all_rewards = []
    all_grades = []

    for i in range(args.episodes):
        print(f"\n[Episode {i+1}/{args.episodes}]")
        result = run_episode(difficulty=args.difficulty)
        all_rewards.append(result["total_reward"])
        all_grades.append(result["grade"].get("score", 0.0))

    if args.episodes > 1:
        print(f"\n{'='*65}")
        print(f"SUMMARY over {args.episodes} episodes | difficulty={args.difficulty}")
        print(f"  Avg Reward:  {sum(all_rewards)/len(all_rewards):.1f}")
        print(f"  Best Reward: {max(all_rewards):.1f}")
        print(f"  Avg Grade:   {sum(all_grades)/len(all_grades):.4f}")
        print(f"  Best Grade:  {max(all_grades):.4f}")
        print(f"{'='*65}")


if __name__ == "__main__":
    main()

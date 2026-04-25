"""
ClusterOps Groq LLM Benchmark.

Runs the Groq-hosted LLM against a specific ClusterOps scenario
and generates a markdown report with step-by-step reasoning.

Configuration:
    GROQ_API_KEY          — Your Groq API key (required)
    ENVIRONMENT_BASE_URL  — ClusterOps server URL (default: http://localhost:8000)

Usage:
    export GROQ_API_KEY=your_key
    python run_groq_test.py
"""

import os
import subprocess
import time
import sys
import requests
import json
from datetime import datetime

# Configuration
ENV_URL = os.getenv("ENVIRONMENT_BASE_URL", "http://localhost:8000")
API_BASE_URL = "https://api.groq.com/openai/v1"
MODEL_NAME = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

SYSTEM_PROMPT = """\
You are an advanced GPU Cluster SRE. Prevent meltdowns (temp >= 100°C) and maximize throughput.

SCENARIO AWARENESS:
- 02_spatial_bleed: Heat radiates to neighbors (+3°C) if a node hits 85°C. Leave idle nodes between heavy jobs!
- 03_heterogeneous: Even IDs are H100 (2x fast, 2x hot). Odd IDs are T4 (cool, slow).
- 04_maintenance: Watch for maintenance warnings and drain nodes before deadlines.

STRICT ACTION RULES:
- "allocate": Assign a job to an IDLE node. Needs: {"action_type": "allocate", "job_id": "...", "node_id": 0}
- "evict": Emergency stop on a BUSY node only if temp > 95°C. Needs: {"action_type": "evict", "node_id": 0}
- "cooldown": Proactive cooling on an IDLE node. Needs: {"action_type": "cooldown", "node_id": 0}
- "wait": Do nothing. Needs: {"action_type": "wait"}

RESPONSE FORMAT (JSON ONLY):
{
  "thought": "Reasoning based on thermal headroom and queue urgency.",
  "action": {"action_type": "allocate", "job_id": "job_1", "node_id": 0}
}
"""


def format_observation(obs):
    nodes = obs.get("gpu_nodes", [])
    queue = obs.get("job_queue", [])
    node_str = "\n".join([
        f"Node {n['id']}: {n['status']} | {n['temperature']}°C | Job: {n['job_type']}"
        for n in nodes
    ])
    queue_str = "\n".join([
        f"- {j['id']} ({j['type']}) dur:{j['duration']} dead:{j['deadline']}"
        for j in queue[:5]
    ])
    metadata = obs.get("metadata", {})
    return (
        f"SCENARIO: {metadata.get('scenario', 'unknown')}\n"
        f"NODES:\n{node_str}\n\n"
        f"QUEUE:\n{queue_str if queue_str else 'Empty'}"
    )


def get_llm_action(api_key, obs_text):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    data = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": obs_text},
        ],
        "temperature": 0.0,
        "max_tokens": 500,
        "response_format": {"type": "json_object"},
    }
    for attempt in range(3):
        try:
            resp = requests.post(
                f"{API_BASE_URL}/chat/completions",
                headers=headers,
                json=data,
                timeout=30,
            )
            if resp.status_code == 429:
                print("! Rate limited. Waiting 30s...", flush=True)
                time.sleep(30)
                continue
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            return json.loads(content)
        except Exception as e:
            if attempt == 2:
                raise e
            time.sleep(5)


def run_groq_test():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("X Error: GROQ_API_KEY not set.")
        return

    difficulty = os.getenv("DIFFICULTY", "medium")
    scenario = os.getenv("SCENARIO", "02_spatial_bleed")

    report_lines = [
        f"# ClusterOps Scenario Benchmark: {scenario}\n",
        f"- **Model**: {MODEL_NAME}",
        f"- **Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- **Difficulty**: {difficulty}",
        f"- **Scenario**: {scenario}\n",
        "## Step-by-Step Execution Log\n",
        "| Step | Reasoning | Action | Result |",
        "| :--- | :--- | :--- | :--- |",
    ]

    print("--- Starting ClusterOps Server ---", flush=True)
    server_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(5)

    try:
        print(f"--- Resetting environment to {difficulty}/{scenario} ---", flush=True)
        resp = requests.post(
            f"{ENV_URL}/reset",
            json={"difficulty": difficulty, "scenario": scenario},
            timeout=10,
        )
        obs = resp.json()["observation"]

        for i in range(1, 11):
            obs_text = format_observation({
                "gpu_nodes": obs["gpu_nodes"],
                "job_queue": obs["job_queue"],
                "metadata": {"scenario": scenario},
            })
            print(f"\n[Step {i}]", flush=True)

            res = get_llm_action(api_key, obs_text)
            thought = res.get("thought", "N/A")
            action = res.get("action", res)

            print(f"THOUGHT: {thought}", flush=True)
            print(f"ACTION: {json.dumps(action)}", flush=True)

            resp = requests.post(f"{ENV_URL}/step", json=action, timeout=10)
            data = resp.json()
            obs = data["observation"]
            reward = data["reward"]

            report_lines.append(
                f"| {i} | {thought} | `{json.dumps(action)}` | "
                f"Reward: {reward:.1f}, Meltdowns: {obs['meltdowns']} |"
            )
            print(f"RESULT: Reward={reward}, Meltdowns={obs['meltdowns']}", flush=True)

            time.sleep(15)

        resp = requests.post(f"{ENV_URL}/grader", timeout=10)
        grade = resp.json()

        report_lines.append("\n## Final Performance Metrics\n")
        report_lines.append("```json")
        report_lines.append(json.dumps(grade, indent=2))
        report_lines.append("```\n")

        report_name = f"scenario_benchmark_{scenario}.md"
        with open(report_name, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))
        print(f"\n✅ Report generated: {report_name}", flush=True)

    except Exception as e:
        print(f"X Test Failed: {e}", flush=True)
    finally:
        server_proc.terminate()


if __name__ == "__main__":
    run_groq_test()

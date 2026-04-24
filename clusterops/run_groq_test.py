import os
import subprocess
import time
import requests
import json
from datetime import datetime

# Configuration
ENV_URL = "http://localhost:8000"
API_BASE_URL = "https://api.groq.com/openai/v1"
MODEL_NAME = "llama-3.1-8b-instant"

SYSTEM_PROMPT = """\
You are an advanced GPU Cluster SRE. Prevent meltdowns (temp >= 100°C) and maximize throughput.

STRICT ACTION RULES:
- "allocate": Assign a job to an IDLE node. Needs: {"action_type": "allocate", "job_id": "...", "node_id": 0}
- "evict": Emergency stop on a BUSY node only if temp > 95°C. Needs: {"action_type": "evict", "node_id": 0}
- "cooldown": Proactive cooling on an IDLE node. Needs: {"action_type": "cooldown", "node_id": 0}
- "wait": Do nothing. Needs: {"action_type": "wait"}

STRATEGY:
- Do NOT evict nodes below 90°C.
- Prioritize jobs with early deadlines.
- Keep at least 1 node idle for emergency VIP jobs.

RESPONSE FORMAT (JSON ONLY):
{
  "thought": "Reasoning based on thermal headroom and queue urgency.",
  "action": {"action_type": "allocate", "job_id": "job_1", "node_id": 0}
}
"""

def format_observation(obs):
    nodes = obs.get("gpu_nodes", [])
    queue = obs.get("job_queue", [])
    node_str = " | ".join([f"N{n['id']}:{n['status'][0].upper()}{n['temperature']:.0f}°C" for n in nodes])
    queue_str = " | ".join([f"{j['id']}:{j['type'][:3]} (dur:{j['duration']}, wait:{j['wait_time']}, dead:{j['deadline']})" for j in queue[:5]])
    return f"THERMAL LIMIT: 100°C | WARNING: 85°C\nNODES: {node_str}\nQUEUE: {queue_str if queue_str else 'Empty'}"

def get_llm_action(api_key, obs_text):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    data = {
        "model": MODEL_NAME,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": obs_text}],
        "temperature": 0.0,
        "max_tokens": 200
    }
    resp = requests.post(f"{API_BASE_URL}/chat/completions", headers=headers, json=data, timeout=30)
    if resp.status_code == 429: raise Exception("rate_limit")
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    try:
        start = content.index("{")
        end = content.rindex("}") + 1
        return json.loads(content[start:end])
    except:
        return {"thought": f"Failed to parse: {content}", "action": {"action_type": "wait"}}

def run_groq_test():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("X Error: GROQ_API_KEY not set.")
        return

    report_lines = [
        f"# ClusterOps Groq Benchmark Report\n",
        f"- **Model**: {MODEL_NAME}",
        f"- **Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- **Difficulty**: Easy\n",
        "## Step-by-Step Execution Log\n",
        "| Step | Reasoning | Action | Result |",
        "| :--- | :--- | :--- | :--- |"
    ]

    print("--- Starting ClusterOps Server ---", flush=True)
    server_proc = subprocess.Popen([os.sys.executable, "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(5)
    
    try:
        print("--- Resetting environment ---", flush=True)
        resp = requests.post(f"{ENV_URL}/reset", json={"difficulty": "easy"}, timeout=10)
        obs = resp.json()["observation"]
        
        for i in range(1, 11): # 10 steps for a real report
            obs_text = format_observation(obs)
            print(f"\n[Step {i}]", flush=True)
            
            # Get Action
            res = get_llm_action(api_key, obs_text)
            thought = res.get("thought", "N/A")
            action = res.get("action", res) # Fallback
            
            print(f"REASONING: {thought}", flush=True)
            print(f"Action: {json.dumps(action)}", flush=True)
                
            # Step Env
            resp = requests.post(f"{ENV_URL}/step", json=action, timeout=10)
            data = resp.json()
            obs = data["observation"]
            reward = data["reward"]
            
            report_lines.append(f"| {i} | {thought} | `{json.dumps(action)}` | Reward: {reward:.1f}, Meltdowns: {obs['meltdowns']} |")
            print(f"Result: Reward={reward}, Meltdowns={obs['meltdowns']}", flush=True)
            
            time.sleep(30) # 30s safety delay as requested
            
        # Final Grade
        resp = requests.post(f"{ENV_URL}/grader", timeout=10)
        grade = resp.json()
        
        report_lines.append("\n## Final Performance Metrics\n")
        report_lines.append("```json")
        report_lines.append(json.dumps(grade, indent=2))
        report_lines.append("```\n")
        
        # Save Report
        with open("groq_benchmark_report.md", "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))
        print("\n✅ Report generated: groq_benchmark_report.md", flush=True)

    except Exception as e:
        print(f"X Test Failed: {e}", flush=True)
    finally:
        server_proc.terminate()

if __name__ == "__main__":
    run_groq_test()

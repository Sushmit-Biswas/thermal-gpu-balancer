"""
ClusterOps Heuristic Test.

Runs a simple smart heuristic directly against the server to verify
the environment is solvable and rewards are non-trivial.

Usage:
    1. Start the server: uvicorn server.app:app --port 8000
    2. Run this script:  python heuristic_test.py
"""

import os
import requests

ENV_URL = os.getenv("ENVIRONMENT_BASE_URL", "http://localhost:8000")


def env_reset(difficulty="medium", scenario="01_baseline"):
    return requests.post(
        f"{ENV_URL}/reset",
        json={"difficulty": difficulty, "scenario": scenario},
    ).json()


def env_step(action):
    return requests.post(f"{ENV_URL}/step", json=action).json()


def run_heuristic_episode(difficulty="medium", scenario="01_baseline"):
    print(f"\n--- Running SMART HEURISTIC | difficulty={difficulty} scenario={scenario} ---")
    data = env_reset(difficulty, scenario)
    obs = data.get("observation", {})
    total_reward = 0.0
    step = 0

    while not data.get("done", False):
        nodes = obs.get("gpu_nodes", [])
        queue = obs.get("job_queue", [])

        # 1. Thermal Emergency: If node > 90C, EVICT immediately!
        action = {"action_type": "wait"}
        for n in nodes:
            if n["status"] == "busy" and n["temperature"] >= 90.0:
                action = {"action_type": "evict", "node_id": n["id"]}
                break

        # 2. Allocation Logic: If we have jobs, pick the COOLEST idle node
        if action["action_type"] == "wait" and queue:
            idle_nodes = [n for n in nodes if n["status"] == "idle"]
            # Sort queue so VIP is highest priority (0), then inference (1), then batch (2)
            sorted_q = sorted(
                queue,
                key=lambda j: {"vip_training": 0, "inference": 1, "batch": 2}.get(j["type"], 3),
            )

            if idle_nodes and sorted_q:
                # Pick coolest node
                best_node = min(idle_nodes, key=lambda x: x["temperature"])
                # Pick highest priority job
                job = sorted_q[0]
                action = {
                    "action_type": "allocate",
                    "job_id": job["id"],
                    "node_id": best_node["id"],
                }

        data = env_step(action)
        obs = data.get("observation", {})
        total_reward += data.get("reward", 0)
        step += 1
        print(
            f"  Step {step:2d} | Action: {action['action_type']:8} | "
            f"Reward: {data.get('reward'):+5.1f} | Total: {total_reward:6.1f}",
            flush=True,
        )

    print(f"\nEPISODE COMPLETE! Total Reward: {total_reward:.1f}")

    # Grader
    grade = requests.post(f"{ENV_URL}/grader").json()
    print(f"Grade: {grade.get('score', 0.0):.4f}")
    return total_reward


if __name__ == "__main__":
    import sys

    difficulty = sys.argv[1] if len(sys.argv) > 1 else "medium"
    scenario = sys.argv[2] if len(sys.argv) > 2 else "01_baseline"

    try:
        run_heuristic_episode(difficulty, scenario)
    except Exception as e:
        print(
            f"Error: {e}. Is the server running? Run 'python -m uvicorn server.app:app' first!"
        )

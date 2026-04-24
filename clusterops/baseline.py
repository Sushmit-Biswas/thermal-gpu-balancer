#!/usr/bin/env python3
"""
ClusterOps Baseline Agent.

A deterministic heuristic agent. Used as the "before training" comparison.

Strategy:
    1. If any busy node is above 80°C, evict its job.
    2. Allocate highest-priority queued jobs to the coolest idle nodes.
    3. Otherwise, wait.
"""

import requests
import sys

ENV_URL = "http://localhost:8000"


def reset(difficulty="medium"):
    resp = requests.post(f"{ENV_URL}/reset", json={"difficulty": difficulty})
    return resp.json()


def step(action_type, job_id="", node_id=-1):
    resp = requests.post(f"{ENV_URL}/step", json={
        "action_type": action_type,
        "job_id": job_id,
        "node_id": node_id,
    })
    return resp.json()


def run_baseline(difficulty="medium"):
    print(f"\n{'='*60}")
    print(f"ClusterOps Baseline Agent | Difficulty: {difficulty}")
    print(f"{'='*60}")

    obs = reset(difficulty)
    obs_data = obs.get("observation", obs)
    total_reward = 0.0
    step_num = 0

    while True:
        done = obs.get("done", False)
        if done:
            break

        gpu_nodes = obs_data.get("gpu_nodes", [])
        job_queue = obs_data.get("job_queue", [])

        action_type = "wait"
        action_job_id = ""
        action_node_id = -1

        # Priority 1: Evict jobs from overheating nodes
        for node in gpu_nodes:
            if node["status"] == "busy" and node["temperature"] >= 80.0:
                action_type = "evict"
                action_node_id = node["id"]
                break

        # Priority 2: Allocate highest-priority jobs to coolest idle nodes
        if action_type == "wait" and job_queue:
            priority_order = {"vip_training": 0, "inference": 1, "batch": 2}
            sorted_jobs = sorted(job_queue, key=lambda j: priority_order.get(j.get("type", "batch"), 99))
            idle_nodes = sorted(
                [n for n in gpu_nodes if n["status"] == "idle"],
                key=lambda n: n["temperature"],
            )
            if sorted_jobs and idle_nodes:
                action_type = "allocate"
                action_job_id = sorted_jobs[0]["id"]
                action_node_id = idle_nodes[0]["id"]

        obs = step(action_type, action_job_id, action_node_id)
        obs_data = obs.get("observation", obs)
        step_reward = obs.get("reward", 0.0) or 0.0
        total_reward += step_reward
        step_num += 1

        if step_num % 10 == 0:
            print(f"  Step {step_num:3d} | Reward: {step_reward:+8.1f} | "
                  f"Completed: {obs_data.get('completed_jobs', 0)} | "
                  f"Meltdowns: {obs_data.get('meltdowns', 0)} | "
                  f"Queue: {len(obs_data.get('job_queue', []))}")

    # Get grade
    grade_resp = requests.post(f"{ENV_URL}/grader").json()

    print(f"\n--- Episode Complete ---")
    print(f"  Total Steps:    {step_num}")
    print(f"  Total Reward:   {total_reward:.1f}")
    print(f"  Jobs Completed: {obs_data.get('completed_jobs', 0)}")
    print(f"  Meltdowns:      {obs_data.get('meltdowns', 0)}")
    print(f"  Queue Left:     {len(obs_data.get('job_queue', []))}")
    print(f"  Grade:          {grade_resp.get('score', 0.0):.4f}")
    print(f"{'='*60}\n")

    return total_reward


def main():
    difficulty = sys.argv[1] if len(sys.argv) > 1 else "medium"
    rewards = []
    num_episodes = 3

    print(f"\nRunning {num_episodes} baseline episodes on '{difficulty}' difficulty...\n")

    for i in range(num_episodes):
        print(f"--- Episode {i+1}/{num_episodes} ---")
        r = run_baseline(difficulty)
        rewards.append(r)

    print(f"\n{'='*60}")
    print(f"BASELINE SUMMARY ({difficulty})")
    print(f"  Average Reward: {sum(rewards)/len(rewards):.1f}")
    print(f"  Best Reward:    {max(rewards):.1f}")
    print(f"  Worst Reward:   {min(rewards):.1f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

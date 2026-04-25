"""
ClusterOps Smart Heuristic Agent.

A more intelligent, heuristic-based agent for the ClusterOps environment.
Uses proactive cooling and priority-based allocation.
"""

import os
import requests

ENV_URL = os.getenv("ENVIRONMENT_BASE_URL", "http://localhost:8000")
PROACTIVE_COOLING_THRESHOLD = 60.0
EMERGENCY_EVICTION_THRESHOLD = 92.0


def choose_action(observation):
    """
    Chooses an action based on a set of heuristics.

    Returns:
        (action_dict, reason_string)
    """
    gpu_nodes = observation.get("gpu_nodes", [])
    job_queue = observation.get("job_queue", [])

    # 1. Emergency Eviction: If any busy node is about to overheat, evict its job.
    for node in gpu_nodes:
        if node["status"] == "busy" and node["temperature"] >= EMERGENCY_EVICTION_THRESHOLD:
            return {
                "action_type": "evict",
                "node_id": node["id"]
            }, f"Emergency evicting from hot node {node['id']}"

    # 2. Proactive Cooling: If any idle node is hot, cool it down.
    for node in gpu_nodes:
        if node["status"] == "idle" and node["temperature"] > PROACTIVE_COOLING_THRESHOLD:
            return {
                "action_type": "cooldown",
                "node_id": node["id"]
            }, f"Proactively cooling hot idle node {node['id']}"

    # 3. Allocate High-Priority Jobs:
    if job_queue:
        # Sort queue by priority (vip_training > inference > batch)
        sorted_queue = sorted(
            job_queue,
            key=lambda j: {"vip_training": 0, "inference": 1, "batch": 2}.get(j["type"], 3),
        )

        idle_nodes = [n for n in gpu_nodes if n["status"] == "idle"]
        if idle_nodes:
            # Find the coolest idle node
            coolest_node = min(idle_nodes, key=lambda n: n["temperature"])

            # Allocate the highest priority job to the coolest node
            job_to_allocate = sorted_queue[0]
            return {
                "action_type": "allocate",
                "job_id": job_to_allocate["id"],
                "node_id": coolest_node["id"]
            }, f"Allocating priority job {job_to_allocate['id']} to coolest node {coolest_node['id']}"

    # 4. If nothing else, wait.
    return {"action_type": "wait"}, "Waiting"


def run_smart_agent(difficulty="medium", scenario="01_baseline"):
    """
    Runs the smart agent against the environment.
    """
    print(f"Starting smart agent on difficulty={difficulty}, scenario={scenario}")

    try:
        resp = requests.post(
            f"{ENV_URL}/reset",
            json={"difficulty": difficulty, "scenario": scenario},
        )
        if not resp.ok:
            print(f"Server not ready. Status: {resp.status_code}")
            return
    except requests.ConnectionError:
        print(f"Connection to the server failed. Is the server running on {ENV_URL}?")
        return

    done = False
    total_reward = 0

    while not done:
        resp_data = resp.json()
        observation = resp_data.get("observation", {})
        metadata = resp_data.get("metadata", {})
        if not observation:
            print("Invalid observation received. Ending run.")
            break

        action, reason = choose_action(observation)

        step_num = metadata.get("step", "N/A")
        print(f"Step {step_num}: {reason}")

        resp = requests.post(f"{ENV_URL}/step", json=action)
        if not resp.ok:
            print(f"Error during step: {resp.text}")
            break

        data = resp.json()
        done = data.get("done", False)
        reward = data.get("reward", 0)
        total_reward += reward

    print("\nEpisode finished.")
    print(f"Total reward: {total_reward}")

    grader_resp = requests.post(f"{ENV_URL}/grader")
    if grader_resp.ok:
        print("Final Score:")
        print(grader_resp.json())
    else:
        print("Could not get final score.")


if __name__ == "__main__":
    import sys
    difficulty = sys.argv[1] if len(sys.argv) > 1 else "medium"
    scenario = sys.argv[2] if len(sys.argv) > 2 else "01_baseline"
    run_smart_agent(difficulty=difficulty, scenario=scenario)

import requests
import json

BASE_URL = "http://127.0.0.1:8000"

def print_state(response, action_taken=""):
    if not response.ok:
        print(f"Error: {response.status_code} - {response.text}")
        return

    data = response.json()
    obs = data.get("observation", {})
    
    if action_taken:
        print(f"Action Taken: {action_taken}")

    print("\n" + "="*50)
    print(f"Step: {obs.get('metadata', {}).get('step', 'N/A')}, Reward: {data.get('reward', 'N/A')}, Done: {data.get('done', 'N/A')}")
    print(f"Feedback: {obs.get('feedback', 'N/A')}")
    print("-"*50)

    print("GPU Nodes:")
    for node in obs.get("gpu_nodes", []):
        print(f"  Node {node['id']}: {node['status']:<10} | Temp: {node['temperature']:.1f}°C | Job: {node.get('job_id', 'None')}")

    print("\nJob Queue:")
    for job in obs.get("job_queue", []):
        print(f"  Job {job['id']}: {job['type']:<15} | Duration: {job['duration']} | Wait Time: {job['wait_time']}")
    
    print("\n" + "="*50 + "\n")


def run_test_scenario():
    # Reset environment to easy
    print("Resetting environment to easy mode...")
    resp = requests.post(f"{BASE_URL}/reset", json={"difficulty": "easy"})
    print_state(resp, "RESET")

    # Let's try to allocate the first job in the queue to the first node
    if resp.ok and resp.json()["observation"]["job_queue"]:
        job_id = resp.json()["observation"]["job_queue"][0]["id"]
        action = {"action_type": "allocate", "job_id": job_id, "node_id": 0}
        print(f"Attempting to allocate {job_id} to node 0...")
        resp = requests.post(f"{BASE_URL}/step", json=action)
        print_state(resp, f"ALLOCATE {job_id} to Node 0")

    # Wait for a few steps to see temperature changes
    for i in range(3):
        print(f"Waiting... (Step {i+1}/3)")
        resp = requests.post(f"{BASE_URL}/step", json={"action_type": "wait"})
        print_state(resp, "WAIT")

    # Try an invalid action
    print("Attempting an invalid action (allocating a non-existent job)...")
    action = {"action_type": "allocate", "job_id": "job_999", "node_id": 1}
    resp = requests.post(f"{BASE_URL}/step", json=action)
    print_state(resp, "ALLOCATE job_999 to Node 1")

    # Run until the episode is done
    while resp.ok and not resp.json().get("done"):
        # A simple agent: if there's a job and an idle node, allocate. Otherwise, wait.
        obs = resp.json()["observation"]
        if obs["job_queue"] and any(n["status"] == "idle" for n in obs["gpu_nodes"]):
            job_to_allocate = obs["job_queue"][0]
            idle_node = next(n for n in obs["gpu_nodes"] if n["status"] == "idle")
            action = {"action_type": "allocate", "job_id": job_to_allocate["id"], "node_id": idle_node["id"]}
            resp = requests.post(f"{BASE_URL}/step", json=action)
            print_state(resp, f"ALLOCATE {action['job_id']} to Node {action['node_id']}")
        else:
            resp = requests.post(f"{BASE_URL}/step", json={"action_type": "wait"})
            print_state(resp, "WAIT")

    print("Episode finished.")
    
    # Get final score
    print("Getting final score...")
    grader_resp = requests.post(f"{BASE_URL}/grader")
    if grader_resp.ok:
        print("Grader Response:")
        print(json.dumps(grader_resp.json(), indent=2))
    else:
        print(f"Error getting score: {grader_resp.status_code} - {grader_resp.text}")


if __name__ == "__main__":
    try:
        # Health check
        health = requests.get(f"{BASE_URL}/health")
        if health.ok:
            print("Server is healthy!")
            run_test_scenario()
        else:
            print(f"Server health check failed: {health.status_code}")
    except requests.exceptions.ConnectionError as e:
        print(f"Connection to the server failed. Is the server running on {BASE_URL}?")
        print("Please run 'python thermal-gpu-balancer/clusterops/server/app.py' in a separate terminal.")


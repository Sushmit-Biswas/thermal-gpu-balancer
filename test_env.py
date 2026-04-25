import requests, json

# Reset
r = requests.post('http://localhost:8000/reset', json={})
obs = r.json()
obs_data = obs.get('observation', obs)
print('Initial queue:', len(obs_data.get('job_queue', [])))
print('Initial nodes:', len(obs_data.get('gpu_nodes', [])))

queue = obs_data.get('job_queue', [])
if queue:
    job = queue[0]
    print(f"Allocating {job['id']} to node 0...")
    r = requests.post('http://localhost:8000/step', json={
        'action': {'action_type': 'allocate', 'job_id': job['id'], 'node_id': 0}
    })
    d = r.json()
    print(f"  Reward: {d.get('reward')} Feedback: {d['observation']['feedback']}")

# Wait 6 steps
for i in range(6):
    r = requests.post('http://localhost:8000/step', json={
        'action': {'action_type': 'wait'}
    })
    d = r.json()
    od = d['observation']
    print(f"  Step {i+2}: reward={d.get('reward'):.1f} completed={od['completed_jobs']} "
          f"meltdowns={od['meltdowns']} node0_temp={od['gpu_nodes'][0]['temperature']} "
          f"node0_status={od['gpu_nodes'][0]['status']}")

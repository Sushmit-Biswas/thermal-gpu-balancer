import gymnasium as gym
from gymnasium import spaces
import numpy as np
import requests

class ClusterOpsEnv(gym.Env):
    """A custom Gym environment for the ClusterOps server."""
    metadata = {'render.modes': ['human']}

    def __init__(self, difficulty='hard', max_nodes=16, max_queue=20):
        super(ClusterOpsEnv, self).__init__()

        self.difficulty = difficulty
        self.base_url = "http://127.0.0.1:8000"
        self.max_nodes = max_nodes
        self.max_queue = max_queue

        # Define action and observation space
        # They must be gym.spaces objects
        # Example: action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        # N_DISCRETE_ACTIONS = 1 (wait) + max_nodes (cooldown) + max_nodes (evict) + (max_queue * max_nodes) (allocate)
        self.n_actions = 1 + self.max_nodes + self.max_nodes + (self.max_queue * self.max_nodes)
        self.action_space = spaces.Discrete(self.n_actions)

        # Observation space:
        # For each node: status (4 states), temp, job_type (4 types), duration
        # For each job in queue: type (4 types), duration, wait_time
        node_state_size = 4 + 1 + 4 + 1 
        queue_state_size = 4 + 1 + 1
        self.observation_space = spaces.Box(low=0, high=1, 
                                            shape=(self.max_nodes * node_state_size + self.max_queue * queue_state_size,), 
                                            dtype=np.float32)

    def _get_obs(self, obs_data):
        """Converts the JSON observation from the server into a numpy array."""
        
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        
        # Node states
        for i, node in enumerate(obs_data.get('gpu_nodes', [])):
            if i < self.max_nodes:
                base_idx = i * (4 + 1 + 4 + 1)
                
                # Status: idle, busy, cooldown, failed
                status_map = {"idle": 0, "busy": 1, "cooldown": 2, "failed": 3}
                obs[base_idx + status_map.get(node['status'], 0)] = 1
                
                # Temperature (normalized)
                obs[base_idx + 4] = node['temperature'] / 100.0
                
                # Job type
                job_type_map = {"vip_training": 0, "inference": 1, "batch": 2}
                if node['job_type']:
                    obs[base_idx + 5 + job_type_map.get(node['job_type'], 3)] = 1
                
                # Job duration remaining (normalized)
                obs[base_idx + 9] = node['job_duration_remaining'] / 10.0

        # Job queue states
        for i, job in enumerate(obs_data.get('job_queue', [])):
            if i < self.max_queue:
                base_idx = self.max_nodes * (4 + 1 + 4 + 1) + i * (4 + 1 + 1)
                
                # Job type
                job_type_map = {"vip_training": 0, "inference": 1, "batch": 2}
                obs[base_idx + job_type_map.get(job['type'], 3)] = 1
                
                # Duration (normalized)
                obs[base_idx + 4] = job['duration'] / 10.0
                
                # Wait time (normalized)
                obs[base_idx + 5] = job['wait_time'] / 50.0
                
        return obs

    def _get_action(self, action_index):
        """Converts a discrete action index into a server-compatible action dictionary."""
        
        # Wait
        if action_index == 0:
            return {"action_type": "wait"}
        
        action_index -= 1

        # Cooldown
        if action_index < self.max_nodes:
            return {"action_type": "cooldown", "node_id": action_index}
        
        action_index -= self.max_nodes

        # Evict
        if action_index < self.max_nodes:
            return {"action_type": "evict", "node_id": action_index}
        
        action_index -= self.max_nodes

        # Allocate
        job_idx = action_index // self.max_nodes
        node_idx = action_index % self.max_nodes
        
        try:
            job_id = self.job_queue[job_idx]['id']
            return {"action_type": "allocate", "job_id": job_id, "node_id": node_idx}
        except IndexError:
            # If the chosen job doesn't exist, default to wait
            return {"action_type": "wait"}


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        response = requests.post(f"{self.base_url}/reset", json={'difficulty': self.difficulty})
        obs_data = response.json()['observation']
        self.job_queue = obs_data.get('job_queue', [])
        return self._get_obs(obs_data), {}

    def step(self, action):
        server_action = self._get_action(action)
        
        response = requests.post(f"{self.base_url}/step", json=server_action)
        data = response.json()
        
        obs_data = data['observation']
        reward = data['reward']
        done = data['done']
        
        self.job_queue = obs_data.get('job_queue', [])
        
        return self._get_obs(obs_data), reward, done, False, {}

    def render(self, mode='human', close=False):
        # For now, we're not implementing a visual renderer
        pass

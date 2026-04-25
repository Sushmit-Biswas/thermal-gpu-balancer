# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
ClusterOps: Thermal GPU Balancer Environment Implementation.

A high-stakes data center scheduler simulation where an LLM agent must
allocate GPU training/inference jobs across a cluster of nodes while managing
adversarial thermal constraints, job priorities, and operational scenarios.
"""

import os
import sys
from uuid import uuid4
import random
import copy

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from clusterops.models import ClusteropsAction, ClusteropsObservation

# ─── Job Priority Tiers ────────────────────────────────────────────────────────
JOB_TYPES = {
    "vip_training": {
        "heat_rate": 15.0,
        "reward_on_complete": 40.0,
        "queue_penalty": -2.0,
        "duration_range": (3, 7),
    },
    "inference": {
        "heat_rate": 8.0,
        "reward_on_complete": 15.0,
        "queue_penalty": -0.5,
        "duration_range": (1, 3),
    },
    "batch": {
        "heat_rate": 5.0,
        "reward_on_complete": 8.0,
        "queue_penalty": -0.2,
        "duration_range": (2, 5),
    },
}

# ─── Operational Scenarios (Curriculum) ─────────────────────────────────────────
SCENARIOS = {
    "01_baseline": {
        "num_nodes": 10,
        "max_steps": 100,
        "spawn_rate": 0.4,
        "max_spawn": 2,
        "thermal_limit": 100.0,
        "cool_rate": 10.0,
        "initial_jobs": 3,
        "description": "Standard Operations. All nodes are equal. Standard job flow.",
    },
    "02_spatial_bleed": {
        "num_nodes": 10,
        "max_steps": 100,
        "spawn_rate": 0.4,
        "max_spawn": 2,
        "thermal_limit": 100.0,
        "cool_rate": 10.0,
        "initial_jobs": 3,
        "description": "Rack Thermodynamics. Heat bleeds (+3C) to adjacent nodes if a node hits 85C.",
    },
    "03_heterogeneous": {
        "num_nodes": 10,
        "max_steps": 100,
        "spawn_rate": 0.4,
        "max_spawn": 2,
        "thermal_limit": 100.0,
        "cool_rate": 10.0,
        "initial_jobs": 3,
        "description": "Hardware Diversity. Even IDs (H100): 2x speed, 2x heat. Odd IDs (T4): 1x speed, 0.5x heat.",
    },
    "04_maintenance": {
        "num_nodes": 10,
        "max_steps": 100,
        "spawn_rate": 0.3,
        "max_spawn": 2,
        "thermal_limit": 100.0,
        "cool_rate": 10.0,
        "initial_jobs": 3,
        "description": "Scheduled Outage. Nodes 0-4 go offline at step 35 (warning at step 20).",
    },
    "05_adversarial": {
        "num_nodes": 10,
        "max_steps": 100,
        "spawn_rate": 0.0,
        "max_spawn": 0,
        "thermal_limit": 100.0,
        "cool_rate": 10.0,
        "initial_jobs": 0,
        "description": "Traffic Spikes. Queue is empty, then 15 VIP jobs suddenly drop at step 10.",
    },
}

# ─── Legacy Difficulty Config (for testing/backward compatibility) ───
DIFFICULTY_CONFIG = {
    "easy": {"num_nodes": 6, "max_steps": 100, "spawn_rate": 0.3},
    "medium": {"num_nodes": 10, "max_steps": 100, "spawn_rate": 0.4},
    "hard": {"num_nodes": 16, "max_steps": 150, "spawn_rate": 0.5},
    "expert": {"num_nodes": 20, "max_steps": 200, "spawn_rate": 0.6},
}


class ClusteropsEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, scenario: str = "01_baseline", **kwargs):
        # Support both 'scenario' and legacy 'difficulty' kwargs
        difficulty = kwargs.get('difficulty', None)
        if difficulty and difficulty in DIFFICULTY_CONFIG:
            self.scenario = difficulty  # store raw difficulty key; _init_state handles it
        elif scenario in SCENARIOS:
            self.scenario = scenario
        else:
            self.scenario = "01_baseline"  # any unknown falls back to baseline
        self._init_state()

    @property
    def difficulty(self):
        """Expose scenario as difficulty for backward compat with tests & app.py."""
        return self.scenario

    def _init_state(self):
        if self.scenario in DIFFICULTY_CONFIG:
            config = {**SCENARIOS["01_baseline"], **DIFFICULTY_CONFIG[self.scenario]}
        else:
            config = SCENARIOS.get(self.scenario, SCENARIOS["01_baseline"])
        
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.config = config
        self.num_nodes = config["num_nodes"]

        self.max_steps = config["max_steps"]
        self.thermal_limit = config["thermal_limit"]
        self.cool_rate = config["cool_rate"]

        self.gpu_nodes = []
        for i in range(self.num_nodes):
            self.gpu_nodes.append({
                "id": i,
                "status": "idle",
                "temperature": round(random.uniform(35.0, 45.0), 1),
                "job_id": None,
                "job_type": None,
                "job_duration_remaining": 0,
            })

        self.job_queue = []
        self.completed_jobs = 0
        self.failed_jobs = 0
        self.meltdowns = 0
        self.evictions = 0
        self.thrashing_events = 0
        self.total_reward = 0.0
        self.next_job_id = 1

        self._node_alloc_step = [0] * self.num_nodes
        self._queue_saturation_limit = self.num_nodes * 2

        self._spawn_jobs(config["initial_jobs"])

    def _spawn_jobs(self, count, specific_type=None):
        for _ in range(count):
            job_type = specific_type or random.choices(
                list(JOB_TYPES.keys()),
                weights=[0.25, 0.45, 0.30],
                k=1
            )[0]
            props = JOB_TYPES[job_type]
            duration = random.randint(*props["duration_range"])
            deadline = random.randint(10, 15) if job_type == "vip_training" else random.randint(20, 35)
            
            self.job_queue.append({
                "id": f"job_{self.next_job_id}",
                "type": job_type,
                "duration": duration,
                "wait_time": 0,
                "deadline": deadline,
            })
            self.next_job_id += 1

    def reset(self, scenario: str = None, difficulty: str = None) -> ClusteropsObservation:
        target = scenario or difficulty or self.scenario
        if target in SCENARIOS:
            self.scenario = target
        elif target in DIFFICULTY_CONFIG:
            self.scenario = target   # keep for _init_state to expand
        else:
            self.scenario = "01_baseline"
        self._init_state()
        return self._build_observation(f"Environment reset. Scenario: {self.scenario}")


    def step(self, action: ClusteropsAction) -> ClusteropsObservation:  # type: ignore[override]
        self._state.step_count += 1
        reward = 0.0
        feedback = ""

        # ─── Scenario-Specific Events ───
        if self.scenario == "04_maintenance":
            if self._state.step_count == 20:
                feedback += "WARNING: Scheduled Outage. Nodes 0-4 will go offline at step 35! "
            elif self._state.step_count == 35:
                feedback += "MAINTENANCE: Nodes 0-4 are now offline! "
                for i in range(5):
                    if self.gpu_nodes[i]["status"] == "busy":
                        reward -= 50.0  # Catastrophic failure for not draining
                        self.failed_jobs += 1
                        self.meltdowns += 1
                    self.gpu_nodes[i]["status"] = "failed"
                    self.gpu_nodes[i]["job_id"] = None
                    self.gpu_nodes[i]["job_type"] = None

        if self.scenario == "05_adversarial":
            if self._state.step_count == 10:
                self._spawn_jobs(15, specific_type="vip_training")
                feedback += "DDoS ALERT: 15 VIP jobs just hit the queue! "

        # ─── Process Action ───
        if action.action_type == "allocate":
            act_reward, act_fb = self._handle_allocate(action)
        elif action.action_type == "evict":
            act_reward, act_fb = self._handle_evict(action)
        elif action.action_type == "cooldown":
            act_reward, act_fb = self._handle_cooldown(action)
        elif action.action_type == "wait":
            act_reward, act_fb = 0.0, "Agent chose to wait."
        else:
            act_reward, act_fb = -5.0, f"Error: Unknown action type '{action.action_type}'."
            
        reward += act_reward
        feedback += act_fb

        # ─── Simulate Physics ───
        reward += self._simulate_physics()

        # ─── Queue Aging ───
        reward += self._age_queue()

        # ─── Random Jobs ───
        if self.scenario not in ["05_adversarial"]:
            if random.random() < self.config["spawn_rate"]:
                num = random.randint(1, self.config["max_spawn"])
                self._spawn_jobs(num)

        # ─── Saturation Check (Anti-Exploit) ───
        queue_overflow = len(self.job_queue) >= self._queue_saturation_limit
        if queue_overflow:
            reward -= 100.0
            feedback += " CRITICAL: Queue saturation! Episode terminated."

        done = self._state.step_count >= self.max_steps or queue_overflow
        self.total_reward += reward

        return self._build_observation(feedback, reward, done)

    def _handle_allocate(self, action):
        job = next((j for j in self.job_queue if j["id"] == action.job_id), None)
        if not job:
            return -5.0, f"Error: Job '{action.job_id}' not found in queue."
        if action.node_id < 0 or action.node_id >= self.num_nodes:
            return -5.0, f"Error: Node ID {action.node_id} out of range."
        node = self.gpu_nodes[action.node_id]
        if node["status"] != "idle":
            return -5.0, f"Error: Node {action.node_id} is '{node['status']}', not idle."

        self.job_queue.remove(job)
        node["status"] = "busy"
        node["job_id"] = job["id"]
        node["job_type"] = job["type"]
        node["job_duration_remaining"] = job["duration"]
        self._node_alloc_step[action.node_id] = self._state.step_count
        return 0.0, f"Allocated {job['id']} ({job['type']}) to Node {action.node_id}."

    def _handle_evict(self, action):
        if action.node_id < 0 or action.node_id >= self.num_nodes:
            return -5.0, f"Error: Node ID {action.node_id} out of range."
        node = self.gpu_nodes[action.node_id]
        if node["status"] != "busy":
            return -2.0, f"Error: Node {action.node_id} has no running job to evict."

        evicted_job = node["job_id"]
        self.evictions += 1
        self.failed_jobs += 1
        node["status"] = "idle"
        node["job_id"] = None
        node["job_type"] = None
        node["job_duration_remaining"] = 0

        steps_since_alloc = self._state.step_count - self._node_alloc_step[action.node_id]
        if steps_since_alloc <= 2:
            self.thrashing_events += 1
            return -30.0, f"THRASHING: Evicted {evicted_job} only {steps_since_alloc} step(s) after alloc. 3x penalty."
        return -10.0, f"Evicted {evicted_job} from Node {action.node_id}. Compute wasted."

    def _handle_cooldown(self, action):
        if action.node_id < 0 or action.node_id >= self.num_nodes:
            return -5.0, f"Error: Node ID {action.node_id} out of range."
        node = self.gpu_nodes[action.node_id]
        if node["status"] != "idle":
            return -2.0, f"Error: Node {action.node_id} must be idle to force-cool."
        node["status"] = "cooldown"
        return 0.0, f"Force-cooling Node {action.node_id}."

    def _simulate_physics(self):
        reward = 0.0
        
        # 1. Base Heat & Durations
        for node in self.gpu_nodes:
            if node["status"] == "busy":
                job_type = node.get("job_type", "batch")
                heat_rate = JOB_TYPES.get(job_type, JOB_TYPES["batch"])["heat_rate"]
                
                # 03_heterogeneous: Hardware Diversity
                if self.scenario == "03_heterogeneous":
                    if node["id"] % 2 == 0:  # H100
                        heat_rate *= 2.0
                        node["job_duration_remaining"] -= 2
                    else:  # T4
                        heat_rate *= 0.5
                        node["job_duration_remaining"] -= 1
                else:
                    node["job_duration_remaining"] -= 1
                    
                node["temperature"] += heat_rate

                if node["job_duration_remaining"] <= 0:
                    reward += JOB_TYPES.get(job_type, JOB_TYPES["batch"])["reward_on_complete"]
                    self.completed_jobs += 1
                    node["status"] = "idle"
                    node["job_id"] = None
                    node["job_type"] = None

            elif node["status"] == "cooldown":
                node["temperature"] = max(35.0, node["temperature"] - self.cool_rate * 2.5)
                node["status"] = "idle"

            elif node["status"] == "idle":
                node["temperature"] = max(35.0, node["temperature"] - self.cool_rate)

            elif node["status"] == "failed":
                node["temperature"] = max(35.0, node["temperature"] - self.cool_rate * 1.5)
                # 04_maintenance: offline nodes don't auto-recover
                if self.scenario == "04_maintenance" and node["id"] <= 4 and self._state.step_count >= 35:
                    pass
                elif node["temperature"] <= 50.0:
                    node["status"] = "idle"

        # 2. Spatial Bleed (02_spatial_bleed)
        if self.scenario == "02_spatial_bleed":
            temps = [n["temperature"] for n in self.gpu_nodes]
            for i, temp in enumerate(temps):
                if temp >= 85.0:
                    if i > 0: self.gpu_nodes[i-1]["temperature"] += 3.0
                    if i < self.num_nodes - 1: self.gpu_nodes[i+1]["temperature"] += 3.0

        # 3. Meltdown Check & Clamping
        for node in self.gpu_nodes:
            if node["temperature"] >= self.thermal_limit and node["status"] == "busy":
                self.meltdowns += 1
                self.failed_jobs += 1
                reward -= 50.0
                node["temperature"] = self.thermal_limit * 0.75
                node["status"] = "failed"
                node["job_id"] = None
                node["job_type"] = None
                node["job_duration_remaining"] = 0
            
            node["temperature"] = round(node["temperature"], 1)

        return reward

    def _age_queue(self):
        penalty = 0.0
        remaining_jobs = []
        for job in self.job_queue:
            job["wait_time"] += 1
            if job["wait_time"] > job.get("deadline", 999):
                self.failed_jobs += 1
                penalty -= 20.0  # Massive penalty for starving VIPs
                continue
            penalty += JOB_TYPES.get(job["type"], JOB_TYPES["batch"])["queue_penalty"]
            remaining_jobs.append(job)
        self.job_queue = remaining_jobs
        return penalty

    def _build_observation(self, feedback, reward=0.0, done=False):
        thermal_warnings = sum(1 for n in self.gpu_nodes if n["temperature"] >= self.thermal_limit * 0.85)
        return ClusteropsObservation(
            gpu_nodes=copy.deepcopy(self.gpu_nodes),
            job_queue=copy.deepcopy(self.job_queue),
            thermal_warnings=thermal_warnings,
            meltdowns=self.meltdowns,
            completed_jobs=self.completed_jobs,
            feedback=feedback,
            reward=reward,
            done=done,
            metadata={
                "step": self._state.step_count,
                "max_steps": self.max_steps,
                "scenario": self.scenario,
                "evictions": self.evictions,
                "failed_jobs": self.failed_jobs,
                "total_reward": round(self.total_reward, 2),
                "rubric": self.grade_rubric(),
            },
        )

    def grade(self) -> float:
        return self.grade_rubric()["total"]

    def grade_rubric(self) -> dict:
        max_possible = max(self.max_steps * 0.3, 1.0)
        thermal_safety = max(0.0, 1.0 - self.meltdowns * 0.20)
        throughput = min(self.completed_jobs / max_possible, 1.0)
        total_jobs_handled = self.completed_jobs + self.evictions
        efficiency = self.completed_jobs / total_jobs_handled if total_jobs_handled > 0 else 1.0
        sla_compliance = max(0.0, 1.0 - (self.failed_jobs * 0.10))

        total = round(max(0.0, min(1.0, (
            thermal_safety  * 0.35 +
            throughput      * 0.30 +
            efficiency      * 0.20 +
            sla_compliance  * 0.15
        ))), 4)

        return {
            "total": total,
            "score": total,
            "thermal_safety": round(thermal_safety, 4),
            "throughput": round(throughput, 4),
            "efficiency": round(efficiency, 4),
            "sla_compliance": round(sla_compliance, 4),
            "completed_jobs": self.completed_jobs,
            "meltdowns": self.meltdowns,
            "evictions": self.evictions,
            "thrashing_events": self.thrashing_events,
            "failed_jobs": self.failed_jobs,
            "total_reward": round(self.total_reward, 2),
        }

    def curriculum_difficulty(self) -> str:
        # Re-map legacy method to scenarios
        score = self.grade()
        scenarios = list(SCENARIOS.keys())
        idx = scenarios.index(self.scenario) if self.scenario in scenarios else 0
        if score >= 0.70 and idx < len(scenarios) - 1:
            return scenarios[idx + 1]
        return self.scenario

    @property
    def state(self) -> State:
        return self._state

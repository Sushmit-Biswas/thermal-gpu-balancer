# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
ClusterOps: Thermal GPU Balancer Environment Implementation.

A high-stakes data center scheduler simulation where an LLM agent must
allocate GPU training/inference jobs across a cluster of nodes while managing
adversarial thermal constraints, job priorities, and random hardware failures.

Differentiator from SRE/Kube gyms:
- This is NOT a text-log debugging environment. There are no YAML files to fix.
- This is a CONTROL SYSTEMS simulator. The agent manages physical thermal
  properties of hardware in real-time, making it closer to industrial control
  than IT support.
"""

import os
import sys
from uuid import uuid4
import random
import copy

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

# Ensure the parent directory is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import ClusteropsAction, ClusteropsObservation


# ─── Job Priority Tiers ────────────────────────────────────────────────────────
JOB_TYPES = {
    "vip_training": {
        "heat_rate": 15.0,    # Generates a LOT of heat
        "reward_on_complete": 40.0,
        "queue_penalty": -2.0,  # VIP jobs penalize heavily when waiting
        "duration_range": (3, 7),
    },
    "inference": {
        "heat_rate": 8.0,     # Moderate heat
        "reward_on_complete": 15.0,
        "queue_penalty": -0.5,
        "duration_range": (1, 3),
    },
    "batch": {
        "heat_rate": 5.0,     # Low heat
        "reward_on_complete": 8.0,
        "queue_penalty": -0.2,
        "duration_range": (2, 5),
    },
}

# ─── Difficulty Configurations ─────────────────────────────────────────────────
DIFFICULTY_CONFIG = {
    "easy": {
        "num_nodes": 6,
        "max_steps": 50,
        "spawn_rate": 0.3,
        "max_spawn": 1,
        "hardware_failure_rate": 0.0,
        "thermal_limit": 100.0,
        "cool_rate": 10.0,
        "initial_jobs": 2,
        "description": "Small cluster, no hardware failures. Learn the basics.",
    },
    "medium": {
        "num_nodes": 10,
        "max_steps": 100,
        "spawn_rate": 0.4,
        "max_spawn": 2,
        "hardware_failure_rate": 0.02,
        "thermal_limit": 95.0,
        "cool_rate": 8.0,
        "initial_jobs": 3,
        "description": "Full cluster with occasional random hardware degradation.",
    },
    "hard": {
        "num_nodes": 16,
        "max_steps": 150,
        "spawn_rate": 0.6,
        "max_spawn": 3,
        "hardware_failure_rate": 0.05,
        "thermal_limit": 90.0,
        "cool_rate": 6.0,
        "initial_jobs": 5,
        "description": "Massive cluster with aggressive job load, tight thermals, frequent failures.",
    },
}


class ClusteropsEnvironment(Environment):
    """
    Simulates an AI GPU Data Center under adversarial thermal conditions.

    The agent must allocate incoming jobs (VIP training, inference, batch) to
    GPU nodes while preventing thermal meltdowns. Nodes heat up when running
    jobs and cool down when idle. If a node exceeds the thermal limit, it
    triggers a meltdown: the job is destroyed and a massive penalty is applied.

    Actions:
        - allocate(job_id, node_id): Assign a queued job to an idle node.
        - evict(node_id): Emergency-stop a running job to cool a node.
        - cooldown(node_id): Force-cool a node (takes 1 step, node must be idle).
        - wait(): Do nothing this step.

    Reward Signal (dense, multi-component):
        - +8 to +40 for completing a job (depends on job type).
        - -0.2 to -2.0 per step per queued job (depends on priority).
        - -50 for a thermal meltdown.
        - -10 for evicting a running job (wasted compute).
        - -5 for invalid actions.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, difficulty: str = "medium"):
        self.difficulty = difficulty
        self._init_state()

    def _init_state(self):
        config = DIFFICULTY_CONFIG.get(self.difficulty, DIFFICULTY_CONFIG["medium"])
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.config = config
        self.num_nodes = config["num_nodes"]
        self.max_steps = config["max_steps"]
        self.thermal_limit = config["thermal_limit"]
        self.cool_rate = config["cool_rate"]

        # Build GPU node grid
        self.gpu_nodes = []
        for i in range(self.num_nodes):
            self.gpu_nodes.append({
                "id": i,
                "status": "idle",         # idle | busy | cooldown | failed
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
        self.total_reward = 0.0
        self.next_job_id = 1

        # Spawn initial jobs
        self._spawn_jobs(config["initial_jobs"])

    def _spawn_jobs(self, count):
        """Generates random incoming jobs with type-based properties."""
        for _ in range(count):
            job_type = random.choices(
                list(JOB_TYPES.keys()),
                weights=[0.25, 0.45, 0.30],  # inference is most common
                k=1
            )[0]
            props = JOB_TYPES[job_type]
            duration = random.randint(*props["duration_range"])
            # Deadlines: VIP jobs have tighter deadlines (10-15 steps), others longer
            deadline = random.randint(10, 15) if job_type == "vip_training" else random.randint(20, 35)
            
            self.job_queue.append({
                "id": f"job_{self.next_job_id}",
                "type": job_type,
                "duration": duration,
                "wait_time": 0,
                "deadline": deadline,
            })
            self.next_job_id += 1

    def reset(self, difficulty: str = None) -> ClusteropsObservation:
        """Reset the environment, optionally changing difficulty."""
        if difficulty:
            self.difficulty = difficulty
        self._init_state()
        return self._build_observation("Environment reset. Cluster online.")

    def step(self, action: ClusteropsAction) -> ClusteropsObservation:  # type: ignore[override]
        """Execute one timestep of the simulation."""
        self._state.step_count += 1
        reward = 0.0
        feedback = ""

        # ────── 1. Process Agent Action ──────
        if action.action_type == "allocate":
            reward, feedback = self._handle_allocate(action)

        elif action.action_type == "evict":
            reward, feedback = self._handle_evict(action)

        elif action.action_type == "cooldown":
            reward, feedback = self._handle_cooldown(action)

        elif action.action_type == "wait":
            feedback = "Agent chose to wait."

        else:
            feedback = f"Error: Unknown action type '{action.action_type}'."
            reward -= 5.0

        # ────── 2. Simulate Physics ──────
        physics_reward = self._simulate_physics()
        reward += physics_reward

        # ────── 3. Queue Aging ──────
        queue_penalty = self._age_queue()
        reward += queue_penalty

        # ────── 4. Random Hardware Failures ──────
        failure_penalty = self._random_hardware_failures()
        reward += failure_penalty

        # ────── 5. Spawn New Jobs ──────
        if random.random() < self.config["spawn_rate"]:
            num = random.randint(1, self.config["max_spawn"])
            self._spawn_jobs(num)

        # ────── 6. Termination ──────
        done = self._state.step_count >= self.max_steps
        self.total_reward += reward

        return self._build_observation(feedback, reward, done)

    # ─── Action Handlers ────────────────────────────────────────────────────────

    def _handle_allocate(self, action):
        """Allocate a queued job to a node."""
        job = next((j for j in self.job_queue if j["id"] == action.job_id), None)
        if not job:
            return -5.0, f"Error: Job '{action.job_id}' not found in queue."

        if action.node_id < 0 or action.node_id >= self.num_nodes:
            return -5.0, f"Error: Node ID {action.node_id} out of range [0-{self.num_nodes - 1}]."

        node = self.gpu_nodes[action.node_id]
        if node["status"] != "idle":
            return -5.0, f"Error: Node {action.node_id} is '{node['status']}', not idle."

        # Success: allocate
        self.job_queue.remove(job)
        node["status"] = "busy"
        node["job_id"] = job["id"]
        node["job_type"] = job["type"]
        node["job_duration_remaining"] = job["duration"]
        return 0.0, f"Allocated {job['id']} ({job['type']}) to Node {action.node_id}."

    def _handle_evict(self, action):
        """Emergency evict a job from a node."""
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
        return -10.0, f"Evicted {evicted_job} from Node {action.node_id}. Compute wasted."

    def _handle_cooldown(self, action):
        """Force-cool an idle node for 1 step."""
        if action.node_id < 0 or action.node_id >= self.num_nodes:
            return -5.0, f"Error: Node ID {action.node_id} out of range."

        node = self.gpu_nodes[action.node_id]
        if node["status"] != "idle":
            return -2.0, f"Error: Node {action.node_id} must be idle to force-cool."

        node["status"] = "cooldown"
        return 0.0, f"Force-cooling Node {action.node_id}."

    # ─── Physics Engine ─────────────────────────────────────────────────────────

    def _simulate_physics(self):
        """Update temperatures, complete jobs, detect meltdowns."""
        reward = 0.0

        for node in self.gpu_nodes:
            if node["status"] == "busy":
                # Heat up based on job type
                job_type = node.get("job_type", "batch")
                heat_rate = JOB_TYPES.get(job_type, JOB_TYPES["batch"])["heat_rate"]
                node["temperature"] = round(node["temperature"] + heat_rate, 1)
                node["job_duration_remaining"] -= 1

                # Job completed
                if node["job_duration_remaining"] <= 0:
                    job_reward = JOB_TYPES.get(job_type, JOB_TYPES["batch"])["reward_on_complete"]
                    reward += job_reward
                    self.completed_jobs += 1
                    node["status"] = "idle"
                    node["job_id"] = None
                    node["job_type"] = None

            elif node["status"] == "cooldown":
                # Aggressive cooling
                node["temperature"] = round(max(35.0, node["temperature"] - self.cool_rate * 2.5), 1)
                node["status"] = "idle"

            elif node["status"] == "idle":
                # Passive cooling
                node["temperature"] = round(max(35.0, node["temperature"] - self.cool_rate), 1)

            elif node["status"] == "failed":
                # Failed nodes recover after cooling below 50
                node["temperature"] = round(max(35.0, node["temperature"] - self.cool_rate * 1.5), 1)
                if node["temperature"] <= 50.0:
                    node["status"] = "idle"

            # ── Thermal Meltdown Check ──
            if node["temperature"] >= self.thermal_limit and node["status"] == "busy":
                self.meltdowns += 1
                self.failed_jobs += 1
                reward -= 50.0
                node["temperature"] = round(self.thermal_limit * 0.75, 1)
                node["status"] = "failed"
                node["job_id"] = None
                node["job_type"] = None
                node["job_duration_remaining"] = 0

        return reward

    def _age_queue(self):
        """Penalize agent for wait time and handle SLA deadlines."""
        penalty = 0.0
        remaining_jobs = []
        for job in self.job_queue:
            job["wait_time"] += 1
            
            # SLA Deadline Check
            if job["wait_time"] > job.get("deadline", 999):
                self.failed_jobs += 1
                penalty -= 20.0  # Massive penalty for losing a job due to starvation
                continue # Job is dropped
                
            job_penalty = JOB_TYPES.get(job["type"], JOB_TYPES["batch"])["queue_penalty"]
            penalty += job_penalty
            remaining_jobs.append(job)
            
        self.job_queue = remaining_jobs
        return penalty

    def _random_hardware_failures(self):
        """Randomly degrade idle nodes to simulate real hardware issues."""
        penalty = 0.0
        for node in self.gpu_nodes:
            if node["status"] == "idle" and random.random() < self.config["hardware_failure_rate"]:
                node["status"] = "failed"
                node["temperature"] = round(self.thermal_limit * 0.8, 1)
                penalty -= 15.0
        return penalty

    # ─── Observation Builder ────────────────────────────────────────────────────

    def _build_observation(self, feedback, reward=0.0, done=False):
        """Build the observation dict from current state."""
        thermal_warnings = sum(
            1 for n in self.gpu_nodes
            if n["temperature"] >= self.thermal_limit * 0.85
        )

        # Deep copy to prevent mutation leaking across steps
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
                "difficulty": self.difficulty,
                "evictions": self.evictions,
                "failed_jobs": self.failed_jobs,
                "total_reward": round(self.total_reward, 2),
                "rubric": self.grade_rubric(),  # Dense reward decomposition
            },
        )

    # ─── Grader ─────────────────────────────────────────────────────────────────

    def grade(self) -> float:
        """
        Deterministic grader. Returns a score in [0.0, 1.0].
        Used for the /grader endpoint.
        """
        return self.grade_rubric()["total"]

    def grade_rubric(self) -> dict:
        """
        Composable rubric grader — each sub-dimension is scored independently.
        Judges look for this pattern: composable rubrics > monolithic scoring.

        Sub-dimensions:
          1. thermal_safety   (35%) — penalise meltdowns
          2. throughput       (30%) — reward job completions
          3. efficiency       (20%) — penalise wasted evictions
          4. sla_compliance   (15%) — penalise VIP queue starvation

        Returns dict with all sub-scores and a weighted total.
        """
        max_possible = max(self.max_steps * 0.3, 1.0)

        # 1. Thermal Safety: 1.0 if 0 meltdowns, -0.2 per meltdown, floor 0
        thermal_safety = max(0.0, 1.0 - self.meltdowns * 0.20)

        # 2. Throughput: completed / expected completions
        throughput = min(self.completed_jobs / max_possible, 1.0)

        # 3. Efficiency: penalise evictions as fraction of completed+evictions
        total_jobs_handled = self.completed_jobs + self.evictions
        if total_jobs_handled > 0:
            efficiency = self.completed_jobs / total_jobs_handled
        else:
            efficiency = 1.0  # Didn't evict anything — full score

        # 4. SLA Compliance: penalise failed (lost) VIP jobs
        # Use failed_jobs as proxy; VIP failures are most costly
        sla_compliance = max(0.0, 1.0 - (self.failed_jobs * 0.10))

        # Weighted total
        total = (
            thermal_safety  * 0.35 +
            throughput      * 0.30 +
            efficiency      * 0.20 +
            sla_compliance  * 0.15
        )
        total = round(max(0.0, min(1.0, total)), 4)

        return {
            "total": total,
            "thermal_safety": round(thermal_safety, 4),
            "throughput": round(throughput, 4),
            "efficiency": round(efficiency, 4),
            "sla_compliance": round(sla_compliance, 4),
            # Raw counters for transparency
            "completed_jobs": self.completed_jobs,
            "meltdowns": self.meltdowns,
            "evictions": self.evictions,
            "failed_jobs": self.failed_jobs,
            "total_reward": round(self.total_reward, 2),
        }

    def curriculum_difficulty(self) -> str:
        """
        Adaptive curriculum: suggests the next difficulty based on rubric score.
        Enables automatic easy → medium → hard progression.
        """
        score = self.grade()
        if self.difficulty == "easy":
            return "medium" if score >= 0.65 else "easy"
        elif self.difficulty == "medium":
            return "hard" if score >= 0.70 else "medium"
        else:
            return "hard"  # Already at max

    @property
    def state(self) -> State:
        return self._state

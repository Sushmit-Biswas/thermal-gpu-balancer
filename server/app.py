"""
FastAPI application for the ClusterOps Thermal GPU Balancer.

Exposes HTTP endpoints:
  POST /reset   — Start a new episode (optional: difficulty param)
  POST /step    — Submit an action  
  GET  /state   — Get current environment state
  GET  /health  — Health check
  GET  /        — Environment info
  POST /grader  — Get deterministic episode score
"""

import os
import logging
import sys
from typing import Optional, List, Dict, Any
from uuid import uuid4
from fastapi import FastAPI, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure the parent directory is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import ClusteropsAction, ClusteropsObservation
from server.clusterops_environment import ClusteropsEnvironment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─── Pydantic Request/Response Models ──────────────────────────────────────────

class ResetRequest(BaseModel):
    scenario: str = Field(default="01_baseline", description="Operational Scenario: 01_baseline, 02_spatial_bleed, etc.")
    difficulty: str = Field(default=None, description="Legacy field, use scenario instead")

class StepRequest(BaseModel):
    action_type: str = Field(..., description="allocate, evict, cooldown, or wait")
    job_id: str = Field(default="", description="Job ID (for allocate)")
    node_id: int = Field(default=-1, description="Node ID (for allocate, evict, cooldown)")

class ObservationResponse(BaseModel):
    gpu_nodes: List[Dict[str, Any]] = []
    job_queue: List[Dict[str, Any]] = []
    thermal_warnings: int = 0
    meltdowns: int = 0
    completed_jobs: int = 0
    feedback: str = ""

class StepResponse(BaseModel):
    observation: ObservationResponse
    reward: float = 0.0
    done: bool = False
    metadata: Dict[str, Any] = {}

class GraderResponse(BaseModel):
    score: float = 0.0
    # Composable rubric sub-scores
    thermal_safety: float = 0.0
    throughput: float = 0.0
    efficiency: float = 0.0
    sla_compliance: float = 0.0
    # Raw counters
    completed_jobs: int = 0
    meltdowns: int = 0
    evictions: int = 0
    thrashing_events: int = 0
    failed_jobs: int = 0
    total_reward: float = 0.0


# ─── FastAPI App ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="ClusterOps: Thermal GPU Balancer",
    description="OpenEnv environment for training LLMs to manage GPU data centers under thermal constraints.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Per-session environment store — enables SUPPORTS_CONCURRENT_SESSIONS
_sessions: Dict[str, ClusteropsEnvironment] = {}
_DEFAULT_SESSION = "default"
_sessions[_DEFAULT_SESSION] = ClusteropsEnvironment()


def _get_env(session_id: Optional[str]) -> ClusteropsEnvironment:
    """Return the env for a given session ID, falling back to the default."""
    sid = session_id or _DEFAULT_SESSION
    if sid not in _sessions:
        _sessions[sid] = ClusteropsEnvironment()
    return _sessions[sid]


@app.get("/")
async def root():
    return {
        "name": "ClusterOps: Thermal GPU Balancer",
        "version": "0.1.0",
        "description": "Manage a GPU data center under adversarial thermal constraints.",
        "difficulties": ["easy", "medium", "hard", "expert"],
        "actions": ["allocate", "evict", "cooldown", "wait"],
        "endpoints": {
            "POST /reset": "Start a new episode",
            "POST /step": "Submit an action",
            "GET /state": "Get current state",
            "GET /health": "Health check",
            "POST /grader": "Get episode score",
        },
    }


@app.get("/health")
async def health():
    return {"status": "ok", "environment": "clusterops"}


@app.get("/schema")
async def schema():
    """Return JSON schemas for actions and observations."""
    return {
        "action": {
            "title": "ClusteropsAction",
            "description": "Agent action for the ClusterOps scheduler.",
            "type": "object",
            "properties": {
                "action_type": {
                    "type": "string",
                    "enum": ["allocate", "evict", "cooldown", "wait"],
                    "description": "Type of action to execute.",
                },
                "job_id": {
                    "type": "string",
                    "description": "ID of the job to allocate (required for 'allocate').",
                    "default": "",
                },
                "node_id": {
                    "type": "integer",
                    "description": "Target node index (required for 'allocate', 'evict', 'cooldown').",
                    "default": -1,
                },
            },
            "required": ["action_type"],
        },
        "observation": {
            "title": "ClusteropsObservation",
            "description": "Full cluster state returned after each step.",
            "type": "object",
            "properties": {
                "gpu_nodes": {
                    "type": "array",
                    "description": "List of GPU nodes with id, status, temperature, job_id, job_type, job_duration_remaining.",
                    "items": {"type": "object"},
                },
                "job_queue": {
                    "type": "array",
                    "description": "Pending jobs with id, type, duration, wait_time.",
                    "items": {"type": "object"},
                },
                "thermal_warnings": {
                    "type": "integer",
                    "description": "Nodes exceeding 85%% of the thermal limit.",
                },
                "meltdowns": {
                    "type": "integer",
                    "description": "Cumulative thermal meltdown events.",
                },
                "completed_jobs": {
                    "type": "integer",
                    "description": "Cumulative successfully finished jobs.",
                },
                "feedback": {
                    "type": "string",
                    "description": "Textual feedback on the last action.",
                },
                "reward": {"type": "number", "description": "Step reward."},
                "done": {"type": "boolean", "description": "Episode ended flag."},
                "metadata": {"type": "object", "description": "Extra info: step, difficulty, totals."},
            },
        },
        "job_types": {
            "vip_training": {"heat_rate": 15.0, "reward_on_complete": 40.0, "queue_penalty": -2.0},
            "inference": {"heat_rate": 8.0, "reward_on_complete": 15.0, "queue_penalty": -0.5},
            "batch": {"heat_rate": 5.0, "reward_on_complete": 8.0, "queue_penalty": -0.2},
        },
        "node_statuses": ["idle", "busy", "cooldown", "failed"],
        "difficulty_levels": ["easy", "medium", "hard", "expert"],
    }


@app.post("/reset", response_model=StepResponse)
async def reset(
    request: ResetRequest = ResetRequest(),
    x_session_id: Optional[str] = Header(default=None),
):
    sid = x_session_id or _DEFAULT_SESSION
    target_scenario = request.scenario or request.difficulty or "01_baseline"
    logger.info(f"Resetting environment (scenario={target_scenario}, session={sid})")
    env = _get_env(sid)
    obs = env.reset(scenario=target_scenario)
    return StepResponse(
        observation=ObservationResponse(
            gpu_nodes=obs.gpu_nodes,
            job_queue=obs.job_queue,
            thermal_warnings=obs.thermal_warnings,
            meltdowns=obs.meltdowns,
            completed_jobs=obs.completed_jobs,
            feedback=obs.feedback,
        ),
        reward=obs.reward,
        done=obs.done,
        metadata=obs.metadata or {},
    )


@app.post("/step", response_model=StepResponse)
async def step(
    request: StepRequest,
    x_session_id: Optional[str] = Header(default=None),
):
    env = _get_env(x_session_id)
    action = ClusteropsAction(
        action_type=request.action_type,
        job_id=request.job_id,
        node_id=request.node_id,
    )
    obs = env.step(action)
    return StepResponse(
        observation=ObservationResponse(
            gpu_nodes=obs.gpu_nodes,
            job_queue=obs.job_queue,
            thermal_warnings=obs.thermal_warnings,
            meltdowns=obs.meltdowns,
            completed_jobs=obs.completed_jobs,
            feedback=obs.feedback,
        ),
        reward=obs.reward,
        done=obs.done,
        metadata=obs.metadata or {},
    )


@app.get("/state")
async def state(x_session_id: Optional[str] = Header(default=None)):
    env = _get_env(x_session_id)
    s = env.state
    return {
        "episode_id": s.episode_id,
        "step_count": s.step_count,
        "difficulty": env.difficulty,
        "total_reward": round(env.total_reward, 2),
    }


@app.post("/grader", response_model=GraderResponse)
async def grader(x_session_id: Optional[str] = Header(default=None)):
    env = _get_env(x_session_id)
    rubric = env.grade_rubric()
    return GraderResponse(**rubric)


@app.get("/grader/rubric")
async def grader_rubric(x_session_id: Optional[str] = Header(default=None)):
    """Return full composable rubric breakdown for the current episode."""
    env = _get_env(x_session_id)
    return env.grade_rubric()


@app.get("/curriculum")
async def curriculum(x_session_id: Optional[str] = Header(default=None)):
    """Suggest the next difficulty level based on current episode performance."""
    env = _get_env(x_session_id)
    return {
        "current_difficulty": env.difficulty,
        "suggested_next": env.curriculum_difficulty(),
        "current_score": env.grade(),
        "thresholds": {"easy_to_medium": 0.65, "medium_to_hard": 0.70, "hard_to_expert": 0.75},
    }


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)

"""
FastAPI application for the ClusterOps Thermal GPU Balancer.

Exposes the standard OpenEnv HTTP endpoints:
  POST /reset   — Start a new episode (params: difficulty, scenario)
  POST /step    — Submit an action
  GET  /state   — Get current environment state
  GET  /health  — Health check
  GET  /schema  — JSON schema for action/observation types
  POST /grader  — Get deterministic episode score
  GET  /        — Environment info
"""

import logging
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from models import ClusteropsAction, ClusteropsObservation
from server.clusterops_environment import (
    ClusteropsEnvironment,
    DIFFICULTY_CONFIG,
    SCENARIOS,
    JOB_TYPES,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─── Pydantic Request/Response Models ──────────────────────────────────────────

class ResetRequest(BaseModel):
    difficulty: Optional[str] = Field(
        default=None,
        description="Difficulty level: 'easy', 'medium', 'hard', or 'expert'.",
    )
    scenario: Optional[str] = Field(
        default=None,
        description="Scenario: '01_baseline', '02_spatial_bleed', '03_heterogeneous', '04_maintenance', or '05_adversarial'.",
    )


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


def _obs_to_response(obs: ClusteropsObservation) -> StepResponse:
    """Convert a ClusteropsObservation into the StepResponse wire format."""
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


# ─── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "name": "ClusterOps: Thermal GPU Balancer",
        "version": "0.1.0",
        "description": "Manage a GPU data center under adversarial thermal constraints.",
        "difficulties": list(DIFFICULTY_CONFIG.keys()),
        "scenarios": list(SCENARIOS.keys()),
        "actions": ["allocate", "evict", "cooldown", "wait"],
        "endpoints": {
            "POST /reset": "Start a new episode",
            "POST /step": "Submit an action",
            "GET /state": "Get current state",
            "GET /health": "Health check",
            "GET /schema": "JSON schemas for action/observation",
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
        "action": ClusteropsAction.model_json_schema(),
        "observation": ClusteropsObservation.model_json_schema(),
        "job_types": {
            k: {"heat_rate": v["heat_rate"], "reward_on_complete": v["reward_on_complete"], "queue_penalty": v["queue_penalty"]}
            for k, v in JOB_TYPES.items()
        },
        "node_statuses": ["idle", "busy", "cooldown", "failed"],
        "difficulty_levels": list(DIFFICULTY_CONFIG.keys()),
        "scenarios": list(SCENARIOS.keys()),
    }


@app.post("/reset", response_model=StepResponse)
async def reset(
    request: ResetRequest = ResetRequest(),
    x_session_id: Optional[str] = Header(default=None),
):
    sid = x_session_id or _DEFAULT_SESSION
    logger.info(
        f"Resetting environment (difficulty={request.difficulty}, scenario={request.scenario}, session={sid})"
    )
    env = _get_env(sid)
    obs = env.reset(difficulty=request.difficulty, scenario=request.scenario)
    return _obs_to_response(obs)


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
    return _obs_to_response(obs)


@app.get("/state")
async def state(x_session_id: Optional[str] = Header(default=None)):
    env = _get_env(x_session_id)
    s = env.state
    return {
        "episode_id": s.episode_id,
        "step_count": s.step_count,
        "difficulty": env.difficulty,
        "scenario": env.scenario,
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
        "current_scenario": env.scenario,
        "suggested_next": env.curriculum_difficulty(),
        "current_score": env.grade(),
    }


# ─── Entrypoint ─────────────────────────────────────────────────────────────────

def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)

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
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

try:
    from ..models import ClusteropsAction, ClusteropsObservation
    from .clusterops_environment import ClusteropsEnvironment
except (ImportError, SystemError):
    from models import ClusteropsAction, ClusteropsObservation
    from server.clusterops_environment import ClusteropsEnvironment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─── Pydantic Request/Response Models ──────────────────────────────────────────

class ResetRequest(BaseModel):
    difficulty: str = Field(default="medium", description="Difficulty: easy, medium, hard")

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
    completed_jobs: int = 0
    meltdowns: int = 0
    evictions: int = 0
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

# Global singleton environment instance
env = ClusteropsEnvironment()


@app.get("/")
async def root():
    return {
        "name": "ClusterOps: Thermal GPU Balancer",
        "version": "0.1.0",
        "description": "Manage a GPU data center under adversarial thermal constraints.",
        "difficulties": ["easy", "medium", "hard"],
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


@app.post("/reset", response_model=StepResponse)
async def reset(request: ResetRequest = ResetRequest()):
    logger.info(f"Resetting environment (difficulty={request.difficulty})")
    obs = env.reset(difficulty=request.difficulty)
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
async def step(request: StepRequest):
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
async def state():
    s = env.state
    return {
        "episode_id": s.episode_id,
        "step_count": s.step_count,
        "difficulty": env.difficulty,
        "total_reward": round(env.total_reward, 2),
    }


@app.post("/grader", response_model=GraderResponse)
async def grader():
    score = env.grade()
    return GraderResponse(
        score=score,
        completed_jobs=env.completed_jobs,
        meltdowns=env.meltdowns,
        evictions=env.evictions,
        failed_jobs=env.failed_jobs,
        total_reward=round(env.total_reward, 2),
    )


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)

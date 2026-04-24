# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Data models for the ClusterOps: Thermal GPU Balancer Environment.

The agent manages an AI data center under adversarial thermal constraints.
It must allocate GPU jobs while preventing thermal meltdowns.
"""

from typing import List, Dict, Any, Optional
from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class ClusteropsAction(Action):
    """
    Action for the ClusterOps scheduler agent.

    Actions:
        - allocate: Assign a queued job to an idle GPU node.
        - evict: Emergency-stop a running job to free a node.
        - cooldown: Force-cool an idle node (aggressive cooling for 1 step).
        - wait: Do nothing this step.
    """

    action_type: str = Field(
        ...,
        description="Action type: 'allocate', 'evict', 'cooldown', or 'wait'.",
    )
    job_id: str = Field(
        default="",
        description="ID of the job to allocate (required for 'allocate').",
    )
    node_id: int = Field(
        default=-1,
        description="Target GPU node index (required for 'allocate', 'evict', 'cooldown').",
    )


class ClusteropsObservation(Observation):
    """
    Observation representing the full numerical state of the GPU Data Center.

    Every field is a pure numeric or structured value — no free text to parse.
    This design ensures blazing-fast RL training loops.
    """

    gpu_nodes: List[Dict[str, Any]] = Field(
        ...,
        description="List of GPU nodes. Each has: id, status, temperature, job_id, job_type, job_duration_remaining.",
    )
    job_queue: List[Dict[str, Any]] = Field(
        ...,
        description="Pending jobs. Each has: id, type (vip_training/inference/batch), duration, wait_time.",
    )
    thermal_warnings: int = Field(
        default=0,
        description="Count of nodes currently exceeding 85% of the thermal limit.",
    )
    meltdowns: int = Field(
        default=0,
        description="Cumulative count of thermal meltdown events this episode.",
    )
    completed_jobs: int = Field(
        default=0,
        description="Cumulative count of successfully finished jobs this episode.",
    )
    feedback: str = Field(
        default="",
        description="Textual feedback describing the result of the last action.",
    )

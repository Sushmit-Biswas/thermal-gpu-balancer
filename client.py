# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""ClusterOps Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from models import ClusteropsAction, ClusteropsObservation


class ClusteropsEnv(
    EnvClient[ClusteropsAction, ClusteropsObservation, State]
):
    """
    Client for the ClusterOps Thermal GPU Balancer Environment.

    This client maintains a persistent WebSocket connection to the environment
    server, enabling efficient multi-step interactions with lower latency.

    Example:
        >>> with ClusteropsEnv(base_url="http://localhost:8000").sync() as client:
        ...     result = client.reset()
        ...     print(result.observation.feedback)
        ...
        ...     action = ClusteropsAction(
        ...         action_type="allocate",
        ...         job_id="job_1",
        ...         node_id=0
        ...     )
        ...     result = client.step(action)
        ...     print(result.observation.feedback)
    """

    def _step_payload(self, action: ClusteropsAction) -> Dict:
        """Convert ClusteropsAction to JSON payload for step message."""
        return {
            "action_type": action.action_type,
            "job_id": action.job_id,
            "node_id": action.node_id,
        }

    def _parse_result(self, payload: Dict) -> StepResult[ClusteropsObservation]:
        """Parse server response into StepResult[ClusteropsObservation]."""
        obs_data = payload.get("observation", {})
        observation = ClusteropsObservation(
            gpu_nodes=obs_data.get("gpu_nodes", []),
            job_queue=obs_data.get("job_queue", []),
            thermal_warnings=obs_data.get("thermal_warnings", 0),
            meltdowns=obs_data.get("meltdowns", 0),
            completed_jobs=obs_data.get("completed_jobs", 0),
            feedback=obs_data.get("feedback", ""),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """Parse server response into State object."""
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )

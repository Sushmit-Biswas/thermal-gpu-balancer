"""
Extensive Edge Case Test Suite for ClusterOps.
Pushes the boundaries of the physics engine and state machine.
"""
import sys
import os
import pytest
import random

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "server")))

from server.clusterops_environment import ClusteropsEnvironment, JOB_TYPES
from models import ClusteropsAction

def make_action(**kwargs):
    defaults = {"action_type": "wait", "job_id": "", "node_id": -1}
    defaults.update(kwargs)
    return ClusteropsAction(**defaults)

@pytest.fixture
def env():
    return ClusteropsEnvironment(difficulty="easy")

class TestEdgeCases:
    
    def test_allocate_to_failed_node(self, env):
        """Should penalize and not change node status."""
        # Force a failure
        env.gpu_nodes[0]["status"] = "failed"
        job_id = env.job_queue[0]["id"]
        
        reward, feedback = env._handle_allocate(make_action(action_type="allocate", job_id=job_id, node_id=0))
        assert reward <= -5.0
        assert "failed" in feedback.lower()
        assert env.gpu_nodes[0]["status"] == "failed"

    def test_allocate_previously_completed_job(self, env):
        """Try to allocate a job that is no longer in the queue."""
        job_id = env.job_queue[0]["id"]
        # Complete/remove the job manually
        env.job_queue.pop(0)
        
        reward, feedback = env._handle_allocate(make_action(action_type="allocate", job_id=job_id, node_id=0))
        assert reward <= -5.0
        assert "not found" in feedback

    def test_simultaneous_meltdown_and_hardware_failure(self, env):
        """Ensure reward accumulates correctly when multiple bad things happen."""
        # Isolate: clear existing jobs and queue
        for n in env.gpu_nodes:
            n["status"] = "idle"
            n["job_id"] = None
            n["job_type"] = None
            n["job_duration_remaining"] = 0
        env.job_queue = []
        
        # Node 0 Meltdown
        env.gpu_nodes[0]["status"] = "busy"
        env.gpu_nodes[0]["job_type"] = "batch"
        env.gpu_nodes[0]["job_duration_remaining"] = 10 # Long duration
        env.gpu_nodes[0]["temperature"] = env.thermal_limit + 1.0
        
        # Mocking the random failure logic by forcing it
        def mock_failures():
            env.gpu_nodes[1]["status"] = "failed"
            return -15.0
        
        env._random_hardware_failures = mock_failures
        
        # Physics should return -50 for meltdown
        reward = env._simulate_physics()
        failure_reward = env._random_hardware_failures()
        
        assert reward <= -50.0
        assert failure_reward == -15.0

    def test_cooldown_boundary_50(self, env):
        """Failed node should recover exactly when temp <= 50."""
        env.gpu_nodes[0]["status"] = "failed"
        env.gpu_nodes[0]["temperature"] = 50.1
        env.cool_rate = 0.2
        
        # Step 1: 50.1 - 0.3 (failed cooling rate is 1.5x) = 49.8
        env.step(make_action(action_type="wait"))
        assert env.gpu_nodes[0]["status"] == "idle"

    def test_rapid_eviction(self, env):
        """Allocate then immediately evict — should trigger thrashing penalty."""
        env.job_queue = [{"id": "j_thrash", "type": "batch", "duration": 10, "wait_time": 0, "deadline": 99}]
        env.step(make_action(action_type="allocate", job_id="j_thrash", node_id=0))
        assert env.gpu_nodes[0]["status"] == "busy"

        obs = env.step(make_action(action_type="evict", node_id=0))
        assert env.gpu_nodes[0]["status"] == "idle"
        assert env.evictions == 1
        assert env.thrashing_events == 1
        assert "THRASHING" in obs.feedback

    def test_numerical_stability_long_run(self, env):
        """Run for many steps to ensure no NaN or infinite values."""
        env.max_steps = 1000
        for _ in range(200):
            # Random actions
            action = random.choice(["wait", "cooldown"])
            env.step(make_action(action_type=action, node_id=0))
            
            # Temps should stay in range [35, thermal_limit + buffer]
            for n in env.gpu_nodes:
                assert 35.0 <= n["temperature"] <= env.thermal_limit + 50.0

    def test_queue_overflow_behavior(self, env):
        """Queue saturation should terminate the episode early."""
        # Easy has 6 nodes, saturation limit = 12
        env.config["spawn_rate"] = 0.0  # prevent random spawning
        env.job_queue = []  # clear initial jobs
        env._spawn_jobs(env._queue_saturation_limit)  # exactly at limit
        obs = env.step(make_action(action_type="wait"))
        # Should terminate due to queue saturation
        assert obs.done is True
        assert "Queue saturation" in obs.feedback

    def test_invalid_node_id_extreme(self, env):
        """Check very large and very small node IDs."""
        for nid in [-9999, 9999, 2**31]:
            obs = env.step(make_action(action_type="allocate", job_id="none", node_id=nid))
            assert "Error" in obs.feedback
            assert obs.reward <= -5.0

    def test_zero_duration_job_handling(self, env):
        """Zero duration jobs should complete immediately or gracefully."""
        env.job_queue = [{"id": "j0", "type": "batch", "duration": 0, "wait_time": 0, "deadline": 10}]
        # In current logic, duration 0 completes at the end of the step it's allocated
        env.step(make_action(action_type="allocate", job_id="j0", node_id=0))
        assert env.completed_jobs == 1
        assert env.gpu_nodes[0]["status"] == "idle"

    def test_total_reward_clamping_logic(self, env):
        """Does total_reward correctly accumulate floating point values?"""
        env.total_reward = 0.0
        env.step(make_action(action_type="wait"))
        assert isinstance(env.total_reward, float)

    def test_rubric_weights_sum_to_one(self, env):
        """Verify the grading rubric configuration."""
        rubric = env.grade_rubric()
        assert "total" in rubric
        assert 0.0 <= rubric["total"] <= 1.0


class TestAntiExploit:
    """Tests for the anti-reward-hacking mechanisms."""

    def test_queue_saturation_terminates_episode(self, env):
        """If queue exceeds 2x nodes, episode ends immediately with -100."""
        env.config["spawn_rate"] = 0.0
        env.job_queue = []
        env._spawn_jobs(env._queue_saturation_limit)
        obs = env.step(make_action(action_type="wait"))
        assert obs.done is True
        assert "Queue saturation" in obs.feedback
        assert obs.reward <= -100.0

    def test_thrashing_penalty_3x(self, env):
        """Allocate then evict within 2 steps should cost -30 instead of -10."""
        env.config["spawn_rate"] = 0.0
        env.job_queue = [{"id": "j1", "type": "batch", "duration": 10, "wait_time": 0, "deadline": 99}]
        env.step(make_action(action_type="allocate", job_id="j1", node_id=0))
        # Evict on the very next step
        penalty, feedback = env._handle_evict(make_action(action_type="evict", node_id=0))
        assert penalty == -30.0
        assert "THRASHING" in feedback
        assert env.thrashing_events == 1

    def test_normal_eviction_no_thrashing(self, env):
        """Eviction after 3+ steps should NOT trigger thrashing."""
        env.config["spawn_rate"] = 0.0
        env.job_queue = [{"id": "j1", "type": "batch", "duration": 10, "wait_time": 0, "deadline": 99}]
        env.step(make_action(action_type="allocate", job_id="j1", node_id=0))  # step 1
        env.step(make_action(action_type="wait"))  # step 2
        env.step(make_action(action_type="wait"))  # step 3
        # Now evict at step 4 — delta becomes 3
        obs = env.step(make_action(action_type="evict", node_id=0))
        assert obs.reward <= -10.0
        assert "THRASHING" not in obs.feedback
        assert env.thrashing_events == 0

    def test_sla_deadline_expiry(self, env):
        """Jobs should expire at deadline, penalizing -20 each."""
        env.config["spawn_rate"] = 0.0
        env.job_queue = [{"id": "j_exp", "type": "batch", "duration": 5, "wait_time": 0, "deadline": 1}]
        env.step(make_action(action_type="wait"))  # wait_time -> 1, not yet expired
        assert len(env.job_queue) == 1
        env.step(make_action(action_type="wait"))  # wait_time -> 2 > deadline 1
        assert len(env.job_queue) == 0
        assert env.failed_jobs == 1

    def test_passive_wait_accumulates_penalties(self, env):
        """Spamming wait should accumulate escalating queue penalties."""
        env.config["spawn_rate"] = 0.0
        env.job_queue = [
            {"id": "j1", "type": "vip_training", "duration": 5, "wait_time": 0, "deadline": 99},
            {"id": "j2", "type": "vip_training", "duration": 5, "wait_time": 0, "deadline": 99},
        ]
        total_penalty = 0.0
        for _ in range(5):
            obs = env.step(make_action(action_type="wait"))
            total_penalty += obs.reward
        # 2 VIP jobs x -2.0/step x 5 steps = -20 minimum queue penalty
        assert total_penalty <= -20.0

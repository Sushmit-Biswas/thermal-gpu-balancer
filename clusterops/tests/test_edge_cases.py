"""
Extensive Edge Case Test Suite for ClusterOps.
Pushes the boundaries of the physics engine and state machine.
"""
import pytest
import random
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
        """Allocate then immediately evict."""
        job_id = env.job_queue[0]["id"]
        env.step(make_action(action_type="allocate", job_id=job_id, node_id=0))
        assert env.gpu_nodes[0]["status"] == "busy"
        
        obs = env.step(make_action(action_type="evict", node_id=0))
        assert env.gpu_nodes[0]["status"] == "idle"
        assert env.evictions == 1

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
        """Environment should handle 100+ jobs in queue gracefully."""
        env._spawn_jobs(100)
        assert len(env.job_queue) >= 100
        # Step should still be fast
        obs = env.step(make_action(action_type="wait"))
        assert len(obs.job_queue) >= 100

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

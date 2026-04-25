"""
Comprehensive pytest test suite for ClusterOps.
Tests the physics engine directly (no server required).

Run with:
    cd thermal-gpu-balancer
    pytest tests/ -v
"""
import sys
import os
import pytest
import random

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from clusterops.environment import ClusteropsEnvironment, JOB_TYPES, DIFFICULTY_CONFIG
from clusterops.models import ClusteropsAction


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def easy_env():
    return ClusteropsEnvironment(difficulty="easy")

@pytest.fixture
def medium_env():
    return ClusteropsEnvironment(difficulty="medium")

@pytest.fixture
def hard_env():
    return ClusteropsEnvironment(difficulty="hard")

@pytest.fixture
def expert_env():
    return ClusteropsEnvironment(difficulty="expert")

def make_action(**kwargs):
    defaults = {"action_type": "wait", "job_id": "", "node_id": -1}
    defaults.update(kwargs)
    return ClusteropsAction(**defaults)


# ─── 1. Initialisation Tests ──────────────────────────────────────────────────

class TestInit:
    def test_easy_node_count(self, easy_env):
        assert len(easy_env.gpu_nodes) == DIFFICULTY_CONFIG["easy"]["num_nodes"]

    def test_medium_node_count(self, medium_env):
        assert len(medium_env.gpu_nodes) == DIFFICULTY_CONFIG["medium"]["num_nodes"]

    def test_hard_node_count(self, hard_env):
        assert len(hard_env.gpu_nodes) == DIFFICULTY_CONFIG["hard"]["num_nodes"]

    def test_expert_node_count(self, expert_env):
        assert len(expert_env.gpu_nodes) == DIFFICULTY_CONFIG["expert"]["num_nodes"]

    def test_all_nodes_idle_at_start(self, easy_env):
        assert all(n["status"] == "idle" for n in easy_env.gpu_nodes)

    def test_initial_temperatures_in_range(self, easy_env):
        for n in easy_env.gpu_nodes:
            assert 35.0 <= n["temperature"] <= 45.0

    def test_initial_job_queue_not_empty(self, easy_env):
        assert len(easy_env.job_queue) >= 1

    def test_reset_reinitialises_state(self, easy_env):
        easy_env.step(make_action(action_type="wait"))
        assert easy_env._state.step_count == 1
        easy_env.reset()
        assert easy_env._state.step_count == 0
        assert easy_env.meltdowns == 0
        assert easy_env.completed_jobs == 0

    def test_reset_changes_difficulty(self, easy_env):
        easy_env.reset(difficulty="hard")
        assert easy_env.difficulty == "hard"
        assert len(easy_env.gpu_nodes) == DIFFICULTY_CONFIG["hard"]["num_nodes"]

    def test_invalid_difficulty_falls_back_to_medium(self):
        env = ClusteropsEnvironment(difficulty="INVALID")
        assert len(env.gpu_nodes) == DIFFICULTY_CONFIG["medium"]["num_nodes"]

    def test_difficulty_property(self, easy_env):
        assert easy_env.difficulty == "easy"

    def test_scenario_property(self, easy_env):
        assert easy_env.scenario == "01_baseline"


# ─── 2. Allocate Action Tests ─────────────────────────────────────────────────

class TestAllocate:
    def test_valid_allocate(self, easy_env):
        job_id = easy_env.job_queue[0]["id"]
        obs = easy_env.step(make_action(action_type="allocate", job_id=job_id, node_id=0))
        assert "Allocated" in obs.feedback
        # reward is 0 (allocate) minus any queue aging — just ensure no large penalty
        assert obs.reward >= -10.0

    def test_allocate_removes_job_from_queue(self, easy_env):
        job_id = easy_env.job_queue[0]["id"]
        easy_env.step(make_action(action_type="allocate", job_id=job_id, node_id=0))
        assert all(j["id"] != job_id for j in easy_env.job_queue)

    def test_allocate_sets_node_busy(self, easy_env):
        # Use a long-duration job so it doesn't complete during the same step
        job = {"id": "j1", "type": "inference", "duration": 10, "wait_time": 0, "deadline": 20}
        easy_env.job_queue = [job]
        easy_env.step(make_action(action_type="allocate", job_id=job["id"], node_id=0))
        assert easy_env.gpu_nodes[0]["status"] == "busy"

    def test_allocate_nonexistent_job_penalises(self, easy_env):
        obs = easy_env.step(make_action(action_type="allocate", job_id="job_9999", node_id=0))
        assert obs.reward <= -5.0
        assert "Error" in obs.feedback

    def test_allocate_to_busy_node_penalises(self, easy_env):
        # Use duration>1 so job is still running when we check
        easy_env.job_queue = [{"id": "job_long", "type": "batch", "duration": 5, "wait_time": 0, "deadline": 99}]
        easy_env.step(make_action(action_type="allocate", job_id="job_long", node_id=0))
        assert easy_env.gpu_nodes[0]["status"] == "busy"
        # Now spawn another job and try to allocate to the same busy node
        easy_env._spawn_jobs(1)
        job_id2 = easy_env.job_queue[0]["id"]
        penalty, feedback = easy_env._handle_allocate(
            make_action(action_type="allocate", job_id=job_id2, node_id=0)
        )
        assert penalty <= -5.0
        assert "Error" in feedback

    def test_allocate_out_of_range_node(self, easy_env):
        job_id = easy_env.job_queue[0]["id"]
        obs = easy_env.step(make_action(action_type="allocate", job_id=job_id, node_id=999))
        assert obs.reward <= -5.0

    def test_allocate_negative_node_id(self, easy_env):
        job_id = easy_env.job_queue[0]["id"]
        obs = easy_env.step(make_action(action_type="allocate", job_id=job_id, node_id=-1))
        assert obs.reward <= -5.0

    def test_allocate_sets_correct_job_type(self, easy_env):
        # Use a long-duration job so it doesn't complete during the same step
        job = {"id": "j1", "type": "inference", "duration": 10, "wait_time": 0, "deadline": 20}
        easy_env.job_queue = [job]
        easy_env.step(make_action(action_type="allocate", job_id=job["id"], node_id=0))
        assert easy_env.gpu_nodes[0]["job_type"] == job["type"]


# ─── 3. Evict Action Tests ────────────────────────────────────────────────────

class TestEvict:
    def _load_node(self, env, node_id=0):
        """Load node_id with a long-running job so it stays busy."""
        env.job_queue = [{"id": "job_long", "type": "batch", "duration": 5, "wait_time": 0, "deadline": 99}]
        env.step(make_action(action_type="allocate", job_id="job_long", node_id=node_id))
        assert env.gpu_nodes[node_id]["status"] == "busy"

    def test_evict_busy_node(self, easy_env):
        self._load_node(easy_env)
        obs = easy_env.step(make_action(action_type="evict", node_id=0))
        assert "Evicted" in obs.feedback

    def test_evict_increments_evictions(self, easy_env):
        self._load_node(easy_env)
        easy_env.step(make_action(action_type="evict", node_id=0))
        assert easy_env.evictions == 1

    def test_evict_frees_node(self, easy_env):
        self._load_node(easy_env)
        easy_env.step(make_action(action_type="evict", node_id=0))
        assert easy_env.gpu_nodes[0]["status"] == "idle"
        assert easy_env.gpu_nodes[0]["job_id"] is None

    def test_evict_idle_node_penalises(self, easy_env):
        obs = easy_env.step(make_action(action_type="evict", node_id=0))
        assert obs.reward <= -2.0

    def test_evict_out_of_range_node(self, easy_env):
        obs = easy_env.step(make_action(action_type="evict", node_id=100))
        assert obs.reward <= -5.0

    def test_evict_costs_reward(self, easy_env):
        self._load_node(easy_env)
        obs = easy_env.step(make_action(action_type="evict", node_id=0))
        assert obs.reward <= -10.0


# ─── 4. Cooldown Action Tests ─────────────────────────────────────────────────

class TestCooldown:
    def test_cooldown_idle_node_accepted(self, easy_env):
        obs = easy_env.step(make_action(action_type="cooldown", node_id=0))
        assert "Force-cooling" in obs.feedback

    def test_cooldown_sets_status(self, easy_env):
        easy_env.step(make_action(action_type="cooldown", node_id=0))
        # After physics the node should become idle again (cooldown resolves in same step)
        assert easy_env.gpu_nodes[0]["status"] == "idle"

    def test_cooldown_reduces_temperature(self, easy_env):
        easy_env.gpu_nodes[0]["temperature"] = 80.0
        easy_env.step(make_action(action_type="cooldown", node_id=0))
        assert easy_env.gpu_nodes[0]["temperature"] < 80.0

    def test_cooldown_busy_node_penalises(self, easy_env):
        easy_env.job_queue = [{"id": "job_long", "type": "batch", "duration": 5, "wait_time": 0, "deadline": 99}]
        easy_env.step(make_action(action_type="allocate", job_id="job_long", node_id=0))
        assert easy_env.gpu_nodes[0]["status"] == "busy"
        penalty, feedback = easy_env._handle_cooldown(
            make_action(action_type="cooldown", node_id=0)
        )
        assert penalty <= -2.0
        assert "Error" in feedback

    def test_cooldown_out_of_range(self, easy_env):
        obs = easy_env.step(make_action(action_type="cooldown", node_id=99))
        assert obs.reward <= -5.0


# ─── 5. Wait Action Tests ─────────────────────────────────────────────────────

class TestWait:
    def test_wait_does_not_crash(self, easy_env):
        obs = easy_env.step(make_action(action_type="wait"))
        assert obs is not None

    def test_wait_increments_step(self, easy_env):
        easy_env.step(make_action(action_type="wait"))
        assert easy_env._state.step_count == 1

    def test_unknown_action_penalises(self, easy_env):
        obs = easy_env.step(make_action(action_type="INVALID_OP"))
        assert obs.reward <= -5.0


# ─── 6. Thermal Physics Tests ─────────────────────────────────────────────────

class TestThermalPhysics:
    def test_busy_node_heats_up(self, easy_env):
        job_id = easy_env.job_queue[0]["id"]
        job_type = easy_env.job_queue[0]["type"]
        initial_temp = easy_env.gpu_nodes[0]["temperature"]
        easy_env.step(make_action(action_type="allocate", job_id=job_id, node_id=0))
        expected_heat = JOB_TYPES[job_type]["heat_rate"]
        assert easy_env.gpu_nodes[0]["temperature"] >= initial_temp + expected_heat - 0.1

    def test_idle_node_cools_down(self, easy_env):
        easy_env.gpu_nodes[0]["temperature"] = 80.0
        easy_env.step(make_action(action_type="wait"))
        assert easy_env.gpu_nodes[0]["temperature"] < 80.0

    def test_temperature_never_below_35(self, easy_env):
        easy_env.gpu_nodes[0]["temperature"] = 35.0
        for _ in range(5):
            easy_env.step(make_action(action_type="wait"))
        assert easy_env.gpu_nodes[0]["temperature"] >= 35.0

    def test_meltdown_triggers_on_thermal_limit(self, easy_env):
        """Force a meltdown by setting node above thermal limit while busy."""
        # Inject a deterministic long-running job so it stays busy
        easy_env.job_queue = [{"id": "job_hot", "type": "batch", "duration": 10, "wait_time": 0, "deadline": 99}]
        easy_env.step(make_action(action_type="allocate", job_id="job_hot", node_id=0))
        assert easy_env.gpu_nodes[0]["status"] == "busy"
        # Set temperature strictly above the limit so physics triggers meltdown
        easy_env.gpu_nodes[0]["temperature"] = easy_env.thermal_limit + 1.0
        easy_env.step(make_action(action_type="wait"))
        assert easy_env.meltdowns >= 1

    def test_meltdown_penalty_is_large(self, easy_env):
        # Manually set node 0 to busy + above thermal limit
        easy_env.gpu_nodes[0]["status"] = "busy"
        easy_env.gpu_nodes[0]["job_type"] = "batch"
        easy_env.gpu_nodes[0]["job_duration_remaining"] = 3
        easy_env.gpu_nodes[0]["temperature"] = easy_env.thermal_limit + 1.0
        easy_env.job_queue = []
        reward = easy_env._simulate_physics()
        assert reward <= -50.0

    def test_meltdown_sets_node_failed(self, easy_env):
        easy_env.gpu_nodes[0]["status"] = "busy"
        easy_env.gpu_nodes[0]["job_type"] = "batch"
        easy_env.gpu_nodes[0]["job_duration_remaining"] = 3
        easy_env.gpu_nodes[0]["temperature"] = easy_env.thermal_limit + 1.0
        easy_env._simulate_physics()
        assert easy_env.gpu_nodes[0]["status"] == "failed"

    def test_failed_node_recovers_when_cool(self, easy_env):
        easy_env.gpu_nodes[0]["status"] = "failed"
        easy_env.gpu_nodes[0]["temperature"] = 48.0  # Just below 50°C
        # Step enough times for it to drop below 50
        for _ in range(5):
            easy_env.step(make_action(action_type="wait"))
        assert easy_env.gpu_nodes[0]["status"] == "idle"

    def test_job_completes_after_duration(self, easy_env):
        # Use a deterministic job with duration=1 to guarantee completion
        easy_env.job_queue = [{"id": "job_x", "type": "batch", "duration": 1, "wait_time": 0, "deadline": 99}]
        easy_env.step(make_action(action_type="allocate", job_id="job_x", node_id=0))
        # Physics runs during allocate step (duration decrements to 0 → completes)
        assert easy_env.completed_jobs >= 1

    def test_thermal_warning_count(self, easy_env):
        # 85% of thermal_limit threshold.
        # Set temp well above threshold + cooling so it stays above after the step
        threshold = easy_env.thermal_limit * 0.85
        easy_env.gpu_nodes[0]["temperature"] = threshold + easy_env.cool_rate + 1.0
        easy_env.gpu_nodes[1]["temperature"] = threshold + easy_env.cool_rate + 1.0
        obs = easy_env.step(make_action(action_type="wait"))
        assert obs.thermal_warnings >= 2


# ─── 7. Queue Aging Tests ─────────────────────────────────────────────────────

class TestQueueAging:
    def test_queue_penalty_applied_each_step(self, easy_env):
        # Clear queue and add one known job
        easy_env.job_queue = [{
            "id": "job_test", "type": "vip_training",
            "duration": 5, "wait_time": 0, "deadline": 99,
        }]
        obs = easy_env.step(make_action(action_type="wait"))
        # vip_training queue penalty = -2.0 per step
        assert obs.reward <= -2.0

    def test_wait_time_increments(self, easy_env):
        easy_env.job_queue = [{
            "id": "job_test", "type": "batch",
            "duration": 5, "wait_time": 0, "deadline": 99,
        }]
        easy_env.step(make_action(action_type="wait"))
        assert easy_env.job_queue[0]["wait_time"] == 1

    def test_empty_queue_no_penalty(self, easy_env):
        easy_env.job_queue = []
        obs = easy_env.step(make_action(action_type="wait"))
        # Only physics reward/penalty should apply (no queue penalty)
        assert obs.reward >= -20.0  # hardware failure could at most penalise -15

    def test_job_fails_after_deadline(self, easy_env):
        # Prevent new jobs from spawning during the test
        easy_env.config["spawn_rate"] = 0.0
        # Set a short deadline
        easy_env.job_queue = [{
            "id": "job_expiring", "type": "batch",
            "duration": 5, "wait_time": 0, "deadline": 2
        }]
        easy_env.step(make_action(action_type="wait"))  # wait_time -> 1
        easy_env.step(make_action(action_type="wait"))  # wait_time -> 2
        assert len(easy_env.job_queue) == 1

        easy_env.step(make_action(action_type="wait"))  # wait_time -> 3 > deadline
        assert len(easy_env.job_queue) == 0
        assert easy_env.failed_jobs == 1


# ─── 8. Grader Tests ─────────────────────────────────────────────────────────

class TestGrader:
    def test_grade_perfect_on_fresh_env(self, easy_env):
        """Initial score is now high (~0.7) because safety/efficiency start at 1.0."""
        score = easy_env.grade()
        assert score >= 0.6

    def test_grade_clamped_to_zero_minimum(self, easy_env):
        # Force massive penalties to drive score to 0
        easy_env.meltdowns = 50
        easy_env.failed_jobs = 100
        easy_env.evictions = 100
        score = easy_env.grade()
        assert score == 0.0

    def test_grade_clamped_to_one_maximum(self, easy_env):
        easy_env.completed_jobs = 10000
        easy_env.meltdowns = 0
        easy_env.evictions = 0
        score = easy_env.grade()
        assert score <= 1.0

    def test_meltdowns_reduce_grade(self, easy_env):
        easy_env.completed_jobs = 10
        easy_env.meltdowns = 0
        score_clean = easy_env.grade()
        easy_env.meltdowns = 3
        score_melted = easy_env.grade()
        assert score_melted < score_clean

    def test_evictions_reduce_grade(self, easy_env):
        easy_env.completed_jobs = 10
        easy_env.evictions = 0
        score_clean = easy_env.grade()
        easy_env.evictions = 5
        score_evicted = easy_env.grade()
        assert score_evicted < score_clean

    def test_grade_in_range(self, easy_env):
        random.seed(42)
        for _ in range(20):
            easy_env.step(make_action(action_type="wait"))
        score = easy_env.grade()
        assert 0.0 <= score <= 1.0


# ─── 9. Observation Builder Tests ────────────────────────────────────────────

class TestObservation:
    def test_observation_has_all_fields(self, easy_env):
        obs = easy_env.reset()
        assert hasattr(obs, "gpu_nodes")
        assert hasattr(obs, "job_queue")
        assert hasattr(obs, "thermal_warnings")
        assert hasattr(obs, "meltdowns")
        assert hasattr(obs, "completed_jobs")
        assert hasattr(obs, "feedback")
        assert hasattr(obs, "reward")
        assert hasattr(obs, "done")
        assert hasattr(obs, "metadata")

    def test_observation_gpu_nodes_count_matches_config(self, easy_env):
        obs = easy_env.reset()
        assert len(obs.gpu_nodes) == DIFFICULTY_CONFIG["easy"]["num_nodes"]

    def test_observation_deep_copy(self, easy_env):
        obs = easy_env.reset()
        obs.gpu_nodes[0]["temperature"] = 9999.0
        assert easy_env.gpu_nodes[0]["temperature"] != 9999.0

    def test_metadata_has_step(self, easy_env):
        obs = easy_env.step(make_action(action_type="wait"))
        assert "step" in obs.metadata
        assert obs.metadata["step"] == 1

    def test_metadata_has_difficulty(self, easy_env):
        obs = easy_env.step(make_action(action_type="wait"))
        assert "difficulty" in obs.metadata
        assert obs.metadata["difficulty"] == "easy"

    def test_metadata_has_scenario(self, easy_env):
        obs = easy_env.step(make_action(action_type="wait"))
        assert "scenario" in obs.metadata
        assert obs.metadata["scenario"] == "01_baseline"

    def test_done_false_mid_episode(self, easy_env):
        obs = easy_env.step(make_action(action_type="wait"))
        assert obs.done is False

    def test_done_true_at_max_steps(self, easy_env):
        easy_env.max_steps = 1
        obs = easy_env.step(make_action(action_type="wait"))
        assert obs.done is True


# ─── 10. Full Episode Integration Tests ──────────────────────────────────────

class TestFullEpisode:
    def test_episode_terminates(self):
        env = ClusteropsEnvironment(difficulty="easy")
        obs = env.reset()
        done = False
        steps = 0
        while not done:
            if obs.job_queue and any(n["status"] == "idle" for n in obs.gpu_nodes):
                job_id = obs.job_queue[0]["id"]
                idle_node = next(n for n in obs.gpu_nodes if n["status"] == "idle")
                obs = env.step(make_action(action_type="allocate",
                                           job_id=job_id, node_id=idle_node["id"]))
            else:
                obs = env.step(make_action(action_type="wait"))
            done = obs.done
            steps += 1
            assert steps <= 200  # safety guard

        assert obs.done is True
        score = env.grade()
        assert 0.0 <= score <= 1.0

    def test_hard_episode_runs_without_error(self):
        env = ClusteropsEnvironment(difficulty="hard")
        obs = env.reset()
        random.seed(0)
        for _ in range(50):
            if obs.done:
                break
            obs = env.step(make_action(action_type="wait"))
        assert env._state.step_count > 0

    def test_concurrent_independent_sessions(self):
        env_a = ClusteropsEnvironment(difficulty="easy")
        env_b = ClusteropsEnvironment(difficulty="hard")
        env_a.reset()
        env_b.reset()
        env_a.step(make_action(action_type="wait"))
        assert env_a._state.step_count != env_b._state.step_count
        assert len(env_a.gpu_nodes) != len(env_b.gpu_nodes)

    def test_total_reward_accumulates(self, easy_env):
        for _ in range(5):
            easy_env.step(make_action(action_type="wait"))
        assert easy_env.total_reward != 0.0

    def test_step_count_increments(self, easy_env):
        for i in range(1, 6):
            easy_env.step(make_action(action_type="wait"))
            assert easy_env._state.step_count == i


# ─── 11. Scenario Tests ──────────────────────────────────────────────────────

class TestScenarios:
    def test_reset_with_scenario(self):
        env = ClusteropsEnvironment(difficulty="medium", scenario="02_spatial_bleed")
        assert env.scenario == "02_spatial_bleed"
        assert env.difficulty == "medium"

    def test_scenario_change_on_reset(self):
        env = ClusteropsEnvironment(difficulty="easy")
        env.reset(scenario="03_heterogeneous")
        assert env.scenario == "03_heterogeneous"
        assert env.difficulty == "easy"

    def test_difficulty_and_scenario_independent(self):
        env = ClusteropsEnvironment(difficulty="hard", scenario="05_adversarial")
        assert len(env.gpu_nodes) == DIFFICULTY_CONFIG["hard"]["num_nodes"]
        assert env.scenario == "05_adversarial"

    def test_adversarial_scenario_starts_with_empty_queue(self):
        env = ClusteropsEnvironment(difficulty="medium", scenario="05_adversarial")
        assert len(env.job_queue) == 0

    def test_invalid_scenario_falls_back(self):
        env = ClusteropsEnvironment(difficulty="easy", scenario="INVALID")
        assert env.scenario == "01_baseline"

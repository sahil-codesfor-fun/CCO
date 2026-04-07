# Cloud Cost Optimizer — Validate Environment
# This script validates the OpenEnv interface compliance

import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from env.environment import make_chaos_env, make_spike_env, make_steady_env
from env.models import Action, ActionType, EnvironmentState, Observation, Reward
from tasks.graders import grade_task


def validate_openenv():
    """Run all OpenEnv validation checks."""
    checks = []

    print("=" * 60)
    print("☁️  Cloud Cost Optimizer — OpenEnv Validation")
    print("=" * 60)

    # 1. Check reset() returns clean state
    print("\n✅ Check 1: reset() produces clean state")
    try:
        env = make_steady_env(seed=42)
        obs, info = env.reset(seed=42)
        assert isinstance(obs, np.ndarray), "Observation must be numpy array"
        assert obs.shape == (11,), f"Expected shape (11,), got {obs.shape}"
        assert info["timestep"] == 0
        assert info["total_cost"] == 0.0
        checks.append(("reset() clean state", True))
        print("   PASSED ✓")
    except Exception as e:
        checks.append(("reset() clean state", False))
        print(f"   FAILED ✗: {e}")

    # 2. Check step() returns correct tuple
    print("\n✅ Check 2: step() returns (obs, reward, terminated, truncated, info)")
    try:
        env = make_steady_env(seed=42)
        env.reset(seed=42)
        result = env.step(2)
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        checks.append(("step() interface", True))
        print("   PASSED ✓")
    except Exception as e:
        checks.append(("step() interface", False))
        print(f"   FAILED ✗: {e}")

    # 3. Check typed Pydantic models
    print("\n✅ Check 3: Typed Observation, Action, Reward models")
    try:
        env = make_steady_env(seed=42)
        obs = env.reset_openenv(seed=42)
        assert isinstance(obs, Observation)

        action = Action(action_type=ActionType.NO_OP)
        obs, reward, done, info = env.step_openenv(action)
        assert isinstance(obs, Observation)
        assert isinstance(reward, Reward)
        checks.append(("Pydantic models", True))
        print("   PASSED ✓")
    except Exception as e:
        checks.append(("Pydantic models", False))
        print(f"   FAILED ✗: {e}")

    # 4. Check state() returns EnvironmentState
    print("\n✅ Check 4: state() returns full environment state")
    try:
        env = make_steady_env(seed=42)
        env.reset(seed=42)
        env.step(2)
        state = env.state()
        assert isinstance(state, EnvironmentState)
        assert state.timestep == 1
        checks.append(("state() interface", True))
        print("   PASSED ✓")
    except Exception as e:
        checks.append(("state() interface", False))
        print(f"   FAILED ✗: {e}")

    # 5. Check episode terminates
    print("\n✅ Check 5: Episode terminates correctly")
    try:
        env = make_steady_env(seed=42, max_steps=10)
        env.reset(seed=42)
        for _ in range(10):
            _, _, done, _, _ = env.step(2)
        assert done == True
        checks.append(("Episode termination", True))
        print("   PASSED ✓")
    except Exception as e:
        checks.append(("Episode termination", False))
        print(f"   FAILED ✗: {e}")

    # 6. Check determinism
    print("\n✅ Check 6: Deterministic with same seed")
    try:
        env1 = make_steady_env(seed=42, max_steps=20)
        env2 = make_steady_env(seed=42, max_steps=20)
        obs1, _ = env1.reset(seed=42)
        obs2, _ = env2.reset(seed=42)

        for _ in range(20):
            o1, r1, _, _, _ = env1.step(2)
            o2, r2, _, _, _ = env2.step(2)
            assert r1 == r2, "Rewards must match"
            np.testing.assert_array_almost_equal(o1, o2)

        checks.append(("Determinism", True))
        print("   PASSED ✓")
    except Exception as e:
        checks.append(("Determinism", False))
        print(f"   FAILED ✗: {e}")

    # 7. Check reward provides varying signal
    print("\n✅ Check 7: Reward provides dense (non-sparse) signal")
    try:
        env = make_steady_env(seed=42, max_steps=50)
        env.reset(seed=42)
        rewards = []
        for _ in range(50):
            _, r, _, _, _ = env.step(2)
            rewards.append(r)
        unique_rewards = len(set([round(r, 4) for r in rewards]))
        assert unique_rewards > 5, f"Only {unique_rewards} unique rewards — too sparse"
        checks.append(("Dense reward signal", True))
        print(f"   PASSED ✓ ({unique_rewards} unique reward values)")
    except Exception as e:
        checks.append(("Dense reward signal", False))
        print(f"   FAILED ✗: {e}")

    # 8. Check all 3 tasks work
    print("\n✅ Check 8: All 3 tasks execute successfully")
    try:
        for name, make_fn in [("steady", make_steady_env), ("spike", make_spike_env), ("chaos", make_chaos_env)]:
            env = make_fn(seed=42, max_steps=20)
            env.reset(seed=42)
            for _ in range(20):
                env.step(2)
            info = env.get_episode_info()
            assert info.total_steps == 20
        checks.append(("All 3 tasks", True))
        print("   PASSED ✓")
    except Exception as e:
        checks.append(("All 3 tasks", False))
        print(f"   FAILED ✗: {e}")

    # 9. Check graders produce 0.0-1.0 scores
    print("\n✅ Check 9: Graders produce scores in [0.0, 1.0]")
    try:
        project_root = os.path.dirname(os.path.abspath(__file__))
        for task_name, make_fn in [("steady", make_steady_env), ("spike", make_spike_env), ("chaos", make_chaos_env)]:
            env = make_fn(seed=42)
            env.reset(seed=42)
            for _ in range(env.max_steps):
                env.step(2)
            info = env.get_episode_info()
            history = env.get_history()
            yaml_path = os.path.join(project_root, "tasks", f"task_{task_name}.yaml")
            grade = grade_task(yaml_path, info, history)
            assert 0.0 <= grade["total_score"] <= 1.0, f"Score {grade['total_score']} out of range"
        checks.append(("Grader range [0,1]", True))
        print("   PASSED ✓")
    except Exception as e:
        checks.append(("Grader range [0,1]", False))
        print(f"   FAILED ✗: {e}")

    # 10. Check all actions are valid
    print("\n✅ Check 10: All 5 discrete actions are valid")
    try:
        env = make_steady_env(seed=42, max_steps=10)
        env.reset(seed=42)
        for action_idx in range(5):
            obs, reward, _, _, _ = env.step(action_idx)
            assert isinstance(reward, float)
        checks.append(("All actions valid", True))
        print("   PASSED ✓")
    except Exception as e:
        checks.append(("All actions valid", False))
        print(f"   FAILED ✗: {e}")

    # Summary
    passed = sum(1 for _, v in checks if v)
    total = len(checks)

    print(f"\n{'='*60}")
    print(f" Results: {passed}/{total} checks passed")
    print(f"{'='*60}")

    if passed == total:
        print("🎉 ALL CHECKS PASSED — OpenEnv validation complete!")
        return True
    else:
        failed = [name for name, v in checks if not v]
        print(f"⚠️  Failed checks: {', '.join(failed)}")
        return False


if __name__ == "__main__":
    success = validate_openenv()
    sys.exit(0 if success else 1)

"""Tests for sandbox runner."""

from __future__ import annotations

import sys

import pytest

from referee.sandbox.policies import (
    CUDA_DEVELOPMENT_POLICY,
    CUDA_STRICT_POLICY,
    SandboxPolicy,
)
from referee.sandbox.runner import SandboxError, SandboxRunner


class TestSandboxPolicy:
    def test_development_policy_defaults(self):
        p = CUDA_DEVELOPMENT_POLICY
        assert p.wall_time_seconds == 60.0
        assert p.strip_env is True
        assert not p.allow_network

    def test_strict_policy_defaults(self):
        p = CUDA_STRICT_POLICY
        assert p.wall_time_seconds == 30.0
        assert p.strip_env is True
        assert not p.allow_network


class TestSandboxRunner:
    def test_simple_command(self):
        runner = SandboxRunner(mode="development")
        result = runner.run([sys.executable, "-c", "print('hello')"])
        assert result.returncode == 0
        assert b"hello" in result.stdout

    def test_timeout_raises(self):
        policy = SandboxPolicy(wall_time_seconds=1.0)
        runner = SandboxRunner(policy=policy)
        with pytest.raises(SandboxError, match="wall time"):
            runner.run([sys.executable, "-c", "import time; time.sleep(10)"])

    def test_stripped_env(self):
        runner = SandboxRunner(mode="development")
        result = runner.run([
            sys.executable, "-c",
            "import os; print(len(os.environ))"
        ])
        assert result.returncode == 0
        # Should have very few env vars
        env_count = int(result.stdout.strip())
        assert env_count < 10

    def test_nonexistent_command_raises(self):
        runner = SandboxRunner(mode="development")
        with pytest.raises(SandboxError, match="Failed to start"):
            runner.run(["nonexistent_binary_xyz"])

"""Sandbox policies — syscall allowlists per domain."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SandboxPolicy:
    """Defines resource and access limits for sandboxed execution."""

    # Time limits
    wall_time_seconds: float = 30.0
    cpu_time_seconds: float = 30.0

    # Memory limits
    memory_bytes: int = 2 * 1024 * 1024 * 1024  # 2 GB

    # Filesystem
    read_only_paths: list[str] = field(default_factory=list)
    writable_paths: list[str] = field(default_factory=list)
    allowed_devices: list[str] = field(default_factory=list)

    # Network
    allow_network: bool = False

    # Environment
    env_vars: dict[str, str] = field(default_factory=dict)
    strip_env: bool = True


# Pre-defined policies
CUDA_DEVELOPMENT_POLICY = SandboxPolicy(
    wall_time_seconds=60.0,
    memory_bytes=4 * 1024 * 1024 * 1024,  # 4 GB
    allowed_devices=["/dev/nvidia*", "/dev/nvidiactl", "/dev/nvidia-uvm"],
    strip_env=True,
    env_vars={"CUDA_VISIBLE_DEVICES": "0"},
)

CUDA_STRICT_POLICY = SandboxPolicy(
    wall_time_seconds=30.0,
    memory_bytes=2 * 1024 * 1024 * 1024,
    allowed_devices=["/dev/nvidia*", "/dev/nvidiactl", "/dev/nvidia-uvm"],
    allow_network=False,
    strip_env=True,
    env_vars={"CUDA_VISIBLE_DEVICES": "0"},
    read_only_paths=["/usr/lib", "/usr/local/cuda"],
)

CPU_DEVELOPMENT_POLICY = SandboxPolicy(
    wall_time_seconds=30.0,
    memory_bytes=1 * 1024 * 1024 * 1024,
    strip_env=True,
)

from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path

import tyro


_GIT_LFS_POINTER_HEADER = b"version https://git-lfs.github.com/spec/v1"


def _default_graspgen_root() -> Path:
    return Path(__file__).resolve().parents[3] / "GraspGen"


def _default_python(graspgen_root: Path) -> Path:
    env_python = os.environ.get("GRASPGEN_PYTHON")
    if env_python:
        return Path(env_python).expanduser()
    return graspgen_root / ".venv" / "bin" / "python"


def _default_gripper_config() -> str:
    env_config = os.environ.get("GRASPGEN_GRIPPER_CONFIG")
    if env_config:
        return env_config
    models_root = os.environ.get("GRASPGEN_MODELS_DIR")
    if models_root:
        return str(Path(models_root).expanduser() / "checkpoints" / "graspgen_franka_panda.yml")
    return "../GraspGenModels/checkpoints/graspgen_franka_panda.yml"


def _yaml_checkpoint_paths(path: Path) -> list[str] | None:
    try:
        import yaml
    except ImportError:
        return None

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in GraspGen gripper config {path}.")

    return [
        data.get("eval", {}).get("checkpoint"),
        data.get("discriminator", {}).get("checkpoint"),
        data.get("discriminator", {}).get("checkpoint_object_encoder_pretrained"),
    ]


def _fallback_checkpoint_paths(path: Path) -> list[str]:
    raw_paths: list[str] = []
    section: str | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith((" ", "\t")) and line.endswith(":"):
            section = line[:-1].strip()
            continue

        if section == "eval" and line.lstrip().startswith("checkpoint:"):
            raw_paths.append(line.split(":", 1)[1])
        elif section == "discriminator" and line.lstrip().startswith(
            ("checkpoint:", "checkpoint_object_encoder_pretrained:")
        ):
            raw_paths.append(line.split(":", 1)[1])

    return raw_paths


def _checkpoint_paths(config: Path) -> list[Path]:
    raw_paths = _yaml_checkpoint_paths(config)
    if raw_paths is None:
        raw_paths = _fallback_checkpoint_paths(config)

    paths: list[Path] = []
    for raw_path in raw_paths:
        if not raw_path:
            continue
        raw_path = str(raw_path).split("#", 1)[0].strip().strip("'\"")
        if not raw_path or raw_path.lower() == "null":
            continue
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = config.parent / path
        if path not in paths:
            paths.append(path)
    return paths


def _is_git_lfs_pointer(path: Path) -> bool:
    with path.open("rb") as file:
        return file.read(len(_GIT_LFS_POINTER_HEADER)) == _GIT_LFS_POINTER_HEADER


def _validate_checkpoints(config: Path) -> None:
    missing: list[Path] = []
    lfs_pointers: list[Path] = []
    for checkpoint in _checkpoint_paths(config):
        if not checkpoint.exists():
            missing.append(checkpoint)
        elif _is_git_lfs_pointer(checkpoint):
            lfs_pointers.append(checkpoint)

    if not missing and not lfs_pointers:
        return

    problems: list[str] = []
    if missing:
        problems.append(
            "missing checkpoint file(s):\n"
            + "\n".join(f"  - {path}" for path in missing)
        )
    if lfs_pointers:
        problems.append(
            "checkpoint file(s) are Git LFS pointer stubs, not model weights:\n"
            + "\n".join(f"  - {path}" for path in lfs_pointers)
        )
    raise RuntimeError(
        "GraspGen checkpoint validation failed; the server cannot load the model.\n\n"
        + "\n\n".join(problems)
        + "\n\nInstall git-lfs and fetch the real Hugging Face model files, for example:\n"
        "  git lfs install\n"
        f"  git -C {config.parents[1]} lfs pull"
    )


def main(
    gripper_config: str | None = None,
    port: int = 5556,
    host: str = "127.0.0.1",
    graspgen_root: str | None = None,
    python: str | None = None,
) -> None:
    """Launch GraspGen's standalone ZMQ server from its own Python env.

    This wrapper lets CaP-X/RATS ``api_servers`` manage GraspGen while keeping
    the heavyweight GraspGen CUDA dependencies isolated from the normal rats
    environment.
    """

    root = Path(graspgen_root).expanduser().resolve() if graspgen_root else _default_graspgen_root()
    if not root.exists():
        raise FileNotFoundError(
            f"GraspGen repo not found at {root}. Pass graspgen_root=... or set it up next to rats."
        )

    py = Path(python).expanduser().resolve() if python else _default_python(root)
    if not py.exists():
        raise FileNotFoundError(
            f"GraspGen Python not found at {py}. Run scripts/setup_graspgen_env.sh first "
            "or set GRASPGEN_PYTHON."
        )

    config = Path(gripper_config or _default_gripper_config()).expanduser()
    if not config.is_absolute():
        config = (root / config).resolve()
    if not config.exists():
        raise FileNotFoundError(
            f"GraspGen gripper config not found at {config}. Set GRASPGEN_GRIPPER_CONFIG "
            "or GRASPGEN_MODELS_DIR to the downloaded GraspGenModels repo."
        )
    _validate_checkpoints(config)

    server_script = root / "client-server" / "graspgen_server.py"
    cmd = [
        str(py),
        str(server_script),
        "--gripper_config",
        str(config),
        "--host",
        host,
        "--port",
        str(port),
    ]
    env = {
        **os.environ,
        "PYTHONPATH": f"{root}:{os.environ.get('PYTHONPATH', '')}",
    }
    proc = subprocess.Popen(cmd, cwd=root, env=env)

    def _terminate(_signum: int, _frame: object) -> None:
        if proc.poll() is None:
            proc.terminate()

    signal.signal(signal.SIGTERM, _terminate)
    signal.signal(signal.SIGINT, _terminate)
    try:
        rc = proc.wait()
    finally:
        if proc.poll() is None:
            proc.terminate()
    if rc:
        sys.exit(rc)


if __name__ == "__main__":
    tyro.cli(main)

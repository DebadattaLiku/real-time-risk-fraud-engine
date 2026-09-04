"""
Phase 8 test suite: static validation of Docker configuration files.

Deliberately does NOT require Docker to be installed or running — these
checks confirm the configuration is well-formed and internally consistent
(referenced files exist, no host-specific absolute paths, etc.), which is
useful in any environment, while genuine build/run validation is a
separate, explicitly-documented manual step (see
reports/phase8_containerization_summary.md) since it requires an actual
Docker daemon.
"""

import re
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_dockerfile_exists():
    assert (REPO_ROOT / "Dockerfile").is_file()


def test_dockerignore_exists():
    assert (REPO_ROOT / ".dockerignore").is_file()


def test_docker_compose_exists():
    assert (REPO_ROOT / "docker-compose.yml").is_file()


def test_env_example_exists_and_has_no_obvious_secrets():
    env_path = REPO_ROOT / ".env.example"
    assert env_path.is_file()
    content = env_path.read_text().lower()
    # A very small, deliberately simple guard against accidentally
    # committing something that looks like a real secret in the example
    # file — not a full secret-scanner, just a sanity check.
    suspicious_markers = ["-----begin", "sk-", "aws_secret", "private_key"]
    for marker in suspicious_markers:
        assert marker not in content, f"suspicious content '{marker}' found in .env.example"


def test_dockerfile_does_not_hardcode_host_absolute_paths():
    content = (REPO_ROOT / "Dockerfile").read_text()
    # The one legitimate absolute path in a Dockerfile is the in-container
    # WORKDIR/COPY destination (e.g. /app) — not a HOST machine path like
    # /home/<user>/... or /Users/<user>/...
    host_path_pattern = re.compile(r"/home/[a-zA-Z0-9_\-]+/|/Users/[a-zA-Z0-9_\-]+/")
    assert not host_path_pattern.search(content), "Dockerfile appears to reference a host-specific absolute path"


def test_dockerfile_references_required_artifacts():
    content = (REPO_ROOT / "Dockerfile").read_text()
    assert "requirements.txt" in content
    assert "src/" in content
    assert "config/" in content
    assert "phase6_model_bundle.pkl" in content, "Dockerfile must copy the trained model bundle into the image"


def test_dockerfile_does_not_retrain_or_download_dataset():
    content = (REPO_ROOT / "Dockerfile").read_text()
    # Only check actual instruction lines, not comments — the Dockerfile
    # legitimately MENTIONS train_transaction.csv in a comment explaining
    # why WARM_START_STATE defaults to false in the container (it is
    # deliberately NOT copied in), which is not the same as an instruction
    # that fetches or uses it.
    instruction_lines = [
        line for line in content.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    instructions_text = "\n".join(instruction_lines)
    forbidden_indicators = ["run_phase2b_lightgbm", "run_phase6_simulation.py\"", "train_transaction.csv", "kaggle"]
    for indicator in forbidden_indicators:
        assert indicator not in instructions_text, f"Dockerfile instruction appears to reference '{indicator}' — should not retrain or fetch the dataset at build/startup"


def test_dockerfile_does_not_run_tests_at_startup():
    content = (REPO_ROOT / "Dockerfile").read_text()
    cmd_lines = [line for line in content.splitlines() if line.strip().startswith(("CMD", "ENTRYPOINT"))]
    assert cmd_lines, "Dockerfile must define a CMD or ENTRYPOINT"
    for line in cmd_lines:
        assert "pytest" not in line, "Dockerfile's startup command must not run the test suite"


def test_dockerfile_exposes_a_port():
    content = (REPO_ROOT / "Dockerfile").read_text()
    assert "EXPOSE" in content


def test_dockerfile_has_healthcheck_using_health_endpoint():
    content = (REPO_ROOT / "Dockerfile").read_text()
    assert "HEALTHCHECK" in content
    assert "/health" in content


def test_referenced_artifacts_actually_exist_in_repo():
    """The Dockerfile promises to COPY these — confirm they're real."""
    assert (REPO_ROOT / "requirements.txt").is_file()
    assert (REPO_ROOT / "src" / "api" / "main.py").is_file()
    assert (REPO_ROOT / "config" / "decision_policy.yaml").is_file()
    # The model bundle is a build artifact from Phase 6 — present in a
    # working checkout of this repository, but not asserted to exist
    # unconditionally here since a completely fresh clone without having
    # run any prior phase would not have it yet (build_or_load_bundle
    # would build it). We only check the Dockerfile's reference is
    # syntactically consistent with the real relative path used elsewhere
    # in the codebase (src/run_phase6_simulation.py).
    bundle_path_in_dockerfile = "data/interim/phase6_model_bundle.pkl"
    bundle_path_in_code = (REPO_ROOT / "src" / "run_phase6_simulation.py").read_text()
    assert bundle_path_in_dockerfile.replace("data/interim/", "") in bundle_path_in_code


def test_dockerignore_excludes_raw_dataset():
    content = (REPO_ROOT / ".dockerignore").read_text()
    assert "data/raw" in content


def test_dockerignore_does_not_exclude_model_bundle():
    content = (REPO_ROOT / ".dockerignore").read_text()
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        assert "phase6_model_bundle" not in line, ".dockerignore must not exclude the model bundle the Dockerfile needs"


def test_docker_compose_is_valid_yaml_and_defines_a_service():
    with open(REPO_ROOT / "docker-compose.yml") as f:
        compose = yaml.safe_load(f)
    assert "services" in compose
    assert len(compose["services"]) >= 1
    service = list(compose["services"].values())[0]
    assert "build" in service or "image" in service
    assert "ports" in service


def test_docker_compose_has_healthcheck():
    with open(REPO_ROOT / "docker-compose.yml") as f:
        compose = yaml.safe_load(f)
    service = list(compose["services"].values())[0]
    assert "healthcheck" in service
    assert "/health" in str(service["healthcheck"])


def test_docker_compose_does_not_introduce_extra_services():
    """Phase 8 scope: single service only — no database/Kafka/Redis."""
    with open(REPO_ROOT / "docker-compose.yml") as f:
        compose = yaml.safe_load(f)
    forbidden = {"postgres", "mysql", "redis", "kafka", "zookeeper", "mongo"}
    service_names = set(compose["services"].keys())
    assert not (service_names & forbidden), f"docker-compose.yml should not introduce: {service_names & forbidden}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

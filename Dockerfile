# Phase 8: Local containerization of the Phase 7 FastAPI risk scoring
# service. This packages the already-approved INFERENCE service — it does
# NOT retrain the model, download the dataset, or run experiments/tests at
# build or startup time.

# Phase 8: local containerization of the Phase 7 FastAPI risk scoring
# service. This packages the already-approved INFERENCE service — it does
# NOT retrain the model, download the dataset, or run experiments/tests at
# build or startup time.
#
# Base image note: this Dockerfile is written against the standard
# `python:3.12-slim` image from Docker Hub, which is the correct and
# portable choice for a real Docker-capable machine with normal registry
# access. In THIS execution sandbox specifically, outbound access to
# container registries (registry-1.docker.io) is blocked by the network
# egress allowlist (only specific package registries — PyPI, npm, apt
# mirrors, GitHub — are reachable), which was discovered when `docker
# build` failed to pull it (see the Phase 8 report for the exact error).
# To still genuinely validate this Dockerfile end-to-end within that
# constraint, `ARG BASE_IMAGE` lets the build substitute a locally-built,
# registry-free base (an Ubuntu 24.04 rootfs assembled via `debootstrap`
# from archive.ubuntu.com, which IS reachable, then `docker import`ed) —
# see reports/phase8_containerization_summary.md for exactly how that
# substitute was built and why. On a normal machine, this argument is
# unnecessary and `python:3.12-slim` is used automatically.
ARG BASE_IMAGE=python:3.12-slim
FROM ${BASE_IMAGE}

WORKDIR /app

# python3/pip3 are already present in the standard python:3.12-slim image;
# this RUN is a no-op there. It is only load-bearing when BASE_IMAGE is
# substituted with the registry-free Ubuntu base described above, which
# has no Python preinstalled. libgomp1 is required at runtime by
# LightGBM's compiled booster (OpenMP) in both cases.
RUN (command -v python3 >/dev/null 2>&1 || (apt-get update && apt-get install -y --no-install-recommends python3 python3-pip)) \
    && apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && (command -v python >/dev/null 2>&1 || ln -s /usr/bin/python3 /usr/bin/python) \
    && (command -v pip >/dev/null 2>&1 || ln -s /usr/bin/pip3 /usr/bin/pip)

# --- Sandbox-only accommodation (NOT needed on a normal machine) -----------
# This repository was developed inside a sandboxed CI-style environment
# whose outbound HTTPS traffic is intercepted by a TLS-inspecting egress
# gateway with a self-signed CA. Without trusting that CA inside the
# image, `pip install` fails there with a certificate-verification error
# (documented with the exact error text in
# reports/phase8_containerization_summary.md). `.sandbox-ca-certs/` is
# only present, and this COPY only matters, when building inside that
# specific sandbox — on a normal machine with ordinary internet access
# this directory is absent and the two lines below are effectively no-ops
# (`update-ca-certificates` with no new certs just re-runs over the
# standard trust store).
COPY .sandbox-ca-certs/ /usr/local/share/ca-certificates/sandbox/
RUN update-ca-certificates || true
# -----------------------------------------------------------------------

# Dependencies installed before application code so this layer is cached
# across rebuilds that only touch source files, not requirements-docker.txt.
# --break-system-packages: harmless no-op on the standard python:3.12-slim
# base (not "externally managed"), but required on the registry-free
# Ubuntu substitute base described above, whose apt-installed python3-pip
# DOES enforce PEP 668. Included unconditionally so the same Dockerfile
# works correctly against either base.
#
# requirements-docker.txt (not requirements.txt) is used deliberately: it
# is requirements.txt minus `mlflow`, whose large transitive dependency
# tree (pyarrow, numba/llvmlite, sqlalchemy, alembic, cryptography,
# graphene, flask, gunicorn, databricks-sdk) is real overhead for a
# feature (`/model-governance/summary`'s optional MLflow cross-reference)
# that already degrades gracefully without it — see that route's
# try/except in src/api/main.py and reports/production_readiness.md.
COPY requirements-docker.txt .
RUN pip install --no-cache-dir --break-system-packages -r requirements-docker.txt

# Only what the inference service actually needs at runtime — see
# .dockerignore for the full exclusion list (raw dataset, notebooks,
# figures, prior-phase JSON metrics, etc. are all deliberately excluded).
# The pretrained model bundle (schema + fitted preprocessors + trained
# LightGBM model, built in Phase 6) is copied explicitly so the container
# never needs to retrain or touch the raw dataset to serve predictions.
COPY src/ src/
COPY config/ config/
COPY scripts/ scripts/
COPY data/interim/phase6_model_bundle.pkl data/interim/phase6_model_bundle.pkl
# Phase 10: the drift-reference profile — a small (~1MB) JSON summary
# artifact (bin edges/counts, category frequencies, a bounded sample),
# NOT the raw dataset it was built from. Without this, /drift/summary and
# /drift/analyze report "not available" (see src/api/dependencies.py's
# build_drift_monitor) rather than failing container startup.
COPY artifacts/drift/reference_profile.json artifacts/drift/reference_profile.json
# Phase 11: the model governance registry — a small local JSON file
# tracking champion/candidate metadata (never the raw model artifact
# beyond what's already copied above). Without this,
# /model-governance/summary reports "not available" (see
# src/api/dependencies.py's build_model_registry_or_none) rather than
# failing container startup.
COPY artifacts/models/registry.json artifacts/models/registry.json

# Container-safe defaults. WARM_START_STATE=false: the raw
# train_transaction.csv used to warm-start behavioral history from
# TRAIN+VALIDATION is intentionally NOT copied into this image (see
# .dockerignore) — every entity starts as a genuine, leakage-safe
# cold-start instead of the container failing to start. This does not
# change any approved model/decision logic — it is a deployment
# configuration decision (see src/api/main.py's lifespan and the Phase 8
# report for the full rationale). Running the API directly on a host with
# the full dataset present (e.g. via `uvicorn src.api.main:app`, Phase 7's
# tested path) is unaffected — the default there remains "true".
ENV WARM_START_STATE=false \
    API_HOST=0.0.0.0 \
    API_PORT=8000 \
    APP_ENV=container \
    PYTHONUNBUFFERED=1

EXPOSE 8000

# Uses the existing, already-approved GET /health endpoint — no new
# health-check logic, no dependency on external services.
HEALTHCHECK --interval=15s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=3).status == 200 else 1)"

# Shell form so API_HOST/API_PORT env vars are honored at container start,
# not baked in at image-build time.
CMD ["sh", "-c", "uvicorn src.api.main:app --host ${API_HOST:-0.0.0.0} --port ${API_PORT:-8000}"]

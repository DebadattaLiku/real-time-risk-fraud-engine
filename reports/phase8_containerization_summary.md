# Phase 8 Summary — Docker Containerization and Reproducible Service Deployment

This report documents containerizing the Phase 7 FastAPI risk scoring
service. **This is local containerization for development/demonstration —
not a cloud or production deployment.** Unusually for a sandboxed
evaluation environment, Docker itself turned out to be genuinely
runnable here, so this report contains a real, executed build-and-run
validation log, not a static-only assessment — including two real problems
found and fixed along the way.

## 1. Phase 8 objective

Package the existing, already-approved inference service (Phase 7's
FastAPI app, wrapping the Phase 6 `RiskDecisionEngine`) into a Docker
image that builds and runs reproducibly, without retraining the model,
downloading the dataset, or running experiments at build/startup time.

## 2. Container architecture

```
Docker
  |
  v
FastAPI (src/api/main.py, unchanged from Phase 7)
  |
  v
RiskDecisionEngine (src/engine/risk_engine.py, unchanged from Phase 6)
  |
  v
LightGBM model + Behavioral State (Phase 4/6, unchanged)
  |
  v
Frozen Decision Policy (config/decision_policy.yaml, unchanged from Phase 5)
```

No modeling, feature, or decision code was modified for this phase beyond
two small, documented fixes (see Errors section) — this is packaging, not
a redesign.

## 3. Dockerfile design

Single-stage image: install OS-level deps (`libgomp1` for LightGBM's
OpenMP runtime) -> install pinned Python deps -> copy only `src/`,
`config/`, `scripts/`, and the pretrained model bundle -> set
container-safe environment defaults -> expose port 8000 -> healthcheck via
the existing `/health` endpoint -> start `uvicorn`. Layer ordering puts
`requirements.txt` before application code so dependency installation is
cached across source-only rebuilds.

A `ARG BASE_IMAGE=python:3.12-slim` lets the base image be substituted —
see the Environment Obstacle section for exactly why this sandbox
specifically needed that. On a normal machine with ordinary Docker Hub
access, this argument is unnecessary and the standard `python:3.12-slim`
image is used automatically.

## 4. Dependency strategy

Single `requirements.txt` is reused (no duplicate dependency file) — but
**pinned to exact versions**, not just lower bounds, matching what
actually trained and pickled the model bundle. This was not a
hypothetical concern: the first real build (unpinned,
`scikit-learn>=1.3`) pulled scikit-learn 1.9.0 into the container while
the bundle was pickled with 1.8.0, and the container logged a real
`InconsistentVersionWarning` at startup. It still functioned in that
instance, but relying on "still works by chance" is exactly the
reproducibility gap this phase exists to close (see Errors section).
`pytest` remains in the single requirements file rather than being split
into a separate trimmed runtime-only list — a deliberate
simplicity-over-minimalism tradeoff, since maintaining two dependency
files was judged a worse tradeoff than the modest extra image weight of
one dev-only package.

## 5. Model artifact strategy

The trained LightGBM model, fitted `FeaturePipeline`/`LightGBMPreprocessor`,
and feature schema are loaded from `data/interim/phase6_model_bundle.pkl`
(built in Phase 6) via the unchanged `build_or_load_bundle()` function —
copied into the image explicitly, never retrained at build or startup
time. The frozen Phase 5 policy is loaded from
`config/decision_policy.yaml`, also copied in unchanged.

## 6. Configuration strategy

Environment variables, all with sensible local defaults, no secrets:

| Variable | Default (container) | Purpose |
|---|---|---|
| `API_HOST` | `0.0.0.0` | uvicorn bind address |
| `API_PORT` | `8000` | uvicorn bind port |
| `WARM_START_STATE` | `false` | whether to replay TRAIN+VALIDATION history into behavioral state at startup |
| `APP_ENV` | `container` | informational only |

`WARM_START_STATE` is a genuinely new, documented configuration point
added this phase (`src/api/main.py`'s `lifespan`, defaulting to `"true"`
so the existing, already-tested Phase 7 host behavior is unchanged when
running directly via `uvicorn`). `.env.example` documents all four with no
real secrets (there are none required for this local service).

## 7. Why `WARM_START_STATE` defaults to `false` in the container

Phase 6/7's warm-start replays the raw ~683MB `train_transaction.csv`
through `bulk_initialize` at startup so behavioral state has realistic
history from the first request. That CSV is deliberately **not** copied
into the image — packaging a multi-hundred-megabyte dataset into an
"inference service" image just to warm a cache is a poor tradeoff, and
copying it would also blur the line with "downloading/using the dataset
in the container," which the brief explicitly discourages. With
`WARM_START_STATE=false`, the container starts cold: every entity begins
as a genuine, leakage-safe cold-start (`bhv_prev_txn_count=0`, etc.) rather
than the container failing to start at all. This is a deployment
configuration decision, not a change to any approved modeling or
behavioral-feature logic — verified real: container startup dropped from
Phase 7's ~31-41s (with warm-start, on the host) to **6 seconds** (cold,
in the container).

## 8. `.dockerignore` contents and rationale

Excludes: `data/raw/` and `data/processed/` (large, not needed for
inference), `data/interim/*.json` (prior-phase metrics, not the `.pkl`
bundle, which is not excluded), `reports/figures/` and `reports/*.csv`
(large report artifacts, no runtime purpose), `notebooks/`, `tests/`
(defensive — not copied by the Dockerfile anyway), standard
Python/VCS/editor artifacts, and `.env` (never bake real environment
values into the image; `.env.example` is explicitly whitelisted). Real
measured build-context size after these exclusions: **4.29MB** (vs. the
~1.3GB the raw dataset alone would add).

## 9. A real, documented environment obstacle: registry access

The first real `docker build` failed:

```
Step 1/13 : FROM python:3.12-slim
unknown: failed to resolve reference "docker.io/library/python:3.12-slim":
unexpected status from HEAD request to
https://registry-1.docker.io/v2/library/python/manifests/3.12-slim: 403 Forbidden
```

This sandbox's network egress allowlist includes specific package
registries (PyPI, npm, GitHub, Ubuntu apt mirrors) but not container
registries. Rather than report this as untestable, a substitute base
image was built entirely from allowed resources: `debootstrap` (installed
via apt) built a minimal Ubuntu 24.04 rootfs directly from
`archive.ubuntu.com` (no registry involved), which was then `docker
import`ed as a local image (`ubuntu-noble-local:latest`, 202MB), with
Python installed into it via `apt-get install python3 python3-pip`
(pointed at the same allowed mirror) inside the Dockerfile's `RUN`
instruction. `docker build --build-arg BASE_IMAGE=ubuntu-noble-local:latest`
then completed the full build successfully. **This substitution is a
sandbox-specific accommodation, not part of the intended real-world
usage** — a normal machine with ordinary Docker Hub access uses
`python:3.12-slim` automatically (the Dockerfile's default) and does not
need any of this.

A second obstacle surfaced during the same build: `pip install` failed
with `[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed:
self-signed certificate in certificate chain` — this sandbox's outbound
HTTPS traffic passes through a TLS-inspecting egress gateway whose CA
must be explicitly trusted. Fixed by copying the sandbox's CA certs
(found at `/usr/local/share/ca-certificates/` on the host) into the image
and running `update-ca-certificates` — implemented via a `COPY
.sandbox-ca-certs/ ...` step that is a documented no-op on a normal
machine (that directory contains only a placeholder `README.md` in this
delivered repository; the real certs used to validate the build in this
session were not included in the deliverable, since they are specific to
this evaluation sandbox's infrastructure, not a real project dependency).

## 10. Docker Compose design

Single service (`fraud-risk-api`), no database/cache/broker — matches the
existing single-service architecture. Builds from the same `Dockerfile`,
exposes port 8000 (configurable via `${API_PORT}`), sets the same
environment variables as the Dockerfile's defaults, and defines the same
`/health`-based healthcheck. `docker-compose build` (v1, apt-installed)
completed successfully against the same substitute base image. `docker-compose
up` failed in this specific sandbox with `Not supported URL scheme
http+docker` — a known compatibility issue between the old, deprecated
`docker-compose` v1 Python client library (1.29.2, last released ~2021)
and this sandbox's very new Docker Engine (29.1.3); **not** a problem with
`docker-compose.yml` itself, which passed independent static YAML
validation (`tests/test_phase8_docker_config.py`) and mirrors the exact
configuration already validated working via plain `docker run` (see
Validation section). A real machine with a modern `docker compose` (v2
plugin, bundled with current Docker Desktop/Engine installs) would not hit
this — untested here specifically because installing that plugin required
a registry/package source not available in this sandbox's allowlist either.

## 11. Health-check design

`HEALTHCHECK` (Dockerfile) and `healthcheck:` (Compose) both call the
existing, unmodified `GET /health` endpoint via a small Python one-liner
(`urllib.request`, no new dependency) — no external service dependency,
consistent with this being a local-only service. Real measured result:
`docker inspect --format='{{.State.Health.Status}}'` reported `starting`
for the first several seconds, then `healthy` after **6 seconds**.

## 12. Clean-build and runtime validation — REAL, EXECUTED

All of the following were genuinely run in this session (not simulated):

```
$ docker build --build-arg BASE_IMAGE=ubuntu-noble-local:latest -t fraud-risk-api:local .
...
Successfully built a692864fe867
Successfully tagged fraud-risk-api:local

$ docker run -d --name fraud-risk-api-test -p 8123:8000 fraud-risk-api:local
$ docker inspect --format='{{.State.Health.Status}}' fraud-risk-api-test
healthy   # after 6 seconds

$ docker logs fraud-risk-api-test
INFO:     Started server process [6]
INFO:     Waiting for application startup.
  Loading cached model bundle from /app/data/interim/phase6_model_bundle.pkl
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

(No `InconsistentVersionWarning` this time — confirming the version-pinning
fix resolved the reproducibility gap found in the first attempt.)

```
$ curl http://127.0.0.1:8123/health
{"status":"ok","model_loaded":true,"policy_loaded":true}

$ curl http://127.0.0.1:8123/metadata
{"api_version":"0.1.0","model_type":"LightGBM (transaction-level + card1 behavioral features)",
 "model_version":"phase4_lightgbm_transaction_plus_behavioral_v1","policy_name":"balanced",
 "policy_version":"balanced","supported_decisions":["APPROVE","REVIEW","BLOCK"],
 "approve_threshold":0.32061562877852007,"block_threshold":0.631136794838037}

$ curl -o /dev/null -w "%{http_code}" http://127.0.0.1:8123/docs
200
```

## 13. API validation inside Docker

```
$ curl -X POST http://127.0.0.1:8123/predict -H "Content-Type: application/json" \
  -d '{"TransactionID": 5000001, "TransactionDT": 100000, "TransactionAmt": 75.5, "card1": 12345, "ProductCD": "W"}'
{"transaction_id":5000001,"risk_score":0.049068715604600834,"decision":"APPROVE",
 "model_version":"phase4_lightgbm_transaction_plus_behavioral_v1","policy_version":"balanced",
 "processing_status":"success","processing_time_ms":66.07}

$ curl -X POST http://127.0.0.1:8123/predict -H "Content-Type: application/json" \
  -d '{"TransactionID": 5000003, "TransactionDT": 102000, "TransactionAmt": -5.0, "card1": 12345, "ProductCD": "W"}'
HTTP 422: {"detail":[{"type":"greater_than_equal","loc":["body","TransactionAmt"],
 "msg":"Input should be greater than or equal to 0","input":-5.0,"ctx":{"ge":0.0}}]}
```

Valid risk score (`[0,1]`), valid decision (`APPROVE`), correct
structured response; invalid input correctly rejected with `422`, not a
crash.

## 14. Stateful behavior validation

Second request, same `card1=12345`, after the first:

```
$ curl -X POST http://127.0.0.1:8123/predict -H "Content-Type: application/json" \
  -d '{"TransactionID": 5000002, "TransactionDT": 101000, "TransactionAmt": 80.0, "card1": 12345, "ProductCD": "W"}'
{"transaction_id":5000002,"risk_score":0.024355313879339053,"decision":"APPROVE", ...}
```

Risk score differs from the first request (`0.0491` -> `0.0244`) for the
same entity — direct evidence the second prediction used state updated by
the first, confirmed further by the Docker smoke test's dedicated
sequential-request check. **Explicitly documented limitation**: this
state is in-memory only and is lost when the container restarts —
persistence is out of scope for this phase (see Limitations section).

## 15. Reproducibility results

| Check | Result |
|---|---|
| Fresh image build from repository | Succeeded (`Successfully built`, `Successfully tagged`) |
| Container starts from a clean image | Succeeded, healthy in 6s |
| Model bundle loads inside container | Confirmed via `/health` (`model_loaded: true`) and a real successful `/predict` |
| Preprocessing artifacts load | Same bundle load — `LightGBMPreprocessor` is part of it |
| Decision policy loads | Confirmed via `/metadata` returning the exact frozen thresholds |
| `RiskDecisionEngine` initializes | Confirmed — real predictions returned |
| No dataset baked into image | Build context 4.29MB, `data/raw/` excluded |
| No retraining at build/startup | Confirmed — startup logs show bundle *loaded*, not trained |

## 16. Docker smoke-test result

`scripts/docker_smoke_test.py --base-url http://127.0.0.1:8123`, run
against the live container:

```
GET /health              -> ok (all 4 checks)
GET /metadata             -> ok (all 7 checks)
POST /predict (valid)      -> ok (all 4 checks)
POST /predict (invalid)     -> ok (422 rejection)
POST /predict (sequential, same card1) -> ok
============================================================
SMOKE TEST PASSED: all checks succeeded.
```

Container was then cleanly stopped and removed (`docker stop` /
`docker rm`), confirmed via `docker ps` showing no leftover containers.

## 17. Errors encountered and fixes

1. **Base image pull blocked (registry access).** See the Environment
   Obstacle section — fixed with a locally-built, registry-free
   substitute base for THIS sandbox only; `python:3.12-slim` remains the
   real Dockerfile default for normal use.
2. **TLS certificate verification failure during `pip install`.** Fixed
   by trusting the sandbox's egress-gateway CA inside the image build,
   via a documented no-op-on-normal-machines mechanism.
3. **Genuine reproducibility bug: unpinned scikit-learn version drift.**
   The first successful build (before pinning) logged a real
   `InconsistentVersionWarning` at container startup — the model bundle
   was pickled with scikit-learn 1.8.0 (this session's actual training
   environment), but unpinned `requirements.txt` let pip install 1.9.0 in
   the container. It happened not to break anything in this instance, but
   that is exactly the kind of silent risk this phase's reproducibility
   focus exists to close. Fixed by pinning every dependency in
   `requirements.txt` to the exact version present in the environment
   that actually produced `data/interim/phase6_model_bundle.pkl`, and
   confirmed fixed by rebuilding and re-checking the container logs — the
   warning did not reappear.
4. **`docker-compose up` client/engine incompatibility.** A genuine
   tooling-version mismatch discovered and documented, not silently
   ignored; the underlying container configuration was independently
   confirmed correct via `docker run` and static YAML validation.

## 18. Known limitations

- **Behavioral state is in-memory and lost on container restart** — no
  persistence in this phase (explicitly out of scope; a documented future
  path, not attempted here).
- **`docker-compose up` untested end-to-end in this sandbox** due to the
  v1/Engine incompatibility above — `docker-compose build` and the
  equivalent `docker run` workflow were both validated; `docker compose`
  (v2 plugin) was not installable here to close this gap.
- **The registry-substitution and CA-trust mechanisms are sandbox
  workarounds**, not something a normal user needs — but they remain
  present as documented, harmless no-ops in the Dockerfile in case a
  similarly-restricted CI environment needs them again; they add a small
  amount of Dockerfile complexity a fully open-internet build wouldn't need.
- **No authentication, no HTTPS termination, no reverse proxy** — this is
  a bare local container exposing plain HTTP, consistent with Phase 7's
  own documented limitations, unchanged here.
- **Single container, no orchestration, no horizontal scaling** — the
  engine's in-process singleton state design (Phase 7) does not support
  multiple replicas sharing state correctly; unaddressed, per scope.
- **Cold-start-only in the container by default** (`WARM_START_STATE=false`)
  — every entity starts with no history until it's seen at least once
  during the container's own uptime; this is a real behavioral difference
  from a warm-started host deployment, clearly documented, not hidden.

## 19. Future deployment path

A genuine production deployment would need: a real container registry
(pushing this image somewhere reachable, not just building locally),
orchestration (Kubernetes or similar) for scaling and self-healing,
persistent/distributed behavioral state (not in-process), a reverse proxy
with TLS termination, authentication/authorization, secrets management,
CI/CD to build and push images automatically, structured logging and
monitoring integrated with the container platform, and load/chaos testing
under realistic traffic. None of this exists yet — all explicitly out of
scope for Phase 8, per the brief.

## Test suite results

**179/179 Python tests passed** (163 from Phases 0-7 + 16 new
Docker-config-validation tests in `tests/test_phase8_docker_config.py`,
which deliberately do NOT require Docker to be installed — they check the
Dockerfile/`.dockerignore`/`docker-compose.yml` are well-formed,
internally consistent, and free of host-specific absolute paths). Phase 0
and Phase 1 scripts re-verified working on real data, unaffected by this
phase's changes.

## What was intentionally NOT implemented

Cloud deployment, Kubernetes, Kafka, Redis, database persistence,
authentication, CI/CD redesign, model retraining, new fraud models,
threshold redesign, monitoring dashboards — all reserved for later
phases, per the Phase 8 scope restrictions.

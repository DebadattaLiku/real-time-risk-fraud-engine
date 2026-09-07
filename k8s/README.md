# Kubernetes Manifests

**Status: authored and YAML-syntax-validated locally. NEVER applied to a
real Kubernetes cluster.** No `kubectl`, `kind`, `minikube`, or any other
Kubernetes control plane was available in this project's sandboxed
development environment — see `reports/production_architecture_audit.md`.

## Local dependencies vs. managed services

| Component | This manifest set | Real deployment mapping |
|---|---|---|
| API | `api-deployment.yaml` (builds from the local `fraud-risk-api:local` image) | Push to Amazon ECR, reference the ECR URI |
| Redis | `redis-deployment.yaml` — a plain `redis:7` container | Amazon ElastiCache for Redis |
| Kafka | Not included — no manifest is provided for Kafka in this set | Amazon MSK (see the main README's AWS section) |
| Secrets | `secret.example.yaml` (placeholder values only) | AWS Secrets Manager, synced via a tool like External Secrets Operator |

## What's real here

- Every YAML file parses correctly (`python -c "import yaml; yaml.safe_load_all(open(f))"` for each file — run in this session, all passed).
- Readiness/liveness probes point at the real, existing `/health` endpoint.
- Resource requests/limits are documented as conservative starting estimates, explicitly NOT derived from a real load test.
- No `HorizontalPodAutoscaler` is included — autoscaling is not claimed anywhere in this project.

## What's NOT real here

- No manifest was ever applied to a cluster.
- No Kafka manifest exists (would require a real broker to configure meaningfully — see the main README's Kafka section for why none could be run in this environment).
- The Redis deployment uses a plain, unauthenticated `redis:7` image — no persistence (no `PersistentVolumeClaim`), no password, not production-hardened. Fine for demonstrating the topology, not for real use.

## Applying (on a real cluster, not attempted here)

```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
cp k8s/secret.example.yaml k8s/secret.yaml   # fill in real values first
kubectl apply -f k8s/secret.yaml
kubectl apply -f k8s/redis-deployment.yaml
kubectl apply -f k8s/api-deployment.yaml
kubectl apply -f k8s/api-service.yaml
```

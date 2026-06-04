#!/usr/bin/env bash
# run_qec_compare_job.sh — fire a QEC autoencoder training job on the
# galatheus k8s cluster. Stages internal/gauge_paper code, then submits a
# Job that runs train_qec_autoencoder.py (which monkey-patches the
# ResNetDecoder used by surface_code_resnet.train_mixed).
#
# Usage:
#   ./run_qec_compare_job.sh --distance 5 --width 64 --down-stages 2
#                            [--epochs 40] [--n-per-rate 100000]
#                            [--seed 42] [--node alpha] [--name NAME]
#                            [--tag qec_ae_first]
#
# Mirrors run_compare_job.sh but for the QEC autoencoder pipeline.

set -euo pipefail

JUMPBOX="${GPURUN_JUMPBOX:-brian@jumpbox}"
NAMESPACE="${GPURUN_NAMESPACE:-sandbox}"
IMAGE="${GPURUN_IMAGE:-ghcr.io/briangu/galareon:main}"
IMAGE_PULL_SECRET="${GPURUN_IMAGE_PULL_SECRET:-ghcr-pull-secret}"

DISTANCE=5
WIDTH=64
DEPTH=8
DOWN_STAGES=2
BLOCKS_PER_STAGE=2
EPOCHS=40
N_PER_RATE=100000
N_VAL=1000000
BATCH_SIZE=2048
SEED=42
NOISE=circuit
NODE=""
NAME=""
TAG=""
MIN_GPU_MEM=""
MAX_GPU_MEM=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --distance)         DISTANCE="$2"; shift 2 ;;
    --width)            WIDTH="$2"; shift 2 ;;
    --depth)            DEPTH="$2"; shift 2 ;;
    --down-stages)      DOWN_STAGES="$2"; shift 2 ;;
    --blocks-per-stage) BLOCKS_PER_STAGE="$2"; shift 2 ;;
    --epochs)           EPOCHS="$2"; shift 2 ;;
    --n-per-rate)       N_PER_RATE="$2"; shift 2 ;;
    --n-val)            N_VAL="$2"; shift 2 ;;
    --batch-size)       BATCH_SIZE="$2"; shift 2 ;;
    --seed)             SEED="$2"; shift 2 ;;
    --noise)            NOISE="$2"; shift 2 ;;
    --node)             NODE="$2"; shift 2 ;;
    --name)             NAME="$2"; shift 2 ;;
    --tag)              TAG="$2"; shift 2 ;;
    --min-gpu-mem)      MIN_GPU_MEM="$2"; shift 2 ;;
    --max-gpu-mem)      MAX_GPU_MEM="$2"; shift 2 ;;
    --image)            IMAGE="$2"; shift 2 ;;
    -h|--help)          sed -n '2,15p' "$0" | sed 's/^# \?//'; exit 0 ;;
    *) echo "run_qec_compare_job.sh: unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$NAME" ]]; then
  NAME="qec-ae-d${DISTANCE}-w${WIDTH}-ds${DOWN_STAGES}-s${SEED}-$(date +%s)"
fi

# Default routing: d>=5 needs 24GB cards.
if [[ -z "$MIN_GPU_MEM" && "$DISTANCE" -ge 5 ]]; then
  MIN_GPU_MEM=20000
fi

LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"
STAGE_REMOTE="/mnt/hyperdisk/sandbox/gauge_paper_code"
RESULTS_DIR="/work/qec_runs/${TAG:+${TAG}/}${NAME}"

echo "[stage] copying gauge_paper code to ${JUMPBOX}:${STAGE_REMOTE}"
ssh -q "$JUMPBOX" "mkdir -p ${STAGE_REMOTE}"
scp -q "$LOCAL_DIR"/*.py "$LOCAL_DIR"/run_qec_compare_job.sh "$JUMPBOX:${STAGE_REMOTE}/"

JSON=$(NAME="$NAME" \
       NAMESPACE="$NAMESPACE" \
       IMAGE="$IMAGE" \
       IMAGE_PULL_SECRET="$IMAGE_PULL_SECRET" \
       NODE="$NODE" \
       SEED="$SEED" \
       DISTANCE="$DISTANCE" \
       WIDTH="$WIDTH" \
       DEPTH="$DEPTH" \
       DOWN_STAGES="$DOWN_STAGES" \
       BLOCKS_PER_STAGE="$BLOCKS_PER_STAGE" \
       EPOCHS="$EPOCHS" \
       N_PER_RATE="$N_PER_RATE" \
       N_VAL="$N_VAL" \
       BATCH_SIZE="$BATCH_SIZE" \
       NOISE="$NOISE" \
       RESULTS_DIR="$RESULTS_DIR" \
       MIN_GPU_MEM="$MIN_GPU_MEM" \
       MAX_GPU_MEM="$MAX_GPU_MEM" \
       TAG="$TAG" \
  python3 - <<'PY'
import json, os

name = os.environ["NAME"]
ns = os.environ["NAMESPACE"]
image = os.environ["IMAGE"]
image_pull_secret = os.environ.get("IMAGE_PULL_SECRET", "") or ""
node = os.environ["NODE"]
seed = os.environ["SEED"]
distance = os.environ["DISTANCE"]
width = os.environ["WIDTH"]
depth = os.environ["DEPTH"]
down_stages = os.environ["DOWN_STAGES"]
blocks_per_stage = os.environ["BLOCKS_PER_STAGE"]
epochs = os.environ["EPOCHS"]
n_per_rate = os.environ["N_PER_RATE"]
n_val = os.environ["N_VAL"]
batch_size = os.environ["BATCH_SIZE"]
noise = os.environ["NOISE"]
results_dir = os.environ["RESULTS_DIR"]
min_gpu_mem = os.environ.get("MIN_GPU_MEM", "") or ""
max_gpu_mem = os.environ.get("MAX_GPU_MEM", "") or ""
tag = os.environ.get("TAG", "") or ""

stage_local = "/work/gauge_paper_code"

train_script = "\n".join([
    "set -eo pipefail",
    f"mkdir -p {results_dir}/ckpts",
    "export GALAREON_PD_ROOT=/app/parity_diffusion",
    " ".join([
        f"python -u {stage_local}/train_qec_autoencoder.py",
        f"--distance {distance}",
        f"--noise {noise}",
        f"--epochs {epochs}",
        f"--width {width}",
        f"--depth {depth}",
        f"--down-stages {down_stages}",
        f"--blocks-per-stage {blocks_per_stage}",
        f"--n-per-rate {n_per_rate}",
        f"--n-val {n_val}",
        f"--batch-size {batch_size}",
        f"--seed {seed}",
        f"--save-dir {results_dir}/ckpts",
        f"--log-file {results_dir}/epochs.log",
        f"2>&1 | tee {results_dir}/run.log",
    ]),
])

main_container = {
    "name": "main",
    "image": image,
    "workingDir": "/app",
    "command": ["bash", "-c", train_script],
    "env": [
        {"name": "PYTHONUNBUFFERED", "value": "1"},
        {"name": "GALAREON_PD_ROOT", "value": "/app/parity_diffusion"},
    ],
    "resources": {"limits": {"nvidia.com/gpu": "1"}},
    "volumeMounts": [{"name": "work", "mountPath": "/work"}],
}

pod_spec = {
    "runtimeClassName": "nvidia",
    "restartPolicy": "Never",
    "tolerations": [
        {"key": "nvidia.com/gpu", "operator": "Exists"},
        {"key": "hpc", "operator": "Equal", "value": "true", "effect": "NoSchedule"},
    ],
    "containers": [main_container],
    "volumes": [
        {"name": "work", "persistentVolumeClaim": {"claimName": "scratch"}},
    ],
}

if image_pull_secret:
    pod_spec["imagePullSecrets"] = [{"name": image_pull_secret}]
if node:
    pod_spec["nodeSelector"] = {"kubernetes.io/hostname": node}

gpu_mem_exprs = []
if min_gpu_mem:
    gpu_mem_exprs.append({"key": "nvidia.com/gpu.memory", "operator": "Gt", "values": [str(min_gpu_mem)]})
if max_gpu_mem:
    gpu_mem_exprs.append({"key": "nvidia.com/gpu.memory", "operator": "Lt", "values": [str(max_gpu_mem)]})
if gpu_mem_exprs:
    pod_spec["affinity"] = {
        "nodeAffinity": {
            "requiredDuringSchedulingIgnoredDuringExecution": {
                "nodeSelectorTerms": [{"matchExpressions": gpu_mem_exprs}],
            },
        },
    }

labels = {"app": "qec-decoder", "distance": f"d{distance}", "seed": f"s{seed}",
          "backbone": "autoencoder", "noise": noise}
if tag:
    labels["tag"] = tag

job = {
    "apiVersion": "batch/v1",
    "kind": "Job",
    "metadata": {"name": name, "namespace": ns, "labels": labels},
    "spec": {
        "backoffLimit": 0,
        "ttlSecondsAfterFinished": 604800,
        "template": {"metadata": {"labels": labels}, "spec": pod_spec},
    },
}
print(json.dumps(job))
PY
)

B64=$(printf '%s' "$JSON" | base64 | tr -d '\n')
ssh -q "$JUMPBOX" "
  source ~/.bash_profile >/dev/null 2>&1
  printf '%s' '$B64' | base64 -d | kubectl create -f -
"

cat <<INFO

Submitted Job: $NAME
  distance:    d=$DISTANCE
  width:       $WIDTH
  down_stages: $DOWN_STAGES
  epochs:      $EPOCHS
  n_per_rate:  $N_PER_RATE
  seed:        $SEED
  results:     $RESULTS_DIR

Monitor:
  ssh $JUMPBOX kubectl -n $NAMESPACE get jobs -l app=qec-decoder,backbone=autoencoder
  ssh $JUMPBOX kubectl -n $NAMESPACE logs -f job/$NAME

Stop:
  ssh $JUMPBOX kubectl -n $NAMESPACE delete job $NAME
INFO

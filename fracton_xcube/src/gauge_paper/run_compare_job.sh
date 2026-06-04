#!/usr/bin/env bash
# run_compare_job.sh — fire a gauge-paper phi-prior vs x-prior A/B
# comparison on the galatheus k8s cluster.
#
# Stages the gauge_paper folder onto the shared scratch PVC, then submits
# a k8s Job that runs train_phi_prior_compare.py with the baked
# parity_diffusion image at /app. The script falls back to /app for
# parity_diffusion modules via the GALAREON_PD_ROOT env var.
#
# Usage:
#   ./run_compare_job.sh [--width 64] [--depth 8] [--epochs 5]
#                        [--batch-size 16] [--seed 42]
#                        [--variant both|x_prior|phi_prior]
#                        [--train-subset 0] [--val-subset 512]
#                        [--name NAME] [--node alpha]
#                        [--tag smoke1]
#
# Default config is a small smoke test (w=48 d=6 e=3 batch=8 train_subset=2048)
# that should complete in well under an hour on a single 24GB GPU and give
# enough signal to know if the phi-prior idea has any traction.

set -euo pipefail

JUMPBOX="${GPURUN_JUMPBOX:-brian@jumpbox}"
NAMESPACE="${GPURUN_NAMESPACE:-sandbox}"
IMAGE="${GPURUN_IMAGE:-ghcr.io/briangu/galareon:main}"
IMAGE_PULL_SECRET="${GPURUN_IMAGE_PULL_SECRET:-ghcr-pull-secret}"

# Defaults — small smoke test
WIDTH=48
DEPTH=6
EPOCHS=3
BATCH_SIZE=8
SEED=42
LR=2e-4
CUBE=32
TRAIN_SUBSET=2048
VAL_SUBSET=512
NUM_WORKERS=4
VARIANT=both
POS_WEIGHT=1.0
GT_BASE_CH=64
GT_LATENT_DIM=32
NO_GAUGE_AUX=false
DOWN_STAGES=2
DCT_LEVEL=0
NAME=""
NODE=""
TAG=""
MIN_GPU_MEM=""
MAX_GPU_MEM=""

usage() { sed -n '2,30p' "$0" | sed 's/^# \?//'; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --width)        WIDTH="$2"; shift 2 ;;
    --depth)        DEPTH="$2"; shift 2 ;;
    --epochs)       EPOCHS="$2"; shift 2 ;;
    --batch-size)   BATCH_SIZE="$2"; shift 2 ;;
    --seed)         SEED="$2"; shift 2 ;;
    --lr)           LR="$2"; shift 2 ;;
    --cube)         CUBE="$2"; shift 2 ;;
    --train-subset) TRAIN_SUBSET="$2"; shift 2 ;;
    --val-subset)   VAL_SUBSET="$2"; shift 2 ;;
    --num-workers)  NUM_WORKERS="$2"; shift 2 ;;
    --variant)      VARIANT="$2"; shift 2 ;;
    --pos-weight)   POS_WEIGHT="$2"; shift 2 ;;
    --gt-base-ch)   GT_BASE_CH="$2"; shift 2 ;;
    --gt-latent-dim) GT_LATENT_DIM="$2"; shift 2 ;;
    --no-gauge-aux) NO_GAUGE_AUX=true; shift ;;
    --down-stages)  DOWN_STAGES="$2"; shift 2 ;;
    --dct-level)    DCT_LEVEL="$2"; shift 2 ;;
    --name)         NAME="$2"; shift 2 ;;
    --node)         NODE="$2"; shift 2 ;;
    --tag)          TAG="$2"; shift 2 ;;
    --min-gpu-mem)  MIN_GPU_MEM="$2"; shift 2 ;;
    --max-gpu-mem)  MAX_GPU_MEM="$2"; shift 2 ;;
    --image)        IMAGE="$2"; shift 2 ;;
    -h|--help)      usage; exit 0 ;;
    *) echo "run_compare_job.sh: unknown arg: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$NAME" ]]; then
  NAME="gauge-cmp-w${WIDTH}-d${DEPTH}-s${SEED}-$(date +%s)"
fi

# Stage gauge_paper folder onto scratch via scp.
LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"
STAGE_REMOTE="/mnt/hyperdisk/sandbox/gauge_paper_code"
RESULTS_DIR="/work/gauge_paper_runs/${TAG:+${TAG}/}${NAME}"

echo "[stage] copying gauge_paper code to ${JUMPBOX}:${STAGE_REMOTE}"
ssh -q "$JUMPBOX" "mkdir -p ${STAGE_REMOTE}"
scp -q "$LOCAL_DIR"/*.py "$LOCAL_DIR"/run_compare_job.sh "$JUMPBOX:${STAGE_REMOTE}/"

# Build the Job manifest with python on this side, base64 it, and
# kubectl-create on the jumpbox.
JSON=$(NAME="$NAME" \
       NAMESPACE="$NAMESPACE" \
       IMAGE="$IMAGE" \
       IMAGE_PULL_SECRET="$IMAGE_PULL_SECRET" \
       NODE="$NODE" \
       SEED="$SEED" \
       WIDTH="$WIDTH" \
       DEPTH="$DEPTH" \
       EPOCHS="$EPOCHS" \
       BATCH_SIZE="$BATCH_SIZE" \
       LR="$LR" \
       CUBE="$CUBE" \
       TRAIN_SUBSET="$TRAIN_SUBSET" \
       VAL_SUBSET="$VAL_SUBSET" \
       NUM_WORKERS="$NUM_WORKERS" \
       VARIANT="$VARIANT" \
       POS_WEIGHT="$POS_WEIGHT" \
       GT_BASE_CH="$GT_BASE_CH" \
       GT_LATENT_DIM="$GT_LATENT_DIM" \
       NO_GAUGE_AUX="$NO_GAUGE_AUX" \
       DOWN_STAGES="$DOWN_STAGES" \
       DCT_LEVEL="$DCT_LEVEL" \
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
width = os.environ["WIDTH"]
depth = os.environ["DEPTH"]
epochs = os.environ["EPOCHS"]
batch_size = os.environ["BATCH_SIZE"]
lr = os.environ["LR"]
cube = os.environ["CUBE"]
train_subset = os.environ["TRAIN_SUBSET"]
val_subset = os.environ["VAL_SUBSET"]
num_workers = os.environ["NUM_WORKERS"]
variant = os.environ["VARIANT"]
pos_weight = os.environ.get("POS_WEIGHT", "1.0")
gt_base_ch = os.environ.get("GT_BASE_CH", "64")
gt_latent_dim = os.environ.get("GT_LATENT_DIM", "32")
no_gauge_aux = os.environ.get("NO_GAUGE_AUX", "false") == "true"
down_stages = os.environ.get("DOWN_STAGES", "2")
dct_level = os.environ.get("DCT_LEVEL", "0")
results_dir = os.environ["RESULTS_DIR"]
min_gpu_mem = os.environ.get("MIN_GPU_MEM", "") or ""
max_gpu_mem = os.environ.get("MAX_GPU_MEM", "") or ""
tag = os.environ.get("TAG", "") or ""

stage_local = "/work/gauge_paper_code"

train_script = "\n".join([
    "set -eo pipefail",
    f"mkdir -p {results_dir}",
    f"cp -r {stage_local}/*.py {stage_local}/run_compare_job.sh {results_dir}/code/ 2>/dev/null || (mkdir -p {results_dir}/code && cp {stage_local}/*.py {stage_local}/run_compare_job.sh {results_dir}/code/)",
    "export GALAREON_PD_ROOT=/app/parity_diffusion",
    " ".join([
        f"python -u {stage_local}/train_phi_prior_compare.py",
        f"--seed {seed}",
        f"--width {width}",
        f"--depth {depth}",
        f"--epochs {epochs}",
        f"--batch-size {batch_size}",
        f"--lr {lr}",
        f"--cube-size {cube}",
        f"--train-subset {train_subset}",
        f"--val-subset {val_subset}",
        f"--num-workers {num_workers}",
        f"--variant {variant}",
        f"--pos-weight {pos_weight}",
        f"--gt-base-ch {gt_base_ch}",
        f"--gt-latent-dim {gt_latent_dim}",
        "--no-gauge-aux" if no_gauge_aux else "",
        f"--down-stages {down_stages}",
        f"--dct-level {dct_level}",
        f"--out-dir {results_dir}",
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

labels = {"app": "gauge-paper", "variant": variant, "seed": f"s{seed}"}
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
  variant:    $VARIANT
  width:      $WIDTH
  depth:      $DEPTH
  epochs:     $EPOCHS
  batch_size: $BATCH_SIZE
  seed:       $SEED
  results:    $RESULTS_DIR

Monitor:
  ssh $JUMPBOX kubectl -n $NAMESPACE get jobs -l app=gauge-paper
  ssh $JUMPBOX kubectl -n $NAMESPACE logs -f job/$NAME

Stop:
  ssh $JUMPBOX kubectl -n $NAMESPACE delete job $NAME
INFO

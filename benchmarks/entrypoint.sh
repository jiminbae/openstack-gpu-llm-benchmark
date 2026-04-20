#!/bin/bash
set -e

echo "=================================================="
echo "  Axolotl Benchmark — LLaMA 3.1 8B QLoRA"
echo "=================================================="

# ── run.sh에서 전달받은 환경변수 ──────────────────────────
CONFIG_FILE="${CONFIG_FILE:-96gb_single.yml}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
LAUNCH_MODE="${LAUNCH_MODE:-single}"
BENCHMARK_TYPE="${BENCHMARK_TYPE:-unknown}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d-%H%M%S)}"

# ── 출력 디렉터리 구성: /workspace/final-result/<타입>/<타임스탬프> ──
OUTPUT_DIR="/workspace/final-result/${BENCHMARK_TYPE}/${TIMESTAMP}"
mkdir -p "${OUTPUT_DIR}"

echo "  CONFIG_FILE  : ${CONFIG_FILE}"
echo "  NUM_PROCESSES: ${NUM_PROCESSES}"
echo "  LAUNCH_MODE  : ${LAUNCH_MODE}"
echo "  BENCHMARK    : ${BENCHMARK_TYPE}"
echo "  TIMESTAMP    : ${TIMESTAMP}"
echo "  OUTPUT_DIR   : ${OUTPUT_DIR}"
echo ""

# ── GPU 정보 출력 ──────────────────────────────────────────
echo "[1/2] GPU 확인"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || \
    nvidia-smi -L
echo ""

# ── 실행 모드별 accelerate 플래그 구성 ────────────────────
EXTRA_FLAGS=""
case "${LAUNCH_MODE}" in
    single)
        :
        ;;
    ddp|fsdp)
        EXTRA_FLAGS="--multi_gpu"
        ;;
    *)
        echo "[ERROR] 알 수 없는 LAUNCH_MODE: ${LAUNCH_MODE}"
        exit 1
        ;;
esac

# ── 학습 실행 ──────────────────────────────────────────────
# --output_dir 는 axolotl이 YAML 값을 덮어쓰도록 CLI override로 전달
echo "[2/2] 학습 시작"
cd /workspace

AXOLOTL_DO_NOT_TRACK=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
HF_HUB_OFFLINE=1 \
accelerate launch \
    --num_processes="${NUM_PROCESSES}" \
    --num_machines=1 \
    ${EXTRA_FLAGS} \
    --mixed_precision=bf16 \
    --dynamo_backend=no \
    -m axolotl.cli.train \
    "${CONFIG_FILE}" \
    --output_dir="${OUTPUT_DIR}"

echo ""
echo "[완료] 결과물 위치: ${OUTPUT_DIR}"
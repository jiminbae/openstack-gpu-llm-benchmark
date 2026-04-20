#!/bin/bash
set -e

# ── 사용법 ────────────────────────────────────────────────
usage() {
    cat <<EOF
사용법: $0 {mig48|single48|single96|ddp48|fsdp48}

  mig48    - 96GB GPU를 MIG로 2개로 쪼갠 후 48GB MIG 인스턴스에서 학습
  single48 - 단일 48GB GPU에서 학습
  single96 - 단일 96GB GPU에서 학습
  ddp48    - 2x 48GB GPU에서 학습 (DDP)
  fsdp48   - 2x 48GB GPU에서 학습 (FSDP2 + QLoRA)
EOF
    exit 1
}

# ── 파라미터 파싱 ──────────────────────────────────────────
BENCHMARK_TYPE="${1:-}"
[ -z "${BENCHMARK_TYPE}" ] && usage

case "${BENCHMARK_TYPE}" in
    mig48)
        CONFIG_FILE="48gb_single.yml"
        NUM_PROCESSES=1
        GPU_MODE="mig"
        LAUNCH_MODE="single"
        DESC="48GB MIG instance"
        ;;
    single48)
        CONFIG_FILE="48gb_single.yml"
        NUM_PROCESSES=1
        GPU_MODE="single"
        LAUNCH_MODE="single"
        DESC="단일 48GB GPU"
        ;;
    single96)
        CONFIG_FILE="96gb_single.yml"
        NUM_PROCESSES=1
        GPU_MODE="single"
        LAUNCH_MODE="single"
        DESC="단일 96GB GPU"
        ;;
    ddp48)
        CONFIG_FILE="48gb_ddp.yml"
        NUM_PROCESSES=2
        GPU_MODE="dual"
        LAUNCH_MODE="ddp"
        DESC="2x 48GB GPU (DDP)"
        ;;
    fsdp48)
        CONFIG_FILE="48gb_fsdp.yml"
        NUM_PROCESSES=2
        GPU_MODE="dual"
        LAUNCH_MODE="fsdp"
        DESC="2x 48GB GPU (FSDP2 + QLoRA)"
        ;;
    *)
        usage
        ;;
esac

# ── 타임스탬프 (한 번만 생성, CONTAINER_NAME과 OUTPUT_DIR에 공통 사용) ──
TIMESTAMP=$(date +%Y%m%d-%H%M%S)

IMAGE_NAME="axolotl-benchmark"
CONTAINER_NAME="axolotl-${BENCHMARK_TYPE}-${TIMESTAMP}"
RESULT_DIR="$(pwd)/output"
CACHE_DIR="$(pwd)/cache"
MODEL_DIR="${HOME}/automated_benchmark/models/Llama-3.1-8B"

# 호스트/컨테이너 양쪽에서 같은 경로 구조
HOST_OUTPUT_SUBDIR="${RESULT_DIR}/${BENCHMARK_TYPE}/${TIMESTAMP}"
CONTAINER_OUTPUT_SUBDIR="/workspace/final-result/${BENCHMARK_TYPE}/${TIMESTAMP}"

echo "=================================================="
echo "  벤치마크 타입 : ${BENCHMARK_TYPE} (${DESC})"
echo "  YAML         : ${CONFIG_FILE}"
echo "  프로세스 수  : ${NUM_PROCESSES}"
echo "  실행 모드    : ${LAUNCH_MODE}"
echo "  타임스탬프   : ${TIMESTAMP}"
echo "  출력 경로    : ${HOST_OUTPUT_SUBDIR}"
echo "=================================================="

# ── 모델 경로 확인 ─────────────────────────────────────────
echo ""
echo "[사전점검] 모델 경로: ${MODEL_DIR}"
if [ ! -f "${MODEL_DIR}/config.json" ]; then
    echo "[ERROR] 모델 파일이 없습니다."
    exit 1
fi
echo "  → 확인 완료"

mkdir -p "${RESULT_DIR}" "${CACHE_DIR}"

# ── GPU 디바이스 결정 ──────────────────────────────────────
echo ""
echo "[사전점검] GPU 디바이스 설정"
case "${GPU_MODE}" in
    mig)
        MIG_UUID=$(nvidia-smi -L | grep "MIG" | grep -oP 'MIG-[a-f0-9\-]+' | head -1)
        if [ -z "${MIG_UUID}" ]; then
            echo "[ERROR] MIG 인스턴스를 찾을 수 없습니다."
            exit 1
        fi
        # 단일 디바이스는 큰따옴표 없이도 OK
        GPU_ARG="device=${MIG_UUID}"
        echo "  → 감지된 MIG UUID: ${MIG_UUID}"
        ;;
    single)
        GPU_ARG="device=0"
        echo "  → 사용 디바이스: device=0"
        ;;
    dual)
        GPU_COUNT=$(nvidia-smi -L | grep -c "^GPU ")
        if [ "${GPU_COUNT}" -lt 2 ]; then
            echo "[ERROR] ${BENCHMARK_TYPE}은 최소 2개의 GPU가 필요합니다. (감지: ${GPU_COUNT}개)"
            exit 1
        fi
        # ★ 다중 디바이스는 반드시 내부 큰따옴표로 감싸기
        GPU_ARG='"device=0,1"'
        echo "  → 사용 디바이스: device=0,1 (${LAUNCH_MODE^^})"
        ;;
esac

# ── [1/2] Docker 이미지 빌드 ───────────────────────────────
echo ""
echo "=================================================="
echo "  [1/2] Docker 이미지 빌드: ${IMAGE_NAME}"
echo "=================================================="
docker buildx build \
    --progress=plain \
    --network=host \
    --load \
    -t "${IMAGE_NAME}" \
    -f Dockerfile \
    .

# ── [2/2] 컨테이너 실행 ────────────────────────────────────
echo ""
echo "=================================================="
echo "  [2/2] 컨테이너 실행: ${CONTAINER_NAME}"
echo "=================================================="
docker run \
    --rm \
    --gpus ${GPU_ARG} \
    --network=host \
    --name "${CONTAINER_NAME}" \
    --shm-size=16g \
    -e CONFIG_FILE="${CONFIG_FILE}" \
    -e NUM_PROCESSES="${NUM_PROCESSES}" \
    -e LAUNCH_MODE="${LAUNCH_MODE}" \
    -e BENCHMARK_TYPE="${BENCHMARK_TYPE}" \
    -e TIMESTAMP="${TIMESTAMP}" \
    -v "${RESULT_DIR}:/workspace/final-result" \
    -v "${MODEL_DIR}:/workspace/model" \
    -v "${CACHE_DIR}:/workspace/cache" \
    "${IMAGE_NAME}"

echo ""
echo "=================================================="
echo "  [완료] 결과물 저장 위치: ${HOST_OUTPUT_SUBDIR}"
echo "=================================================="
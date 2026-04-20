FROM nvidia/cuda:13.0.3-cudnn-devel-ubuntu24.04

# ── 기본 환경 ──────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
# ── 시스템 패키지 ──────────────────────────────────────────
# Ubuntu 24.04 기본 Python 3.12 사용
RUN apt-get update && apt-get install -y \
    python3.12 python3.12-dev python3.12-venv \
    python3-pip git wget curl build-essential \
    libssl-dev libffi-dev ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# ── 작업 디렉토리 ──────────────────────────────────────────
WORKDIR /workspace
RUN mkdir -p /workspace/cache /workspace/result /workspace/final-result/48gb

# ── 가상환경 생성 및 활성화 ─────────────────────────────────
RUN python -m venv /axo-venv
ENV PATH="/axo-venv/bin:$PATH"

# ── pip 업그레이드 ──────────────────────────────────────────
RUN pip install --upgrade pip packaging==26.0 setuptools wheel ninja

# ── [1단계] PyTorch 먼저 설치 ──────────────────────────────
# axolotl --no-build-isolation 설치 시 torch 버전을 감지해야 하므로 선행 필수
RUN pip install \
    torch==2.11.0+cu130 \
    torchvision==0.26.0+cu130 \
    torchaudio==2.11.0+cu130 \
    --index-url https://download.pytorch.org/whl/cu130

# ── [2단계] Flash Attention (prebuilt wheel) ───────────────
# torch 설치 후 즉시 설치 — axolotl[flash-attn] 소스 빌드 차단
RUN pip install \
    "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.4/flash_attn-2.8.3+cu130torch2.11-cp312-cp312-linux_x86_64.whl"

# ── [3단계] Axolotl 0.16.1 설치 ───────────────────────────
# --no-build-isolation: 설치된 torch를 감지해 의존성 버전을 올바르게 결정
# transformers==4.53.1, accelerate==1.8.1, bitsandbytes==0.46.0 등은
# axolotl이 알아서 관리하므로 별도 지정 불필요
RUN pip install --no-build-isolation axolotl==0.16.1

# ── [4단계] 나머지 벤치마크 의존성 ────────────────────────
# xformers는 PyTorch wheel 인덱스에서 받아야 하므로 별도 명령으로 분리
RUN pip install xformers==0.0.35 --index-url https://download.pytorch.org/whl/cu130

# 나머지는 PyPI에서 설치
RUN pip install \
    "lm_eval[hf]==0.4.11" \
    evaluate==0.4.1 \
    rouge_score==0.1.2 \
    sacrebleu==2.6.0 \
    scipy==1.17.1 \
    wandb \
    tensorboard \
    optimum \
    torchao==0.17.0 \
    hf_transfer

# ── 설정 파일 복사 ─────────────────────────────────────────
COPY 48gb_single.yml /workspace/48gb_single.yml
COPY 96gb_single.yml /workspace/96gb_single.yml
COPY 48gb_ddp.yml    /workspace/48gb_ddp.yml
COPY 48gb_fsdp.yml   /workspace/48gb_fsdp.yml
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
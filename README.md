# OpenStack 클라우드 환경에서 GPU 가상화 방식에 따른 LLM 파인튜닝 성능 비교 연구

**LLM Fine-tuning Performance Comparison by GPU Virtualization Method in OpenStack Cloud Environments**

> 오픈스택 클라우드 환경 내 LLM 워크로드를 위한 GPU 가상화 성능 분석: MIG와 PCIe 패스스루를 중심으로
>
> GPU Performance Analysis for LLM Workloads in OpenStack Environments: MIG vs. Passthrough

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11.0-orange)
![CUDA](https://img.shields.io/badge/CUDA-13.0.3-green)
![Axolotl](https://img.shields.io/badge/Axolotl-0.16.1-purple)
![Docker](https://img.shields.io/badge/Docker-required-blue)

---

**[한국어](#한국어) | [English](#english)**

---

## 한국어

### 연구 개요

프라이빗 클라우드 환경에서 **동일한 48GB VRAM 조건** 아래, 두 가지 GPU 할당 방식의 LLM 파인튜닝 성능을 실증적으로 비교합니다.

| 구성 | 노드 | GPU | 방식 |
|---|---|---|---|
| **MIG 48GB** | compute1 | RTX PRO 6000 Blackwell 96GB → MIG 2g.48gb | PCIe Passthrough + MIG 분할 |
| **물리 48GB** | compute2 | RTX A6000 48GB × 1장 | PCIe Passthrough |

벤치마크 모델: **LLaMA 3.1 8B QLoRA** (Axolotl 기반 Docker 자동화)

### 리포지토리 구조

```
capstone-project1/
├── benchmarks/             # 벤치마크 자동화 도구 (메인)
│   ├── run.sh              # 메인 실행 스크립트
│   ├── Dockerfile          # 학습 환경 정의
│   ├── entrypoint.sh       # 컨테이너 내 학습 실행
│   ├── 48gb_single.yml     # 단일 48GB / MIG용 Axolotl 설정
│   ├── 96gb_single.yml     # 단일 96GB용 Axolotl 설정
│   ├── 48gb_ddp.yml        # 2x 48GB DDP용 Axolotl 설정
│   ├── 48gb_fsdp.yml       # 2x 48GB FSDP2용 Axolotl 설정
│   └── models/             # 모델 파일 위치 (로컬 배치 필요)
├── meeting-notes/          # 회의록
├── presentations/          # 발표 자료
└── README.md
```

### 벤치마크 도구 (`benchmarks/`)

#### 실행 모드

| 모드 | GPU 구성 | 설명 |
|---|---|---|
| `mig48` | 96GB GPU → MIG 48GB | MIG로 분할된 48GB 인스턴스에서 학습 |
| `single48` | 1x 48GB GPU | 단일 48GB GPU에서 학습 |
| `single96` | 1x 96GB GPU | 단일 96GB GPU에서 학습 |
| `ddp48` | 2x 48GB GPU (DDP) | DistributedDataParallel 기반 다중 GPU |
| `fsdp48` | 2x 48GB GPU (FSDP2) | Fully Sharded Data Parallel v2 + QLoRA |

#### 요구 사항

**하드웨어**
- `mig48`: MIG 지원 GPU (Ampere 이상), 사전 MIG 파티션 설정 필요
- `ddp48`, `fsdp48`: 최소 GPU 2장 필요

**소프트웨어**
- Docker (buildx 지원)
- NVIDIA Container Toolkit
- NVIDIA 드라이버 (CUDA 13.0 호환)

**모델**
LLaMA 3.1 8B 모델을 아래 경로에 사전 다운로드:
```
~/automated_benchmark/models/Llama-3.1-8B/
```

#### 사용법

```bash
cd benchmarks/

./run.sh single48   # 단일 48GB GPU
./run.sh mig48      # MIG 48GB 인스턴스
./run.sh single96   # 단일 96GB GPU
./run.sh ddp48      # 2x 48GB DDP
./run.sh fsdp48     # 2x 48GB FSDP2
```

#### MIG 모드 사전 준비

`mig48` 실행 전 호스트에서 MIG 파티션을 생성해야 합니다:

```bash
# MIG 활성화
sudo nvidia-smi -mig 1

# 48GB x 2 파티션 생성 (RTX PRO 6000 Blackwell 기준, 프로파일 ID: 5)
sudo nvidia-smi mig -cgi 5,5
sudo nvidia-smi mig -cci

# MIG UUID 확인
nvidia-smi -L
```

스크립트는 `nvidia-smi -L`에서 첫 번째 MIG 인스턴스를 자동 감지합니다.

#### 실행 흐름

```
run.sh 실행
  │
  ├─ [사전점검] 모델 파일 확인, GPU 디바이스 감지
  ├─ [1/2] Docker 이미지 빌드 (axolotl-benchmark)
  └─ [2/2] 컨테이너 실행
              │
              └─ entrypoint.sh
                    ├─ GPU 정보 출력
                    └─ accelerate launch → axolotl.cli.train
```

#### 출력 결과

결과는 타임스탬프 기반 디렉터리에 저장됩니다:

```
benchmarks/output/
└── single48/
    └── 20260420-143022/
        ├── checkpoint-*/
        ├── adapter_config.json
        └── adapter_model.safetensors
```

#### 학습 하이퍼파라미터 (공통)

| 항목 | 값 |
|---|---|
| 모델 | LLaMA 3.1 8B |
| 방식 | QLoRA (4bit) |
| LoRA rank | 32 (48GB), 64 (96GB/DDP/FSDP) |
| Sequence length | 20,480 tokens |
| Micro batch size | 1 |
| Gradient accumulation | 16 (단일), 8 (멀티 GPU) |
| Effective batch size | 16 (전 구성 동일) |
| Max steps | 20 |
| Optimizer | paged_adamw_32bit |
| Precision | bf16 + tf32 |
| Flash Attention | 활성화 |

#### 기술 스택

| 구성 요소 | 버전 |
|---|---|
| CUDA | 13.0.3 |
| Python | 3.12 |
| PyTorch | 2.11.0+cu130 |
| Flash Attention | 2.8.3 |
| Axolotl | 0.16.1 |
| xformers | 0.0.35 |
| Base OS | Ubuntu 24.04 |

### 인프라 환경

| 노드 | 역할 | GPU | OS |
|---|---|---|---|
| user-Super-Server | OpenStack 컨트롤러 | - | Ubuntu 24.04 |
| compute1 | compute 노드 | RTX PRO 6000 Blackwell 96GB × 1 | Ubuntu 24.04 |
| compute2 | compute 노드 | RTX A6000 48GB × 4 | Ubuntu 24.04 |

- OpenStack: Keystone, Glance, Nova, Neutron, Placement
- GPU 할당: PCIe Passthrough (VFIO)
- 커널: 6.17.0 (HWE)

### 참고 사항

- 컨테이너는 `--rm` 옵션으로 실행되어 종료 시 자동 삭제됩니다. 결과물은 볼륨 마운트로 호스트에 영구 저장됩니다.
- `HF_HUB_OFFLINE=1` 설정으로 학습 중 외부 네트워크 접속을 차단합니다. 모델은 반드시 로컬에 준비되어 있어야 합니다.
- 멀티 GPU 모드에서는 `--shm-size=16g`로 공유 메모리를 할당합니다.
- DDP와 FSDP 모드에서 gradient_accumulation_steps를 8로 줄여 effective batch size 16을 유지합니다 (단일 GPU 구성과 공정 비교).

---

## English

### Research Overview

This project empirically compares LLM fine-tuning performance under **identical 48GB VRAM conditions** with two different GPU allocation strategies in a private cloud environment.

| Configuration | Node | GPU | Method |
|---|---|---|---|
| **MIG 48GB** | compute1 | RTX PRO 6000 Blackwell 96GB → MIG 2g.48gb | PCIe Passthrough + MIG partition |
| **Physical 48GB** | compute2 | RTX A6000 48GB × 1 | PCIe Passthrough |

Benchmark model: **LLaMA 3.1 8B QLoRA** (Docker-automated with Axolotl)

### Repository Structure

```
capstone-project1/
├── benchmarks/             # Benchmark automation tool (main)
│   ├── run.sh              # Main execution script
│   ├── Dockerfile          # Training environment definition
│   ├── entrypoint.sh       # In-container training runner
│   ├── 48gb_single.yml     # Axolotl config for single 48GB / MIG
│   ├── 96gb_single.yml     # Axolotl config for single 96GB
│   ├── 48gb_ddp.yml        # Axolotl config for 2x 48GB DDP
│   ├── 48gb_fsdp.yml       # Axolotl config for 2x 48GB FSDP2
│   └── models/             # Model file location (must be placed locally)
├── meeting-notes/          # Meeting notes
├── presentations/          # Presentation slides
└── README.md
```

### Benchmark Tool (`benchmarks/`)

#### Execution Modes

| Mode | GPU Configuration | Description |
|---|---|---|
| `mig48` | 96GB GPU → MIG 48GB | Train on MIG-partitioned 48GB instance |
| `single48` | 1x 48GB GPU | Train on single 48GB GPU |
| `single96` | 1x 96GB GPU | Train on single 96GB GPU |
| `ddp48` | 2x 48GB GPU (DDP) | Multi-GPU with DistributedDataParallel |
| `fsdp48` | 2x 48GB GPU (FSDP2) | Fully Sharded Data Parallel v2 + QLoRA |

#### Requirements

**Hardware**
- `mig48`: MIG-capable GPU (Ampere or later), MIG partition must be configured in advance
- `ddp48`, `fsdp48`: At least 2 GPUs required

**Software**
- Docker (with buildx support)
- NVIDIA Container Toolkit
- NVIDIA driver (CUDA 13.0 compatible)

**Model**
LLaMA 3.1 8B model must be downloaded locally at:
```
~/automated_benchmark/models/Llama-3.1-8B/
```

#### Usage

```bash
cd benchmarks/

./run.sh single48   # Single 48GB GPU
./run.sh mig48      # MIG 48GB instance
./run.sh single96   # Single 96GB GPU
./run.sh ddp48      # 2x 48GB DDP
./run.sh fsdp48     # 2x 48GB FSDP2
```

#### MIG Mode Prerequisites

Before running `mig48`, create MIG partitions on the host:

```bash
# Enable MIG mode
sudo nvidia-smi -mig 1

# Create 2x 48GB partitions (Profile ID 5 for RTX PRO 6000 Blackwell)
sudo nvidia-smi mig -cgi 5,5
sudo nvidia-smi mig -cci

# Verify MIG UUID
nvidia-smi -L
```

The script automatically detects the first MIG instance from `nvidia-smi -L`.

#### Execution Flow

```
run.sh
  │
  ├─ [Pre-check] Verify model files, detect GPU devices
  ├─ [1/2] Build Docker image (axolotl-benchmark)
  └─ [2/2] Run container
              │
              └─ entrypoint.sh
                    ├─ Print GPU info
                    └─ accelerate launch → axolotl.cli.train
```

#### Output

Results are saved in timestamp-based directories:

```
benchmarks/output/
└── single48/
    └── 20260420-143022/
        ├── checkpoint-*/
        ├── adapter_config.json
        └── adapter_model.safetensors
```

#### Training Hyperparameters (Common)

| Parameter | Value |
|---|---|
| Model | LLaMA 3.1 8B |
| Method | QLoRA (4bit) |
| LoRA rank | 32 (48GB), 64 (96GB/DDP/FSDP) |
| Sequence length | 20,480 tokens |
| Micro batch size | 1 |
| Gradient accumulation | 16 (single), 8 (multi-GPU) |
| Effective batch size | 16 (identical across all configs) |
| Max steps | 20 |
| Optimizer | paged_adamw_32bit |
| Precision | bf16 + tf32 |
| Flash Attention | Enabled |

#### Tech Stack

| Component | Version |
|---|---|
| CUDA | 13.0.3 |
| Python | 3.12 |
| PyTorch | 2.11.0+cu130 |
| Flash Attention | 2.8.3 |
| Axolotl | 0.16.1 |
| xformers | 0.0.35 |
| Base OS | Ubuntu 24.04 |

### Infrastructure

| Node | Role | GPU | OS |
|---|---|---|---|
| user-Super-Server | OpenStack Controller | - | Ubuntu 24.04 |
| compute1 | Compute Node | RTX PRO 6000 Blackwell 96GB × 1 | Ubuntu 24.04 |
| compute2 | Compute Node | RTX A6000 48GB × 4 | Ubuntu 24.04 |

- OpenStack: Keystone, Glance, Nova, Neutron, Placement
- GPU allocation: PCIe Passthrough (VFIO)
- Kernel: 6.17.0 (HWE)

### Notes

- Containers run with `--rm` and are automatically removed on exit. Results are permanently saved to the host via volume mount.
- `HF_HUB_OFFLINE=1` prevents any Hugging Face Hub access during training. The model must be available locally.
- Multi-GPU modes use `--shm-size=16g` for NCCL shared memory.
- DDP and FSDP modes use `gradient_accumulation_steps=8` to maintain an effective batch size of 16, matching single-GPU configs for a fair comparison.

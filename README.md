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

| 실험 | 모드 | GPU 구성 | 목적 |
|---|---|---|---|
| **실험 1** | `mig48` vs `single48` | MIG 48GB vs 물리 48GB | 동일 VRAM, 가상화 기법 차이 비교 |
| **실험 2** | `single96` vs `fsdp48` | 96GB × 1 vs 48GB × 2 (FSDP2) | 단일 대용량 vs FSDP 분산 비교 |

벤치마크 모델: **LLaMA 3.1 8B QLoRA** (Axolotl 기반 Docker 자동화)

### 리포지토리 구조

```
capstone-project1/
├── benchmarks/benchmark-tool     # 벤치마크 자동화 도구 (메인)
│   ├── run.sh                    # 메인 실행 스크립트
│   ├── Dockerfile                # 학습 환경 정의
│   ├── entrypoint.sh             # 컨테이너 내 학습 실행
│   ├── 48gb_single.yml           # 단일 48GB / MIG용 Axolotl 설정
│   ├── 96gb_single.yml           # 단일 96GB용 Axolotl 설정
│   ├── 48gb_ddp.yml              # 2x 48GB DDP용 Axolotl 설정
│   ├── 48gb_fsdp.yml             # 2x 48GB FSDP2용 Axolotl 설정
│   └── models/                   # 모델 파일 위치 (로컬 배치 필요)
├── meeting-notes/                # 회의록
├── presentations/                # 발표 자료
└── README.md
```

### 벤치마크 도구

[![benchmarks](https://img.shields.io/badge/📁%20benchmarks-blue)](./benchmarks/benchmark-tool)

### 인프라 환경

| 노드 | 역할 | GPU | OS |
|---|---|---|---|
| user-Super-Server | OpenStack 컨트롤러 | - | Ubuntu 24.04 |
| compute1 | compute 노드 | RTX PRO 6000 Blackwell 96GB × 1 | Ubuntu 24.04 |
| compute2 | compute 노드 | RTX A6000 48GB × 4 | Ubuntu 24.04 |

- OpenStack: Keystone, Glance, Nova, Neutron, Placement
- GPU 할당: PCIe Passthrough (VFIO)
- 커널: 6.8.0-110-generic

### 참고 사항

- 컨테이너는 `--rm` 옵션으로 실행되어 종료 시 자동 삭제됩니다. 결과물은 볼륨 마운트로 호스트에 영구 저장됩니다.
- `HF_HUB_OFFLINE=1` 설정으로 학습 중 외부 네트워크 접속을 차단합니다. 모델은 반드시 로컬에 준비되어 있어야 합니다.
- 멀티 GPU 모드에서는 `--shm-size=16g`로 공유 메모리를 할당합니다.
- DDP와 FSDP 모드에서 gradient_accumulation_steps를 8로 줄여 effective batch size 16을 유지합니다 (단일 GPU 구성과 공정 비교).

---

## English

### Research Overview

This project empirically compares LLM fine-tuning performance under **identical 48GB VRAM conditions** with two different GPU allocation strategies in a private cloud environment.

| Experiment | Mode | GPU Configuration | Purpose |
|---|---|---|---|
| **Experiment 1** | `mig48` vs `single48` | MIG 48GB vs Physical 48GB | Same VRAM, different virtualization methods |
| **Experiment 2** | `single96` vs `fsdp48` | 96GB × 1 vs 48GB × 2 (FSDP2) | Single large GPU vs FSDP distributed training |

Benchmark model: **LLaMA 3.1 8B QLoRA** (Docker-automated with Axolotl)

### Repository Structure

```
capstone-project1/
├── benchmarks/benchmark-tool     # Benchmark automation tool (main)
│   ├── run.sh                    # Main execution script
│   ├── Dockerfile                # Training environment definition
│   ├── entrypoint.sh             # In-container training runner
│   ├── 48gb_single.yml           # Axolotl config for single 48GB / MIG
│   ├── 96gb_single.yml           # Axolotl config for single 96GB
│   ├── 48gb_ddp.yml              # Axolotl config for 2x 48GB DDP
│   ├── 48gb_fsdp.yml             # Axolotl config for 2x 48GB FSDP2
│   └── models/                   # Model file location (must be placed locally)
├── meeting-notes/                # Meeting notes
├── presentations/                # Presentation slides
└── README.md
```

### Benchmark Tool

[![benchmarks](https://img.shields.io/badge/📁%20benchmarks-blue)](./benchmarks/benchmark-tool)

### Infrastructure

| Node | Role | GPU | OS |
|---|---|---|---|
| user-Super-Server | OpenStack Controller | - | Ubuntu 24.04 |
| compute1 | Compute Node | RTX PRO 6000 Blackwell 96GB × 1 | Ubuntu 24.04 |
| compute2 | Compute Node | RTX A6000 48GB × 4 | Ubuntu 24.04 |

- OpenStack: Keystone, Glance, Nova, Neutron, Placement
- GPU allocation: PCIe Passthrough (VFIO)
- Kernel: 6.8.0-110-generic

### Notes

- Containers run with `--rm` and are automatically removed on exit. Results are permanently saved to the host via volume mount.
- `HF_HUB_OFFLINE=1` prevents any Hugging Face Hub access during training. The model must be available locally.
- Multi-GPU modes use `--shm-size=16g` for NCCL shared memory.
- DDP and FSDP modes use `gradient_accumulation_steps=8` to maintain an effective batch size of 16, matching single-GPU configs for a fair comparison.

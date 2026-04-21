# Axolotl QLoRA 벤치마크

LLaMA 3.1 8B 모델을 다양한 GPU 구성에서 Axolotl로 QLoRA 파인튜닝할 때의 성능을 측정하기 위한 벤치마크 도구입니다. Docker 기반으로 환경을 격리해 재현성을 보장하며, 단일 GPU부터 MIG 파티션, DDP, FSDP2까지 5가지 실행 모드를 지원합니다.

## 주요 기능

- **5가지 벤치마크 모드**: MIG 파티션, 단일 48GB, 단일 96GB, 2x48GB DDP, 2x48GB FSDP2
- **Docker 격리 환경**: CUDA 13.0 + PyTorch 2.11 + Flash Attention 2.8.3 기반 재현 가능한 빌드
- **자동화된 실행 파이프라인**: GPU 자동 감지, 이미지 빌드, 컨테이너 실행까지 단일 스크립트로 처리
- **타임스탬프 기반 결과 관리**: 실행마다 `output/<타입>/<타임스탬프>/` 경로에 결과를 분리 저장

## 벤치마크 모드

| 모드         | GPU 구성               | 설명                                    |
| ------------ | ---------------------- | --------------------------------------- |
| `mig48`      | 96GB GPU → MIG 48GB    | MIG로 분할된 48GB 인스턴스에서 학습     |
| `single48`   | 1x 48GB GPU            | 단일 48GB GPU에서 학습                  |
| `single96`   | 1x 96GB GPU            | 단일 96GB GPU에서 학습                  |
| `ddp48`      | 2x 48GB GPU (DDP)      | DistributedDataParallel 기반 다중 GPU   |
| `fsdp48`     | 2x 48GB GPU (FSDP2)    | Fully Sharded Data Parallel v2 + QLoRA  |

## 요구 사항

### 하드웨어
- NVIDIA GPU (벤치마크 모드에 따라 48GB / 96GB 필요)
- `ddp48`, `fsdp48` 모드는 최소 2장의 GPU 필요
- `mig48` 모드는 MIG 지원 GPU(A100, H100 등)에서 사전 MIG 설정 필요

### 소프트웨어
- Docker (`buildx` 지원)
- NVIDIA Container Toolkit
- NVIDIA 드라이버 (CUDA 13.0 호환 버전)

### 모델
- LLaMA 3.1 8B 모델이 아래 경로에 사전 다운로드되어 있어야 합니다:
  ```
  ~/automated_benchmark/models/Llama-3.1-8B/
  ```

## 프로젝트 구조

```
.
├── run.sh              # 메인 실행 스크립트 (이미지 빌드 + 컨테이너 실행)
├── Dockerfile          # 벤치마크 환경 정의
├── entrypoint.sh       # 컨테이너 내부 학습 실행 스크립트
├── 48gb_single.yml     # 단일 48GB / MIG용 Axolotl 설정
├── 96gb_single.yml     # 단일 96GB용 Axolotl 설정
├── 48gb_ddp.yml        # 2x 48GB DDP용 Axolotl 설정
├── 48gb_fsdp.yml       # 2x 48GB FSDP2용 Axolotl 설정
├── output/             # 벤치마크 결과 저장 (자동 생성)
└── cache/              # 데이터셋 전처리 캐시 (자동 생성)
```

## 사용법

### 기본 실행

원하는 벤치마크 모드를 인자로 전달하면 됩니다:

```bash
./run.sh single48     # 단일 48GB GPU
./run.sh single96     # 단일 96GB GPU
./run.sh mig48        # MIG 48GB 인스턴스
./run.sh ddp48        # 2x 48GB DDP
./run.sh fsdp48       # 2x 48GB FSDP2
```

### 실행 흐름

`run.sh`를 실행하면 다음 순서로 진행됩니다:

1. **사전 점검**: 모델 파일 존재 확인, GPU 디바이스 감지
2. **Docker 이미지 빌드**: `axolotl-benchmark` 이미지 생성
3. **컨테이너 실행**: GPU를 마운트하고 `entrypoint.sh` 실행
4. **학습 수행**: `accelerate launch`로 Axolotl 학습 시작
5. **결과 저장**: 호스트의 `output/<타입>/<타임스탬프>/`에 체크포인트/로그 저장

### MIG 모드 사전 준비

`mig48`을 사용하려면 호스트에서 먼저 MIG 파티션을 생성해야 합니다. 예시:

```bash
sudo nvidia-smi -mig 1
sudo nvidia-smi mig -cgi 9,9 -C    # 96GB GPU를 48GB x 2로 분할 (프로파일은 GPU마다 상이)
nvidia-smi -L                       # MIG UUID 확인
```

스크립트는 `nvidia-smi -L`에서 첫 번째로 감지된 MIG 인스턴스를 자동 사용합니다.

## 출력 결과

각 실행 결과는 타임스탬프 기반 디렉터리에 저장됩니다:

```
output/
└── single48/
    └── 20260420-143022/
        ├── checkpoint-*/
        ├── adapter_config.json
        ├── adapter_model.safetensors
        └── ...
```

## 기술 스택

| 구성 요소         | 버전          |
| ----------------- | ------------- |
| CUDA              | 13.0.3        |
| Python            | 3.12          |
| PyTorch           | 2.11.0+cu130  |
| Flash Attention   | 2.8.3         |
| Axolotl           | 0.16.1        |
| xformers          | 0.0.35        |
| Base OS           | Ubuntu 24.04  |

## 참고 사항

- 컨테이너는 `--rm` 옵션으로 실행되므로 종료 시 자동 삭제됩니다. 결과물은 볼륨 마운트를 통해 호스트의 `output/` 디렉터리에 영구 저장됩니다.
- 데이터셋 전처리 결과는 `cache/` 디렉터리에 저장되어 재실행 시 재사용됩니다.
- `HF_HUB_OFFLINE=1`로 설정되어 있어 학습 중 Hugging Face Hub에 접속하지 않습니다. 모델은 반드시 로컬에 준비되어 있어야 합니다.
- 멀티 GPU 모드에서는 `--shm-size=16g` 옵션으로 공유 메모리가 할당되어 NCCL 통신에 사용됩니다.

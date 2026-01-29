# AnyDoor: Zero-shot 객체 단위 이미지 커스터마이징

![License](https://img.shields.io/badge/License-MIT-blue.svg)


AnyDoor는 **제로샷(Zero-shot) 객체 단위 이미지 커스터마이징** 시스템으로, 한 이미지의 객체를 다른 이미지로 자연스럽게 전송할 수 있습니다.
## 주요 기능

- **제로샷 객체 전송**: 추가 학습 없이 어떤 객체든 다른 이미지로 전송
- **현실적인 합성**: 조명, 스케일, 외형을 자동으로 적응
- **다양한 응용**: 가상 피팅, 가구 배치, 씬 편집 등
- **고품질 결과**: Stable Diffusion 2.1 기반 생성

## 작동 원리

AnyDoor는 다음 요소들을 결합합니다:

1. **참조 이미지 + 마스크**: 전송할 객체
2. **타겟 이미지 + 마스크**: 객체를 배치할 위치
3. **DINOv2 인코더**: 의미적 조건부 생성 (외형 유지)
4. **ControlNet**: 공간적 가이던스 (배치 위치)
5. **Sobel 엣지맵**: 고주파 세부사항 보존

```
참조 이미지 + 타겟 이미지 + 마스크들
    ↓
[DINOv2 인코더] → 의미적 특징 (1024차원)
    ↓
[콜라주 생성] → Sobel 엣지맵 (고주파수)
    ↓
[ControlNet] → 제어 특징 (다중 스케일)
    ↓
[Stable Diffusion 2.1] + [ControlNet 주입] → 생성된 이미지
    ↓
[자르기/붙여넣기] → 최종 결과
```

## 설치 방법

### 요구사항

- Python 3.8+
- CUDA 지원 GPU (권장: 16GB+ VRAM)
- PyTorch 2.0.0


## 사용 방법

### 단일 이미지 추론

```python
python run_inference.py
```

**입력 준비:**
- `ref_image`: 참조 객체가 있는 이미지 (512×512 권장)
- `ref_mask`: 객체 영역을 나타내는 이진 마스크
- `tar_image`: 타겟 이미지 (임의 크기)
- `tar_mask`: 객체를 배치할 영역의 이진 마스크

**파라미터:**
- `guidance_scale`: 분류기 없는 가이던스 강도 (기본값: 5.0)
- `steps`: 확산 샘플링 단계 (기본값: 50)
- `seed`: 재현성을 위한 난수 시드

### Replicate API로 추론

```python
# predict.py 사용 (Cog 호환)
python predict.py \
  --ref_image path/to/reference.jpg \
  --ref_mask path/to/ref_mask.png \
  --tar_image path/to/target.jpg \
  --tar_mask path/to/tar_mask.png \
  --control_strength 1.0 \
  --steps 50 \
  --guidance_scale 5.0
```

## 프로젝트 구조

```
Anydoor-main/
├── cldm/                   # Control Latent Diffusion Model (핵심 아키텍처)
│   ├── cldm.py            # ControlNet + ControlLDM 클래스
│   ├── ddim_hacked.py     # DDIM 샘플러 구현
│   └── model.py           # 모델 로딩 유틸리티
├── ldm/                    # Latent Diffusion Model (Stable Diffusion 기반)
│   ├── models/            # 핵심 확산 모델 컴포넌트
│   │   ├── autoencoder.py # VAE 오토인코더
│   │   └── diffusion/     # DDPM, DDIM, DPM-Solver 샘플러
│   ├── modules/           # 네트워크 빌딩 블록
│   │   ├── attention.py   # 교차 주의 메커니즘
│   │   ├── encoders/      # 텍스트/이미지 인코더 (CLIP, DINOv2)
│   │   └── diffusionmodules/ # UNet, ResBlocks
│   └── util.py            # 일반 유틸리티
├── dinov2/                # Facebook DINOv2 (비전 인코더)
│   └── dinov2/models/     # Vision Transformer 아키텍처
├── datasets/              # 데이터셋 구현
│   ├── base.py           # BaseDataset 클래스
│   └── data_utils.py     # 이미지/마스크 처리 유틸리티
├── configs/               # 설정 파일
│   ├── anydoor.yaml      # 모델 아키텍처 설정
│   └── inference.yaml    # 추론 설정
├── run_inference.py       # 단일 이미지 추론 스크립트
├── run_train_anydoor.py  # 학습 스크립트
├── predict.py            # Cog 예측기 (Replicate 배포용)
└── requirements.txt      # Python 의존성
```

## 핵심 파일 설명

### run_inference.py
단일 이미지 추론 또는 배치 데이터셋 추론을 수행합니다.

**주요 함수:**
- `process_pairs()`: 참조와 타겟 이미지 쌍 준비
  - 마스크에서 경계 상자 추출
  - 콜라주 생성 (타겟 위치에 참조 객체 배치)
  - Sobel 엣지 검출 적용
  - 512×512로 리사이징
- `inference_single_image()`: 메인 추론 파이프라인
  - 모델과 DDIM 샘플러 로드
  - 조건 설정 (참조 이미지 + 제어 힌트)
  - 확산 샘플링 실행 (50단계 DDIM)
  - 잠재 표현을 이미지로 디코딩
  - 원본 해상도로 자르기 및 붙여넣기

### cldm/cldm.py
핵심 모델 아키텍처를 정의합니다.

**ControlNet 클래스:**
- 제어 신호 주입으로 UNet 확장
- 힌트 이미지 (마스크가 있는 콜라주)를 공간 제어로 사용
- 각 해상도 레벨에 대한 제로 초기화 컨볼루션 생성

**ControlLDM 클래스:**
- 전체 파이프라인 조율
- 참조 이미지 조건화에 DINOv2 인코더 사용
- 비조건 조건화로 분류기 없는 가이던스 지원

### datasets/data_utils.py
이미지 처리 유틸리티를 제공합니다.

**마스크/박스 연산:**
- `get_bbox_from_mask()`: 이진 마스크에서 경계 상자 추출
- `expand_bbox()`: 무작위 비율로 박스 확장
- `box2squre()`: 박스를 정사각형으로 만들기

**이미지 변환:**
- `pad_to_square()`: 상수 값으로 정사각형 패딩
- `resize_and_pad()`: 종횡비 유지하며 리사이징

**엣지 검출:**
- `sobel()`: Sobel 엣지맵 계산 (고주파 가이던스)

## 학습

### 학습 데이터셋

프로젝트는 12개의 다양한 데이터셋을 사용합니다:

- **이미지 데이터**: Saliency, SAM, LVIS
- **비디오 데이터**: YouTube-VOS, VIPSeg, YouTube-VIS, UVO, MOSE
- **의류/피팅**: VITON-HD, FashionTryon
- **3D 데이터**: MVImageNet

### 학습 실행

```bash
python run_train_anydoor.py
```

**학습 설정:**
- 해상도: 512×512 (잠재 공간 64×64)
- 배치 크기: 16 (A100 GPU 2개 사용)
- 옵티마이저: Adam, 학습률 1e-5
- 반복 횟수: 최소 300,000
- 정밀도: Float16 (mixed precision)
- 전략: DDP 분산 학습

## 모델 아키텍처

### 핵심 컴포넌트

**1. DINOv2 인코더**
- 참조 이미지(224×224)에서 1024차원 의미적 특징 추출
- 고정됨, 학습 불가능한 인코더
- 생성을 위한 외형/콘텐츠 정보 제공

**2. ControlNet**
- 콜라주 이미지(512×512) + 마스크를 4채널 힌트로 사용
- 다중 스케일 공간 특징 추출
- 13개의 서로 다른 UNet 해상도 레벨에 제어 주입

**3. Stable Diffusion 2.1**
- VAE 오토인코더: 3채널 이미지 ↔ 4채널 잠재 (1/8 해상도)
- UNet: 타임스텝 + 텍스트 임베딩으로 조건화된 잠재 노이즈 제거
- 1000 확산 타임스텝, 선형 노이즈 스케줄

**4. 샘플링 전략**
- 50단계 DDIM 샘플러
- 분류기 없는 가이던스 (비조건 = 제로)
- 조정 가능한 가이던스 스케일 (추론 시 일반적으로 5.0)

## 입출력 형식

### 입력

- **객체(장애물) 이미지**: 누끼 이미지
- **참조 마스크**: 객체 경계를 정의하는 이진 마스크
- **타겟 이미지**: RGB 이미지 (임의 크기)
- **타겟 마스크**: 배치 영역을 정의하는 이진 마스크

### 출력

- 객체가 자연스럽게 삽입된 생성 이미지
- 스케일 적응 및 외형 전송 처리
- 응용: 가상 피팅, 객체 재배치, 씬 편집


<img width="9840" height="2160" alt="output" src="https://github.com/user-attachments/assets/fec23fcf-05f6-4fd6-b8c2-c29a8c535820" />

## 설정 파일

### configs/anydoor.yaml

```yaml
모델 설정:
- Stable Diffusion 2.1 백본을 사용한 ControlLDM
- ControlNet: 4채널 힌트 (콜라주+마스크)
- UNet: 320→640→1280→1280 채널 진행
- 주의 해상도: 4, 2, 1
- 트랜스포머 깊이: 1, 컨텍스트 차원: 1024
- VAE: 4차원 잠재 공간, 스케일 0.18215
- 조건화를 위한 DINOv2 인코더 (고정)
```

## 주요 혁신

AnyDoor는 다음을 독특하게 결합합니다:

1. **공간 가이던스를 위한 ControlNet** (객체 배치 위치)
2. **의미적 조건화를 위한 DINOv2** (외형 유지)
3. **고주파 힌트로서의 Sobel 엣지맵** (세밀한 디테일 보존)
4. **분류기 없는 가이던스** (생성 품질 향상)
5. **다중 데이터셋 학습** (도메인 간 일반화)

이를 통해 작업별 파인튜닝 없이 제로샷 객체 커스터마이징이 가능하며, 가상 피팅, 가구 배치, 씬 편집 등에 적용할 수 있습니다.

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 상업적 사용이 허용됩니다.



## 리소스

- 📄 **논문**: https://arxiv.org/abs/2307.09481
- 🎨 **HuggingFace 데모**: https://huggingface.co/spaces/xichenhku/AnyDoor-online
- 🌐 **ModelScope 데모**: https://modelscope.cn/studios/damo/AnyDoor-online/summary
- 🚀 **Replicate**: https://replicate.com/lucataco/anydoor
- 💾 **체크포인트**: ModelScope 및 HuggingFace에서 제공



---



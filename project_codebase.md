# Dual Arm Visuomotor Imitation Learning Codebase

## README.md
```md
# Dual Arm Visuomotor Imitation Learning Project

This is an **independent Visuomotor Imitation Learning (IL)** project that is completely isolated from the existing Reinforcement Learning (RL, `dual_arm0`) environment.
It supports the entire pipeline, from teleoperation demonstration data collection in the Isaac Sim environment to the training and simulation evaluation of three state-of-the-art robot manipulation imitation learning algorithms (**BC, Diffusion Policy, ACT**).

**Key Feature:** This project heavily utilizes **Visuomotor policies**. It introduces an RGB Front Camera (320x240) to the simulation. Ground truth object states are explicitly removed from the policy observation space; instead, the agents must infer the object state purely from the visual input (via a versatile Vision Encoder supporting ResNet, MobileNet, EfficientNet, and ViT) combined with proprioception (joint states & TCP poses).

---

## 1. Project Structure

```text
/home/optimus/isaac_lab/dual_arm_il_visuo/
├── README.md                  # Project usage guide (This document)
├── requirements.txt           # Required Python libraries list
├── run_pipeline.sh            # Automated pipeline script (Collect -> Train -> Eval)
├── configs/                   # Environment and model hyperparameters
│   ├── env_cfg.py             # Teleop/IL custom Isaac Lab environment settings (Camera added, grasping tuned)
│   ├── bc_cfg.yaml            # Behavior Cloning settings
│   ├── diffusion_cfg.yaml     # Diffusion Policy settings
│   └── act_cfg.yaml           # ACT (Action Chunking with Transformers) settings
├── teleop/                    # Teleoperation & Data collection
│   ├── dual_arm_teleop.py     # Active-Arm Toggle SE(3) keyboard controller
│   └── collect_demos.py       # Isaac Sim teleop and HDF5 episode saving script (Saves images)
├── dataset/                   # Dataset pipeline
│   └── il_dataset.py          # PyTorch HDF5 Dataset (Image preprocessing, Normalization, Horizon slicing, Chunking)
├── models/                    # 3 Major Imitation Learning Algorithms
│   ├── vision_encoder.py      # Versatile Vision Backbone (Supports ResNet, MobileNet, EfficientNet, ViT)
│   ├── bc/                    # MLP / RNN Behavior Cloning
│   ├── diffusion/             # 1D Temporal UNet Diffusion Policy (Chi et al. 2023)
│   └── act/                   # CVAE + Transformer ACT (Zhao et al. 2023)
├── scripts/                   # Execution scripts
│   ├── generate_scripted_demos.py          # Script-based single demo auto-generator
│   ├── generate_scripted_demos_parallel.py # [Recommended] Parallel high-speed demo auto-generator
│   ├── replay_demos.py                     # Verify collected HDF5 demos via simulation replay
│   ├── train.py                            # Integrated high-speed GPU training script for the 3 algorithms
│   ├── eval.py                             # Sequential policy rollout evaluation in Isaac Sim
│   └── eval_parallel.py                    # Parallel large-scale policy evaluation in Isaac Sim
├── data/                      # Directory for collected demo files (.hdf5)
└── checkpoints/               # Directory for trained model checkpoints
```

---


## 1.5. Recent Major Optimizations 🚀

To maximize training and evaluation throughput, the following system-level optimizations have been applied to this project:

1. **Catastrophic Chunk Thrashing Fix (HDF5 Locality)**:
   - When training with large Visuomotor datasets (HDF5 containing massive image arrays), PyTorch's default `random_split` and `shuffle=True` random sampler caused catastrophic "Chunk Thrashing" inside DataLoader workers due to completely random disk I/O.
   - **Solution**: We implemented `ChunkedRandomSampler` in `scripts/train.py`, which groups indices into large sequential chunks (e.g., 20,000) preserving spatial locality while shuffling. This single fix guarantees a 99%+ HDF5 cache hit rate, effectively eliminating the CPU/Disk I/O bottleneck.

2. **Ultra-Fast GPU Data Augmentation**:
   - Previously, torchvision transformations (`ColorJitter`, `RandomCrop`) were applied on the CPU inside the dataset's `__getitem__`. This bottlenecked training when GPU batch sizes were large.
   - **Solution**: We moved the augmentation logic directly into the GPU training loop. The Dataset now outputs `uint8` tensors (saving 75% PCIe bandwidth). Inside `train.py`, these are transferred to the GPU, cast to `float32`, divided by `255.0`, and then augmented via `apply_gpu_augmentation()` utilizing raw GPU horsepower.

3. **Asynchronous Video Encoding (AsyncVideoWriter)**:
   - When running parallel headless evaluation (`scripts/eval_parallel.py` with `--num_envs 128`), creating video grids and synchronously writing to `cv2.VideoWriter` completely blocked the main Reinforcement Learning loop, causing the GPU to wait for the CPU encoder.
   - **Solution**: We introduced `AsyncVideoWriter` using Python threading and queues. The main RL loop pushes `numpy` frames into a non-blocking queue in O(1) time, while a background daemon thread handles the heavy `mp4v` compression. This allows 100% GPU utilization during evaluation. We also added a `--record_video` toggle flag.

4. **Robust Model Resumption**:
>    - `scripts/train.py` was updated so that when resuming training from an older checkpoint (especially one trained before data augmentation was introduced), it automatically ignores the old `best_val_loss` (resetting it to infinity). This prevents distribution shift lockouts and ensures the script properly saves new `best_model.pt` weights on the newly augmented dataset.

5. **100% Visual Accuracy Kinematic Replay**:
>    - `scripts/replay_demos.py` now leverages Isaac Lab's physics by bypassing PD control and forcing explicit joint states (`env.scene["robot"].set_joint_position_target(...)`). This guarantees zero lag and perfectly replicates the collected demonstrations.

## 2. Teleoperation (Keyboard Controls)

To intuitively control both arms (14 DoF + 2 grippers), an **Active-Arm Toggle (Tab key)** method is provided.

| Category | Key | Description |
| :--- | :---: | :--- |
| **Arm Selection** | `TAB` | Toggle between Right Arm (Pick) <-> Left Arm (Place) |
| | `1` / `2` | 1: Force select Right Arm, 2: Force select Left Arm |
| **Translation** | `W` / `S` | Move X-axis Forward(+) / Backward(-) |
| | `A` / `D` | Move Y-axis Left(+) / Right(-) |
| | `Q` / `E` | Move Z-axis Up(+) / Down(-) |
| **Rotation** | `Z` / `X` | Roll (X-axis rotation) |
| | `T` / `G` | Pitch (Y-axis rotation) |
| | `C` / `V` | Yaw (Z-axis rotation) |
| **Gripper** | `SPACE` or `K` | Toggle open/close gripper of the active arm |
| | `O` / `P` | Force open (`O`) / close (`P`) gripper of the active arm |
| **Data Collection** | `ENTER` or `Y` | **[Save Success]** Save the current episode to HDF5 and reset |
| | `BACKSPACE` or `N` | **[Discard]** Discard the mistaken episode without saving and reset |
| | `R` | Reset environment immediately |
| **Speed Control** | `UP` / `DOWN` | Increase / Decrease movement speed per input |

---

## 3. Step-by-Step Usage Guide

### ⓪ Step 0: Activate Virtual Environment
Before running any scripts, you must activate the Isaac Lab virtual environment in your terminal:
```bash
source ~/isaac_lab/bin/activate
```

### ① Step 1: Demo Data Collection (Choose 1 of 3 methods)

#### [Method A] Script-based Single Demo Auto-generator (`generate_scripted_demos.py`)
Generates perfect, high-quality demos sequentially using a single robot when debugging or visual confirmation is needed. (Wait time optimization patch applied)

```bash
# Auto-collect 50 demos while watching the GUI screen
python scripts/generate_scripted_demos.py --num_demos=50
```

#### [Method B] Script-based Parallel Demo Auto-generator (`generate_scripted_demos_parallel.py`)
Collects high-quality demos in parallel using multiple environments simultaneously for significantly faster data collection.

```bash
# Auto-collect 100 demos in parallel using 16 environments without GUI (Headless)
python scripts/generate_scripted_demos_parallel.py --num_demos=100 --num_envs=16 --headless
```

#### [Method C] Manual Keyboard Teleoperation Collection (`collect_demos.py`)
Launches the Isaac Sim GUI and allows manual recording of successful episodes by controlling the robot directly with a keyboard.

```bash
# Collect 20 baseline demos (auto-accumulated and saved in data/demos.hdf5)
python teleop/collect_demos.py --num_demos=20
```

> **Tip**: If you make a mistake during operation, pressing `BACKSPACE` or `N` will immediately discard the data without polluting the dataset and start a new episode.

---

### ② Step 2: Verify Collected Demos (`replay_demos.py`)
Replay the recorded HDF5 trajectories in the simulation physics environment to ensure they operate stably.

```bash
# Replay demo 0
python scripts/replay_demos.py --demo_idx=0

# Sequentially replay all collected demos
python scripts/replay_demos.py --demo_idx=-1

# Replay multiple demos in parallel (e.g., 4 environments simultaneously)
python scripts/replay_demos.py --demo_idx=-1 --num_envs=4
```

---

### ③ Step 3: Imitation Learning Model Training (`train.py`)
Performs high-speed GPU parallel training in a pure PyTorch environment without launching Isaac Sim. The Vision Encoder (default: ResNet-18) is trained end-to-end with the policy algorithms. 

You can easily swap out the vision backbone by passing the `backbone_type` parameter to the `VisionEncoder` (e.g., `mobilenet_v3_small`, `efficientnet_b0`, `vit_b_16`).

```bash
# 1. Behavior Cloning (MLP Baseline)
python scripts/train.py --algo=bc --epochs=100

# 2. Diffusion Policy (1D Temporal UNet)
python scripts/train.py --algo=diffusion --epochs=150

# 3. ACT (Action Chunking with Transformers)
python scripts/train.py --algo=act --epochs=150
```

- Weights recording the lowest Validation Loss during training are automatically preserved at `checkpoints/{algo}/best_model.pt`.
- Normalization statistics for input observations/actions are saved at `checkpoints/{algo}/stats.pkl`.
- **Resume Training**: You can resume training from a saved checkpoint by passing the `--resume` flag:
  ```bash
  python scripts/train.py --algo=diffusion --epochs=100 --resume="checkpoints/diffusion/best_model.pt"
  ```
- **Training Speed Optimization (`--num_workers`)**: By default, it uses 16 CPU workers to parallel load HDF5 images (`pin_memory=True`). If epoch time is long, you can increase this up to your machine's core count.
  ```bash
  python scripts/train.py --algo=diffusion --epochs=150 --num_workers=32
  ```

---

### ④ Step 4: Isaac Sim Simulation Evaluation (Choose 1 of 2 methods)

#### [Method A] Sequential Evaluation (`eval.py`)
Connects the trained model to the simulation robot to measure the actual closed-loop success rate based on visual input.

```bash
# Evaluate Diffusion Policy sequentially
python scripts/eval.py --algo=diffusion --num_episodes=10

# Evaluate ACT (Temporal Ensembling applied)
python scripts/eval.py --algo=act --num_episodes=10
```

#### [Method B] Parallel Evaluation (`eval_parallel.py`)
Significantly accelerates large-scale evaluation (e.g. 100 episodes) by launching multiple Isaac Sim environments and batching visual policy inferences.

```bash
# Evaluate 2000 episodes in parallel using 128 headless environments for maximum GPU utilization
# (Note: Omit --record_video for maximum speed. Add it only if you want mp4 outputs)
python scripts/eval_parallel.py --algo diffusion --num_episodes 2000 --num_envs 128 --headless

# Or evaluate while recording both front and global observer cameras:
python scripts/eval_parallel.py --algo diffusion --num_episodes 2000 --num_envs 64 --headless --record_video
```

---

## 4. Camera Configuration (Visuomotor Setup)

In Isaac Lab, cameras must be explicitly defined as dataclass fields in the Scene configuration. We provide a custom `VisuomotorSceneCfg` in `configs/env_cfg.py` for this purpose.

```python
@configclass
class VisuomotorSceneCfg(DualArmSceneCfg):
    # Agent Vision Camera (Input to Vision Encoder)
    front_camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/FrontCamera", # Spawns at the root of the environment
        update_period=0.0,
        height=240, width=320,              # Camera Resolution
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(1.2, 0.0, 1.0),            # Camera Position (X, Y, Z)
            rot=(0.5, 0.5, 0.5, 0.5),       # Camera Rotation (Quaternion)
            convention="ros"
        ),
    )
```

### Adding a Wrist Camera (Eye-in-Hand)
To attach a camera dynamically to the robot's wrist without modifying the USD file, simply spawn a camera with a `prim_path` that is a child of the robot's hand link:

```python
    wrist_camera: CameraCfg = CameraCfg(
        # Spawns as a child of the robot's hand, so it automatically tracks hand movement!
        prim_path="{ENV_REGEX_NS}/Robot/panda_hand/WristCamera",
        height=240, width=320,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(...),
        # Offset is relative to the hand link
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.05), rot=(1.0, 0.0, 0.0, 0.0), convention="ros"),
    )
```
*Note: The vision encoder expects the specified input size, but ViT models will automatically resize the image to 224x224 internally.*

---

## 5. Relationship with Existing RL Environment

- All code in this folder (`dual_arm_il_visuo`) does not modify any files in `/home/optimus/isaac_lab/dual_arm0`.
- Simulation scenes (`DualArmSceneCfg`) and robot definitions are safely inherited (`configs/env_cfg.py`) and reused, meaning it causes absolutely no interference with ongoing training or tuning on the RL side.

---

# Dual Arm Visuomotor Imitation Learning (비주얼 모터 모방 학습) 프로젝트

기존 강화학습(RL, `dual_arm0`) 환경과 완벽히 격리된 **독립적인 Visuomotor 모방 학습(Imitation Learning, IL)** 프로젝트입니다.  
Isaac Sim 환경에서의 텔레오퍼레이션(수동 조작) 시연 데이터 수집부터 최신 로봇 조작 모방 학습 3대 알고리즘(**BC, Diffusion Policy, ACT**) 학습 및 시뮬레이션 평가까지 전 과정을 지원합니다.

**핵심 특징:** 이 프로젝트는 **카메라 영상을 활용(Visuomotor)**하도록 완전히 업그레이드 되었습니다. 시뮬레이션 내에 전면 카메라(RGB 320x240)가 추가되었으며, 모델의 Observation에서 물체의 실제 좌표(Ground truth state) 정보가 명시적으로 제거되었습니다. 모델은 오로지 **카메라 이미지(ResNet, MobileNet, EfficientNet, ViT 등 다양한 백본 지원)와 로봇 자신의 관절/TCP 상태만**을 보고 상황을 파악하여 행동해야 합니다.

---

## 1. 프로젝트 구조

```text
/home/optimus/isaac_lab/dual_arm_il_visuo/
├── README.md                  # 프로젝트 사용 가이드 (본 문서)
├── requirements.txt           # 필요한 파이썬 라이브러리 목록
├── run_pipeline.sh            # 데이터수집-학습-평가 자동화 쉘 스크립트
├── configs/                   # 환경 및 모델 하이퍼파라미터
│   ├── env_cfg.py             # 텔레옵/IL 맞춤형 Isaac Lab 환경 설정 (카메라 추가, 그라스핑 마찰력 튜닝)
│   ├── bc_cfg.yaml            # Behavior Cloning 설정
│   ├── diffusion_cfg.yaml     # Diffusion Policy 설정
│   └── act_cfg.yaml           # ACT (Action Chunking with Transformers) 설정
├── teleop/                    # 텔레오퍼레이션 & 데이터 수집
│   ├── dual_arm_teleop.py     # Active-Arm Toggle SE(3) 키보드 제어기
│   └── collect_demos.py       # Isaac Sim 텔레옵 및 HDF5 에피소드 저장 스크립트 (이미지 포함)
├── dataset/                   # 데이터셋 파이프라인
│   └── il_dataset.py          # PyTorch HDF5 Dataset (이미지 전처리, 정규화, Horizon 슬라이싱, Chunking)
├── models/                    # 모방학습 3대 알고리즘 구현체
│   ├── vision_encoder.py      # 다목적 비전 백본 인코더 (ResNet, MobileNet, EfficientNet, ViT 지원)
│   ├── bc/                    # MLP / RNN Behavior Cloning
│   ├── diffusion/             # 1D Temporal UNet Diffusion Policy (Chi et al. 2023)
│   └── act/                   # CVAE + Transformer ACT (Zhao et al. 2023)
├── scripts/                   # 실행 스크립트
│   ├── generate_scripted_demos.py          # 스크립트 기반 고품질 단일 데모 자동 생성기
│   ├── generate_scripted_demos_parallel.py # [추천] 스크립트 기반 고품질 데모 병렬 고속 생성기
│   ├── replay_demos.py                     # 수집된 HDF5 데모 시뮬레이션 재생 검증
│   ├── train.py                            # 3종 알고리즘 통합 고속 GPU 학습 스크립트
│   ├── eval.py                             # Isaac Sim 환경에서 비전 기반 정책 순차 평가
│   └── eval_parallel.py                    # Isaac Sim 환경에서 비전 기반 정책 병렬 초고속 평가
├── data/                      # 수집된 데모 파일 (.hdf5) 저장 경로
└── checkpoints/               # 훈련된 모델 체크포인트 저장 경로
```

---


## 1.5. 최근 주요 최적화 사항 🚀

이 프로젝트는 모델 학습 및 평가 속도를 극대화하기 위해 다음과 같은 시스템 레벨의 최적화가 완벽하게 적용되어 있습니다.

1. **데이터로더 Chunk Thrashing 방지 (HDF5 캐시 최적화)**:
   - 수십 GB 단위의 이미지 배열이 포함된 HDF5 데이터셋으로 학습할 때, PyTorch의 기본 `random_split`과 무작위 샘플러(`shuffle=True`)를 사용하면 램덤 디스크 I/O로 인해 심각한 병목 현상(Chunk Thrashing)이 발생합니다.
   - **해결책**: `scripts/train.py`에 `ChunkedRandomSampler`를 직접 구현했습니다. 인덱스들을 거대한 순차 청크(예: 20,000개)로 묶어서 섞음으로써 물리적인 메모리 지역성(Spatial Locality)을 보존합니다. 이를 통해 HDF5 캐시 히트율을 99% 이상으로 유지하여 디스크 I/O 병목을 완벽히 해결했습니다.

2. **초고속 GPU 데이터 증강 (Data Augmentation)**:
   - 기존에는 `ColorJitter`, `RandomCrop` 등의 이미지 변환 작업이 데이터셋(`__getitem__`) 내부에서 CPU를 통해 처리되었습니다. 이는 배치 크기가 커질수록 CPU 병목을 유발했습니다.
   - **해결책**: 데이터 증강 로직을 GPU 학습 루프 안으로 옮겼습니다. 데이터로더는 `uint8` 텐서를 그대로 반환하여 PCIe 대역폭 낭비를 75% 줄이고, GPU로 넘어간 뒤에야 `float32`로 변환 및 정규화(/255.0)를 수행합니다. 그 후 `apply_gpu_augmentation()` 함수가 GPU 코어를 활용해 초고속으로 이미지를 증강합니다.

3. **비동기 비디오 인코딩 (AsyncVideoWriter)**:
   - 병렬 시뮬레이션 평가(`scripts/eval_parallel.py`) 시 `--num_envs 128` 등 거대한 환경을 띄우면, 수많은 카메라 영상을 합치고 `cv2.VideoWriter`로 압축(인코딩)하는 작업이 메인 RL 루프를 동기적으로 멈춰 세워 GPU가 놀게 되는 현상이 있었습니다.
   - **해결책**: 백그라운드 스레드(Thread)와 큐(Queue)를 활용하는 `AsyncVideoWriter` 클래스를 도입했습니다. 메인 루프는 프레임을 큐에 순식간에 던져넣고 바로 다음 액션을 계산하며, 백그라운드 스레드가 남는 CPU 자원으로 여유롭게 mp4 인코딩을 수행합니다. 덕분에 평가 시 GPU 활용도를 100% 가깝게 유지할 수 있습니다. (필요시에만 녹화하는 `--record_video` 플래그도 추가)

4. **견고한 학습 이어서 하기 (Model Resumption Fix)**:
   - 데이터 증강 로직이 도입되기 전의 구형 체크포인트에서 학습을 이어서(Resume) 할 경우, 데이터 분포의 차이로 인해 이전에 기록된 `best_val_loss`를 영원히 넘지 못하는 문제가 발생할 수 있습니다.
   - **해결책**: `scripts/train.py`에서 체크포인트 로드 시 기존의 검증 손실 기록을 강제로 무시(`float("inf")`)하도록 수정하여, 변경된 데이터 분포에서도 새롭게 가장 뛰어난 `best_model.pt`를 문제없이 갱신해 나가도록 조치했습니다.

5. **100% 정확도의 키네마틱 데모 재생기 (Kinematic Replay)**:
>    - `scripts/replay_demos.py`가 물리 엔진의 PD 제어기를 거치지 않고, 로봇의 관절 상태를 직접 강제 할당(`set_joint_position_target`)하도록 고도화되었습니다. 이를 통해 제어 지연(Lag) 없이 수집된 HDF5 데모 데이터를 시각적으로 100% 동일하게 재생하고 CCTV 및 전체 관찰자 시점(Global View) 비디오로 저장할 수 있습니다.

## 2. 텔레오퍼레이션 (키보드 조작법)

양팔(14자유도 + 2개 그리퍼)을 직관적으로 다룰 수 있도록 **활성 팔 전환(Tab 키)** 방식을 제공합니다.

| 분류 | 키 | 동작 설명 |
| :--- | :---: | :--- |
| **팔 선택** | `TAB` | 오른팔(Pick) <-> 왼팔(Place) 전환 토글 |
| | `1` / `2` | 1번: 오른팔 즉시 선택, 2번: 왼팔 즉시 선택 |
| **위치 이동** | `W` / `S` | X축 전진(+) / 후진(-) |
| | `A` / `D` | Y축 좌측(+) / 우측(-) |
| | `Q` / `E` | Z축 상승(+) / 하강(-) |
| **자세 회전** | `Z` / `X` | Roll (X축 회전) |
| | `T` / `G` | Pitch (Y축 회전) |
| | `C` / `V` | Yaw (Z축 회전) |
| **그리퍼** | `SPACE` or `K` | 현재 활성 팔의 그리퍼 열기/닫기 토글 |
| | `O` / `P` | 현재 활성 팔의 그리퍼 강제 열기(`O`) / 닫기(`P`) |
| **데이터 수집** | `ENTER` or `Y` | **[성공 저장]** 현재 에피소드를 HDF5에 확정 저장 후 리셋 |
| | `BACKSPACE` or `N` | **[폐기]** 실수한 에피소드를 저장하지 않고 폐기 후 리셋 |
| | `R` | 환경 즉시 리셋 |
| **속도 조절** | `UP` / `DOWN` | 1회 입력당 이동 속도 증가 / 감소 |

---

## 3. 사용 단계별 가이드

### ⓪ 0단계: 가상환경 활성화 (필수)
모든 터미널 작업(스크립트 실행) 전에는 반드시 Isaac Lab 가상환경을 활성화해야 합니다:
```bash
source ~/isaac_lab/bin/activate
```

### ① 1단계: 데모 데이터 수집 (3가지 방법 중 선택)

#### [방법 A] 스크립트 기반 단일 데모 자동 생성기 (`generate_scripted_demos.py`)
디버깅이나 시각적 확인이 필요할 때 1대의 로봇이 순차적으로 완벽한 고품질 데모를 생성합니다. (대기 시간 최적화 패치 적용 완료)

```bash
# GUI 화면을 보면서 50개 데모 자동 수집
python scripts/generate_scripted_demos.py --num_demos=50
```

#### [방법 B] 스크립트 기반 병렬 데모 자동 생성기 (`generate_scripted_demos_parallel.py`)
다수의 환경을 동시에 실행하여 고품질 데모 데이터를 초고속으로 병렬 수집합니다. 

```bash
# 16개의 환경을 GUI 없이(Headless) 동시 실행하여 100개의 데모 초고속 병렬 수집
python scripts/generate_scripted_demos_parallel.py --num_demos=100 --num_envs=16 --headless
```

#### [방법 C] 키보드 텔레오퍼레이션 수동 수집 (`collect_demos.py`)
Isaac Sim GUI를 띄우고 직접 키보드로 조작하여 성공 에피소드를 수동 녹화합니다.

```bash
# 기본 20개 데모 수집 (data/demos.hdf5에 자동 누적 저장)
python teleop/collect_demos.py --num_demos=20
```

> **팁**: 조작 중 실수가 발생했을 때 `BACKSPACE`나 `N`을 누르면 데이터셋에 오염되지 않고 즉시 버려지며 새 에피소드가 시작됩니다.

---

### ② 2단계: 수집된 데모 검증 (`replay_demos.py`)
녹화된 HDF5 궤적이 시뮬레이션 물리 환경에서 안정적으로 동작하는지 재생해 봅니다.

```bash
# 0번 데모 1개 재생
python scripts/replay_demos.py --demo_idx=0

# 전체 수집된 데모 순차 재생
python scripts/replay_demos.py --demo_idx=-1

# 여러 개의 데모를 동시에 재생 (예: 4개 환경 동시 렌더링)
python scripts/replay_demos.py --demo_idx=-1 --num_envs=4
```

---

### ③ 3단계: 비전-모방학습 모델 훈련 (`train.py`)
Isaac Sim을 켜지 않고 순수 PyTorch 환경에서 고속 GPU 병렬 학습을 수행합니다. (기본 설정인 ResNet-18을 포함한 다양한 비전 인코더도 함께 End-to-End로 학습됩니다.)

💡 **비전 백본(Vision Backbone) 변경 가능**:
정책 파일에서 `VisionEncoder`를 생성할 때 `backbone_type` 파라미터를 넘겨주어 쉽게 모델을 교체할 수 있습니다. 
지원 목록: `resnet18`(기본), `resnet50`, `mobilenet_v3_small`, `efficientnet_b0`, `vit_b_16`

```bash
# 1. Behavior Cloning (MLP 베이스라인)
python scripts/train.py --algo=bc --epochs=100

# 2. Diffusion Policy (1D Temporal UNet)
python scripts/train.py --algo=diffusion --epochs=150

# 3. ACT (Action Chunking with Transformers)
python scripts/train.py --algo=act --epochs=150
```

- 학습 중 최저 Validation Loss를 기록한 가중치는 `checkpoints/{algo}/best_model.pt`로 자동 보존됩니다.
- 입력 관측치/액션의 정규화 통계치는 `checkpoints/{algo}/stats.pkl`에 저장됩니다.
- **학습 이어하기 (Resume)**: `--resume` 옵션을 사용해 기존에 학습된 가중치부터 이어서 학습할 수 있습니다.
  ```bash
  python scripts/train.py --algo=diffusion --epochs=100 --resume="checkpoints/diffusion/best_model.pt"
  ```
- **학습 속도 최적화 (`--num_workers`)**: 기본적으로 16개의 CPU 워커를 사용해 HDF5 이미지를 병렬로 로드(`pin_memory=True`)합니다. 1에포크 소요 시간이 길다면 장비의 코어 수에 맞춰 워커를 늘려보세요.
  ```bash
  # 32코어를 사용하여 디스크 I/O 병목 돌파 및 학습 속도 극대화
  python scripts/train.py --algo=diffusion --epochs=150 --num_workers=32
  ```

---

### ④ 4단계: Isaac Sim 시뮬레이션 평가 (2가지 방법 중 선택)

#### [방법 A] 순차 평가 (`eval.py`)
학습된 모델을 시뮬레이션 로봇에 1대1로 연결하여 실제 클로즈드 루프 성공률을 측정합니다. 모델은 제공되는 카메라 영상과 로봇 관절 상태만으로 예측을 수행합니다.

```bash
# Diffusion Policy 순차 평가
python scripts/eval.py --algo=diffusion --num_episodes=10

# ACT 평가 (Temporal Ensembling 적용)
python scripts/eval.py --algo=act --num_episodes=10
```

#### [방법 B] 병렬 평가 (`eval_parallel.py`)
다중 환경을 띄우고 이미지 처리 및 모델 추론(Inference)을 Batch 단위로 수행하여 대규모 에피소드(예: 100회) 평가 시간을 획기적으로 단축합니다.

```bash
# 128개의 환경을 GUI 없이(Headless) 동시 띄워 총 2000 에피소드 극강의 속도로 병렬 평가
# (주의: 최고 속도를 원한다면 --record_video 옵션을 빼세요. 영상 저장이 필요할 때만 추가하세요.)
python scripts/eval_parallel.py --algo diffusion --num_episodes 2000 --num_envs 128 --headless

# 또는 CCTV 격자 화면 및 전체 관찰자 시점(Global View) 비디오를 함께 녹화하며 평가:
python scripts/eval_parallel.py --algo diffusion --num_episodes 2000 --num_envs 64 --headless --record_video
```

---

## 4. 카메라 설정 (Visuomotor 셋업)

Isaac Lab에서는 카메라를 씬(Scene) 설정의 데이터클래스 필드로 명시적으로 선언해야 시뮬레이션에 정상적으로 스폰(Spawn)됩니다. 이를 위해 `configs/env_cfg.py`에 `VisuomotorSceneCfg`를 새롭게 정의했습니다.

```python
@configclass
class VisuomotorSceneCfg(DualArmSceneCfg):
    # 에이전트 입력용 정면 비전 카메라
    front_camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/FrontCamera", # 환경의 최상위 경로에 스폰 (고정됨)
        update_period=0.0,
        height=240, width=320,              # 해상도 변경
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(1.2, 0.0, 1.0),            # 카메라 위치 (X, Y, Z)
            rot=(0.5, 0.5, 0.5, 0.5),       # 카메라 회전 (쿼터니언)
            convention="ros"
        ),
    )
```

### 📷 손목 카메라 (Eye-in-Hand) 추가 방법
로봇 원본 USD 파일을 건드리지 않고, 코드를 통해서만 손목에 카메라를 달려면 `prim_path`를 로봇 손목 객체의 자식 경로로 설정하면 됩니다.

```python
    wrist_camera: CameraCfg = CameraCfg(
        # 로봇 손목(panda_hand)의 하위 객체로 스폰되므로, 로봇이 움직이면 카메라도 자동으로 따라다닙니다!
        prim_path="{ENV_REGEX_NS}/Robot/panda_hand/WristCamera",
        height=240, width=320,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(...),
        # offset은 손목 기준 '상대 좌표'가 됩니다.
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.05), rot=(1.0, 0.0, 0.0, 0.0), convention="ros"),
    )
```
*참고: ViT 백본을 사용할 경우, 입력된 해상도와 무관하게 내부적으로 224x224 크기로 자동 변환(Resize)되어 처리됩니다.*

---

## 5. 기존 RL 환경과의 관계

- 본 폴더(`dual_arm_il_visuo`)의 모든 코드는 `/home/optimus/isaac_lab/dual_arm0`의 파일을 수정하지 않습니다.
- 시뮬레이션 씬(`DualArmSceneCfg`) 및 로봇 정의는 환경에 설치된 설정을 안전하게 상속(`configs/env_cfg.py`)받아 재사용하므로, RL 쪽에서 현재 진행 중인 학습이나 튜닝에 아무런 간섭을 주지 않습니다.

---

## 6. Antigravity AI 자동 생성 프롬프트 (재구축용)

나중에 다른 프로젝트나 새로운 환경에서 지금과 **똑같은 Visuomotor 모방 학습 환경을 에러 없이 한 번에 자동 생성**하고 싶다면, Antigravity AI에게 아래 프롬프트를 그대로 복사해서 붙여넣으세요. 이 프롬프트 안에는 우리가 겪었던 Isaac Lab의 까다로운 환경 스폰 규칙과 Python 모듈 충돌 방지 노하우가 모두 담겨 있습니다.

> **[복사할 프롬프트 시작]**
> Isaac Lab 기반의 양팔 로봇(Dual Arm)을 위한 독립적인 Visuomotor 모방 학습(Imitation Learning) 프로젝트를 구축해 줘. 
> 
> 먼저, 반드시 아래와 같이 정확히 동일한 폴더 및 파일 구조로 작성해 줘:
> 
> ```text
> /home/optimus/isaac_lab/dual_arm_il_visuo/
> ├── README.md
> ├── requirements.txt
> ├── run_pipeline.sh
> ├── configs/
> │   ├── env_cfg.py
> │   ├── bc_cfg.yaml
> │   ├── diffusion_cfg.yaml
> │   └── act_cfg.yaml
> ├── teleop/
> │   ├── dual_arm_teleop.py
> │   └── collect_demos.py
> ├── dataset/
> │   └── il_dataset.py
> ├── models/
> │   ├── vision_encoder.py
> │   ├── bc/
> │   ├── diffusion/
> │   └── act/
> ├── scripts/
> │   ├── generate_scripted_demos.py
> │   ├── generate_scripted_demos_parallel.py
> │   ├── replay_demos.py
> │   ├── train.py
> │   ├── eval.py
> │   └── eval_parallel.py
> └── data/
> ```
> 
> 다음의 아키텍처 원칙과 Isaac Lab의 매우 중요한 규칙들을 준수해서 위 파일들의 코드를 채워 줘:
> 
> 1. **씬(Scene) 및 카메라 스폰 규칙 (가장 중요)**: 
>    `configs/env_cfg.py`에서 기존 `DualArmSceneCfg`를 상속받는 `VisuomotorSceneCfg`를 만들고, `@configclass` 데이터클래스 필드로서 `front_camera: CameraCfg`를 명시적으로 선언해 줘. (절대 `__post_init__` 내부에서 `self.scene.front_camera` 형태로 동적 할당하지 말 것! Isaac Lab의 InteractiveScene이 카메라를 인식하지 못하고 스폰을 누락시킴).
> 
> 2. **관측치(Observation) 설정**:
>    `VisuomotorObsCfg`는 `ObservationManagerCfg`를 상속받지 말고 단독 `@configclass`로 작성해 줘. 내부에 `image` (카메라) 그룹과 `policy` (Proprioception: joint 및 TCP) 그룹을 포함시켜야 해. 그리고 이 관측치 설정을 `DualArmILEnvCfg`의 클래스 레벨 필드로 지정해 줘.
> 
> 3. **독립적인 환경 레지스트리 강제 등록**:
>    기존 RL 환경(`dual_arm0`) 설정이 오버라이드 되는 것을 막기 위해, 모든 실행 스크립트(`generate_scripted_demos.py`, `eval.py`, `collect_demos.py`)의 최상단에서 `gym.register`를 사용하여 `"Isaac-Dual-Arm-IL-v0"`라는 완전히 새로운 ID를 등록해. 이때 `kwargs={"env_cfg_entry_point": DualArmILEnvCfg}`를 반드시 전달하고, `gym.make` 호출 시에도 이 새로운 ID를 사용해 줘.
> 
> 4. **로컬 모듈 임포트 강제 (Shadowing 방지)**:
>    설정 파일 등을 임포트할 때 절대 `try... except ModuleNotFoundError` 블록을 사용해서 예전 `dual_arm_il` 패키지를 임포트하려고 시도하지 마. 사용자의 시스템에 예전 버전이 pip로 설치되어 있을 경우 로컬 코드의 수정사항이 완전히 무시되는 Shadowing 현상이 발생해. 무조건 로컬 임포트(`from configs.env_cfg import DualArmILEnvCfg`)만 사용해 줘.
> 
> 5. **비전 인코더 구현**:
>    `models/vision_encoder.py`를 만들고, 입력 이미지를 처리하는 다목적 Vision Encoder를 PyTorch로 작성해 줘. `backbone_type` 파라미터를 통해 `resnet18`, `resnet50`, `mobilenet_v3_small`, `efficientnet_b0`, `vit_b_16`를 동적으로 스왑할 수 있어야 해. (ViT 모델을 위해 이미지가 224x224로 자동 리사이징 되도록 처리해 줘).
> 
> 6. **스크립트 기반 고품질 데모 생성기**:
>    `generate_scripted_demos.py`를 작성하여 DLS IK(역운동학)를 이용해 부드러운 양팔 궤적(Pick & Place)을 자동으로 생성하고 HDF5로 저장하도록 해 줘. 에피소드 저장 시에는 반드시 `image` 관측치 데이터도 함께 저장해야 해.
>
> 7. **데이터셋 지연 로딩 (OOM 프리징 방지)**:
>    수십 GB에 달하는 이미지 데이터가 포함된 HDF5를 로드하는 `il_dataset.py`를 작성할 때, `__init__`에서 절대 이미지 배열 전체를 메모리에 올리지 마. `__getitem__` 내부에서 `h5py.File`을 통해 그때그때 필요한 시퀀스만 읽어오는 지연 로딩(Lazy-loading) 방식으로 구현해야 시스템 프리징을 막을 수 있어.
>
> 8. **TensorBoard 로깅 적용**:
>    학습 스크립트(`train.py`) 작성 시 `torch.utils.tensorboard`의 `SummaryWriter`를 도입하여 Train/Val Loss와 Learning Rate가 기록되도록 코드를 구성해 줘.
> 
> 9. **각 파일별 핵심 내용 (정확한 동작을 위한 세부 지침)**:
>    - `configs/env_cfg.py`: `DualArmSceneCfg`를 상속한 `VisuomotorSceneCfg` 작성 (손목 카메라 옵션 포함). `ObservationManagerCfg` 없이 `VisuomotorObsCfg`를 독립 작성.
>    - `dataset/il_dataset.py`: HDF5 파일에서 `obs`, `actions`만 메모리에 올리고 `images`는 `self.get_h5_file()`를 통해 `__getitem__`에서 지연 로딩. 이미지 텐서는 `(C, H, W)` 형태로 0~1 사이 float32 정규화.
>    - `models/vision_encoder.py`: `torchvision.models`를 활용하여 `resnet18`, `resnet50`, `vit_b_16` 등의 백본을 불러오고, 마지막 FC 레이어를 제거하여 1D Feature Vector를 반환하는 `VisionEncoder` 클래스 구현.
> 
>    - `scripts/train.py`: HDF5 I/O 병목 방지를 위한 `ChunkedRandomSampler` 구현. PCIe 대역폭 절약을 위해 데이터로더에서는 `uint8`을 반환하고 GPU에서 `float32` 캐스팅 및 `ColorJitter`, `RandomCrop` 증강을 초고속으로 수행하는 로직 반영.
>    - `scripts/eval_parallel.py`: RL 루프의 CPU 병목을 막기 위해 큐(Queue)와 백그라운드 스레드(Thread)를 활용하는 `AsyncVideoWriter` 클래스 도입. `--record_video` 플래그 처리 및 DDPM(100스텝) 추론 사용.

>    - `scripts/train.py`: `build_model()` 함수로 BC/Diffusion/ACT 모델 인스턴스화. `argparse`에 `--num_workers` (기본값 16) 인자를 추가하여 `DataLoader`에 전달하고 `pin_memory=True`, `persistent_workers=True`를 설정해 디스크 I/O 병목을 해결. `AdamW` 옵티마이저와 `CosineAnnealingLR` 스케줄러 사용. 에포크마다 TensorBoard에 기록하고 `best_model.pt` 저장.
>    - `scripts/eval.py` & `eval_parallel.py`: `gym.make("Isaac-Dual-Arm-IL-v0")` 호출 전 반드시 `gym.register`로 로컬 `DualArmILEnvCfg` 강제 매핑. 모델 로드 후 시뮬레이션 환경에서 루프를 돌며 평가. 병렬 평가는 `img_queue`를 이용하여 다중 환경 시각 처리 수행.
>    - `teleop/collect_demos.py`: 키보드 이벤트를 받아 로봇을 제어하고, 에피소드 성공(Y) 시 현재까지의 `obs`, `actions`, `images`를 버퍼에서 HDF5로 `demo_0`, `demo_1` 그룹으로 Append.
>    - `scripts/generate_scripted_demos_parallel.py`: Isaac Lab의 `DifferentialInverseKinematics`를 활용하여 다중 환경에서 병렬로 Pick & Place 궤적을 고속 생성하고 HDF5에 자동 저장.
>    - `run_pipeline.sh`: 데이터 수집(`generate_scripted_demos_parallel.py`), 학습(`train.py`), 병렬 평가(`eval_parallel.py`)를 논스톱으로 수행하는 자동화 스크립트.
> 10. **코드 100% 완벽 복제 지침 (`setup_project.sh` 또는 `project_codebase.md`)**:
>     각 파일의 로직을 스스로 처음부터 다시 구현할 필요 없이, 프로젝트에 동봉된 `setup_project.sh` 스크립트를 실행하거나 `project_codebase.md` 파일을 읽어줘. 이 파일들 안에는 3,400줄에 달하는 모든 핵심 코드들이 하드코딩 되어 있어서, 스크립트를 실행하거나 마크다운의 코드 블록을 그대로 파싱하여 저장하면 100% 동일하게 에러 없이 자동 생성될 거야. 
> **[복사할 프롬프트 끝]**

```

## requirements.txt
```txt
# Dependencies for Imitation Learning in Isaac Lab
h5py>=3.8.0
pyyaml>=6.0
torch>=2.0.0
numpy>=1.22.0
scipy>=1.9.0
tqdm>=4.65.0
einops>=0.7.0

```

## run_pipeline.sh
```sh
#!/bin/bash
# ==============================================================================
# Dual Arm Imitation Learning (Visuomotor): Full Pipeline Script 
# ==============================================================================

set -e  # 에러 발생 시 즉시 종료

# 설정 (자율적으로 수정 가능)
ALGO="diffusion"
NUM_DEMOS=2000   # Handover는 난이도가 높으므로 많은 데모 권장
NUM_ENVS=16      # 병렬 환경 개수 (메모리에 맞게 조절)
EPOCHS=1500      # 학습 에폭
RUN_EVAL=true    # 병렬 평가(eval_parallel.py)를 실행할지 여부 (true/false)
NUM_EVAL_EPISODES=100 # 평가할 에피소드 수
PYTHON_EXEC="~/isaac_lab/bin/python"

echo "============================================================================="
echo "🚀 Visuomotor 파이프라인 시작 (Data Collection -> Train -> Eval)"
echo "알고리즘: $ALGO | 데모 개수: $NUM_DEMOS | 학습 에폭: $EPOCHS"
echo "============================================================================="

# 1. 기존 결과 파일 초기화
> pipeline_summary.txt

# 2. 데이터 수집
echo -e "\n[1/3] 🎥 $NUM_DEMOS 개의 Visuomotor 데모 데이터 수집 중... (병렬 $NUM_ENVS 환경)"
eval $PYTHON_EXEC scripts/generate_scripted_demos_parallel.py --num_demos $NUM_DEMOS --num_envs $NUM_ENVS --headless
echo "✅ 데이터 수집 완료."

# 3. 모델 학습
echo -e "\n[2/3] 🧠 $ALGO 모델 학습 중 ($EPOCHS Epochs)..."
eval $PYTHON_EXEC scripts/train.py --algo $ALGO --epochs $EPOCHS --num_workers 32
echo "✅ 학습 완료."

# 4. 병렬 평가 (eval_parallel.py)
if [ "$RUN_EVAL" = true ] || [ "$RUN_EVAL" = "true" ]; then
    echo -e "\n[3/3] 🚀 병렬 평가 (eval_parallel.py) $NUM_EVAL_EPISODES 에피소드 진행 중..."
    PAR_LOG="results_eval_parallel.txt"
    eval $PYTHON_EXEC scripts/eval_parallel.py --algo $ALGO --num_episodes $NUM_EVAL_EPISODES --num_envs $NUM_ENVS --headless > $PAR_LOG 2>&1 || true
    PAR_SUCCESS=$(grep "Success Rate" $PAR_LOG | tail -n 1 | xargs)
    echo "✅ 병렬 평가 완료. ($PAR_SUCCESS)"
    echo "[eval_parallel.py] $PAR_SUCCESS" >> pipeline_summary.txt
else
    echo -e "\n[3/3] ⏭️ 병렬 평가 (eval_parallel.py) 건너뜀 (RUN_EVAL=false)"
    echo "[eval_parallel.py] Skipped" >> pipeline_summary.txt
fi

echo "============================================================================="
echo "🎉 Visuomotor 파이프라인이 종료되었습니다! 최종 결과 요약:"
cat pipeline_summary.txt
echo "============================================================================="

```

## configs/env_cfg.py
```py
# Copyright (c) 2026, Dual Arm Imitation Learning Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Environment configuration for Dual Arm Imitation Learning.

This configuration inherits directly from the RL environment (dual_arm0)
without modifying any RL code, while customizing parameters for
teleoperation, demonstration collection, and IL policy rollout.
"""

from isaaclab.utils import configclass
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg
import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils

# Import the base configuration from dual_arm0 (installed in environment)
from dual_arm0.tasks.dual_arm.dual_arm_env_cfg import DualArmEnvCfg, DualArmSceneCfg
from dual_arm0.tasks.dual_arm import rewards as base_rewards

from isaaclab.sensors import CameraCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm


@configclass
class VisuomotorImageObsCfg(ObsGroup):
    """Image observations for the Visuomotor policy."""
    rgb = ObsTerm(
        func=mdp.image,
        params={"sensor_cfg": SceneEntityCfg("front_camera"), "data_type": "rgb"},
    )
    
    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = False

import dual_arm0.tasks.dual_arm.observations as custom_obs
@configclass
class VisuomotorProprioCfg(ObsGroup):
    """Proprioceptive observations for policy network."""
    # Joint positions and velocities
    joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot")})
    joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot")})
    
    # TCP poses (no object-relative terms)
    pick_tcp_pos = ObsTerm(func=custom_obs.pick_tcp_pos_w, params={"asset_name": "robot", "pick_hand_regex": "panda_hand_0"})
    pick_tcp_quat = ObsTerm(func=custom_obs.pick_tcp_quat_w, params={"asset_name": "robot", "pick_hand_regex": "panda_hand_0"})
    place_tcp_pos = ObsTerm(func=custom_obs.place_tcp_pos_w, params={"asset_name": "robot", "place_hand_regex": "panda_hand$"})
    place_tcp_quat = ObsTerm(func=custom_obs.place_tcp_quat_w, params={"asset_name": "robot", "place_hand_regex": "panda_hand$"})
    
    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True

@configclass
class VisuomotorObsCfg:
    image: VisuomotorImageObsCfg = VisuomotorImageObsCfg()
    policy: VisuomotorProprioCfg = VisuomotorProprioCfg()


@configclass
class VisuomotorSceneCfg(DualArmSceneCfg):
    # Explicitly define the camera as a dataclass field so InteractiveScene spawns it!
    front_camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/FrontCamera",
        update_period=0.0,
        height=240,
        width=320,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=14.0, focus_distance=400.0, horizontal_aperture=20.955
        ),
        offset=CameraCfg.OffsetCfg(pos=(1.8, 0.0, 1.5), rot=(-0.2883, 0.6457, 0.6457, -0.2883), convention="ros"),
    )

@configclass
class DualArmILEnvCfg(DualArmEnvCfg):
    """Environment configuration tailored for Imitation Learning."""
    
    # Define at the class level so dataclasses.fields() detects it!
    observations: VisuomotorObsCfg = VisuomotorObsCfg()
    scene: VisuomotorSceneCfg = VisuomotorSceneCfg()
    
    def __post_init__(self):
        super().__post_init__()
        
        # Override the configs AFTER the parent class initializes
        self.observations = VisuomotorObsCfg()
        
        # Re-initialize the scene using our new class that has the camera
        self.scene = VisuomotorSceneCfg()

        # For teleoperation and evaluation, default to 1 environment
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5

        # Episode settings for teleoperation & evaluation
        self.episode_length_s = 50.0  # 50 seconds per episode (1500 steps at 30Hz)
        self.decimation = 2           # 60Hz / 2 = 30Hz control frequency

        # Camera viewer position for comfortable human teleoperation view
        self.viewer.eye = (1.2, 0.0, 1.0)
        self.viewer.lookat = (0.35, 0.0, 0.2)

        # -------------------------------------------------------------
        # Gripper Clamping Force & Grasp Stability Enhancements
        # -------------------------------------------------------------
        import copy
        self.scene.robot = copy.deepcopy(self.scene.robot)
        self.scene.object = copy.deepcopy(self.scene.object)

        if "hand" in self.scene.robot.actuators:
            self.scene.robot.actuators["hand"].stiffness = 3000.0
            self.scene.robot.actuators["hand"].damping = 100.0
            self.scene.robot.actuators["hand"].effort_limit = 300.0

        if hasattr(self.scene.object.spawn, "mass_props") and self.scene.object.spawn.mass_props is not None:
            self.scene.object.spawn.mass_props.mass = 0.1

        if hasattr(self.scene.object, "spawn") and hasattr(self.scene.object.spawn, "size"):
            self.scene.object.spawn.size = (0.04, 0.04, 0.25)
        if hasattr(self.scene.object, "init_state") and hasattr(self.scene.object.init_state, "pos"):
            self.scene.object.init_state.pos = (0.5, 0.0, 0.020)

        if hasattr(self.scene.object.spawn, "physics_material"):
            self.scene.object.spawn.physics_material = sim_utils.RigidBodyMaterialCfg(
                static_friction=3.0,
                dynamic_friction=3.0,
                friction_combine_mode="max",
                restitution=0.0,
                restitution_combine_mode="min",
            )

        if hasattr(self.scene, "target") and hasattr(self.scene.target, "init_state"):
            self.scene.target.init_state.pos = (0.5, 0.5, 0.001)
            if hasattr(self.scene.target, "spawn") and hasattr(self.scene.target.spawn, "radius"):
                self.scene.target.spawn.radius = 0.125

        if hasattr(self, "events") and hasattr(self.events, "reset_target"):
            self.events.reset_target.params["pose_range"] = {"x": (-0.2, 0.2), "y": (-0.2, 0.2), "z": (0.0, 0.0)}



```

## configs/bc_cfg.yaml
```yaml
# Behavior Cloning (BC) Configuration

algo: "bc"
model_type: "mlp" # "mlp" or "rnn"

# Network architecture
hidden_dims: [512, 512, 256]
activation: "relu"
dropout: 0.1

# RNN specific (if model_type == "rnn")
rnn_hidden_dim: 256
rnn_num_layers: 2
seq_len: 10

# Training hyperparameters
learning_rate: 1.0e-3
weight_decay: 1.0e-5
batch_size: 128
epochs: 100
lr_drop_epoch: 60

# Checkpointing & Logging
save_interval: 10
eval_interval: 5

```

## configs/diffusion_cfg.yaml
```yaml
# Diffusion Policy Configuration (1D Temporal UNet)

algo: "diffusion"

# Horizon parameters
pred_horizon: 24       # Number of future actions predicted by model (Tp)
obs_horizon: 2         # Number of historical observations conditioned on (To)
act_horizon: 8         # Number of action steps executed before replanning (Ta)

# UNet Architecture
down_dims: [256, 512, 1024]
kernel_size: 5
n_groups: 8
cond_predict_scale: true

# Diffusion Scheduler (DDPM / DDIM)
num_train_timesteps: 100
beta_schedule: "squaredcos_cap_v2" # or "linear"
prediction_type: "epsilon"         # predict noise epsilon or sample x_0
num_inference_steps: 15            # DDIM accelerated sampling steps for fast closed-loop control

# Training hyperparameters
learning_rate: 1.0e-4
weight_decay: 1.0e-6
batch_size: 64
epochs: 150
lr_scheduler: "cosine"
lr_warmup_steps: 500

# Checkpointing & Logging
save_interval: 10
eval_interval: 5

```

## configs/act_cfg.yaml
```yaml
# Action Chunking with Transformers (ACT) Configuration

algo: "act"

# Chunking parameters
chunk_size: 24         # Action chunk length (num future actions predicted per step)
kl_weight: 10.0        # Weight for CVAE latent KL divergence loss

# Transformer architecture
d_model: 256
nhead: 8
num_encoder_layers: 4
num_decoder_layers: 6
dim_feedforward: 1024
dropout: 0.1
latent_dim: 32         # Dimension of CVAE style latent variable z

# Inference / Rollout
temporal_ensembling: true
temporal_ensemble_coeff: 0.01  # Exponential weighting for overlapping action predictions

# Training hyperparameters
learning_rate: 1.0e-4
weight_decay: 1.0e-4
batch_size: 64
epochs: 150
lr_drop_epoch: 100

# Checkpointing & Logging
save_interval: 10
eval_interval: 5

```

## teleop/dual_arm_teleop.py
```py
# Copyright (c) 2026, Dual Arm Imitation Learning Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Dual Arm Teleoperation Controller for Isaac Sim.

Provides SE(3) task-space control for both Franka arms with an Active-Arm Toggle (Tab key),
Differential IK calculation, and episode recording controls (Confirm/Discard/Reset).
"""

from __future__ import annotations

import weakref
from collections.abc import Callable
import numpy as np
import torch
from scipy.spatial.transform import Rotation

import carb
import omni


class DualArmTeleopController:
    """Keyboard controller for Dual Franka arms in Isaac Lab.

    Key bindings:
        -------------------------------------------------------------
        [Arm Selection]
        TAB              : Toggle active arm (Right Arm <-> Left Arm)
        1                : Force select Right Arm (Pick Arm)
        2                : Force select Left Arm (Place Arm)

        [Task-Space Movement (Active Arm)]
        W / S            : Move Forward / Backward (+X / -X)
        A / D            : Move Left / Right (+Y / -Y)
        Q / E            : Move Up / Down (+Z / -Z)
        Z / X            : Roll (+ / -)
        T / G            : Pitch (+ / -)
        C / V            : Yaw (+ / -)

        [Gripper Control]
        SPACE or K       : Toggle Gripper of active arm (Open <-> Close)
        O                : Open active gripper
        P                : Close active gripper

        [Data Collection Workflow]
        ENTER or Y       : Confirm & Save current episode to HDF5
        BACKSPACE or N   : Discard current episode and reset
        R                : Reset environment

        [Sensitivity]
        UP / DOWN        : Increase / Decrease translation sensitivity
        -------------------------------------------------------------
    """

    def __init__(self, device: str = "cuda:0"):
        self.device = device

        # Sensitivities
        self.pos_step = 0.015   # 1.5 cm per press / tick
        self.rot_step = 0.03    # ~1.7 degrees per press / tick

        # Active arm tracking: "right" or "left"
        self.active_arm = "right"

        # Gripper states: True = closed (-1.0), False = open (+1.0)
        self.right_gripper_closed = False
        self.left_gripper_closed = False

        # Delta accumulators
        self.delta_pos = np.zeros(3, dtype=np.float32)
        self.delta_rot = np.zeros(3, dtype=np.float32)

        # Status flags for demonstration workflow
        self.flag_save_episode = False
        self.flag_discard_episode = False
        self.flag_reset_env = False

        # Acquire Omniverse input interface
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()

        # Subscribe to keyboard events with weakref
        self._sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=weakref.proxy(self): obj._on_keyboard_event(event, *args),
        )

        # Key state tracking for smooth continuous holding
        self._keys_down = set()

        print("[DualArmTeleopController] Initialized successfully.")
        self.print_instructions()

    def __del__(self):
        if hasattr(self, "_sub") and self._sub is not None:
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._sub)
            self._sub = None

    def print_instructions(self):
        print("=" * 65)
        print(" Dual Arm Teleoperation - Key Bindings")
        print("=" * 65)
        print("  TAB          : Toggle Active Arm (Current: " + self.active_arm.upper() + ")")
        print("  W / S        : Move X (+ / -)")
        print("  A / D        : Move Y (+ / -)")
        print("  Q / E        : Move Z (+ / -)")
        print("  Z / X        : Roll  (+ / -)")
        print("  T / G        : Pitch (+ / -)")
        print("  C / V        : Yaw   (+ / -)")
        print("  SPACE / K    : Toggle Active Gripper (Open <-> Close)")
        print("  ENTER / Y    : [SAVE] Confirm & Save Demo")
        print("  BACKSPACE / N: [DISCARD] Discard Demo & Reset")
        print("  R            : [RESET] Reset Environment")
        print("=" * 65)

    def reset_episode_flags(self):
        """Reset workflow flags at the beginning of each episode."""
        self.flag_save_episode = False
        self.flag_discard_episode = False
        self.flag_reset_env = False
        self.delta_pos[:] = 0.0
        self.delta_rot[:] = 0.0

    def reset(self):
        """Full reset of controller state."""
        self.reset_episode_flags()
        self.right_gripper_closed = False
        self.left_gripper_closed = False
        self.active_arm = "right"

    def _on_keyboard_event(self, event, *args, **kwargs):
        name = event.input.name

        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            self._keys_down.add(name)

            # --- Arm Switching ---
            if name == "TAB":
                self.active_arm = "left" if self.active_arm == "right" else "right"
                print(f"[Teleop] Switched active arm to: >> {self.active_arm.upper()} <<")
            elif name in ("NUM_1", "KEY_1"):
                self.active_arm = "right"
                print("[Teleop] Active arm: >> RIGHT (Pick) <<")
            elif name in ("NUM_2", "KEY_2"):
                self.active_arm = "left"
                print("[Teleop] Active arm: >> LEFT (Place) <<")

            # --- Gripper Toggles ---
            elif name in ("SPACE", "K"):
                if self.active_arm == "right":
                    self.right_gripper_closed = not self.right_gripper_closed
                    status = "CLOSED" if self.right_gripper_closed else "OPEN"
                    print(f"[Teleop] Right Gripper: {status}")
                else:
                    self.left_gripper_closed = not self.left_gripper_closed
                    status = "CLOSED" if self.left_gripper_closed else "OPEN"
                    print(f"[Teleop] Left Gripper: {status}")
            elif name == "O":
                if self.active_arm == "right":
                    self.right_gripper_closed = False
                else:
                    self.left_gripper_closed = False
                print(f"[Teleop] {self.active_arm.capitalize()} Gripper OPEN")
            elif name == "P":
                if self.active_arm == "right":
                    self.right_gripper_closed = True
                else:
                    self.left_gripper_closed = True
                print(f"[Teleop] {self.active_arm.capitalize()} Gripper CLOSED")

            # --- Workflow Flags ---
            elif name in ("ENTER", "Y"):
                self.flag_save_episode = True
                print("\n>>> [DEMO CONFIRMED] Saving episode to dataset! <<<")
            elif name in ("BACKSPACE", "N"):
                self.flag_discard_episode = True
                print("\n>>> [DEMO DISCARDED] Episode dropped! Resetting... <<<")
            elif name == "R":
                self.flag_reset_env = True
                print("[Teleop] Reset requested.")

            # --- Sensitivity Adjust ---
            elif name == "UP":
                self.pos_step = min(0.05, self.pos_step + 0.005)
                print(f"[Teleop] Sensitivity increased: {self.pos_step*100:.1f} cm/step")
            elif name == "DOWN":
                self.pos_step = max(0.002, self.pos_step - 0.005)
                print(f"[Teleop] Sensitivity decreased: {self.pos_step*100:.1f} cm/step")

        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            self._keys_down.discard(name)

    def get_delta_command(self) -> tuple[str, np.ndarray, np.ndarray, float, float]:
        """Poll currently held keys and compute delta position and rotation for the active arm.

        Returns:
            active_arm: "right" or "left"
            dpos: np.ndarray (3,) [dx, dy, dz] in meters
            drot: np.ndarray (3,) [droll, dpitch, dyaw] in radians
            right_gripper_val: float (-1.0 for close, +1.0 for open)
            left_gripper_val: float (-1.0 for close, +1.0 for open)
        """
        dpos = np.zeros(3, dtype=np.float32)
        drot = np.zeros(3, dtype=np.float32)

        # X axis (Forward / Backward)
        if "W" in self._keys_down:
            dpos[0] += self.pos_step
        if "S" in self._keys_down:
            dpos[0] -= self.pos_step

        # Y axis (Left / Right)
        if "A" in self._keys_down:
            dpos[1] += self.pos_step
        if "D" in self._keys_down:
            dpos[1] -= self.pos_step

        # Z axis (Up / Down)
        if "Q" in self._keys_down:
            dpos[2] += self.pos_step
        if "E" in self._keys_down:
            dpos[2] -= self.pos_step

        # Roll
        if "Z" in self._keys_down:
            drot[0] += self.rot_step
        if "X" in self._keys_down:
            drot[0] -= self.rot_step

        # Pitch
        if "T" in self._keys_down:
            drot[1] += self.rot_step
        if "G" in self._keys_down:
            drot[1] -= self.rot_step

        # Yaw
        if "C" in self._keys_down:
            drot[2] += self.rot_step
        if "V" in self._keys_down:
            drot[2] -= self.rot_step

        right_grip = -1.0 if self.right_gripper_closed else 1.0
        left_grip = -1.0 if self.left_gripper_closed else 1.0

        return self.active_arm, dpos, drot, right_grip, left_grip

```

## teleop/collect_demos.py
```py
# Copyright (c) 2026, Dual Arm Imitation Learning Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Demonstration Collection Script for Dual Arm via Teleoperation.

Collects expert demonstrations in Isaac Sim using the DualArmTeleopController,
and exports successful trajectories into an HDF5 dataset compatible with
BC, Diffusion Policy, and ACT.

Usage:
    # Run from the isaac_lab root or dual_arm_il directory:
    python /home/optimus/isaac_lab/dual_arm_il/teleop/collect_demos.py --num_demos=20
"""

import argparse
import os
import sys
import time
import h5py
import numpy as np
import torch

# Add project root and parent to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT_ROOT = os.path.dirname(PROJECT_ROOT)
for path in [PROJECT_ROOT, PARENT_ROOT]:
    if path not in sys.path:
        sys.path.insert(0, path)

# Isaac Lab AppLauncher (Must be called before importing omni or isaaclab modules)
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Collect dual arm demonstrations via teleoperation.")
parser.add_argument("--num_demos", type=int, default=20, help="Target number of demonstrations to collect.")
parser.add_argument(
    "--dataset_file",
    type=str,
    default=os.path.join(PROJECT_ROOT, "data", "demos.hdf5"),
    help="Path to the output HDF5 dataset file.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# FORCE ENABLE CAMERAS for Visuomotor project!
args_cli.enable_cameras = True

# Launch Isaac Sim simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Imports after simulation app launch
import gymnasium as gym
from isaaclab.envs import ManagerBasedRLEnv
from configs.env_cfg import DualArmILEnvCfg
from teleop.dual_arm_teleop import DualArmTeleopController

gym.register(
    id="Isaac-Dual-Arm-IL-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": DualArmILEnvCfg,
    },
)


def solve_dls_ik(
    jacobian: torch.Tensor,
    delta_pose: torch.Tensor,
    damping: float = 0.05,
) -> torch.Tensor:
    """Damped Least Squares (DLS) Inverse Kinematics.

    Args:
        jacobian: Jacobian matrix of shape (1, 6, 7).
        delta_pose: Desired task-space delta [dx, dy, dz, droll, dpitch, dyaw] of shape (1, 6).
        damping: Damping factor lambda.

    Returns:
        delta_q: Change in joint angles of shape (1, 7).
    """
    # J: (1, 6, 7)
    j = jacobian.squeeze(0)  # (6, 7)
    e = delta_pose.squeeze(0)  # (6,)

    # J * J^T + lambda^2 * I
    jjt = torch.matmul(j, j.transpose(0, 1))  # (6, 6)
    identity = torch.eye(6, device=j.device, dtype=j.dtype)
    inv_term = torch.inverse(jjt + (damping ** 2) * identity)  # (6, 6)

    # J^T * inv * e
    delta_q = torch.matmul(j.transpose(0, 1), torch.matmul(inv_term, e))  # (7,)
    return delta_q.unsqueeze(0)


def save_episode_to_hdf5(hdf5_path: str, ep_idx: int, observations: list, actions: list, rewards: list, object_poses: list, robot_joint_poses: list, init_robot_pos: list, init_robot_quat: list, init_target_pos: list, init_target_quat: list):
    """Save a single successful demonstration to the HDF5 file."""
    os.makedirs(os.path.dirname(os.path.abspath(hdf5_path)), exist_ok=True)

    mode = "a" if os.path.exists(hdf5_path) else "w"
    with h5py.File(hdf5_path, mode) as f:
        data_group = f.require_group("data")
        demo_group = data_group.create_group(f"demo_{ep_idx}")

        obs_array = np.array(observations, dtype=np.float32)
        act_array = np.array(actions, dtype=np.float32)
        rew_array = np.array(rewards, dtype=np.float32)

        demo_group.create_dataset("obs", data=obs_array, compression="gzip")
        demo_group.create_dataset("actions", data=act_array, compression="gzip")
        demo_group.create_dataset("rewards", data=rew_array, compression="gzip")
        demo_group.create_dataset("object_poses", data=np.array(object_poses, dtype=np.float32), compression="gzip")
        demo_group.create_dataset("robot_joint_poses", data=np.array(robot_joint_poses, dtype=np.float32), compression="gzip")
        demo_group.create_dataset("init_robot_pos", data=np.array(init_robot_pos, dtype=np.float32))
        demo_group.create_dataset("init_robot_quat", data=np.array(init_robot_quat, dtype=np.float32))
        demo_group.create_dataset("init_target_pos", data=np.array(init_target_pos, dtype=np.float32))
        demo_group.create_dataset("init_target_quat", data=np.array(init_target_quat, dtype=np.float32))
        demo_group.attrs["num_samples"] = len(act_array)

        # Update total samples attribute in data group
        total_samples = f["data"].attrs.get("total", 0) + len(act_array)
        f["data"].attrs["total"] = total_samples

    print(f"[Dataset] Successfully saved demo_{ep_idx} ({len(act_array)} steps) to {hdf5_path}")
    
def main():
    env_cfg = DualArmILEnvCfg()
    env_cfg.sim.device = args_cli.device

    # Initialize environment
    env: ManagerBasedRLEnv = gym.make("Isaac-Dual-Arm-IL-v0", cfg=env_cfg).unwrapped
    robot = robot

    # Locate body indices and joint indices
    # Right arm: "panda_hand_0", joints "panda_joint[1-7]_0"
    # Left arm:  "panda_hand",   joints "panda_joint[1-7]"
    right_hand_idx = robot.find_bodies("panda_hand_0")[0][0]
    left_hand_idx = robot.find_bodies("panda_hand$")[0][0]

    # Find 14 arm joint indices in articulation
    left_arm_joint_ids, _ = robot.find_joints("panda_joint[1-7]$")
    right_arm_joint_ids, _ = robot.find_joints("panda_joint[1-7]_0")

    print(f"[Collect] Right hand body index: {right_hand_idx}, Left hand body index: {left_hand_idx}")
    print(f"[Collect] Left arm joints: {left_arm_joint_ids}")
    print(f"[Collect] Right arm joints: {right_arm_joint_ids}")

    # For fixed-base articulation in PhysX, the base link is omitted in Jacobian matrices:
    is_fixed = getattr(robot, "is_fixed_base", True)
    jacobi_right_hand_idx = right_hand_idx - 1 if is_fixed else right_hand_idx
    jacobi_left_hand_idx = left_hand_idx - 1 if is_fixed else left_hand_idx

    # Inspect the Action Manager's arm_action term
    arm_term = env.action_manager._terms["arm_action"]

    # Initialize Teleop Controller
    teleop = DualArmTeleopController(device=args_cli.device)

    # Inspect existing dataset to get starting demo index
    existing_demos = 0
    if os.path.exists(args_cli.dataset_file):
        with h5py.File(args_cli.dataset_file, "r") as f:
            if "data" in f:
                existing_demos = len([k for k in f["data"].keys() if k.startswith("demo_")])
    print(f"[Collect] Existing demonstrations found: {existing_demos}")

    collected_count = existing_demos
    target_count = existing_demos + args_cli.num_demos

    # Main collection loop
    while simulation_app.is_running() and collected_count < target_count:
        obs, _ = env.reset()
        teleop.reset()

        # Commanded joint targets buffer (starts at current joint positions)
        current_joint_pos = robot.data.joint_pos.clone()
        commanded_left_joints = current_joint_pos[:, left_arm_joint_ids].clone()
        commanded_right_joints = current_joint_pos[:, right_arm_joint_ids].clone()

        ep_obs = []
        ep_actions = []
        ep_rewards = []
        object_poses = []
        robot_joint_poses = []
        init_robot_pos = robot.data.root_pos_w.cpu().numpy()[0].copy()
        init_robot_quat = robot.data.root_quat_w.cpu().numpy()[0].copy()
        init_target_pos = target.data.root_pos_w.cpu().numpy()[0].copy()
        init_target_quat = target.data.root_quat_w.cpu().numpy()[0].copy()

        print(f"\n>>> Starting Episode for Demo #{collected_count} (Target: {target_count}) <<<")

        step_idx = 0
        while simulation_app.is_running():
            step_idx += 1

            # 1. Read teleoperation command
            active_arm, dpos, drot, r_grip, l_grip = teleop.get_delta_command()

            # 2. Compute IK if there is a delta movement
            delta_pose = np.concatenate([dpos, drot])  # 6D: [dx, dy, dz, drx, dry, drz]
            has_motion = np.linalg.norm(dpos) > 1e-5 or np.linalg.norm(drot) > 1e-5

            if has_motion:
                delta_pose_t = torch.tensor(delta_pose, dtype=torch.float32, device=args_cli.device).unsqueeze(0)
                jacobians = robot.root_physx_view.get_jacobians()  # (1, num_bodies, 6, num_dofs)

                if active_arm == "right":
                    # Jacobians for right hand w.r.t right arm joints
                    j_right = jacobians[:, jacobi_right_hand_idx, :, :][:, :, right_arm_joint_ids]
                    dq_right = solve_dls_ik(j_right, delta_pose_t, damping=0.08)
                    commanded_right_joints += dq_right
                else:
                    # Jacobians for left hand w.r.t left arm joints
                    j_left = jacobians[:, jacobi_left_hand_idx, :, :][:, :, left_arm_joint_ids]
                    dq_left = solve_dls_ik(j_left, delta_pose_t, damping=0.08)
                    commanded_left_joints += dq_left

            # 3. Assemble 16D action with exact action inversion
            target_all_joints = robot.data.default_joint_pos.clone()
            target_all_joints[:, left_arm_joint_ids] = commanded_left_joints
            target_all_joints[:, right_arm_joint_ids] = commanded_right_joints

            arm_targets = target_all_joints[:, arm_term._joint_ids]
            raw_arm_action = (arm_targets - arm_term._offset) / arm_term._scale

            left_grip_t = torch.tensor([[l_grip]], dtype=torch.float32, device=args_cli.device)
            right_grip_t = torch.tensor([[r_grip]], dtype=torch.float32, device=args_cli.device)

            action = torch.cat([raw_arm_action, left_grip_t, right_grip_t], dim=-1)

            # 4. Record step transition
            policy_obs = obs["policy"].squeeze(0).detach().cpu().numpy()
            action_np = action.squeeze(0).detach().cpu().numpy()

            ep_obs.append(policy_obs)
            ep_actions.append(action_np)
            object_poses.append(obj.data.root_state_w.cpu().numpy()[0])
            robot_joint_poses.append(robot.data.joint_pos.cpu().numpy()[0])

            # 5. Step simulation
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_rewards.append(float(reward.mean().item()))

            # 6. Check workflow flags
            if teleop.flag_save_episode:
                save_episode_to_hdf5(
                    args_cli.dataset_file,
                    collected_count,
                    ep_obs,
                    ep_actions,
                    ep_rewards,
                    object_poses,
                    robot_joint_poses,
                    init_robot_pos,
                    init_robot_quat,
                    init_target_pos,
                    init_target_quat,
                )
                collected_count += 1
                teleop.reset_episode_flags()
                break

            if teleop.flag_discard_episode:
                print(f"[Collect] Episode discarded by user. Total retained: {collected_count}")
                teleop.reset_episode_flags()
                break

            if teleop.flag_reset_env or terminated.item() or truncated.item():
                if terminated.item():
                    print("[Collect] Episode terminated (e.g. object dropped). Discarding...")
                teleop.reset_episode_flags()
                break

    print(f"\n[Collect] Finished! Total demonstrations collected: {collected_count}")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()

```

## dataset/il_dataset.py
```py
# Copyright (c) 2026, Dual Arm Imitation Learning Project.
# SPDX-License-Identifier: BSD-3-Clause

"""PyTorch Dataset loader for Dual Arm Imitation Learning.

Supports HDF5 datasets generated by collect_demos.py and handles:
- Normalization (mean/std or min/max)
- Single-step slicing for standard BC
- Sequence / History slicing for RNN BC
- Horizon Chunking for Diffusion Policy (To, Tp) and ACT (chunk_size)
"""

from __future__ import annotations

import os
import pickle
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import random


class DualArmDataset(Dataset):
    """Dataset class for dual arm demonstrations stored in HDF5."""

    def __init__(
        self,
        dataset_path: str,
        algo: str = "bc",
        pred_horizon: int = 16,
        obs_horizon: int = 2,
        stats: dict | None = None,
        normalize: bool = True,
        is_train: bool = False,
    ):
        """Initialize the dataset.

        Args:
            dataset_path: Path to the HDF5 file containing 'data/demo_x'.
            algo: "bc", "diffusion", or "act".
            pred_horizon: Future action sequence length (for diffusion / act).
            obs_horizon: Past observation sequence length (for diffusion).
            stats: Precomputed normalization stats (dict with 'obs_mean', 'obs_std', etc.).
            normalize: Whether to normalize obs and actions to zero mean / unit var.
            is_train: Whether to apply data augmentation (RandomCrop, ColorJitter).
        """
        super().__init__()
        self.dataset_path = dataset_path
        self.algo = algo.lower()
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.normalize = normalize
        self.is_train = is_train

        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

        # Load all trajectories into memory for high-throughput GPU training
        self.demos_obs = []
        self.demos_act = []
        self.indices = []  # Maps flat dataset index to (demo_idx, timestep)
        self.demo_keys = []

        with h5py.File(dataset_path, "r") as f:
            data_grp = f["data"]
            self.demo_keys = sorted([k for k in data_grp.keys() if k.startswith("demo_")], key=lambda x: int(x.split("_")[1]))

            for demo_idx, key in enumerate(self.demo_keys):
                obs = np.array(data_grp[key]["obs"], dtype=np.float32)
                act = np.array(data_grp[key]["actions"], dtype=np.float32)
                ep_len = len(act)

                self.demos_obs.append(obs)
                self.demos_act.append(act)

                for t in range(ep_len):
                    self.indices.append((demo_idx, t))

        self.num_samples = len(self.indices)
        print(f"[Dataset] Loaded {len(self.demos_obs)} demos with {self.num_samples} total transitions.")

        # Compute or apply normalization statistics
        all_obs = np.concatenate(self.demos_obs, axis=0)
        all_act = np.concatenate(self.demos_act, axis=0)

        self.obs_dim = all_obs.shape[1]
        self.act_dim = all_act.shape[1]

        if stats is None:
            self.stats = {
                "obs_mean": np.mean(all_obs, axis=0),
                "obs_std": np.std(all_obs, axis=0) + 1e-6,
                "act_mean": np.mean(all_act, axis=0),
                "act_std": np.std(all_act, axis=0) + 1e-6,
                "act_min": np.min(all_act, axis=0),
                "act_max": np.max(all_act, axis=0),
            }
        else:
            self.stats = stats
            
        # We will lazy-load the HDF5 file handle per-worker to avoid RAM explosion (1000 demos * 23MB = 23GB images)
        self.h5_file = None

    def get_h5_file(self):
        """Returns a cached HDF5 file handle. Safe for num_workers=0."""
        if self.h5_file is None:
            # Allocate 2GB of RAM per worker for chunk caching to drastically reduce Gzip CPU overhead
            self.h5_file = h5py.File(self.dataset_path, "r", rdcc_nbytes=1024**3 * 2)
        return self.h5_file

    def save_stats(self, save_path: str):
        """Save normalization statistics to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(self.stats, f)
        print(f"[Dataset] Normalization statistics saved to {save_path}")

    @staticmethod
    def load_stats(stats_path: str) -> dict:
        """Load normalization statistics from disk."""
        with open(stats_path, "rb") as f:
            return pickle.load(f)

    def normalize_obs(self, obs: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        """Normalize observation."""
        if not self.normalize:
            return obs
        if isinstance(obs, torch.Tensor):
            mean = torch.as_tensor(self.stats["obs_mean"], device=obs.device, dtype=obs.dtype)
            std = torch.as_tensor(self.stats["obs_std"], device=obs.device, dtype=obs.dtype)
            return (obs - mean) / std
        return (obs - self.stats["obs_mean"]) / self.stats["obs_std"]

    def unnormalize_actions(self, act: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        """Denormalize action back to environment space."""
        if not self.normalize:
            return act
        if isinstance(act, torch.Tensor):
            mean = torch.as_tensor(self.stats["act_mean"], device=act.device, dtype=act.dtype)
            std = torch.as_tensor(self.stats["act_std"], device=act.device, dtype=act.dtype)
            return act * std + mean
        return act * self.stats["act_std"] + self.stats["act_mean"]

    def normalize_actions(self, act: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        """Normalize action."""
        if not self.normalize:
            return act
        if isinstance(act, torch.Tensor):
            mean = torch.as_tensor(self.stats["act_mean"], device=act.device, dtype=act.dtype)
            std = torch.as_tensor(self.stats["act_std"], device=act.device, dtype=act.dtype)
            return (act - mean) / std
        return (act - self.stats["act_mean"]) / self.stats["act_std"]

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        demo_idx, t = self.indices[idx]
        demo_obs = self.demos_obs[demo_idx]
        demo_act = self.demos_act[demo_idx]
        ep_len = len(demo_act)
        
        # Lazy read image from HDF5 to save memory
        demo_key = self.demo_keys[demo_idx]
        h5_f = self.get_h5_file()
        img_dataset = h5_f["data"][demo_key]["images"]

        def preprocess_image(img_arr):
            # img_arr is (..., H, W, C) uint8
            if img_arr.ndim == 3: # single image (H, W, C)
                img_t = torch.from_numpy(img_arr)
                img_t = img_t.permute(2, 0, 1).unsqueeze(0) # (1, C, H, W)
            elif img_arr.ndim == 4: # sequence of images (T, H, W, C)
                img_t = torch.from_numpy(img_arr)
                img_t = img_t.permute(0, 3, 1, 2) # (T, C, H, W)
            else:
                return img_arr
            
            # Note: Data augmentation (ColorJitter, RandomCrop) was moved to train.py 
            # to be executed on the GPU, eliminating the massive CPU bottleneck.
                
            return img_t.squeeze(0) if img_arr.ndim == 3 else img_t

        if self.algo == "bc":
            # Standard single-step Behavior Cloning
            obs = demo_obs[t]
            img = img_dataset[t]
            act = demo_act[t]

            if self.normalize:
                obs = (obs - self.stats["obs_mean"]) / self.stats["obs_std"]
                act = (act - self.stats["act_mean"]) / self.stats["act_std"]

            return {
                "obs": torch.tensor(obs, dtype=torch.float32),
                "rgb_image": preprocess_image(img),
                "action": torch.tensor(act, dtype=torch.float32),
            }

        elif self.algo == "diffusion":
            # Diffusion Policy: To past observations, Tp future actions
            start_t = max(0, t - self.obs_horizon + 1)
            obs_seq = demo_obs[start_t : t + 1]
            img_seq = img_dataset[start_t : t + 1]
            if len(obs_seq) < self.obs_horizon:
                pad_len = self.obs_horizon - len(obs_seq)
                obs_seq = np.concatenate([np.repeat(obs_seq[:1], pad_len, axis=0), obs_seq], axis=0)
                img_seq = np.concatenate([np.repeat(img_seq[:1], pad_len, axis=0), img_seq], axis=0)

            end_t = min(ep_len, t + self.pred_horizon)
            act_seq = demo_act[t:end_t]
            if len(act_seq) < self.pred_horizon:
                pad_len = self.pred_horizon - len(act_seq)
                act_seq = np.concatenate([act_seq, np.repeat(act_seq[-1:], pad_len, axis=0)], axis=0)

            if self.normalize:
                obs_seq = (obs_seq - self.stats["obs_mean"]) / self.stats["obs_std"]
                act_seq = (act_seq - self.stats["act_mean"]) / self.stats["act_std"]

            return {
                "obs": torch.tensor(obs_seq, dtype=torch.float32),
                "rgb_image": preprocess_image(img_seq),
                "action": torch.tensor(act_seq, dtype=torch.float32),
            }

        elif self.algo == "act":
            # ACT: Current observation + Action chunk of size pred_horizon
            obs = demo_obs[t]
            img = img_dataset[t]

            end_t = min(ep_len, t + self.pred_horizon)
            act_seq = demo_act[t:end_t]
            if len(act_seq) < self.pred_horizon:
                pad_len = self.pred_horizon - len(act_seq)
                act_seq = np.concatenate([act_seq, np.repeat(act_seq[-1:], pad_len, axis=0)], axis=0)

            is_pad = np.zeros(self.pred_horizon, dtype=np.float32)
            actual_len = min(self.pred_horizon, ep_len - t)
            is_pad[actual_len:] = 1.0

            if self.normalize:
                obs = (obs - self.stats["obs_mean"]) / self.stats["obs_std"]
                act_seq = (act_seq - self.stats["act_mean"]) / self.stats["act_std"]

            return {
                "obs": torch.tensor(obs, dtype=torch.float32),
                "rgb_image": preprocess_image(img),
                "action": torch.tensor(act_seq, dtype=torch.float32),
                "is_pad": torch.tensor(is_pad, dtype=torch.float32),
            }

        else:
            raise ValueError(f"Unsupported algo: {self.algo}")

```

## models/vision_encoder.py
```py
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

class VisionEncoder(nn.Module):
    def __init__(self, backbone_type="resnet18", feature_dim=512, pretrained=True):
        super().__init__()
        self.backbone_type = backbone_type.lower()
        
        # 1. Select the backbone network
        if self.backbone_type == "resnet18":
            net = models.resnet18(pretrained=pretrained)
            self.backbone = nn.Sequential(*list(net.children())[:-1])
            backbone_out_dim = 512
            
        elif self.backbone_type == "resnet50":
            net = models.resnet50(pretrained=pretrained)
            self.backbone = nn.Sequential(*list(net.children())[:-1])
            backbone_out_dim = 2048
            
        elif self.backbone_type == "mobilenet_v3_small":
            net = models.mobilenet_v3_small(pretrained=pretrained)
            self.backbone = nn.Sequential(net.features, nn.AdaptiveAvgPool2d(1))
            backbone_out_dim = 576
            
        elif self.backbone_type == "efficientnet_b0":
            net = models.efficientnet_b0(pretrained=pretrained)
            self.backbone = nn.Sequential(net.features, nn.AdaptiveAvgPool2d(1))
            backbone_out_dim = 1280
            
        elif self.backbone_type == "vit_b_16":
            net = models.vit_b_16(pretrained=pretrained)
            # Remove classification head to output the raw 768-dim class token
            net.heads = nn.Identity()
            self.backbone = net
            backbone_out_dim = 768
            
        else:
            raise ValueError(f"Unsupported backbone_type: {self.backbone_type}. Supported types: resnet18, resnet50, mobilenet_v3_small, efficientnet_b0, vit_b_16")
        
        # 2. Projection layer to match the requested feature_dim (e.g., 512)
        self.proj = nn.Linear(backbone_out_dim, feature_dim) if feature_dim != backbone_out_dim else nn.Identity()
        
        # 3. Standard ImageNet normalization
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def forward(self, x):
        # x shape: (B, C, H, W) or (B, T, C, H, W)
        is_sequence = x.ndim == 5
        if is_sequence:
            B, T, C, H, W = x.shape
            x = x.reshape(B * T, C, H, W)
            
        # ViT requires exactly 224x224 image size
        if self.backbone_type == "vit_b_16":
            x = TF.resize(x, [224, 224], antialias=True)
            
        # Normalize
        x = self.normalize(x)
        
        # Extract features
        features = self.backbone(x)
        
        # CNNs return (B*T, Channels, 1, 1), ViT returns (B*T, Channels)
        if features.dim() > 2:
            features = torch.flatten(features, 1)
        
        # Project to target dimension
        features = self.proj(features)
        
        if is_sequence:
            features = features.reshape(B, T, -1)
            
        return features

```

## scripts/generate_scripted_demos.py
```py
# Copyright (c) 2026, Dual Arm Imitation Learning Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Scripted Automatic Demonstration Generator for Dual Arm Handover Task.

Generates high-quality, optimal demonstration trajectories automatically using
Waypoint State Machine and Differential Inverse Kinematics.

Features:
- 100% automated: No human teleoperation or manual effort required.
- Robust Closed-Loop IK: DLS position control with joint limit protection.
- Exact Action Inversion: Automatically accounts for Isaac Lab's default joint offsets and action scales.
- Direct export to HDF5: Ready for immediate training with BC, Diffusion Policy, or ACT.

Usage:
    # Fast Headless generation (recommended):
    python scripts/generate_scripted_demos.py --num_demos=500 --headless

    # GUI mode (visualize robot motion):
    python scripts/generate_scripted_demos.py --num_demos=10
"""

import argparse
import os
import sys
import time
import h5py
import numpy as np
import torch

# Add project root and parent to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT_ROOT = os.path.dirname(PROJECT_ROOT)
for path in [PROJECT_ROOT, PARENT_ROOT]:
    if path not in sys.path:
        sys.path.insert(0, path)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Generate scripted demonstrations for Dual Arm Handover.")
parser.add_argument("--num_demos", type=int, default=50, help="Number of successful demonstrations to generate.")
parser.add_argument(
    "--dataset_file",
    type=str,
    default=os.path.join(PROJECT_ROOT, "data", "demos.hdf5"),
    help="Path to output HDF5 dataset.",
)
parser.add_argument("--max_steps_per_ep", type=int, default=700, help="Max steps before episode timeout.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# FORCE ENABLE CAMERAS for Visuomotor project!
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from isaaclab.envs import ManagerBasedRLEnv
from configs.env_cfg import DualArmILEnvCfg, VisuomotorObsCfg

# Explicitly register our IL environment with Gym to ensure Isaac Lab's parse_env_cfg uses our config
gym.register(
    id="Isaac-Dual-Arm-IL-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": DualArmILEnvCfg,
    },
)


def solve_pose_dls_ik(
    jacobian: torch.Tensor,
    delta_pose: torch.Tensor,
    damping: float = 0.05,
    q_current: torch.Tensor | None = None,
    q_nominal: torch.Tensor | None = None,
    k_null: float = 0.5,
) -> torch.Tensor:
    """Damped Least Squares (DLS) Inverse Kinematics with 7-DOF nullspace elbow flare projection.

    Args:
        jacobian: Full Jacobian of shape (1, 6, 7).
        delta_pose: Desired 6D delta [dx, dy, dz, wx, wy, wz] of shape (1, 6).
        damping: Damping factor lambda.
        q_current: Current joint positions of shape (1, 7).
        q_nominal: Desired nominal joint posture for nullspace projection (1, 7).
        k_null: Nullspace projection gain.

    Returns:
        delta_q: Change in joint angles of shape (1, 7).
    """
    j = jacobian.squeeze(0)  # (6, 7)
    e = delta_pose.squeeze(0)  # (6,)
    jjt = torch.matmul(j, j.transpose(0, 1))  # (6, 6)
    identity = torch.eye(6, device=j.device, dtype=j.dtype)
    inv_term = torch.inverse(jjt + (damping ** 2) * identity)
    j_pinv = torch.matmul(j.transpose(0, 1), inv_term)  # (7, 6)
    delta_q = torch.matmul(j_pinv, e)  # (7,)

    if q_current is not None and q_nominal is not None:
        eye_n = torch.eye(j.shape[1], device=j.device, dtype=j.dtype)
        null_proj = eye_n - torch.matmul(j_pinv, j)  # (7, 7)
        q_err = (q_nominal - q_current).squeeze(0)  # (7,)
        delta_q_null = torch.matmul(null_proj, k_null * q_err)  # (7,)
        delta_q = delta_q + delta_q_null

    return delta_q.unsqueeze(0)



def get_tcp_jacobian(jacobian: torch.Tensor, wrist_pos: torch.Tensor, tcp_pos: torch.Tensor) -> torch.Tensor:
    r = tcp_pos - wrist_pos  # (N, 3)
    rx, ry, rz = r[:, 0], r[:, 1], r[:, 2]
    zero = torch.zeros_like(rx)
    r_skew = torch.stack([
        torch.stack([zero, -rz, ry], dim=-1),
        torch.stack([rz, zero, -rx], dim=-1),
        torch.stack([-ry, rx, zero], dim=-1)
    ], dim=1)  # (N, 3, 3)
    j_v = jacobian[:, :3, :]   # (N, 3, 7)
    j_w = jacobian[:, 3:6, :]  # (N, 3, 7)
    j_v_tcp = j_v - torch.bmm(r_skew, j_w)
    return torch.cat([j_v_tcp, j_w], dim=1)  # (N, 6, 7)


def compute_desired_grasp_rot(
    wrist_quat: torch.Tensor,
    cube_z_dir: torch.Tensor,
    gain: float = 0.5,
    max_rot_step: float = 0.04,
    directed: bool = False,
) -> torch.Tensor:
    """Compute angular error vector to align Franka gripper perpendicular to baton.

    Ensures:
    1. Local Z axis (palm normal) points vertically downwards [0, 0, -1].
    2. Local X axis (palm lateral) is parallel to the baton longitudinal axis.
    3. Local Y axis (finger opening/closing) is perpendicular to the baton,
       pinching the 3cm thickness cleanly.

    Args:
        wrist_quat: Hand orientation quaternion (1, 4) in (w, x, y, z).
        cube_z_dir: Baton length direction unit vector (1, 3).
        gain: Proportional convergence gain.
        max_rot_step: Maximum angular velocity step in radians.
        directed: If True, do not use 180-degree symmetry. Align exactly to cube_z_dir.

    Returns:
        e_rot_step: (1, 3) angular delta vector [wx, wy, wz].
    """
    w = wrist_quat[:, 0]
    x = wrist_quat[:, 1]
    y = wrist_quat[:, 2]
    z = wrist_quat[:, 3]

    # Current rotation matrix columns in world coordinates
    x_curr = torch.stack([
        1.0 - 2.0 * (y * y + z * z),
        2.0 * (x * y + w * z),
        2.0 * (x * z - w * y),
    ], dim=-1)

    y_curr = torch.stack([
        2.0 * (x * y - w * z),
        1.0 - 2.0 * (x * x + z * z),
        2.0 * (y * z + w * x),
    ], dim=-1)

    z_curr = torch.stack([
        2.0 * (x * z + w * y),
        2.0 * (y * z - w * x),
        1.0 - 2.0 * (x * x + y * y),
    ], dim=-1)

    # Desired Z axis: straight down
    z_des = torch.tensor([[0.0, 0.0, -1.0]], device=wrist_quat.device, dtype=wrist_quat.dtype)

    # Baton horizontal projection
    baton_xy = cube_z_dir.clone()
    baton_xy[:, 2] = 0.0
    norm_xy = torch.norm(baton_xy, dim=-1, keepdim=True)
    if norm_xy.item() > 1e-4:
        x_cand = baton_xy / norm_xy
    else:
        x_cand = torch.tensor([[1.0, 0.0, 0.0]], device=wrist_quat.device, dtype=wrist_quat.dtype)

    if directed:
        x_des = x_cand
    else:
        # Align with whichever 180-degree symmetric direction is closer to current x_curr
        dot = torch.sum(x_curr * x_cand, dim=-1, keepdim=True)
        sign = torch.sign(dot + 1e-6)
        x_des = sign * x_cand

    # Desired Y axis = Z_des x X_des (finger pinching axis perpendicular to baton)
    y_des = torch.cross(z_des, x_des, dim=-1)

    # Orientation error via Lie algebra so(3) cross-product error
    e_rot = 0.5 * (
        torch.cross(x_curr, x_des, dim=-1) +
        torch.cross(y_curr, y_des, dim=-1) +
        torch.cross(z_curr, z_des, dim=-1)
    )

    # Apply gain and clamp
    e_rot_step = e_rot * gain
    err_norm = torch.norm(e_rot_step)
    if err_norm > max_rot_step:
        e_rot_step = e_rot_step * (max_rot_step / err_norm)

    return e_rot_step


def compute_tcp(wrist_pos: torch.Tensor, wrist_quat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Franka TCP position and Z-axis direction from wrist pose."""
    w, x, y, z = wrist_quat[:, 0], wrist_quat[:, 1], wrist_quat[:, 2], wrist_quat[:, 3]
    z_dir_x = 2.0 * (x * z + w * y)
    z_dir_y = 2.0 * (y * z - w * x)
    z_dir_z = 1.0 - 2.0 * (x * x + y * y)
    z_dir = torch.stack([z_dir_x, z_dir_y, z_dir_z], dim=-1)
    tcp_pos = wrist_pos + 0.1034 * z_dir
    return tcp_pos, z_dir


def save_episode_to_hdf5(hdf5_path: str, ep_idx: int, observations: list, images: list, actions: list, rewards: list, object_poses: list, robot_joint_poses: list, init_robot_pos: list, init_robot_quat: list, init_target_pos: list, init_target_quat: list):
    os.makedirs(os.path.dirname(os.path.abspath(hdf5_path)), exist_ok=True)
    mode = "a" if os.path.exists(hdf5_path) else "w"
    with h5py.File(hdf5_path, mode) as f:
        data_group = f.require_group("data")
        demo_group = data_group.create_group(f"demo_{ep_idx}")

        obs_array = np.array(observations, dtype=np.float32)
        img_array = np.array(images, dtype=np.uint8)
        act_array = np.array(actions, dtype=np.float32)
        rew_array = np.array(rewards, dtype=np.float32)

        demo_group.create_dataset("obs", data=obs_array, compression="gzip")
        # For images, gzip can be slow, but we'll use it to save disk space
        demo_group.create_dataset("images", data=img_array, compression="gzip")
        demo_group.create_dataset("actions", data=act_array, compression="gzip")
        demo_group.create_dataset("rewards", data=rew_array, compression="gzip")
        demo_group.create_dataset("object_poses", data=np.array(object_poses, dtype=np.float32), compression="gzip")
        demo_group.create_dataset("robot_joint_poses", data=np.array(robot_joint_poses, dtype=np.float32), compression="gzip")
        demo_group.create_dataset("init_robot_pos", data=np.array(init_robot_pos, dtype=np.float32))
        demo_group.create_dataset("init_robot_quat", data=np.array(init_robot_quat, dtype=np.float32))
        demo_group.create_dataset("init_target_pos", data=np.array(init_target_pos, dtype=np.float32))
        demo_group.create_dataset("init_target_quat", data=np.array(init_target_quat, dtype=np.float32))
        demo_group.attrs["num_samples"] = len(act_array)

        total_samples = f["data"].attrs.get("total", 0) + len(act_array)
        f["data"].attrs["total"] = total_samples

    print(f"[Dataset] Successfully saved demo_{ep_idx} ({len(act_array)} steps) to {hdf5_path}")


# State Machine Phase Constants
PHASE_NAMES = [
    "INIT",
    "RIGHT_HOVER",
    "RIGHT_DESCEND",
    "RIGHT_GRASP",
    "RIGHT_LIFT",
    "RIGHT_HANDOVER",
    "LEFT_APPROACH",
    "LEFT_GRASP",
    "RIGHT_RELEASE",
    "RIGHT_LIFT_CLEAR",
    "RIGHT_RETREAT",
    "LEFT_HOVER_TARGET",
    "LEFT_LOWER_TARGET",
    "LEFT_RELEASE",
    "LEFT_RETREAT",
    "SUCCESS",
]
PHASE_INIT = 0
PHASE_RIGHT_HOVER = 1
PHASE_RIGHT_DESCEND = 2
PHASE_RIGHT_GRASP = 3
PHASE_RIGHT_LIFT = 4
PHASE_RIGHT_HANDOVER = 5
PHASE_LEFT_APPROACH = 6
PHASE_LEFT_GRASP = 7
PHASE_RIGHT_RELEASE = 8
PHASE_RIGHT_LIFT_CLEAR = 9
PHASE_RIGHT_RETREAT = 10
PHASE_LEFT_HOVER_TARGET = 11
PHASE_LEFT_LOWER_TARGET = 12
PHASE_LEFT_RELEASE = 13
PHASE_LEFT_RETREAT = 14
PHASE_SUCCESS = 15


def main():
    cfg = DualArmILEnvCfg()
    cfg.observations = VisuomotorObsCfg()
    cfg.sim.device = args_cli.device

    print("[Scripted Demo] Initializing Isaac Lab Environment...")
    env: ManagerBasedRLEnv = gym.make("Isaac-Dual-Arm-IL-v0", cfg=cfg).unwrapped
    robot = env.scene["robot"]
    obj = env.scene["object"]
    target = env.scene["target"]

    # Body Indices
    right_hand_idx = robot.find_bodies("panda_hand_0")[0][0]
    left_hand_idx = robot.find_bodies("panda_hand$")[0][0]

    # For fixed-base articulation in PhysX, the base link is omitted in Jacobian matrices:
    is_fixed = getattr(robot, "is_fixed_base", True)
    jacobi_right_hand_idx = right_hand_idx - 1 if is_fixed else right_hand_idx
    jacobi_left_hand_idx = left_hand_idx - 1 if is_fixed else left_hand_idx

    # Joint Indices in Articulation
    left_arm_joint_ids, left_arm_names = robot.find_joints("panda_joint[1-7]$")
    right_arm_joint_ids, right_arm_names = robot.find_joints("panda_joint[1-7]_0")

    # Gripper finger joint indices (for real-time grip width and fake grasp verification)
    left_gripper_joint_ids, _ = robot.find_joints("panda_finger_joint[1-2]$")
    right_gripper_joint_ids, _ = robot.find_joints("panda_finger_joint[1-2]_0")

    # Safe standby postures (arm base and elbow rotated away from center workspace)
    left_standby_joints = robot.data.default_joint_pos[:, left_arm_joint_ids].clone()
    left_standby_joints[:, 0] = 0.60  # +34 deg, outward to +Y quadrant

    right_standby_joints = robot.data.default_joint_pos[:, right_arm_joint_ids].clone()
    right_standby_joints[:, 0] = -0.60  # -34 deg, outward to -Y quadrant

    right_standby_joints[:, 4] += 0.5
    left_standby_joints[:, 4] -= 0.5

    # Nominal joint postures for 7-DOF nullspace elbow flaring
    q_nominal_left = left_standby_joints.clone()
    q_nominal_right = right_standby_joints.clone()

    # Fixed world orientations for stable, jitter-free transport
    fixed_z_down = torch.tensor([[0.0, 0.0, -1.0]], device=args_cli.device)
    fixed_x_along = torch.tensor([[1.0, 0.0, 0.0]], device=args_cli.device)

    # Shift joint IDs for Jacobian lookup (0 for fixed base robot)
    num_base_dofs = getattr(robot, "num_base_dofs", 0)
    jacobi_right_joint_ids = [j + num_base_dofs for j in right_arm_joint_ids]
    jacobi_left_joint_ids = [j + num_base_dofs for j in left_arm_joint_ids]

    # Inspect the Action Manager's arm_action term
    arm_term = env.action_manager._terms["arm_action"]
    print(f"[Scripted Demo] Action Manager 'arm_action' controls {len(arm_term._joint_ids)} joints.")
    print(f"  Scale : {arm_term._scale}")

    # Joint limits for safety clamping
    soft_lower = robot.data.soft_joint_pos_limits[:, :, 0]
    soft_upper = robot.data.soft_joint_pos_limits[:, :, 1]

    # Check existing dataset
    existing_demos = 0
    if os.path.exists(args_cli.dataset_file):
        with h5py.File(args_cli.dataset_file, "r") as f:
            if "data" in f:
                existing_demos = len([k for k in f["data"].keys() if k.startswith("demo_")])

    print(f"[Scripted Demo] Existing demonstrations: {existing_demos}")
    collected_count = existing_demos
    target_count = existing_demos + args_cli.num_demos

    start_time = time.time()
    while simulation_app.is_running() and collected_count < target_count:
        obs, _ = env.reset()

        # Initialize commanded joint targets from the robot's safe standby positions
        commanded_left = left_standby_joints.clone()
        commanded_right = right_standby_joints.clone()

        left_gripper_cmd = 1.0   # Open (+1.0)
        right_gripper_cmd = 1.0  # Open (+1.0)

        phase = PHASE_INIT
        phase_timer = 0
        latched_align_sign = None
        latched_cube_z = None
        target_cube_z = None
        target_cube_z_for_wrist = None
        latched_left_grasp_offset = None

        ep_obs = []
        ep_images = []
        ep_actions = []
        ep_rewards = []
        object_poses = []
        robot_joint_poses = []
        init_robot_pos = robot.data.root_pos_w.cpu().numpy()[0].copy()
        init_robot_quat = robot.data.root_quat_w.cpu().numpy()[0].copy()
        init_target_pos = target.data.root_pos_w.cpu().numpy()[0].copy()
        init_target_quat = target.data.root_quat_w.cpu().numpy()[0].copy()

        print(f"\n>>> Generating Scripted Demo #{collected_count} (Target: {target_count}) <<<")

        for step in range(args_cli.max_steps_per_ep):
            if not simulation_app.is_running():
                break

            phase_timer += 1

            # Object and Target Poses
            obj_pos = obj.data.root_pos_w.clone()        # (1, 3)
            obj_quat = obj.data.root_quat_w.clone()      # (1, 4)
            target_pos = target.data.root_pos_w.clone()  # (1, 3)

            # Fail-fast: If baton drops during mid-air phases (Handover to Hover Target), abort episode
            if 5 <= phase <= 11 and obj_pos[0, 2].item() < 0.05:
                print(f"  [X] Dropped baton in mid-air at phase {PHASE_NAMES[phase]}! Discarding episode...")
                break

            # Franka Wrist and TCP Poses
            wrist_pos_r = robot.data.body_pos_w[:, right_hand_idx]
            wrist_quat_r = robot.data.body_quat_w[:, right_hand_idx]
            tcp_pos_r, z_dir_r = compute_tcp(wrist_pos_r, wrist_quat_r)

            wrist_pos_l = robot.data.body_pos_w[:, left_hand_idx]
            wrist_quat_l = robot.data.body_quat_w[:, left_hand_idx]
            tcp_pos_l, z_dir_l = compute_tcp(wrist_pos_l, wrist_quat_l)

            # Real-time gripper widths (for fake grasp detection)
            right_gripper_width = torch.sum(robot.data.joint_pos[:, right_gripper_joint_ids], dim=-1).item()
            left_gripper_width = torch.sum(robot.data.joint_pos[:, left_gripper_joint_ids], dim=-1).item()

            # Baton Geometry: Long axis vector (Cuboid Z-axis in world frame)
            ow, ox, oy, oz = obj_quat[:, 0], obj_quat[:, 1], obj_quat[:, 2], obj_quat[:, 3]
            cube_z = torch.stack([
                2.0 * (ox * oz + ow * oy),
                2.0 * (oy * oz - ow * ox),
                1.0 - 2.0 * (ox * ox + oy * oy),
            ], dim=-1)

            # Latch grasp alignment sign at episode start to prevent sign-flipping glitch
            if latched_align_sign is None:
                latched_align_sign = torch.sign(cube_z[:, 1:2] + 1e-6).clone()
                latched_cube_z = cube_z.clone()
                target_cube_z = torch.tensor([[0.0, latched_align_sign.item(), 0.0]], device=args_cli.device)

            # Optimal Grasp Targets:
            # Distance 0.055m (5.5cm from center) reduces cantilever droop torque by 35%
            # while leaving 11cm of clearance between right and left grippers!
            pick_target = obj_pos - latched_align_sign * 0.055 * latched_cube_z
            if latched_align_sign is not None:
                place_target = obj_pos + latched_align_sign * 0.055 * target_cube_z
            else:
                place_target = obj_pos + latched_align_sign * 0.055 * latched_cube_z

            # Handover zone (between both arms)
            handover_pos = env.scene.env_origins + torch.tensor([[0.30, 0.0, 0.20]], device=args_cli.device)
            # Left arm pre-handover hover position: 20cm in +Y, 12cm above handover zone
            left_wait_pos = handover_pos.clone()
            left_wait_pos[:, 1] += 0.20
            left_wait_pos[:, 2] += 0.12

            delta_r_pos = torch.zeros((1, 3), device=args_cli.device)
            delta_r_rot = torch.zeros((1, 3), device=args_cli.device)
            delta_l_pos = torch.zeros((1, 3), device=args_cli.device)
            delta_l_rot = torch.zeros((1, 3), device=args_cli.device)
            
            k_null_r = 0.05
            k_null_l = 0.05
            damping_r = 0.05
            damping_l = 0.05

            # --- State Machine Logic ---
            if phase == PHASE_INIT:
                right_gripper_cmd = 1.0
                left_gripper_cmd = 1.0
                # Smoothly swing left arm to safe standby posture (+Y quadrant)
                commanded_left = commanded_left + torch.clamp(left_standby_joints - commanded_left, min=-0.04, max=0.04)
                if phase_timer > 15:
                    phase = PHASE_RIGHT_HOVER
                    phase_timer = 0

            elif phase == PHASE_RIGHT_HOVER:
                # Left arm holds safe standby posture away from center
                commanded_left = commanded_left + torch.clamp(left_standby_joints - commanded_left, min=-0.04, max=0.04)
                hover_tgt = pick_target.clone()
                hover_tgt[:, 2] = pick_target[:, 2] + 0.08
                err = hover_tgt - tcp_pos_r
                dist = torch.norm(err)
                dist_xy = torch.norm(hover_tgt[:, :2] - tcp_pos_r[:, :2])
                dist_z = torch.abs(hover_tgt[:, 2] - tcp_pos_r[:, 2])
                step_size = min(0.012, dist.item())
                delta_r_pos = (err / (dist + 1e-6)) * step_size
                delta_r_rot = compute_desired_grasp_rot(wrist_quat_r, latched_align_sign * latched_cube_z, directed=True)
                if (dist_xy < 0.025 and dist_z < 0.03) or phase_timer > 90:
                    phase = PHASE_RIGHT_DESCEND
                    phase_timer = 0

            elif phase == PHASE_RIGHT_DESCEND:
                # Left arm holds standby
                commanded_left = commanded_left + torch.clamp(left_standby_joints - commanded_left, min=-0.04, max=0.04)
                # Descend to center of cuboid (TCP Z=0.025, to avoid fingertips crashing into table)
                descend_tgt = pick_target.clone()
                descend_tgt[:, 2] = torch.clamp(pick_target[:, 2], min=0.025)
                err = descend_tgt - tcp_pos_r
                dist = torch.norm(err)
                dist_xy = torch.norm(descend_tgt[:, :2] - tcp_pos_r[:, :2])
                dist_z = torch.abs(descend_tgt[:, 2] - tcp_pos_r[:, 2])
                step_size = min(0.012, dist.item())
                delta_r_pos = (err / (dist + 1e-6)) * step_size
                delta_r_rot = compute_desired_grasp_rot(wrist_quat_r, latched_align_sign * latched_cube_z, gain=0.2, max_rot_step=0.020, directed=True)
                reached = (dist_xy < 0.010 and dist_z < 0.005)
                if reached or phase_timer > 160:
                    if reached:
                        print(f"  [Right Grasp Reached!] Step {step:03d} | R_TCP_Z: {tcp_pos_r[0,2]:.3f} | Target_Z: {descend_tgt[0,2]:.3f} (diff: {dist_z.item()*1000:.1f}mm)")
                    else:
                        print(f"  [Right Grasp Timeout!] Step {step:03d} | R_TCP_Z: {tcp_pos_r[0,2]:.3f} | Target_Z: {descend_tgt[0,2]:.3f} (diff: {dist_z.item()*1000:.1f}mm)")
                    phase = PHASE_RIGHT_GRASP
                    phase_timer = 0

            elif phase == PHASE_RIGHT_GRASP:
                # Left arm holds standby
                commanded_left = commanded_left + torch.clamp(left_standby_joints - commanded_left, min=-0.04, max=0.04)
                # Close right gripper tightly around baton
                right_gripper_cmd = -1.0
                if phase_timer > 25:
                    if right_gripper_width > 0.025:
                        print(f"  [Right Grasp Verified!] Step {step:03d} | Width: {right_gripper_width*1000:.1f}mm")
                        
                        w, x, y, z = wrist_quat_r[:, 0], wrist_quat_r[:, 1], wrist_quat_r[:, 2], wrist_quat_r[:, 3]
                        x_curr = torch.stack([1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z), 2.0 * (x * z - w * y)], dim=-1)
                        latched_wrist_sign = latched_align_sign.clone()
                        target_cube_z_for_wrist = latched_wrist_sign * target_cube_z
                        
                        phase = PHASE_RIGHT_LIFT
                        phase_timer = 0
                    else:
                        print(f"  [Right Grasp Missed!] Step {step:03d} | Width: {right_gripper_width*1000:.1f}mm < 25mm. Retrying descend...")
                        right_gripper_cmd = 1.0
                        if phase_timer > 40:
                            phase = PHASE_RIGHT_DESCEND
                            phase_timer = 0

            elif phase == PHASE_RIGHT_LIFT:

                # Left arm pre-moves to handover wait position to save time
                err_l = left_wait_pos - tcp_pos_l
                dist_l = torch.norm(err_l)
                step_size_l = min(0.012, dist_l.item())
                delta_l_pos = (err_l / (dist_l + 1e-6)) * step_size_l
                left_orient_target = -torch.sign(target_cube_z[:, 1:2] + 1e-6) * target_cube_z
                delta_l_rot = compute_desired_grasp_rot(wrist_quat_l, left_orient_target, directed=True)
                # Lift to stable handover height Z = 0.18m with locked horizontal orientation
                lift_tgt = pick_target.clone()
                lift_tgt[:, 2] = 0.18
                err = lift_tgt - tcp_pos_r
                dist = torch.norm(err)
                step_size = min(0.012, dist.item())
                delta_r_pos = (err / (dist + 1e-6)) * step_size
                delta_r_rot = compute_desired_grasp_rot(wrist_quat_r, latched_align_sign * latched_cube_z, gain=0.2, max_rot_step=0.020, directed=True)
                if dist < 0.025 or phase_timer > 70:
                    phase = PHASE_RIGHT_HANDOVER
                    phase_timer = 0

            elif phase == PHASE_RIGHT_HANDOVER:
                # Right arm transports baton to handover zone and slowly twists 90 degrees
                # Completely eliminate nullspace and massively increase damping to safely pass through wrist singularity
                k_null_r = 0.0
                damping_r = 0.25
                err = handover_pos - tcp_pos_r
                dist = torch.norm(err)
                step_size = min(0.012, dist.item())
                delta_r_pos = (err / (dist + 1e-6)) * step_size
                # Extremely slow and smooth rotation to prevent inertial shaking of the 25cm baton
                delta_r_rot = compute_desired_grasp_rot(wrist_quat_r, target_cube_z_for_wrist, gain=0.1, max_rot_step=0.020, directed=True)

                # Left arm pre-moves to handover wait position to save time
                err_l = left_wait_pos - tcp_pos_l
                dist_l = torch.norm(err_l)
                step_size_l = min(0.012, dist_l.item())
                delta_l_pos = (err_l / (dist_l + 1e-6)) * step_size_l
                left_orient_target = -torch.sign(target_cube_z[:, 1:2] + 1e-6) * target_cube_z
                delta_l_rot = compute_desired_grasp_rot(wrist_quat_l, left_orient_target, directed=True)

                # Wait until BOTH position and orientation (90-degree twist) are fully reached
                # Orientation is aligned when the right wrist's local X-axis is strictly parallel to target_cube_z_for_wrist
                w, x, y, z = wrist_quat_r[:, 0], wrist_quat_r[:, 1], wrist_quat_r[:, 2], wrist_quat_r[:, 3]
                x_curr_r = torch.stack([1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z), 2.0 * (x * z - w * y)], dim=-1)
                rot_aligned = (1.0 - torch.sum(x_curr_r * target_cube_z_for_wrist, dim=-1)) < 0.005
                
                if (dist < 0.03 and rot_aligned.item()) or phase_timer > 400:
                    phase = PHASE_LEFT_APPROACH
                    phase_timer = 0

            elif phase == PHASE_LEFT_APPROACH:
                # Right arm holds baton rock-solid at handover zone
                k_null_r = 0.05
                right_gripper_cmd = -1.0
                delta_r_pos = torch.zeros((1, 3), device=args_cli.device)
                delta_r_rot = torch.zeros((1, 3), device=args_cli.device)

                # Left arm descends: first move horizontally to align above, then straight down
                left_grasp_tgt = place_target.clone()
                left_grasp_tgt[:, 2] = place_target[:, 2] - 0.006
                dist_xy = torch.norm(left_grasp_tgt[:, :2] - tcp_pos_l[:, :2])
                
                if dist_xy > 0.03:
                    curr_tgt = left_grasp_tgt.clone()
                    curr_tgt[:, 2] = left_wait_pos[:, 2]
                else:
                    curr_tgt = left_grasp_tgt.clone()
                    
                err = curr_tgt - tcp_pos_l
                dist = torch.norm(err)
                dist_z = torch.abs(left_grasp_tgt[:, 2] - tcp_pos_l[:, 2])
                step_size = min(0.012, dist.item())
                delta_l_pos = (err / (dist + 1e-6)) * step_size
                left_orient_target = -torch.sign(target_cube_z[:, 1:2] + 1e-6) * target_cube_z
                delta_l_rot = compute_desired_grasp_rot(wrist_quat_l, left_orient_target, directed=True)
                reached_l = (dist_xy < 0.025 and dist_z < 0.015)
                if reached_l or phase_timer > 160:
                    if reached_l:
                        print(f"  [Left Handover Reached!] Step {step:03d} | L_TCP_Z: {tcp_pos_l[0,2]:.3f} | Place_Z: {place_target[0,2]:.3f}")
                    else:
                        print(f"  [Left Handover Timeout!] Step {step:03d} | L_TCP_Z: {tcp_pos_l[0,2]:.3f} | Place_Z: {place_target[0,2]:.3f}")
                    phase = PHASE_LEFT_GRASP
                    phase_timer = 0

            elif phase == PHASE_LEFT_GRASP:
                k_null_r = 0.05
                # Right arm continues holding
                right_gripper_cmd = -1.0
                delta_r_pos = torch.zeros((1, 3), device=args_cli.device)
                delta_r_rot = torch.zeros((1, 3), device=args_cli.device)
                # Left arm closes gripper
                left_gripper_cmd = -1.0
                if phase_timer > 35:
                    if left_gripper_width > 0.025:
                        print(f"  [Left Grasp Verified!] Step {step:03d} | Width: {left_gripper_width*1000:.1f}mm")
                        latched_left_grasp_offset = tcp_pos_l - obj_pos
                        phase = PHASE_RIGHT_RELEASE
                        phase_timer = 0
                    else:
                        print(f"  [Left Grasp Missed!] Step {step:03d} | Width: {left_gripper_width*1000:.1f}mm < 25mm. Retrying...")
                        left_gripper_cmd = 1.0
                        if phase_timer > 50:
                            phase = PHASE_LEFT_APPROACH
                            phase_timer = 0

            elif phase == PHASE_RIGHT_RELEASE:
                k_null_r = 0.05
                # Left arm holds tightly
                left_gripper_cmd = -1.0
                delta_l_pos = torch.zeros((1, 3), device=args_cli.device)
                delta_l_rot = torch.zeros((1, 3), device=args_cli.device)
                # Right arm opens gripper
                right_gripper_cmd = 1.0
                delta_r_pos = torch.zeros((1, 3), device=args_cli.device)
                delta_r_rot = torch.zeros((1, 3), device=args_cli.device)
                if phase_timer > 15:
                    phase = PHASE_RIGHT_LIFT_CLEAR
                    phase_timer = 0

            elif phase == PHASE_RIGHT_LIFT_CLEAR:
                k_null_r = 0.05
                # Left arm firmly holds baton
                left_gripper_cmd = -1.0
                delta_l_pos = torch.zeros((1, 3), device=args_cli.device)
                delta_l_rot = torch.zeros((1, 3), device=args_cli.device)
                # Right arm moves -Y (away from left arm) AND UP to avoid collision
                right_gripper_cmd = 1.0
                lift_clear_tgt = handover_pos.clone()
                lift_clear_tgt[:, 1] -= 0.15  # Move -15cm in Y (away from left arm)
                lift_clear_tgt[:, 2] = 0.35   # Lift up to Z=0.35m
                err = lift_clear_tgt - tcp_pos_r
                dist = torch.norm(err)
                step_size = min(0.012, dist.item())
                delta_r_pos = (err / (dist + 1e-6)) * step_size
                delta_r_rot = compute_desired_grasp_rot(wrist_quat_r, target_cube_z_for_wrist, gain=0.2, max_rot_step=0.020, directed=True)
                if dist < 0.03 or phase_timer > 60:
                    phase = PHASE_RIGHT_RETREAT
                    phase_timer = 0

            elif phase == PHASE_RIGHT_RETREAT:
                # Right arm retreats backwards towards base to clear the table
                retreat_tgt = lift_clear_tgt.clone()
                retreat_tgt[:, 0] -= 0.15  # Move back 15cm
                err = retreat_tgt - tcp_pos_r
                step_size = min(0.012, torch.norm(err).item())
                delta_r_pos = (err / (torch.norm(err) + 1e-6)) * step_size
                delta_r_rot = compute_desired_grasp_rot(wrist_quat_r, target_cube_z)
                left_gripper_cmd = -1.0
                delta_l_pos = torch.zeros((1, 3), device=args_cli.device)
                delta_l_rot = torch.zeros((1, 3), device=args_cli.device)
                right_gripper_cmd = 1.0
                # Right arm smoothly moves to safe right standby posture (-Y quadrant)
                commanded_right = commanded_right + torch.clamp(right_standby_joints - commanded_right, min=-0.04, max=0.04)
                if phase_timer > 15:
                    phase = PHASE_LEFT_HOVER_TARGET
                    phase_timer = 0

            elif phase == PHASE_LEFT_HOVER_TARGET:
                # Right arm stays safely parked in right standby posture
                commanded_right = commanded_right + torch.clamp(right_standby_joints - commanded_right, min=-0.04, max=0.04)
                # Manual tuning to compensate for physical drop/slip offset (pull back from +X, +Y)
                manual_offset = torch.tensor([[-0.08, -0.08, 0.0]], device=args_cli.device)
                final_place_target = target_pos + latched_left_grasp_offset + manual_offset
                tgt_hover = final_place_target.clone()
                tgt_hover[:, 2] = target_pos[:, 2] + 0.12
                err = tgt_hover - tcp_pos_l
                dist = torch.norm(err)
                step_size = min(0.016, dist.item())
                delta_l_pos = (err / (dist + 1e-6)) * step_size
                # Locked fixed horizontal orientation to eliminate rotational jitter/slipping
                left_orient_target = -torch.sign(target_cube_z[:, 1:2] + 1e-6) * target_cube_z
                delta_l_rot = compute_desired_grasp_rot(wrist_quat_l, left_orient_target, gain=0.10, max_rot_step=0.010, directed=True)
                dist_xy = torch.norm(tgt_hover[:, :2] - tcp_pos_l[:, :2])
                dist_z = torch.abs(tgt_hover[:, 2] - tcp_pos_l[:, 2])
                if (dist_xy < 0.01 and dist_z < 0.015) or phase_timer > 70:
                    phase = PHASE_LEFT_LOWER_TARGET
                    phase_timer = 0

            elif phase == PHASE_LEFT_LOWER_TARGET:
                # Right arm stays safely parked in right standby posture
                commanded_right = commanded_right + torch.clamp(right_standby_joints - commanded_right, min=-0.04, max=0.04)
                # Gently lower baton onto target (Z=0.022m, soft landing 2mm above table surface)
                manual_offset = torch.tensor([[-0.08, -0.08, 0.0]], device=args_cli.device)
                final_place_target = target_pos + latched_left_grasp_offset + manual_offset
                tgt_lower = final_place_target.clone()
                tgt_lower[:, 2] = 0.022
                err = tgt_lower - tcp_pos_l
                dist = torch.norm(err)
                dist_z = torch.abs(tgt_lower[:, 2] - tcp_pos_l[:, 2])
                step_size = min(0.008, dist.item())
                delta_l_pos = (err / (dist + 1e-6)) * step_size
                left_orient_target = -torch.sign(target_cube_z[:, 1:2] + 1e-6) * target_cube_z
                delta_l_rot = compute_desired_grasp_rot(wrist_quat_l, left_orient_target, gain=0.10, max_rot_step=0.010, directed=True)
                if (dist < 0.015 or dist_z < 0.006) or phase_timer > 60:
                    phase = PHASE_LEFT_RELEASE
                    phase_timer = 0

            elif phase == PHASE_LEFT_RELEASE:
                # Right arm stays parked in right standby posture
                commanded_right = commanded_right + torch.clamp(right_standby_joints - commanded_right, min=-0.04, max=0.04)
                # Open left gripper to release baton on target
                left_gripper_cmd = 1.0
                if phase_timer > 10:
                    phase = PHASE_LEFT_RETREAT
                    phase_timer = 0

            elif phase == PHASE_LEFT_RETREAT:
                # Right arm stays parked in right standby posture
                commanded_right = commanded_right + torch.clamp(right_standby_joints - commanded_right, min=-0.04, max=0.04)
                left_gripper_cmd = 1.0
                # Move left arm up and retreat
                tgt_up = target_pos.clone()
                tgt_up[:, 2] += 0.20
                err = tgt_up - tcp_pos_l
                dist = torch.norm(err)
                step_size = min(0.012, dist.item())
                delta_l_pos = (err / (dist + 1e-6)) * step_size
                if dist < 0.03 or phase_timer > 40:
                    phase = PHASE_SUCCESS
                    phase_timer = 0

            # Real-time 3D state and tracking error monitoring
            if step % 25 == 0:
                dist_pick = torch.norm(pick_target - tcp_pos_r).item() * 1000
                dist_place = torch.norm(place_target - tcp_pos_l).item() * 1000
                print(
                    f"  Step {step:03d} | {PHASE_NAMES[phase]:<18} | "
                    f"Pick:[{pick_target[0,0]:.2f},{pick_target[0,1]:.2f},{pick_target[0,2]:.2f}] | "
                    f"R_TCP:[{tcp_pos_r[0,0]:.2f},{tcp_pos_r[0,1]:.2f},{tcp_pos_r[0,2]:.2f}]({dist_pick:.1f}mm) | "
                    f"Place:[{place_target[0,0]:.2f},{place_target[0,1]:.2f},{place_target[0,2]:.2f}] | "
                    f"L_TCP:[{tcp_pos_l[0,0]:.2f},{tcp_pos_l[0,1]:.2f},{tcp_pos_l[0,2]:.2f}]({dist_place:.1f}mm)"
                )

            # --- 6D Inverse Kinematics with 7-DOF Nullspace Elbow Flare Projection ---
            jacobians = robot.root_physx_view.get_jacobians()

            delta_r_pose = torch.cat([delta_r_pos, delta_r_rot], dim=-1)
            if torch.norm(delta_r_pose) > 1e-5:
                j_r_wrist = jacobians[:, jacobi_right_hand_idx, :6, :][:, :, jacobi_right_joint_ids]
                j_r_tcp = get_tcp_jacobian(j_r_wrist, wrist_pos_r, tcp_pos_r)
                dq_r = solve_pose_dls_ik(
                    j_r_tcp,
                    delta_r_pose,
                    damping=damping_r,
                    q_current=commanded_right,
                    q_nominal=q_nominal_right,
                    k_null=k_null_r,
                )
                dq_r = torch.clamp(dq_r, min=-0.08, max=0.08)
                commanded_right += dq_r
                # Clamp within safe joint limits
                commanded_right = torch.clamp(
                    commanded_right,
                    min=soft_lower[:, right_arm_joint_ids] + 0.02,
                    max=soft_upper[:, right_arm_joint_ids] - 0.02,
                )

            delta_l_pose = torch.cat([delta_l_pos, delta_l_rot], dim=-1)
            if torch.norm(delta_l_pose) > 1e-5:
                j_l_wrist = jacobians[:, jacobi_left_hand_idx, :6, :][:, :, jacobi_left_joint_ids]
                j_l_tcp = get_tcp_jacobian(j_l_wrist, wrist_pos_l, tcp_pos_l)
                
                # Dynamic task-space relaxation: Free the wrist orientation during flight to maximize reach
                if phase == PHASE_LEFT_HOVER_TARGET:
                    j_l_tcp = j_l_tcp.clone()
                    j_l_tcp[:, 3:6, :] = 0.0
                    
                dq_l = solve_pose_dls_ik(
                    j_l_tcp,
                    delta_l_pose,
                    damping=damping_l,
                    q_current=commanded_left,
                    q_nominal=q_nominal_left,
                    k_null=k_null_l,
                )
                dq_l = torch.clamp(dq_l, min=-0.08, max=0.08)
                commanded_left += dq_l
                commanded_left = torch.clamp(
                    commanded_left,
                    min=soft_lower[:, left_arm_joint_ids] + 0.02,
                    max=soft_upper[:, left_arm_joint_ids] - 0.02,
                )

            # --- Action Inversion & Mapping ---
            # 1. Update full articulation joint targets
            target_all_joints = robot.data.default_joint_pos.clone()
            target_all_joints[:, left_arm_joint_ids] = commanded_left
            target_all_joints[:, right_arm_joint_ids] = commanded_right

            # 2. Extract joints in the exact order arm_action expects
            arm_targets = target_all_joints[:, arm_term._joint_ids]

            # 3. Invert default offset and scale so applied == arm_targets
            raw_arm_action = (arm_targets - arm_term._offset) / arm_term._scale

            # 4. Grippers
            l_grip_t = torch.tensor([[left_gripper_cmd]], dtype=torch.float32, device=args_cli.device)
            r_grip_t = torch.tensor([[right_gripper_cmd]], dtype=torch.float32, device=args_cli.device)
            action = torch.cat([raw_arm_action, l_grip_t, r_grip_t], dim=-1)

            # Record step transition
            policy_obs = obs["policy"].squeeze(0).detach().cpu().numpy()
            rgb_image = obs["image"]["rgb"].squeeze(0).detach().cpu().numpy()
            if rgb_image.dtype != np.uint8:
                if rgb_image.max() <= 1.0:
                    rgb_image = (rgb_image * 255.0)
                rgb_image = np.clip(rgb_image, 0, 255).astype(np.uint8)

            action_np = action.squeeze(0).detach().cpu().numpy()
            ep_obs.append(policy_obs)
            ep_images.append(rgb_image)
            ep_actions.append(action_np)
            object_poses.append(obj.data.root_state_w.cpu().numpy()[0])
            robot_joint_poses.append(robot.data.joint_pos.cpu().numpy()[0])

            # Step simulation
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_rewards.append(float(reward.mean().item()))

            # Check Termination / Success
            if phase == PHASE_SUCCESS:
                dist_to_target = torch.norm(obj_pos[0, :2] - target_pos[0, :2]).item()
                if dist_to_target < 0.15 and obj_pos[0, 2].item() < 0.06:
                    save_episode_to_hdf5(
                        args_cli.dataset_file,
                        collected_count,
                        ep_obs,
                        ep_images,
                        ep_actions,
                        ep_rewards,
                        object_poses,
                        robot_joint_poses,
                        init_robot_pos,
                        init_robot_quat,
                        init_target_pos,
                        init_target_quat,
                    )
                    collected_count += 1
                    print(f"  [✓] Successfully collected Demo #{collected_count-1} in {step} steps! (dist={dist_to_target:.3f}m)")
                    break
                else:
                    print(f"  [X] Object not on target (dist={dist_to_target:.3f}m, z={obj_pos[0, 2]:.3f}m). Discarding...")
                    break

            if terminated.item() or truncated.item():
                print(f"  [X] Episode terminated early at step {step} (phase={PHASE_NAMES[phase]}). Discarding...")
                break

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f" Scripted Demo Generation Finished!")
    print(f" Total Demos in Dataset : {collected_count}")
    print(f" Time Elapsed           : {elapsed:.1f} seconds ({elapsed/max(1, collected_count-existing_demos):.1f} s/demo)")
    print(f" Saved Dataset          : {args_cli.dataset_file}")
    print("=" * 60)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()

```

## scripts/generate_scripted_demos_parallel.py
```py
# Copyright (c) 2026, Dual Arm Imitation Learning Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Scripted Automatic Demonstration Generator for Dual Arm Handover Task.

Generates high-quality, optimal demonstration trajectories automatically using
Waypoint State Machine and Differential Inverse Kinematics.

Features:
- 100% automated: No human teleoperation or manual effort required.
- Robust Closed-Loop IK: DLS position control with joint limit protection.
- Exact Action Inversion: Automatically accounts for Isaac Lab's default joint offsets and action scales.
- Direct export to HDF5: Ready for immediate training with BC, Diffusion Policy, or ACT.

Usage:
    # Fast Headless generation (recommended):
    python scripts/generate_scripted_demos.py --num_demos=500 --headless

    # GUI mode (visualize robot motion):
    python scripts/generate_scripted_demos.py --num_demos=10
"""

import argparse
import os
import sys
import time
import h5py
import numpy as np
import torch

# Add project root and parent to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT_ROOT = os.path.dirname(PROJECT_ROOT)
for path in [PROJECT_ROOT, PARENT_ROOT]:
    if path not in sys.path:
        sys.path.insert(0, path)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Generate scripted demonstrations for Dual Arm Handover.")
parser.add_argument("--num_demos", type=int, default=50, help="Number of successful demonstrations to generate.")
parser.add_argument(
    "--dataset_file",
    type=str,
    default=os.path.join(PROJECT_ROOT, "data", "demos.hdf5"),
    help="Path to output HDF5 dataset.",
)
parser.add_argument("--max_steps_per_ep", type=int, default=700, help="Max steps before episode timeout.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments to run.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# FORCE ENABLE CAMERAS for Visuomotor project!
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from isaaclab.envs import ManagerBasedRLEnv
from configs.env_cfg import DualArmILEnvCfg, VisuomotorObsCfg

# Explicitly register our IL environment with Gym to ensure Isaac Lab's parse_env_cfg uses our config
gym.register(
    id="Isaac-Dual-Arm-IL-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": DualArmILEnvCfg,
    },
)


def solve_pose_dls_ik(
    jacobian: torch.Tensor,
    delta_pose: torch.Tensor,
    damping: torch.Tensor,
    q_current: torch.Tensor | None = None,
    q_nominal: torch.Tensor | None = None,
    k_null: torch.Tensor | None = None,
) -> torch.Tensor:
    """Damped Least Squares (DLS) Inverse Kinematics with 7-DOF nullspace elbow flare projection.
    """
    j = jacobian  # (N, 6, 7)
    e = delta_pose.unsqueeze(-1)  # (N, 6, 1)
    jjt = torch.bmm(j, j.transpose(1, 2))  # (N, 6, 6)
    identity = torch.eye(6, device=j.device, dtype=j.dtype).unsqueeze(0)
    
    # damping is (N,)
    d_sq = (damping ** 2).view(-1, 1, 1)
    
    inv_term = torch.inverse(jjt + d_sq * identity)
    j_pinv = torch.bmm(j.transpose(1, 2), inv_term)  # (N, 7, 6)
    delta_q = torch.bmm(j_pinv, e).squeeze(-1)  # (N, 7)

    if q_current is not None and q_nominal is not None and k_null is not None:
        eye_n = torch.eye(j.shape[2], device=j.device, dtype=j.dtype).unsqueeze(0)
        null_proj = eye_n - torch.bmm(j_pinv, j)  # (N, 7, 7)
        q_err = (q_nominal - q_current).unsqueeze(-1)  # (N, 7, 1)
        k_n = k_null.view(-1, 1, 1)
        delta_q_null = torch.bmm(null_proj, k_n * q_err).squeeze(-1)  # (N, 7)
        delta_q = delta_q + delta_q_null

    return delta_q



def get_tcp_jacobian(jacobian: torch.Tensor, wrist_pos: torch.Tensor, tcp_pos: torch.Tensor) -> torch.Tensor:
    r = tcp_pos - wrist_pos  # (N, 3)
    rx, ry, rz = r[:, 0], r[:, 1], r[:, 2]
    zero = torch.zeros_like(rx)
    r_skew = torch.stack([
        torch.stack([zero, -rz, ry], dim=-1),
        torch.stack([rz, zero, -rx], dim=-1),
        torch.stack([-ry, rx, zero], dim=-1)
    ], dim=1)  # (N, 3, 3)
    j_v = jacobian[:, :3, :]   # (N, 3, 7)
    j_w = jacobian[:, 3:6, :]  # (N, 3, 7)
    j_v_tcp = j_v - torch.bmm(r_skew, j_w)
    return torch.cat([j_v_tcp, j_w], dim=1)  # (N, 6, 7)


def compute_desired_grasp_rot(
    wrist_quat: torch.Tensor,
    cube_z_dir: torch.Tensor,
    gain: float = 0.5,
    max_rot_step: float = 0.04,
    directed: bool = False,
) -> torch.Tensor:
    """Compute angular error vector to align Franka gripper perpendicular to baton.

    Ensures:
    1. Local Z axis (palm normal) points vertically downwards [0, 0, -1].
    2. Local X axis (palm lateral) is parallel to the baton longitudinal axis.
    3. Local Y axis (finger opening/closing) is perpendicular to the baton,
       pinching the 3cm thickness cleanly.

    Args:
        wrist_quat: Hand orientation quaternion (1, 4) in (w, x, y, z).
        cube_z_dir: Baton length direction unit vector (1, 3).
        gain: Proportional convergence gain.
        max_rot_step: Maximum angular velocity step in radians.
        directed: If True, do not use 180-degree symmetry. Align exactly to cube_z_dir.

    Returns:
        e_rot_step: (1, 3) angular delta vector [wx, wy, wz].
    """
    w = wrist_quat[:, 0]
    x = wrist_quat[:, 1]
    y = wrist_quat[:, 2]
    z = wrist_quat[:, 3]

    # Current rotation matrix columns in world coordinates
    x_curr = torch.stack([
        1.0 - 2.0 * (y * y + z * z),
        2.0 * (x * y + w * z),
        2.0 * (x * z - w * y),
    ], dim=-1)

    y_curr = torch.stack([
        2.0 * (x * y - w * z),
        1.0 - 2.0 * (x * x + z * z),
        2.0 * (y * z + w * x),
    ], dim=-1)

    z_curr = torch.stack([
        2.0 * (x * z + w * y),
        2.0 * (y * z - w * x),
        1.0 - 2.0 * (x * x + y * y),
    ], dim=-1)

    # Desired Z axis: straight down
    z_des = torch.tensor([[0.0, 0.0, -1.0]], device=wrist_quat.device, dtype=wrist_quat.dtype)

    # Baton horizontal projection
    baton_xy = cube_z_dir.clone()
    baton_xy[:, 2] = 0.0
    norm_xy = torch.norm(baton_xy, dim=-1, keepdim=True)
    if norm_xy.item() > 1e-4:
        x_cand = baton_xy / norm_xy
    else:
        x_cand = torch.tensor([[1.0, 0.0, 0.0]], device=wrist_quat.device, dtype=wrist_quat.dtype)

    if directed:
        x_des = x_cand
    else:
        # Align with whichever 180-degree symmetric direction is closer to current x_curr
        dot = torch.sum(x_curr * x_cand, dim=-1, keepdim=True)
        sign = torch.sign(dot + 1e-6)
        x_des = sign * x_cand

    # Desired Y axis = Z_des x X_des (finger pinching axis perpendicular to baton)
    y_des = torch.cross(z_des, x_des, dim=-1)

    # Orientation error via Lie algebra so(3) cross-product error
    e_rot = 0.5 * (
        torch.cross(x_curr, x_des, dim=-1) +
        torch.cross(y_curr, y_des, dim=-1) +
        torch.cross(z_curr, z_des, dim=-1)
    )

    # Apply gain and clamp
    e_rot_step = e_rot * gain
    err_norm = torch.norm(e_rot_step)
    if err_norm > max_rot_step:
        e_rot_step = e_rot_step * (max_rot_step / err_norm)

    return e_rot_step


def compute_tcp(wrist_pos: torch.Tensor, wrist_quat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Franka TCP position and Z-axis direction from wrist pose."""
    w, x, y, z = wrist_quat[:, 0], wrist_quat[:, 1], wrist_quat[:, 2], wrist_quat[:, 3]
    z_dir_x = 2.0 * (x * z + w * y)
    z_dir_y = 2.0 * (y * z - w * x)
    z_dir_z = 1.0 - 2.0 * (x * x + y * y)
    z_dir = torch.stack([z_dir_x, z_dir_y, z_dir_z], dim=-1)
    tcp_pos = wrist_pos + 0.1034 * z_dir
    return tcp_pos, z_dir


def save_episode_to_hdf5(hdf5_path: str, ep_idx: int, observations: list, images: list, actions: list, rewards: list, object_poses: list, robot_joint_poses: list, init_robot_pos: list, init_robot_quat: list, init_target_pos: list, init_target_quat: list):
    os.makedirs(os.path.dirname(os.path.abspath(hdf5_path)), exist_ok=True)
    mode = "a" if os.path.exists(hdf5_path) else "w"
    with h5py.File(hdf5_path, mode) as f:
        data_group = f.require_group("data")
        demo_group = data_group.create_group(f"demo_{ep_idx}")

        obs_array = np.array(observations, dtype=np.float32)
        img_array = np.array(images, dtype=np.uint8)
        act_array = np.array(actions, dtype=np.float32)
        rew_array = np.array(rewards, dtype=np.float32)

        demo_group.create_dataset("obs", data=obs_array, compression="gzip")
        # For images, gzip can be slow, but we'll use it to save disk space
        demo_group.create_dataset("images", data=img_array, compression="gzip")
        demo_group.create_dataset("actions", data=act_array, compression="gzip")
        demo_group.create_dataset("rewards", data=rew_array, compression="gzip")
        demo_group.create_dataset("object_poses", data=np.array(object_poses, dtype=np.float32), compression="gzip")
        demo_group.create_dataset("robot_joint_poses", data=np.array(robot_joint_poses, dtype=np.float32), compression="gzip")
        demo_group.create_dataset("init_robot_pos", data=np.array(init_robot_pos, dtype=np.float32))
        demo_group.create_dataset("init_robot_quat", data=np.array(init_robot_quat, dtype=np.float32))
        demo_group.create_dataset("init_target_pos", data=np.array(init_target_pos, dtype=np.float32))
        demo_group.create_dataset("init_target_quat", data=np.array(init_target_quat, dtype=np.float32))
        demo_group.attrs["num_samples"] = len(act_array)

        total_samples = f["data"].attrs.get("total", 0) + len(act_array)
        f["data"].attrs["total"] = total_samples

    print(f"[Dataset] Successfully saved demo_{ep_idx} ({len(act_array)} steps) to {hdf5_path}")


# State Machine Phase Constants
PHASE_NAMES = [
    "INIT",
    "RIGHT_HOVER",
    "RIGHT_DESCEND",
    "RIGHT_GRASP",
    "RIGHT_LIFT",
    "RIGHT_HANDOVER",
    "LEFT_APPROACH",
    "LEFT_GRASP",
    "RIGHT_RELEASE",
    "RIGHT_LIFT_CLEAR",
    "RIGHT_RETREAT",
    "LEFT_HOVER_TARGET",
    "LEFT_LOWER_TARGET",
    "LEFT_RELEASE",
    "LEFT_RETREAT",
    "SUCCESS",
]
PHASE_INIT = 0
PHASE_RIGHT_HOVER = 1
PHASE_RIGHT_DESCEND = 2
PHASE_RIGHT_GRASP = 3
PHASE_RIGHT_LIFT = 4
PHASE_RIGHT_HANDOVER = 5
PHASE_LEFT_APPROACH = 6
PHASE_LEFT_GRASP = 7
PHASE_RIGHT_RELEASE = 8
PHASE_RIGHT_LIFT_CLEAR = 9
PHASE_RIGHT_RETREAT = 10
PHASE_LEFT_HOVER_TARGET = 11
PHASE_LEFT_LOWER_TARGET = 12
PHASE_LEFT_RELEASE = 13
PHASE_LEFT_RETREAT = 14
PHASE_SUCCESS = 15
PHASE_DONE = 99


def main():
    cfg = DualArmILEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.observations = VisuomotorObsCfg()
    cfg.sim.device = args_cli.device

    print("[Scripted Demo] Initializing Isaac Lab Environment...")
    env: ManagerBasedRLEnv = gym.make("Isaac-Dual-Arm-IL-v0", cfg=cfg).unwrapped
    robot = env.scene["robot"]
    obj = env.scene["object"]
    target = env.scene["target"]

    # Body Indices
    right_hand_idx = robot.find_bodies("panda_hand_0")[0][0]
    left_hand_idx = robot.find_bodies("panda_hand$")[0][0]

    # For fixed-base articulation in PhysX, the base link is omitted in Jacobian matrices:
    is_fixed = getattr(robot, "is_fixed_base", True)
    jacobi_right_hand_idx = right_hand_idx - 1 if is_fixed else right_hand_idx
    jacobi_left_hand_idx = left_hand_idx - 1 if is_fixed else left_hand_idx

    # Joint Indices in Articulation
    left_arm_joint_ids, left_arm_names = robot.find_joints("panda_joint[1-7]$")
    right_arm_joint_ids, right_arm_names = robot.find_joints("panda_joint[1-7]_0")

    # Gripper finger joint indices (for real-time grip width and fake grasp verification)
    left_gripper_joint_ids, _ = robot.find_joints("panda_finger_joint[1-2]$")
    right_gripper_joint_ids, _ = robot.find_joints("panda_finger_joint[1-2]_0")

    # Safe standby postures (arm base and elbow rotated away from center workspace)
    left_standby_joints = robot.data.default_joint_pos[:, left_arm_joint_ids].clone()
    left_standby_joints[:, 0] = 0.60  # +34 deg, outward to +Y quadrant

    right_standby_joints = robot.data.default_joint_pos[:, right_arm_joint_ids].clone()
    right_standby_joints[:, 0] = -0.60  # -34 deg, outward to -Y quadrant

    right_standby_joints[:, 4] += 0.5
    left_standby_joints[:, 4] -= 0.5

    # Nominal joint postures for 7-DOF nullspace elbow flaring
    q_nominal_left = left_standby_joints.clone()
    q_nominal_right = right_standby_joints.clone()

    # Fixed world orientations for stable, jitter-free transport
    fixed_z_down = torch.tensor([[0.0, 0.0, -1.0]], device=args_cli.device)
    fixed_x_along = torch.tensor([[1.0, 0.0, 0.0]], device=args_cli.device)

    # Shift joint IDs for Jacobian lookup (0 for fixed base robot)
    num_base_dofs = getattr(robot, "num_base_dofs", 0)
    jacobi_right_joint_ids = [j + num_base_dofs for j in right_arm_joint_ids]
    jacobi_left_joint_ids = [j + num_base_dofs for j in left_arm_joint_ids]

    # Inspect the Action Manager's arm_action term
    arm_term = env.action_manager._terms["arm_action"]
    print(f"[Scripted Demo] Action Manager 'arm_action' controls {len(arm_term._joint_ids)} joints.")
    print(f"  Scale : {arm_term._scale}")

    # Joint limits for safety clamping
    soft_lower = robot.data.soft_joint_pos_limits[:, :, 0]
    soft_upper = robot.data.soft_joint_pos_limits[:, :, 1]

    # Check existing dataset
    existing_demos = 0
    if os.path.exists(args_cli.dataset_file):
        with h5py.File(args_cli.dataset_file, "r") as f:
            if "data" in f:
                existing_demos = len([k for k in f["data"].keys() if k.startswith("demo_")])

    print(f"[Scripted Demo] Existing demonstrations: {existing_demos}")
    collected_count = existing_demos
    target_count = existing_demos + args_cli.num_demos
    num_envs = args_cli.num_envs

    start_time = time.time()
    while simulation_app.is_running() and collected_count < target_count:
        obs, _ = env.reset()

        # Initialize commanded joint targets from the robot's safe standby positions
        commanded_left = left_standby_joints.clone()
        commanded_right = right_standby_joints.clone()

        left_gripper_cmd = torch.ones((num_envs, 1), device=args_cli.device)
        right_gripper_cmd = torch.ones((num_envs, 1), device=args_cli.device)

        phases = [PHASE_INIT] * num_envs
        phase_timers = [0] * num_envs
        latched_align_signs = [None] * num_envs
        latched_cube_zs = [None] * num_envs
        target_cube_zs = [None] * num_envs
        target_cube_zs_for_wrist = [None] * num_envs
        latched_left_grasp_offsets = [None] * num_envs

        ep_obs = [[] for _ in range(num_envs)]
        ep_images = [[] for _ in range(num_envs)]
        ep_actions = [[] for _ in range(num_envs)]
        ep_rewards = [[] for _ in range(num_envs)]
        object_poses = [[] for _ in range(num_envs)]
        robot_joint_poses = [[] for _ in range(num_envs)]
        init_robot_pos = robot.data.root_pos_w.cpu().numpy().copy()
        init_robot_quat = robot.data.root_quat_w.cpu().numpy().copy()
        init_target_pos = target.data.root_pos_w.cpu().numpy().copy()
        init_target_quat = target.data.root_quat_w.cpu().numpy().copy()

        print(f"\n>>> Generating Scripted Demo #{collected_count} (Target: {target_count}) <<<")

        for step in range(args_cli.max_steps_per_ep):
            if not simulation_app.is_running():
                break

            # Object and Target Poses (all envs)
            obj_pos_all = obj.data.root_pos_w.clone()
            obj_quat_all = obj.data.root_quat_w.clone()
            target_pos_all = target.data.root_pos_w.clone()

            # Franka Wrist and TCP Poses (all envs)
            wrist_pos_r_all = robot.data.body_pos_w[:, right_hand_idx]
            wrist_quat_r_all = robot.data.body_quat_w[:, right_hand_idx]
            tcp_pos_r_all, z_dir_r_all = compute_tcp(wrist_pos_r_all, wrist_quat_r_all)

            wrist_pos_l_all = robot.data.body_pos_w[:, left_hand_idx]
            wrist_quat_l_all = robot.data.body_quat_w[:, left_hand_idx]
            tcp_pos_l_all, z_dir_l_all = compute_tcp(wrist_pos_l_all, wrist_quat_l_all)
            
            delta_r_pos = torch.zeros((num_envs, 3), device=args_cli.device)
            delta_r_rot = torch.zeros((num_envs, 3), device=args_cli.device)
            delta_l_pos = torch.zeros((num_envs, 3), device=args_cli.device)
            delta_l_rot = torch.zeros((num_envs, 3), device=args_cli.device)
            
            # Variables for damping and k_null (could differ per env if we needed, but script uses constants, we can just arrays)
            k_null_r = torch.full((num_envs,), 0.05, device=args_cli.device)
            k_null_l = torch.full((num_envs,), 0.05, device=args_cli.device)
            damping_r = torch.full((num_envs,), 0.05, device=args_cli.device)
            damping_l = torch.full((num_envs,), 0.05, device=args_cli.device)

            for i in range(num_envs):
                if phases[i] == PHASE_DONE:
                    continue

                phase_timers[i] += 1
                
                obj_pos = obj_pos_all[i:i+1]
                obj_quat = obj_quat_all[i:i+1]
                target_pos = target_pos_all[i:i+1]
                
                wrist_pos_r = wrist_pos_r_all[i:i+1]
                wrist_quat_r = wrist_quat_r_all[i:i+1]
                tcp_pos_r = tcp_pos_r_all[i:i+1]
                
                wrist_pos_l = wrist_pos_l_all[i:i+1]
                wrist_quat_l = wrist_quat_l_all[i:i+1]
                tcp_pos_l = tcp_pos_l_all[i:i+1]
                
                right_gripper_width = torch.sum(robot.data.joint_pos[i, right_gripper_joint_ids], dim=-1).item()
                left_gripper_width = torch.sum(robot.data.joint_pos[i, left_gripper_joint_ids], dim=-1).item()

                delta_r_pos_i = torch.zeros((1, 3), device=args_cli.device)
                delta_r_rot_i = torch.zeros((1, 3), device=args_cli.device)
                delta_l_pos_i = torch.zeros((1, 3), device=args_cli.device)
                delta_l_rot_i = torch.zeros((1, 3), device=args_cli.device)

                phase_timers[i] += 1

                # Object and Target Poses

                # Fail-fast: If baton drops during mid-air phases[i]s (Handover to Hover Target), abort episode
                if 5 <= phases[i] <= 11 and obj_pos[0, 2].item() < 0.05:
                    print(f"  [X] Env {i} Dropped baton in mid-air at phase {PHASE_NAMES[phases[i]]}! Discarding episode...")
                    phases[i] = PHASE_DONE
                    continue

                # Franka Wrist and TCP Poses


                # Real-time gripper widths (for fake grasp detection)

                # Baton Geometry: Long axis vector (Cuboid Z-axis in world frame)
                ow, ox, oy, oz = obj_quat[:, 0], obj_quat[:, 1], obj_quat[:, 2], obj_quat[:, 3]
                cube_z = torch.stack([
                    2.0 * (ox * oz + ow * oy),
                    2.0 * (oy * oz - ow * ox),
                    1.0 - 2.0 * (ox * ox + oy * oy),
                ], dim=-1)

                # Latch grasp alignment sign at episode start to prevent sign-flipping glitch
                if latched_align_signs[i] is None:
                    latched_align_signs[i] = torch.sign(cube_z[:, 1:2] + 1e-6).clone()
                    latched_cube_zs[i] = cube_z.clone()
                    target_cube_zs[i] = torch.tensor([[0.0, latched_align_signs[i].item(), 0.0]], device=args_cli.device)

                # Optimal Grasp Targets:
                # Distance 0.055m (5.5cm from center) reduces cantilever droop torque by 35%
                # while leaving 11cm of clearance between right and left grippers!
                pick_target = obj_pos - latched_align_signs[i] * 0.055 * latched_cube_zs[i]
                if latched_align_signs[i] is not None:
                    place_target = obj_pos + latched_align_signs[i] * 0.055 * target_cube_zs[i]
                else:
                    place_target = obj_pos + latched_align_signs[i] * 0.055 * latched_cube_zs[i]

                # Handover zone (between both arms)
                handover_pos = env.scene.env_origins[i:i+1] + torch.tensor([[0.30, 0.0, 0.20]], device=args_cli.device)
                # Left arm pre-handover hover position: 20cm in +Y, 12cm above handover zone
                left_wait_pos = handover_pos.clone()
                left_wait_pos[:, 1] += 0.20
                left_wait_pos[:, 2] += 0.12


                k_null_r[i] = 0.05
                k_null_l[i] = 0.05
                damping_r[i] = 0.05
                damping_l[i] = 0.05

                # --- State Machine Logic ---
                if phases[i] == PHASE_INIT:
                    right_gripper_cmd[i, 0] = 1.0
                    left_gripper_cmd[i, 0] = 1.0
                    # Smoothly swing left arm to safe standby posture (+Y quadrant)
                    commanded_left[i:i+1] = commanded_left[i:i+1] + torch.clamp(left_standby_joints[i:i+1] - commanded_left[i:i+1], min=-0.04, max=0.04)
                    if phase_timers[i] > 15:
                        phases[i] = PHASE_RIGHT_HOVER
                        phase_timers[i] = 0

                elif phases[i] == PHASE_RIGHT_HOVER:
                    # Left arm holds safe standby posture away from center
                    commanded_left[i:i+1] = commanded_left[i:i+1] + torch.clamp(left_standby_joints[i:i+1] - commanded_left[i:i+1], min=-0.04, max=0.04)
                    hover_tgt = pick_target.clone()
                    hover_tgt[:, 2] = pick_target[:, 2] + 0.08
                    err = hover_tgt - tcp_pos_r
                    dist = torch.norm(err)
                    dist_xy = torch.norm(hover_tgt[:, :2] - tcp_pos_r[:, :2])
                    dist_z = torch.abs(hover_tgt[:, 2] - tcp_pos_r[:, 2])
                    step_size = min(0.012, dist.item())
                    delta_r_pos_i = (err / (dist + 1e-6)) * step_size
                    delta_r_rot_i = compute_desired_grasp_rot(wrist_quat_r, latched_align_signs[i] * latched_cube_zs[i], directed=True)
                    if (dist_xy < 0.025 and dist_z < 0.03) or phase_timers[i] > 90:
                        phases[i] = PHASE_RIGHT_DESCEND
                        phase_timers[i] = 0

                elif phases[i] == PHASE_RIGHT_DESCEND:
                    # Left arm holds standby
                    commanded_left[i:i+1] = commanded_left[i:i+1] + torch.clamp(left_standby_joints[i:i+1] - commanded_left[i:i+1], min=-0.04, max=0.04)
                    # Descend to center of cuboid (TCP Z=0.025, to avoid fingertips crashing into table)
                    descend_tgt = pick_target.clone()
                    descend_tgt[:, 2] = torch.clamp(pick_target[:, 2], min=0.025)
                    err = descend_tgt - tcp_pos_r
                    dist = torch.norm(err)
                    dist_xy = torch.norm(descend_tgt[:, :2] - tcp_pos_r[:, :2])
                    dist_z = torch.abs(descend_tgt[:, 2] - tcp_pos_r[:, 2])
                    step_size = min(0.012, dist.item())
                    delta_r_pos_i = (err / (dist + 1e-6)) * step_size
                    delta_r_rot_i = compute_desired_grasp_rot(wrist_quat_r, latched_align_signs[i] * latched_cube_zs[i], gain=0.2, max_rot_step=0.020, directed=True)
                    reached = (dist_xy < 0.010 and dist_z < 0.005)
                    if reached or phase_timers[i] > 160:
                        if reached:
                            print(f"  [Right Grasp Reached!] Step {step:03d} | R_TCP_Z: {tcp_pos_r[0,2]:.3f} | Target_Z: {descend_tgt[0,2]:.3f} (diff: {dist_z.item()*1000:.1f}mm)")
                        else:
                            print(f"  [Right Grasp Timeout!] Step {step:03d} | R_TCP_Z: {tcp_pos_r[0,2]:.3f} | Target_Z: {descend_tgt[0,2]:.3f} (diff: {dist_z.item()*1000:.1f}mm)")
                        phases[i] = PHASE_RIGHT_GRASP
                        phase_timers[i] = 0

                elif phases[i] == PHASE_RIGHT_GRASP:
                    # Left arm holds standby
                    commanded_left[i:i+1] = commanded_left[i:i+1] + torch.clamp(left_standby_joints[i:i+1] - commanded_left[i:i+1], min=-0.04, max=0.04)
                    # Close right gripper tightly around baton
                    right_gripper_cmd[i, 0] = -1.0
                    if phase_timers[i] > 25:
                        if right_gripper_width > 0.025:
                            print(f"  [Right Grasp Verified!] Step {step:03d} | Width: {right_gripper_width*1000:.1f}mm")

                            w, x, y, z = wrist_quat_r[:, 0], wrist_quat_r[:, 1], wrist_quat_r[:, 2], wrist_quat_r[:, 3]
                            x_curr = torch.stack([1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z), 2.0 * (x * z - w * y)], dim=-1)
                            latched_wrist_sign = latched_align_signs[i].clone()
                            target_cube_zs_for_wrist[i] = latched_wrist_sign * target_cube_zs[i]

                            phases[i] = PHASE_RIGHT_LIFT
                            phase_timers[i] = 0
                        else:
                            print(f"  [Right Grasp Missed!] Step {step:03d} | Width: {right_gripper_width*1000:.1f}mm < 25mm. Retrying descend...")
                            right_gripper_cmd[i, 0] = 1.0
                            if phase_timers[i] > 40:
                                phases[i] = PHASE_RIGHT_DESCEND
                                phase_timers[i] = 0

                elif phases[i] == PHASE_RIGHT_LIFT:

                    # Left arm pre-moves to handover wait position to save time
                    err_l = left_wait_pos - tcp_pos_l
                    dist_l = torch.norm(err_l)
                    step_size_l = min(0.012, dist_l.item())
                    delta_l_pos_i = (err_l / (dist_l + 1e-6)) * step_size_l
                    left_orient_target = -torch.sign(target_cube_zs[i][:, 1:2] + 1e-6) * target_cube_zs[i]
                    delta_l_rot_i = compute_desired_grasp_rot(wrist_quat_l, left_orient_target, directed=True)
                    # Lift to stable handover height Z = 0.18m with locked horizontal orientation
                    lift_tgt = pick_target.clone()
                    lift_tgt[:, 2] = 0.18
                    err = lift_tgt - tcp_pos_r
                    dist = torch.norm(err)
                    step_size = min(0.012, dist.item())
                    delta_r_pos_i = (err / (dist + 1e-6)) * step_size
                    delta_r_rot_i = compute_desired_grasp_rot(wrist_quat_r, latched_align_signs[i] * latched_cube_zs[i], gain=0.2, max_rot_step=0.020, directed=True)
                    if dist < 0.025 or phase_timers[i] > 70:
                        phases[i] = PHASE_RIGHT_HANDOVER
                        phase_timers[i] = 0

                elif phases[i] == PHASE_RIGHT_HANDOVER:
                    # Right arm transports baton to handover zone and slowly twists 90 degrees
                    # Completely eliminate nullspace and massively increase damping to safely pass through wrist singularity
                    k_null_r[i] = 0.0
                    damping_r[i] = 0.25
                    err = handover_pos - tcp_pos_r
                    dist = torch.norm(err)
                    step_size = min(0.012, dist.item())
                    delta_r_pos_i = (err / (dist + 1e-6)) * step_size
                    # Extremely slow and smooth rotation to prevent inertial shaking of the 25cm baton
                    delta_r_rot_i = compute_desired_grasp_rot(wrist_quat_r, target_cube_zs_for_wrist[i], gain=0.1, max_rot_step=0.020, directed=True)

                    # Left arm pre-moves to handover wait position to save time
                    err_l = left_wait_pos - tcp_pos_l
                    dist_l = torch.norm(err_l)
                    step_size_l = min(0.012, dist_l.item())
                    delta_l_pos_i = (err_l / (dist_l + 1e-6)) * step_size_l
                    left_orient_target = -torch.sign(target_cube_zs[i][:, 1:2] + 1e-6) * target_cube_zs[i]
                    delta_l_rot_i = compute_desired_grasp_rot(wrist_quat_l, left_orient_target, directed=True)

                    # Wait until BOTH position and orientation (90-degree twist) are fully reached
                    # Orientation is aligned when the right wrist's local X-axis is strictly parallel to target_cube_zs_for_wrist[i]
                    w, x, y, z = wrist_quat_r[:, 0], wrist_quat_r[:, 1], wrist_quat_r[:, 2], wrist_quat_r[:, 3]
                    x_curr_r = torch.stack([1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z), 2.0 * (x * z - w * y)], dim=-1)
                    rot_aligned = (1.0 - torch.sum(x_curr_r * target_cube_zs_for_wrist[i], dim=-1)) < 0.005

                    if (dist < 0.03 and rot_aligned.item()) or phase_timers[i] > 400:
                        phases[i] = PHASE_LEFT_APPROACH
                        phase_timers[i] = 0

                elif phases[i] == PHASE_LEFT_APPROACH:
                    # Right arm holds baton rock-solid at handover zone
                    k_null_r[i] = 0.05
                    right_gripper_cmd[i, 0] = -1.0

                    # Left arm descends: first move horizontally to align above, then straight down
                    left_grasp_tgt = place_target.clone()
                    left_grasp_tgt[:, 2] = place_target[:, 2] - 0.006
                    dist_xy = torch.norm(left_grasp_tgt[:, :2] - tcp_pos_l[:, :2])

                    if dist_xy > 0.03:
                        curr_tgt = left_grasp_tgt.clone()
                        curr_tgt[:, 2] = left_wait_pos[:, 2]
                    else:
                        curr_tgt = left_grasp_tgt.clone()

                    err = curr_tgt - tcp_pos_l
                    dist = torch.norm(err)
                    dist_z = torch.abs(left_grasp_tgt[:, 2] - tcp_pos_l[:, 2])
                    step_size = min(0.012, dist.item())
                    delta_l_pos_i = (err / (dist + 1e-6)) * step_size
                    left_orient_target = -torch.sign(target_cube_zs[i][:, 1:2] + 1e-6) * target_cube_zs[i]
                    delta_l_rot_i = compute_desired_grasp_rot(wrist_quat_l, left_orient_target, directed=True)
                    reached_l = (dist_xy < 0.025 and dist_z < 0.015)
                    if reached_l or phase_timers[i] > 160:
                        if reached_l:
                            print(f"  [Left Handover Reached!] Step {step:03d} | L_TCP_Z: {tcp_pos_l[0,2]:.3f} | Place_Z: {place_target[0,2]:.3f}")
                        else:
                            print(f"  [Left Handover Timeout!] Step {step:03d} | L_TCP_Z: {tcp_pos_l[0,2]:.3f} | Place_Z: {place_target[0,2]:.3f}")
                        phases[i] = PHASE_LEFT_GRASP
                        phase_timers[i] = 0

                elif phases[i] == PHASE_LEFT_GRASP:
                    k_null_r[i] = 0.05
                    # Right arm continues holding
                    right_gripper_cmd[i, 0] = -1.0
                    # Left arm closes gripper
                    left_gripper_cmd[i, 0] = -1.0
                    if phase_timers[i] > 35:
                        if left_gripper_width > 0.025:
                            print(f"  [Left Grasp Verified!] Step {step:03d} | Width: {left_gripper_width*1000:.1f}mm")
                            latched_left_grasp_offsets[i] = tcp_pos_l - obj_pos
                            phases[i] = PHASE_RIGHT_RELEASE
                            phase_timers[i] = 0
                        else:
                            print(f"  [Left Grasp Missed!] Step {step:03d} | Width: {left_gripper_width*1000:.1f}mm < 25mm. Retrying...")
                            left_gripper_cmd[i, 0] = 1.0
                            if phase_timers[i] > 50:
                                phases[i] = PHASE_LEFT_APPROACH
                                phase_timers[i] = 0

                elif phases[i] == PHASE_RIGHT_RELEASE:
                    k_null_r[i] = 0.05
                    # Left arm holds tightly
                    left_gripper_cmd[i, 0] = -1.0
                    # Right arm opens gripper
                    right_gripper_cmd[i, 0] = 1.0
                    if phase_timers[i] > 15:
                        phases[i] = PHASE_RIGHT_LIFT_CLEAR
                        phase_timers[i] = 0

                elif phases[i] == PHASE_RIGHT_LIFT_CLEAR:
                    k_null_r[i] = 0.05
                    # Left arm firmly holds baton
                    left_gripper_cmd[i, 0] = -1.0
                    # Right arm moves -Y (away from left arm) AND UP to avoid collision
                    right_gripper_cmd[i, 0] = 1.0
                    lift_clear_tgt = handover_pos.clone()
                    lift_clear_tgt[:, 1] -= 0.15  # Move -15cm in Y (away from left arm)
                    lift_clear_tgt[:, 2] = 0.35   # Lift up to Z=0.35m
                    err = lift_clear_tgt - tcp_pos_r
                    dist = torch.norm(err)
                    step_size = min(0.012, dist.item())
                    delta_r_pos_i = (err / (dist + 1e-6)) * step_size
                    delta_r_rot_i = compute_desired_grasp_rot(wrist_quat_r, target_cube_zs_for_wrist[i], gain=0.2, max_rot_step=0.020, directed=True)
                    if dist < 0.03 or phase_timers[i] > 60:
                        phases[i] = PHASE_RIGHT_RETREAT
                        phase_timers[i] = 0

                elif phases[i] == PHASE_RIGHT_RETREAT:
                    # Right arm retreats backwards towards base to clear the table
                    retreat_tgt = lift_clear_tgt.clone()
                    retreat_tgt[:, 0] -= 0.15  # Move back 15cm
                    err = retreat_tgt - tcp_pos_r
                    step_size = min(0.012, torch.norm(err).item())
                    delta_r_pos_i = (err / (torch.norm(err) + 1e-6)) * step_size
                    delta_r_rot_i = compute_desired_grasp_rot(wrist_quat_r, target_cube_zs[i])
                    left_gripper_cmd[i, 0] = -1.0
                    right_gripper_cmd[i, 0] = 1.0
                    # Right arm smoothly moves to safe right standby posture (-Y quadrant)
                    commanded_right[i:i+1] = commanded_right[i:i+1] + torch.clamp(right_standby_joints[i:i+1] - commanded_right[i:i+1], min=-0.04, max=0.04)
                    if phase_timers[i] > 15:
                        phases[i] = PHASE_LEFT_HOVER_TARGET
                        phase_timers[i] = 0

                elif phases[i] == PHASE_LEFT_HOVER_TARGET:
                    # Right arm stays safely parked in right standby posture
                    commanded_right[i:i+1] = commanded_right[i:i+1] + torch.clamp(right_standby_joints[i:i+1] - commanded_right[i:i+1], min=-0.04, max=0.04)
                    # Manual tuning to compensate for physical drop/slip offset (pull back from +X, +Y)
                    manual_offset = torch.tensor([[-0.08, -0.08, 0.0]], device=args_cli.device)
                    final_place_target = target_pos + latched_left_grasp_offsets[i] + manual_offset
                    tgt_hover = final_place_target.clone()
                    tgt_hover[:, 2] = target_pos[:, 2] + 0.12
                    err = tgt_hover - tcp_pos_l
                    dist = torch.norm(err)
                    step_size = min(0.016, dist.item())
                    delta_l_pos_i = (err / (dist + 1e-6)) * step_size
                    # Locked fixed horizontal orientation to eliminate rotational jitter/slipping
                    left_orient_target = -torch.sign(target_cube_zs[i][:, 1:2] + 1e-6) * target_cube_zs[i]
                    delta_l_rot_i = compute_desired_grasp_rot(wrist_quat_l, left_orient_target, gain=0.10, max_rot_step=0.010, directed=True)
                    dist_xy = torch.norm(tgt_hover[:, :2] - tcp_pos_l[:, :2])
                    dist_z = torch.abs(tgt_hover[:, 2] - tcp_pos_l[:, 2])
                    if (dist_xy < 0.01 and dist_z < 0.015) or phase_timers[i] > 70:
                        phases[i] = PHASE_LEFT_LOWER_TARGET
                        phase_timers[i] = 0

                elif phases[i] == PHASE_LEFT_LOWER_TARGET:
                    # Right arm stays safely parked in right standby posture
                    commanded_right[i:i+1] = commanded_right[i:i+1] + torch.clamp(right_standby_joints[i:i+1] - commanded_right[i:i+1], min=-0.04, max=0.04)
                    # Gently lower baton onto target (Z=0.022m, soft landing 2mm above table surface)
                    manual_offset = torch.tensor([[-0.08, -0.08, 0.0]], device=args_cli.device)
                    final_place_target = target_pos + latched_left_grasp_offsets[i] + manual_offset
                    tgt_lower = final_place_target.clone()
                    tgt_lower[:, 2] = 0.022
                    err = tgt_lower - tcp_pos_l
                    dist = torch.norm(err)
                    dist_z = torch.abs(tgt_lower[:, 2] - tcp_pos_l[:, 2])
                    step_size = min(0.008, dist.item())
                    delta_l_pos_i = (err / (dist + 1e-6)) * step_size
                    left_orient_target = -torch.sign(target_cube_zs[i][:, 1:2] + 1e-6) * target_cube_zs[i]
                    delta_l_rot_i = compute_desired_grasp_rot(wrist_quat_l, left_orient_target, gain=0.10, max_rot_step=0.010, directed=True)
                    if (dist < 0.015 or dist_z < 0.006) or phase_timers[i] > 60:
                        phases[i] = PHASE_LEFT_RELEASE
                        phase_timers[i] = 0

                elif phases[i] == PHASE_LEFT_RELEASE:
                    # Right arm stays parked in right standby posture
                    commanded_right[i:i+1] = commanded_right[i:i+1] + torch.clamp(right_standby_joints[i:i+1] - commanded_right[i:i+1], min=-0.04, max=0.04)
                    # Open left gripper to release baton on target
                    left_gripper_cmd[i, 0] = 1.0
                    if phase_timers[i] > 10:
                        phases[i] = PHASE_LEFT_RETREAT
                        phase_timers[i] = 0

                elif phases[i] == PHASE_LEFT_RETREAT:
                    # Right arm stays parked in right standby posture
                    commanded_right[i:i+1] = commanded_right[i:i+1] + torch.clamp(right_standby_joints[i:i+1] - commanded_right[i:i+1], min=-0.04, max=0.04)
                    left_gripper_cmd[i, 0] = 1.0
                    # Move left arm up and retreat
                    tgt_up = target_pos.clone()
                    tgt_up[:, 2] += 0.20
                    err = tgt_up - tcp_pos_l
                    dist = torch.norm(err)
                    step_size = min(0.012, dist.item())
                    delta_l_pos_i = (err / (dist + 1e-6)) * step_size
                    if dist < 0.03 or phase_timers[i] > 40:
                        phases[i] = PHASE_SUCCESS
                        phase_timers[i] = 0

                # Real-time 3D state and tracking error monitoring
                if step % 25 == 0:
                    dist_pick = torch.norm(pick_target - tcp_pos_r).item() * 1000
                    dist_place = torch.norm(place_target - tcp_pos_l).item() * 1000
                    print(
                        f"  Step {step:03d} | {PHASE_NAMES[phases[i]]:<18} | "
                        f"Pick:[{pick_target[0,0]:.2f},{pick_target[0,1]:.2f},{pick_target[0,2]:.2f}] | "
                        f"R_TCP:[{tcp_pos_r[0,0]:.2f},{tcp_pos_r[0,1]:.2f},{tcp_pos_r[0,2]:.2f}]({dist_pick:.1f}mm) | "
                        f"Place:[{place_target[0,0]:.2f},{place_target[0,1]:.2f},{place_target[0,2]:.2f}] | "
                        f"L_TCP:[{tcp_pos_l[0,0]:.2f},{tcp_pos_l[0,1]:.2f},{tcp_pos_l[0,2]:.2f}]({dist_place:.1f}mm)"
                    )


                delta_r_pos[i] = delta_r_pos_i[0]
                delta_r_rot[i] = delta_r_rot_i[0]
                delta_l_pos[i] = delta_l_pos_i[0]
                delta_l_rot[i] = delta_l_rot_i[0]

            if all(p == PHASE_DONE for p in phases):
                break

            # --- 6D Inverse Kinematics with 7-DOF Nullspace Elbow Flare Projection ---
            jacobians = robot.root_physx_view.get_jacobians()

            delta_r_pose = torch.cat([delta_r_pos, delta_r_rot], dim=-1)
            if torch.norm(delta_r_pose) > 1e-5:
                j_r_wrist = jacobians[:, jacobi_right_hand_idx, :6, :][:, :, jacobi_right_joint_ids]
                j_r_tcp = get_tcp_jacobian(j_r_wrist, wrist_pos_r_all, tcp_pos_r_all)
                dq_r = solve_pose_dls_ik(
                    j_r_tcp,
                    delta_r_pose,
                    damping=damping_r.unsqueeze(-1),
                    q_current=commanded_right,
                    q_nominal=q_nominal_right,
                    k_null=k_null_r,
                )
                dq_r = torch.clamp(dq_r, min=-0.08, max=0.08)
                commanded_right += dq_r
                # Clamp within safe joint limits
                commanded_right = torch.clamp(
                    commanded_right,
                    min=soft_lower[:, right_arm_joint_ids] + 0.02,
                    max=soft_upper[:, right_arm_joint_ids] - 0.02,
                )

            delta_l_pose = torch.cat([delta_l_pos, delta_l_rot], dim=-1)
            if torch.norm(delta_l_pose) > 1e-5:
                j_l_wrist = jacobians[:, jacobi_left_hand_idx, :6, :][:, :, jacobi_left_joint_ids]
                j_l_tcp = get_tcp_jacobian(j_l_wrist, wrist_pos_l_all, tcp_pos_l_all)
                
                # Dynamic task-space relaxation: Free the wrist orientation during flight to maximize reach
                j_l_tcp = j_l_tcp.clone()
                for i in range(num_envs):
                    if phases[i] == PHASE_LEFT_HOVER_TARGET:
                        j_l_tcp[i, 3:6, :] = 0.0
                    
                dq_l = solve_pose_dls_ik(
                    j_l_tcp,
                    delta_l_pose,
                    damping=damping_l.unsqueeze(-1),
                    q_current=commanded_left,
                    q_nominal=q_nominal_left,
                    k_null=k_null_l,
                )
                dq_l = torch.clamp(dq_l, min=-0.08, max=0.08)
                commanded_left += dq_l
                commanded_left = torch.clamp(
                    commanded_left,
                    min=soft_lower[:, left_arm_joint_ids] + 0.02,
                    max=soft_upper[:, left_arm_joint_ids] - 0.02,
                )

            # --- Action Inversion & Mapping ---
            # 1. Update full articulation joint targets
            target_all_joints = robot.data.default_joint_pos.clone()
            target_all_joints[:, left_arm_joint_ids] = commanded_left
            target_all_joints[:, right_arm_joint_ids] = commanded_right

            # 2. Extract joints in the exact order arm_action expects
            arm_targets = target_all_joints[:, arm_term._joint_ids]

            # 3. Invert default offset and scale so applied == arm_targets
            raw_arm_action = (arm_targets - arm_term._offset) / arm_term._scale

            # 4. Grippers
            action = torch.cat([raw_arm_action, left_gripper_cmd, right_gripper_cmd], dim=-1)

            # Record step transition
            policy_obs = obs["policy"].detach().cpu().numpy()
            rgb_image = obs["image"]["rgb"].detach().cpu().numpy()
            action_np = action.detach().cpu().numpy()

            for i in range(num_envs):
                if phases[i] != PHASE_DONE:
                    ep_obs[i].append(policy_obs[i])
                    
                    img = rgb_image[i]
                    if img.dtype != np.uint8:
                        if img.max() <= 1.0:
                            img = (img * 255.0)
                        img = np.clip(img, 0, 255).astype(np.uint8)
                    ep_images[i].append(img)
                    
                    ep_actions[i].append(action_np[i])
                    object_poses[i].append(obj.data.root_state_w.cpu().numpy()[i])
                    robot_joint_poses[i].append(robot.data.joint_pos.cpu().numpy()[i])

            # Step simulation
            obs, reward, terminated, truncated, _ = env.step(action)
            
            for i in range(num_envs):
                if phases[i] != PHASE_DONE:
                    ep_rewards[i].append(float(reward[i].item()))

            # Check Termination / Success
            for i in range(num_envs):
                if phases[i] == PHASE_SUCCESS:
                    dist_to_target = torch.norm(obj_pos_all[i, :2] - target_pos_all[i, :2]).item()
                    if dist_to_target < 0.15 and obj_pos_all[i, 2].item() < 0.06:
                        save_episode_to_hdf5(
                            args_cli.dataset_file,
                            collected_count,
                            ep_obs[i],
                            ep_images[i],
                            ep_actions[i],
                            ep_rewards[i],
                            object_poses[i],
                            robot_joint_poses[i],
                            init_robot_pos[i] if init_robot_pos.ndim > 1 else init_robot_pos,
                            init_robot_quat[i] if init_robot_quat.ndim > 1 else init_robot_quat,
                            init_target_pos[i] if init_target_pos.ndim > 1 else init_target_pos,
                            init_target_quat[i] if init_target_quat.ndim > 1 else init_target_quat,
                        )
                        collected_count += 1
                        print(f"  [✓] Env {i} Successfully collected Demo #{collected_count-1} in {step} steps! (dist={dist_to_target:.3f}m)")
                    else:
                        print(f"  [X] Env {i} Object not on target (dist={dist_to_target:.3f}m, z={obj_pos_all[i, 2]:.3f}m). Discarding...")
                    phases[i] = PHASE_DONE

            if all(p == PHASE_DONE for p in phases):
                break

            if torch.any(terminated) or torch.any(truncated):
                for i in range(num_envs):
                    if terminated[i].item() or truncated[i].item():
                        if phases[i] != PHASE_DONE:
                            print(f"  [X] Env {i} Episode terminated early at step {step} (phase={PHASE_NAMES[phases[i]]}). Discarding...")
                            phases[i] = PHASE_DONE
                if all(p == PHASE_DONE for p in phases):
                    break

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f" Scripted Demo Generation Finished!")
    print(f" Total Demos in Dataset : {collected_count}")
    print(f" Time Elapsed           : {elapsed:.1f} seconds ({elapsed/max(1, collected_count-existing_demos):.1f} s/demo)")
    print("=" * 60)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()

```

## scripts/replay_demos.py
```py
# Copyright (c) 2026, Dual Arm Imitation Learning Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Replay Demonstrations Script (Kinematic / Visuomotor).

Replays recorded demonstration trajectories in Isaac Sim kinematically
to visually inspect and verify motion quality and task success.
"""

import argparse
import math
import os
import sys
import time
import h5py
import torch
import cv2
import numpy as np

from isaaclab.app import AppLauncher

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT_ROOT = os.path.dirname(PROJECT_ROOT)
for path in [PROJECT_ROOT, PARENT_ROOT]:
    if path not in sys.path:
        sys.path.insert(0, path)

parser = argparse.ArgumentParser(description="Kinematic Replay of multiple demonstrations.")
parser.add_argument(
    "--dataset",
    type=str,
    default=os.path.join(PROJECT_ROOT, "data", "demos.hdf5"),
    help="Path to HDF5 dataset file.",
)
parser.add_argument("--demo_idx", type=int, default=0, help="Starting index of demo to replay. Negative indices supported.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of demos to play simultaneously.")
parser.add_argument("--delay", type=float, default=0.033, help="Delay between frames in seconds.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True  # Force enable cameras for rendering

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from isaaclab.envs import ManagerBasedRLEnv
try:
    from configs.env_cfg import VisuomotorObsCfg
except ImportError:
    from dual_arm_il.configs.env_cfg import VisuomotorObsCfg
try:
    from configs.env_cfg import DualArmILEnvCfg
except ImportError:
    from dual_arm_il.configs.env_cfg import DualArmILEnvCfg

def main():
    if not os.path.exists(args_cli.dataset):
        raise FileNotFoundError(f"Dataset not found: {args_cli.dataset}")

    with h5py.File(args_cli.dataset, "r") as f:
        data_grp = f["data"]
        demo_keys = sorted([k for k in data_grp.keys() if k.startswith("demo_")], key=lambda x: int(x.split("_")[1]))

        if len(demo_keys) == 0:
            print("[Replay] No demonstrations found in dataset!")
            simulation_app.close()
            return
            
        num_avail = len(demo_keys)
        num_parallel = min(args_cli.num_envs, num_avail)
        
        if args_cli.demo_idx < 0:
            # If negative (e.g. -1), pick the last num_parallel demos
            start_idx = max(0, num_avail + args_cli.demo_idx - num_parallel + 1)
        else:
            start_idx = min(args_cli.demo_idx, num_avail - 1)
            
        end_idx = min(num_avail, start_idx + num_parallel)
        target_demos = demo_keys[start_idx : end_idx]
        num_parallel = len(target_demos)  # update to actual
        
        print(f"[Kinematic Replay] Preparing to play {num_parallel} demos in parallel: {target_demos}")

        # Check if kinematic data is available
        if "object_poses" not in data_grp[target_demos[0]]:
            print("ERROR: Dataset does not contain 'object_poses' and 'robot_joint_poses'.")
            print("Please regenerate the dataset using the updated generate_scripted_demos.py script!")
            simulation_app.close()
            return

        all_obj_traj = []
        all_joint_traj = []
        all_init_robot_pos = []
        all_init_robot_quat = []
        all_init_target_pos = []
        all_init_target_quat = []
        max_length = 0
        
        for key in target_demos:
            obj_traj = data_grp[key]["object_poses"][:]
            joint_traj = data_grp[key]["robot_joint_poses"][:]
            all_obj_traj.append(obj_traj)
            all_joint_traj.append(joint_traj)
            if len(obj_traj) > max_length:
                max_length = len(obj_traj)
                
            if "init_robot_pos" in data_grp[key]:
                all_init_robot_pos.append(data_grp[key]["init_robot_pos"][:])
                all_init_robot_quat.append(data_grp[key]["init_robot_quat"][:])
            else:
                all_init_robot_pos.append(None)
                all_init_robot_quat.append(None)
                
            if "init_target_pos" in data_grp[key]:
                all_init_target_pos.append(data_grp[key]["init_target_pos"][:])
                all_init_target_quat.append(data_grp[key]["init_target_quat"][:])
            else:
                all_init_target_pos.append(None)
                all_init_target_quat.append(None)

    env_cfg = DualArmILEnvCfg()
    env_cfg.sim.device = args_cli.device
    
    # -------------------------------------------------------------
    # Viewer & Recording Setup (Fixing Cropped FOV)
    # -------------------------------------------------------------
    env_cfg.viewer.resolution = (1920, 1080)
    if num_parallel == 1:
        env_cfg.viewer.eye = (1.5, 0.0, 1.2)
        env_cfg.viewer.lookat = (0.0, 0.0, 0.0)
    else:
        # Pull camera back and up for grid view
        offset = max(2.0, (num_parallel ** 0.5) * 1.5)
        env_cfg.viewer.eye = (offset, offset, offset * 0.8)
        env_cfg.viewer.lookat = (0.0, 0.0, 0.0)
        env_cfg.viewer.origin_type = "world"
    
    # FOR KINEMATIC REPLAY: Disable physics on the object so it doesn't fall or get pushed
    if hasattr(env_cfg.scene, "object") and hasattr(env_cfg.scene.object, "spawn"):
        from isaaclab.sim import RigidBodyPropertiesCfg
        env_cfg.scene.object.spawn.rigid_props = RigidBodyPropertiesCfg(
            kinematic_enabled=True,
            disable_gravity=True,
        )
    env_cfg.scene.num_envs = num_parallel
    env_cfg.observations = VisuomotorObsCfg()

    try:
        gym.register(
            id="Isaac-Dual-Arm-IL-v0",
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={"env_cfg_entry_point": env_cfg.__class__},
        )
    except Exception:
        pass
    env: ManagerBasedRLEnv = gym.make("Isaac-Dual-Arm-IL-v0", cfg=env_cfg, render_mode="rgb_array").unwrapped
    env.reset()
    
    # Setup Video Writers
    os.makedirs("videos", exist_ok=True)
    cctv_video_out = None
    global_video_out = None
    
    print(f"\n--- Starting KINEMATIC parallel replay (Max steps: {max_length}) ---")
    
    obj_state = env.scene["object"].data.default_root_state.clone()
    tgt_state = env.scene["target"].data.default_root_state.clone()
    rob_state = env.scene["robot"].data.default_root_state.clone()
    j_pos = env.scene["robot"].data.default_joint_pos.clone()
    j_vel = env.scene["robot"].data.default_joint_vel.clone() * 0.0
    
    for step_idx in range(max_length):
        if not simulation_app.is_running():
            break

        for i in range(num_parallel):
            o_traj = all_obj_traj[i]
            j_traj = all_joint_traj[i]
            idx = min(step_idx, len(o_traj) - 1)
            
            # If init_robot_pos was recorded, we use it to deduce the original env_origin during generation.
            # Assuming the robot base is spawned at local (0,0,0), its recorded world position IS the old env_origin.
            if all_init_robot_pos[i] is not None:
                gen_env_origin = torch.tensor(all_init_robot_pos[i], device=env.device)
                gen_env_origin[2] = 0.0  # Environment origins are always 2D (Z=0)
                
                # Robot world pos = (Old World Pos - Old Env Origin) + New Env Origin
                rob_state[i, :3] = (torch.tensor(all_init_robot_pos[i], device=env.device) - gen_env_origin) + env.scene.env_origins[i]
                rob_state[i, 3:7] = torch.tensor(all_init_robot_quat[i], device=env.device)
            else:
                gen_env_origin = torch.zeros(3, device=env.device)
            
            # Object world pos = (Old World Pos - Old Env Origin) + New Env Origin
            obj_state[i, :3] = (torch.tensor(o_traj[idx, :3], device=env.device) - gen_env_origin) + env.scene.env_origins[i]
            obj_state[i, 3:7] = torch.tensor(o_traj[idx, 3:7], device=env.device)
            j_pos[i] = torch.tensor(j_traj[idx], device=env.device)
            
            if all_init_target_pos[i] is not None:
                tgt_state[i, :3] = (torch.tensor(all_init_target_pos[i], device=env.device) - gen_env_origin) + env.scene.env_origins[i]
                tgt_state[i, 3:7] = torch.tensor(all_init_target_quat[i], device=env.device)
            
        env.scene["object"].write_root_state_to_sim(obj_state)
        env.scene["target"].write_root_state_to_sim(tgt_state)
        env.scene["robot"].write_root_state_to_sim(rob_state)
        env.scene["robot"].write_joint_state_to_sim(j_pos, j_vel)
        
        env.sim.step()
        
        # Update scene and compute observations to render camera
        env.scene.update(dt=env.physics_dt)
        obs_dict = env.observation_manager.compute()
        
        # Display camera view
        if "image" not in obs_dict:
            print(f"DEBUG: 'image' not in obs_dict. Keys are: {list(obs_dict.keys())}")
        if "image" in obs_dict:
            # 1. CCTV Video (from 'rgb' sensor)
            if "rgb" in obs_dict["image"] and obs_dict["image"]["rgb"] is not None:
                rgb_data = obs_dict["image"]["rgb"]
                rgb_np = rgb_data.clone().detach().cpu().numpy()
                if rgb_np.dtype != np.uint8:
                    if rgb_np.max() <= 10.0: # Allow HDR specular highlights > 1.0
                        rgb_np = (rgb_np * 255.0)
                    rgb_np = np.clip(rgb_np, 0, 255).astype(np.uint8)
                
                # Create a 2D grid
                n_imgs = num_parallel
                n_cols = math.ceil(math.sqrt(n_imgs))
                n_rows = math.ceil(n_imgs / n_cols)
                h_img, w_img, c_img = rgb_np[0].shape
                grid_img = np.zeros((n_rows * h_img, n_cols * w_img, c_img), dtype=rgb_np.dtype)
                for i in range(num_parallel):
                    row, col = divmod(i, n_cols)
                    grid_img[row*h_img:(row+1)*h_img, col*w_img:(col+1)*w_img] = rgb_np[i]
                
                # Convert to BGR for OpenCV
                if grid_img.shape[-1] == 3:
                    grid_img = cv2.cvtColor(grid_img, cv2.COLOR_RGB2BGR)
                elif grid_img.shape[-1] == 4:
                    grid_img = cv2.cvtColor(grid_img, cv2.COLOR_RGBA2BGR)
                    
                # Add camera position text to the first image
                cam_pos = env.scene["front_camera"].data.pos_w[0].cpu().numpy()
                cam_text = f"Cam Pos: [{cam_pos[0]:.2f}, {cam_pos[1]:.2f}, {cam_pos[2]:.2f}]"
                cv2.putText(grid_img, cam_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Write to CCTV video
                if cctv_video_out is None:
                    h, w = grid_img.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    cctv_video_out = cv2.VideoWriter('videos/cctv_video.mp4', fourcc, 30.0, (w, h))
                cctv_video_out.write(grid_img)

            # 2. Global Overarching Video (from env.render())
            global_img = env.render()
            if global_img is not None:
                g_rgb_np = np.array(global_img)
                if g_rgb_np.dtype != np.uint8:
                    if g_rgb_np.max() <= 10.0:
                        g_rgb_np = (g_rgb_np * 255.0)
                    g_rgb_np = np.clip(g_rgb_np, 0, 255).astype(np.uint8)
                
                if g_rgb_np.shape[-1] == 3:
                    g_rgb_np = cv2.cvtColor(g_rgb_np, cv2.COLOR_RGB2BGR)
                elif g_rgb_np.shape[-1] == 4:
                    g_rgb_np = cv2.cvtColor(g_rgb_np, cv2.COLOR_RGBA2BGR)
                    
                if global_video_out is None:
                    h, w = g_rgb_np.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    global_video_out = cv2.VideoWriter('videos/global_video.mp4', fourcc, 30.0, (w, h))
                global_video_out.write(g_rgb_np)
        
        if args_cli.delay > 0:
            time.sleep(args_cli.delay)

    print("Finished kinematic replay.")
    time.sleep(2.0)
    if cctv_video_out is not None:
        cctv_video_out.release()
        print("Saved CCTV video to videos/cctv_video.mp4")
    if global_video_out is not None:
        global_video_out.release()
        print("Saved Global overarching video to videos/global_video.mp4")
    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()

```

## scripts/train.py
```py
# Copyright (c) 2026, Dual Arm Imitation Learning Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Unified Training Script for Dual Arm Imitation Learning.

Supports:
    - Behavior Cloning (BC)
    - Diffusion Policy
    - Action Chunking with Transformers (ACT)

Usage:
    python scripts/train.py --algo=bc --epochs=100
    python scripts/train.py --algo=diffusion --epochs=150
    python scripts/train.py --algo=act --epochs=150
"""

import argparse
import os
import sys
import yaml
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataset.il_dataset import DualArmDataset
from models import MLPBCPolicy, RNNBCPolicy, DiffusionPolicy, ACTPolicy
import copy
import random
from torch.utils.data import Sampler, Subset

class ChunkedRandomSampler(Sampler):
    """
    Groups dataset indices into large sequential chunks, shuffles the chunks, 
    and then shuffles indices within each chunk.
    This guarantees 99%+ HDF5 chunk cache hit rates by preserving spatial locality,
    preventing catastrophic 'Chunk Thrashing' caused by purely random DataLoader sampling.
    """
    def __init__(self, data_source, chunk_size=20000):
        self.data_source = data_source
        self.chunk_size = chunk_size
        
    def __iter__(self):
        indices = list(range(len(self.data_source)))
        chunks = [indices[i:i + self.chunk_size] for i in range(0, len(indices), self.chunk_size)]
        random.shuffle(chunks)
        for chunk in chunks:
            random.shuffle(chunk)
            for idx in chunk:
                yield idx

    def __len__(self):
        return len(self.data_source)
import random

def apply_gpu_augmentation(images):
    """
    Applies Color Jitter and Random Crop purely on the GPU to completely eliminate CPU/Dataloader bottleneck.
    Expects images tensor: (B, T, C, H, W) float [0, 1]
    """
    B, T, C, H, W = images.shape
    device = images.device

    # 1. Color Jitter (Probability 80%)
    if random.random() < 0.8:
        # Generate random factors per batch element (B, 1, 1, 1, 1)
        # Apply the exact same jitter to all T frames in each sequence
        b_f = torch.empty(B, 1, 1, 1, 1, device=device).uniform_(0.8, 1.2)
        c_f = torch.empty(B, 1, 1, 1, 1, device=device).uniform_(0.8, 1.2)
        
        # Apply brightness (multiply)
        images = images * b_f
        
        # Apply contrast (interpolate with mean)
        # mean over C, H, W (last 3 dims)
        mean = images.mean(dim=[-3, -2, -1], keepdim=True)
        images = (images - mean) * c_f + mean
        
        images = torch.clamp(images, 0.0, 1.0)
        
    # 2. Random Crop (Translation)
    pad = 4
    # Fold B and T into a 4D tensor (B*T, C, H, W) because PyTorch replicate pad requires 3D/4D tensors
    images_4d = images.view(B * T, C, H, W)
    images_padded = torch.nn.functional.pad(images_4d, (pad, pad, pad, pad), mode='replicate')
    images_padded = images_padded.view(B, T, C, H + pad * 2, W + pad * 2)
    
    # We slice uniquely per batch element, but consistently across T
    # Using a fast loop over B to slice (very fast on GPU as it just modifies tensor views)
    cropped = torch.empty_like(images)
    for i in range(B):
        top = random.randint(0, pad * 2)
        left = random.randint(0, pad * 2)
        cropped[i] = images_padded[i, :, :, top:top+H, left:left+W]
        
    return cropped

class EMAModel:
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
                
    def step(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
                
    def copy_to(self, target_model):
        for name, param in target_model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name])


def parse_args():
    parser = argparse.ArgumentParser(description="Train imitation learning policies.")
    parser.add_argument(
        "--algo",
        type=str,
        default="diffusion",
        choices=["bc", "diffusion", "act"],
        help="Algorithm to train: bc, diffusion, or act.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=os.path.join(PROJECT_ROOT, "data", "demos.hdf5"),
        help="Path to HDF5 demonstration dataset.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML (defaults to configs/{algo}_cfg.yaml).",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate.")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--resume", type=str, default=None, help="Path to a checkpoint file to resume training from.")
    parser.add_argument("--num_workers", type=int, default=32, help="Number of CPU workers for DataLoader.")
    return parser.parse_args()


from torch.utils.tensorboard import SummaryWriter

def load_config(algo: str, custom_cfg_path: str | None) -> dict:
    if custom_cfg_path is None:
        cfg_path = os.path.join(PROJECT_ROOT, "configs", f"{algo}_cfg.yaml")
    else:
        cfg_path = custom_cfg_path

    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def build_model(algo: str, cfg: dict, obs_dim: int, act_dim: int) -> torch.nn.Module:
    if algo == "bc":
        model_type = cfg.get("model_type", "mlp")
        if model_type == "rnn":
            return RNNBCPolicy(
                obs_dim=obs_dim,
                act_dim=act_dim,
                hidden_dim=cfg.get("rnn_hidden_dim", 256),
                num_layers=cfg.get("rnn_num_layers", 2),
                dropout=cfg.get("dropout", 0.1),
            )
        else:
            return MLPBCPolicy(
                obs_dim=obs_dim,
                act_dim=act_dim,
                hidden_dims=cfg.get("hidden_dims", [512, 512, 256]),
                dropout=cfg.get("dropout", 0.1),
                activation=cfg.get("activation", "relu"),
            )

    elif algo == "diffusion":
        return DiffusionPolicy(
            obs_dim=obs_dim,
            act_dim=act_dim,
            pred_horizon=cfg.get("pred_horizon", 16),
            obs_horizon=cfg.get("obs_horizon", 2),
            num_train_timesteps=cfg.get("num_train_timesteps", 100),
            beta_schedule=cfg.get("beta_schedule", "squaredcos_cap_v2"),
            down_dims=cfg.get("down_dims", [256, 512, 1024]),
            kernel_size=cfg.get("kernel_size", 5),
            n_groups=cfg.get("n_groups", 8),
        )

    elif algo == "act":
        return ACTPolicy(
            obs_dim=obs_dim,
            act_dim=act_dim,
            chunk_size=cfg.get("chunk_size", 24),
            d_model=cfg.get("d_model", 256),
            nhead=cfg.get("nhead", 8),
            num_encoder_layers=cfg.get("num_encoder_layers", 4),
            num_decoder_layers=cfg.get("num_decoder_layers", 6),
            dim_feedforward=cfg.get("dim_feedforward", 1024),
            latent_dim=cfg.get("latent_dim", 32),
            kl_weight=cfg.get("kl_weight", 10.0),
            dropout=cfg.get("dropout", 0.1),
            temporal_ensembling=cfg.get("temporal_ensembling", True),
            temporal_ensemble_coeff=cfg.get("temporal_ensemble_coeff", 0.01),
        )
    else:
        raise ValueError(f"Unknown algo: {algo}")


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    cfg = load_config(args.algo, args.config)

    # Overrides
    epochs = args.epochs or cfg.get("epochs", 100)
    batch_size = args.batch_size or cfg.get("batch_size", 64)
    lr = args.lr or cfg.get("learning_rate", 1.0e-4)
    weight_decay = cfg.get("weight_decay", 1.0e-5)

    print("=" * 60)
    print(f" Training Imitation Learning Policy: {args.algo.upper()}")
    print("=" * 60)
    print(f" Dataset    : {args.dataset}")
    print(f" Batch Size : {batch_size}")
    print(f" Epochs     : {epochs}")
    print(f" Device     : {args.device}")
    print(f" LR         : {lr}")
    print("=" * 60)

    # Setup directories
    save_dir = os.path.join(PROJECT_ROOT, "checkpoints", args.algo)
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize TensorBoard Writer
    tb_dir = os.path.join(save_dir, "logs")
    writer = SummaryWriter(log_dir=tb_dir)
    print(f" TensorBoard logged to: {tb_dir}")

    # Load Dataset
    pred_h = cfg.get("pred_horizon", 16) if args.algo == "diffusion" else cfg.get("chunk_size", 24)
    obs_h = cfg.get("obs_horizon", 2)

    full_dataset = DualArmDataset(
        dataset_path=args.dataset,
        algo=args.algo,
        pred_horizon=pred_h,
        obs_horizon=obs_h,
    )

    # Save normalization stats for evaluation
    stats_path = os.path.join(save_dir, "stats.pkl")
    full_dataset.save_stats(stats_path)

    # Train / Val Split (90% train, 10% val)
    # CRITICAL: We MUST use sequential split instead of random_split.
    # random_split shuffles the dataset indices globally, destroying spatial locality
    # and causing catastrophic HDF5 chunk thrashing in DataLoader workers.
    val_size = max(1, int(0.1 * len(full_dataset)))
    train_size = len(full_dataset) - val_size
    train_dataset = Subset(full_dataset, range(0, train_size))
    val_dataset = Subset(full_dataset, range(train_size, len(full_dataset)))
    
    # Enable Data Augmentation for training dataset only
    val_dataset.dataset = copy.copy(full_dataset)
    val_dataset.dataset.is_train = False
    train_dataset.dataset.is_train = True

    # Optimize DataLoader for lazy HDF5 loading
    num_workers = args.num_workers
    print(f" CPU Workers: {num_workers}")
    
    # Use ChunkedRandomSampler to preserve HDF5 cache locality (20k transitions per chunk)
    sampler = ChunkedRandomSampler(train_dataset, chunk_size=20000)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=sampler, 
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=8 if num_workers > 0 else None
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=8 if num_workers > 0 else None
    )

    # Instantiate Model
    model = build_model(args.algo, cfg, full_dataset.obs_dim, full_dataset.act_dim)
    model.to(args.device)
    raw_model = model  # Keep reference to raw model for deepcopy, EMA, and saving

    # Optimizer & Scheduler (Use raw_model)
    optimizer = torch.optim.AdamW(raw_model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    # PyTorch AMP Scaler
    scaler = torch.cuda.amp.GradScaler()
    
    # Initialize EMA (Use raw_model)
    ema = EMAModel(raw_model)

    start_epoch = 1
    best_val_loss = float("inf")
    global_step = 0

    if args.resume:
        if os.path.exists(args.resume):
            print(f"[Model] Resuming training from checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=args.device, weights_only=False)
            
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                raw_model.load_state_dict(checkpoint["model_state_dict"])
                if "optimizer_state_dict" in checkpoint:
                    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                if "scheduler_state_dict" in checkpoint:
                    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                if "scaler_state_dict" in checkpoint:
                    scaler.load_state_dict(checkpoint["scaler_state_dict"])
                if "ema_shadow" in checkpoint:
                    # Clean _orig_mod. prefix from keys if they were accidentally saved with them
                    cleaned_shadow = {}
                    for k, v in checkpoint["ema_shadow"].items():
                        clean_k = k.replace("_orig_mod.", "")
                        cleaned_shadow[clean_k] = v
                    ema.shadow = cleaned_shadow
                if "epoch" in checkpoint:
                    start_epoch = checkpoint["epoch"] + 1
                if "val_loss" in checkpoint:
                    # Ignore old val_loss because the validation set distribution changed significantly
                    # (due to removing random_split data leakage and adding Augmentation).
                    # This forces the script to generate a new best_model.pt.
                    best_val_loss = float("inf")
                print(f"  --> Resumed from epoch {start_epoch - 1}")
            else:
                raw_model.load_state_dict(checkpoint)
                print(f"  --> Resumed raw weights only.")
        else:
            print(f"[Warning] Checkpoint not found: {args.resume}. Starting from scratch.")

    # Apply Torch Compile for 10-20% speedup on GPU AFTER loading weights
    try:
        model = torch.compile(raw_model)
        print("[Train] torch.compile() applied successfully for speedup.")
    except Exception as e:
        print(f"[Train] torch.compile() failed or not supported: {e}")

    total_params = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
    print(f"[Model] {args.algo.upper()} created with {total_params:,} trainable parameters.")

    # Training Loop
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        num_train_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for batch in pbar:
            batch = {k: v.to(args.device) for k, v in batch.items()}
            
            if "rgb_image" in batch:
                # Cast uint8 to float32 on GPU (Saves 75% PCIe bandwidth)
                batch["rgb_image"] = batch["rgb_image"].float() / 255.0
                # Apply ultra-fast GPU augmentation!
                batch["rgb_image"] = apply_gpu_augmentation(batch["rgb_image"])

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                loss_dict = model.compute_loss(batch)
                loss = loss_dict["loss"]
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            # Step EMA
            ema.step()

            train_loss_sum += loss.item()
            num_train_batches += 1
            global_step += 1
            pbar.set_postfix({"train_loss": f"{loss.item():.4f}"})
            
            # Log step-wise training loss
            writer.add_scalar("Loss/Train_Step", loss.item(), global_step)

        scheduler.step()
        avg_train_loss = train_loss_sum / max(1, num_train_batches)

        # Validation (using EMA model)
        # Create a temporary model copy for EMA validation
        ema_val_model = copy.deepcopy(raw_model)
        ema.copy_to(ema_val_model)
        ema_val_model.eval()
        
        val_loss_sum = 0.0
        num_val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(args.device) for k, v in batch.items()}
                if "rgb_image" in batch:
                    batch["rgb_image"] = batch["rgb_image"].float() / 255.0
                    
                with torch.cuda.amp.autocast():
                    loss_dict = ema_val_model.compute_loss(batch)
                val_loss_sum += loss_dict["loss"].item()
                num_val_batches += 1

        avg_val_loss = val_loss_sum / max(1, num_val_batches)
        print(f"Epoch {epoch:3d} | Train Loss: {avg_train_loss:.5f} | EMA Val Loss: {avg_val_loss:.5f}")

        # Log epoch-wise metrics to TensorBoard
        writer.add_scalar("Loss/Train_Epoch", avg_train_loss, epoch)
        writer.add_scalar("Loss/Val_Epoch", avg_val_loss, epoch)
        writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)

        # Prepare Checkpoint Dictionary
        checkpoint_dict = {
            "epoch": epoch,
            "algo": args.algo,
            "config": cfg,
            "obs_dim": full_dataset.obs_dim,
            "act_dim": full_dataset.act_dim,
            "model_state_dict": ema_val_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "ema_shadow": ema.shadow,
            "val_loss": best_val_loss,
        }

        # Save Best Checkpoint (Save EMA weights)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_ckpt_path = os.path.join(save_dir, "best_model.pt")
            
            torch.save(checkpoint_dict, best_ckpt_path)
            print(f"  --> Saved new best model to {best_ckpt_path} (Val Loss: {best_val_loss:.5f})")

        # Periodic checkpoint
        if epoch % cfg.get("save_interval", 20) == 0:
            ckpt_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pt")
            torch.save(checkpoint_dict, ckpt_path)

    writer.close()
    print("\n[Training Complete]")
    print(f"Best model saved at: {os.path.join(save_dir, 'best_model.pt')}")


if __name__ == "__main__":
    main()

```

## scripts/eval.py
```py
# Copyright (c) 2026, Dual Arm Imitation Learning Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Closed-loop Evaluation Script for Dual Arm Imitation Learning Policies.

Tests trained BC, Diffusion Policy, or ACT models in the Isaac Sim simulation
and logs task success metrics.

Usage:
    python scripts/eval.py --algo=diffusion --num_episodes=10
    python scripts/eval.py --algo=act --num_episodes=10
    python scripts/eval.py --algo=bc --num_episodes=10
"""

import argparse
import collections
import os
import sys
import torch

# Isaac Lab AppLauncher
from isaaclab.app import AppLauncher

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT_ROOT = os.path.dirname(PROJECT_ROOT)
for path in [PROJECT_ROOT, PARENT_ROOT]:
    if path not in sys.path:
        sys.path.insert(0, path)

parser = argparse.ArgumentParser(description="Evaluate trained imitation learning policies.")
parser.add_argument("--algo", type=str, default="diffusion", choices=["bc", "diffusion", "act"])
parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint (.pt).")
parser.add_argument("--stats", type=str, default=None, help="Path to normalization stats.pkl.")
parser.add_argument("--num_episodes", type=int, default=10, help="Number of evaluation episodes.")
parser.add_argument("--max_steps_per_ep", type=int, default=700, help="Max steps per episode (~23 seconds at 30Hz).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# FORCE ENABLE CAMERAS for Visuomotor project!
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from isaaclab.envs import ManagerBasedRLEnv
from configs.env_cfg import DualArmILEnvCfg

gym.register(
    id="Isaac-Dual-Arm-IL-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": DualArmILEnvCfg,
    },
)

from dataset.il_dataset import DualArmDataset
from models import MLPBCPolicy, RNNBCPolicy, DiffusionPolicy, ACTPolicy


def main():
    algo = args_cli.algo.lower()
    ckpt_path = args_cli.checkpoint or os.path.join(PROJECT_ROOT, "checkpoints", algo, "best_model.pt")
    stats_path = args_cli.stats or os.path.join(PROJECT_ROOT, "checkpoints", algo, "stats.pkl")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"Stats file not found: {stats_path}")

    print("=" * 60)
    print(f" Evaluating Policy: {algo.upper()}")
    print(f" Checkpoint : {ckpt_path}")
    print(f" Stats      : {stats_path}")
    print("=" * 60)

    # Load Normalization Stats
    stats = DualArmDataset.load_stats(stats_path)
    obs_mean = torch.tensor(stats["obs_mean"], device=args_cli.device, dtype=torch.float32)
    obs_std = torch.tensor(stats["obs_std"], device=args_cli.device, dtype=torch.float32)
    act_mean = torch.tensor(stats["act_mean"], device=args_cli.device, dtype=torch.float32)
    act_std = torch.tensor(stats["act_std"], device=args_cli.device, dtype=torch.float32)

    # Load Checkpoint & Instantiate Model
    checkpoint = torch.load(ckpt_path, map_location=args_cli.device)
    cfg = checkpoint.get("config", {})
    obs_dim = checkpoint.get("obs_dim", len(obs_mean))
    act_dim = checkpoint.get("act_dim", len(act_mean))

    if algo == "bc":
        model = MLPBCPolicy(obs_dim=obs_dim, act_dim=act_dim, hidden_dims=cfg.get("hidden_dims", [512, 512, 256]))
    elif algo == "diffusion":
        model = DiffusionPolicy(
            obs_dim=obs_dim,
            act_dim=act_dim,
            pred_horizon=cfg.get("pred_horizon", 16),
            obs_horizon=cfg.get("obs_horizon", 2),
            num_train_timesteps=cfg.get("num_train_timesteps", 100),
            beta_schedule=cfg.get("beta_schedule", "squaredcos_cap_v2"),
            down_dims=cfg.get("down_dims", [256, 512, 1024]),
        )
    elif algo == "act":
        model = ACTPolicy(
            obs_dim=obs_dim,
            act_dim=act_dim,
            chunk_size=cfg.get("chunk_size", 24),
            d_model=cfg.get("d_model", 256),
            nhead=cfg.get("nhead", 8),
            latent_dim=cfg.get("latent_dim", 32),
            temporal_ensembling=cfg.get("temporal_ensembling", True),
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(args_cli.device)
    model.eval()

    # Create Environment
    env_cfg = DualArmILEnvCfg()
    env_cfg.sim.device = args_cli.device
    print("[Eval] Initializing Isaac Lab Environment...")
    env: ManagerBasedRLEnv = gym.make("Isaac-Dual-Arm-IL-v0", cfg=env_cfg).unwrapped

    obs_horizon = cfg.get("obs_horizon", 2)
    act_horizon = cfg.get("act_horizon", 8)

    num_success = 0
    total_episodes = args_cli.num_episodes

    print(f"\nStarting {total_episodes} evaluation episodes...\n")

    for ep_idx in range(1, total_episodes + 1):
        obs, _ = env.reset()
        if hasattr(model, "reset_temporal_ensemble"):
            model.reset_temporal_ensemble()

        # Observation queue for Diffusion Policy
        obs_queue = collections.deque(maxlen=obs_horizon)
        # Action queue for multi-step receding horizon execution
        action_queue = collections.deque()

        ep_success = False

        for step in range(args_cli.max_steps_per_ep):
            if not simulation_app.is_running():
                break

            # Extract and normalize observation
            raw_obs = obs["policy"].squeeze(0)  # (obs_dim,)
            norm_obs = (raw_obs - obs_mean) / obs_std

            with torch.no_grad():
                if algo == "bc":
                    norm_action = model(norm_obs)
                    action = norm_action * act_std + act_mean

                elif algo == "diffusion":
                    obs_queue.append(norm_obs)
                    while len(obs_queue) < obs_horizon:
                        obs_queue.append(norm_obs)

                    if len(action_queue) == 0:
                        obs_tensor = torch.stack(list(obs_queue), dim=0).unsqueeze(0)  # (1, To, obs_dim)
                        pred_act_chunk = model.predict_action(obs_tensor, num_inference_steps=15)
                        pred_act_chunk = pred_act_chunk.squeeze(0)  # (Tp, act_dim)

                        # Enqueue first act_horizon actions
                        for a_idx in range(min(act_horizon, len(pred_act_chunk))):
                            act_unnorm = pred_act_chunk[a_idx] * act_std + act_mean
                            action_queue.append(act_unnorm)

                    action = action_queue.popleft()

                elif algo == "act":
                    norm_action = model.predict_action_step(norm_obs, current_step=step)
                    action = norm_action * act_std + act_mean

            # Step environment: action shape (1, act_dim)
            action_step = action.unsqueeze(0)
            obs, reward, terminated, truncated, _ = env.step(action_step)

            # Check task success: baton placed near red target
            obj_pos = env.scene["object"].data.root_pos_w.squeeze(0)
            target_pos = env.scene["target"].data.root_pos_w.squeeze(0)
            dist_to_target = torch.norm(obj_pos[:2] - target_pos[:2]).item()

            if dist_to_target < 0.12 and obj_pos[2].item() < 0.05:
                ep_success = True
                print(f"[Episode {ep_idx}] SUCCESS! Placed at target (dist: {dist_to_target:.3f}m, step: {step})")
                break

            if terminated.item() or truncated.item():
                break

        if ep_success:
            num_success += 1
        else:
            print(f"[Episode {ep_idx}] Failed or timed out.")

    success_rate = (num_success / total_episodes) * 100.0
    print("\n" + "=" * 60)
    print(f" Evaluation Completed: {algo.upper()}")
    print(f" Total Episodes : {total_episodes}")
    print(f" Successes      : {num_success}")
    print(f" Success Rate   : {success_rate:.1f}%")
    print("=" * 60)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()

```

## scripts/eval_parallel.py
```py
import argparse
import collections
import os
import sys
import math
import torch
import cv2
import numpy as np
import json

from isaaclab.app import AppLauncher

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT_ROOT = os.path.dirname(PROJECT_ROOT)
for path in [PROJECT_ROOT, PARENT_ROOT]:
    if path not in sys.path:
        sys.path.append(path)

# Parse Arguments
parser = argparse.ArgumentParser(description="Evaluate Visuomotor Imitation Learning Policy in Parallel.")
parser.add_argument("--algo", type=str, required=True, choices=["bc", "diffusion", "act"])
parser.add_argument("--checkpoint", type=str, required=False, default=None)
parser.add_argument("--num_episodes", type=int, default=100, help="Total episodes to evaluate")
parser.add_argument("--num_envs", type=int, default=16, help="Number of parallel environments")
parser.add_argument("--max_steps_per_ep", type=int, default=700)
parser.add_argument("--record_video", action="store_true", help="Record videos of the evaluation (Warning: causes CPU bottleneck and slows down evaluation).")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True # Force cameras on for Visuomotor

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from isaaclab.envs import ManagerBasedRLEnv
from configs.env_cfg import DualArmILEnvCfg, VisuomotorObsCfg
from models import MLPBCPolicy, DiffusionPolicy, ACTPolicy

gym.register(
    id="Isaac-Dual-Arm-IL-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": DualArmILEnvCfg,
    },
)
import threading
import queue

class AsyncVideoWriter:
    """Writes video frames in a background thread to prevent CPU bottlenecking the main RL loop."""
    def __init__(self, path, fourcc, fps, size):
        self.writer = cv2.VideoWriter(path, fourcc, fps, size)
        self.q = queue.Queue(maxsize=300) # Buffer approx 10 seconds of frames
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def write(self, frame):
        if not self.q.full():
            self.q.put(frame)
        else:
            print("[Warning] Video encoding queue full! Dropping frame to preserve RL speed.")

    def _worker(self):
        while self.running or not self.q.empty():
            try:
                frame = self.q.get(timeout=0.1)
                self.writer.write(frame)
                self.q.task_done()
            except queue.Empty:
                pass

    def release(self):
        self.running = False
        self.thread.join()
        self.writer.release()

def main():
    device = torch.device(args_cli.device)
    algo = args_cli.algo
    
    # Auto-resolve checkpoint path if not provided
    ckpt_path = args_cli.checkpoint
    if ckpt_path is None:
        ckpt_path = os.path.join(PROJECT_ROOT, "checkpoints", algo, "best_model.pt")

    save_dir = os.path.dirname(ckpt_path)
    stats_path = os.path.join(save_dir, "stats.pkl")

    if not os.path.exists(stats_path):
        print(f"[Error] Stats file not found at {stats_path}")
        sys.exit(1)
    
    from dataset.il_dataset import DualArmDataset
    stats = DualArmDataset.load_stats(stats_path)
    obs_mean = torch.tensor(stats["obs_mean"], device=args_cli.device, dtype=torch.float32)
    obs_std = torch.tensor(stats["obs_std"], device=args_cli.device, dtype=torch.float32)
    act_mean = torch.tensor(stats["act_mean"], device=args_cli.device, dtype=torch.float32)
    act_std = torch.tensor(stats["act_std"], device=args_cli.device, dtype=torch.float32)

    print(f"[Model] Loading checkpoint from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = checkpoint.get("config", {})
    obs_dim = checkpoint.get("obs_dim", len(obs_mean))
    act_dim = checkpoint.get("act_dim", len(act_mean))

    if algo == "bc":
        model = MLPBCPolicy(obs_dim=obs_dim, act_dim=act_dim, hidden_dims=cfg.get("hidden_dims", [512, 512, 256]))
    elif algo == "diffusion":
        model = DiffusionPolicy(
            obs_dim=obs_dim,
            act_dim=act_dim,
            pred_horizon=cfg.get("pred_horizon", 16),
            obs_horizon=cfg.get("obs_horizon", 2),
            num_train_timesteps=cfg.get("num_train_timesteps", 100),
            beta_schedule=cfg.get("beta_schedule", "squaredcos_cap_v2"),
            down_dims=cfg.get("down_dims", [256, 512, 1024]),
        )
    elif algo == "act":
        model = ACTPolicy(
            obs_dim=obs_dim,
            act_dim=act_dim,
            chunk_size=cfg.get("chunk_size", 24),
            d_model=cfg.get("d_model", 256),
            nhead=cfg.get("nhead", 8),
            latent_dim=cfg.get("latent_dim", 32),
            temporal_ensembling=cfg.get("temporal_ensembling", True),
        )

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    # Create Parallel Environment
    env_cfg = DualArmILEnvCfg()
    env_cfg.observations = VisuomotorObsCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device

    # -------------------------------------------------------------
    # Viewer & Recording Setup (Fixing Cropped FOV)
    # -------------------------------------------------------------
    env_cfg.viewer.resolution = (1920, 1080)
    if args_cli.num_envs > 1:
        import math
        # Isaac Lab arranges environments in a 2D grid
        n_cols = math.ceil(math.sqrt(args_cli.num_envs))
        env_spacing = env_cfg.scene.env_spacing
        cx = (n_cols - 1) * env_spacing / 2.0
        cy = (n_cols - 1) * env_spacing / 2.0
        
        # Pull the camera back and up to capture all parallel environments in the grid
        env_cfg.viewer.eye = (cx + 3.0 + n_cols * 1.5, cy, 2.0 + n_cols * 1.5)
        env_cfg.viewer.lookat = (cx, cy, 0.5)
    print("[Eval] Initializing Isaac Lab Environment...")
    env: ManagerBasedRLEnv = gym.make("Isaac-Dual-Arm-IL-v0", cfg=env_cfg, render_mode="rgb_array").unwrapped

    obs_horizon = cfg.get("obs_horizon", 2)
    act_horizon = cfg.get("act_horizon", 8)

    num_envs = args_cli.num_envs
    total_episodes = args_cli.num_episodes
    num_batches = math.ceil(total_episodes / num_envs)

    print(f"\nStarting {total_episodes} evaluation episodes across {num_batches} batches (Batch Size: {num_envs})...\n")

    total_success = 0
    total_evaluated = 0
    video_out = None
    global_video_out = None

    for batch_idx in range(num_batches):
        print(f"\n=== Batch {batch_idx + 1}/{num_batches} ===")
        obs, _ = env.reset()
        if hasattr(model, "reset_temporal_ensemble"):
            model.reset_temporal_ensemble()

        obs_queue = collections.deque(maxlen=obs_horizon)
        img_queue = collections.deque(maxlen=obs_horizon)
        action_queue = collections.deque()

        batch_success = torch.zeros(num_envs, dtype=torch.bool, device=device)
        batch_done = torch.zeros(num_envs, dtype=torch.bool, device=device)
        last_safe_action = None

        for step in range(args_cli.max_steps_per_ep):
            if not simulation_app.is_running():
                break

            # Extract and normalize observation
            raw_obs = obs["policy"].to(device)
            norm_obs = (raw_obs - obs_mean) / obs_std

            # Extract image (num_envs, H, W, C) -> (num_envs, C, H, W)
            raw_img = obs["image"]["rgb"].to(device)
            
            # --- Record front_camera video ---
            # --- Record Videos (Only if enabled, to prevent CPU bottleneck) ---
            if args_cli.record_video:
                rgb_np = raw_img.clone().detach().cpu().numpy()
                if rgb_np.dtype != np.uint8:
                    if rgb_np.max() <= 1.0:
                        rgb_np = (rgb_np * 255.0)
                    rgb_np = np.clip(rgb_np, 0, 255).astype(np.uint8)
                    
                # Create a 2D grid instead of a 1D strip to prevent video player cropping
                n_imgs = len(rgb_np)
                n_cols = math.ceil(math.sqrt(n_imgs))
                n_rows = math.ceil(n_imgs / n_cols)
                h_img, w_img, c_img = rgb_np[0].shape
                grid_img = np.zeros((n_rows * h_img, n_cols * w_img, c_img), dtype=rgb_np.dtype)
                for i, img in enumerate(rgb_np):
                    row, col = divmod(i, n_cols)
                    grid_img[row*h_img:(row+1)*h_img, col*w_img:(col+1)*w_img] = img
                
                if grid_img.shape[-1] == 3:
                    grid_img = cv2.cvtColor(grid_img, cv2.COLOR_RGB2BGR)
                elif grid_img.shape[-1] == 4:
                    grid_img = cv2.cvtColor(grid_img, cv2.COLOR_RGBA2BGR)
                    
                if video_out is None:
                    h, w = grid_img.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_out = AsyncVideoWriter(os.path.join(save_dir, 'eval_front_camera.mp4'), fourcc, 30.0, (w, h))
                video_out.write(grid_img)
                
                # --- Record Global Observer Video ---
                global_img = env.render()
                if global_img is not None:
                    if isinstance(global_img, list):
                        global_img = global_img[0]
                    g_rgb_np = global_img.cpu().numpy() if torch.is_tensor(global_img) else np.array(global_img)
                    if g_rgb_np.shape[-1] == 3:
                        g_rgb_np = cv2.cvtColor(g_rgb_np, cv2.COLOR_RGB2BGR)
                    elif g_rgb_np.shape[-1] == 4:
                        g_rgb_np = cv2.cvtColor(g_rgb_np, cv2.COLOR_RGBA2BGR)
                        
                    if global_video_out is None:
                        h, w = g_rgb_np.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        global_video_out = AsyncVideoWriter(os.path.join(save_dir, 'eval_global_camera.mp4'), fourcc, 30.0, (w, h))
                    global_video_out.write(g_rgb_np)
            # ---------------------------------
            
            img_obs = raw_img.float()
            if raw_img.dtype == torch.uint8:
                img_obs = img_obs / 255.0
            
            img_obs = torch.clamp(img_obs, 0.0, 1.0)
            img_obs = img_obs.permute(0, 3, 1, 2)

            with torch.no_grad():
                if algo == "bc":
                    norm_action = model(norm_obs, img_obs)
                    action = norm_action * act_std + act_mean

                elif algo == "diffusion":
                    obs_queue.append(norm_obs)
                    img_queue.append(img_obs)
                    
                    while len(obs_queue) < obs_horizon:
                        obs_queue.append(norm_obs)
                        img_queue.append(img_obs)

                    if len(action_queue) == 0:
                        obs_tensor = torch.stack(list(obs_queue), dim=0).permute(1, 0, 2)
                        img_tensor = torch.stack(list(img_queue), dim=0).permute(1, 0, 2, 3, 4)
                        
                        infer_steps = 15 # will be ignored by use_ddim=False
                        pred_act_chunk = model.predict_action(obs_tensor, img_tensor, num_inference_steps=infer_steps, use_ddim=False)
                        
                        start_a_idx = obs_horizon - 1
                        end_a_idx = start_a_idx + act_horizon
                        for a_idx in range(start_a_idx, min(end_a_idx, pred_act_chunk.shape[1])):
                            act_unnorm = pred_act_chunk[:, a_idx, :] * act_std + act_mean
                            action_queue.append(act_unnorm)

                    action = action_queue.popleft()

                elif algo == "act":
                    norm_action = model.predict_action_step(norm_obs, img_obs, current_step=step)
                    action = norm_action * act_std + act_mean

            # Freeze done environments
            if last_safe_action is None:
                last_safe_action = action.clone()
            else:
                last_safe_action[~batch_done] = action[~batch_done].clone()
                action[batch_done] = last_safe_action[batch_done]

            # Step environment
            obs, reward, terminated, truncated, _ = env.step(action)

            # Check task success and failures
            obj_pos = env.scene["object"].data.root_pos_w 
            target_pos = env.scene["target"].data.root_pos_w 
            dist_to_target = torch.norm(obj_pos[:, :2] - target_pos[:, :2], dim=1)

            # Success condition
            is_success = (dist_to_target < 0.15) & (obj_pos[:, 2] < 0.06)
            newly_succeeded = is_success & ~batch_done
            
            for env_idx in newly_succeeded.nonzero(as_tuple=True)[0]:
                print(f"  [Env {env_idx.item()}] SUCCESS! (Step: {step})")
            
            batch_success |= newly_succeeded
            batch_done |= is_success

            # Failure conditions
            is_dropped = obj_pos[:, 2] < -0.05
            joint_vel = env.scene["robot"].data.joint_vel
            is_spinning = torch.any(torch.abs(joint_vel) > 10.0, dim=1)
            
            # terminated is a tensor? In Isaac Lab, terminated is (num_envs,)
            term_mask = terminated if isinstance(terminated, torch.Tensor) else torch.tensor(terminated, device=device)
            trunc_mask = truncated if isinstance(truncated, torch.Tensor) else torch.tensor(truncated, device=device)

            newly_failed = (is_dropped | is_spinning | term_mask | trunc_mask) & ~batch_done
            for env_idx in newly_failed.nonzero(as_tuple=True)[0]:
                reasons = []
                if is_dropped[env_idx]: reasons.append("Dropped")
                if is_spinning[env_idx]: reasons.append(f"Spinning (vel={joint_vel[env_idx].abs().max().item():.2f})")
                if term_mask[env_idx]: reasons.append("Terminated")
                if trunc_mask[env_idx]: reasons.append("Truncated")
                print(f"  [Env {env_idx.item()}] FAILED: {', '.join(reasons)}. (Step: {step})")
            
            batch_done |= newly_failed

            if batch_done.all():
                print(f"All environments in batch {batch_idx + 1} finished early at step {step}.")
                break

        # Calculate results for this batch
        valid_envs_in_batch = min(num_envs, total_episodes - total_evaluated)
        total_success += batch_success[:valid_envs_in_batch].sum().item()
        total_evaluated += valid_envs_in_batch

        print(f"Batch {batch_idx + 1} Success Rate: {batch_success[:valid_envs_in_batch].sum().item()} / {valid_envs_in_batch}")

    success_rate = (total_success / total_evaluated) * 100.0 if total_evaluated > 0 else 0
    print("\n" + "=" * 60)
    print(f" Parallel Evaluation Completed: {algo.upper()}")
    print(f" Total Evaluated: {total_evaluated}")
    print(f" Successes      : {total_success}")
    print(f" Success Rate   : {success_rate:.1f}%")
    print("=" * 60)

    results_dict = {
        "algo": algo,
        "checkpoint": ckpt_path,
        "total_evaluated": total_evaluated,
        "successes": total_success,
        "success_rate_percent": success_rate,
        "max_steps_per_ep": args_cli.max_steps_per_ep
    }
    
    results_path = os.path.join(save_dir, "eval_results.json")
    with open(results_path, "w") as f:
        json.dump(results_dict, f, indent=4)
    print(f"Saved evaluation results to {results_path}")

    if video_out is not None:
        video_out.release()
        print(f"Saved evaluation front camera video to {os.path.join(save_dir, 'eval_front_camera.mp4')}")
        
    if global_video_out is not None:
        global_video_out.release()
        print(f"Saved evaluation global camera video to {os.path.join(save_dir, 'eval_global_camera.mp4')}")

    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()

```


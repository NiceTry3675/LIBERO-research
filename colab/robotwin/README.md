# Colab에서 RoboTwin 2.0 돌리기

WSL2에는 NVIDIA Vulkan 드라이버가 없어 RoboTwin이 고정한 SAPIEN 3.0.0b1이 렌더링하지 못한다. Colab GPU VM에서는 그대로 돈다. 2026-10-06에 L4 세션에서 확인했다.

## 순서

이 컴퓨터의 프로젝트 루트에서 실행한다.

```bash
bash colab/robotwin/launch.sh
```

L4 세션을 만들고 스크립트와 HF 토큰을 올린 뒤 `bootstrap.sh`를 VM에서 백그라운드로 시작한다. 같은 이름의 세션이 이미 있으면 그 세션을 쓴다. RoboDawn과 비교하는 공식 평가처럼 무작위화 장면이 필요하면 `--textures`를 붙인다. `--session`과 `--gpu`로 세션 이름과 GPU를 바꿀 수 있다.

`/content/bootstrap.log` 끝에 `BOOTSTRAP_DONE`이 나오면 끝난 것이다. 실패하면 `BOOTSTRAP_FAILED`와 실패한 단계가 찍힌다. 각 부분이 걸린 시간은 `TIME` 줄로 남는다. 끝나면 반드시 `colab stop -s robotwin`.

| 스크립트 | 하는 일 |
| --- | --- |
| `launch.sh` | 이 컴퓨터에서 실행한다. 세션 생성, 스크립트와 토큰 업로드, 설치 시작. |
| `bootstrap.sh` | VM의 진입점. 에셋 내려받기를 백그라운드로 먼저 시작하고, 그동안 환경 설치와 RoboTwin 설치를 차례로 한다. |
| `download_assets.sh` | 에셋 압축 파일을 병렬 전송으로 받아 `/content/assets_stage`에 동시에 푼다. RoboTwin 환경과 따로, 최신 huggingface_hub를 일회용 uv 환경에서 쓴다. |
| `setup_env.sh` | micromamba로 Python 3.10, CUDA 12.1 컴파일러, ffmpeg 환경(`/content/mamba/envs/rt`)을 만들고 torch 2.4.1(cu121)과 SAPIEN 3.0.0b1을 설치한다. |
| `install_rt.sh` | RoboTwin 복제, 의존성, XPolicyLab, sapien·mplib 패치, cuRobo v0.7.8 빌드. 마지막에 에셋을 기다렸다가 `RoboTwin/assets`로 옮긴다. |
| `run_expert.sh` | 전문가 스크립트로 과제를 몇 에피소드 돌려 시간을 잰다. 배경 질감이 없으면 무작위화 장면은 건너뛴다. |

- Colab의 gcc 13은 CUDA 12.1과 맞지 않아 conda-forge의 gcc 12로 cuRobo를 빌드한다.
- pytorch3d는 점군 샘플링에만 쓰여 설치하지 않는다.

### 배경 질감

배경 질감 11GB는 기본으로 받지 않는다. RoboTwin은 무작위화 설정에서만 이 질감을 읽고, 깨끗한 장면은 질감 없이 돈다. 개발과 디버깅은 깨끗한 장면으로 하고, RoboDawn과 숫자를 비교하는 평가만 `--textures`로 받는다. RoboDawn의 RoboTwin 성공률은 무작위화 장면에서 잰 값이고, 그 설정은 에피소드의 98%에 무작위 배경을 쓴다. 질감 없이 무작위화 설정을 돌리면 RoboTwin이 멈춘다.

### HF 토큰

`launch.sh`는 프로젝트 `.env`의 `HUGGINGFACE_API_KEY`만 꺼내 VM의 `/content/.hf_token`에 파일로 올린다. 실행하는 코드에는 토큰을 넣지 않는다. Colab CLI가 실행한 코드를 세션 기록에 남기기 때문이다. `.env`의 다른 키는 VM으로 가지 않는다. 토큰이 없으면 익명으로 받는다. 토큰은 요청 횟수 제한을 넉넉하게 할 뿐 대역폭은 그대로다. 내려받기에는 읽기 전용 토큰이면 충분하다.

## 확인 결과 (L4)

| 과제 | 장면 | 에피소드 | 시간 |
| --- | --- | --- | --- |
| beat_block_hammer | 깨끗함 | 3/3 | 113초 |
| place_empty_cup | 깨끗함 | 3/3 | 122초 |
| place_empty_cup | 무작위화 | 2/2 | 115초 |

시간은 시드 탐색과 데이터 수집을 합친 값이다. 렌더링은 GPU로 하며 단순한 장면에서 프레임당 약 1ms였다. L4 세션은 시간당 약 1.54 컴퓨트 단위를 쓴다.

첫 설치는 모든 단계를 차례로 돌려 약 40분 걸렸다. 에셋 내려받기와 cuRobo 빌드가 대부분이었다. 병렬화한 지금 스크립트로는 아직 재지 않았다.

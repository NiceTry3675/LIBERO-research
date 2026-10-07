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
| `prepare_robodawn.sh` | 고정 커밋의 RoboDawn 하네스와 시연을 받고, 수집한 기록과 오프라인으로 비교한다. |

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

## RoboDawn 기준선 재현 준비

`install_rt.sh`는 RoboTwin을 `96c1feab536306b50c26af200044fcdf126e8904`로 고정한다.
XPolicyLab도 해당 커밋의 gitlink를 유지한다. 기존 설치를 최신 커밋에서 그대로 재사용하면 같은 평가 조건이 아니다.

2026-10-07에 수집한 과제 50개·에피소드 500개는 `robodawn_site/`에 있다.
`scripts/audit_robodawn.py`는 공개 RoboDawn 커밋 `9247f366cd31f278e10f2fbe5fe8469b5f1b5b94`에서
500개 seed와 시연 entry, 실제 시연 이미지 파일을 점검하고 실행 manifest를 만든다.
저장한 사용자 프롬프트 10개와 시스템 프롬프트 2개는 공개 하네스의 생성 결과와 문자 단위로 일치한다.
중간 턴의 기억과 카메라 설명은 수집한 텍스트를 입력으로 재사용했다. 전체 에피소드 기록이나 렌더링을 재생한 검증은 아니다.

Colab CLI가 없는 컴퓨터에서는 다음 묶음을 만든 뒤, `reproduce_robodawn.ipynb`를 Colab에서 연다.
L4 GPU 런타임에서 첫 셀에 zip을 업로드하고 위에서부터 실행한다.

```bash
python scripts/package_robodawn.py
```

묶음은 `outputs/robodawn_reproduction.zip`에 생긴다. 설치 스크립트·실행기·수집 기록이 들어가며
API 키, `.env`, GPU 에셋은 포함하지 않는다. 노트북은 공개 코드와 에셋을 GPU VM에서 받는다.

설치가 끝난 GPU VM에서 직접 실행할 때:

```bash
cd /content/branchlab
bash colab/robotwin/prepare_robodawn.sh
# OPENROUTER_API_KEY는 셸에 안전하게 설정하거나 --api-key-file로 전달한다.
/content/mamba/envs/rt/bin/python scripts/run_robodawn_baseline.py --task adjust_bottle --dry-run
/content/mamba/envs/rt/bin/python scripts/run_robodawn_baseline.py --task adjust_bottle --episodes 1
/content/mamba/envs/rt/bin/python scripts/run_robodawn_baseline.py --task place_empty_cup --episodes 1
```

처음에는 두 과제에서 에피소드 0 하나씩 돌린다. 조건 점검이 통과하면 `--episodes 10`으로 이어서 실행한다.
`--start-episode`로 시작 인덱스를 바꿀 수 있으며 범위는 수집한 0~9 안으로 제한된다.
과제마다 완료된 에피소드는 같은 결과 폴더에서 건너뛴다.

실행기는 공개 하네스의 시뮬레이터·프롬프트·시연 선택·종료 조건을 그대로 사용한다.
OpenRouter 모델 이름과 공급자 제한만 연결하고 `reasoning.enabled=true`를 보낸다.
`src/branchlab/vlm.py`는 Clef 오프라인 비교용이므로 이 기준선 실행에는 쓰지 않는다.

| 조건 | 값 |
| --- | --- |
| 장면 / seed base | `demo_randomized` / `0` |
| 지시문 | `unseen`, 공개 하네스의 seed별 생성 방식 |
| 턴 / 명령 / 응답 토큰 한도 | `45` / `4` / `8000` |
| 시연 | 명령 primer 6장 + 초기 물체 좌우에 따라 고른 과제 시연 1개 |
| 현재 관측 | `agent_camera`, `top_camera`, `left_camera`, `right_camera` |
| 모델 / 공급자 | `google/gemini-3.8-flash` / `google-vertex`, fallback 없음 |
| 추론 | 활성화, OpenRouter 기본 강도. 실제 사용량은 실행 후 점검 |

실제 시연을 읽어 센 요청 이미지 수는 수집한 500개 에피소드에서 14~48장이다.
시연 entry 매핑과 시연 원본 seed는 `robodawn_site/reproduction_manifest.json`에 남겼다.
entry 선택이 사이트와 같은지는 GPU 실행 후 첫 턴의 `trace.json`에서 확인한다.

결과는 `outputs/robodawn/gemini_flash/<task>/shard_0/`에 저장된다.
공개 하네스의 `results.json`, `trace.json`, `llm_calls.jsonl`, 이미지·영상 외에 다음 기록을 남긴다.

- `reproduction_config.json`: 고정 커밋, seed와 예상 시연, 요청 조건
- `transport.jsonl`: 시도별 지연, 요청 공급자·추론 설정, 반환 공급자, 종료 이유, 토큰·비용 usage
- `condition_check.json`: seed·시연 일치, 반환 공급자, 양수 reasoning 토큰 확인. 조건이 어긋나면 실행기가 실패로 끝난다.

양수 reasoning 토큰은 추론이 동작했다는 근거이며 원 실험의 추론 강도가 같다는 증거는 아니다.
원 게이트웨이의 강도와 실제 토큰 사용량은 공개 사이트에 없었다.
또한 확인한 Vertex endpoint 메타데이터에는 `temperature`가 나열되지 않아 공개 하네스의 `temperature=0`이
그대로 적용되는지는 실제 endpoint에서 확인해야 한다. 기본 강도를 사용한 요청임을 기록하고 결과를 해석한다.
스모크 실행이나 조건 점검만으로 성공률 재현을 완료했다고 표시하지 않는다.

공개 코드와 API 설정 근거:
[RoboDawn](https://github.com/Hugo-AGI/RoboDawn/tree/9247f366cd31f278e10f2fbe5fe8469b5f1b5b94),
[OpenRouter 추론 설정](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens),
[공급자 제한](https://openrouter.ai/docs/guides/routing/provider-selection).

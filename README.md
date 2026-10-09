# branchlab

LIBERO 분기 실험 환경에서 시작해, 지금은 RoboDawn 하네스(RoboTwin 2.0)에서 큰 VLM의 판단을 결정 모델 Clef가 적용하는 연구를 한다. 현재 방향과 원칙은 [연구_방향.md](연구_방향.md)에 있다. 이전 구상서는 [v6](구상서_v6.md), [v5](구상서_v5.md).

## 설치

[uv](https://docs.astral.sh/uv/)가 필요하다. LIBERO를 `third_party/`에 고정 커밋으로 받고, `.venv`에 의존성을 설치한다.

```bash
./scripts/setup.sh
```

## 확인

```bash
uv run python scripts/smoke_test.py
```

과제 생성, 스텝 실행, 카메라 렌더링, 스냅숏 복원 후 재실행의 일치를 확인한다.

## 실험 과제

LIBERO-Object 3번 과제 "바비큐 소스를 집어 바구니에 넣기"를 쓴다. 병이 좁아서 믿음 위치가 2cm 틀려도 잡기가 성공하므로, 실패는 잡기 운이 아니라 잘못된 믿음에서 나온다. 가리개는 케첩 병이다. 과제와 물체 이름은 `scenarios.py`에 모여 있다.

## 구조

- `src/branchlab/env.py`: 과제 환경 생성, 초기 상태 로드
- `src/branchlab/snapshot.py`: 상태 저장·복원 (MuJoCo 상태 + 그리퍼·컨트롤러 내부 상태)
- `src/branchlab/perception.py`: 관측 모델. 위치 잡음·누락을 넣은 물체 인식(`PerceptionModel`), 주입하는 인식 오류(`Fault`), 관측으로만 갱신되는 믿음(`Belief`)
- `src/branchlab/visibility.py`: 고정 카메라에서 물체가 얼마나 보이는지 광선으로 판정. 관측 모델이 가려진 물체를 빼는 데 씀
- `src/branchlab/events.py`, `scenarios.py`: 외부 사건(물체 이동)과, 증상("대상 없음")은 같고 원인은 다른 방해 상황 여섯 가지
- `src/branchlab/executor.py`: 믿음과 고유감각만 보고 움직이는 스크립트 집기·놓기 실행기. 내려가기 전에 대상이 관측으로 확인돼야 잡는다(기본은 주 카메라, 탐색 뒤에는 손목 카메라)
- `src/branchlab/facts.py`: 믿음과 실행 상태를 이름 붙은 사실 목록(텍스트)으로 변환
- `src/branchlab/runner.py`: 에피소드 실행(`Session`). 판단 시점(단계 전환, 진행 정체)에만 물체를 관측하고, 판단 시점에서 멈춰 스냅숏·복원할 수 있음
- `src/branchlab/interventions.py`: 네 개입(계속 실행, 재관측, 국소 복구, 재계획)과 단계적 확대
- `src/branchlab/clef.py`: 결정 모델 Clef 호출기. Cloudflare Workers AI 직접 또는 OpenRouter 경유, 응답 캐시
- `src/branchlab/vlm.py`: 생성형 VLM 호출기(OpenRouter). Clef와 같은 질문을 JSON 답으로 받는다. Gemini는 BYOK가 걸린 Vertex로 고정
- `scripts/`: 설치, 스모크 테스트, 실행기 평가(`eval_executor.py`), 사실 목록 출력(`show_facts.py`), 원인별 방해 상황 실행(`run_scenarios.py`), 분기 실험 파일럿(`pilot.py`)과 집계(`analyze_pilot.py`), RoboDawn 시연으로 하는 Clef 오프라인 시험(`clef_offline.py`, `clef_servo_offline.py`)
- `colab/robotwin/`: Colab GPU에서 RoboTwin 2.0을 설치하고 돌리는 스크립트 (WSL2에서는 렌더링이 안 됨)
- `reports/`: 파일럿 보고서와 Clef 오프라인 실험 보고서
- `third_party/LIBERO`: LIBERO 원본 (git 추적 제외)
- `third_party/robodawn`: RoboDawn 저장소 일부 사본, RoboTwin 시연 데이터와 하네스 (git 추적 제외)
- `.libero/config.yaml`: LIBERO 경로 설정 (`~/.libero` 대신 프로젝트 안에 둠)

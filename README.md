# branchlab

LIBERO 분기 실험 환경. 연구 내용은 [구상서_v5.md](구상서_v5.md) 참고.

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

## 구조

- `src/branchlab/env.py`: 과제 환경 생성, 초기 상태 로드
- `src/branchlab/snapshot.py`: 상태 저장·복원 (MuJoCo 상태 + 그리퍼·컨트롤러 내부 상태)
- `src/branchlab/perception.py`: 관측 모델. 위치 잡음·누락을 넣은 물체 인식(`PerceptionModel`), 주입하는 인식 오류(`Fault`), 관측으로만 갱신되는 믿음(`Belief`)
- `src/branchlab/executor.py`: 믿음과 고유감각만 보고 움직이는 스크립트 집기·놓기 실행기
- `src/branchlab/facts.py`: 믿음과 실행 상태를 이름 붙은 사실 목록(텍스트)으로 변환
- `src/branchlab/runner.py`: 에피소드 실행. 판단 시점(단계 전환, 진행 정체)에만 물체를 관측
- `scripts/`: 설치, 스모크 테스트, 실행기 평가(`eval_executor.py`), 사실 목록 출력(`show_facts.py`)
- `third_party/LIBERO`: LIBERO 원본 (git 추적 제외)
- `.libero/config.yaml`: LIBERO 경로 설정 (`~/.libero` 대신 프로젝트 안에 둠)

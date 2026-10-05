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
- `src/branchlab/perception.py`: 실행기가 보는 관측(`Percept`). 지금은 시뮬레이터 정답값 그대로
- `src/branchlab/executor.py`: 관측만 보고 움직이는 스크립트 집기·놓기 실행기
- `scripts/`: 설치, 스모크 테스트, 실행기 평가(`eval_executor.py`)
- `third_party/LIBERO`: LIBERO 원본 (git 추적 제외)
- `.libero/config.yaml`: LIBERO 경로 설정 (`~/.libero` 대신 프로젝트 안에 둠)

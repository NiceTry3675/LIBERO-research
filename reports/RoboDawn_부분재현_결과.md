# RoboDawn 부분 재현 결과 (2026-10-08)

## 요약
- 재현 판정: **일치** (완료 에피소드 100/100)
- 전체 성공률: 우리 57.0% vs 사이트 56.0% (차이 +1.0%p, 95% 구간 -12.7%p ~ +14.7%p)
- 에피소드 일치율: 69.0% (100개 중 69개 에피소드에서 성공·실패가 같음)
- 조건 점검 실패: 없음 (조건 점검 22개 모두 `errors: []`. 본 실행 flex 20개와 standard 시험 2개)

해석: 전체 성공률은 원 결과와 거의 같다. 다만 에피소드 단위로는 31개가 엇갈렸다. 과제별로 보면 adjust_bottle(10 vs 7), place_object_stand(9 vs 6), open_laptop(5 vs 3)은 우리가 높고, place_fan(2 vs 4), move_can_pot(6 vs 8)은 낮았다. 과제당 10개라서 과제별 판정은 하지 않는다(README 기준).

## 과제별

`outputs/robodawn_colab/compare_flex.txt`의 과제별 표를 그대로 옮긴다.

```
task                        run   site  agree  turns run/site  ended
adjust_bottle             10/10   7/10   7/10            8/16  {'success': 10}
click_bell                10/10  10/10  10/10            5/10  {'success': 10}
handover_block             1/10   2/10   9/10           36/39  {'success': 1, 'max_turns': 7, 'object_fell': 2}
lift_pot                   0/10   0/10  10/10           30/36  {'max_turns': 5, 'object_fell': 5}
move_can_pot               6/10   8/10   6/10           13/15  {'object_fell': 3, 'success': 6, 'max_turns': 1}
open_laptop                5/10   3/10   4/10           28/34  {'success': 5, 'max_turns': 5}
place_a2b_right            8/10   9/10   7/10            9/12  {'object_fell': 2, 'success': 8}
place_empty_cup            6/10   7/10   7/10           23/18  {'success': 6, 'max_turns': 3, 'object_fell': 1}
place_fan                  2/10   4/10   4/10           26/22  {'object_fell': 4, 'max_turns': 4, 'success': 2}
place_object_stand         9/10   6/10   5/10           17/17  {'success': 9, 'object_fell': 1}

100 episodes: run 57.0%, site 56.0%, difference +1.0% (95% interval -12.7% to +14.7%) -> consistent with the site
episode agreement 69.0%

calls 1967 (answered 1967, failed attempts 0), traffic {'ON_DEMAND_FLEX': 1967}
tokens: input 54.1M (cached 62%), output 8.44M (reasoning 8.01M, reply 0.43M)
cut off by the token budget: 48 replies
call time: median 49.7s, 90th percentile 87.0s
rough cost (flex): $24.78; check the GCP bill for the real amount
```

중복 실행된 에피소드는 없다(비교 출력에 "run more than once" 줄이 없다).

## 실행 조건 확인
- RoboTwin 커밋: 두 VM 모두 `STEP_CLONE_OK 96c1feab536306b50c26af200044fcdf126e8904`
- STEP_ARCH: 두 VM 모두 `8.9` (L4)
- 두 VM 사용: 예. robotwin이 1/2 몫(6과제, 60 에피소드), robotwin2가 2/2 몫(4과제, 40 에피소드)을 맡았다. 두 번째 세션도 문제없이 만들어졌다.
- RoboDawn 준비: 두 VM 모두 `seed_pairs 500, demo_entries 500, demo_images_loaded true, all_user_prompts_exact true, all_system_prompts_exact true` 다음에 `ROBODAWN_READY`
- flex 트래픽 비율: 100% (1967/1967 `ON_DEMAND_FLEX`). 실패한 요청 0, HTTP 429/5xx 0
- 응답 예산(8000토큰)에 잘린 응답: 48개 (전체 1967개 중 2.4%)
- 캐시 비율: 62% (flex 본 실행). 시험 실행만 보면 59%
- 프로세스당 GPU 메모리
  - 4-3 시험 실행 중(프로세스 1개, place_empty_cup 로딩): 4639 MiB. GPU 전체 4668 / 23034 MiB
  - 본 실행 중(02:55, 프로세스 3개): 프로세스마다 약 4.6 / 6.1 / 8.4 GB였고, 에피소드를 거칠수록 늘었다. GPU 전체는 19.3 GB(robotwin), 19.1 GB(robotwin2) / 23.0 GB. 이 증가가 아래 "문제와 조치"의 샤드 실패 원인이다.
- 실험 조건(하네스 `third_party/robodawn`, `robodawn_site/`, `scripts/run_robodawn_baseline.py`의 요청 조건, 동시 실행 수 3, 샤드 크기 5)은 바꾸지 않았다. 저장소의 파일은 하나도 수정하지 않았다.

## 시간과 비용
- 시작 2026-10-08 00:21 KST (사전 점검), 종료 08:21 KST (두 VM 종료 확인)
- 단계별 소요
  - 첫 설치 시도: 00:21–00:49. 두 VM 모두 Colab이 회수해 잃었다(아래 참조).
  - 설치(재시도): 00:52 시작, 약 18분. robotwin 1095초, robotwin2 1054초(setup_env 149/141초, install_rt 946/913초, 에셋 내려받기 64초와 압축 풀기 103/100초). 확인은 01:13.
  - 준비(`setup`): 01:14 시작, 1분 이내에 `ROBODAWN_READY`
  - 시험(flex): 01:15:30–01:23 (약 8분, 두 에피소드 모두 5턴에 성공)
  - 본 실행: 01:29 시작. robotwin2는 07:57에 40/40, robotwin은 08:01에 60/60. 약 6.5시간이며, 실패한 샤드를 이어서 돌린 시간을 포함한다.
  - standard 지연 기준: 08:04:31–08:18 (약 14분)
  - 마무리(`fetch --full`, 종료): 08:05–08:21
- 호출 시간
  - flex: 중앙값 49.7초, 90% 87.0초 (1967회). 시험 실행만 보면 27.8초 / 49.1초.
  - standard: 중앙값 38.8초, 90% 63.5초 (14회, `ON_DEMAND`, 두 에피소드 5턴·9턴 성공) (`compare_standard.txt`)
- Colab 컴퓨트 단위: 시작 잔액 623.06, 종료 잔액 599.53이므로 **23.53 단위**를 썼다. 회수된 첫 설치 시도분 약 0.9단위가 포함돼 있다. 두 대 동시 사용률은 3.08/시간이었다.
- Vertex 대략 비용: flex $24.78과 standard $0.33을 합쳐 **약 $25.1**이다. 실제 금액은 GCP 청구서로 확인해야 한다.
- 예산 한도(Colab 80단위, $50)는 넘지 않았다.

## 문제와 조치

1. **Colab VM이 유휴 상태로 회수됨 (첫 설치 시도를 잃음)**
   - 증상: 00:49에 두 VM이 `colab sessions`에서 사라졌다. 마지막 `colab exec` 뒤 약 15분이 지난 시점이었고, 설치는 진행 중이었다.
   - 원인: Colab CLI는 커널이 활동하는 동안만 세션을 살려 둔다(CLI README의 "Activity-Based Liveness"). 그런데 `launch.sh`와 `subset.sh`는 작업을 분리된 `Popen`으로 띄우고 곧바로 돌아오므로 커널이 유휴 상태가 된다.
   - 조치: 이 컴퓨터에서 4분마다 각 세션에 `print("alive")`를 `colab exec`으로 보내는 keep-alive 루프를 돌렸다(저장소 밖 임시 스크립트). 그 뒤 약 7시간 반 동안 VM이 사라진 일은 없었다. 다시 설치하고 나서 진행했다.
   - 제안: 이 keep-alive를 `launch.sh`/`subset.sh` 쪽에 넣거나 README에 적어 두는 것이 좋겠다. 저장소 파일은 수정하지 않았다.
2. **`colab exec`이 가끔 끝나지 않음**
   - 첫 `launch.sh`가 새 커널에 처음 보낸 exec(`.hf_token` 권한 변경)에서 ReadTimeout으로 멈췄다. 같은 동작(chmod, bootstrap 시작)을 손으로 다시 실행해 마쳤다.
   - 그 밖에도 `status` exec 하나와 keep-alive exec 하나가 무기한 멈췄다. 이후 모든 감시 호출에 로컬 시간 제한(perl `alarm`)을 걸었다. 일시적인 "Connection was lost"도 1회 있었다.
   - 실행 결과에는 영향이 없다.
3. **샤드 실패 11회: `RuntimeError: cannot create buffer`**
   - 어디서: robotwin 5회(lift_pot 0-4, place_a2b_right 5-9, place_empty_cup 5-9, place_fan 0-4, place_fan 5-9), robotwin2 6회(handover_block 5-9, move_can_pot 5-9, open_laptop 0-4, open_laptop 5-9, place_object_stand 0-4, place_object_stand 5-9)
   - 증상: 모두 같은 오류였다. 다음 에피소드를 시작하는 `env.reset_episode` → `camera.take_picture`에서 났고, 대부분 프로세스 하나가 에피소드 2개를 마친 뒤였다.
   - 원인: 위에서 잰 대로 프로세스당 GPU 메모리가 에피소드마다 늘어, 3개를 동시에 돌리면 23 GB L4에서 Vulkan 버퍼를 만들지 못한다.
   - 결과에 미치는 영향: 실패가 에피소드 시작 전에 나므로 진행 중이던 에피소드를 잃지 않았고, 성공·실패에 따라 실패하는 구조도 아니다.
   - 조치: 동시 실행 수 3은 그대로 두고, 각 VM의 대기열이 끝나 `SUBSET_DONE`이 나오면 같은 `run --part K/2` 명령을 다시 실행했다(robotwin2 04:41, robotwin 05:38). 끝난 에피소드는 건너뛴다. 큐가 도는 중에 다시 실행하면 진행 중인 샤드와 겹치므로 끝날 때까지 기다렸다.
   - 같은 샤드가 두 번 실패한 경우는 없다. 재실행 한 번으로 모든 샤드가 끝났다.
   - 마지막에 남은 에피소드가 없음을 확인하는 재실행이 각 VM에서 한 번씩 더 돌았다. 모델 호출 없이 바로 끝났다.
   - 제안: L4에서 3개를 동시에 돌리려면 하네스 밖에서 메모리 증가를 해결하거나, 샤드 크기를 2로 줄여 한 프로세스가 에피소드 3개째를 시작하지 않게 하는 방법을 검토할 만하다(이번에는 조건 유지를 위해 바꾸지 않았다).
4. **설치 스크립트 수정**: 없음.
5. **키 처리**: 서비스 계정 키와 `.env`의 내용은 읽거나 출력하지 않았다. 키는 `subset.sh setup`이 파일째 올렸고, `subset.sh stop`이 VM에서 지운 뒤 VM을 껐다(두 VM 모두 "key removed"). 업로드 묶음(`outputs/robodawn_reproduction.zip`)에 키나 `.env`가 없는 것도 확인했다.
6. **커밋과 push**: 하지 않았다.

## 파일
- 결과 위치: `outputs/robodawn_colab/robotwin/robodawn/`, `outputs/robodawn_colab/robotwin2/robodawn/` (`fetch --full`이므로 이미지·영상 포함. 각각 1.1 GB와 973 MB, 영상 62개와 40개)
  - 진행 기록: 각 폴더의 `subset_flex_<세션>/status.tsv`와 샤드별 로그
  - standard 기준: `robotwin/robodawn/gemini_flash_standard/`
- 비교: `outputs/robodawn_colab/compare_flex.txt`, `outputs/robodawn_colab/compare_standard.txt`

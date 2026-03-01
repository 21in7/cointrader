# 맥미니 로컬 학습 후 LXC 배포 구현 계획

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 맥미니에서 LightGBM 모델을 학습하고, 학습된 모델 파일(`lgbm_filter.pkl`)을 Proxmox LXC 컨테이너로 자동 전송하여 봇이 즉시 사용할 수 있도록 한다.

**Architecture:**
- 맥미니에서 `scripts/train_model.py`를 직접 실행하여 모델 학습 (M 시리즈 칩 병렬 처리 활용)
- 학습 완료 후 `scp` 또는 `rsync`로 LXC 호스트에 모델 파일 전송
- LXC 컨테이너 내 `models/` 볼륨 마운트 경로에 파일이 도달하면 봇이 자동으로 핫 리로드

**Tech Stack:** Python 3.12, LightGBM, joblib, scp/rsync, SSH, docker-compose volume mount

---

## 전제 조건 확인

- 맥미니에 Python 3.12 + 의존성 설치 가능
- LXC 호스트 IP: `10.1.10.24`
- SSH 키 인증 등록 완료 (`ssh root@10.1.10.24` 비밀번호 없이 접속 가능)
- LXC 컨테이너에서 `./models`가 `/app/models`로 볼륨 마운트 중

---

## Task 1: 맥미니 환경 준비

**Files:**
- Read: `requirements.txt`

**Step 1: 의존성 설치 확인**

```bash
# 맥미니 터미널에서 실행
cd /Users/gihyeon/github/cointrader
pip install -r requirements.txt
```

Expected: 모든 패키지 설치 완료 (lightgbm, pandas, joblib 등)

**Step 2: 데이터 수집**

```bash
python scripts/fetch_history.py --symbol XRPUSDT --interval 1m --days 90 --output data/xrpusdt_1m.parquet
```

Expected: `저장 완료: data/xrpusdt_1m.parquet (약 130,000행)`

**Step 3: 학습 실행 (맥미니 전체 코어 활용)**

```bash
# M 시리즈 맥미니는 cpu_count()가 올바르게 반환되므로 --jobs 생략 가능
python scripts/train_model.py --data data/xrpusdt_1m.parquet
```

Expected 출력:
```
캔들 수: 130000
병렬 처리: N코어 사용 (총 129940개 인덱스)
...
검증 AUC: 0.XXXX
모델 저장: models/lgbm_filter.pkl
```

**Step 4: 학습 결과 확인**

```bash
ls -lh models/lgbm_filter.pkl
python -c "import joblib; m = joblib.load('models/lgbm_filter.pkl'); print('모델 로드 OK:', type(m))"
```

Expected: 파일 존재, 로드 성공

---

## Task 2: LXC 전송 스크립트 작성

**Files:**
- Create: `scripts/deploy_model.sh`

**Step 1: 전송 스크립트 작성**

`scripts/deploy_model.sh` 파일을 생성한다:

```bash
#!/usr/bin/env bash
# 맥미니에서 학습한 모델을 LXC 컨테이너 볼륨 경로로 전송한다.
# 사용법: bash scripts/deploy_model.sh [LXC_HOST] [LXC_MODELS_PATH]
#
# 예시:
#   bash scripts/deploy_model.sh 10.1.10.28 /path/to/cointrader/models
#   bash scripts/deploy_model.sh root@10.1.10.28 /root/cointrader/models

set -euo pipefail

LXC_HOST="${1:-root@10.1.10.24}"
LXC_MODELS_PATH="${2:-/root/cointrader/models}"
LOCAL_MODEL="models/lgbm_filter.pkl"
LOCAL_LOG="models/training_log.json"

if [[ ! -f "$LOCAL_MODEL" ]]; then
  echo "[오류] 모델 파일 없음: $LOCAL_MODEL"
  echo "먼저 python scripts/train_model.py 를 실행하세요."
  exit 1
fi

echo "=== 모델 전송 시작 ==="
echo "  대상: ${LXC_HOST}:${LXC_MODELS_PATH}"
echo "  파일: $LOCAL_MODEL"

# 기존 모델을 prev로 백업 (원격)
ssh "${LXC_HOST}" "
  if [ -f '${LXC_MODELS_PATH}/lgbm_filter.pkl' ]; then
    cp '${LXC_MODELS_PATH}/lgbm_filter.pkl' '${LXC_MODELS_PATH}/lgbm_filter_prev.pkl'
    echo '  기존 모델 백업 완료'
  fi
  mkdir -p '${LXC_MODELS_PATH}'
"

# 모델 파일 전송
rsync -avz --progress \
  "$LOCAL_MODEL" \
  "${LXC_HOST}:${LXC_MODELS_PATH}/lgbm_filter.pkl"

# 학습 로그도 함께 전송 (있을 경우)
if [[ -f "$LOCAL_LOG" ]]; then
  rsync -avz "$LOCAL_LOG" "${LXC_HOST}:${LXC_MODELS_PATH}/training_log.json"
  echo "  학습 로그 전송 완료"
fi

echo "=== 전송 완료 ==="
echo ""
echo "봇이 실행 중이라면 아래 명령으로 모델을 즉시 리로드할 수 있습니다:"
echo "  docker exec cointrader python -c \\"
echo "    \"from src.ml_filter import MLFilter; f=MLFilter(); f.reload_model(); print('리로드 완료')\""
```

**Step 2: 실행 권한 부여**

```bash
chmod +x scripts/deploy_model.sh
```

**Step 3: 커밋**

```bash
git add scripts/deploy_model.sh
git commit -m "feat: add deploy_model.sh for mac-to-lxc model transfer"
```

---

## Task 3: LXC 경로 확인 및 SSH 접속 테스트

**Step 1: LXC 호스트 SSH 접속 확인**

```bash
# 맥미니 터미널에서 (SSH 키 등록 완료 상태)
ssh root@10.1.10.24 "echo 접속 성공"
```

Expected: `접속 성공`

**Step 2: LXC 컨테이너 내 models 경로 확인**

LXC 호스트에서 docker-compose.yml의 볼륨 마운트 경로를 확인한다:

```bash
ssh root@10.1.10.24 "docker inspect cointrader | grep -A5 Mounts"
```

Expected 출력 예시:
```json
"Mounts": [
  {
    "Source": "/root/cointrader/models",
    "Destination": "/app/models",
    ...
  }
]
```

`Source` 경로가 LXC 호스트에서 실제로 파일을 복사해야 할 위치다.

**Step 3: 경로 기록**

확인된 경로를 메모해 둔다. 예:
- LXC 호스트: `root@10.1.10.24`
- models 볼륨 소스: `/root/cointrader/models` (또는 실제 확인된 경로)

---

## Task 4: 모델 전송 실행

**Step 1: 전송 스크립트 실행**

```bash
# 맥미니 터미널에서 (cointrader 프로젝트 루트)
bash scripts/deploy_model.sh root@10.1.10.24 /root/cointrader/models
```

Expected:
```
=== 모델 전송 시작 ===
  대상: root@10.1.10.24:/root/cointrader/models
  파일: models/lgbm_filter.pkl
  기존 모델 백업 완료
lgbm_filter.pkl ... 전송 완료
  학습 로그 전송 완료
=== 전송 완료 ===
```

**Step 2: LXC에서 파일 존재 확인**

```bash
ssh root@10.1.10.24 "ls -lh /root/cointrader/models/"
```

Expected: `lgbm_filter.pkl`, `lgbm_filter_prev.pkl`, `training_log.json` 확인

---

## Task 5: 봇 핫 리로드 확인

**Step 1: 봇 컨테이너에서 모델 리로드**

봇이 실행 중인 경우 `MLFilter.reload_model()`을 트리거한다.

방법 A — 컨테이너 재시작 (가장 간단):
```bash
ssh root@10.1.10.24 "cd /root/cointrader && docker compose restart cointrader"
```

방법 B — 핫 리로드 (재시작 없이):
```bash
ssh root@10.1.10.24 "docker exec cointrader python -c \
  \"import sys; sys.path.insert(0,'src'); \
    from src.ml_filter import MLFilter; \
    f = MLFilter(); \
    print('모델 로드:', f.is_model_loaded())\""
```

**Step 2: 봇 로그에서 모델 로드 확인**

```bash
ssh root@10.1.10.24 "docker logs cointrader --tail 20"
```

Expected 로그:
```
INFO | ML 필터 모델 로드 완료: models/lgbm_filter.pkl
```

---

## Task 6: 자동화 스크립트 통합 (선택 사항)

**Files:**
- Create: `scripts/train_and_deploy.sh`

전체 파이프라인(수집 → 학습 → 전송)을 한 번에 실행하는 스크립트:

```bash
#!/usr/bin/env bash
# 맥미니에서 전체 학습 파이프라인을 실행하고 LXC로 배포한다.
# 사용법: bash scripts/train_and_deploy.sh [LXC_HOST] [LXC_MODELS_PATH]

set -euo pipefail

LXC_HOST="${1:-root@10.1.10.24}"
LXC_MODELS_PATH="${2:-/root/cointrader/models}"

echo "=== [1/3] 데이터 수집 ==="
python scripts/fetch_history.py --symbol XRPUSDT --interval 1m --days 90

echo ""
echo "=== [2/3] 모델 학습 ==="
python scripts/train_model.py --data data/xrpusdt_1m.parquet

echo ""
echo "=== [3/3] LXC 배포 ==="
bash scripts/deploy_model.sh "$LXC_HOST" "$LXC_MODELS_PATH"

echo ""
echo "=== 전체 파이프라인 완료 ==="
```

```bash
chmod +x scripts/train_and_deploy.sh
git add scripts/train_and_deploy.sh
git commit -m "feat: add train_and_deploy.sh for full pipeline on mac"
```

---

## 운영 워크플로우 요약

```
맥미니 (빠른 학습)                     LXC 컨테이너 (운영)
─────────────────────                  ────────────────────
1. fetch_history.py                    
2. train_model.py                      
3. deploy_model.sh ──── rsync ────→   models/lgbm_filter.pkl
                                       (볼륨 마운트로 컨테이너에 즉시 반영)
                                       
                                       docker compose restart
                                       → MLFilter.reload_model()
                                       → 새 모델로 거래 재개
```

---

## 주의사항

- `models/lgbm_filter.pkl`은 joblib으로 직렬화된 LightGBM 모델이다. **Python 버전이 다르면 로드 실패**할 수 있다. 맥미니와 LXC 컨테이너의 Python 버전을 일치시킬 것 (현재 Python 3.12 기준).
- Docker 이미지 내 Python 버전 확인: `docker exec cointrader python --version`
- 버전 불일치 시 맥미니에서도 동일 버전 가상환경을 사용하거나, Docker 컨테이너 안에서 학습하는 방식으로 전환해야 한다.
- `retrainer.py`의 자동 재학습(매일 새벽 3시)은 LXC에서 계속 동작한다. 맥미니에서 수동 학습한 모델이 자동 재학습으로 덮어쓰여질 수 있으므로, 자동 재학습 스케줄과 충돌하지 않도록 타이밍을 조율한다.

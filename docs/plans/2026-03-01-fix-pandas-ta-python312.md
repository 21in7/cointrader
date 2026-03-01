# pandas-ta Python 버전 호환성 수정 계획

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Jenkins CI에서 `pandas-ta==0.4.71b0`이 Python 3.11에서 설치 실패하는 문제를 해결한다.

**Architecture:** `pandas-ta==0.4.71b0`은 Python >=3.12를 요구하므로, Dockerfile의 베이스 이미지를 `python:3.11-slim`에서 `python:3.12-slim`으로 업그레이드한다. `requirements.txt`의 의존 패키지 버전도 Python 3.12와 호환되는 버전으로 정리한다.

**Tech Stack:** Docker, Python 3.12-slim, pandas-ta 0.4.71b0, python-binance 1.0.19

---

## 문제 분석

Jenkins 빌드 로그 오류:
```
ERROR: Ignored the following versions that require a different python version:
  0.4.67b0 Requires-Python >=3.12; 0.4.71b0 Requires-Python >=3.12
ERROR: Could not find a version that satisfies the requirement pandas-ta==0.4.71b0
```

**원인:** `requirements.txt`에 `pandas-ta==0.4.71b0`이 명시되어 있으나, Dockerfile 베이스 이미지가 `python:3.11-slim`이라 Python 3.12 이상을 요구하는 `pandas-ta`를 설치할 수 없다.

**해결 방향:** Dockerfile 베이스 이미지를 `python:3.12-slim`으로 변경한다.

---

## Task 1: Dockerfile 베이스 이미지 업그레이드

**Files:**
- Modify: `Dockerfile:1`

**Step 1: Dockerfile 수정**

`Dockerfile` 1번째 줄을 다음과 같이 변경한다:

변경 전:
```dockerfile
FROM python:3.11-slim
```

변경 후:
```dockerfile
FROM python:3.12-slim
```

**Step 2: 변경 내용 확인**

```bash
head -1 /Users/gihyeon/github/cointrader/Dockerfile
```

Expected:
```
FROM python:3.12-slim
```

---

## Task 2: requirements.txt 의존성 호환성 검토 및 수정

**Files:**
- Modify: `requirements.txt`

**Step 1: 현재 requirements.txt 내용 확인**

```bash
cat /Users/gihyeon/github/cointrader/requirements.txt
```

Expected (현재 내용):
```
python-binance==1.0.19
pandas>=2.2.0
pandas-ta==0.4.71b0
python-dotenv==1.0.0
httpx>=0.27.0
pytest>=8.1.0
pytest-asyncio>=0.24.0
aiohttp==3.9.3
websockets==12.0
loguru==0.7.2
```

**Step 2: pandas-ta 0.4.71b0의 의존성 확인**

PyPI 정보에 따르면 `pandas-ta==0.4.71b0`은 다음을 요구한다:
- `numba==0.61.2`
- `numpy>=2.2.6`
- `pandas>=2.3.2`

`requirements.txt`의 `pandas>=2.2.0`은 `pandas>=2.3.2`를 만족하므로 문제없다.  
단, `numba`가 명시되어 있지 않아 pandas-ta 설치 시 자동으로 설치된다.

**Step 3: requirements.txt 수정 (pandas 최소 버전 상향)**

`pandas>=2.2.0`을 `pandas>=2.3.2`로 변경하여 pandas-ta의 요구사항을 명시적으로 반영한다:

변경 전:
```
pandas>=2.2.0
```

변경 후:
```
pandas>=2.3.2
```

**Step 4: 변경 내용 확인**

```bash
grep "pandas" /Users/gihyeon/github/cointrader/requirements.txt
```

Expected:
```
pandas>=2.3.2
pandas-ta==0.4.71b0
```

---

## Task 3: 로컬 Docker 빌드 테스트

> Docker가 설치된 환경에서만 실행한다.

**Step 1: Docker 빌드**

```bash
cd /Users/gihyeon/github/cointrader
docker build -t cointrader:test .
```

Expected: 빌드 성공 (`Successfully tagged cointrader:test` 또는 `#N DONE`)

**Step 2: 빌드된 이미지의 Python 버전 확인**

```bash
docker run --rm cointrader:test python --version
```

Expected:
```
Python 3.12.x
```

**Step 3: pandas-ta import 확인**

```bash
docker run --rm cointrader:test python -c "import pandas_ta; print(pandas_ta.__version__)"
```

Expected:
```
0.4.71b0
```

**Step 4: 테스트 이미지 정리**

```bash
docker rmi cointrader:test
```

---

## Task 4: git 커밋 및 Gitea push

**Files:**
- Modify: `Dockerfile`
- Modify: `requirements.txt`

**Step 1: git 상태 확인**

```bash
cd /Users/gihyeon/github/cointrader
git status
```

Expected:
```
modified:   Dockerfile
modified:   requirements.txt
```

**Step 2: 변경 내용 검토**

```bash
git diff Dockerfile requirements.txt
```

Expected:
- `Dockerfile`: `-FROM python:3.11-slim` → `+FROM python:3.12-slim`
- `requirements.txt`: `-pandas>=2.2.0` → `+pandas>=2.3.2`

**Step 3: 스테이징**

```bash
git add Dockerfile requirements.txt
```

**Step 4: 커밋**

```bash
git commit -m "fix: upgrade to Python 3.12 to support pandas-ta>=0.4.67b0"
```

Expected: 커밋 성공

**Step 5: Gitea push**

```bash
git push origin main
```

Expected: push 성공 후 Jenkins가 자동으로 새 빌드를 트리거함

**Step 6: 커밋 확인**

```bash
git log --oneline -3
```

Expected: 방금 만든 커밋이 최상단에 표시됨

---

## Task 5: Jenkins 빌드 재실행 및 결과 확인

**Step 1: Jenkins 빌드 트리거**

Gitea push 후 Jenkins Webhook이 설정되어 있다면 자동으로 빌드가 트리거된다.  
수동으로 트리거하려면 Jenkins 웹 UI에서 `cointrader` 파이프라인 → `Build Now` 클릭.

**Step 2: 빌드 로그에서 성공 확인**

Jenkins 빌드 로그에서 다음 내용이 나타나야 한다:

```
#9 [5/7] RUN pip install --no-cache-dir -r requirements.txt
...
Successfully installed pandas-ta-0.4.71b0 ...
#9 DONE xx.xs
```

오류 없이 `[Build Docker Image]` 스테이지가 완료되어야 한다.

**Step 3: 전체 파이프라인 성공 확인**

Jenkins 빌드 결과가 `SUCCESS`로 표시되어야 한다:
```
Finished: SUCCESS
```

---

## 트러블슈팅

| 문제 | 원인 | 해결 |
|------|------|------|
| `python-binance==1.0.19` 설치 실패 | Python 3.12 비호환 | `python-binance>=1.0.19`로 변경하거나 최신 버전 확인 |
| `aiohttp==3.9.3` 설치 실패 | Python 3.12 비호환 | `aiohttp>=3.9.3`으로 완화하거나 최신 버전으로 업그레이드 |
| `numba` 설치 시간 초과 | numba 컴파일 시간 | 빌드 타임아웃 설정 증가 또는 `--timeout=300` 추가 |
| Jenkins Webhook 미동작 | Gitea Webhook 미설정 | Gitea 저장소 설정 → Webhooks → Jenkins URL 추가 |

---

## 참고: Python 3.12 호환성 체크리스트

Python 3.11 → 3.12 주요 변경사항 중 이 프로젝트에 영향 가능한 항목:

- `asyncio` 동작 변경: `asyncio.get_event_loop()` deprecated → `asyncio.get_running_loop()` 권장
- `typing` 모듈 일부 변경: `Union[X, Y]` → `X | Y` 문법 지원 강화
- `datetime.utcnow()` deprecated → `datetime.now(timezone.utc)` 권장

현재 코드베이스(`src/`, `tests/`)에서 위 패턴 사용 여부를 확인하고 필요 시 수정한다.

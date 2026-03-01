# Dockerfile & docker-compose.yml 작성 및 Gitea 업로드 구현 계획

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** cointrader 프로젝트에 Dockerfile과 docker-compose.yml을 추가하고, 변경사항을 커밋하여 Gitea(10.1.10.28:3000)에 push한다.

**Architecture:** Python 3.11 slim 이미지 기반의 멀티스테이지 없는 단일 Dockerfile을 작성하고, docker-compose.yml로 환경변수(.env)를 주입하여 컨테이너를 실행한다. 로그 디렉토리는 볼륨으로 마운트하여 컨테이너 재시작 시에도 보존한다.

**Tech Stack:** Docker, docker-compose v2, Python 3.11-slim, python-dotenv

---

## Task 1: Dockerfile 작성

**Files:**
- Create: `Dockerfile`

**Step 1: Dockerfile 생성**

`/Users/gihyeon/github/cointrader/Dockerfile` 파일을 아래 내용으로 생성한다:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p logs

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

CMD ["python", "main.py"]
```

**Step 2: Dockerfile 내용 확인**

```bash
cat /Users/gihyeon/github/cointrader/Dockerfile
```

Expected: 위 내용이 그대로 출력됨

**Step 3: Docker 빌드 테스트 (Docker가 설치된 경우)**

```bash
cd /Users/gihyeon/github/cointrader
docker build -t cointrader:test .
```

Expected: `Successfully built <image_id>` 또는 `Successfully tagged cointrader:test`

> Docker가 설치되지 않은 환경이라면 이 단계는 건너뛴다.

---

## Task 2: docker-compose.yml 작성

**Files:**
- Create: `docker-compose.yml`

**Step 1: docker-compose.yml 생성**

`/Users/gihyeon/github/cointrader/docker-compose.yml` 파일을 아래 내용으로 생성한다:

```yaml
services:
  cointrader:
    build: .
    container_name: cointrader
    restart: unless-stopped
    env_file:
      - .env
    volumes:
      - ./logs:/app/logs
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "5"
```

**Step 2: docker-compose.yml 내용 확인**

```bash
cat /Users/gihyeon/github/cointrader/docker-compose.yml
```

Expected: 위 내용이 그대로 출력됨

**Step 3: docker-compose 문법 검증 (docker compose가 설치된 경우)**

```bash
cd /Users/gihyeon/github/cointrader
docker compose config
```

Expected: 파싱된 YAML 설정이 오류 없이 출력됨

---

## Task 3: .dockerignore 작성

**Files:**
- Create: `.dockerignore`

**Step 1: .dockerignore 생성**

`/Users/gihyeon/github/cointrader/.dockerignore` 파일을 아래 내용으로 생성한다:

```
.env
.venv
__pycache__
*.pyc
*.pyo
.pytest_cache
logs/
*.log
.git
docs/
tests/
```

> `.env`를 반드시 포함시켜 빌드 컨텍스트에서 제외한다. 이미지에 API 키가 포함되는 것을 방지한다.

**Step 2: .dockerignore 내용 확인**

```bash
cat /Users/gihyeon/github/cointrader/.dockerignore
```

Expected: 위 내용이 그대로 출력됨

---

## Task 4: git 커밋

**Files:**
- Modify: `Dockerfile` (신규)
- Modify: `docker-compose.yml` (신규)
- Modify: `.dockerignore` (신규)

**Step 1: git 상태 확인**

```bash
cd /Users/gihyeon/github/cointrader
git status
```

Expected: `Dockerfile`, `docker-compose.yml`, `.dockerignore`가 untracked files로 표시됨

**Step 2: 스테이징**

```bash
cd /Users/gihyeon/github/cointrader
git add Dockerfile docker-compose.yml .dockerignore
```

**Step 3: 스테이징 내용 검토 (`.env` 포함 여부 확인)**

```bash
git diff --cached --name-only
```

Expected:
```
.dockerignore
Dockerfile
docker-compose.yml
```

`.env`가 목록에 **없어야** 한다. 만약 있다면 즉시 `git reset HEAD .env` 실행 후 중단.

**Step 4: 커밋**

```bash
git commit -m "chore: add Dockerfile, docker-compose.yml, .dockerignore"
```

Expected: `main` 브랜치에 새 커밋 생성

**Step 5: 커밋 확인**

```bash
git log --oneline -3
```

Expected: 방금 만든 커밋이 최상단에 표시됨

---

## Task 5: Gitea push

> 이 Task는 Gitea 원격 저장소가 이미 설정되어 있다고 가정한다.  
> 아직 설정하지 않았다면 `docs/plans/2026-03-01-upload-to-gitea.md`의 Task 2~3을 먼저 완료한다.

**Step 1: 현재 원격 저장소 확인**

```bash
cd /Users/gihyeon/github/cointrader
git remote -v
```

Expected:
```
origin  http://10.1.10.28:3000/<사용자명>/cointrader.git (fetch)
origin  http://10.1.10.28:3000/<사용자명>/cointrader.git (push)
```

origin이 없다면 아래 명령으로 추가 (`<사용자명>` 교체 필요):
```bash
git remote add origin http://10.1.10.28:3000/<사용자명>/cointrader.git
```

**Step 2: push**

```bash
git push origin main
```

> Gitea 계정의 사용자명과 비밀번호(또는 액세스 토큰)를 입력하라는 프롬프트가 나타남

Expected:
```
Enumerating objects: ...
Writing objects: 100% ...
```

**Step 3: push 결과 확인**

```bash
git log --oneline origin/main -3
```

Expected: 로컬 커밋 히스토리와 동일하게 표시됨

**Step 4: Gitea 웹 UI에서 파일 확인**

브라우저에서 `http://10.1.10.28:3000/<사용자명>/cointrader` 접속 후 다음 파일이 있는지 확인:
- `Dockerfile`
- `docker-compose.yml`
- `.dockerignore`

---

## 트러블슈팅

| 문제 | 원인 | 해결 |
|------|------|------|
| `docker build` 시 `gcc` 설치 실패 | 네트워크 문제 | `apt-get` 단계를 제거하고 빌드 재시도 (pandas-ta가 gcc 없이 설치되는지 확인) |
| `docker compose config` 오류 | YAML 들여쓰기 오류 | 탭 대신 스페이스 2칸 사용 여부 확인 |
| push 시 `Authentication failed` | 잘못된 계정 정보 | Gitea 웹 UI 로그인 테스트 후 동일 계정 사용 |
| push 시 `non-fast-forward` | 원격에 이미 다른 커밋 존재 | `git pull --rebase origin main` 후 재시도 |
| 컨테이너 실행 시 `.env` 없음 오류 | `.env` 파일 미생성 | `.env.example`을 복사하여 `.env` 생성 후 값 입력 |

---

## 참고: 컨테이너 실행 방법

```bash
# .env 파일 준비
cp .env.example .env
# .env 파일에 실제 API 키와 Discord Webhook URL 입력

# 빌드 및 백그라운드 실행
docker compose up -d --build

# 로그 확인
docker compose logs -f

# 중지
docker compose down
```

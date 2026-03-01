# Jenkins + Gitea 이미지 레지스트리 CI/CD 구현 계획

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Jenkins가 Gitea(10.1.10.28:3000)의 코드 변경을 감지하면 Docker 이미지를 빌드하여 Gitea Container Registry(10.1.10.28:5000 또는 Gitea 내장 패키지 레지스트리)에 push하고, docker-compose.yml이 해당 이미지를 pull해서 실행하도록 전체 CI/CD 파이프라인을 구성한다.

**Architecture:**
- Jenkins는 Gitea webhook을 통해 main 브랜치 push 이벤트를 수신한다.
- Jenkinsfile(파이프라인 스크립트)이 프로젝트 루트에 위치하며, `docker build → docker push → (선택) 원격 배포` 단계를 수행한다.
- Gitea의 내장 Container Registry(Packages)를 이미지 저장소로 사용한다. 이미지 이름 형식: `10.1.10.28:3000/gihyeon/cointrader:<tag>`
- docker-compose.yml은 `build: .` 대신 레지스트리 이미지를 직접 참조하도록 수정한다.

**Tech Stack:** Jenkins, Gitea Container Registry, Docker, docker-compose v2, Jenkinsfile(Declarative Pipeline)

---

## 사전 확인 사항

- Gitea 서버: `http://10.1.10.28:3000`
- Gitea 저장소: `http://10.1.10.28:3000/gihyeon/cointrader.git`
- Gitea Container Registry 주소: `10.1.10.28:3000` (HTTP 사용 시 Docker insecure-registries 설정 필요)
- Jenkins 서버 주소: 별도 확인 필요 (아래 Task 1에서 확인)
- 현재 Dockerfile: `FROM python:3.12-slim` 기반, `/app`에서 `python main.py` 실행

---

## Task 1: 환경 사전 점검

**Files:**
- 확인: `Dockerfile`
- 확인: `docker-compose.yml`

**Step 1: Gitea Container Registry(Packages) 활성화 확인**

브라우저에서 `http://10.1.10.28:3000/gihyeon/cointrader/packages` 접속.
- 패키지 탭이 보이면 활성화된 것.
- 안 보이면 Gitea 관리자 패널 → `Site Administration` → `Configuration` → `Enable Packages` 체크 필요.

**Step 2: Gitea Access Token 생성 (Jenkins용)**

`http://10.1.10.28:3000/user/settings/applications` 접속:
- Token Name: `jenkins-cointrader`
- 권한: `read:packages`, `write:packages` (또는 전체 권한)
- `Generate Token` 클릭 후 **토큰 값을 반드시 복사** (다시 볼 수 없음)

**Step 3: Docker insecure-registries 설정 (HTTP 레지스트리 사용 시)**

Jenkins가 실행되는 서버(또는 로컬 Mac)에서:

```bash
# /etc/docker/daemon.json 또는 Docker Desktop의 경우 Settings > Docker Engine
cat /etc/docker/daemon.json
```

아래 내용이 없으면 추가:
```json
{
  "insecure-registries": ["10.1.10.28:3000"]
}
```

Docker Desktop 사용 시: `Settings` → `Docker Engine` → JSON에 위 내용 병합 → `Apply & Restart`

**Step 4: Docker login 테스트**

```bash
docker login 10.1.10.28:3000 -u gihyeon -p <위에서_생성한_토큰>
```

Expected:
```
Login Succeeded
```

---

## Task 2: Jenkinsfile 작성

**Files:**
- Create: `Jenkinsfile`

**Step 1: Jenkinsfile 생성**

`/Users/gihyeon/github/cointrader/Jenkinsfile` 파일을 아래 내용으로 생성:

```groovy
pipeline {
    agent any

    environment {
        REGISTRY      = '10.1.10.28:3000'
        IMAGE_NAME    = 'gihyeon/cointrader'
        IMAGE_TAG     = "${env.BUILD_NUMBER}"
        FULL_IMAGE    = "${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
        LATEST_IMAGE  = "${REGISTRY}/${IMAGE_NAME}:latest"
        GITEA_CREDS   = credentials('gitea-registry-credentials')
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Build Image') {
            steps {
                sh "docker build -t ${FULL_IMAGE} -t ${LATEST_IMAGE} ."
            }
        }

        stage('Push to Gitea Registry') {
            steps {
                sh """
                    echo ${GITEA_CREDS_PSW} | docker login ${REGISTRY} -u ${GITEA_CREDS_USR} --password-stdin
                    docker push ${FULL_IMAGE}
                    docker push ${LATEST_IMAGE}
                """
            }
        }

        stage('Cleanup') {
            steps {
                sh """
                    docker rmi ${FULL_IMAGE} || true
                    docker rmi ${LATEST_IMAGE} || true
                """
            }
        }
    }

    post {
        success {
            echo "Build #${env.BUILD_NUMBER} pushed: ${FULL_IMAGE}"
        }
        failure {
            echo "Build #${env.BUILD_NUMBER} FAILED"
        }
    }
}
```

> **참고:**
> - `GITEA_CREDS`는 Jenkins Credentials에 등록할 Username+Password 자격증명 ID다 (Task 3에서 등록).
> - `IMAGE_TAG`는 Jenkins 빌드 번호를 사용한다. 태그 전략을 git 커밋 해시로 바꾸려면 `"${env.GIT_COMMIT[0..7]}"` 사용.
> - `Cleanup` 스테이지는 Jenkins 서버 디스크 절약을 위해 빌드 후 로컬 이미지를 삭제한다.

**Step 2: Jenkinsfile 내용 확인**

```bash
cat /Users/gihyeon/github/cointrader/Jenkinsfile
```

Expected: 위 내용이 출력됨

---

## Task 3: Jenkins에 Gitea Credentials 등록

**Step 1: Jenkins 웹 UI 접속**

`http://<jenkins-서버-주소>:8080` 접속 (Jenkins 서버 주소 확인 필요)

**Step 2: Credentials 등록**

`Jenkins` → `Manage Jenkins` → `Credentials` → `System` → `Global credentials` → `Add Credentials`:

| 항목 | 값 |
|------|----|
| Kind | Username with password |
| Scope | Global |
| Username | `gihyeon` |
| Password | Task 1 Step 2에서 생성한 Gitea Access Token |
| ID | `gitea-registry-credentials` |
| Description | Gitea Container Registry for cointrader |

`Create` 클릭

**Step 3: 등록 확인**

Credentials 목록에 `gitea-registry-credentials`가 표시되는지 확인

---

## Task 4: Jenkins Pipeline Job 생성

**Step 1: 새 Pipeline Job 생성**

`Jenkins` → `New Item`:
- Item name: `cointrader`
- Type: `Pipeline`
- `OK` 클릭

**Step 2: Pipeline 설정**

`Pipeline` 섹션에서:
- Definition: `Pipeline script from SCM`
- SCM: `Git`
- Repository URL: `http://10.1.10.28:3000/gihyeon/cointrader.git`
- Credentials: (Gitea 저장소 접근용 credentials 추가, 없으면 Task 3과 동일하게 추가)
- Branch Specifier: `*/main`
- Script Path: `Jenkinsfile`

`Save` 클릭

**Step 3: 수동 빌드 테스트**

`Build Now` 클릭 → Console Output 확인

Expected:
```
[Pipeline] stage: Build Image
Successfully built ...
[Pipeline] stage: Push to Gitea Registry
Login Succeeded
The push refers to repository [10.1.10.28:3000/gihyeon/cointrader]
...
latest: digest: sha256:... size: ...
[Pipeline] stage: Cleanup
Finished: SUCCESS
```

---

## Task 5: Gitea Webhook 설정 (자동 트리거)

**Step 1: Gitea 저장소 Webhook 추가**

`http://10.1.10.28:3000/gihyeon/cointrader/settings/hooks` 접속:
- `Add Webhook` → `Gitea`
- Target URL: `http://<jenkins-서버-주소>:8080/gitea-webhook/post`
  - Jenkins Gitea Plugin 사용 시 위 URL 형식
  - 일반 Generic Webhook 사용 시: `http://<jenkins-서버-주소>:8080/job/cointrader/build?token=<토큰>`
- Trigger: `Push Events`
- Branch filter: `main`
- `Add Webhook` 클릭

**Step 2: Jenkins에 Gitea Plugin 설치 (미설치 시)**

`Manage Jenkins` → `Plugins` → `Available plugins` → `Gitea` 검색 → 설치 후 재시작

**Step 3: Webhook 테스트**

Gitea Webhook 설정 페이지에서 `Test Delivery` 클릭

Expected: Jenkins에서 새 빌드가 자동으로 시작됨

---

## Task 6: docker-compose.yml 수정

**Files:**
- Modify: `docker-compose.yml`

현재 `docker-compose.yml`은 `build: .`으로 로컬 빌드를 사용한다. 이를 레지스트리 이미지를 pull해서 실행하도록 변경한다.

**Step 1: docker-compose.yml 수정**

`/Users/gihyeon/github/cointrader/docker-compose.yml`을 아래 내용으로 교체:

```yaml
services:
  cointrader:
    image: 10.1.10.28:3000/gihyeon/cointrader:latest
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

> **변경 사항:**
> - `build: .` → `image: 10.1.10.28:3000/gihyeon/cointrader:latest`
> - 이제 `docker compose up -d`를 실행하면 로컬 빌드 없이 레지스트리에서 이미지를 pull한다.
> - 배포 서버에서 최신 이미지로 업데이트하려면: `docker compose pull && docker compose up -d`

**Step 2: 변경 내용 확인**

```bash
cat /Users/gihyeon/github/cointrader/docker-compose.yml
```

Expected: `image:` 필드가 레지스트리 주소를 가리킴

**Step 3: (선택) 로컬 개발용 docker-compose.override.yml 생성**

로컬에서 소스 코드를 직접 빌드해서 테스트하고 싶을 때를 위한 override 파일:

```yaml
# docker-compose.override.yml (로컬 개발 전용, git에 포함하지 않아도 됨)
services:
  cointrader:
    build: .
    image: cointrader:local
```

이 파일이 있으면 `docker compose up -d`가 자동으로 `build: .`을 사용한다. 프로덕션 서버에는 이 파일을 두지 않는다.

---

## Task 7: 변경사항 커밋 및 Push

**Step 1: 변경 파일 확인**

```bash
cd /Users/gihyeon/github/cointrader
git status
```

Expected: `Jenkinsfile`(new), `docker-compose.yml`(modified)이 표시됨

**Step 2: 스테이징**

```bash
git add Jenkinsfile docker-compose.yml
```

**Step 3: `.env` 미포함 확인**

```bash
git diff --cached --name-only
```

Expected: `Jenkinsfile`, `docker-compose.yml` 두 파일만 표시됨

**Step 4: 커밋**

```bash
git commit -m "ci: Jenkins pipeline + Gitea registry CI/CD 설정"
```

Expected: `main` 브랜치에 새 커밋 생성

**Step 5: Gitea에 Push**

```bash
git push origin main
```

Expected: Push 성공 + (Webhook 설정 완료 시) Jenkins 빌드 자동 시작

---

## Task 8: 엔드-투-엔드 검증

**Step 1: 코드 변경 후 push 테스트**

```bash
cd /Users/gihyeon/github/cointrader
# 아무 파일이나 사소하게 변경 (예: README 한 줄 추가)
echo "# CI/CD test" >> README.md
git add README.md
git commit -m "test: CI/CD 파이프라인 검증용 더미 커밋"
git push origin main
```

**Step 2: Jenkins 빌드 자동 시작 확인**

Jenkins UI에서 `cointrader` 잡의 빌드가 자동으로 시작되는지 확인 (30초 이내)

**Step 3: Gitea 레지스트리에 이미지 push 확인**

`http://10.1.10.28:3000/gihyeon/cointrader/packages` 접속 → `cointrader` 컨테이너 패키지에 새 태그가 생성되었는지 확인

**Step 4: 이미지 pull 테스트**

```bash
docker pull 10.1.10.28:3000/gihyeon/cointrader:latest
```

Expected:
```
latest: Pulling from gihyeon/cointrader
...
Status: Downloaded newer image for 10.1.10.28:3000/gihyeon/cointrader:latest
```

**Step 5: docker compose로 실행 테스트**

```bash
cd /Users/gihyeon/github/cointrader
docker compose up -d
docker compose logs -f --tail=20
```

Expected: 컨테이너가 정상 시작되고 로그가 출력됨

---

## 트러블슈팅

| 문제 | 원인 | 해결 |
|------|------|------|
| `http: server gave HTTP response to HTTPS client` | Docker가 HTTPS로 레지스트리 접근 시도 | `daemon.json`에 `insecure-registries` 추가 후 Docker 재시작 |
| `unauthorized: authentication required` | Credentials 미등록 또는 토큰 만료 | Task 1 Step 2에서 새 토큰 발급 후 Jenkins Credentials 업데이트 |
| `connection refused` to Jenkins | Jenkins URL 오타 또는 방화벽 | Jenkins 서버 주소 재확인 |
| Webhook이 Jenkins를 트리거하지 않음 | Jenkins URL이 Gitea 서버에서 접근 불가 | Jenkins가 Gitea 서버와 같은 네트워크에 있는지 확인, 방화벽 8080 포트 오픈 |
| `image not found` on docker compose pull | 이미지가 아직 push되지 않음 | Jenkins 빌드 완료 후 재시도 |
| Jenkins에서 `docker: command not found` | Jenkins 에이전트에 Docker 미설치 | Jenkins 서버에 Docker 설치 또는 Docker-in-Docker 설정 |

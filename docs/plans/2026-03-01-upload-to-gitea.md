# Gitea 셀프호스팅 서버 업로드 구현 계획

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 현재 cointrader 프로젝트를 셀프호스팅 Gitea 서버(10.1.10.28:3000)에 업로드한다.

**Architecture:** 기존 로컬 git 저장소에 Gitea 원격 저장소를 추가하고, 미커밋 변경사항을 정리한 뒤 전체 히스토리를 push한다. `.env` 파일은 절대 포함하지 않으며, `.gitignore`가 올바르게 설정되어 있는지 확인 후 진행한다.

**Tech Stack:** git, Gitea REST API (또는 웹 UI), zsh

---

## 사전 확인 사항

### 현재 git 상태 요약

- **브랜치:** `main`
- **기존 커밋:** 10개 (b1a7632 ~ 117fd9e)
- **미커밋 변경사항 (modified):**
  - `.env.example`
  - `requirements.txt`
  - `src/bot.py`
  - `src/config.py`
  - `src/exchange.py`
- **삭제된 파일:** `src/database.py`
- **추적되지 않는 파일 (untracked):**
  - `docs/` (전체 디렉토리)
  - `src/notifier.py`
- **`.gitignore`에 의해 제외됨:** `.env`, `__pycache__/`, `*.pyc`, `.pytest_cache/`, `logs/`, `*.log`, `.venv/`

---

## Task 1: 미커밋 변경사항 스테이징 및 커밋

**Files:**
- Modify: `.env.example`
- Modify: `requirements.txt`
- Modify: `src/bot.py`
- Modify: `src/config.py`
- Modify: `src/exchange.py`
- Delete: `src/database.py`
- Create: `src/notifier.py`
- Create: `docs/` (전체)

**Step 1: git 상태 최종 확인**

```bash
git -C /Users/gihyeon/github/cointrader status
```

Expected: modified 파일들과 untracked 파일들이 목록에 표시됨

**Step 2: 모든 변경사항 스테이징**

```bash
cd /Users/gihyeon/github/cointrader
git add -A
```

> `-A` 옵션은 수정, 삭제, 신규 파일을 모두 스테이징한다. `.env`는 `.gitignore`에 있으므로 자동 제외된다.

**Step 3: 스테이징 내용 검토 (`.env` 포함 여부 반드시 확인)**

```bash
git diff --cached --name-only
```

Expected: `.env` 파일이 목록에 **없어야** 한다. 만약 있다면 즉시 `git reset HEAD .env` 실행 후 중단.

**Step 4: 커밋**

```bash
git commit -m "feat: Discord 알림, 포지션 복구, 설정 개선 및 docs 추가"
```

Expected: `main` 브랜치에 새 커밋 생성

**Step 5: 커밋 확인**

```bash
git log --oneline -3
```

Expected: 방금 만든 커밋이 최상단에 표시됨

---

## Task 2: Gitea에 원격 저장소 생성

**Step 1: Gitea 웹 UI 접속**

브라우저에서 `http://10.1.10.28:3000` 접속 후 로그인

**Step 2: 새 저장소 생성**

- 우상단 `+` 버튼 → `New Repository` 클릭
- **Repository Name:** `cointrader`
- **Visibility:** Private (권장) 또는 Public
- **Initialize this repository:** **체크 해제** (로컬에 이미 히스토리가 있으므로 빈 저장소로 생성해야 함)
- `Create Repository` 클릭

**Step 3: 저장소 URL 확인**

생성 후 표시되는 URL 메모:
```
http://10.1.10.28:3000/<사용자명>/cointrader.git
```

---

## Task 3: 로컬 저장소에 Gitea 원격 추가 및 Push

**Step 1: 현재 원격 저장소 확인**

```bash
cd /Users/gihyeon/github/cointrader
git remote -v
```

Expected: 아무것도 없거나 기존 origin이 있을 수 있음

**Step 2: Gitea 원격 추가**

기존 origin이 없는 경우:
```bash
git remote add origin http://10.1.10.28:3000/<사용자명>/cointrader.git
```

기존 origin이 있는 경우 (다른 URL):
```bash
git remote set-url origin http://10.1.10.28:3000/<사용자명>/cointrader.git
```

> `<사용자명>`은 Gitea 로그인 계정명으로 교체

**Step 3: 원격 추가 확인**

```bash
git remote -v
```

Expected:
```
origin  http://10.1.10.28:3000/<사용자명>/cointrader.git (fetch)
origin  http://10.1.10.28:3000/<사용자명>/cointrader.git (push)
```

**Step 4: main 브랜치 push**

```bash
git push -u origin main
```

> Gitea 계정의 사용자명과 비밀번호(또는 액세스 토큰)를 입력하라는 프롬프트가 나타남

Expected:
```
Enumerating objects: ...
Counting objects: ...
Writing objects: 100% ...
Branch 'main' set up to track remote branch 'main' from 'origin'.
```

**Step 5: Push 결과 확인**

```bash
git log --oneline origin/main
```

Expected: 로컬 커밋 히스토리와 동일하게 표시됨

---

## Task 4: Gitea 웹 UI에서 업로드 검증

**Step 1: 브라우저에서 저장소 확인**

`http://10.1.10.28:3000/<사용자명>/cointrader` 접속

**Step 2: 파일 목록 확인**

다음 파일/폴더가 있어야 함:
- `src/` (bot.py, config.py, exchange.py, notifier.py, indicators.py, risk_manager.py, logger_setup.py, data_stream.py, config.py 등)
- `tests/`
- `docs/`
- `main.py`
- `requirements.txt`
- `.env.example`
- `.gitignore`

다음 파일이 **없어야** 함:
- `.env`
- `__pycache__/`
- `.venv/`
- `logs/`

**Step 3: 커밋 히스토리 확인**

Gitea UI에서 `Commits` 탭 클릭 → 11개 커밋이 모두 표시되는지 확인

---

## 선택 사항: SSH 키 설정 (비밀번호 없이 push하려면)

매번 비밀번호 입력이 번거롭다면 SSH 키를 등록할 수 있다.

**Step 1: SSH 키 생성 (없는 경우)**

```bash
ssh-keygen -t ed25519 -C "cointrader@gitea" -f ~/.ssh/id_gitea
```

**Step 2: 공개 키 복사**

```bash
cat ~/.ssh/id_gitea.pub
```

**Step 3: Gitea에 SSH 키 등록**

Gitea 웹 UI → 우상단 프로필 → `Settings` → `SSH / GPG Keys` → `Add Key` → 공개 키 붙여넣기

**Step 4: SSH config 설정**

```bash
cat >> ~/.ssh/config << 'EOF'
Host gitea-local
    HostName 10.1.10.28
    Port 22
    User git
    IdentityFile ~/.ssh/id_gitea
EOF
```

**Step 5: 원격 URL을 SSH로 변경**

```bash
git remote set-url origin git@gitea-local:<사용자명>/cointrader.git
```

---

## 트러블슈팅

| 문제 | 원인 | 해결 |
|------|------|------|
| `Connection refused` | Gitea 서버 미실행 또는 방화벽 | `http://10.1.10.28:3000` 접속 가능한지 브라우저로 먼저 확인 |
| `Repository not found` | 저장소 미생성 또는 URL 오타 | Task 2 재확인, URL의 사용자명 확인 |
| `Authentication failed` | 잘못된 계정 정보 | Gitea 웹 UI 로그인 테스트 후 동일 계정 사용 |
| `non-fast-forward` | 원격에 이미 커밋 존재 | `git push --force origin main` (단, 원격 데이터 덮어씌워짐 주의) |
| `.env` 파일이 push됨 | `.gitignore` 미적용 | `git rm --cached .env && git commit -m "chore: remove .env from tracking"` |

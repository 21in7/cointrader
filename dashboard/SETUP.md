# Trading Dashboard

봇과 통합된 대시보드. 봇 로그를 읽기 전용으로 마운트하여 실시간 시각화합니다.

## 구조

```
dashboard/
├── SETUP.md
├── api/
│   ├── Dockerfile
│   ├── log_parser.py
│   ├── dashboard_api.py
│   └── entrypoint.sh
└── ui/
    ├── Dockerfile
    ├── nginx.conf
    ├── package.json
    ├── vite.config.js
    ├── index.html
    └── src/
        ├── main.jsx
        └── App.jsx
```

## 실행

루트 디렉토리에서 봇과 함께 실행:

```bash
# 전체 (봇 + 대시보드)
docker compose up -d --build

# 대시보드만
docker compose up -d --build dashboard-api dashboard-ui
```

## 접속

`http://<서버IP>:8080`

## 동작 방식

- `dashboard-api`: 로그 파서 + FastAPI 서버 (봇 로그 → SQLite → REST API)
- `dashboard-ui`: React + Vite (빌드 후 nginx에서 서빙, API 프록시)
- 봇 로그 디렉토리를 `:ro` (읽기 전용) 마운트
- 대시보드 DB는 Docker named volume (`dashboard-data`)에 저장

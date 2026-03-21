# Dashboard Code Review R2

**날짜**: 2026-03-21
**상태**: Completed
**커밋**: e362329

## 원본 리뷰 (23건) → 재평가 결과

원래 23건의 코드 리뷰 항목 중 15건이 과잉 지적으로 판단되어 삭제, 5건은 Low로 하향, 3건만 Medium 유지. 이후 내부망 전용 환경 감안하여 #3 Health 503도 Low로 하향.

## 삭제 (15건)

| # | 이슈 | 삭제 사유 |
|---|------|----------|
| 1 | SQL Injection — reset_db | 테이블명 하드코딩 리스트, 코드 명백 |
| 4 | CORS wildcard + CSRF | X-API-Key 커스텀 헤더가 preflight 강제, CSRF 벡터 없음 |
| 5 | get_symbols 쿼리 비효율 | bot_status 수십 건, 최적화 불필요 |
| 6 | Signal handler sys.exit() | _shutdown=True → commit → close → exit 순서 이미 방어적 |
| 7 | SIGHUP DB 미초기화 | reset_db API가 5개 테이블 DELETE 후 SIGHUP, 설계상 올바름 |
| 8 | stale trade 삭제 f-string | parameterized query 패턴, 안전 |
| 9 | 파싱 순서 의존성 | 각 패턴이 서로 다른 한국어 키워드, 충돌 불가 |
| 10 | PID 파일 경쟁 조건 | Docker 단일 인스턴스 |
| 12 | 인라인 스타일 | S 객체로 변수화, 이 규모에서 CSS framework은 오버엔지니어링 |
| 15 | fmtTime 라벨 충돌 | 96캔들=24시간, 날짜 겹칠 일 없음 |
| 16 | prompt()/confirm() | 관리자 전용 드문 기능, 네이티브 dialog 적절 |
| 17 | 함수형 dataKey | 차트 2개 기준선, 성능 영향 0 |
| 20 | API Dockerfile requirements 미분리 | 패키지 2개, requirements.txt 오버헤드만 추가 |
| 21 | Nginx resolver 하드코딩 | Docker-only 환경 |
| 23 | private 메서드 직접 테스트 | 파서 핵심 로직 검증, 자주 리팩토링될 구조 아님 |

## Low로 하향 (5건, 수정 미진행)

| # | 이슈 | 하향 사유 |
|---|------|----------|
| 2 | DB PRAGMA 반복 | SQLite connect()는 파일 open 수준, 15초 폴링에서 병목 아님 |
| 11 | App.jsx 모놀리식 | 737줄이나 컴포넌트 파일 내 잘 분리, 추가 기능 계획 없으면 YAGNI |
| 13 | 부분 API 실패 | 이전 값 유지가 전체 초기화보다 나은 동작, 15초 후 자동 복구 |
| 18 | pos.id undefined | _handle_entry 중복 체크로 발생 확률 극히 낮음 |
| 22 | 테스트 환경변수 오염 | dashboard_api import하는 테스트 파일 하나뿐 |

## Low로 하향 (내부망 감안)

| # | 이슈 | 하향 사유 |
|---|------|----------|
| 3 | Health 에러 시 200→503 | 내부망 전용, 로드밸런서 health check 시나리오 없음 |

## 수정 완료 (2건)

### #14 Trades 페이지네이션 (Medium)

**문제**: API가 offset 파라미터를 지원하는데 프론트엔드에서 항상 `limit=50&offset=0`만 호출. tradesTotal > 50이면 나머지를 볼 수 없음.

**수정** (`dashboard/ui/src/App.jsx`):
- `tradesPage` state 추가
- fetchAll API 호출에 `offset=${tradesPage * 50}` 반영
- `useCallback` dependency에 `tradesPage` 추가
- 심볼 변경 시 `setTradesPage(0)` 리셋
- Trades 탭 하단에 이전/다음 페이지네이션 컨트롤 추가 (범위 표시: `1–50 / 총건수`)

### #19 package-lock.json + npm ci (Medium)

**문제**: `dashboard/ui/Dockerfile`에서 `COPY package.json .` + `npm install`만 사용. package-lock.json이 존재하는데 활용하지 않아 빌드 재현성 미보장.

**수정** (`dashboard/ui/Dockerfile`):
- `COPY package.json package-lock.json .`
- `RUN npm ci`

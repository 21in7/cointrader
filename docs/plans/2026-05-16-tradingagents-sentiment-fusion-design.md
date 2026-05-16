# TradingAgents Sentiment Fusion — 설계 문서

> 작성일: 2026-05-16 · 상태: **설계 완료, 백테스트 대기** · 유형: design

## 1. 배경 & 동기

`2026-05-04 strategy-post-mortem` 결론: 7전 7패의 공통 원인은 추론 능력 부족이
아니라 **공개 *가격 파생* 시그널(RSI/MACD/BB/EMA/StochRSI/OI/펀딩/LS비율)만으로
방향을 예측하는 패러다임의 한계**. 같은 정보원에 다른 알고리즘을 얹는 시도는 모두
실패했다.

본 설계는 [TradingAgents](https://github.com/TauricResearch/TradingAgents)의
멀티에이전트 프레임워크 중 **Sentiment Analyst 1개 모듈만 추출**하여, *가격이 아닌
다른 정보 모달리티(군중/뉴스 텍스트 센티먼트)*를 일반 봇의 진입 게이트로 도입한다.
post-mortem 정합상 **순응 신호가 아니라 contrarian/veto 필터**로만 사용한다(아래
§3.2 가설 참조).

전체 TradingAgents 그래프(LangGraph 12~16 LLM 콜, 결정당 2~5분)는 15m 봉에
부적합하므로 **융합하지 않는다**. 추출 대상은 그래프 의존이 없는 단일 분석기뿐이다.

## 2. 가설 (strategy-research Step 1)

```
가설: 군중/뉴스 센티먼트의 극단값이 일반 봇 진입 신호의 수익성을 높인다
      (순응이 아닌 contrarian/veto 게이트로 적용 시)

검증 기준:
- OOS PF > 1.0 (최소), > 1.2 (목표) — 베이스라인(센티먼트 OFF) 대비 개선
- 거래 수 > 50 (게이트 적용 후에도)
- IS/OOS PF 격차 < 30%
- LONG/SHORT 대칭성 유지 (한쪽만 수익이면 폐기)
측정: Walk-forward 백테스트, 베이스라인 대비 ablation

폐기 기준 (하나라도 해당 시 즉시 폐기):
- OOS PF < 1.0  또는  베이스라인 대비 PF 악화
- LONG/SHORT 대칭성 실패
- 게이트 적용 후 거래 수 < 30
```

베이스라인 = 현 XRP 단독 봇 설정(`NO_ML_FILTER=true`, ML OFF). 센티먼트 게이트가
**베이스라인을 능가하지 못하면 8번째 실패로 간주하고 폐기**한다.

## 3. 아키텍처

### 3.1 TradingAgents 추출 범위

| 추출 | 파일 (TradingAgents) | 비고 |
|------|----------------------|------|
| Sentiment Analyst closure | `agents/analysts/sentiment_analyst.py` | 데이터 self-prefetch, **LLM 1콜**, LangGraph 무의존 |
| StockTwits fetcher | `dataflows/stocktwits.py` | API키 불필요, `$BTC.X` 캐시태그 동작 (라이브 전용) |
| Reddit fetcher | `dataflows/reddit.py` | API키 불필요, subreddit 인자 → `CryptoCurrency`,`Bitcoin` (라이브 전용) |
| Alpha Vantage NEWS_SENTIMENT | `dataflows/alpha_vantage_news.py` | **크립토 + 과거 시계열** → 백테스트 데이터원 |
| 5등급 파서 | `graph/signal_processing.py` 정규식 | LLM 무관, 결정론 |

**버린다**: 전체 그래프, bull/bear 디베이트, reflection, trader/risk/portfolio,
yfinance-주식 레이어.

추출 산출물은 신규 파일 `src/sentiment_provider.py` 한 곳에 벤더링·트림한다.
**의존성 0**: 프로젝트 `.venv`에 langchain/openai 미설치 확인(2026-05-16) →
LLM-free 봇에 거대 의존성 추가는 과도. MLX 서버가 OpenAI 호환이므로 이미 설치된
`httpx`로 `/v1/chat/completions` 직접 호출, fetcher는 stdlib `urllib`. TradingAgents의
가치(프롬프트·3소스 패턴·graceful fetcher·등급 체계)만 추출하고 LangChain 플러밍은
제거. 분석기 출력도 prose 리포트 대신 **엄격 JSON**(5등급 label + confidence +
rationale)으로 강제 → 정규식 파싱 단계 제거, 결정론 강화.

### 3.2 적용 모드 (post-mortem 정합)

`SENTIMENT_MODE` 환경변수로 선택:
- `veto`   : 신호 방향과 센티먼트가 정면 충돌하면 진입 거부 (보수적 기본값)
- `contrarian` : 센티먼트 **극단**(VeryBullish/VeryBearish)일 때 그 반대 방향만 허용
- `confirm`(비권장) : 동일 방향 확인 — post-mortem상 실패 가능성 높음, ablation 비교용으로만

순응(`confirm`)을 기본값으로 두지 않는다. 군중 극단의 페이드가 유일하게
post-mortem과 정합하는 가설이다.

### 3.3 일반 봇 게이트 융합 (`src/bot.py`)

핵심 발견: **일반 봇엔 `(방향, 컨텍스트) → 허용/거부` veto 슬롯이 이미 존재하고
현재 비어 있다**(`NO_ML_FILTER=true`로 ML OFF, `src/ml_filter.py:38`).
`SentimentFilter`는 `MLFilter`의 1:1 드롭인 형제로 추가한다.

| 항목 | 기존 ML 선례 (위치) | 센티먼트 적용 |
|------|---------------------|----------------|
| graceful 미로드 | `MLFilter.is_model_loaded()` `src/ml_filter.py:94` | `SentimentFilter.is_ready()` — stale/미가동 시 통과 |
| 진입 게이트 (신규) | `bot.py:426-429` `if ml_filter.is_model_loaded(): if not should_enter(): return` | 동일 위치에 `sentiment_filter` 게이트 추가 |
| 진입 게이트 (역신호 재진입) | `bot.py:840-851` `_close_and_reenter` 내 동일 패턴 | 동일하게 추가 (일관성) |
| hot-reload | `ml_filter.check_and_reload()` mtime `src/bot.py:362`, `src/ml_filter.py:105` | 스코어 파일 mtime 폴링 (사이드카가 write) |
| 비활성 플래그 | `NO_ML_FILTER` env | `SENTIMENT_ENABLED`(기본 false, 동일 철학) |

### 3.4 사이드카 (`src/sentiment_stream.py`)

선례: `UserDataStream`이 `run()`에서 콜백과 함께 생성(`bot.py:878-881`)되어
`self.stream.start()`, `self._position_monitor()`와 같은
`asyncio.gather()`(`bot.py:883-895`)에 묶임. `SentimentStream`은 그 gather()의
4번째 코루틴으로 들어간다.

```
SentimentStream (N=1~4h 주기, SENTIMENT_REFRESH_SEC)
  reddit.fetch(coin,["CryptoCurrency","Bitcoin"]) + stocktwits.fetch("BTC.X")
   └ (옵션) jina-reranker @ :8787 로 헤드라인 관련도 1차 필터
  → sentiment_provider 분석기 (로컬 MLX 1콜, temp=0)
  → 5등급 → score ∈ [-1,+1] + ts  →  data/sentiment/{symbol}.json (atomic write)
        │                                  ▲ mtime hot-reload
        ▼
bot.process_candle(): get_signal → sentiment_filter 게이트 → _open_position
```

군중심리는 시간 단위로 느리게 변하므로 봉(15m)마다 추론하지 않는다.

## 4. 로컬 추론 (MLX — 실측 확인 완료)

2026-05-16 실측:
- 가동 중: `mlx_vlm.server` (mlx-vlm 0.4.4, mlx-lm 0.31.3도 동일 venv
  `/Users/gihyeon/Model/mlx/.venv`), 모델 `mlx-community/gemma-4-e4b-it-4bit`,
  엔드포인트 `http://localhost:8080/v1` (OpenAI 호환, `0.0.0.0` 바인딩, 키 불필요),
  peak 6.5GB. 상위 fallback `gemma-4-26b-a4b-it-4bit` 캐시 보유.
  보너스: `jina-reranker-v3-mlx` @ `:8787` (뉴스 관련도 리랭킹용).
- 감성분류 기능 테스트 통과: enum 5등급 단일어 출력 제약 정확히 준수.
- **결정론 실측 확인**: `temperature=0` 동일 프롬프트 2회 → 비트단위 동일 출력.
  → "TradingAgents 비결정성 → validator 백테스트 재현 불가"라는 최대 검증
  블로커가 이 환경에서 **해소됨** (이론 아닌 실측).

연결: `langchain-openai.ChatOpenAI(base_url="http://localhost:8080/v1",
api_key="x", model="mlx-community/gemma-4-e4b-it-4bit", temperature=0)`.
TradingAgents `llm_clients/factory.py:41-43`(ollama 스타일 OpenAI 호환 경로,
`openai_client.py:154`)와 동형 — 단 분석기 closure만 추출하므로 langchain-openai
직접 사용이 더 간단.

배포: 서버는 Mac. 프로덕션 LXC 봇은 **train-on-Mac→LXC 기존 패턴 재사용** —
사이드카·추론은 Mac에서, 스코어 JSON만 LXC가 mtime hot-reload(`check_and_reload`
동형). LAN 직결(0.0.0.0:8080)은 대안이나 봇-인프라 결합도 증가로 비권장.

## 5. 데이터 전략 (핵심 제약)

| 용도 | 데이터원 | 가용성 |
|------|----------|--------|
| 라이브 사이드카 | Reddit/StockTwits 공개 EP | **현재 시점만** — 과거 스냅샷 없음 |
| **백테스트** | Alpha Vantage `NEWS_SENTIMENT` (크립토, 과거 시계열) | 키 필요, time-range 쿼리 |

**미해결 블로커**: 라이브에서 쓸 Reddit/StockTwits는 과거 데이터가 없어
백테스트 불가. 백테스트는 Alpha Vantage 과거 뉴스 센티먼트로 *대리(proxy)*
검증한다. 즉 **백테스트 신호원 ≠ 라이브 신호원**. 이 modality gap은 본 전략의
최대 리스크이며, 백테스트 PASS 후에도 라이브 섀도우 검증을 의무화하는 이유다
(§7 게이트 C). Alpha Vantage edge가 borderline이면, 라이브 사이드카로 Reddit/
StockTwits 스코어를 무거래 적립(shadow logging)하여 자체 데이터셋 구축 후 재검증.

로컬 추론이 무료·결정론이므로 과거 텍스트 대량 일괄 추론이 비로소 현실적
(API였다면 비용·레이트리밋으로 불가).

## 6. 백테스트 설계 (strategy-research Step 3~4)

### 6.1 데이터

- 심볼: XRPUSDT (현 운영 단독 심볼). 데이터: `data/xrpusdt/combined_15m.parquet`
  (6.6M, ~2026-05-16). BTC/ETH 상관 컬럼 임베디드.
- 센티먼트: `scripts/build_sentiment_dataset.py`(신규) — Alpha Vantage
  NEWS_SENTIMENT를 심볼·기간별로 수집 → 로컬 MLX 일괄 추론(전용 인스턴스,
  라이브 :8080 비점유) → `data/xrpusdt/sentiment_15m.parquet`
  (15m 타임스탬프 정렬, **look-ahead 금지**: 각 봉 t의 스코어는 t 시점까지
  공개된 뉴스만 사용 — published_at ≤ candle_open).

### 6.2 백테스트 엔진 주입 지점

기존 `src/backtester.py` 인프라 재사용. ML 확률 게이트 선례
`_get_ml_proba()`(`backtester.py:252`)와 동형으로 센티먼트 게이트 추가:

- 진입 chokepoint: `_try_enter(sym, signal, df_ind, candle_idx, features, ts)`
  — 호출처 `backtester.py:389`(역신호 재진입), `:398`(신규 진입).
- 신규 `_get_sentiment_score(ts, sym) -> float | None`: 사전 적재
  `sentiment_15m.parquet`을 ts로 lookup (LLM 호출 없음, 결정론·고속).
- `_try_enter` 진입 직전 게이트: `SENTIMENT_MODE`에 따라 veto/contrarian 판정.
  센티먼트 결측/stale → None → 게이트 통과(graceful, 베이스라인과 동일 거동).
- `BacktestConfig`에 `sentiment_mode`, `sentiment_threshold`,
  `sentiment_extreme_band` 추가. `scripts/run_backtest.py` argparse에
  `--sentiment-mode`, `--sentiment-threshold` 추가 (기존 `--no-ml` 패턴 모방).

### 6.3 검증 프로토콜

1. **베이스라인**: `run_backtest.py --symbol XRPUSDT --walk-forward
   --no-ml`(센티먼트 OFF) — 현 운영 설정 PF 기록.
2. **ablation**: 동일 분할에 `--sentiment-mode {veto,contrarian}` 각각 →
   베이스라인 대비 ΔPF, Δ거래수, 방향별 PF.
3. Walk-forward: `--train-months 6 --test-months 1` (기존 관례).
   ※ 센티먼트 게이트는 학습 파라미터가 없으므로 WF는 *시기 강건성* 확인 용도
   (regime별 일관성). IS/OOS 격차는 데이터-스누핑(임계값 튜닝) 방지로 측정.
4. **결정**: §2 폐기 기준 적용. 결과를
   `docs/plans/2026-05-16-tradingagents-sentiment-fusion-result.md`에 기록.

### 6.4 결과 메트릭 (strategy-research Step 4 표 준수)

| 메트릭 | 최소 | 목표 | 비고 |
|--------|------|------|------|
| Profit Factor | >1.0 | >1.2 | IS·OOS 모두, 베이스라인 대비 개선 필수 |
| 승률 | >40% | >50% | LONG/SHORT 분리 |
| 거래 수 | >50 | >100 | 게이트 적용 후 |
| 최대 낙폭 | <20% | <10% | |
| IS/OOS 격차 | <30% | <15% | 임계값 스누핑 지표 |

## 7. 단계별 의사결정 게이트 (각 게이트 통과 시에만 다음 진행)

- **게이트 A — 데이터·추출 검증**
  - `src/sentiment_provider.py` 추출 + 로컬 MLX 연결, 결정론 회귀 테스트(temp=0
    동일 입력 → 동일 스코어) green.
  - `build_sentiment_dataset.py`로 XRP 과거 센티먼트 parquet 생성, look-ahead
    무결성 검증(published_at ≤ candle_open) 통과.
- **게이트 B — 오프라인 edge**
  - §6 백테스트. §2 기준 PASS 시에만 라이브 통합 진행. FAIL 시 `-result.md`에
    폐기 기록 후 종료.
- **게이트 C — 라이브 섀도우**
  - `SentimentStream` + `SentimentFilter`를 **무거래 로깅 모드**로 프로덕션
    병행 가동(게이트 판정만 기록, 주문 영향 없음). Reddit/StockTwits 실신호원
    기준 수 주 수집 → 백테스트(Alpha Vantage proxy)와 거동 일치 확인.
- **게이트 D — 실거래 활성화**
  - 게이트 C 일치 확인 후 `SENTIMENT_ENABLED=true`. 킬스위치·리스크 한도는
    기존 그대로 적용.

## 8. 구현 단계 (게이트 B PASS 후 착수, 본 문서는 설계만)

1. `src/sentiment_provider.py` — TradingAgents Sentiment Analyst + reddit/
   stocktwits/alpha_vantage fetcher 벤더링·트림, 크립토 subreddit/캐시태그,
   로컬 MLX(`temperature=0`) 바인딩, 응답 캐시 `data/sentiment_cache/`.
2. `scripts/build_sentiment_dataset.py` — Alpha Vantage 과거 수집 + MLX 일괄
   추론 + 15m 정렬 + look-ahead 가드 → `data/{symbol}/sentiment_15m.parquet`.
3. `src/backtester.py` — `_get_sentiment_score()`, `_try_enter` 게이트,
   `BacktestConfig` 필드. `scripts/run_backtest.py` argparse 확장.
4. (게이트 B PASS 후) `src/sentiment_filter.py` — `MLFilter` 형제 게이트.
5. `src/config.py` — `SENTIMENT_*` env (`SymbolStrategyParams` `config.py:9`
   확장 + per-symbol override 패턴 `config.py:91-99`).
6. `src/sentiment_stream.py` — 비동기 사이드카, `bot.run()` gather() 합류.
7. `src/bot.py` — `bot.py:426-429`, `:840-851` 두 게이트 site에 sentiment 추가.
8. 테스트: 결정론 회귀, graceful 폴백(stale/미가동→통과), look-ahead 가드,
   게이트 모드별 단위 테스트. README/ARCHITECTURE 동기화.

## 9. 리스크 & 미해결 질문

- **R1 (최대)**: 백테스트 신호원(Alpha Vantage 뉴스) ≠ 라이브 신호원(Reddit/
  StockTwits 군중). modality gap → 게이트 C 섀도우 검증 의무.
- **R2**: 센티먼트도 *공개* 신호. post-mortem 패러다임 한계 재현 위험 →
  contrarian/veto만 허용, confirm 기본 금지로 완화.
- **R3**: gemma-4-e4b(4B) 미묘 뉴스 해석 한계. borderline 시 26b-a4b ablation
  1회 비교. 단 로컬 edge 부재 시 신호 자체 한계일 개연 큼.
- **R4**: 백테스트 대량 추론이 라이브 :8080 점유 → 전용 MLX 인스턴스 분리.
- **R5**: Alpha Vantage 무료 티어 레이트리밋 → 수집 스크립트 백오프·재개 지원.
- **Q1**: 센티먼트 스코어를 게이트(bool)로만 쓸지, `build_features_aligned`의
  추가 피처로도 넣을지 — 게이트 B에서 게이트-only 우선 검증, 피처화는 후속.
- **Q2**: 멀티심볼 확장 시 심볼별 센티먼트 분리 필요 — 현재 XRP 단독이므로 보류.

## 10. 폐기된 전략 참조 (반복 방지)

LS Ratio / FR+OI 단독 / Binance 공개 API 전수 / ML Filter / MTF Pullback —
모두 *가격 파생 공개 신호*. 본 전략은 *비가격 텍스트 모달리티*라는 점에서만
차별화되며, 그 차별성이 edge로 입증되지 않으면 위 목록에 추가하고 폐기한다.

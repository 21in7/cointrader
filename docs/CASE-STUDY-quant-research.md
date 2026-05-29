# 사례 연구 — 반증 우선(Falsification-First) 정량 트레이딩 리서치

> 한 줄 요약: **13개 가설을 사전 폐기기준 + 데이터-우선 검증으로 체계적으로
> 반증/비배포 판정하고, 자본 손실 $0으로 "리테일·공개데이터·암호화폐에는
> 배포 가능한 알파가 없다"를 빠르고 싸게 입증한 정량 리서치 프로그램.**
> 산출물은 알파가 아니라 *재사용 가능한 리서치 규율 + 실행 인프라*.

이 문서는 "13번 실패했다"가 아니라 **"나쁜 아이디어를 자본을 걸기 전에
싸게 죽이는 방법론을 가지고 있고, 그 증거가 13건 있다"**를 보여주기 위한
포트폴리오 자산이다. 최신 웨이브(11–13)는 *백테스트를 짜기도 전에* 구조적
속성(공적분·펀딩경제성·예측edge-vs-비용·walk-forward 생존)으로 PASS/FAIL을
가르는 **데이터-only precheck 게이트**와, *내 자신의 분석*의 버그를 커밋 전에
잡은 **적대적 검증 패스**를 추가했다. 음성 결과를 규율 있게 다룬 증거는
대부분의 지원자가 보여주지 못한다.

---

## 1. 방법론 (이게 진짜 자산)

모든 가설에 동일한 루프를 강제했다:

1. **가설을 구조화** — 시그널/메커니즘을 측정 가능한 명제로.
2. **사전 폐기기준을 *실행 전에* 못박음** — PF/Sharpe/N/대칭성/연도일관성
   임계를 결정 후 변경 금지. 결과를 보고 기준을 movement 하지 않음.
3. **가장 싼 검증부터** — 자본·실거래 전에 데이터-only 백테스트. 데이터
   확보 가능성 자체를 게이트로(불가하면 그 자리에서 폐기).
4. **정직한 판정** — 스크립트는 숫자만, 판정은 사전기준 대조로 사람이.
   음성 결과를 1급 산출물로 문서화(`docs/plans/*-result.md`).
5. **패러다임 단위 에스컬레이션** — 파라미터 만지작거리기 금지. 한
   패러다임이 죽으면 *카테고리*를 바꿈(방향예측 → 구조캐리 → 이벤트 →
   비-알파). 같은 루프의 N회차를 돌지 않음.

핵심: **검증의 비용을 결과의 확신도보다 먼저 최소화**한다. 13건 모두
$0 / 수 시간~1일 내에 판정했다. 2차 웨이브(11–13)는 "가장 싼 검증"을 한
단계 더 당겼다 — 백테스트조차 짜기 전에 *구조적 전제*를 데이터-only로
PASS/FAIL(precheck 게이트). 전제가 깨지면 백테스트를 작성하지 않는다.

---

## 2. 13개 실험 (전부 `docs/plans/`에 설계+결과 문서 존재)

| # | 가설 | 패러다임 | 반증 이유 | 일반화 교훈 |
|---|------|----------|-----------|-------------|
| 1 | LS Ratio 시그널 | 방향예측 | edge 없음 확정 | 공개 포지셔닝 비차별 |
| 2 | Funding+OI 단독 | 방향예측 | SHORT만 +, LONG/SHORT 비대칭 | 한쪽만 수익=과적합 신호 |
| 3 | Binance 공개 API 전수 | 방향예측 | 단독 edge 전무 | 공개데이터=경쟁 차익소거 |
| 4 | ML 필터(LGBM/ONNX) | ML | ML OFF > ML ON | 알파 없는 피처엔 ML 무의미 |
| 5 | MTF Pullback Bot | 방향예측 | OOS 실패 | IS 과적합, OOS가 진실 |
| 6 | MTF OOS 재검 | 방향예측 | fees_only PF 0.84, 대칭성 실패 | 수수료 후가 진짜 |
| 7 | MTF + BTC 필터 | 방향예측 | 필터 추가가 베이스보다 악화 | 필터는 음EV를 양으로 못 바꿈 |
| 8 | TradingAgents 센티먼트 융합 | LLM/센티먼트 | N=19 표본부족·뉴스-only modality gap | 백테스트≠라이브 신호원이면 검증 무력 |
| 9 | 델타중립 펀딩 캐리 | 구조적·시장중립 | 이상화 gross +4.6%/yr이나 net −0.37%, 실은 레짐의존 강세프리미엄 | gross≠net; "구조적" 주장은 레짐분해로 검증 |
| 10 | 신규상장 마이크로구조 | 비대칭/이벤트 | N=323서 net 음, 부분데이터 +275bps 힌트가 전체N서 소멸 | 저N·체리픽·생존편향을 규율이 잡아냄 |
| 11 | 통계적 차익(공적분) | 시장중립 | 알트 공적분 전무(EG/Johansen 일치, half-life 랜덤워크), 크로스알트 0/15 | co-trending≠공적분; precheck가 백테스트 전 차단 |
| 12 | 리드-래그(BTC/ETH→알트, 15m) | 방향/마이크로구조 | 예측 edge 0.2~2.1bps ≪ 15bps 비용 (TRX는 통계적 진짜나 거래불가) | 통계적 유의≠경제적 유의; 거래가능분은 차익소거됨 |
| 13 | 저빈도 모멘텀(TSMOM/XSMOM) | 방향/추세 | 이식가능 edge 없음: 2/56 통과는 ETH한정·robust 단위 BH 탈락; crisis-alpha는 크래시 형태의존(LUNA 포착/FTX 7-8 휩쏘)이지 일반성질 아님 | 단일자산 통과=false-positive 함정; 다자산+적대검증 필수 |

(1–7은 `2026-05-04-strategy-post-mortem.md`에 집약, 8–10은 직전 사이클,
11–13은 데이터-only precheck **2차 웨이브**. #9(펀딩 캐리)는 이번에 BTC/ETH로
재확인됨(carry ≈ 자금조달비, basis 실측까지). #13은 walk-forward + 8자산
crisis-alpha 일반성 검증 + 적대적 검증까지 추적. 각 건 `*-design.md`/`*-result.md` 추적 가능.)

---

## 3. 메타 교훈 — 시니어 판단의 증거

각 항목은 면접/리뷰에서 *판단력*을 보여주는 구체 사례다:

- **표본 수 규율**: #8에서 단일심볼 entry-filter가 25개월 ≤22거래(N=19)임을
  발견 → "통계적으로 검증 불가"를 결과로 인정(억지 결론 금지). #10에선
  반대로 N=323 이벤트스터디를 설계해 검정력 확보.
- **과적합을 실시간으로 포착**(최고 증거): #10 부분데이터(95건)에서
  +275bps "힌트"가 보였으나 *흥분 대신* "단일셀 집중·저N 아티팩트"로
  의심 → 전체 N=323에서 −82bps로 소멸. 사전기준+전체표본 규율이 비싼
  실거래 실패를 차단.
- **생존편향을 방향까지 추론**: #10 데이터가 *현재 거래중* 심볼만 →
  편향이 엣지를 *과대* 방향임을 명시, "그조차 음이면 확정 폐기" 논리.
- **gross≠net**: #9에서 이상화 gross +4.6%/yr이 spot 자금조달비(~5%/yr)를
  못 넘음을 비용 매트릭스로 분해.
- **숨은 노출 적발**: #9 "시장중립" 주장을 레짐 분해로 깨고 *강세프리미엄
  방향베팅*임을 폭로.
- **재현성 엔지니어링**: #8에서 비결정적 LLM을 `temperature=0` + 응답
  해시 캐시로 *비트단위 재현 가능*하게 만들어 백테스트 검증 가능성 확보
  (의존성 0, langchain 없이 httpx 직접).
- **음성 결과의 1급 취급**: 13개 `*-result.md` + plan history + 메모리에
  "왜 죽었나 + 재시도 금지 조건"을 영구 기록 → 미래의 자신이 같은 루프
  재실행 방지.
- **백테스트보다 더 싼 킬(precheck 게이트)**: #11–13에서 자본·백테스트
  *전에* 구조적 전제(공적분이 존재하나? 펀딩이 비용을 넘나? 예측 edge가
  비용을 넘나? walk-forward에서 살아남나?)를 데이터-only로 PASS/FAIL.
  전제가 깨지면 백테스트 자체를 작성하지 않음 — "가장 싼 검증"을 한 단계
  더 당김. (예: #11 공적분 부재 → 스프레드 백테스트 미작성.)
- **적대적 검증이 *내* 버그를 잡음**(최고 증거): #13 crisis-alpha 다자산
  결과를 커밋 *전* 적대적 검증(구현/통계/프레이밍 3렌즈 + 종합)에 걸어
  EW 포트폴리오 inner-join 버그를 발견 → 잘못된 주장("EW가 buy&hold에 열위,
  크래시 −7%")을 정정(outer-join 시 Sharpe 0.58>0.45). 핵심 판정(REJECTED)이
  모든 임계서 강건함도 독립 재현. 자기 결과를 의심하는 규율이 false-positive를
  자기 손으로 차단.
- **단일자산 통과의 함정 적발**: #13 ETH 모멘텀 단독 "통과"(2/56)를 다자산
  walk-forward로 *ETH-specific*임을 폭로(robust 단위 BH 탈락, crisis-alpha
  형태의존). 사전지정 "majors+포트폴리오" 헤드라인이 cherry-pick을 차단.

---

## 4. 엔지니어링 산출물 (재사용 가능 인프라)

- **async 멀티심볼 실행**: `src/bot.py` — 심볼별 독립 봇 `asyncio.gather`,
  공유 `RiskManager` 싱글톤(글로벌 손실한도·동일방향 제한, `asyncio.Lock`).
- **이중 킬스위치**: Fast(8연속손실)/Slow(15거래 PF<0.75), JSONL 영속·부팅
  소급검증 — 음EV 시스템이 *천천히 죽지 않게* 하는 안전장치.
- **벡터화 백테스터**: `src/backtester.py` — walk-forward, 슬리피지/수수료
  모델, 결정론적, 커스텀 이벤트스터디 확장(`funding_carry_backtest.py`,
  `listing_microstructure_backtest.py`).
- **결정론적 로컬 LLM 통합**: `src/sentiment_provider.py` — OpenAI 호환
  로컬 MLX 직접 호출, 응답 캐시로 재현성, graceful degradation.
- **데이터 파이프라인**: Binance fapi 수집·캐시·idempotent 재개
  (`collect_listings.py`, `build_sentiment_dataset.py`).
- **테스트 규율**: 게이트/프로바이더 회귀 테스트(`tests/test_sentiment_*`),
  baseline no-op 보장 검증, look-ahead 가드 단위테스트.

---

## 5. 정직한 결론

리테일·공개데이터·암호화폐 선물에서 방향예측·ML·LLM·구조캐리·통계적차익·
리드래그·이벤트비대칭·저빈도모멘텀 **모든 패러다임이 비용·편향 직시 후
배포 가능한 net 엣지 없음**으로 반증/비배포 판정됐다. 가장 가까이 간 모멘텀
조차 "방향성 알파"가 아니라 *형태·자산 의존적 crisis-convexity*로 수렴했고,
그마저 일반 성질이 아님이 다자산·적대적 검증으로 확인됐다. 이는 실패가 아니라
*효율적 시장에 대한 올바른 발견*이며, 핵심 가치는:

> **느리고 비싼 NO 대신, 빠르고 싼 NO.**
> 자본 $0 손실로 "여기엔 없다"를 13번 입증했고, 그 과정에서 재사용 가능한
> 반증 규율(precheck 게이트·적대적 검증 포함)과 실행 인프라를 만들었다.

---

## 6. 사용·재현·확장하는 법

이건 OSS(MIT)다 — 남들이 같은 막다른 길을 반복하지 않도록.

**결과 재현**: 모든 가설에 `docs/plans/*-design.md`(사전 폐기기준 포함) +
`*-result.md` + 실행 스크립트가 있다.
```bash
python scripts/funding_carry_backtest.py            # #9, 완전 재현
python scripts/run_backtest.py --symbol XRPUSDT --sentiment-mode off  # #8 베이스라인
python -m src.statarb.precheck                      # #11 공적분 precheck 게이트
python -m src.momentum.crisis_alpha                 # #13 다자산 crisis-alpha 일반성
```

**결과 반박**: 여기서 가장 가치 있는 기여는 근거 있는 "당신의 폐기 판정이
X 때문에 틀렸다"이다. 폐기기준이 명시돼 있어 대조 검증 가능.

**컴포넌트 재사용** (각각 비교적 독립적):
- 결정론적 LLM 프로바이더 — `temperature=0` + 응답해시 캐시, 의존성 0
  (`src/sentiment_provider.py`)
- 비용모델 포함 벡터화 walk-forward 백테스터 (`src/backtester.py`)
- idempotent·재개 가능 Binance fapi 파이프라인 (`scripts/collect_*.py`,
  `scripts/build_sentiment_dataset.py`)
- 듀얼레이어 킬스위치 (`src/bot.py`)
- 데이터-only precheck 게이트 — 공적분(EG/Johansen+BH 보정)·OU half-life·
  펀딩경제성·예측edge-vs-비용·walk-forward·block bootstrap, 임의 자산군 재사용
  (`src/{statarb,carry,leadlag,momentum}/`)

**새 반증 실험 추가**: 동일 규율을 따를 것 — 실행 *전* 폐기기준 확정,
데이터-only 검증 우선, 결과와 무관하게 정직한 `*-result.md`.
[`CONTRIBUTING.md`](../CONTRIBUTING.md) 참조.

**방법론은 트레이딩 밖으로 일반화된다**: 사전 폐기기준 + 최소비용 검증 +
음성 결과의 1급 취급은 과적합·생존편향·gross-net 혼동으로 자기기만하기
쉬운 모든 경험적/ML 리서치에 적용된다.

**근거 자료 위치**: `docs/plans/2026-05-04-strategy-post-mortem.md`(1–7),
`docs/plans/2026-05-16~17-*-result.md`(8–10),
`docs/plans/2026-05-29-*-result.md`(11–13: statarb·carry·leadlag·momentum·
walkforward·crisis-alpha), `CLAUDE.md` plan history, `scripts/*_backtest.py` +
`src/{statarb,carry,leadlag,momentum}/`(재현 가능 코드).

# 의사결정 로그: ML 필터 비활성화 & XRP 단독 운영

**일자**: 2026-03-21
**결정자**: gihyeon
**상태**: 확정, 운영 반영 완료

---

## 1. ML 필터를 왜 껐는가

### 결론: ML OFF > ML ON (전 심볼)

Walk-Forward 검증 결과, ML 필터를 끈 상태가 모든 심볼에서 더 나은 성과를 보였다.

| 심볼 | ML OFF PF | ML ON PF | 차이 | ML OFF Return | ML ON Return |
|------|-----------|----------|------|---------------|--------------|
| **XRPUSDT** | **1.16** | 0.71 | -0.45 (61%↓) | +12.17% | -25.62% |
| DOGEUSDT | 1.18 | 0.78 | -0.40 (34%↓) | +16.11% | -28.50% |
| SOLUSDT | 0.09 | 0.25 | — | -321.85% | -48.83% |

### 원인 분석

**1) Ablation 실험 — 모델이 독립적 알파를 제공하지 못함**
- 실험 A: 전체 26개 피처 (baseline AUC)
- 실험 B: signal_strength 제거
- 실험 C: signal_strength + side 제거
- **A→C AUC 하락: 0.08~0.09** (판정 기준: ≤0.05 유용, 0.05~0.10 조건부, ≥0.10 재설계)
- 해석: 모델이 기존 기술적 신호(RSI, MACD, ADX)를 단순 재확인하는 수준. 독립적 예측력 부재.

**2) 학습 데이터 부족**
- Walk-Forward 각 폴드 학습 세트에 유효 신호 ~27건
- 1:1 언더샘플링 후 양성 샘플 ~13건/폴드 → LightGBM 학습에 극히 부족
- 과적합 → 일반화 실패

**3) Purged Gap 적용 후 성능 추가 하락**
- 라벨 생성에 24캔들(6h) lookahead 사용 → 학습/검증 사이에 24캔들 embargo 추가
- 이전에 label leakage로 부풀려진 성능이 정정됨

### 운영 설정
```
NO_ML_FILTER=true  # .env
```
모델 파일은 유지 (향후 재검증용). `ml_filter.py`의 hot-reload 로직도 그대로 남겨둠.

---

## 2. SOL/DOGE/TRX를 왜 뺐는가

### 결론: XRP만 PF > 1.0 달성

| 심볼 | Strategy Sweep 최고 PF | Walk-Forward PF (ML OFF) | 판정 |
|------|----------------------|--------------------------|------|
| **XRPUSDT** | 1.68 | **1.16** | ✅ 운영 유지 |
| DOGEUSDT | 1.80 | 1.18* | ❌ 제외 |
| TRXUSDT | 3.87 (16건) | — | ❌ 제외 |
| SOLUSDT | 2.83 | **0.09** | ❌ 제외 |

*DOGE PF 1.18은 WR 25%로 소수 대형 승리에 의존 → 안정성 부족

### 핵심 교훈: 과적합 탐지

**SOLUSDT 사례가 가장 극적:**
- Strategy Sweep (1년 전체 백테스트): PF 2.83, Return +90.93%
- Walk-Forward (시계열 CV): PF 0.09, Return -321.85%
- **과적합 정도: PF 2.83 → 0.09 (97% 하락)**

→ 전체 기간 백테스트 결과만으로 심볼을 선택하면 안 됨. 반드시 Walk-Forward로 검증해야 함.

### 운영 설정
```
SYMBOLS=XRPUSDT  # .env (이전: XRPUSDT,SOLUSDT,DOGEUSDT,TRXUSDT)
```

---

## 3. ML을 다시 켜려면 어떤 조건이 필요한가

### 필수 조건 (AND)

1. **데이터 양**: Walk-Forward 폴드당 유효 신호 100건 이상
   - 현재 ~27건 → 약 4배 필요
   - 방법: (a) 더 긴 수집 기간 (1년→3년), (b) 15m→5m 타임프레임 (데이터 3배), (c) 새 피처로 유효 신호 비율 증가

2. **독립적 알파**: Ablation A→C AUC 하락 ≤ 0.05
   - signal_strength와 side를 제거해도 모델이 독립적으로 예측할 수 있어야 함
   - 현재 0.08~0.09 → 새 피처(L/S ratio, OI 파생 등)가 이 갭을 메워야 함

3. **Walk-Forward 검증**: ML ON PF > ML OFF PF (최소 0.1 이상 차이)
   - 단순히 PF > 1.0이 아니라, ML OFF 대비 개선이 있어야 함
   - 검증 거래 수 50건 이상

4. **과적합 지표**: Strategy Sweep PF vs Walk-Forward PF 비율 < 2.0
   - SOL처럼 Sweep 2.83 / WF 0.09 = 31배 차이 → 극심한 과적합
   - 비율 2.0 이하면 합리적 범위

### 유망한 다음 시도

| 개선 방향 | 기대 효과 | 현재 상태 |
|-----------|-----------|-----------|
| **L/S Ratio 피처 추가** | 독립적 알파 (상관 0.12~0.14) | 수집 시작 (2026-03-22), 1개월 뒤 검증 가능 |
| **학습 데이터 3년 확보** | 폴드당 샘플 3배 증가 | 미착수 |
| **Cross-symbol 피처** | BTC/ETH 탑 트레이더 동향 → XRP 예측 | L/S ratio 수집 후 가능 |
| **다른 모델 (XGBoost, CatBoost)** | 소규모 데이터에 더 적합할 수 있음 | 미착수 |

### 재검증 타임라인

```
2026-03-22: L/S ratio 수집 시작 (top_acct + global, 3심볼)
2026-04-22: 1개월 데이터 축적 (~17,000건)
            → 상관분석 재실행 (5일 → 30일 데이터로 신뢰도 확인)
            → L/S ratio 피처를 ML에 추가하여 Ablation 재실험
            → Walk-Forward ML ON vs OFF 재비교
```

---

## 4. 관련 문서 & 코드

| 참조 | 위치 |
|------|------|
| ML 비활성화 커밋 | `dacefaa` (docs: update for XRP-only operation) |
| ML 비교 결과 (XRP) | `results/xrpusdt/ml_comparison_20260321_200332.json` |
| ML 비교 결과 (DOGE) | `results/dogeusdt/ml_comparison_20260321_200334.json` |
| Strategy Sweep 결과 | `results/{symbol}/strategy_sweep_*.json` |
| Purged Gap 계획 | `docs/plans/2026-03-21-purged-gap-and-ablation.md` |
| ML 검증 파이프라인 | `docs/plans/2026-03-21-ml-validation-pipeline.md` |
| ML 검증 결과 | `docs/plans/2026-03-21-ml-validation-result.md` |
| L/S Ratio 수집 스크립트 | `scripts/collect_ls_ratio.py` |
| 운영 설정 | `.env` → `NO_ML_FILTER=true`, `SYMBOLS=XRPUSDT` |

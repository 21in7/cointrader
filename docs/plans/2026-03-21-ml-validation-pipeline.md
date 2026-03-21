# ML Validation Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** ML 필터의 실전 가치를 검증하는 `--compare-ml` CLI를 추가하여, 완화된 임계값에서 ML on/off Walk-Forward 백테스트를 자동 비교하고 PF/승률/MDD 개선폭을 리포트한다.

**Architecture:** `scripts/run_backtest.py`에 `--compare-ml` 플래그를 추가한다. 이 플래그가 활성화되면 WalkForwardBacktester를 `use_ml=True`와 `use_ml=False`로 각각 실행하고, 결과를 나란히 비교하는 리포트를 출력한다. 기존 `Backtester`/`WalkForwardBacktester` 코드는 변경하지 않는다.

**Tech Stack:** Python, LightGBM, src/backtester.py (기존 모듈 재사용)

**선행 완료 항목 (이미 구현됨):**
- ✅ 학습 전용 상수 (TRAIN_SIGNAL_THRESHOLD=2, TRAIN_ADX_THRESHOLD=15, etc.)
- ✅ Purged gap (embargo=LOOKAHEAD) in all walk-forward functions
- ✅ Ablation A/B/C CLI (`--ablation`)
- ✅ `BacktestConfig.use_ml` 플래그
- ✅ `run_backtest.py --no-ml` 지원

**판단 기준 (합의됨):**
- ML on vs ML off의 **상대 PF 개선폭**으로 판단 (절대 기준 아님)
- PF 개선 + 승률 개선 + MDD 감소 → 투입 가치 있음
- PF 변화 미미 → ML 기여 낮음

---

## File Structure

| 파일 | 변경 유형 | 역할 |
|------|-----------|------|
| `scripts/run_backtest.py` | Modify | `--compare-ml` CLI + 비교 리포트 |
| `CLAUDE.md` | Modify | plan history 업데이트 |

---

### Task 1: `--compare-ml` CLI 추가

**Files:**
- Modify: `scripts/run_backtest.py:29-55, 151-211`

- [ ] **Step 1: argparse에 --compare-ml 추가**

`scripts/run_backtest.py`의 `parse_args()` 함수에:

```python
p.add_argument("--compare-ml", action="store_true",
               help="ML on vs off Walk-Forward 비교 (--walk-forward 자동 활성화)")
```

- [ ] **Step 2: compare_ml 함수 작성**

`scripts/run_backtest.py`에 `compare_ml()` 함수 추가:

```python
def compare_ml(symbols: list[str], args):
    """ML on vs ML off Walk-Forward 백테스트 비교.

    완화된 임계값(threshold=2)에서 ML 필터의 실질적 가치를 검증한다.
    판단 기준: 상대 PF 개선폭 (절대 기준 아님).
    """
    base_kwargs = dict(
        symbols=symbols,
        start=args.start,
        end=args.end,
        initial_balance=args.balance,
        leverage=args.leverage,
        fee_pct=args.fee,
        slippage_pct=args.slippage,
        ml_threshold=args.ml_threshold,
        atr_sl_mult=args.sl_atr,
        atr_tp_mult=args.tp_atr,
        signal_threshold=args.signal_threshold,
        adx_threshold=args.adx_threshold,
        volume_multiplier=args.vol_multiplier,
        train_months=args.train_months,
        test_months=args.test_months,
    )

    results = {}
    for label, use_ml in [("ML OFF", False), ("ML ON", True)]:
        print(f"\n{'='*60}")
        print(f"  Walk-Forward 백테스트: {label}")
        print(f"{'='*60}")

        cfg = WalkForwardConfig(**base_kwargs, use_ml=use_ml)
        wf = WalkForwardBacktester(cfg)
        result = wf.run()
        results[label] = result
        print_summary(result["summary"], cfg, mode="walk_forward")
        if result.get("folds"):
            print_fold_table(result["folds"])

    # 비교 리포트
    _print_comparison(results, symbols)

    # 결과 저장
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if len(symbols) == 1:
        out_dir = Path(f"results/{symbols[0].lower()}")
    else:
        out_dir = Path("results/combined")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"ml_comparison_{ts}.json"

    comparison = {
        "timestamp": datetime.now().isoformat(),
        "symbols": symbols,
        "ml_off": results["ML OFF"]["summary"],
        "ml_on": results["ML ON"]["summary"],
    }

    def sanitize(obj):
        if isinstance(obj, bool):
            return obj
        if isinstance(obj, (int, float)):
            if isinstance(obj, float) and obj == float("inf"):
                return "Infinity"
            return obj
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [sanitize(v) for v in obj]
        return obj

    with open(path, "w") as f:
        json.dump(sanitize(comparison), f, indent=2, ensure_ascii=False)
    print(f"\n비교 결과 저장: {path}")


def _print_comparison(results: dict, symbols: list[str]):
    """ML on vs off 비교 리포트 출력."""
    off = results["ML OFF"]["summary"]
    on = results["ML ON"]["summary"]

    print(f"\n{'='*64}")
    print(f"  ML ON vs OFF 비교 ({', '.join(symbols)})")
    print(f"{'='*64}")
    print(f"  {'지표':<20} {'ML OFF':>12} {'ML ON':>12} {'Delta':>12}")
    print(f"{'─'*64}")

    metrics = [
        ("총 거래", "total_trades", "d"),
        ("총 PnL (USDT)", "total_pnl", ".2f"),
        ("수익률 (%)", "return_pct", ".2f"),
        ("승률 (%)", "win_rate", ".1f"),
        ("Profit Factor", "profit_factor", ".2f"),
        ("MDD (%)", "max_drawdown_pct", ".2f"),
        ("Sharpe", "sharpe_ratio", ".2f"),
    ]

    for label, key, fmt in metrics:
        v_off = off.get(key, 0)
        v_on = on.get(key, 0)
        # inf 처리
        if v_off == float("inf"):
            v_off_str = "INF"
        else:
            v_off_str = f"{v_off:{fmt}}"
        if v_on == float("inf"):
            v_on_str = "INF"
        else:
            v_on_str = f"{v_on:{fmt}}"

        if isinstance(v_off, (int, float)) and isinstance(v_on, (int, float)) \
                and v_off != float("inf") and v_on != float("inf"):
            delta = v_on - v_off
            sign = "+" if delta > 0 else ""
            delta_str = f"{sign}{delta:{fmt}}"
        else:
            delta_str = "N/A"

        print(f"  {label:<20} {v_off_str:>12} {v_on_str:>12} {delta_str:>12}")

    # 판정
    pf_off = off.get("profit_factor", 0)
    pf_on = on.get("profit_factor", 0)
    wr_off = off.get("win_rate", 0)
    wr_on = on.get("win_rate", 0)
    mdd_off = off.get("max_drawdown_pct", 0)
    mdd_on = on.get("max_drawdown_pct", 0)

    print(f"{'─'*64}")

    if pf_off == float("inf") or pf_on == float("inf"):
        print(f"  판정: PF=INF — 한쪽 모드에서 손실 거래 없음 (거래 수 부족 가능), 판단 보류")
    elif pf_off == 0:
        print(f"  판정: ML OFF PF=0 — baseline 거래 없음, 판단 불가")
    else:
        pf_improvement = pf_on - pf_off
        wr_improvement = wr_on - wr_off
        mdd_improvement = mdd_off - mdd_on  # MDD는 낮을수록 좋음

        # 판정 임계값 (초기값 — 실제 백테스트 결과를 보고 조정 가능)
        improvements = []
        if pf_improvement > 0.1:
            improvements.append(f"PF +{pf_improvement:.2f}")
        if wr_improvement > 2.0:
            improvements.append(f"승률 +{wr_improvement:.1f}%p")
        if mdd_improvement > 1.0:
            improvements.append(f"MDD -{mdd_improvement:.1f}%p")

        if len(improvements) >= 2:
            verdict = f"✅ ML 필터 투입 가치 있음 ({', '.join(improvements)})"
        elif len(improvements) == 1:
            verdict = f"⚠️ ML 필터 조건부 투입 ({improvements[0]}, 다른 지표 변화 미미)"
        else:
            verdict = f"❌ ML 필터 기여 미미 (PF {pf_improvement:+.2f}, 승률 {wr_improvement:+.1f}%p)"
        print(f"  판정: {verdict}")

    print(f"{'='*64}\n")
```

- [ ] **Step 3: main()에 --compare-ml 분기 추가**

`scripts/run_backtest.py`의 `main()` 함수에서 `if args.walk_forward:` 블록 **앞에** 추가:

```python
if args.compare_ml:
    if args.no_ml:
        logger.warning("--no-ml is ignored when using --compare-ml")
    compare_ml(symbols, args)
    return
```

- [ ] **Step 4: 전체 테스트 통과 확인**

Run: `bash scripts/run_tests.sh`
Expected: ALL PASS (기존 테스트 영향 없음)

- [ ] **Step 5: 커밋**

```bash
git add scripts/run_backtest.py
git commit -m "feat(backtest): add --compare-ml for ML on/off walk-forward comparison"
```

---

### Task 2: CLAUDE.md 업데이트

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: plan history 업데이트**

```markdown
| 2026-03-21 | `ml-validation-pipeline` (plan) | Completed |
```

- [ ] **Step 2: 커밋**

```bash
git add CLAUDE.md
git commit -m "docs: update plan history with ml-validation-pipeline"
```

---

## 구현 후 실행 가이드

### Phase 1: Ablation 진단 (이미 구현됨)

```bash
# 심볼별 ablation 실행
python scripts/train_model.py --symbol XRPUSDT --ablation
python scripts/train_model.py --symbol SOLUSDT --ablation
python scripts/train_model.py --symbol DOGEUSDT --ablation
```

판단:
- A→C 드롭 ≤ 0.05 → Phase 2로 진행
- A→C 드롭 ≥ 0.10 → ML 재설계 필요 (중단)

### Phase 2: ML on/off 비교 (이 플랜에서 구현)

```bash
# 완화된 임계값(threshold=2)로 ML 비교
python scripts/run_backtest.py --symbol XRPUSDT --compare-ml \
  --signal-threshold 2 --adx-threshold 15 --vol-multiplier 1.5 --walk-forward

python scripts/run_backtest.py --symbol SOLUSDT --compare-ml \
  --signal-threshold 2 --adx-threshold 15 --vol-multiplier 1.5 --walk-forward

python scripts/run_backtest.py --symbol DOGEUSDT --compare-ml \
  --signal-threshold 2 --adx-threshold 15 --vol-multiplier 1.5 --walk-forward
```

판단: 상대 PF 개선폭으로 ML 가치 평가

### Phase 3: 실전 점진적 전환 (코드 변경 불필요)

Phase 1, 2 모두 긍정적이면 `.env`로 1심볼부터 적용:

```bash
# .env에 추가 (1심볼만 먼저)
SIGNAL_THRESHOLD_XRPUSDT=2
ADX_THRESHOLD_XRPUSDT=15
VOL_MULTIPLIER_XRPUSDT=1.5

# 나머지 심볼은 기존 값 유지
# SIGNAL_THRESHOLD_SOLUSDT=3  (기본값)
# SIGNAL_THRESHOLD_DOGEUSDT=3 (기본값)
```

1~2주 운영 후 kill switch 미발동 + PnL 양호하면 나머지 심볼도 전환.

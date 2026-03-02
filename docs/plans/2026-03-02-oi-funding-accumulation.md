# OI/펀딩비 누적 저장 (접근법 B) 구현 계획

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** `fetch_history.py`의 데이터 수집 방식을 덮어쓰기(Overwrite)에서 Upsert(병합)로 변경해, 매일 실행할 때마다 기존 parquet의 OI/펀딩비 0.0 구간이 실제 값으로 채워지며 고품질 데이터가 무한히 누적되도록 한다.

**Architecture:**
- `fetch_history.py`에 `--upsert` 플래그 추가 (기본값 True). 기존 parquet이 있으면 로드 후 신규 데이터와 timestamp 기준 병합(Upsert). 없으면 기존처럼 새로 생성.
- Upsert 규칙: 기존 행의 `oi_change` / `funding_rate`가 0.0이면 신규 값으로 덮어씀. 신규 행은 그냥 추가. 중복 제거 후 시간순 정렬.
- `train_and_deploy.sh`의 `--days` 인자를 35일로 조정 (30일 API 한도 + 5일 버퍼).
- LXC 운영서버는 모델 파일만 받으므로 변경 없음. 맥미니의 `data/` 폴더에만 누적.

**Tech Stack:** pandas, parquet (pyarrow), pytest

---

## Task 1: fetch_history.py — upsert_parquet() 함수 추가 및 --upsert 플래그

**Files:**
- Modify: `scripts/fetch_history.py`
- Test: `tests/test_fetch_history.py` (신규 생성)

### Step 1: 실패 테스트 작성

`tests/test_fetch_history.py` 파일을 새로 만든다.

```python
"""fetch_history.py의 upsert_parquet() 함수 테스트."""
import pandas as pd
import numpy as np
import pytest
from pathlib import Path


def _make_parquet(tmp_path: Path, rows: dict) -> Path:
    """테스트용 parquet 파일 생성 헬퍼."""
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp")
    path = tmp_path / "test.parquet"
    df.to_parquet(path)
    return path


def test_upsert_fills_zero_oi_with_real_value(tmp_path):
    """기존 행의 oi_change=0.0이 신규 데이터의 실제 값으로 덮어써진다."""
    from scripts.fetch_history import upsert_parquet

    existing_path = _make_parquet(tmp_path, {
        "timestamp": ["2026-01-01 00:00", "2026-01-01 00:15"],
        "close": [1.0, 1.1],
        "oi_change": [0.0, 0.0],
        "funding_rate": [0.0, 0.0],
    })

    new_df = pd.DataFrame({
        "close": [1.0, 1.1],
        "oi_change": [0.05, 0.03],
        "funding_rate": [0.0001, 0.0001],
    }, index=pd.to_datetime(["2026-01-01 00:00", "2026-01-01 00:15"], utc=True))
    new_df.index.name = "timestamp"

    result = upsert_parquet(existing_path, new_df)

    assert result.loc["2026-01-01 00:00+00:00", "oi_change"] == pytest.approx(0.05)
    assert result.loc["2026-01-01 00:15+00:00", "oi_change"] == pytest.approx(0.03)


def test_upsert_appends_new_rows(tmp_path):
    """신규 타임스탬프 행이 기존 데이터 아래에 추가된다."""
    from scripts.fetch_history import upsert_parquet

    existing_path = _make_parquet(tmp_path, {
        "timestamp": ["2026-01-01 00:00"],
        "close": [1.0],
        "oi_change": [0.05],
        "funding_rate": [0.0001],
    })

    new_df = pd.DataFrame({
        "close": [1.1],
        "oi_change": [0.03],
        "funding_rate": [0.0002],
    }, index=pd.to_datetime(["2026-01-01 00:15"], utc=True))
    new_df.index.name = "timestamp"

    result = upsert_parquet(existing_path, new_df)

    assert len(result) == 2
    assert "2026-01-01 00:15+00:00" in result.index.astype(str).tolist() or \
           pd.Timestamp("2026-01-01 00:15", tz="UTC") in result.index


def test_upsert_keeps_nonzero_existing_oi(tmp_path):
    """기존 행의 oi_change가 이미 0이 아니면 덮어쓰지 않는다."""
    from scripts.fetch_history import upsert_parquet

    existing_path = _make_parquet(tmp_path, {
        "timestamp": ["2026-01-01 00:00"],
        "close": [1.0],
        "oi_change": [0.07],   # 이미 실제 값 존재
        "funding_rate": [0.0003],
    })

    new_df = pd.DataFrame({
        "close": [1.0],
        "oi_change": [0.05],   # 다른 값으로 덮어쓰려 해도
        "funding_rate": [0.0001],
    }, index=pd.to_datetime(["2026-01-01 00:00"], utc=True))
    new_df.index.name = "timestamp"

    result = upsert_parquet(existing_path, new_df)

    # 기존 값(0.07)이 유지되어야 한다
    assert result.iloc[0]["oi_change"] == pytest.approx(0.07)


def test_upsert_no_existing_file_returns_new_df(tmp_path):
    """기존 parquet 파일이 없으면 신규 데이터를 그대로 반환한다."""
    from scripts.fetch_history import upsert_parquet

    nonexistent_path = tmp_path / "nonexistent.parquet"
    new_df = pd.DataFrame({
        "close": [1.0, 1.1],
        "oi_change": [0.05, 0.03],
        "funding_rate": [0.0001, 0.0001],
    }, index=pd.to_datetime(["2026-01-01 00:00", "2026-01-01 00:15"], utc=True))
    new_df.index.name = "timestamp"

    result = upsert_parquet(nonexistent_path, new_df)

    assert len(result) == 2
    assert result.iloc[0]["oi_change"] == pytest.approx(0.05)


def test_upsert_result_is_sorted_by_timestamp(tmp_path):
    """결과 DataFrame이 timestamp 기준 오름차순 정렬되어 있다."""
    from scripts.fetch_history import upsert_parquet

    existing_path = _make_parquet(tmp_path, {
        "timestamp": ["2026-01-01 00:15"],
        "close": [1.1],
        "oi_change": [0.0],
        "funding_rate": [0.0],
    })

    new_df = pd.DataFrame({
        "close": [1.0, 1.1, 1.2],
        "oi_change": [0.05, 0.03, 0.02],
        "funding_rate": [0.0001, 0.0001, 0.0002],
    }, index=pd.to_datetime(
        ["2026-01-01 00:00", "2026-01-01 00:15", "2026-01-01 00:30"], utc=True
    ))
    new_df.index.name = "timestamp"

    result = upsert_parquet(existing_path, new_df)

    assert result.index.is_monotonic_increasing
    assert len(result) == 3
```

### Step 2: 테스트 실패 확인

```bash
.venv/bin/pytest tests/test_fetch_history.py -v
```

Expected: `FAILED` — `ImportError: cannot import name 'upsert_parquet' from 'scripts.fetch_history'`

### Step 3: fetch_history.py에 upsert_parquet() 함수 구현

`scripts/fetch_history.py`의 `main()` 함수 바로 위에 추가한다.

```python
def upsert_parquet(path: Path | str, new_df: pd.DataFrame) -> pd.DataFrame:
    """
    기존 parquet 파일에 신규 데이터를 Upsert(병합)한다.

    규칙:
    - 기존 행의 oi_change / funding_rate가 0.0이면 신규 값으로 덮어씀
    - 기존 행의 oi_change / funding_rate가 이미 0이 아니면 유지
    - 신규 타임스탬프 행은 그냥 추가
    - 결과는 timestamp 기준 오름차순 정렬, 중복 제거

    Args:
        path: 기존 parquet 경로 (없으면 new_df 그대로 반환)
        new_df: 새로 수집한 DataFrame (timestamp index)

    Returns:
        병합된 DataFrame
    """
    path = Path(path)
    if not path.exists():
        return new_df.sort_index()

    existing = pd.read_parquet(path)

    # timestamp index 통일 (tz-aware UTC)
    if existing.index.tz is None:
        existing.index = existing.index.tz_localize("UTC")
    if new_df.index.tz is None:
        new_df.index = new_df.index.tz_localize("UTC")

    # 기존 데이터에서 oi_change / funding_rate가 0.0인 행만 신규 값으로 업데이트
    UPSERT_COLS = ["oi_change", "funding_rate"]
    overlap_idx = existing.index.intersection(new_df.index)

    for col in UPSERT_COLS:
        if col not in existing.columns or col not in new_df.columns:
            continue
        # 겹치는 행 중 기존 값이 0.0인 경우에만 신규 값으로 교체
        zero_mask = existing.loc[overlap_idx, col] == 0.0
        update_idx = overlap_idx[zero_mask]
        if len(update_idx) > 0:
            existing.loc[update_idx, col] = new_df.loc[update_idx, col]

    # 신규 타임스탬프 행 추가 (기존에 없는 것만)
    new_only_idx = new_df.index.difference(existing.index)
    if len(new_only_idx) > 0:
        existing = pd.concat([existing, new_df.loc[new_only_idx]])

    return existing.sort_index()
```

### Step 4: main()에 --upsert 플래그 추가 및 저장 로직 수정

`main()` 함수의 `parser` 정의 부분에 인자 추가:

```python
parser.add_argument(
    "--no-upsert", action="store_true",
    help="기존 parquet을 Upsert하지 않고 새로 덮어씀 (기본: Upsert 활성화)",
)
```

그리고 단일 심볼 저장 부분:
```python
# 기존:
df.to_parquet(args.output)

# 변경:
if not args.no_upsert:
    df = upsert_parquet(args.output, df)
df.to_parquet(args.output)
```

멀티 심볼 저장 부분도 동일하게:
```python
# 기존:
merged.to_parquet(output)

# 변경:
if not args.no_upsert:
    merged = upsert_parquet(output, merged)
merged.to_parquet(output)
```

### Step 5: 테스트 통과 확인

```bash
.venv/bin/pytest tests/test_fetch_history.py -v
```

Expected: 전체 PASS

### Step 6: 커밋

```bash
git add scripts/fetch_history.py tests/test_fetch_history.py
git commit -m "feat: add upsert_parquet to accumulate OI/funding data incrementally"
```

---

## Task 2: train_and_deploy.sh — 데이터 수집 일수 35일로 조정

**Files:**
- Modify: `scripts/train_and_deploy.sh`

### Step 1: 현재 상태 확인

`scripts/train_and_deploy.sh`에서 `--days 365` 부분을 찾는다.

### Step 2: 수정

`train_and_deploy.sh`에서 `fetch_history.py` 호출 부분을 수정한다.

기존:
```bash
python scripts/fetch_history.py \
    --symbols XRPUSDT BTCUSDT ETHUSDT \
    --interval 15m \
    --days 365 \
    --output data/combined_15m.parquet
```

변경:
```bash
# OI/펀딩비 API 제한(30일) + 버퍼 5일 = 35일치 신규 수집 후 기존 parquet에 Upsert
python scripts/fetch_history.py \
    --symbols XRPUSDT BTCUSDT ETHUSDT \
    --interval 15m \
    --days 35 \
    --output data/combined_15m.parquet
```

**이유**: 매일 실행 시 35일치만 새로 가져와 기존 누적 parquet에 Upsert한다.
- 최초 실행 시(`data/combined_15m.parquet` 없음): 35일치로 시작
- 이후 매일: 35일치 신규 데이터로 기존 파일의 0.0 구간을 채우고 최신 행 추가
- 시간이 지날수록 OI/펀딩비 실제 값이 있는 구간이 1달 → 2달 → ... 로 늘어남

**주의**: 최초 실행 시 캔들 데이터도 35일치만 있으므로, 첫 실행은 수동으로
`--days 365 --no-upsert`로 전체 캔들을 먼저 수집하는 것을 권장한다.
README에 이 내용을 추가한다.

### Step 3: 커밋

```bash
git add scripts/train_and_deploy.sh
git commit -m "feat: fetch 35 days for daily upsert instead of overwriting 365 days"
```

---

## Task 3: 전체 테스트 통과 확인 및 README 업데이트

### Step 1: 전체 테스트 실행

```bash
.venv/bin/pytest tests/ --ignore=tests/test_mlx_filter.py --ignore=tests/test_database.py -v
```

Expected: 전체 PASS

### Step 2: README.md 업데이트

**"ML 모델 학습" 섹션의 "전체 파이프라인 (권장)" 부분 아래에 아래 내용을 추가한다:**

```markdown
### 최초 실행 (캔들 전체 수집)

처음 실행하거나 `data/combined_15m.parquet`가 없을 때는 전체 캔들을 먼저 수집한다.
이후 매일 크론탭이 `train_and_deploy.sh`를 실행하면 35일치 신규 데이터가 자동으로 Upsert된다.

```bash
# 최초 1회: 1년치 캔들 전체 수집 (OI/펀딩비는 최근 30일만 실제 값, 나머지 0.0)
python scripts/fetch_history.py \
    --symbols XRPUSDT BTCUSDT ETHUSDT \
    --interval 15m \
    --days 365 \
    --no-upsert \
    --output data/combined_15m.parquet

# 이후 매일 자동 실행 (크론탭 또는 train_and_deploy.sh):
# 35일치 신규 데이터를 기존 파일에 Upsert → OI/펀딩비 0.0 구간이 야금야금 채워짐
bash scripts/train_and_deploy.sh
```
```

**"주요 기능" 섹션에 아래 항목 추가:**

```markdown
- **OI/펀딩비 누적 학습**: 매일 35일치 신규 데이터를 기존 parquet에 Upsert. 시간이 지날수록 실제 OI/펀딩비 값이 있는 학습 구간이 1달 → 2달 → 반년으로 늘어남
```

### Step 3: 최종 커밋

```bash
git add README.md
git commit -m "docs: document OI/funding incremental accumulation strategy"
```

---

## 구현 후 검증 포인트

1. `data/combined_15m.parquet`에서 날짜별 `oi_change` 값 분포 확인:
   ```python
   import pandas as pd
   df = pd.read_parquet("data/combined_15m.parquet")
   print(df["oi_change"].describe())
   print((df["oi_change"] == 0.0).sum(), "개 행이 아직 0.0")
   ```
2. 매일 실행 후 0.0 행 수가 줄어드는지 확인
3. 모델 학습 시 `oi_change` / `funding_rate` 피처의 non-zero 비율이 증가하는지 확인

---

## 아키텍처 메모 (LXC 운영서버 관련)

- **LXC 운영서버(10.1.10.24)**: 변경 없음. 모델 파일(`*.pkl` / `*.onnx`)만 받음
- **맥미니**: `data/combined_15m.parquet`를 누적 보관. 매일 35일치 Upsert 후 학습
- **데이터 흐름**: 맥미니 parquet 누적 → 학습 → 모델 → LXC 배포
- **봇 실시간 OI/펀딩비**: 접근법 A(Task 1~4)에서 이미 구현됨. LXC 봇이 캔들마다 REST API로 실시간 수집

# 온체인 netflow 디리스크 오버레이 (비방향성) 결과

## 맥락
같은 날 directional netflow precheck FAIL(0/12) 후 **(c) 비방향성 피벗**. 레그 비대칭
진단(숏 레그가 강하게 음 = 유입→하방 정보 가능성)에 근거. momentum/crisis-alpha/ETH-L2가
반복 시사한 "리스크 오버레이" 가설의 정직검증 — netflow를 *방향*이 아니라 *노출 조절*에.

**핵심 위험**: 리스크 오버레이는 자기기만이 쉽다(고변동 자산서 아무 디리스크나 Sharpe↑).
→ 설계 핵심 = **anti-fooling 삼중 차단**: B1 변동성관리(빈도매칭), B2 랜덤 null(부트스트랩),
레짐(크래시구간 제거).

## 방법론 — falsification-first
사전등록(`src/onchain/overlay.py` 상단 상수, 사후 변경 금지):
- 데이터: 같은 `data/{btc,eth}usdt/onchain_daily.parquet` (CM community, BTC 15yr/ETH 11yr)
- 전략: 기본 롱(exp=1), `netflow_z_{t−1−LAG}>THR` 시 플랫(exp=0). **숏 없음**.
  z90/LAG1/THR1.0 1차. `shift(1+LAG)` lookahead·flash개정 차단. 토글당 5bps.
- 벤치: B0 buy&hold, B1 변동성관리(30d 실현변동성 상위, 디리스크 빈도 매칭),
  B2 랜덤 디리스크(빈도 매칭, N=1000 부트스트랩 null).

### 사전등록 게이트
| # | 게이트 | 기준 |
|---|---|---|
| 1 | 개선 | overlay Sharpe>B0 AND MDD<B0(덜 심각) |
| 2 | 정보(핵심) | 랜덤 null(B2) Sharpe-gain p<0.05 **AND** overlay Sharpe≥B1(vol) |
| 3 | 레짐 | 크래시구간(2018-01,COVID,LUNA,FTX) 제거 후 Sharpe gain>0 |
| 4 | 안정성 | OOS(마지막 30%)서 overlay Sharpe>B0 |
| P | 포트폴리오 | BTC·ETH 둘 다 or EW |

## 결과 — BTC/ETH/EW 전부 FAIL

| asset | OVsh | BHsh | **VOLsh** | OVmdd | BHmdd | gain | p_rnd | g-crash | OOS ov/bh | g1234 | PASS |
|---|---|---|---|---|---|---|---|---|---|---|---|
| BTC | +0.91 | +0.82 | **+0.99** | −90.0% | −92.7% | +0.089 | **0.030** | +0.091 | 0.03/0.04 | ✓·✗·✓·✗ | FAIL |
| ETH | +0.65 | +0.66 | +0.53 | −86.8% | −94.0% | −0.001 | 0.270 | +0.033 | 0.20/0.04 | ✗·✗·✓·✓ | FAIL |
| EW | +0.80 | +0.77 | +0.62 | −83.3% | −88.4% | +0.029 | 0.076 | +0.051 | 0.36/0.34 | ✓·✗·✓·✓ | FAIL |

디리스크 빈도: BTC 11.1%, ETH 9.0%, EW 17.5%.

### 결정적 실패 = 게이트 2(정보), 두 가지 양상
1. **BTC — 랜덤은 이기나 vol엔 진다**: p_random=0.030(<0.05 ✓, 랜덤 디리스크보다 나음)
   이지만 overlay Sharpe 0.91 **< 변동성관리 0.99**. netflow의 하방 정보는 **실현변동성이
   이미 (더 잘) 포착**. netflow는 vol보다 못한 리스크 타이머.
2. **ETH/EW — 랜덤도 못 이김**: p_random 0.270 / 0.076 (>0.05). 개선이 랜덤 디리스크
   분포 안에 들어감 = 우연과 구분 안 됨.

### 보조 관찰
- **개선폭 자체가 경제적으로 미미**(+0.03~0.09 Sharpe)하고 **임계 취약**: THR=0.5에선
  전부 **음**(−0.01/−0.09/−0.12), THR 1.0~2.0만 소폭 양. 부호가 파라미터 의존.
- **게이트 3은 통과**(gain_excl_crash 전부 +): 작은 이득이 크래시 에피소드에만 몰리진
  않음 = crisis-alpha 함정은 아님. 단 게이트 2가 죽여서 무의미.

## 결론 — FAIL, "정보는 있으나 vol에 지배됨"

netflow 디리스크는 buy&hold·랜덤보다 *약간* 나을 때가 있으나(BTC), **단순 변동성관리
오버레이에 지배된다**(0.91<0.99). netflow가 하방에 대해 가진 한계 정보는 **무료 가격
파생 실현변동성이 이미 더 잘 담고 있다**. ETH/EW는 랜덤조차 못 이긴다.

**anti-fooling 벤치마크가 핵심 역할**: B1·B2 없이 봤다면 BTC의 +0.089 Sharpe gain +
p=0.030(vs 랜덤)은 "승리"로 보였을 것. 변동성 벤치(B1)가 그 이득이 **netflow 고유가
아니라 vol-timing의 열등 버전**임을 폭로. 정직검증 설계가 false-positive를 차단.

## 원질문("유료 API 수익 가능?")에 대한 누적 답
일봉 거래소 netflow는 **방향성 알파로도(directional 0/12), 리스크 오버레이로도(overlay
전패)** 무료 가격 데이터로 할 수 있는 것을 못 넘는다. **일봉 netflow 유료 결제 비추천**
재확인. 온체인의 *다른 차원 정보*는 실재할 수 있으나, 거래소 집계 netflow의 일봉 형태는
가격(수익률·변동성)에 이미 반영돼 잔존 거래가능 정보가 없다.

## Caveat — 배제하지 *못하는* 것 (불변)
1. **유료 티어 고해상도**: 인트라데이/시간별 플로우, 엔티티 레벨(고래/스마트머니), 거래소별
   세분, 스테이블코인 플로우. 일봉 집계는 가장 무딘 형태 — 단 directional+overlay 양쪽
   FAIL로 입증 부담 매우 높음.
2. **다른 온체인 메트릭**: MVRV/활성주소/SOPR 바스켓 미검증. netflow가 가장 이론적
   동기 강했고 그게 실패.
3. **다른 오버레이 형태**: 부분 디리스크/연속 스케일/멀티신호 결합은 미검증(DoF↑ 주의).

## 보존 자산
- `src/onchain/overlay.py` — 디리스크 오버레이 + Sharpe/MDD + B1(vol 빈도매칭)·B2(랜덤
  부트스트랩 null)·레짐(크래시제거)·IS-OOS, 한글 플롯 4. 임의 신호의 오버레이 평가에 재사용.
- 산출물: `results/onchain/onchain_overlay_2026-06-08.json` + equity/underwater/디리스크일/랜덤null 플롯 4.

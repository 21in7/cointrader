import { useState, useEffect, useCallback, useRef } from "react";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  AreaChart, Area, LineChart, Line, CartesianGrid, Cell,
} from "recharts";

/* ── API ──────────────────────────────────────────────────────── */
const api = async (path) => {
  try {
    const r = await fetch(`/api${path}`);
    if (!r.ok) throw new Error(r.statusText);
    return await r.json();
  } catch (e) {
    console.error(`API [${path}]:`, e);
    return null;
  }
};

/* ── 유틸 ─────────────────────────────────────────────────────── */
const fmt = (n, d = 4) => (n != null ? Number(n).toFixed(d) : "—");
const fmtTime = (iso) => {
  if (!iso) return "—";
  const d = new Date(iso);
  return `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
};
const fmtDate = (s) => (s ? s.slice(5, 10).replace("-", "/") : "—");
const pnlColor = (v) => (v > 0 ? "#34d399" : v < 0 ? "#f87171" : "rgba(255,255,255,0.5)");
const pnlSign = (v) => (v > 0 ? `+${fmt(v)}` : fmt(v));

/* ── 스타일 변수 ──────────────────────────────────────────────── */
const S = {
  sans: "'Satoshi','DM Sans',system-ui,sans-serif",
  mono: "'JetBrains Mono','Fira Code',monospace",
  bg: "#08080f",
  surface: "rgba(255,255,255,0.015)",
  surface2: "rgba(255,255,255,0.03)",
  border: "rgba(255,255,255,0.06)",
  text3: "rgba(255,255,255,0.35)",
  text4: "rgba(255,255,255,0.2)",
  green: "#34d399",
  red: "#f87171",
  indigo: "#818cf8",
  amber: "#f59e0b",
};

/* ── Badge ────────────────────────────────────────────────────── */
const Badge = ({ children, bg = "rgba(255,255,255,0.06)", color = "rgba(255,255,255,0.5)" }) => (
  <span style={{
    display: "inline-block", fontSize: 10, fontWeight: 600, padding: "2px 8px",
    borderRadius: 6, background: bg, color, fontFamily: S.mono,
    letterSpacing: 0.5, marginLeft: 4,
  }}>{children}</span>
);

/* ── StatCard ─────────────────────────────────────────────────── */
const StatCard = ({ icon, label, value, sub, accent }) => (
  <div style={{
    background: `linear-gradient(135deg, ${S.surface2} 0%, rgba(255,255,255,0.008) 100%)`,
    border: `1px solid ${S.border}`, borderRadius: 14,
    padding: "18px 20px", position: "relative", overflow: "hidden",
  }}>
    <div style={{
      position: "absolute", top: -20, right: -20, width: 70, height: 70,
      borderRadius: "50%", background: accent, filter: "blur(28px)",
    }} />
    <div style={{
      fontSize: 10, color: S.text3, letterSpacing: 1.5,
      textTransform: "uppercase", fontFamily: S.mono, marginBottom: 6,
    }}>
      {icon && <span style={{ marginRight: 5 }}>{icon}</span>}{label}
    </div>
    <div style={{ fontSize: 26, fontWeight: 700, color: "#fff", fontFamily: S.sans, letterSpacing: -0.5 }}>
      {value}
    </div>
    {sub && (
      <div style={{ fontSize: 11, color: accent, fontFamily: S.mono, marginTop: 2 }}>{sub}</div>
    )}
  </div>
);

/* ── ChartTooltip ─────────────────────────────────────────────── */
const ChartTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{
      background: "rgba(10,10,18,0.95)", border: "1px solid rgba(255,255,255,0.08)",
      borderRadius: 10, padding: "10px 14px", fontSize: 11, fontFamily: S.mono,
    }}>
      <div style={{ color: "rgba(255,255,255,0.4)", marginBottom: 4 }}>{label}</div>
      {payload.filter(p => p.name !== "과매수" && p.name !== "과매도" && p.name !== "임계값").map((p, i) => (
        <div key={i} style={{ color: p.color || "#fff", marginBottom: 1 }}>
          {p.name}: {typeof p.value === "number" ? p.value.toFixed(4) : p.value}
        </div>
      ))}
    </div>
  );
};

/* ── TradeRow ──────────────────────────────────────────────────── */
const TradeRow = ({ trade, isExpanded, onToggle }) => {
  const pnl = trade.net_pnl || 0;
  const isShort = trade.direction === "SHORT";
  const priceDiff = trade.entry_price && trade.exit_price
    ? ((trade.entry_price - trade.exit_price) / trade.entry_price * 100 * (isShort ? 1 : -1)).toFixed(2)
    : "—";

  const sections = [
    {
      title: "리스크 관리",
      items: [
        ["손절가 (SL)", trade.sl, S.red],
        ["익절가 (TP)", trade.tp, S.green],
        ["수량", trade.quantity, "rgba(255,255,255,0.6)"],
      ],
    },
    {
      title: "기술 지표",
      items: [
        ["RSI", trade.rsi, trade.rsi > 70 ? S.amber : S.indigo],
        ["MACD Hist", trade.macd_hist, trade.macd_hist >= 0 ? S.green : S.red],
        ["ATR", trade.atr, "rgba(255,255,255,0.6)"],
      ],
    },
    {
      title: "손익 상세",
      items: [
        ["예상 수익", trade.expected_pnl, S.green],
        ["순수익", trade.net_pnl, pnlColor(trade.net_pnl)],
        ["수수료", trade.commission ? -trade.commission : null, S.red],
      ],
    },
  ];

  return (
    <div style={{ marginBottom: 6 }}>
      <div
        onClick={onToggle}
        style={{
          background: isExpanded ? "rgba(99,102,241,0.06)" : S.surface,
          border: `1px solid ${isExpanded ? "rgba(99,102,241,0.15)" : "rgba(255,255,255,0.04)"}`,
          borderRadius: isExpanded ? "14px 14px 0 0" : 14,
          padding: "14px 18px", cursor: "pointer",
          display: "grid",
          gridTemplateColumns: "36px 1.5fr 0.8fr 0.8fr 0.8fr 32px",
          alignItems: "center", gap: 10, transition: "all 0.15s ease",
        }}
      >
        <div style={{
          width: 30, height: 30, borderRadius: 8,
          background: isShort ? "rgba(239,68,68,0.1)" : "rgba(52,211,153,0.1)",
          display: "flex", alignItems: "center", justifyContent: "center",
          fontSize: 12, fontWeight: 700,
          color: isShort ? S.red : S.green, fontFamily: S.mono,
        }}>
          {isShort ? "S" : "L"}
        </div>

        <div>
          <div style={{ fontSize: 13, fontWeight: 600, color: "#fff", fontFamily: S.sans }}>
            {(trade.symbol || "XRPUSDT").replace("USDT", "/USDT")}
            <Badge
              bg={isShort ? "rgba(239,68,68,0.1)" : "rgba(52,211,153,0.1)"}
              color={isShort ? S.red : S.green}
            >
              {trade.direction}
            </Badge>
            <Badge>{trade.leverage || 10}x</Badge>
          </div>
          <div style={{ fontSize: 10, color: S.text3, marginTop: 2, fontFamily: S.mono }}>
            {fmtDate(trade.entry_time)} {fmtTime(trade.entry_time)} → {fmtTime(trade.exit_time)}
            {trade.close_reason && (
              <span style={{ marginLeft: 6, color: S.text4 }}>({trade.close_reason})</span>
            )}
          </div>
        </div>

        <div style={{ textAlign: "right" }}>
          <div style={{ fontSize: 12, color: "rgba(255,255,255,0.6)", fontFamily: S.mono }}>
            {fmt(trade.entry_price)}
          </div>
          <div style={{ fontSize: 9, color: S.text4 }}>진입가</div>
        </div>

        <div style={{ textAlign: "right" }}>
          <div style={{ fontSize: 12, color: "rgba(255,255,255,0.6)", fontFamily: S.mono }}>
            {fmt(trade.exit_price)}
          </div>
          <div style={{ fontSize: 9, color: S.text4 }}>청산가</div>
        </div>

        <div style={{ textAlign: "right" }}>
          <div style={{ fontSize: 14, fontWeight: 700, color: pnlColor(pnl), fontFamily: S.mono }}>
            {pnlSign(pnl)}
          </div>
          <div style={{ fontSize: 9, color: pnlColor(pnl), opacity: 0.7 }}>{priceDiff}%</div>
        </div>

        <div style={{
          textAlign: "center", color: S.text4, fontSize: 12,
          transition: "transform 0.15s",
          transform: isExpanded ? "rotate(180deg)" : "",
        }}>▾</div>
      </div>

      {isExpanded && (
        <div style={{
          background: "rgba(99,102,241,0.025)",
          border: "1px solid rgba(99,102,241,0.15)",
          borderTop: "none", borderRadius: "0 0 14px 14px",
          padding: "18px 22px",
          display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 14,
        }}>
          {sections.map((sec, si) => (
            <div key={si}>
              <div style={{
                fontSize: 9, color: S.text4, letterSpacing: 1.2,
                fontFamily: S.mono, textTransform: "uppercase", marginBottom: 10,
              }}>
                {sec.title}
              </div>
              {sec.items.map(([label, val, color], ii) => (
                <div key={ii} style={{ display: "flex", justifyContent: "space-between", marginBottom: 5 }}>
                  <span style={{ fontSize: 11, color: "rgba(255,255,255,0.4)" }}>{label}</span>
                  <span style={{ fontSize: 11, color, fontFamily: S.mono }}>
                    {val != null ? fmt(val) : "—"}
                  </span>
                </div>
              ))}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

/* ── 차트 컨테이너 ────────────────────────────────────────────── */
const ChartBox = ({ title, children }) => (
  <div style={{
    background: S.surface, border: `1px solid rgba(255,255,255,0.05)`,
    borderRadius: 14, padding: 18,
  }}>
    <div style={{
      fontSize: 10, color: S.text3, letterSpacing: 1.2,
      fontFamily: S.mono, textTransform: "uppercase", marginBottom: 14,
    }}>
      {title}
    </div>
    {children}
  </div>
);

/* ── 탭 정의 ──────────────────────────────────────────────────── */
const TABS = [
  { id: "overview", label: "Overview", icon: "◆" },
  { id: "trades", label: "Trades", icon: "◈" },
  { id: "chart", label: "Chart", icon: "◇" },
];

/* ═══════════════════════════════════════════════════════════════ */
/* 메인 대시보드                                                    */
/* ═══════════════════════════════════════════════════════════════ */
export default function App() {
  const [tab, setTab] = useState("overview");
  const [expanded, setExpanded] = useState(null);
  const [isLive, setIsLive] = useState(false);
  const [lastUpdate, setLastUpdate] = useState(null);

  const [symbols, setSymbols] = useState([]);
  const symbolsRef = useRef([]);
  const [selectedSymbol, setSelectedSymbol] = useState(null); // null = ALL

  const [stats, setStats] = useState({
    total_trades: 0, wins: 0, losses: 0,
    total_pnl: 0, total_fees: 0, avg_pnl: 0,
    best_trade: 0, worst_trade: 0,
  });
  const [positions, setPositions] = useState([]);
  const [botStatus, setBotStatus] = useState({});
  const [trades, setTrades] = useState([]);
  const [daily, setDaily] = useState([]);
  const [candles, setCandles] = useState([]);

  /* ── 데이터 폴링 ─────────────────────────────────────────── */
  const fetchAll = useCallback(async () => {
    const sym = selectedSymbol ? `?symbol=${selectedSymbol}` : "";
    const symRequired = selectedSymbol || symbolsRef.current[0] || "XRPUSDT";

    const [symRes, sRes, pRes, tRes, dRes, cRes] = await Promise.all([
      api("/symbols"),
      api(`/stats${sym}`),
      api(`/position${sym}`),
      api(`/trades${sym}${sym ? "&" : "?"}limit=50`),
      api(`/daily${sym}`),
      api(`/candles?symbol=${symRequired}&limit=96`),
    ]);

    if (symRes?.symbols) {
      symbolsRef.current = symRes.symbols;
      setSymbols(symRes.symbols);
    }
    if (sRes && sRes.total_trades !== undefined) {
      setStats(sRes);
      setIsLive(true);
      setLastUpdate(new Date());
    }
    if (pRes) {
      setPositions(pRes.positions || []);
      if (pRes.bot) setBotStatus(pRes.bot);
    }
    if (tRes?.trades) setTrades(tRes.trades);
    if (dRes?.daily) setDaily(dRes.daily);
    if (cRes?.candles) setCandles(cRes.candles);
  }, [selectedSymbol]);

  useEffect(() => {
    fetchAll();
    const iv = setInterval(fetchAll, 15000);
    return () => clearInterval(iv);
  }, [fetchAll]);

  /* ── 파생 데이터 ─────────────────────────────────────────── */
  const winRate = stats.total_trades > 0
    ? ((stats.wins / stats.total_trades) * 100).toFixed(0) : "0";

  // 일별 → 날짜순 정렬 (오래된 순)
  const dailyAsc = [...daily].reverse();
  const dailyLabels = dailyAsc.map((d) => fmtDate(d.date));
  const dailyPnls = dailyAsc.map((d) => d.net_pnl || 0);

  // 누적 수익
  const cumData = [];
  let cum = 0;
  dailyAsc.forEach((d) => {
    cum += d.net_pnl || 0;
    cumData.push({ date: fmtDate(d.date), cumPnl: +cum.toFixed(4) });
  });

  // 캔들 차트용
  const candleLabels = candles.map((c) => fmtTime(c.ts));

  /* ── 현재 가격 (봇 상태 또는 마지막 캔들) ──────────────────── */
  const currentPrice = selectedSymbol
    ? (botStatus[`${selectedSymbol}:current_price`] || (candles.length ? candles[candles.length - 1].price : null))
    : (candles.length ? candles[candles.length - 1].price : null);

  /* ── 공통 차트 축 스타일 ─────────────────────────────────── */
  const axisStyle = {
    tick: { fill: "rgba(255,255,255,0.25)", fontSize: 10, fontFamily: "JetBrains Mono" },
    axisLine: false, tickLine: false,
  };

  return (
    <div style={{
      minHeight: "100vh", background: S.bg, color: "#fff",
      fontFamily: S.sans, padding: "28px 20px",
      position: "relative", overflow: "hidden",
    }}>
      {/* BG glow */}
      <div style={{
        position: "fixed", inset: 0, pointerEvents: "none",
        background: "radial-gradient(ellipse 50% 35% at 15% 5%,rgba(99,102,241,0.05) 0%,transparent 70%),radial-gradient(ellipse 40% 40% at 85% 90%,rgba(52,211,153,0.03) 0%,transparent 70%)",
      }} />

      <div style={{ maxWidth: 960, margin: "0 auto", position: "relative" }}>
        {/* ═══ 헤더 ═══════════════════════════════════════════ */}
        <div style={{
          display: "flex", justifyContent: "space-between",
          alignItems: "flex-start", marginBottom: 28, flexWrap: "wrap", gap: 16,
        }}>
          <div>
            <div style={{
              display: "flex", alignItems: "center", gap: 10, marginBottom: 6,
            }}>
              <div style={{
                width: 8, height: 8, borderRadius: "50%",
                background: isLive ? S.green : S.amber,
                boxShadow: isLive
                  ? "0 0 10px rgba(52,211,153,0.5)"
                  : "0 0 10px rgba(245,158,11,0.5)",
                animation: "pulse 2s infinite",
              }} />
              <span style={{
                fontSize: 10, color: S.text3, letterSpacing: 2,
                textTransform: "uppercase", fontFamily: S.mono,
              }}>
                {isLive ? "Live" : "Connecting…"}
                {selectedSymbol
                  ? ` · ${selectedSymbol.replace("USDT", "/USDT")}`
                  : ` · ${symbols.length} symbols`}
                {currentPrice && (
                  <span style={{ color: "rgba(255,255,255,0.5)", marginLeft: 8 }}>
                    {fmt(currentPrice)}
                  </span>
                )}
              </span>
            </div>
            <h1 style={{ fontSize: 28, fontWeight: 700, margin: 0, letterSpacing: -0.8 }}>
              Trading Dashboard
            </h1>
          </div>

          {/* 오픈 포지션 — 복수 표시 */}
          {positions.length > 0 && (
            <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
              {positions.map((pos) => (
                <div key={pos.id} style={{
                  background: "linear-gradient(135deg,rgba(99,102,241,0.08) 0%,rgba(99,102,241,0.02) 100%)",
                  border: "1px solid rgba(99,102,241,0.15)", borderRadius: 14,
                  padding: "12px 18px",
                }}>
                  <div style={{ fontSize: 9, color: S.text3, letterSpacing: 1.2, fontFamily: S.mono, marginBottom: 4 }}>
                    {(pos.symbol || "").replace("USDT", "/USDT")}
                  </div>
                  <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
                    <Badge
                      bg={pos.direction === "SHORT" ? "rgba(239,68,68,0.12)" : "rgba(52,211,153,0.12)"}
                      color={pos.direction === "SHORT" ? S.red : S.green}
                    >
                      {pos.direction} {pos.leverage || 10}x
                    </Badge>
                    <span style={{ fontSize: 14, fontWeight: 700, fontFamily: S.mono }}>
                      {fmt(pos.entry_price)}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* ═══ 심볼 필터 ═══════════════════════════════════════ */}
        <div style={{
          display: "flex", gap: 4, marginBottom: 12,
          background: "rgba(255,255,255,0.02)", borderRadius: 12,
          padding: 4, width: "fit-content",
        }}>
          <button
            onClick={() => setSelectedSymbol(null)}
            style={{
              background: selectedSymbol === null ? "rgba(99,102,241,0.15)" : "transparent",
              border: "none",
              color: selectedSymbol === null ? S.indigo : S.text3,
              padding: "6px 14px", borderRadius: 8, cursor: "pointer",
              fontSize: 11, fontWeight: 600, fontFamily: S.mono,
            }}
          >ALL</button>
          {symbols.map((sym) => (
            <button
              key={sym}
              onClick={() => setSelectedSymbol(sym)}
              style={{
                background: selectedSymbol === sym ? "rgba(99,102,241,0.15)" : "transparent",
                border: "none",
                color: selectedSymbol === sym ? S.indigo : S.text3,
                padding: "6px 14px", borderRadius: 8, cursor: "pointer",
                fontSize: 11, fontWeight: 600, fontFamily: S.mono,
              }}
            >{sym.replace("USDT", "")}</button>
          ))}
        </div>

        {/* ═══ 탭 ═════════════════════════════════════════════ */}
        <div style={{
          display: "flex", gap: 4, marginBottom: 24,
          background: "rgba(255,255,255,0.02)", borderRadius: 12,
          padding: 4, width: "fit-content",
        }}>
          {TABS.map((t) => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              style={{
                background: tab === t.id ? "rgba(255,255,255,0.08)" : "transparent",
                border: "none",
                color: tab === t.id ? "#fff" : S.text3,
                padding: "8px 18px", borderRadius: 9, cursor: "pointer",
                fontSize: 12, fontWeight: 500, fontFamily: S.sans,
                transition: "all 0.15s",
              }}
            >
              <span style={{ marginRight: 6, fontSize: 10 }}>{t.icon}</span>
              {t.label}
            </button>
          ))}
        </div>

        {/* ═══ OVERVIEW ═══════════════════════════════════════ */}
        {tab === "overview" && (
          <div>
            {/* Stats */}
            <div style={{
              display: "grid", gridTemplateColumns: "repeat(4,1fr)",
              gap: 10, marginBottom: 24,
            }}>
              <StatCard icon="💰" label="총 수익" value={pnlSign(stats.total_pnl)} sub="USDT" accent="rgba(52,211,153,0.4)" />
              <StatCard icon="📊" label="승률" value={`${winRate}%`} sub={`${stats.wins}W / ${stats.losses}L`} accent="rgba(129,140,248,0.4)" />
              <StatCard icon="⚡" label="총 거래" value={stats.total_trades} sub={`평균 ${fmt(stats.avg_pnl)} USDT`} accent="rgba(251,191,36,0.3)" />
              <StatCard icon="🎯" label="베스트" value={`+${fmt(stats.best_trade)}`} sub={`최저 ${fmt(stats.worst_trade)}`} accent="rgba(99,102,241,0.3)" />
            </div>

            {/* 차트 */}
            <div style={{
              display: "grid", gridTemplateColumns: "1fr 1fr",
              gap: 10, marginBottom: 24,
            }}>
              <ChartBox title="일별 손익">
                <ResponsiveContainer width="100%" height={180}>
                  <BarChart data={dailyAsc.map((d) => ({ date: fmtDate(d.date), pnl: d.net_pnl || 0 }))}>
                    <XAxis dataKey="date" {...axisStyle} />
                    <YAxis {...axisStyle} />
                    <Tooltip content={<ChartTooltip />} />
                    <Bar dataKey="pnl" name="순수익" radius={[5, 5, 0, 0]}>
                      {dailyAsc.map((d, i) => (
                        <Cell key={i} fill={(d.net_pnl || 0) >= 0 ? S.green : S.red} fillOpacity={0.75} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </ChartBox>

              <ChartBox title="누적 수익 곡선">
                <ResponsiveContainer width="100%" height={180}>
                  <AreaChart data={cumData}>
                    <defs>
                      <linearGradient id="gCum" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor={S.indigo} stopOpacity={0.25} />
                        <stop offset="100%" stopColor={S.indigo} stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <XAxis dataKey="date" {...axisStyle} />
                    <YAxis {...axisStyle} />
                    <Tooltip content={<ChartTooltip />} />
                    <Area
                      type="monotone" dataKey="cumPnl" name="누적"
                      stroke={S.indigo} strokeWidth={2} fill="url(#gCum)"
                      dot={{ fill: S.indigo, r: 3.5, strokeWidth: 0 }}
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </ChartBox>
            </div>

            {/* 최근 거래 */}
            <div style={{
              fontSize: 10, color: S.text3, letterSpacing: 1.2,
              fontFamily: S.mono, textTransform: "uppercase", marginBottom: 10,
            }}>
              최근 거래
            </div>
            {trades.length === 0 && (
              <div style={{
                textAlign: "center", color: S.text3, padding: 40,
                fontFamily: S.mono, fontSize: 12,
              }}>
                거래 내역 없음 — 로그 파싱 대기 중
              </div>
            )}
            {trades.slice(0, 3).map((t) => (
              <TradeRow
                key={t.id}
                trade={t}
                isExpanded={expanded === t.id}
                onToggle={() => setExpanded(expanded === t.id ? null : t.id)}
              />
            ))}
            {trades.length > 3 && (
              <div
                onClick={() => setTab("trades")}
                style={{
                  textAlign: "center", padding: 12, color: S.indigo,
                  fontSize: 12, cursor: "pointer", fontFamily: S.mono,
                  background: "rgba(99,102,241,0.04)", borderRadius: 10,
                  marginTop: 6,
                }}
              >
                전체 {trades.length}건 보기 →
              </div>
            )}
          </div>
        )}

        {/* ═══ TRADES ═════════════════════════════════════════ */}
        {tab === "trades" && (
          <div>
            <div style={{
              fontSize: 10, color: S.text3, letterSpacing: 1.2,
              fontFamily: S.mono, textTransform: "uppercase", marginBottom: 12,
            }}>
              전체 거래 내역 ({trades.length}건)
            </div>
            {trades.map((t) => (
              <TradeRow
                key={t.id}
                trade={t}
                isExpanded={expanded === t.id}
                onToggle={() => setExpanded(expanded === t.id ? null : t.id)}
              />
            ))}
          </div>
        )}

        {/* ═══ CHART ══════════════════════════════════════════ */}
        {tab === "chart" && (
          <div>
            <ChartBox title={`${(selectedSymbol || symbols[0] || "XRP").replace("USDT", "")}/USDT 15m 가격`}>
              <ResponsiveContainer width="100%" height={240}>
                <AreaChart data={candles.map((c) => ({ ts: fmtTime(c.ts), price: c.price || c.close }))}>
                  <defs>
                    <linearGradient id="gP" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#6366f1" stopOpacity={0.15} />
                      <stop offset="100%" stopColor="#6366f1" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.03)" />
                  <XAxis dataKey="ts" {...axisStyle} interval="preserveStartEnd" />
                  <YAxis domain={["auto", "auto"]} {...axisStyle} />
                  <Tooltip content={<ChartTooltip />} />
                  <Area
                    type="monotone" dataKey="price" name="가격"
                    stroke="#6366f1" strokeWidth={1.5} fill="url(#gP)" dot={false}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </ChartBox>

            <div style={{
              display: "grid", gridTemplateColumns: "1fr 1fr",
              gap: 10, marginTop: 12,
            }}>
              <ChartBox title="RSI">
                <ResponsiveContainer width="100%" height={150}>
                  <LineChart data={candles.map((c) => ({ ts: fmtTime(c.ts), rsi: c.rsi }))}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.03)" />
                    <XAxis dataKey="ts" {...axisStyle} interval="preserveStartEnd" />
                    <YAxis domain={[0, 100]} {...axisStyle} />
                    <Tooltip content={<ChartTooltip />} />
                    <Line type="monotone" dataKey={() => 70} stroke="rgba(248,113,113,0.2)" strokeDasharray="4 4" dot={false} name="과매수" />
                    <Line type="monotone" dataKey={() => 30} stroke="rgba(139,92,246,0.2)" strokeDasharray="4 4" dot={false} name="과매도" />
                    <Line type="monotone" dataKey="rsi" name="RSI" stroke={S.amber} strokeWidth={1.5} dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              </ChartBox>

              <ChartBox title="ADX">
                <ResponsiveContainer width="100%" height={150}>
                  <AreaChart data={candles.map((c) => ({ ts: fmtTime(c.ts), adx: c.adx }))}>
                    <defs>
                      <linearGradient id="gA" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor={S.green} stopOpacity={0.15} />
                        <stop offset="100%" stopColor={S.green} stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.03)" />
                    <XAxis dataKey="ts" {...axisStyle} interval="preserveStartEnd" />
                    <YAxis {...axisStyle} />
                    <Tooltip content={<ChartTooltip />} />
                    <Line type="monotone" dataKey={() => 25} stroke="rgba(52,211,153,0.3)" strokeDasharray="4 4" dot={false} name="임계값" />
                    <Area type="monotone" dataKey="adx" name="ADX" stroke={S.green} strokeWidth={1.5} fill="url(#gA)" dot={false} />
                  </AreaChart>
                </ResponsiveContainer>
              </ChartBox>
            </div>
          </div>
        )}

        {/* ═══ 푸터 ═══════════════════════════════════════════ */}
        <div style={{
          textAlign: "center", padding: "24px 0 8px", marginTop: 24,
          borderTop: "1px solid rgba(255,255,255,0.03)",
          display: "flex", justifyContent: "center", alignItems: "center", gap: 16,
        }}>
          <span style={{ fontSize: 10, color: "rgba(255,255,255,0.12)", fontFamily: S.mono }}>
            {lastUpdate
              ? `Synced: ${lastUpdate.toLocaleTimeString("ko-KR")} · 15s polling`
              : "API 연결 대기 중…"}
          </span>
          <button
            onClick={async () => {
              if (!confirm("DB를 초기화하고 로그를 처음부터 다시 파싱합니다. 계속할까요?")) return;
              try {
                const r = await fetch("/api/reset", { method: "POST" });
                if (r.ok) { alert("초기화 완료. 잠시 후 데이터가 다시 채워집니다."); location.reload(); }
                else alert("초기화 실패: " + r.statusText);
              } catch (e) { alert("초기화 실패: " + e.message); }
            }}
            style={{
              fontSize: 10, fontFamily: S.mono, padding: "3px 10px",
              background: "rgba(255,255,255,0.04)", color: "rgba(255,255,255,0.2)",
              border: "1px solid rgba(255,255,255,0.06)", borderRadius: 6, cursor: "pointer",
            }}
          >Reset DB</button>
        </div>
      </div>

      <style>{`
        @keyframes pulse { 0%,100% { opacity:1; } 50% { opacity:0.4; } }
        button:hover { filter: brightness(1.1); }
      `}</style>
    </div>
  );
}

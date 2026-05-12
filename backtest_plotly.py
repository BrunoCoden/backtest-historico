#!/usr/bin/env python3
import argparse
import sys
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ==== Indicators (copied from bots; do not import from other repos) ====

def compute_bollinger_bands(
    df: pd.DataFrame,
    length: int,
    mult: float,
    profile: str = "tradingview",
) -> pd.DataFrame:
    if df is None or df.empty:
        idx = df.index if df is not None else None
        return pd.DataFrame(index=idx)

    length = max(int(length), 1)
    mult = float(mult)
    profile = str(profile).strip().lower()
    if profile == "tradingview":
        ddof = 0
        min_periods = length
    elif profile == "legacy":
        ddof = 1
        min_periods = 1
    else:
        raise ValueError(f"Perfil Bollinger no soportado: {profile}")

    close = df["Close"].astype("float64")
    basis = close.rolling(length, min_periods=min_periods).mean()
    deviation = close.rolling(length, min_periods=min_periods).std(ddof=ddof)
    upper = basis + mult * deviation
    lower = basis - mult * deviation

    idx = df.index
    return pd.DataFrame(
        {
            "basis": basis,
            "upper": upper,
            "lower": lower,
            "deviation": deviation,
            "close": close,
        },
        index=idx,
    )


def compute_atr(df: pd.DataFrame, period: int) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype="float64")
    period = max(int(period), 1)
    high = df["High"].astype("float64")
    low = df["Low"].astype("float64")
    close = df["Close"].astype("float64")
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    # RMA (Wilder)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    return atr


def compute_supertrend(df: pd.DataFrame, period: int, factor: float) -> pd.DataFrame:
    if df is None or df.empty:
        idx = df.index if df is not None else None
        return pd.DataFrame(index=idx)

    period = max(int(period), 1)
    factor = float(factor)
    high = df["High"].astype("float64")
    low = df["Low"].astype("float64")
    close = df["Close"].astype("float64")

    atr = compute_atr(df, period)
    hl2 = (high + low) / 2.0
    upper = hl2 + factor * atr
    lower = hl2 - factor * atr

    final_upper = upper.copy()
    final_lower = lower.copy()
    supertrend = pd.Series(index=df.index, dtype="float64")
    direction = pd.Series(index=df.index, dtype="int64")

    for i in range(len(df)):
        if i == 0:
            supertrend.iat[i] = lower.iat[i]
            direction.iat[i] = 1
            continue

        if upper.iat[i] < final_upper.iat[i - 1] or close.iat[i - 1] > final_upper.iat[i - 1]:
            final_upper.iat[i] = upper.iat[i]
        else:
            final_upper.iat[i] = final_upper.iat[i - 1]

        if lower.iat[i] > final_lower.iat[i - 1] or close.iat[i - 1] < final_lower.iat[i - 1]:
            final_lower.iat[i] = lower.iat[i]
        else:
            final_lower.iat[i] = final_lower.iat[i - 1]

        prev_st = supertrend.iat[i - 1]
        if prev_st == final_upper.iat[i - 1]:
            if close.iat[i] <= final_upper.iat[i]:
                supertrend.iat[i] = final_upper.iat[i]
                direction.iat[i] = -1
            else:
                supertrend.iat[i] = final_lower.iat[i]
                direction.iat[i] = 1
        else:
            if close.iat[i] >= final_lower.iat[i]:
                supertrend.iat[i] = final_lower.iat[i]
                direction.iat[i] = 1
            else:
                supertrend.iat[i] = final_upper.iat[i]
                direction.iat[i] = -1

    idx = df.index
    return pd.DataFrame(
        {
            "supertrend": supertrend,
            "direction": direction,
            "upper": final_upper,
            "lower": final_lower,
            "atr": atr,
            "close": close,
        },
        index=idx,
    )


# ==== Backtest engine ====

@dataclass
class Position:
    direction: str  # long/short
    entry_price: float
    entry_time: pd.Timestamp
    sl_price: Optional[float]
    tp_price: Optional[float]
    entry_reason: str


@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: str
    entry_price: float
    exit_price: float
    entry_reason: str
    exit_reason: str
    pnl: float
    pnl_pct: float


@dataclass
class Signal:
    time: pd.Timestamp
    direction: str
    price: float
    reason: str


def _normalize_rule(rule: str) -> str:
    r = rule.strip()
    if r.upper().endswith("T"):
        r = r[:-1] + "min"
    return r


def _find_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    cols_lc = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        hit = cols_lc.get(cand.lower())
        if hit is not None:
            return hit
    return None


def _resample_ohlcv(df_ticks: pd.DataFrame, rule: str, tz: str) -> pd.DataFrame:
    ts_col = _find_col(df_ticks, ["timestamp_ms_utc", "bucket_start_ms_utc"])
    if ts_col is None:
        raise ValueError("Falta columna timestamp_ms_utc/bucket_start_ms_utc")

    ts_raw = df_ticks[ts_col]
    if pd.api.types.is_numeric_dtype(ts_raw):
        dt = pd.to_datetime(ts_raw, unit="ms", utc=True, errors="coerce")
    else:
        dt = pd.to_datetime(ts_raw, utc=True, errors="coerce")
    if dt.isna().all():
        raise ValueError("No se pudo parsear columna temporal a datetime")
    if tz:
        dt = dt.dt.tz_convert(tz)
    df = df_ticks.copy()
    df["dt"] = dt
    df = df.set_index("dt")

    rule = _normalize_rule(rule)

    # Formato tick-level: timestamp + price/qty
    price_col = _find_col(df, ["price"])
    qty_col = _find_col(df, ["qty"])
    if price_col is not None and qty_col is not None:
        ohlc = df[price_col].astype("float64").resample(rule).ohlc()
        vol = df[qty_col].astype("float64").resample(rule).sum()
        out = ohlc.join(vol).dropna()
        out.columns = ["Open", "High", "Low", "Close", "Volume"]
        return out

    # Formato OHLC: timestamp + open/high/low/close/volume
    open_col = _find_col(df, ["Open", "open"])
    high_col = _find_col(df, ["High", "high"])
    low_col = _find_col(df, ["Low", "low"])
    close_col = _find_col(df, ["Close", "close"])
    if None in (open_col, high_col, low_col, close_col):
        raise ValueError("Faltan columnas price/qty o open/high/low/close")

    open_rs = df[open_col].astype("float64").resample(rule).first()
    high_rs = df[high_col].astype("float64").resample(rule).max()
    low_rs = df[low_col].astype("float64").resample(rule).min()
    close_rs = df[close_col].astype("float64").resample(rule).last()
    vol_col = _find_col(df, ["Volume", "volume", "qty"])
    if vol_col is not None:
        vol_rs = df[vol_col].astype("float64").resample(rule).sum()
    else:
        # Compatibilidad con parquets OHLC sin volumen.
        vol_rs = open_rs * 0.0

    out = pd.DataFrame(
        {
            "Open": open_rs,
            "High": high_rs,
            "Low": low_rs,
            "Close": close_rs,
            "Volume": vol_rs,
        }
    ).dropna(subset=["Open", "High", "Low", "Close"])
    return out


def _calc_pnl(direction: str, entry: float, exit: float, notional: float, fee_rate: float) -> tuple[float, float]:
    qty = notional / entry if entry > 0 else 0.0
    if direction == "long":
        gross = (exit - entry) * qty
    else:
        gross = (entry - exit) * qty
    fees = notional * fee_rate * 2
    net = gross - fees
    pnl_pct = net / notional if notional > 0 else 0.0
    return net, pnl_pct


# ==== Signal generators ====


def generate_bollinger_signals(ohlcv: pd.DataFrame, bb: pd.DataFrame, bb_direction: int = 0):
    pending = None
    last_direction = None
    signals: list[Signal] = []

    for i in range(len(ohlcv)):
        ts = ohlcv.index[i]
        close_now = float(ohlcv["Close"].iat[i])
        upper_now = float(bb["upper"].iat[i])
        lower_now = float(bb["lower"].iat[i])
        if np.isnan(close_now) or np.isnan(upper_now) or np.isnan(lower_now):
            continue

        direction = None
        if pending:
            pend_dir = pending["direction"]
            break_ts = pending["break_ts"]
            if pend_dir == "long" and bb_direction != -1:
                if break_ts is not None and ts > break_ts and close_now > lower_now:
                    direction = "long"
                    pending = None
            elif pend_dir == "short" and bb_direction != 1:
                if break_ts is not None and ts > break_ts and close_now < upper_now:
                    direction = "short"
                    pending = None

        if direction is None:
            if close_now < lower_now and bb_direction != -1:
                pending = {"direction": "long", "band": lower_now, "break_ts": ts}
            elif close_now > upper_now and bb_direction != 1:
                pending = {"direction": "short", "band": upper_now, "break_ts": ts}
            continue

        if last_direction == direction:
            continue
        last_direction = direction
        signals.append(Signal(time=ts, direction=direction, price=close_now, reason="bollinger_signal"))

    return signals


def generate_supertrend_signals(ohlcv: pd.DataFrame, st: pd.DataFrame):
    signals: list[Signal] = []
    direction_series = st.get("direction")
    if direction_series is None or direction_series.empty:
        return signals

    for i in range(1, len(ohlcv)):
        prev_dir = int(direction_series.iat[i - 1])
        curr_dir = int(direction_series.iat[i])
        if curr_dir == prev_dir:
            continue
        ts = ohlcv.index[i]
        close_now = float(ohlcv["Close"].iat[i])
        # Estrategia contraria
        direction = "long" if curr_dir < 0 else "short"
        signals.append(Signal(time=ts, direction=direction, price=close_now, reason="supertrend_signal"))

    return signals


# ==== Strategy simulator ====


def run_backtest(
    ohlcv: pd.DataFrame,
    signals: list[Signal],
    entry_mode: str,
    sl_pct: float,
    tp_pct: float | None,
    notional: float,
    fee_rate: float,
    reentry_on_tp: bool,
    same_dir_sl_hook: Optional[Callable[[pd.Timestamp, str, float], None]] = None,
    position_state_hook: Optional[Callable[[pd.Timestamp, Optional[str]], None]] = None,
    intrabar_ohlcv: Optional[pd.DataFrame] = None,
    sl_tp_intrabar: bool = False,
) -> tuple[list[Trade], list[dict]]:
    signals_by_time = {s.time: s for s in signals}
    trades: list[Trade] = []
    markers = []
    position: Optional[Position] = None
    pending_entry: Optional[Signal] = None

    intrabar_enabled = bool(sl_tp_intrabar and intrabar_ohlcv is not None and not intrabar_ohlcv.empty)
    ib_index = None
    ib_low = None
    ib_high = None
    bar_start_pos = None
    bar_end_pos = None
    if intrabar_enabled:
        ib = intrabar_ohlcv[["High", "Low"]].copy()
        ib = ib.sort_index()
        ib_index = ib.index
        ib_ns = ib_index.view("i8")
        ib_low = ib["Low"].to_numpy(dtype="float64")
        ib_high = ib["High"].to_numpy(dtype="float64")

        bar_index = ohlcv.index
        bar_ns = bar_index.view("i8")
        if len(bar_ns) >= 2:
            diffs = np.diff(bar_ns)
            step_ns = int(np.median(diffs[diffs > 0])) if np.any(diffs > 0) else int(pd.Timedelta(minutes=1).value)
        else:
            step_ns = int(pd.Timedelta(minutes=1).value)
        next_bar_ns = np.empty_like(bar_ns)
        if len(bar_ns) >= 2:
            next_bar_ns[:-1] = bar_ns[1:]
        if len(bar_ns) >= 1:
            next_bar_ns[-1] = bar_ns[-1] + step_ns
        bar_start_pos = np.searchsorted(ib_ns, bar_ns, side="left")
        bar_end_pos = np.searchsorted(ib_ns, next_bar_ns, side="left")

    def open_position(sig: Signal, price: float, ts: pd.Timestamp):
        nonlocal position
        if price <= 0:
            return
        sl = price * (1 - sl_pct) if sig.direction == "long" else price * (1 + sl_pct)
        tp = None
        if tp_pct is not None and tp_pct > 0:
            tp = price * (1 + tp_pct) if sig.direction == "long" else price * (1 - tp_pct)
        position = Position(
            direction=sig.direction,
            entry_price=price,
            entry_time=ts,
            sl_price=sl,
            tp_price=tp,
            entry_reason=sig.reason,
        )
        markers.append({"ts": ts, "price": price, "type": f"entry_{sig.direction}"})

    def close_position(exit_price: float, ts: pd.Timestamp, reason: str):
        nonlocal position
        if position is None:
            return
        pnl, pnl_pct = _calc_pnl(position.direction, position.entry_price, exit_price, notional, fee_rate)
        trades.append(
            Trade(
                entry_time=position.entry_time,
                exit_time=ts,
                direction=position.direction,
                entry_price=position.entry_price,
                exit_price=exit_price,
                entry_reason=position.entry_reason,
                exit_reason=reason,
                pnl=pnl,
                pnl_pct=pnl_pct,
            )
        )
        markers.append({"ts": ts, "price": exit_price, "type": f"exit_{position.direction}"})
        position = None

    for i in range(len(ohlcv)):
        ts = ohlcv.index[i]
        row = ohlcv.iloc[i]
        o, h, l, c = float(row.Open), float(row.High), float(row.Low), float(row.Close)

        # 1) ejecutar entrada pendiente al open
        if pending_entry is not None:
            open_position(pending_entry, o, ts)
            pending_entry = None

        # 2) evaluar SL/TP intra-vela
        if position is not None:
            hit_sl = False
            hit_tp = False
            hit_ts = ts

            if intrabar_enabled and ib_index is not None and ib_low is not None and ib_high is not None:
                s = int(bar_start_pos[i])
                e = int(bar_end_pos[i])
                if e > s:
                    sl_hit_idx = None
                    tp_hit_idx = None

                    if position.direction == "long":
                        if position.sl_price is not None:
                            sl_hits = np.flatnonzero(ib_low[s:e] <= position.sl_price)
                            if sl_hits.size:
                                sl_hit_idx = s + int(sl_hits[0])
                        if position.tp_price is not None:
                            tp_hits = np.flatnonzero(ib_high[s:e] >= position.tp_price)
                            if tp_hits.size:
                                tp_hit_idx = s + int(tp_hits[0])
                    else:
                        if position.sl_price is not None:
                            sl_hits = np.flatnonzero(ib_high[s:e] >= position.sl_price)
                            if sl_hits.size:
                                sl_hit_idx = s + int(sl_hits[0])
                        if position.tp_price is not None:
                            tp_hits = np.flatnonzero(ib_low[s:e] <= position.tp_price)
                            if tp_hits.size:
                                tp_hit_idx = s + int(tp_hits[0])

                    if sl_hit_idx is not None or tp_hit_idx is not None:
                        if sl_hit_idx is not None and (tp_hit_idx is None or sl_hit_idx <= tp_hit_idx):
                            hit_sl = True
                            hit_ts = ib_index[sl_hit_idx]
                        else:
                            hit_tp = True
                            hit_ts = ib_index[tp_hit_idx]
                else:
                    # Fallback a high/low de la vela si no hay sub-velas en el tramo.
                    if position.direction == "long":
                        if position.sl_price is not None and l <= position.sl_price:
                            hit_sl = True
                        if position.tp_price is not None and h >= position.tp_price:
                            hit_tp = True
                    else:
                        if position.sl_price is not None and h >= position.sl_price:
                            hit_sl = True
                        if position.tp_price is not None and l <= position.tp_price:
                            hit_tp = True
            else:
                if position.direction == "long":
                    if position.sl_price is not None and l <= position.sl_price:
                        hit_sl = True
                    if position.tp_price is not None and h >= position.tp_price:
                        hit_tp = True
                else:
                    if position.sl_price is not None and h >= position.sl_price:
                        hit_sl = True
                    if position.tp_price is not None and l <= position.tp_price:
                        hit_tp = True

            if hit_sl or hit_tp:
                # Prioridad: SL primero (conservador)
                if hit_sl:
                    exit_price = position.sl_price
                    close_position(exit_price, hit_ts, "stop_loss")
                    # flip automático al SL
                    flip_dir = "short" if position is None else None
                    if flip_dir is None:
                        # position ya está None, definimos flip por exit_reason
                        pass
                    flip_dir = "short" if (trades[-1].direction == "long") else "long"
                    if position is None:
                        flip_sig = Signal(time=hit_ts, direction=flip_dir, price=exit_price, reason="stop_loss_reversal")
                        # abrir inmediatamente al precio del SL
                        open_position(flip_sig, exit_price, hit_ts)
                else:
                    exit_price = position.tp_price
                    close_position(exit_price, hit_ts, "take_profit")
                    if reentry_on_tp:
                        flip_dir = "short" if (trades[-1].direction == "long") else "long"
                        flip_sig = Signal(time=hit_ts, direction=flip_dir, price=exit_price, reason="tp_reversal")
                        open_position(flip_sig, exit_price, hit_ts)
                if position_state_hook is not None:
                    position_state_hook(ts, None if position is None else position.direction)
                continue

        # 3) señal al cierre de la vela
        sig = signals_by_time.get(ts)
        if sig:
            if position is None:
                if entry_mode == "close":
                    open_position(sig, c, ts)
                else:
                    pending_entry = sig
            else:
                if position.direction == sig.direction:
                    # actualizar SL/TP con el precio de la señal
                    anchor = c
                    new_sl = anchor * (1 - sl_pct) if sig.direction == "long" else anchor * (1 + sl_pct)
                    position.sl_price = new_sl
                    if same_dir_sl_hook is not None:
                        same_dir_sl_hook(ts, sig.direction, float(new_sl))
                    if tp_pct is not None and tp_pct > 0:
                        position.tp_price = anchor * (1 + tp_pct) if sig.direction == "long" else anchor * (1 - tp_pct)
                else:
                    # cerrar y abrir nueva
                    close_position(c, ts, "signal")
                    if entry_mode == "close":
                        open_position(sig, c, ts)
                    else:
                        pending_entry = sig

        if position_state_hook is not None:
            position_state_hook(ts, None if position is None else position.direction)

    return trades, markers


# ==== CLI ====

def main() -> int:
    parser = argparse.ArgumentParser(description="Backtest + Plotly para estrategias (Bollinger / Supertrend)")
    parser.add_argument("parquet_path", help="Ruta al parquet de aggTrades")
    parser.add_argument("--strategy", required=True, choices=["bollinger", "supertrend", "supertrend2"], help="Estrategia")
    parser.add_argument("--tf", default="30T", help="Timeframe (ej: 30T, 1H)")
    parser.add_argument("--entry", default="close", choices=["close", "next_open"], help="Entrada")
    parser.add_argument("--start", default="", help="Inicio (ISO, opcional)")
    parser.add_argument("--end", default="", help="Fin (ISO, opcional)")
    parser.add_argument("--notional", type=float, default=30.0, help="Notional fijo")
    parser.add_argument("--fee", type=float, default=0.0004, help="Fee rate (ej: 0.0004)")
    parser.add_argument("--out", default="plot.html", help="Salida HTML")
    parser.add_argument("--out-trades", default="trades.csv", help="CSV de trades")
    parser.add_argument("--tz", default="America/Argentina/Buenos_Aires", help="Timezone")
    parser.add_argument("--bb-length", type=int, default=20)
    parser.add_argument("--bb-mult", type=float, default=2.0)
    parser.add_argument("--bb-profile", choices=["tradingview", "legacy"], default="tradingview")
    parser.add_argument("--bb-direction", type=int, default=0)
    parser.add_argument("--st-period", type=int, default=10)
    parser.add_argument("--st-factor", type=float, default=3.0)
    parser.add_argument("--sl", type=float, default=0.02, help="Stop loss pct")
    parser.add_argument("--tp", type=float, default=0.0, help="Take profit pct (0 = sin TP)")
    args = parser.parse_args()

    p = Path(args.parquet_path)
    if not p.exists():
        print(f"No existe: {p}", file=sys.stderr)
        return 1

    df_ticks = pd.read_parquet(p)
    ohlcv = _resample_ohlcv(df_ticks, args.tf, args.tz)
    if args.start:
        ohlcv = ohlcv[ohlcv.index >= pd.Timestamp(args.start)]
    if args.end:
        ohlcv = ohlcv[ohlcv.index <= pd.Timestamp(args.end)]
    if ohlcv.empty:
        print("No hay datos en el rango seleccionado.", file=sys.stderr)
        return 1

    # build signals + overlays
    overlays = []
    if args.strategy == "bollinger":
        bb = compute_bollinger_bands(ohlcv, args.bb_length, args.bb_mult, profile=args.bb_profile)
        overlays.append(("upper", bb["upper"]))
        overlays.append(("lower", bb["lower"]))
        overlays.append(("basis", bb["basis"]))
        signals = generate_bollinger_signals(ohlcv, bb, args.bb_direction)
        tp_pct = None  # Bollinger actual: sin TP en thresholds
        trades, markers = run_backtest(
            ohlcv, signals, args.entry, args.sl, tp_pct, args.notional, args.fee, reentry_on_tp=True
        )
    elif args.strategy == "supertrend":
        st = compute_supertrend(ohlcv, args.st_period, args.st_factor)
        overlays.append(("supertrend", st["supertrend"]))
        signals = generate_supertrend_signals(ohlcv, st)
        tp_pct = None
        trades, markers = run_backtest(
            ohlcv, signals, args.entry, args.sl, tp_pct, args.notional, args.fee, reentry_on_tp=True
        )
    else:  # supertrend2
        st = compute_supertrend(ohlcv, args.st_period, args.st_factor)
        overlays.append(("supertrend", st["supertrend"]))
        signals = generate_supertrend_signals(ohlcv, st)
        tp_pct = args.tp if args.tp > 0 else None
        trades, markers = run_backtest(
            ohlcv, signals, args.entry, args.sl, tp_pct, args.notional, args.fee, reentry_on_tp=False
        )

    # Export trades
    trades_df = pd.DataFrame([t.__dict__ for t in trades])
    trades_df.to_csv(args.out_trades, index=False)

    # Plot
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=[0.75, 0.25])
    fig.add_trace(
        go.Candlestick(
            x=ohlcv.index,
            open=ohlcv["Open"],
            high=ohlcv["High"],
            low=ohlcv["Low"],
            close=ohlcv["Close"],
            name="OHLC",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(go.Bar(x=ohlcv.index, y=ohlcv["Volume"], name="Volume"), row=2, col=1)

    for name, series in overlays:
        fig.add_trace(go.Scatter(x=series.index, y=series.values, mode="lines", name=name), row=1, col=1)

    # markers
    for m in markers:
        color = "green" if "entry_long" in m["type"] else "red" if "entry_short" in m["type"] else "orange"
        fig.add_trace(
            go.Scatter(
                x=[m["ts"]],
                y=[m["price"]],
                mode="markers",
                marker=dict(size=8, color=color),
                name=m["type"],
                showlegend=False,
            ),
            row=1,
            col=1,
        )

    fig.update_layout(
        title=f"{args.strategy} | tf={args.tf} | entry={args.entry}",
        xaxis_rangeslider_visible=True,
        xaxis=dict(
            rangeselector=dict(
                buttons=[
                    dict(count=7, label="7d", step="day", stepmode="backward"),
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=3, label="3m", step="month", stepmode="backward"),
                    dict(step="all", label="all"),
                ]
            )
        ),
    )

    div_id = "chart"
    chart_html = fig.to_html(full_html=False, include_plotlyjs="cdn", div_id=div_id)

    trades_json = []
    for t in trades:
        trades_json.append(
            {
                "entry_time": t.entry_time.isoformat() if hasattr(t.entry_time, "isoformat") else str(t.entry_time),
                "exit_time": t.exit_time.isoformat() if hasattr(t.exit_time, "isoformat") else str(t.exit_time),
                "direction": t.direction,
                "pnl": t.pnl,
                "pnl_pct": t.pnl_pct,
            }
        )

    range_min = ohlcv.index.min().isoformat()
    range_max = ohlcv.index.max().isoformat()
    title = f"{args.strategy} | tf={args.tf} | entry={args.entry}"

    html = """<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<title>__TITLE__</title>
</head>
<body style="margin:0; font-family:Arial, sans-serif; background:#111; color:#eee;">
  <div style="padding:10px 14px; border-bottom:1px solid #333;">
    <div style="font-weight:bold; font-size:16px;">__TITLE__</div>
    <div id="metrics" style="margin-top:6px; font-size:14px;"></div>
  </div>
  __CHART__
<script>
const trades = __TRADES__;
const rangeMin = new Date("__RANGE_MIN__");
const rangeMax = new Date("__RANGE_MAX__");
const metricsEl = document.getElementById('metrics');

function inRange(tr, start, end) {
  const entry = new Date(tr.entry_time);
  const exit = new Date(tr.exit_time);
  return entry <= end && exit >= start;
}

function updateMetrics(start, end) {
  let count = 0;
  let wins = 0;
  let pnl = 0;
  let pnlPct = 0;
  for (const tr of trades) {
    if (!inRange(tr, start, end)) continue;
    count += 1;
    pnl += tr.pnl;
    pnlPct += tr.pnl_pct;
    if (tr.pnl > 0) wins += 1;
  }
  const losses = count - wins;
  const winrate = count > 0 ? (wins / count * 100).toFixed(2) : '0.00';
  metricsEl.textContent = \"Rango: \" + start.toISOString() + \" → \" + end.toISOString() + \" | Trades: \" + count +
    \" | Wins: \" + wins + \" | Losses: \" + losses + \" | Winrate: \" + winrate + \"% | PnL: \" + pnl.toFixed(2) +
    \" | PnL%: \" + (pnlPct*100).toFixed(2) + \"%\";
}

const chartDiv = document.getElementById('__DIV_ID__');
chartDiv.on('plotly_relayout', (evt) => {
  const r0 = evt['xaxis.range[0]'];
  const r1 = evt['xaxis.range[1]'];
  if (r0 && r1) {
    updateMetrics(new Date(r0), new Date(r1));
  }
});

updateMetrics(rangeMin, rangeMax);
</script>
</body>
</html>"""

    html = (
        html.replace("__TITLE__", title)
        .replace("__CHART__", chart_html)
        .replace("__TRADES__", json.dumps(trades_json))
        .replace("__RANGE_MIN__", range_min)
        .replace("__RANGE_MAX__", range_max)
        .replace("__DIV_ID__", div_id)
    )

    out_path = Path(args.out)
    out_path.write_text(html, encoding="utf-8")
    print(f"OK -> {out_path}")
    print(f"Trades -> {args.out_trades}")
    print(f"Total trades: {len(trades)}")
    if trades:
        total_pnl = sum(t.pnl for t in trades)
        total_pct = sum(t.pnl_pct for t in trades)
        print(f"PnL total: {total_pnl:.2f} USDT | {total_pct*100:.2f}%")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

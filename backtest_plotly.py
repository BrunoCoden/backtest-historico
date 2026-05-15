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
    profit_lock_done: bool = False


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


@dataclass
class PendingRangeOrder:
    side: int  # 1 long, -1 short
    price: float
    bar_i: int
    order_type: str


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


def compute_range3_channels(
    ohlcv: pd.DataFrame,
    lookback_bars: int,
    pct_upper: float,
    pct_middle: float,
    pct_lower: float,
) -> pd.DataFrame:
    lookback_bars = max(int(lookback_bars), 2)
    sum_pct = float(pct_upper) + float(pct_middle) + float(pct_lower)
    safe_sum = sum_pct if sum_pct != 0 else 1.0
    w_upper = float(pct_upper) / safe_sum
    w_middle = float(pct_middle) / safe_sum

    high = ohlcv["High"].astype("float64")
    low = ohlcv["Low"].astype("float64")
    max_line = high.rolling(lookback_bars, min_periods=lookback_bars).max()
    min_line = low.rolling(lookback_bars, min_periods=lookback_bars).min()
    range_size = max_line - min_line
    maxfloor = max_line - range_size * w_upper
    minroof = maxfloor - range_size * w_middle
    return pd.DataFrame(
        {
            "max": max_line,
            "maxfloor": maxfloor,
            "minroof": minroof,
            "min": min_line,
            "range": range_size,
        },
        index=ohlcv.index,
    )


def _range3_bb_raw_signals(
    ohlcv: pd.DataFrame,
    bb: pd.DataFrame,
    signal_type: str,
    avoid_repeated: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    close = ohlcv["Close"].to_numpy(dtype="float64")
    high = ohlcv["High"].to_numpy(dtype="float64")
    low = ohlcv["Low"].to_numpy(dtype="float64")
    upper = bb["upper"].to_numpy(dtype="float64")
    lower = bb["lower"].to_numpy(dtype="float64")
    n = len(ohlcv)
    lower_raw = np.zeros(n, dtype=bool)
    upper_raw = np.zeros(n, dtype=bool)
    if signal_type == "Cruce de cierre":
        for i in range(1, n):
            if np.isnan(lower[i]) or np.isnan(lower[i - 1]) or np.isnan(close[i]) or np.isnan(close[i - 1]):
                continue
            lower_raw[i] = close[i - 1] <= lower[i - 1] and close[i] > lower[i]
            if np.isnan(upper[i]) or np.isnan(upper[i - 1]):
                continue
            upper_raw[i] = close[i - 1] >= upper[i - 1] and close[i] < upper[i]
    elif signal_type == "Toque simple":
        lower_raw = (low <= lower) & ~np.isnan(lower)
        upper_raw = (high >= upper) & ~np.isnan(upper)
    else:
        lower_raw = (low <= lower) & (close > lower) & ~np.isnan(lower)
        upper_raw = (high >= upper) & (close < upper) & ~np.isnan(upper)

    if avoid_repeated:
        lower_sig = lower_raw & ~np.r_[False, lower_raw[:-1]]
        upper_sig = upper_raw & ~np.r_[False, upper_raw[:-1]]
    else:
        lower_sig = lower_raw
        upper_sig = upper_raw
    return lower_sig, upper_sig, lower_sig | upper_sig


def run_range3_bb_backtest(
    ohlcv: pd.DataFrame,
    notional: float,
    fee_rate: float,
    lookback_bars: int = 200,
    pct_upper: float = 25.0,
    pct_middle: float = 50.0,
    pct_lower: float = 25.0,
    bb_length: int = 20,
    bb_mult: float = 2.0,
    signal_type: str = "Mecha + cierre",
    avoid_repeated: bool = True,
    classify_with: str = "Mecha",
    ambiguous_priority: str = "Ignorar",
    new_extreme_bars: int = 3,
    pending_order_type: str = "Stop en banda",
    max_pending_bars: int = 0,
    replace_pending_opposite: bool = True,
    update_pending_only_new_extreme: bool = True,
    use_stop_loss_pct: bool = True,
    stop_loss_pct: float = 0.02,
    use_opposite_take_profit: bool = False,
    entry_mode: str = "next_open",
    intrabar_ohlcv: Optional[pd.DataFrame] = None,
) -> tuple[list[Trade], list[dict], pd.DataFrame, pd.DataFrame]:
    channels = compute_range3_channels(ohlcv, lookback_bars, pct_upper, pct_middle, pct_lower)
    bb = compute_bollinger_bands(ohlcv, bb_length, bb_mult, profile="tradingview")
    lower_sig, upper_sig, bb_sig = _range3_bb_raw_signals(ohlcv, bb, signal_type, avoid_repeated)

    trades: list[Trade] = []
    markers: list[dict] = []
    position: Optional[Position] = None
    pending_order: Optional[PendingRangeOrder] = None
    pending_market: Optional[Signal] = None

    high = ohlcv["High"].to_numpy(dtype="float64")
    low = ohlcv["Low"].to_numpy(dtype="float64")
    close = ohlcv["Close"].to_numpy(dtype="float64")
    open_ = ohlcv["Open"].to_numpy(dtype="float64")
    idx = ohlcv.index
    max_line = channels["max"].to_numpy(dtype="float64")
    maxfloor = channels["maxfloor"].to_numpy(dtype="float64")
    minroof = channels["minroof"].to_numpy(dtype="float64")
    min_line = channels["min"].to_numpy(dtype="float64")
    range_size = channels["range"].to_numpy(dtype="float64")

    intrabar_enabled = bool(intrabar_ohlcv is not None and not intrabar_ohlcv.empty)
    ib_index = None
    ib_low = None
    ib_high = None
    bar_start_pos = None
    bar_end_pos = None
    if intrabar_enabled:
        ib = intrabar_ohlcv[["High", "Low"]].copy().sort_index()
        ib_index = ib.index
        ib_ns = ib_index.view("i8")
        ib_low = ib["Low"].to_numpy(dtype="float64")
        ib_high = ib["High"].to_numpy(dtype="float64")

        bar_ns = ohlcv.index.view("i8")
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

    def open_position(direction: str, price: float, ts: pd.Timestamp, reason: str):
        nonlocal position
        if price <= 0:
            return
        if position is not None:
            if position.direction == direction:
                return
            close_position(price, ts, "reverse_signal")
        sl = None
        if use_stop_loss_pct:
            sl = price * (1 - stop_loss_pct) if direction == "long" else price * (1 + stop_loss_pct)
        position = Position(
            direction=direction,
            entry_price=price,
            entry_time=ts,
            sl_price=sl,
            tp_price=None,
            entry_reason=reason,
        )
        markers.append({"ts": ts, "price": price, "type": f"entry_{direction}"})

    def pending_hit(po: PendingRangeOrder, hi: float, lo: float) -> bool:
        if po.side == 1:
            return hi >= po.price if po.order_type == "Stop en banda" else lo <= po.price
        return lo <= po.price if po.order_type == "Stop en banda" else hi >= po.price

    def cancel_pending():
        nonlocal pending_order
        pending_order = None

    def evaluate_execution_path(i: int, hi: float, lo: float, event_ts: pd.Timestamp, allow_pending_fill: bool):
        nonlocal pending_order
        if pending_order is not None and allow_pending_fill and i > pending_order.bar_i and pending_hit(pending_order, hi, lo):
            side = pending_order.side
            price = pending_order.price
            reason = "range3_pending_long" if side == 1 else "range3_pending_short"
            markers.append({"ts": event_ts, "price": price, "type": "pending_fill_long" if side == 1 else "pending_fill_short"})
            pending_order = None
            open_position("long" if side == 1 else "short", price, event_ts, reason)

        if position is None:
            return

        exit_price = None
        reason = ""
        if use_stop_loss_pct and position.sl_price is not None:
            if position.direction == "long" and lo <= position.sl_price:
                exit_price = position.sl_price
                reason = "stop_loss"
            elif position.direction == "short" and hi >= position.sl_price:
                exit_price = position.sl_price
                reason = "stop_loss"

        if exit_price is None and use_opposite_take_profit and not np.isnan(maxfloor[i]) and not np.isnan(minroof[i]):
            if position.direction == "long" and hi >= maxfloor[i]:
                exit_price = maxfloor[i]
                reason = "tp_opposite_zone"
            elif position.direction == "short" and lo <= minroof[i]:
                exit_price = minroof[i]
                reason = "tp_opposite_zone"

        if exit_price is not None:
            close_position(float(exit_price), event_ts, reason)

    for i in range(len(ohlcv)):
        ts = idx[i]
        o, h, l, c = open_[i], high[i], low[i], close[i]

        if pending_market is not None:
            open_position(pending_market.direction, o, ts, pending_market.reason)
            pending_market = None

        used_intrabar = False
        if intrabar_enabled and ib_index is not None and ib_low is not None and ib_high is not None:
            s = int(bar_start_pos[i])
            e = int(bar_end_pos[i])
            if e > s:
                used_intrabar = True
                for j in range(s, e):
                    evaluate_execution_path(i, float(ib_high[j]), float(ib_low[j]), ib_index[j], allow_pending_fill=True)

        if not used_intrabar:
            evaluate_execution_path(i, h, l, ts, allow_pending_fill=True)

        if np.isnan(range_size[i]) or range_size[i] <= 0:
            continue

        nuevo_max = high[i] >= max_line[i]
        nuevo_min = low[i] <= min_line[i]
        start_i = max(0, i - max(1, int(new_extreme_bars)) + 1)
        hubo_nuevo_max = bool(np.any(high[start_i : i + 1] >= max_line[start_i : i + 1]))
        hubo_nuevo_min = bool(np.any(low[start_i : i + 1] <= min_line[start_i : i + 1]))
        hubo_nuevo_extremo = hubo_nuevo_max or hubo_nuevo_min
        hay_nuevo_extremo_ahora = nuevo_max or nuevo_min

        ref_short = high[i] if classify_with == "Mecha" else close[i]
        ref_long = low[i] if classify_with == "Mecha" else close[i]
        in_short_zone = ref_short <= max_line[i] and ref_short >= maxfloor[i]
        in_long_zone = ref_long >= min_line[i] and ref_long <= minroof[i]
        ambiguous = bool(bb_sig[i] and in_short_zone and in_long_zone)
        short_signal = bool(bb_sig[i] and ((in_short_zone and not in_long_zone) or (ambiguous and ambiguous_priority == "Priorizar SHORT")))
        long_signal = bool(bb_sig[i] and ((in_long_zone and not in_short_zone) or (ambiguous and ambiguous_priority == "Priorizar LONG")))

        long_pendiente = long_signal and hubo_nuevo_extremo
        short_pendiente = short_signal and hubo_nuevo_extremo
        long_directo = long_signal and not hubo_nuevo_extremo
        short_directo = short_signal and not hubo_nuevo_extremo

        if long_directo or short_directo:
            cancel_pending()
            direction = "long" if long_directo else "short"
            sig = Signal(time=ts, direction=direction, price=c, reason="range3_direct")
            markers.append({"ts": ts, "price": c, "type": f"signal_{direction}"})
            if entry_mode == "close":
                open_position(direction, c, ts, "range3_direct")
            else:
                pending_market = sig

        if long_pendiente:
            if replace_pending_opposite or pending_order is None or pending_order.side != -1:
                pending_order = PendingRangeOrder(side=1, price=float(minroof[i]), bar_i=i, order_type=pending_order_type)
                markers.append({"ts": ts, "price": float(minroof[i]), "type": "pending_set_long"})

        if short_pendiente:
            if replace_pending_opposite or pending_order is None or pending_order.side != 1:
                pending_order = PendingRangeOrder(side=-1, price=float(maxfloor[i]), bar_i=i, order_type=pending_order_type)
                markers.append({"ts": ts, "price": float(maxfloor[i]), "type": "pending_set_short"})

        if (
            update_pending_only_new_extreme
            and pending_order is not None
            and hay_nuevo_extremo_ahora
            and i > pending_order.bar_i
        ):
            if pending_order.side == 1:
                pending_order.price = float(minroof[i])
                pending_order.bar_i = i
                markers.append({"ts": ts, "price": pending_order.price, "type": "pending_update_long"})
            else:
                pending_order.price = float(maxfloor[i])
                pending_order.bar_i = i
                markers.append({"ts": ts, "price": pending_order.price, "type": "pending_update_short"})

        if max_pending_bars > 0 and pending_order is not None and i - pending_order.bar_i > max_pending_bars:
            markers.append({"ts": ts, "price": pending_order.price, "type": "pending_cancel"})
            pending_order = None

    return trades, markers, channels, bb


def run_range3_bb_lock_backtest(
    ohlcv: pd.DataFrame,
    notional: float,
    fee_rate: float,
    lookback_bars: int = 200,
    pct_upper: float = 25.0,
    pct_middle: float = 50.0,
    pct_lower: float = 25.0,
    bb_length: int = 20,
    bb_mult: float = 2.0,
    signal_type: str = "Mecha + cierre",
    avoid_repeated: bool = True,
    classify_with: str = "Mecha",
    ambiguous_priority: str = "Ignorar",
    previous_extreme_bars: int = 3,
    pending_order_type: str = "Stop en banda",
    stop_loss_pct: float = 0.02,
    profit_lock_trigger_pct: float = 0.03,
    profit_lock_sl_pct: float = 0.005,
    use_trailing_stop: bool = False,
    trailing_step_pct: float = 0.01,
    intrabar_ohlcv: Optional[pd.DataFrame] = None,
    active_start: Optional[pd.Timestamp] = None,
    active_end: Optional[pd.Timestamp] = None,
) -> tuple[list[Trade], list[dict], pd.DataFrame, pd.DataFrame]:
    channels = compute_range3_channels(ohlcv, lookback_bars, pct_upper, pct_middle, pct_lower)
    bb = compute_bollinger_bands(ohlcv, bb_length, bb_mult, profile="tradingview")
    lower_sig, upper_sig, bb_sig = _range3_bb_raw_signals(ohlcv, bb, signal_type, avoid_repeated)

    trades: list[Trade] = []
    markers: list[dict] = []
    position: Optional[Position] = None
    pending_order: Optional[PendingRangeOrder] = None

    high = ohlcv["High"].to_numpy(dtype="float64")
    low = ohlcv["Low"].to_numpy(dtype="float64")
    close = ohlcv["Close"].to_numpy(dtype="float64")
    idx = ohlcv.index
    if len(idx) >= 2:
        idx_diffs = idx.to_series().diff().dropna()
        idx_step = idx_diffs.median() if not idx_diffs.empty else pd.Timedelta(minutes=1)
    else:
        idx_step = pd.Timedelta(minutes=1)
    signal_idx = list(idx[1:]) + ([idx[-1] + idx_step] if len(idx) else [])
    max_line = channels["max"].to_numpy(dtype="float64")
    maxfloor = channels["maxfloor"].to_numpy(dtype="float64")
    minroof = channels["minroof"].to_numpy(dtype="float64")
    min_line = channels["min"].to_numpy(dtype="float64")
    range_size = channels["range"].to_numpy(dtype="float64")

    intrabar_enabled = bool(intrabar_ohlcv is not None and not intrabar_ohlcv.empty)
    ib_index = None
    ib_low = None
    ib_high = None
    bar_start_pos = None
    bar_end_pos = None
    if intrabar_enabled:
        ib = intrabar_ohlcv[["High", "Low"]].copy().sort_index()
        ib_index = ib.index
        ib_ns = ib_index.view("i8")
        ib_low = ib["Low"].to_numpy(dtype="float64")
        ib_high = ib["High"].to_numpy(dtype="float64")

        bar_ns = ohlcv.index.view("i8")
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

    def open_position(direction: str, price: float, ts: pd.Timestamp, reason: str):
        nonlocal position
        if price <= 0:
            return
        if position is not None:
            if position.direction == direction:
                return
            close_position(price, ts, "reverse_signal")
        sl = price * (1 - stop_loss_pct) if direction == "long" else price * (1 + stop_loss_pct)
        position = Position(
            direction=direction,
            entry_price=price,
            entry_time=ts,
            sl_price=sl,
            tp_price=None,
            entry_reason=reason,
            profit_lock_done=False,
        )
        markers.append({"ts": ts, "price": price, "type": f"entry_{direction}"})

    def pending_hit(po: PendingRangeOrder, hi: float, lo: float) -> bool:
        if po.side == 1:
            return hi >= po.price if po.order_type == "Stop en banda" else lo <= po.price
        return lo <= po.price if po.order_type == "Stop en banda" else hi >= po.price

    def evaluate_execution_path(i: int, hi: float, lo: float, event_ts: pd.Timestamp, allow_pending_fill: bool):
        nonlocal pending_order, position
        if pending_order is not None and allow_pending_fill and i > pending_order.bar_i and pending_hit(pending_order, hi, lo):
            side = pending_order.side
            price = pending_order.price
            reason = "range3_pending_long" if side == 1 else "range3_pending_short"
            markers.append({"ts": event_ts, "price": price, "type": "pending_fill_long" if side == 1 else "pending_fill_short"})
            pending_order = None
            open_position("long" if side == 1 else "short", price, event_ts, reason)

        if position is None:
            return

        if use_trailing_stop:
            step_pct = max(float(trailing_step_pct), 1e-12)
            if position.direction == "long":
                steps = int(np.floor(max(0.0, (hi / position.entry_price - 1.0)) / step_pct))
                new_sl = position.entry_price * (1 - stop_loss_pct + steps * step_pct)
                if steps > 0 and (position.sl_price is None or new_sl > position.sl_price):
                    position.sl_price = new_sl
                    markers.append({"ts": event_ts, "price": position.sl_price, "type": "trail_sl_long"})
            elif position.direction == "short":
                steps = int(np.floor(max(0.0, (1.0 - lo / position.entry_price)) / step_pct))
                new_sl = position.entry_price * (1 + stop_loss_pct - steps * step_pct)
                if steps > 0 and (position.sl_price is None or new_sl < position.sl_price):
                    position.sl_price = new_sl
                    markers.append({"ts": event_ts, "price": position.sl_price, "type": "trail_sl_short"})
        elif not position.profit_lock_done:
            if position.direction == "long" and hi >= position.entry_price * (1 + profit_lock_trigger_pct):
                position.sl_price = position.entry_price * (1 + profit_lock_sl_pct)
                position.profit_lock_done = True
                markers.append({"ts": event_ts, "price": position.sl_price, "type": "profit_lock_long"})
            elif position.direction == "short" and lo <= position.entry_price * (1 - profit_lock_trigger_pct):
                position.sl_price = position.entry_price * (1 - profit_lock_sl_pct)
                position.profit_lock_done = True
                markers.append({"ts": event_ts, "price": position.sl_price, "type": "profit_lock_short"})

        if position.sl_price is None:
            return

        if position.direction == "long" and lo <= position.sl_price:
            reason = "trailing_stop" if use_trailing_stop else ("profit_lock_sl" if position.profit_lock_done else "stop_loss")
            close_position(float(position.sl_price), event_ts, reason)
        elif position.direction == "short" and hi >= position.sl_price:
            reason = "trailing_stop" if use_trailing_stop else ("profit_lock_sl" if position.profit_lock_done else "stop_loss")
            close_position(float(position.sl_price), event_ts, reason)

    def valid_signal(i: int) -> int:
        if np.isnan(range_size[i]) or range_size[i] <= 0:
            return 0
        ref_short = high[i] if classify_with == "Mecha" else close[i]
        ref_long = low[i] if classify_with == "Mecha" else close[i]
        in_short_zone = ref_short <= max_line[i] and ref_short >= maxfloor[i]
        in_long_zone = ref_long >= min_line[i] and ref_long <= minroof[i]
        close_in_short_zone = close[i] <= max_line[i] and close[i] >= maxfloor[i]
        close_in_long_zone = close[i] >= min_line[i] and close[i] <= minroof[i]
        ambiguous = bool(bb_sig[i] and in_short_zone and in_long_zone)
        short_signal = bool(
            upper_sig[i]
            and close_in_short_zone
            and ((in_short_zone and not in_long_zone) or (ambiguous and ambiguous_priority == "Priorizar SHORT"))
        )
        long_signal = bool(
            lower_sig[i]
            and close_in_long_zone
            and ((in_long_zone and not in_short_zone) or (ambiguous and ambiguous_priority == "Priorizar LONG"))
        )
        if long_signal and not short_signal:
            return 1
        if short_signal and not long_signal:
            return -1
        return 0

    for i in range(len(ohlcv)):
        ts = idx[i]
        h, l, c = high[i], low[i], close[i]

        used_intrabar = False
        if intrabar_enabled and ib_index is not None and ib_low is not None and ib_high is not None:
            s = int(bar_start_pos[i])
            e = int(bar_end_pos[i])
            if e > s:
                used_intrabar = True
                for j in range(s, e):
                    evaluate_execution_path(i, float(ib_high[j]), float(ib_low[j]), ib_index[j], allow_pending_fill=True)

        if not used_intrabar:
            evaluate_execution_path(i, h, l, ts, allow_pending_fill=True)

        if pending_order is not None and i > pending_order.bar_i:
            new_max_now = bool(high[i] >= max_line[i])
            new_min_now = bool(low[i] <= min_line[i])
            if (pending_order.side == -1 and new_max_now) or (pending_order.side == 1 and new_min_now):
                markers.append(
                    {
                        "ts": ts,
                        "price": pending_order.price,
                        "type": "pending_cancel_new_extreme_short" if pending_order.side == -1 else "pending_cancel_new_extreme_long",
                    }
                )
                pending_order = None

        side = valid_signal(i)
        if side == 0:
            continue
        signal_ts = signal_idx[i]
        if active_start is not None and signal_ts < active_start:
            continue
        if active_end is not None and signal_ts > active_end:
            continue

        start_i = max(0, i - max(0, int(previous_extreme_bars)))
        recent_max = bool(np.any(high[start_i : i + 1] >= max_line[start_i : i + 1]))
        recent_min = bool(np.any(low[start_i : i + 1] <= min_line[start_i : i + 1]))
        recent_extreme = recent_max or recent_min

        # Si hay una pendiente y aparece señal opuesta, se descarta.
        if pending_order is not None:
            if pending_order.side == side:
                if recent_extreme:
                    price = float(minroof[i]) if side == 1 else float(maxfloor[i])
                    if not np.isnan(price) and price > 0:
                        pending_order = PendingRangeOrder(side=side, price=price, bar_i=i, order_type=pending_order_type)
                        markers.append({"ts": signal_ts, "price": price, "type": "pending_set_long" if side == 1 else "pending_set_short"})
                else:
                    pending_order = None
                    open_position("long" if side == 1 else "short", c, signal_ts, "range3_consecutive_close")
            continue

        if position is not None:
            current_side = 1 if position.direction == "long" else -1
            if side != current_side:
                if recent_extreme:
                    price = float(minroof[i]) if side == 1 else float(maxfloor[i])
                    if not np.isnan(price) and price > 0:
                        pending_order = PendingRangeOrder(side=side, price=price, bar_i=i, order_type=pending_order_type)
                        markers.append({"ts": signal_ts, "price": price, "type": "pending_set_long" if side == 1 else "pending_set_short"})
                else:
                    open_position("long" if side == 1 else "short", c, signal_ts, "range3_reverse_signal")
            continue

        if recent_extreme:
            price = float(minroof[i]) if side == 1 else float(maxfloor[i])
            if not np.isnan(price) and price > 0:
                pending_order = PendingRangeOrder(side=side, price=price, bar_i=i, order_type=pending_order_type)
                markers.append({"ts": signal_ts, "price": price, "type": "pending_set_long" if side == 1 else "pending_set_short"})
        else:
            open_position("long" if side == 1 else "short", c, signal_ts, "range3_direct_close")

    return trades, markers, channels, bb


def run_range3_extremes_backtest(
    ohlcv: pd.DataFrame,
    notional: float,
    fee_rate: float,
    lookback_bars: int = 200,
    pct_upper: float = 25.0,
    pct_middle: float = 50.0,
    pct_lower: float = 25.0,
    pending_order_type: str = "Stop en banda",
    replace_pending_opposite: bool = True,
    use_stop_loss_pct: bool = True,
    stop_loss_pct: float = 0.02,
    take_profit_pct: float = 0.05,
    intrabar_ohlcv: Optional[pd.DataFrame] = None,
) -> tuple[list[Trade], list[dict], pd.DataFrame]:
    channels = compute_range3_channels(ohlcv, lookback_bars, pct_upper, pct_middle, pct_lower)

    trades: list[Trade] = []
    markers: list[dict] = []
    position: Optional[Position] = None
    pending_order: Optional[PendingRangeOrder] = None

    high = ohlcv["High"].to_numpy(dtype="float64")
    low = ohlcv["Low"].to_numpy(dtype="float64")
    idx = ohlcv.index
    max_line = channels["max"].to_numpy(dtype="float64")
    maxfloor = channels["maxfloor"].to_numpy(dtype="float64")
    minroof = channels["minroof"].to_numpy(dtype="float64")
    min_line = channels["min"].to_numpy(dtype="float64")
    range_size = channels["range"].to_numpy(dtype="float64")

    intrabar_enabled = bool(intrabar_ohlcv is not None and not intrabar_ohlcv.empty)
    ib_index = None
    ib_low = None
    ib_high = None
    bar_start_pos = None
    bar_end_pos = None
    if intrabar_enabled:
        ib = intrabar_ohlcv[["High", "Low"]].copy().sort_index()
        ib_index = ib.index
        ib_ns = ib_index.view("i8")
        ib_low = ib["Low"].to_numpy(dtype="float64")
        ib_high = ib["High"].to_numpy(dtype="float64")

        bar_ns = ohlcv.index.view("i8")
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

    def open_position(direction: str, price: float, ts: pd.Timestamp, reason: str):
        nonlocal position
        if price <= 0:
            return
        if position is not None:
            if position.direction == direction:
                return
            close_position(price, ts, "reverse_order")
        sl = None
        if use_stop_loss_pct:
            sl = price * (1 - stop_loss_pct) if direction == "long" else price * (1 + stop_loss_pct)
        tp = None
        if take_profit_pct > 0:
            tp = price * (1 + take_profit_pct) if direction == "long" else price * (1 - take_profit_pct)
        position = Position(
            direction=direction,
            entry_price=price,
            entry_time=ts,
            sl_price=sl,
            tp_price=tp,
            entry_reason=reason,
        )
        markers.append({"ts": ts, "price": price, "type": f"entry_{direction}"})

    def pending_hit(po: PendingRangeOrder, hi: float, lo: float) -> bool:
        if po.side == 1:
            return hi >= po.price if po.order_type == "Stop en banda" else lo <= po.price
        return lo <= po.price if po.order_type == "Stop en banda" else hi >= po.price

    def set_pending(side: int, price: float, bar_i: int, ts: pd.Timestamp):
        nonlocal pending_order
        if np.isnan(price) or price <= 0:
            return
        if position is not None:
            current_side = 1 if position.direction == "long" else -1
            if side == current_side:
                return
        if (
            pending_order is not None
            and pending_order.side != side
            and not replace_pending_opposite
        ):
            return
        pending_order = PendingRangeOrder(side=side, price=float(price), bar_i=bar_i, order_type=pending_order_type)
        markers.append(
            {
                "ts": ts,
                "price": float(price),
                "type": "pending_set_long" if side == 1 else "pending_set_short",
            }
        )

    def evaluate_execution_path(i: int, hi: float, lo: float, event_ts: pd.Timestamp, allow_pending_fill: bool):
        nonlocal pending_order
        if pending_order is not None and allow_pending_fill and i > pending_order.bar_i and pending_hit(pending_order, hi, lo):
            side = pending_order.side
            price = pending_order.price
            reason = "range3_extreme_long" if side == 1 else "range3_extreme_short"
            markers.append({"ts": event_ts, "price": price, "type": "pending_fill_long" if side == 1 else "pending_fill_short"})
            pending_order = None
            open_position("long" if side == 1 else "short", price, event_ts, reason)

        if position is None:
            return

        exit_price = None
        reason = ""
        if use_stop_loss_pct and position.sl_price is not None:
            if position.direction == "long" and lo <= position.sl_price:
                exit_price = position.sl_price
                reason = "stop_loss"
            elif position.direction == "short" and hi >= position.sl_price:
                exit_price = position.sl_price
                reason = "stop_loss"

        if exit_price is None and position.tp_price is not None:
            if position.direction == "long" and hi >= position.tp_price:
                exit_price = position.tp_price
                reason = "take_profit"
            elif position.direction == "short" and lo <= position.tp_price:
                exit_price = position.tp_price
                reason = "take_profit"

        if exit_price is not None:
            close_position(float(exit_price), event_ts, reason)

    for i in range(len(ohlcv)):
        ts = idx[i]
        h, l = high[i], low[i]

        used_intrabar = False
        if intrabar_enabled and ib_index is not None and ib_low is not None and ib_high is not None:
            s = int(bar_start_pos[i])
            e = int(bar_end_pos[i])
            if e > s:
                used_intrabar = True
                for j in range(s, e):
                    evaluate_execution_path(i, float(ib_high[j]), float(ib_low[j]), ib_index[j], allow_pending_fill=True)

        if not used_intrabar:
            evaluate_execution_path(i, h, l, ts, allow_pending_fill=True)

        if np.isnan(range_size[i]) or range_size[i] <= 0:
            continue

        new_max = high[i] >= max_line[i]
        new_min = low[i] <= min_line[i]
        if new_max:
            set_pending(-1, float(maxfloor[i]), i, ts)
        if new_min:
            set_pending(1, float(minroof[i]), i, ts)

    return trades, markers, channels


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
    parser = argparse.ArgumentParser(description="Backtest + Plotly para estrategias")
    parser.add_argument("parquet_path", help="Ruta al parquet de aggTrades")
    parser.add_argument(
        "--strategy",
        required=True,
        choices=["bollinger", "supertrend", "supertrend2", "range3_bb", "range3_extremes", "range3_bb_lock"],
        help="Estrategia",
    )
    parser.add_argument("--tf", default="30T", help="Timeframe (ej: 30T, 1H)")
    parser.add_argument("--entry", default="close", choices=["close", "next_open"], help="Entrada")
    parser.add_argument("--start", default="", help="Inicio (ISO, opcional)")
    parser.add_argument("--end", default="", help="Fin (ISO, opcional)")
    parser.add_argument("--notional", type=float, default=30.0, help="Notional fijo")
    parser.add_argument("--fee", type=float, default=0.0004, help="Fee rate (ej: 0.0004)")
    parser.add_argument("--out", default="plot.html", help="Salida HTML")
    parser.add_argument("--out-trades", default="trades.csv", help="CSV de trades")
    parser.add_argument("--tz", default="America/Argentina/Buenos_Aires", help="Timezone")
    parser.add_argument(
        "--intrabar-parquet",
        default="",
        help="Parquet intrabar opcional para ejecucion tick/1s dentro de cada vela principal",
    )
    parser.add_argument(
        "--intrabar-tf",
        default="1s",
        help="Timeframe para resamplear el parquet intrabar (ej: 1s, 5s, 1min)",
    )
    parser.add_argument("--bb-length", type=int, default=20)
    parser.add_argument("--bb-mult", type=float, default=2.0)
    parser.add_argument("--bb-profile", choices=["tradingview", "legacy"], default="tradingview")
    parser.add_argument("--bb-direction", type=int, default=0)
    parser.add_argument("--st-period", type=int, default=10)
    parser.add_argument("--st-factor", type=float, default=3.0)
    parser.add_argument("--sl", type=float, default=0.02, help="Stop loss pct")
    parser.add_argument("--tp", type=float, default=0.0, help="Take profit pct (0 = sin TP)")
    parser.add_argument("--range-lookback", type=int, default=200)
    parser.add_argument("--range-pct-upper", type=float, default=25.0)
    parser.add_argument("--range-pct-middle", type=float, default=50.0)
    parser.add_argument("--range-pct-lower", type=float, default=25.0)
    parser.add_argument(
        "--range-bb-signal-type",
        choices=["Mecha + cierre", "Cruce de cierre", "Toque simple"],
        default="Mecha + cierre",
    )
    parser.add_argument("--range-allow-repeated-bb", action="store_true")
    parser.add_argument("--range-classify-with", choices=["Mecha", "Close"], default="Mecha")
    parser.add_argument(
        "--range-ambiguous-priority",
        choices=["Ignorar", "Priorizar SHORT", "Priorizar LONG"],
        default="Ignorar",
    )
    parser.add_argument("--range-new-extreme-bars", type=int, default=3)
    parser.add_argument(
        "--range-pending-order-type",
        choices=["Stop en banda", "Limit en banda"],
        default="Stop en banda",
    )
    parser.add_argument("--range-max-pending-bars", type=int, default=0)
    parser.add_argument("--range-keep-opposite-pending", action="store_true")
    parser.add_argument("--range-update-pending-every-bar", action="store_true")
    parser.add_argument("--range-disable-sl", action="store_true")
    parser.add_argument("--range-use-opposite-tp", action="store_true")
    parser.add_argument("--range-extremes-disable-tp", action="store_true")
    parser.add_argument("--profit-lock-trigger", type=float, default=0.03)
    parser.add_argument("--profit-lock-sl", type=float, default=0.005)
    parser.add_argument("--range-use-trailing-stop", action="store_true")
    parser.add_argument("--range-trailing-step", type=float, default=0.01)
    args = parser.parse_args()

    p = Path(args.parquet_path)
    if not p.exists():
        print(f"No existe: {p}", file=sys.stderr)
        return 1

    df_ticks = pd.read_parquet(p)
    ohlcv = _resample_ohlcv(df_ticks, args.tf, args.tz)
    start_ts = pd.Timestamp(args.start) if args.start else None
    end_ts = pd.Timestamp(args.end) if args.end else None
    if args.strategy == "range3_bb_lock" and start_ts is not None:
        warmup_bars = max(args.range_lookback + args.bb_length + args.range_new_extreme_bars + 20, 260)
        warmup_start = start_ts - pd.tseries.frequencies.to_offset(_normalize_rule(args.tf)) * warmup_bars
        ohlcv = ohlcv[ohlcv.index >= warmup_start]
    elif start_ts is not None:
        ohlcv = ohlcv[ohlcv.index >= start_ts]
    if end_ts is not None:
        ohlcv = ohlcv[ohlcv.index <= end_ts]
    if ohlcv.empty:
        print("No hay datos en el rango seleccionado.", file=sys.stderr)
        return 1

    intrabar_ohlcv = None
    if args.intrabar_parquet:
        ib_path = Path(args.intrabar_parquet)
        if not ib_path.exists():
            print(f"No existe intrabar parquet: {ib_path}", file=sys.stderr)
            return 1
        df_intrabar = pd.read_parquet(ib_path)
        intrabar_ohlcv = _resample_ohlcv(df_intrabar, args.intrabar_tf, args.tz)
        if args.start:
            intrabar_ohlcv = intrabar_ohlcv[intrabar_ohlcv.index >= pd.Timestamp(args.start)]
        if args.end:
            intrabar_ohlcv = intrabar_ohlcv[intrabar_ohlcv.index <= pd.Timestamp(args.end)]
        if intrabar_ohlcv.empty:
            print("No hay datos intrabar en el rango seleccionado.", file=sys.stderr)
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
    elif args.strategy == "supertrend2":
        st = compute_supertrend(ohlcv, args.st_period, args.st_factor)
        overlays.append(("supertrend", st["supertrend"]))
        signals = generate_supertrend_signals(ohlcv, st)
        tp_pct = args.tp if args.tp > 0 else None
        trades, markers = run_backtest(
            ohlcv, signals, args.entry, args.sl, tp_pct, args.notional, args.fee, reentry_on_tp=False
        )
    elif args.strategy == "range3_bb":
        trades, markers, channels, bb = run_range3_bb_backtest(
            ohlcv=ohlcv,
            notional=args.notional,
            fee_rate=args.fee,
            lookback_bars=args.range_lookback,
            pct_upper=args.range_pct_upper,
            pct_middle=args.range_pct_middle,
            pct_lower=args.range_pct_lower,
            bb_length=args.bb_length,
            bb_mult=args.bb_mult,
            signal_type=args.range_bb_signal_type,
            avoid_repeated=not args.range_allow_repeated_bb,
            classify_with=args.range_classify_with,
            ambiguous_priority=args.range_ambiguous_priority,
            new_extreme_bars=args.range_new_extreme_bars,
            pending_order_type=args.range_pending_order_type,
            max_pending_bars=args.range_max_pending_bars,
            replace_pending_opposite=not args.range_keep_opposite_pending,
            update_pending_only_new_extreme=not args.range_update_pending_every_bar,
            use_stop_loss_pct=not args.range_disable_sl,
            stop_loss_pct=args.sl,
            use_opposite_take_profit=args.range_use_opposite_tp,
            entry_mode=args.entry,
            intrabar_ohlcv=intrabar_ohlcv,
        )
        overlays.extend(
            [
                ("range_max", channels["max"]),
                ("range_maxfloor", channels["maxfloor"]),
                ("range_minroof", channels["minroof"]),
                ("range_min", channels["min"]),
                ("bb_upper", bb["upper"]),
                ("bb_lower", bb["lower"]),
                ("bb_basis", bb["basis"]),
            ]
        )
    elif args.strategy == "range3_bb_lock":
        trades, markers, channels, bb = run_range3_bb_lock_backtest(
            ohlcv=ohlcv,
            notional=args.notional,
            fee_rate=args.fee,
            lookback_bars=args.range_lookback,
            pct_upper=args.range_pct_upper,
            pct_middle=args.range_pct_middle,
            pct_lower=args.range_pct_lower,
            bb_length=args.bb_length,
            bb_mult=args.bb_mult,
            signal_type=args.range_bb_signal_type,
            avoid_repeated=not args.range_allow_repeated_bb,
            classify_with=args.range_classify_with,
            ambiguous_priority=args.range_ambiguous_priority,
            previous_extreme_bars=args.range_new_extreme_bars,
            pending_order_type=args.range_pending_order_type,
            stop_loss_pct=args.sl,
            profit_lock_trigger_pct=args.profit_lock_trigger,
            profit_lock_sl_pct=args.profit_lock_sl,
            use_trailing_stop=args.range_use_trailing_stop,
            trailing_step_pct=args.range_trailing_step,
            intrabar_ohlcv=intrabar_ohlcv,
            active_start=start_ts,
            active_end=end_ts,
        )
        overlays.extend(
            [
                ("range_max", channels["max"]),
                ("range_maxfloor", channels["maxfloor"]),
                ("range_minroof", channels["minroof"]),
                ("range_min", channels["min"]),
                ("bb_upper", bb["upper"]),
                ("bb_lower", bb["lower"]),
                ("bb_basis", bb["basis"]),
            ]
        )
    else:  # range3_extremes
        trades, markers, channels = run_range3_extremes_backtest(
            ohlcv=ohlcv,
            notional=args.notional,
            fee_rate=args.fee,
            lookback_bars=args.range_lookback,
            pct_upper=args.range_pct_upper,
            pct_middle=args.range_pct_middle,
            pct_lower=args.range_pct_lower,
            pending_order_type=args.range_pending_order_type,
            replace_pending_opposite=not args.range_keep_opposite_pending,
            use_stop_loss_pct=not args.range_disable_sl,
            stop_loss_pct=args.sl,
            take_profit_pct=0.0 if args.range_extremes_disable_tp else (args.tp if args.tp > 0 else 0.05),
            intrabar_ohlcv=intrabar_ohlcv,
        )
        overlays.extend(
            [
                ("range_max", channels["max"]),
                ("range_maxfloor", channels["maxfloor"]),
                ("range_minroof", channels["minroof"]),
                ("range_min", channels["min"]),
            ]
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

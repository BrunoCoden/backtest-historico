#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from backtest_plotly import (
    compute_bollinger_bands,
    generate_bollinger_signals,
)


@dataclass
class Position:
    direction: str  # long | short
    entry_price: float
    entry_time_ns: int
    sl_price: float


def _pick_col(df: pd.DataFrame, names: list[str]) -> str | None:
    cols_lc = {c.lower(): c for c in df.columns}
    for n in names:
        if n in df.columns:
            return n
        hit = cols_lc.get(n.lower())
        if hit is not None:
            return hit
    return None


def _load_ticks_window(
    parquet_path: Path,
    ts_start_ba: pd.Timestamp,
    ts_end_ba: pd.Timestamp,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    schema_names = pq.ParquetFile(parquet_path).schema.names
    lc = {n.lower(): n for n in schema_names}
    ts_col = lc.get("timestamp_ms_utc") or lc.get("bucket_start_ms_utc")
    read_cols: list[str] | None = None
    if columns:
        wanted = list(columns)
        if ts_col and ts_col not in wanted:
            wanted.append(ts_col)
        read_cols = [c for c in wanted if c in schema_names]

    if ts_col is None:
        return pd.read_parquet(parquet_path, columns=read_cols)

    filters = [
        (ts_col, ">=", int(ts_start_ba.tz_convert("UTC").timestamp() * 1000)),
        (ts_col, "<=", int(ts_end_ba.tz_convert("UTC").timestamp() * 1000)),
    ]
    return pd.read_parquet(parquet_path, filters=filters, columns=read_cols)


def _calc_pnl_pct(direction: str, entry: float, exit: float, notional: float, fee_rate: float) -> float:
    qty = notional / entry if entry > 0 else 0.0
    if direction == "long":
        gross = (exit - entry) * qty
    else:
        gross = (entry - exit) * qty
    fee = (entry * qty + exit * qty) * fee_rate
    pnl = gross - fee
    return pnl / notional if notional > 0 else 0.0


def _entry_exec_price(direction: str, ref_price: float, spread_bps: float, slippage_bps: float) -> float:
    # Long entra por ask (peor), short por bid (peor).
    spread = spread_bps / 10000.0
    slippage = slippage_bps / 10000.0
    if direction == "long":
        return ref_price * (1.0 + spread / 2.0 + slippage)
    return ref_price * (1.0 - spread / 2.0 - slippage)


def _exit_exec_price(direction: str, ref_price: float, spread_bps: float, slippage_bps: float) -> float:
    # Long sale por bid (peor), short sale por ask (peor).
    spread = spread_bps / 10000.0
    slippage = slippage_bps / 10000.0
    if direction == "long":
        return ref_price * (1.0 - spread / 2.0 - slippage)
    return ref_price * (1.0 + spread / 2.0 + slippage)


def _first_sl_touch_idx(
    low_arr: np.ndarray,
    high_arr: np.ndarray,
    start_pos: int,
    end_pos: int,
    direction: str,
    sl_price: float,
) -> int | None:
    if end_pos <= start_pos:
        return None
    if direction == "long":
        hits = np.flatnonzero(low_arr[start_pos:end_pos] <= sl_price)
    else:
        hits = np.flatnonzero(high_arr[start_pos:end_pos] >= sl_price)
    if hits.size == 0:
        return None
    return start_pos + int(hits[0])


def run(args: argparse.Namespace) -> int:
    tz = "America/Argentina/Buenos_Aires"
    calc_start = pd.Timestamp(args.calc_start)
    calc_end = pd.Timestamp(args.calc_end)
    out_start = pd.Timestamp(args.out_start)
    out_end = pd.Timestamp(args.out_end)
    out_start_ns = int(out_start.tz_convert("UTC").value)
    out_end_ns = int(out_end.tz_convert("UTC").value)

    ticks = _load_ticks_window(
        Path(args.parquet),
        calc_start,
        calc_end,
        columns=["timestamp_ms_utc", "bucket_start_ms_utc", "open", "high", "low", "close", "Open", "High", "Low", "Close"],
    )
    ts_col = _pick_col(ticks, ["timestamp_ms_utc", "bucket_start_ms_utc"])
    open_col = _pick_col(ticks, ["open", "Open"])
    low_col = _pick_col(ticks, ["low", "Low"])
    high_col = _pick_col(ticks, ["high", "High"])
    close_col = _pick_col(ticks, ["close", "Close"])
    if ts_col is None or open_col is None or low_col is None or high_col is None or close_col is None:
        raise RuntimeError("Parquet sin columnas necesarias de tiempo/OHLC.")
    ticks = ticks.sort_values(ts_col).reset_index(drop=True)

    ts_raw = ticks[ts_col]
    if pd.api.types.is_numeric_dtype(ts_raw):
        ts_ms = pd.to_numeric(ts_raw, errors="coerce").to_numpy(dtype="float64")
    else:
        ts_dt_utc = pd.to_datetime(ts_raw, utc=True, errors="coerce")
        ts_ms = (ts_dt_utc.view("i8").astype("float64") / 1_000_000.0)
    good = np.isfinite(ts_ms)
    ts_ms_i64 = ts_ms[good].astype("int64", copy=False)

    open_arr = ticks.loc[good, open_col].to_numpy(dtype="float64")
    high_arr = ticks.loc[good, high_col].to_numpy(dtype="float64")
    low_arr64 = ticks.loc[good, low_col].to_numpy(dtype="float64")
    close_arr = ticks.loc[good, close_col].to_numpy(dtype="float64")

    # Serie 1s para detección de toque SL (trabajar en ns evita DatetimeIndex gigante).
    sec_ns = ts_ms_i64 * 1_000_000
    low_arr = low_arr64.astype("float32")
    high_arr_1s = high_arr.astype("float32")

    # Resample 1h alineado a America/Argentina/Buenos_Aires (UTC-3 fijo).
    bucket_ms = 3_600_000
    offset_ms = -3 * bucket_ms
    hour_bucket_ms = ((ts_ms_i64 + offset_ms) // bucket_ms) * bucket_ms - offset_ms
    grp_open = pd.Series(open_arr).groupby(hour_bucket_ms, sort=True).first()
    grp_high = pd.Series(high_arr).groupby(hour_bucket_ms, sort=True).max()
    grp_low = pd.Series(low_arr64).groupby(hour_bucket_ms, sort=True).min()
    grp_close = pd.Series(close_arr).groupby(hour_bucket_ms, sort=True).last()
    ohlcv_1h = pd.DataFrame(
        {
            "Open": grp_open,
            "High": grp_high,
            "Low": grp_low,
            "Close": grp_close,
            "Volume": 0.0,
        }
    )
    ohlcv_1h.index = pd.to_datetime(ohlcv_1h.index.to_numpy(dtype="int64"), unit="ms", utc=True).tz_convert(tz)

    del ts_ms, ts_ms_i64, open_arr, high_arr, low_arr64, close_arr
    del ticks
    gc.collect()
    ohlcv_1h = ohlcv_1h[(ohlcv_1h.index >= calc_start) & (ohlcv_1h.index <= calc_end)].copy()
    if ohlcv_1h.empty or sec_ns.size == 0:
        raise RuntimeError("No hay datos para el rango solicitado.")

    bb = compute_bollinger_bands(ohlcv_1h, length=args.bb_length, mult=args.bb_mult, profile=args.bb_profile)
    signals = generate_bollinger_signals(ohlcv_1h, bb, bb_direction=args.bb_direction)
    sig_by_ts = {s.time: s for s in signals}

    # Arrays 1s para slicing rápido por hora.
    hour_index = ohlcv_1h.index
    h1_arr = ohlcv_1h["High"].to_numpy(dtype="float64")
    l1_arr = ohlcv_1h["Low"].to_numpy(dtype="float64")
    c_arr = ohlcv_1h["Close"].to_numpy(dtype="float64")
    hour_ns = hour_index.view("i8")
    next_hour_ns = np.empty_like(hour_ns)
    next_hour_ns[:-1] = hour_ns[1:]
    # Última vela: hasta +1h
    next_hour_ns[-1] = hour_ns[-1] + int(pd.Timedelta(hours=1).value)

    hour_start_pos = np.searchsorted(sec_ns, hour_ns, side="left")
    hour_end_pos = np.searchsorted(sec_ns, next_hour_ns, side="left")

    sl_values = [round(float(x), 8) for x in np.arange(args.sl_min, args.sl_max + 1e-12, args.sl_step)]
    results = []

    for sl_pct in sl_values:
        pos: Position | None = None
        total_pct = 0.0
        trades = 0

        for i, ts in enumerate(hour_index):
            h1 = float(h1_arr[i])
            l1 = float(l1_arr[i])
            c = float(c_arr[i])
            s_pos = int(hour_start_pos[i])
            e_pos = int(hour_end_pos[i])

            # 1) Intrabar: sólo SL (no TP), con revisión 1s.
            # Compatibilidad con el motor base: como máximo 1 ejecución por vela.
            if pos is not None:
                # Fast gate: si en toda la vela 1h no hay chance de tocar SL, evita escaneo 1s.
                touch_possible = not (
                    (pos.direction == "long" and l1 > pos.sl_price)
                    or (pos.direction == "short" and h1 < pos.sl_price)
                )
                if touch_possible:
                        hit_idx = _first_sl_touch_idx(
                            low_arr=low_arr,
                            high_arr=high_arr_1s,
                            start_pos=s_pos,
                            end_pos=e_pos,
                            direction=pos.direction,
                            sl_price=pos.sl_price,
                        )
                        if hit_idx is not None:
                            hit_ns = int(sec_ns[hit_idx])
                            exit_ref = pos.sl_price
                            exit_px = _exit_exec_price(pos.direction, exit_ref, args.spread_bps, args.slippage_bps)
                            pnl_pct = _calc_pnl_pct(pos.direction, pos.entry_price, exit_px, args.notional, args.fee_rate) * 100.0
                            if out_start_ns <= hit_ns <= out_end_ns:
                                total_pct += pnl_pct
                            trades += 1

                        # Reversa automática en SL y salto a próxima vela 1h.
                            flip = "short" if pos.direction == "long" else "long"
                            entry_ref = exit_ref
                            entry_px = _entry_exec_price(flip, entry_ref, args.spread_bps, args.slippage_bps)
                            new_sl = entry_px * (1.0 - sl_pct) if flip == "long" else entry_px * (1.0 + sl_pct)
                            pos = Position(direction=flip, entry_price=entry_px, entry_time_ns=hit_ns, sl_price=new_sl)
                            continue

            # 2) Señal al cierre de 1h.
            sig = sig_by_ts.get(ts)
            if sig is None:
                continue

            if pos is None:
                entry_px = _entry_exec_price(sig.direction, c, args.spread_bps, args.slippage_bps)
                sl_price = entry_px * (1.0 - sl_pct) if sig.direction == "long" else entry_px * (1.0 + sl_pct)
                pos = Position(direction=sig.direction, entry_price=entry_px, entry_time_ns=int(ts.value), sl_price=sl_price)
                continue

            if pos.direction == sig.direction:
                # misma dirección: actualizar SL
                anchor = _entry_exec_price(sig.direction, c, args.spread_bps, args.slippage_bps)
                pos.sl_price = anchor * (1.0 - sl_pct) if sig.direction == "long" else anchor * (1.0 + sl_pct)
            else:
                # señal contraria: cerrar y abrir nueva al cierre
                exit_px = _exit_exec_price(pos.direction, c, args.spread_bps, args.slippage_bps)
                pnl_pct = _calc_pnl_pct(pos.direction, pos.entry_price, exit_px, args.notional, args.fee_rate) * 100.0
                if out_start <= ts <= out_end:
                    total_pct += pnl_pct
                trades += 1

                entry_px = _entry_exec_price(sig.direction, c, args.spread_bps, args.slippage_bps)
                sl_price = entry_px * (1.0 - sl_pct) if sig.direction == "long" else entry_px * (1.0 + sl_pct)
                pos = Position(direction=sig.direction, entry_price=entry_px, entry_time_ns=int(ts.value), sl_price=sl_price)

        results.append({"sl_pct": sl_pct * 100.0, "pnl_total_pct": total_pct, "trades": trades})

    res = pd.DataFrame(results).sort_values("pnl_total_pct", ascending=False).reset_index(drop=True)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out_csv, index=False)

    best = res.iloc[0]
    print(f"OK -> {out_csv}")
    print(f"best_sl_pct={float(best['sl_pct']):.4f}")
    print(f"best_pnl_total_pct={float(best['pnl_total_pct']):.6f}")
    print(f"best_trades={int(best['trades'])}")
    print("top10:")
    print(res.head(10).to_string(index=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Barrido de SL para ETHUSDT (Bollinger 1h), con toque de SL validado en velas de 1s."
    )
    parser.add_argument(
        "--parquet",
        default="data/velas crudas/ETHUSDT/ETHUSDT_202101_202602_merged_tmp.parquet",
    )
    parser.add_argument("--calc-start", default="2024-12-01T00:00:00-03:00")
    parser.add_argument("--calc-end", default="2026-02-28T23:59:59-03:00")
    parser.add_argument("--out-start", default="2025-01-01T00:00:00-03:00")
    parser.add_argument("--out-end", default="2026-02-28T23:59:59-03:00")
    parser.add_argument("--notional", type=float, default=30.0)
    parser.add_argument("--fee-rate", type=float, default=0.0004)
    parser.add_argument("--spread-bps", type=float, default=0.0)
    parser.add_argument("--slippage-bps", type=float, default=0.0)
    parser.add_argument("--bb-length", type=int, default=20)
    parser.add_argument("--bb-mult", type=float, default=2.0)
    parser.add_argument("--bb-profile", choices=["tradingview", "legacy"], default="tradingview")
    parser.add_argument("--bb-direction", type=int, default=0)
    parser.add_argument("--sl-min", type=float, default=0.005)   # 0.5%
    parser.add_argument("--sl-max", type=float, default=0.06)    # 6.0%
    parser.add_argument("--sl-step", type=float, default=0.0025) # 0.25%
    parser.add_argument(
        "--out-csv",
        default="data/estadisticas/sl_sweep_eth_1h_intrabar1s_202501_202602_no_tp.csv",
    )
    args = parser.parse_args()

    try:
        return run(args)
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

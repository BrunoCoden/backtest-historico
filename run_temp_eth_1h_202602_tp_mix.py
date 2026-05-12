#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import shutil
from typing import Optional

import numpy as np
import pandas as pd

from backtest_plotly import (
    Signal,
    _calc_pnl,
    _normalize_rule,
    _resample_ohlcv,
    compute_bollinger_bands,
    generate_bollinger_signals,
)
from export_tabla_senales import (
    ENTRY_COL,
    EXIT_COL,
    NEW_SL_COL,
    PNL_CLOSE_COL,
    POS_TREND_COL,
    _append_pnl_total_row,
    _collapse_labels,
    _entry_label,
    _exit_label,
)

TZ_BA = "America/Argentina/Buenos_Aires"


@dataclass
class TempPosition:
    direction: str
    entry_price: float
    entry_time: pd.Timestamp
    sl_price: float
    tp_price: float
    origin: str  # signal|flip
    entry_reason: str
    sl_pct_used: float
    tp_pct_used: float


@dataclass
class TempTrade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: str
    entry_price: float
    exit_price: float
    entry_reason: str
    exit_reason: str
    pnl: float
    pnl_pct: float
    origin: str
    sl_pct_used: float
    tp_pct_used: float


def _parse_local_ts(s: str) -> Optional[pd.Timestamp]:
    txt = (s or "").strip()
    if not txt:
        return None
    ts = pd.Timestamp(txt)
    if ts.tzinfo is None:
        ts = ts.tz_localize(TZ_BA)
    else:
        ts = ts.tz_convert(TZ_BA)
    return ts


def _align_ts_to_tf(ts: pd.Timestamp, tf: str) -> pd.Timestamp:
    return pd.Timestamp(ts).floor(_normalize_rule(tf))


def _normalize_tf_for_pandas(tf: str) -> str:
    return _normalize_rule(tf).replace("H", "h")


def _fmt_pct_tag(v: float) -> str:
    return f"{v * 100.0:.2f}pct"


def _tf_tag(tf: str) -> str:
    return tf.replace(" ", "").replace("/", "_")


def _pnl_folder_suffix(total_pct: float) -> str:
    sign = "pos" if total_pct >= 0 else "neg"
    return f"pnl_{sign}{abs(total_pct):.2f}pct"


def _fmt_ts_local_no_tz(ts: pd.Timestamp) -> str:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize(TZ_BA)
    else:
        t = t.tz_convert(TZ_BA)
    return t.strftime("%y-%m-%d %H:%M:%S")


def _round_cols(df: pd.DataFrame, cols: list[str], decimals: int = 2) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").round(decimals)
    return out


def _append_trades_total_row(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "entry_time" in out.columns:
        out = out[out["entry_time"].astype(str) != "TOTAL PNL %"].copy()
    total_pnl = float(pd.to_numeric(out.get("pnl"), errors="coerce").sum(min_count=1) or 0.0)
    total_pnl_pct = float(pd.to_numeric(out.get("pnl_pct"), errors="coerce").sum(min_count=1) or 0.0)
    row = {c: "" for c in out.columns}
    if "entry_time" in row:
        row["entry_time"] = "TOTAL PNL %"
    if "pnl" in row:
        row["pnl"] = round(total_pnl, 2)
    if "pnl_pct" in row:
        row["pnl_pct"] = round(total_pnl_pct, 2)
    return pd.concat([out, pd.DataFrame([row], columns=out.columns)], ignore_index=True)


def _ensure_unique_dir(path: Path) -> Path:
    if not path.exists():
        return path
    for i in range(1, 1000):
        cand = path.with_name(f"{path.name}_dup{i:02d}")
        if not cand.exists():
            return cand
    raise RuntimeError(f"No se pudo resolver carpeta única para {path}")


def _load_monthly_raw(symbol: str, yyyymm: str, root: Path) -> pd.DataFrame:
    p = root / symbol / f"{symbol}_{yyyymm}_1s_ohlc.parquet"
    if not p.exists():
        raise FileNotFoundError(f"No existe parquet mensual: {p}")
    return pd.read_parquet(p)


def _pick_ts_col(df: pd.DataFrame) -> str:
    cols = {c.lower(): c for c in df.columns}
    ts_col = cols.get("timestamp_ms_utc") or cols.get("bucket_start_ms_utc")
    if ts_col is None:
        raise ValueError("Falta timestamp_ms_utc/bucket_start_ms_utc")
    return ts_col


def _merge_raw(df_a: pd.DataFrame, df_b: pd.DataFrame) -> pd.DataFrame:
    merged = pd.concat([df_a, df_b], ignore_index=True)
    ts_col = _pick_ts_col(merged)
    merged = merged.sort_values(ts_col)
    merged = merged.drop_duplicates(subset=[ts_col], keep="last")
    return merged


def _iter_yyyymm(start_ba: pd.Timestamp, end_ba: pd.Timestamp) -> list[str]:
    y, m = int(start_ba.year), int(start_ba.month)
    ey, em = int(end_ba.year), int(end_ba.month)
    out: list[str] = []
    while (y < ey) or (y == ey and m <= em):
        out.append(f"{y:04d}{m:02d}")
        m += 1
        if m > 12:
            y += 1
            m = 1
    return out


def _load_raw_range(symbol: str, root: Path, calc_start: pd.Timestamp, calc_end: pd.Timestamp) -> tuple[pd.DataFrame, list[str]]:
    months = _iter_yyyymm(calc_start, calc_end)
    chunks: list[pd.DataFrame] = []
    missing: list[str] = []
    for ym in months:
        p = root / symbol / f"{symbol}_{ym}_1s_ohlc.parquet"
        if not p.exists():
            missing.append(str(p))
            continue
        chunks.append(pd.read_parquet(p))
    if missing:
        raise FileNotFoundError("Faltan parquets crudos:\\n" + "\\n".join(missing))
    if not chunks:
        raise ValueError("No se cargaron parquets para el rango solicitado")
    merged = pd.concat(chunks, ignore_index=True)
    ts_col = _pick_ts_col(merged)
    merged = merged.sort_values(ts_col).drop_duplicates(subset=[ts_col], keep="last")
    return merged, months


def _filter_raw_window(df: pd.DataFrame, start_ba: Optional[pd.Timestamp], end_ba: Optional[pd.Timestamp]) -> pd.DataFrame:
    ts_col = _pick_ts_col(df)
    ts_raw = df[ts_col]
    if pd.api.types.is_numeric_dtype(ts_raw):
        dt = pd.to_datetime(ts_raw, unit="ms", utc=True, errors="coerce")
    else:
        dt = pd.to_datetime(ts_raw, utc=True, errors="coerce")
    dt_ba = dt.dt.tz_convert(TZ_BA)
    mask = pd.Series(True, index=df.index)
    if start_ba is not None:
        mask &= dt_ba >= start_ba
    if end_ba is not None:
        mask &= dt_ba <= end_ba
    return df.loc[mask].copy()


def _run_temp_backtest(
    ohlcv_tf: pd.DataFrame,
    signals: list[Signal],
    sl_pct: float,
    tp_signal_pct: float,
    tp_flip_pct: float,
    disable_tp: bool,
    disable_tp_signal: bool,
    disable_tp_flip: bool,
    flip_tp_reversal: bool,
    notional: float,
    fee_rate: float,
    intrabar_ohlcv: Optional[pd.DataFrame],
    use_intrabar_1s: bool,
) -> tuple[list[TempTrade], dict[pd.Timestamp, float], dict[pd.Timestamp, str]]:
    signals_by_time = {s.time: s for s in signals}
    trades: list[TempTrade] = []
    sl_update_by_ts: dict[pd.Timestamp, float] = {}
    position_by_ts: dict[pd.Timestamp, str] = {}

    position: Optional[TempPosition] = None
    wait_next_signal = False

    intrabar_enabled = bool(use_intrabar_1s and intrabar_ohlcv is not None and not intrabar_ohlcv.empty)
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

        bar_index = ohlcv_tf.index
        bar_ns = bar_index.view("i8")
        if len(bar_ns) >= 2:
            diffs = np.diff(bar_ns)
            step_ns = int(np.median(diffs[diffs > 0])) if np.any(diffs > 0) else int(pd.Timedelta(hours=1).value)
        else:
            step_ns = int(pd.Timedelta(minutes=1).value)
        next_bar_ns = np.empty_like(bar_ns)
        if len(bar_ns) >= 2:
            next_bar_ns[:-1] = bar_ns[1:]
        if len(bar_ns) >= 1:
            next_bar_ns[-1] = bar_ns[-1] + step_ns
        bar_start_pos = np.searchsorted(ib_ns, bar_ns, side="left")
        bar_end_pos = np.searchsorted(ib_ns, next_bar_ns, side="left")

    def _tp_for_origin(origin: str) -> float:
        return float(tp_signal_pct if origin == "signal" else tp_flip_pct)

    def _tp_enabled_for_origin(origin: str) -> bool:
        if bool(disable_tp):
            return False
        if origin == "signal" and bool(disable_tp_signal):
            return False
        if origin == "flip" and bool(disable_tp_flip):
            return False
        return True

    def open_position(direction: str, price: float, ts: pd.Timestamp, origin: str, reason: str) -> None:
        nonlocal position
        if price <= 0:
            return
        tp_enabled = _tp_enabled_for_origin(origin)
        tp_pct_used = 0.0 if not tp_enabled else _tp_for_origin(origin)
        sl_price = price * (1 - sl_pct) if direction == "long" else price * (1 + sl_pct)
        tp_price = np.nan if not tp_enabled else (price * (1 + tp_pct_used) if direction == "long" else price * (1 - tp_pct_used))
        position = TempPosition(
            direction=direction,
            entry_price=float(price),
            entry_time=pd.Timestamp(ts),
            sl_price=float(sl_price),
            tp_price=float(tp_price),
            origin=origin,
            entry_reason=reason,
            sl_pct_used=float(sl_pct),
            tp_pct_used=float(tp_pct_used),
        )

    def close_position(exit_price: float, ts: pd.Timestamp, reason: str) -> Optional[TempTrade]:
        nonlocal position
        if position is None:
            return None
        pnl, pnl_pct = _calc_pnl(position.direction, position.entry_price, float(exit_price), notional, fee_rate)
        tr = TempTrade(
            entry_time=position.entry_time,
            exit_time=pd.Timestamp(ts),
            direction=position.direction,
            entry_price=float(position.entry_price),
            exit_price=float(exit_price),
            entry_reason=position.entry_reason,
            exit_reason=reason,
            pnl=float(pnl),
            pnl_pct=float(pnl_pct),
            origin=position.origin,
            sl_pct_used=float(position.sl_pct_used),
            tp_pct_used=float(position.tp_pct_used),
        )
        position = None
        trades.append(tr)
        return tr

    for i in range(len(ohlcv_tf)):
        ts = ohlcv_tf.index[i]
        row = ohlcv_tf.iloc[i]
        h = float(row.High)
        l = float(row.Low)
        c = float(row.Close)

        # 1) Evaluación intravela SL/TP
        if position is not None:
            hit_sl = False
            hit_tp = False
            hit_ts = ts
            tp_active = np.isfinite(float(position.tp_price))

            if intrabar_enabled and ib_index is not None and ib_low is not None and ib_high is not None:
                s = int(bar_start_pos[i])
                e = int(bar_end_pos[i])
                if e > s:
                    sl_hit_idx = None
                    tp_hit_idx = None
                    if position.direction == "long":
                        sl_hits = np.flatnonzero(ib_low[s:e] <= position.sl_price)
                        if sl_hits.size:
                            sl_hit_idx = s + int(sl_hits[0])
                        if tp_active:
                            tp_hits = np.flatnonzero(ib_high[s:e] >= position.tp_price)
                            if tp_hits.size:
                                tp_hit_idx = s + int(tp_hits[0])
                    else:
                        sl_hits = np.flatnonzero(ib_high[s:e] >= position.sl_price)
                        if sl_hits.size:
                            sl_hit_idx = s + int(sl_hits[0])
                        if tp_active:
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
                    if position.direction == "long":
                        hit_sl = l <= position.sl_price
                        hit_tp = bool(tp_active and h >= position.tp_price)
                    else:
                        hit_sl = h >= position.sl_price
                        hit_tp = bool(tp_active and l <= position.tp_price)
            else:
                if position.direction == "long":
                    hit_sl = l <= position.sl_price
                    hit_tp = bool(tp_active and h >= position.tp_price)
                else:
                    hit_sl = h >= position.sl_price
                    hit_tp = bool(tp_active and l <= position.tp_price)

            if hit_sl or hit_tp:
                current_origin = position.origin
                if hit_sl:
                    exit_price = float(position.sl_price)
                    tr = close_position(exit_price, hit_ts, "stop_loss")
                    if tr is not None and current_origin == "signal":
                        flip_dir = "short" if tr.direction == "long" else "long"
                        open_position(flip_dir, exit_price, hit_ts, origin="flip", reason="stop_loss_reversal")
                        wait_next_signal = False
                    else:
                        # Cierre de posición flip: no reentrada automática.
                        wait_next_signal = True
                else:
                    exit_price = float(position.tp_price)
                    tr = close_position(exit_price, hit_ts, "take_profit")
                    if tr is not None and current_origin == "flip" and bool(flip_tp_reversal):
                        flip_dir = "short" if tr.direction == "long" else "long"
                        open_position(flip_dir, exit_price, hit_ts, origin="flip", reason="take_profit_reversal")
                        wait_next_signal = False
                    else:
                        # Tras TP, no reentrada automática: esperar próxima señal.
                        wait_next_signal = True

                position_by_ts[ts] = "" if position is None else ("Long" if position.direction == "long" else "Short")
                continue

        # 2) Señal de Bollinger al cierre de vela
        sig = signals_by_time.get(ts)
        if sig is not None:
            if wait_next_signal and position is None:
                open_position(sig.direction, c, ts, origin="signal", reason=sig.reason)
                wait_next_signal = False
            elif position is None:
                open_position(sig.direction, c, ts, origin="signal", reason=sig.reason)
            else:
                if position.direction == sig.direction:
                    # Mismo lado: actualizamos SL/TP anclado al cierre de señal.
                    tp_enabled = _tp_enabled_for_origin(position.origin)
                    tp_pct_used = 0.0 if not tp_enabled else _tp_for_origin(position.origin)
                    new_sl = c * (1 - sl_pct) if sig.direction == "long" else c * (1 + sl_pct)
                    new_tp = np.nan if not tp_enabled else (c * (1 + tp_pct_used) if sig.direction == "long" else c * (1 - tp_pct_used))
                    position.sl_price = float(new_sl)
                    position.tp_price = float(new_tp)
                    sl_update_by_ts[pd.Timestamp(ts)] = float(new_sl)
                else:
                    old_origin = position.origin
                    _ = close_position(c, ts, "signal")
                    if old_origin == "flip":
                        # Cierre de flip: bloquear hasta la próxima señal.
                        wait_next_signal = True
                    else:
                        open_position(sig.direction, c, ts, origin="signal", reason=sig.reason)
                        wait_next_signal = False

        position_by_ts[ts] = "" if position is None else ("Long" if position.direction == "long" else "Short")

    return trades, sl_update_by_ts, position_by_ts


def _build_output_table(
    ohlcv_tf_calc: pd.DataFrame,
    band_upper: pd.Series,
    band_lower: pd.Series,
    trades: list[TempTrade],
    sl_update_by_ts: dict[pd.Timestamp, float],
    position_by_ts: dict[pd.Timestamp, str],
    tf: str,
    out_start: pd.Timestamp,
    out_end: pd.Timestamp,
) -> pd.DataFrame:
    entries_by_ts: dict[pd.Timestamp, list[str]] = {}
    exits_by_ts: dict[pd.Timestamp, list[str]] = {}
    pnl_close_by_ts: dict[pd.Timestamp, float] = {}

    for tr in trades:
        entry_ts = _align_ts_to_tf(pd.Timestamp(tr.entry_time), tf)
        exit_ts = _align_ts_to_tf(pd.Timestamp(tr.exit_time), tf)
        entries_by_ts.setdefault(entry_ts, []).append(_entry_label(tr.entry_reason))
        exits_by_ts.setdefault(exit_ts, []).append(_exit_label(tr.exit_reason))
        pnl_close_by_ts[exit_ts] = float(pnl_close_by_ts.get(exit_ts, 0.0) + tr.pnl_pct * 100.0)

    df_base = pd.DataFrame(index=ohlcv_tf_calc.index)
    df_base["Fecha"] = df_base.index.strftime("%y-%m-%d")
    df_base["Hora"] = df_base.index.strftime("%H:%M:%S")
    df_base["Precio Maximo Vela"] = ohlcv_tf_calc["High"].astype("float64")
    df_base["Precio minimo Vela"] = ohlcv_tf_calc["Low"].astype("float64")
    df_base["Precio Banda Superior"] = band_upper.reindex(df_base.index).astype("float64")
    df_base["Precio Banda Inferior"] = band_lower.reindex(df_base.index).astype("float64")

    sl_series = pd.Series(sl_update_by_ts, dtype="float64")
    pos_series = pd.Series(position_by_ts, dtype="object")
    pnl_series = pd.Series(pnl_close_by_ts, dtype="float64")

    df_base[NEW_SL_COL] = sl_series.reindex(df_base.index)
    df_base[POS_TREND_COL] = pos_series.reindex(df_base.index).fillna("")
    df_base[ENTRY_COL] = [_collapse_labels(entries_by_ts.get(ts, [])) for ts in df_base.index]
    df_base[EXIT_COL] = [_collapse_labels(exits_by_ts.get(ts, [])) for ts in df_base.index]
    df_base[PNL_CLOSE_COL] = pnl_series.reindex(df_base.index)

    df_out = df_base[(df_base.index >= out_start) & (df_base.index <= out_end)].copy()
    if df_out.empty:
        raise ValueError("No hay velas de salida en el rango solicitado")

    df_out = _append_pnl_total_row(
        df_out,
        pnl_col=PNL_CLOSE_COL,
        label_col="Fecha",
        label_value="TOTAL PNL %",
    )
    df_out = _round_cols(
        df_out,
        cols=[
            "Precio Maximo Vela",
            "Precio minimo Vela",
            "Precio Banda Superior",
            "Precio Banda Inferior",
            NEW_SL_COL,
            PNL_CLOSE_COL,
        ],
        decimals=2,
    )
    return df_out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Corrida temporal Bollinger con flip SL y espera de señal tras cierre de flip/TP."
    )
    parser.add_argument("--symbol", default="ETHUSDT")
    parser.add_argument("--tf", default="1H")
    parser.add_argument("--out-start", default="2026-02-01T00:00:00-03:00")
    parser.add_argument("--out-end", default="2026-02-28T23:59:59-03:00")
    parser.add_argument("--calc-start", default="2026-01-01T00:00:00-03:00")
    parser.add_argument("--calc-end", default="2026-02-28T23:59:59-03:00")
    parser.add_argument("--sl-pct", type=float, default=0.02)
    parser.add_argument("--tp-signal-pct", type=float, default=0.05)
    parser.add_argument("--tp-flip-pct", type=float, default=0.03)
    parser.add_argument(
        "--disable-tp",
        dest="disable_tp",
        action="store_true",
        default=False,
        help="Desactiva TP para todas las posiciones (signal y flip).",
    )
    parser.add_argument(
        "--enable-tp",
        dest="disable_tp",
        action="store_false",
        help="Mantiene TP habilitado (default).",
    )
    parser.add_argument(
        "--disable-tp-signal",
        dest="disable_tp_signal",
        action="store_true",
        default=False,
        help="Desactiva TP solo para posiciones origin=signal.",
    )
    parser.add_argument(
        "--enable-tp-signal",
        dest="disable_tp_signal",
        action="store_false",
        help="Mantiene TP en signal (default).",
    )
    parser.add_argument(
        "--disable-tp-flip",
        dest="disable_tp_flip",
        action="store_true",
        default=False,
        help="Desactiva TP solo para posiciones origin=flip.",
    )
    parser.add_argument(
        "--enable-tp-flip",
        dest="disable_tp_flip",
        action="store_false",
        help="Mantiene TP en flip (default).",
    )
    parser.add_argument(
        "--flip-tp-reverse",
        dest="flip_tp_reverse",
        action="store_true",
        default=False,
        help="Si un trade origin=flip cierra por TP, abre reversa inmediata con origin=flip.",
    )
    parser.add_argument(
        "--no-flip-tp-reverse",
        dest="flip_tp_reverse",
        action="store_false",
        help="Desactiva reversa inmediata al TP de flip (default).",
    )
    parser.add_argument("--fee", type=float, default=0.0004)
    parser.add_argument("--notional", type=float, default=30.0)
    parser.add_argument("--bb-length", type=int, default=20)
    parser.add_argument("--bb-mult", type=float, default=2.0)
    parser.add_argument("--bb-direction", type=int, default=0)
    parser.add_argument("--bb-profile", choices=["tradingview", "legacy"], default="tradingview")
    parser.add_argument("--raw-root", default="data/velas crudas")
    parser.add_argument(
        "--out-root",
        default="data/pruebas combinaciones TPs y SLs",
        help="Carpeta raíz para salidas por defecto cuando no se pasa --out-csv.",
    )
    parser.add_argument(
        "--out-csv",
        default="",
        help="CSV técnico de salida (si vacío, se autogenera dentro de --out-root).",
    )
    parser.add_argument(
        "--out-trades",
        default="",
        help="CSV de trades de auditoría (si vacío, se autogenera junto al técnico).",
    )
    parser.add_argument(
        "--sl-intrabar-1s",
        dest="sl_intrabar_1s",
        action="store_true",
        default=True,
        help="Evalúa SL/TP con velas de 1s (default: true).",
    )
    parser.add_argument(
        "--no-sl-intrabar-1s",
        dest="sl_intrabar_1s",
        action="store_false",
        help="Desactiva evaluación SL/TP intrabar 1s.",
    )
    parser.add_argument(
        "--run-label",
        default="",
        help="Etiqueta opcional estable para identificar la corrida (si no se pasa, usa timestamp).",
    )
    parser.add_argument(
        "--append-pnl-to-folder",
        dest="append_pnl_to_folder",
        action="store_true",
        default=True,
        help="Agrega sufijo pnl_pos/negXX.XXpct al nombre de carpeta autogenerada (default: true).",
    )
    parser.add_argument(
        "--no-append-pnl-to-folder",
        dest="append_pnl_to_folder",
        action="store_false",
        help="No agrega sufijo de PNL al nombre de carpeta autogenerada.",
    )
    parser.add_argument(
        "--atomic-folder",
        dest="atomic_folder",
        action="store_true",
        default=True,
        help="Escribe en carpeta temporal __running y renombra al final (default: true).",
    )
    parser.add_argument(
        "--no-atomic-folder",
        dest="atomic_folder",
        action="store_false",
        help="Escribe directo en carpeta final autogenerada.",
    )

    args = parser.parse_args()

    symbol = str(args.symbol).upper().strip()

    tf = _normalize_tf_for_pandas(str(args.tf))
    out_start = _parse_local_ts(args.out_start)
    out_end = _parse_local_ts(args.out_end)
    calc_start = _parse_local_ts(args.calc_start)
    calc_end = _parse_local_ts(args.calc_end)
    if out_start is None or out_end is None or calc_start is None or calc_end is None:
        raise ValueError("Rangos temporales inválidos")
    if calc_start > calc_end:
        raise ValueError("--calc-start no puede ser mayor que --calc-end")
    if out_start > out_end:
        raise ValueError("--out-start no puede ser mayor que --out-end")
    if out_start < calc_start or out_end > calc_end:
        raise ValueError("La ventana de salida debe quedar dentro de la ventana de cálculo")

    raw_root = Path(args.raw_root)
    raw, loaded_months = _load_raw_range(symbol, raw_root, calc_start, calc_end)
    raw = _filter_raw_window(raw, calc_start, calc_end)

    if raw.empty:
        raise ValueError("No hay datos crudos en el rango de cálculo")

    ohlcv_tf = _resample_ohlcv(raw, tf, TZ_BA)
    ohlcv_tf = ohlcv_tf[(ohlcv_tf.index >= calc_start) & (ohlcv_tf.index <= calc_end)]
    if ohlcv_tf.empty:
        raise ValueError("No hay velas TF en el rango de cálculo")

    ohlcv_1s = _resample_ohlcv(raw, "1s", TZ_BA)
    ohlcv_1s = ohlcv_1s[(ohlcv_1s.index >= calc_start) & (ohlcv_1s.index <= calc_end)]

    bb = compute_bollinger_bands(
        ohlcv_tf,
        length=int(args.bb_length),
        mult=float(args.bb_mult),
        profile=str(args.bb_profile),
    )
    band_upper = bb["upper"]
    band_lower = bb["lower"]
    signals = generate_bollinger_signals(ohlcv_tf, bb, int(args.bb_direction))

    trades, sl_updates, pos_by_ts = _run_temp_backtest(
        ohlcv_tf=ohlcv_tf,
        signals=signals,
        sl_pct=float(args.sl_pct),
        tp_signal_pct=float(args.tp_signal_pct),
        tp_flip_pct=float(args.tp_flip_pct),
        disable_tp=bool(args.disable_tp),
        disable_tp_signal=bool(args.disable_tp_signal),
        disable_tp_flip=bool(args.disable_tp_flip),
        flip_tp_reversal=bool(args.flip_tp_reverse),
        notional=float(args.notional),
        fee_rate=float(args.fee),
        intrabar_ohlcv=ohlcv_1s,
        use_intrabar_1s=bool(args.sl_intrabar_1s),
    )

    df_out = _build_output_table(
        ohlcv_tf_calc=ohlcv_tf,
        band_upper=band_upper,
        band_lower=band_lower,
        trades=trades,
        sl_update_by_ts=sl_updates,
        position_by_ts=pos_by_ts,
        tf=tf,
        out_start=out_start,
        out_end=out_end,
    )

    pnl_total = pd.to_numeric(df_out.loc[df_out["Fecha"].astype(str) == "TOTAL PNL %", PNL_CLOSE_COL], errors="coerce").iloc[-1]

    out_csv: Path
    out_trades: Path
    auto_outputs = not args.out_csv.strip()
    final_dir: Optional[Path] = None
    write_dir: Optional[Path] = None

    if auto_outputs:
        run_id = str(args.run_label).strip() or datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir_name = (
            f"TF_{_tf_tag(tf)}"
            f"_SL_{_fmt_pct_tag(float(args.sl_pct))}"
            f"_TPs_{_fmt_pct_tag(float(args.tp_signal_pct))}"
            f"_TPf_{_fmt_pct_tag(float(args.tp_flip_pct))}"
            f"_{run_id}"
        )
        final_dir_name = base_dir_name
        if bool(args.append_pnl_to_folder):
            final_dir_name = f"{base_dir_name}_{_pnl_folder_suffix(float(pnl_total))}"
        final_dir = _ensure_unique_dir(Path(args.out_root) / final_dir_name)

        if bool(args.atomic_folder):
            running_candidate = final_dir.with_name(f"{final_dir.name}__running")
            if running_candidate.exists():
                shutil.rmtree(running_candidate)
            write_dir = running_candidate
        else:
            write_dir = final_dir
        write_dir.mkdir(parents=True, exist_ok=True)

        out_csv = write_dir / f"{symbol}_{out_start.strftime('%Y%m%d')}_{out_end.strftime('%Y%m%d')}_{_tf_tag(tf)}_bollinger_temp.csv"
        out_trades = write_dir / f"{symbol}_{out_start.strftime('%Y%m%d')}_{out_end.strftime('%Y%m%d')}_{_tf_tag(tf)}_bollinger_temp_trades.csv"
    else:
        out_csv = Path(args.out_csv)
        if args.out_trades.strip():
            out_trades = Path(args.out_trades)
        else:
            out_trades = out_csv.with_name(f"{out_csv.stem}_trades.csv")
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        out_trades.parent.mkdir(parents=True, exist_ok=True)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_csv, index=False)

    trades_df = pd.DataFrame([
        {
            "entry_time": _fmt_ts_local_no_tz(t.entry_time),
            "exit_time": _fmt_ts_local_no_tz(t.exit_time),
            "direction": t.direction,
            "origin": t.origin,
            "entry_reason": t.entry_reason,
            "exit_reason": t.exit_reason,
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "sl_pct_used": t.sl_pct_used,
            "tp_pct_used": t.tp_pct_used,
            "pnl": t.pnl,
            "pnl_pct": t.pnl_pct * 100.0,
        }
        for t in trades
    ])
    trades_df = _round_cols(
        trades_df,
        cols=["entry_price", "exit_price", "pnl", "pnl_pct"],
        decimals=2,
    )
    trades_df = _append_trades_total_row(trades_df)
    out_trades.parent.mkdir(parents=True, exist_ok=True)
    trades_df.to_csv(out_trades, index=False)

    if auto_outputs and final_dir is not None and write_dir is not None and write_dir != final_dir:
        if final_dir.exists():
            raise FileExistsError(f"Carpeta final ya existe: {final_dir}")
        write_dir.rename(final_dir)
        out_csv = final_dir / out_csv.name
        out_trades = final_dir / out_trades.name

    print(f"OK -> {out_csv}")
    print(f"OK (trades) -> {out_trades}")
    print(f"Symbol={symbol} TF={tf} | calc={calc_start}..{calc_end} | out={out_start}..{out_end}")
    print(f"Raw meses cargados: {', '.join(loaded_months)}")
    print(
        f"SL={float(args.sl_pct):.4f} TP_signal={float(args.tp_signal_pct):.4f} "
        f"TP_flip={float(args.tp_flip_pct):.4f} intrabar_1s={bool(args.sl_intrabar_1s)} "
        f"flip_tp_reverse={bool(args.flip_tp_reverse)} disable_tp={bool(args.disable_tp)} "
        f"disable_tp_signal={bool(args.disable_tp_signal)} disable_tp_flip={bool(args.disable_tp_flip)}"
    )
    print(f"Signals={len(signals)} Trades={len(trades)} PNL_total_pct={float(pnl_total):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

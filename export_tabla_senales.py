#!/usr/bin/env python3
import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import pyarrow.parquet as pq

from backtest_plotly import (
    Signal,
    _resample_ohlcv,
    compute_bollinger_bands,
    compute_supertrend,
    generate_bollinger_signals,
    generate_supertrend_signals,
    run_backtest,
)

TZ_BA = "America/Argentina/Buenos_Aires"

BOT_BOLLINGER_DIR = Path("/home/diego/bot")
BOT_DEX_DIR = Path("/home/diego/botDex")
BOT_ST2_DIR = Path("/home/diego/Supertrend2-0")

ENTRY_COL = "Señal de entrada(por SL,TP o estrategia)"
EXIT_COL = "Señal de salida(por SL,TP o estrategia)"
NEW_SL_COL = "Precio Nuevo SL"
POS_TREND_COL = "Tendencia Posicion Actual"
PNL_CLOSE_COL = "PNL Cierre %"
TOUCH_COLS = ["Fecha Toque", "Hora Toque", "Precio Toque", "Tipo Toque", "Nro Toque En Vela"]
PRICE_COLS = [
    "Precio Maximo Vela",
    "Precio minimo Vela",
    "Precio Banda Superior",
    "Precio Banda Inferior",
    "Precio Toque",
    NEW_SL_COL,
    PNL_CLOSE_COL,
]


@dataclass
class _SimPosition:
    direction: str
    sl_price: Optional[float]
    tp_price: Optional[float]


def _normalize_rule(rule: str) -> str:
    r = rule.strip()
    if r.upper().endswith("T"):
        r = r[:-1] + "min"
    return r


def _align_ts_to_tf(ts: pd.Timestamp, tf: str) -> pd.Timestamp:
    return pd.Timestamp(ts).floor(_normalize_rule(tf))


def _read_env_value(env_path: Path, key: str) -> Optional[str]:
    if not env_path.exists():
        return None
    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            if k.strip() == key:
                return v.strip().strip('"').strip("'")
    except Exception:
        return None
    return None


def _strategy_defaults() -> dict:
    defaults = {
        "bollinger": {"sl": 0.02, "tp": 0.0},
        "supertrend": {"sl": 0.02, "tp": 0.0},
        "supertrend2": {"sl": 0.02, "tp": 0.05},
    }

    env_boll = BOT_BOLLINGER_DIR / ".env"
    v = _read_env_value(env_boll, "WATCHER_CONTRA_THRESHOLD_PCT")
    if v:
        try:
            defaults["bollinger"]["sl"] = float(v)
        except Exception:
            pass

    env_dex = BOT_DEX_DIR / ".env"
    v = _read_env_value(env_dex, "STRAT_STOP_LOSS_PCT")
    if v:
        try:
            defaults["supertrend"]["sl"] = float(v)
        except Exception:
            pass

    env_st2 = BOT_ST2_DIR / ".env"
    v = _read_env_value(env_st2, "WATCHER_CONTRA_THRESHOLD_PCT")
    if not v:
        v = _read_env_value(env_st2, "STRAT_STOP_LOSS_PCT")
    if v:
        try:
            defaults["supertrend2"]["sl"] = float(v)
        except Exception:
            pass
    v = _read_env_value(env_st2, "WATCHER_TAKE_PROFIT_PCT")
    if v:
        try:
            defaults["supertrend2"]["tp"] = float(v)
        except Exception:
            pass

    return defaults


def _parse_local_ts(value: str) -> Optional[pd.Timestamp]:
    if not value:
        return None
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return None
    if ts.tzinfo is None:
        return ts.tz_localize(TZ_BA)
    return ts.tz_convert(TZ_BA)


def _find_ts_col(df: pd.DataFrame) -> Optional[str]:
    for c in ("timestamp_ms_utc", "bucket_start_ms_utc"):
        if c in df.columns:
            return c
        for cc in df.columns:
            if cc.lower() == c:
                return cc
    return None


def _filter_ticks_window(
    df_ticks: pd.DataFrame,
    ts_start_ba: Optional[pd.Timestamp],
    ts_end_ba: Optional[pd.Timestamp],
) -> pd.DataFrame:
    if ts_start_ba is None and ts_end_ba is None:
        return df_ticks
    ts_col = _find_ts_col(df_ticks)
    if ts_col is None:
        # Si no encontramos timestamp, devolvemos sin filtrar y dejamos que falle en _resample_ohlcv con mensaje claro.
        return df_ticks

    ts_raw = df_ticks[ts_col]
    if pd.api.types.is_numeric_dtype(ts_raw):
        start_ms = int(ts_start_ba.tz_convert("UTC").timestamp() * 1000) if ts_start_ba is not None else None
        end_ms = int(ts_end_ba.tz_convert("UTC").timestamp() * 1000) if ts_end_ba is not None else None
        mask = pd.Series(True, index=df_ticks.index)
        if start_ms is not None:
            mask &= ts_raw >= start_ms
        if end_ms is not None:
            mask &= ts_raw <= end_ms
        return df_ticks.loc[mask]

    ts_dt = pd.to_datetime(ts_raw, utc=True, errors="coerce")
    start_utc = ts_start_ba.tz_convert("UTC") if ts_start_ba is not None else None
    end_utc = ts_end_ba.tz_convert("UTC") if ts_end_ba is not None else None
    mask = ts_dt.notna()
    if start_utc is not None:
        mask &= ts_dt >= start_utc
    if end_utc is not None:
        mask &= ts_dt <= end_utc
    return df_ticks.loc[mask]


def _load_ticks_with_window(
    parquet_path: Path,
    ts_start_ba: Optional[pd.Timestamp],
    ts_end_ba: Optional[pd.Timestamp],
) -> pd.DataFrame:
    # Fast path: predicate pushdown en parquet por columna temporal.
    # Si falla por engine/formato, hace fallback a lectura completa.
    try:
        schema_names = pq.ParquetFile(parquet_path).schema.names
        lc_map = {n.lower(): n for n in schema_names}
        ts_col = lc_map.get("timestamp_ms_utc") or lc_map.get("bucket_start_ms_utc")
        if ts_col is not None and (ts_start_ba is not None or ts_end_ba is not None):
            filters: list[tuple[str, str, int]] = []
            if ts_start_ba is not None:
                filters.append((ts_col, ">=", int(ts_start_ba.tz_convert("UTC").timestamp() * 1000)))
            if ts_end_ba is not None:
                filters.append((ts_col, "<=", int(ts_end_ba.tz_convert("UTC").timestamp() * 1000)))
            return pd.read_parquet(parquet_path, filters=filters)
    except Exception:
        pass
    return pd.read_parquet(parquet_path)


def _entry_label(reason: str) -> str:
    r = (reason or "").lower()
    if "stop_loss" in r:
        return "SL"
    if "tp" in r or "take_profit" in r:
        return "TP"
    return "estrategia"


def _exit_label(reason: str) -> str:
    r = (reason or "").lower()
    if "stop_loss" in r:
        return "SL"
    if "take_profit" in r or "tp" in r:
        return "TP"
    return "estrategia"


def _collapse_labels(labels: list[str]) -> str:
    if not labels:
        return ""
    out = []
    seen = set()
    for lbl in labels:
        if lbl in seen:
            continue
        seen.add(lbl)
        out.append(lbl)
    return "|".join(out)


def _compute_bollinger_bands_with_source(
    ohlcv: pd.DataFrame,
    length: int,
    mult: float,
    price_source: str,
    profile: str,
) -> pd.DataFrame:
    length = max(int(length), 1)
    profile = str(profile).strip().lower()
    if profile == "tradingview":
        ddof = 0
        min_periods = length
    elif profile == "legacy":
        ddof = 1
        min_periods = 1
    else:
        raise ValueError(f"bb_profile no soportado: {profile}")

    if price_source == "close":
        return compute_bollinger_bands(ohlcv, length, mult, profile=profile)
    if price_source == "entry_exit":
        price = ((ohlcv["Open"].astype("float64") + ohlcv["Close"].astype("float64")) / 2.0).astype("float64")
        basis = price.rolling(length, min_periods=min_periods).mean()
        deviation = price.rolling(length, min_periods=min_periods).std(ddof=ddof)
        upper = basis + mult * deviation
        lower = basis - mult * deviation
        return pd.DataFrame({"basis": basis, "upper": upper, "lower": lower})
    raise ValueError(f"bb_price_source no soportado: {price_source}")


def _prepare_strategy(
    ohlcv: pd.DataFrame,
    strategy: str,
    bb_length: int,
    bb_mult: float,
    bb_direction: int,
    bb_price_source: str,
    bb_profile: str,
    st_period: int,
    st_factor: float,
    tp_pct_cfg: float,
) -> tuple[pd.Series, pd.Series, list[Signal], Optional[float], bool]:
    if strategy == "bollinger":
        bands = _compute_bollinger_bands_with_source(ohlcv, bb_length, bb_mult, bb_price_source, bb_profile)
        band_upper = bands["upper"]
        band_lower = bands["lower"]
        signals = generate_bollinger_signals(ohlcv, bands, bb_direction)
        tp_pct = None
        reentry_on_tp = True
    elif strategy == "supertrend":
        st = compute_supertrend(ohlcv, st_period, st_factor)
        band_upper = st["upper"]
        band_lower = st["lower"]
        signals = generate_supertrend_signals(ohlcv, st)
        tp_pct = None
        reentry_on_tp = True
    else:  # supertrend2
        st = compute_supertrend(ohlcv, st_period, st_factor)
        band_upper = st["upper"]
        band_lower = st["lower"]
        signals = generate_supertrend_signals(ohlcv, st)
        tp_pct = tp_pct_cfg if tp_pct_cfg > 0 else None
        reentry_on_tp = False

    return band_upper, band_lower, signals, tp_pct, reentry_on_tp


def _simulate_intrabar_touches(
    ohlcv_1s: pd.DataFrame,
    signals: list[Signal],
    sl_pct: float,
    tp_pct: Optional[float],
    reentry_on_tp: bool,
) -> list[dict]:
    signals_by_time = {s.time: s for s in signals}
    position: Optional[_SimPosition] = None
    pending_entry: Optional[Signal] = None
    touches: list[dict] = []

    def open_position(sig: Signal, price: float):
        nonlocal position
        if price <= 0:
            return
        sl = price * (1 - sl_pct) if sig.direction == "long" else price * (1 + sl_pct)
        tp = None
        if tp_pct is not None and tp_pct > 0:
            tp = price * (1 + tp_pct) if sig.direction == "long" else price * (1 - tp_pct)
        position = _SimPosition(direction=sig.direction, sl_price=sl, tp_price=tp)

    def close_position() -> Optional[str]:
        nonlocal position
        if position is None:
            return None
        direction = position.direction
        position = None
        return direction

    for i in range(len(ohlcv_1s)):
        ts = ohlcv_1s.index[i]
        row = ohlcv_1s.iloc[i]
        o = float(row.Open)
        h = float(row.High)
        l = float(row.Low)
        c = float(row.Close)

        if pending_entry is not None:
            open_position(pending_entry, o)
            pending_entry = None

        if position is not None:
            hit_sl = False
            hit_tp = False
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
                if hit_sl:  # prioridad SL
                    exit_price = float(position.sl_price)
                    closed_dir = close_position()
                    touches.append({"ts": ts, "price": exit_price, "type": "SL"})
                    flip_dir = "short" if closed_dir == "long" else "long"
                    flip_sig = Signal(time=ts, direction=flip_dir, price=exit_price, reason="stop_loss_reversal")
                    open_position(flip_sig, exit_price)
                else:
                    exit_price = float(position.tp_price)
                    closed_dir = close_position()
                    touches.append({"ts": ts, "price": exit_price, "type": "TP"})
                    if reentry_on_tp:
                        flip_dir = "short" if closed_dir == "long" else "long"
                        flip_sig = Signal(time=ts, direction=flip_dir, price=exit_price, reason="tp_reversal")
                        open_position(flip_sig, exit_price)
                continue

        sig = signals_by_time.get(ts)
        if sig is None:
            continue

        if position is None:
            open_position(sig, c)  # entrada a cierre de vela
            continue

        if position.direction == sig.direction:
            anchor = c
            position.sl_price = anchor * (1 - sl_pct) if sig.direction == "long" else anchor * (1 + sl_pct)
            if tp_pct is not None and tp_pct > 0:
                position.tp_price = anchor * (1 + tp_pct) if sig.direction == "long" else anchor * (1 - tp_pct)
        else:
            close_position()  # cierre por señal (no es toque)
            open_position(sig, c)

    return touches


def _expand_with_touches(
    df_base: pd.DataFrame,
    tf: str,
    touches: list[dict],
) -> pd.DataFrame:
    rule = _normalize_rule(tf)
    by_candle: dict[pd.Timestamp, list[dict]] = defaultdict(list)
    base_index = set(df_base.index)
    for t in touches:
        candle_ts = pd.Timestamp(t["ts"]).floor(rule)
        if candle_ts in base_index:
            by_candle[candle_ts].append(t)

    rows = []
    ordered_cols = list(df_base.columns)
    if PNL_CLOSE_COL in ordered_cols:
        ordered_cols.remove(PNL_CLOSE_COL)
    ordered_cols = ordered_cols + TOUCH_COLS + ([PNL_CLOSE_COL] if PNL_CLOSE_COL in df_base.columns else [])
    for ts, row in df_base.iterrows():
        base_row = dict(row)
        for c in TOUCH_COLS:
            base_row[c] = ""
        rows.append(base_row)

        candle_touches = sorted(by_candle.get(ts, []), key=lambda x: x["ts"])
        for seq, touch in enumerate(candle_touches, start=1):
            extra_row = dict(row)
            extra_row[ENTRY_COL] = ""
            extra_row[EXIT_COL] = str(touch["type"])
            if NEW_SL_COL in extra_row:
                extra_row[NEW_SL_COL] = ""
            if PNL_CLOSE_COL in extra_row:
                extra_row[PNL_CLOSE_COL] = ""
            touch_ts = pd.Timestamp(touch["ts"])
            extra_row["Fecha Toque"] = touch_ts.strftime("%Y-%m-%d")
            extra_row["Hora Toque"] = touch_ts.strftime("%H:%M:%S")
            extra_row["Precio Toque"] = float(touch["price"])
            extra_row["Tipo Toque"] = str(touch["type"])
            extra_row["Nro Toque En Vela"] = seq
            rows.append(extra_row)

    return pd.DataFrame(rows, columns=ordered_cols)


def _build_readable_df(df_raw: pd.DataFrame, open_by_candle: pd.Series, close_by_candle: pd.Series) -> pd.DataFrame:
    df = df_raw.copy()

    for col in TOUCH_COLS:
        if col not in df.columns:
            df[col] = ""

    touch_mask = df["Tipo Toque"].fillna("").astype(str).str.strip() != ""
    base_mask = ~touch_mask
    df["Tipo Fila"] = touch_mask.map({True: "TOQUE", False: "BASE"})
    df["ID Vela"] = base_mask.astype("int64").cumsum()
    df["FechaHora Vela"] = df["Fecha"].astype(str) + " " + df["Hora"].astype(str)
    open_keys = open_by_candle.index.strftime("%Y-%m-%d %H:%M:%S")
    open_vals = pd.to_numeric(open_by_candle, errors="coerce").round(2)
    open_map = dict(zip(open_keys, open_vals))
    df["Apertura Vela"] = pd.to_numeric(df["FechaHora Vela"].map(open_map), errors="coerce").round(2)
    close_keys = close_by_candle.index.strftime("%Y-%m-%d %H:%M:%S")
    close_vals = pd.to_numeric(close_by_candle, errors="coerce").round(2)
    close_map = dict(zip(close_keys, close_vals))
    df["Cierre Vela"] = pd.to_numeric(df["FechaHora Vela"].map(close_map), errors="coerce").round(2)

    touch_date = df["Fecha Toque"].fillna("").astype(str).str.strip()
    touch_time = df["Hora Toque"].fillna("").astype(str).str.strip()
    has_touch_dt = (touch_date != "") & (touch_time != "")
    df["FechaHora Toque"] = ""
    df.loc[has_touch_dt, "FechaHora Toque"] = touch_date[has_touch_dt] + " " + touch_time[has_touch_dt]

    for col in PRICE_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").round(2)

    # En filas TOQUE dejamos solo datos del toque para evitar ruido visual.
    compact_cols = [
        "Apertura Vela",
        "Precio Maximo Vela",
        "Precio minimo Vela",
        "Cierre Vela",
        "Precio Banda Superior",
        "Precio Banda Inferior",
        POS_TREND_COL,
        NEW_SL_COL,
        PNL_CLOSE_COL,
        ENTRY_COL,
    ]
    for c in compact_cols:
        if c in df.columns:
            df[c] = df[c].astype("object")
            df.loc[touch_mask, c] = ""

    cols = {
        "ID Vela": df["ID Vela"],
        "Tipo Fila": df["Tipo Fila"],
        "FechaHora": df["FechaHora Vela"],
        "Apertura": df["Apertura Vela"],
        "Max": df["Precio Maximo Vela"],
        "Min": df["Precio minimo Vela"],
        "Vela": df["Cierre Vela"],
        "Ba. Sup": df["Precio Banda Superior"],
        "Ba. Inf": df["Precio Banda Inferior"],
    }
    if NEW_SL_COL in df.columns:
        cols["Nuevo SL"] = df[NEW_SL_COL]
    if POS_TREND_COL in df.columns:
        cols["Tendencia"] = df[POS_TREND_COL]
    cols.update(
        {
            "Entrada": df[ENTRY_COL] if ENTRY_COL in df.columns else "",
            "Salida": df[EXIT_COL] if EXIT_COL in df.columns else "",
            "FechaHora Toque": df["FechaHora Toque"],
            "Precio Toque": df["Precio Toque"],
            "Tipo Toque": df["Tipo Toque"],
            "Toque #": df["Nro Toque En Vela"],
        }
    )
    cols[PNL_CLOSE_COL] = df[PNL_CLOSE_COL] if PNL_CLOSE_COL in df.columns else ""
    df_readable = pd.DataFrame(cols)
    return df_readable.fillna("")


def _append_pnl_total_row(
    df: pd.DataFrame,
    pnl_col: str,
    label_col: Optional[str] = None,
    label_value: str = "TOTAL PNL %",
    total_value: Optional[float] = None,
) -> pd.DataFrame:
    if pnl_col not in df.columns:
        return df
    total = total_value
    if total is None:
        total = pd.to_numeric(df[pnl_col], errors="coerce").sum(min_count=1)
    if pd.isna(total):
        total = 0.0
    row = {col: "" for col in df.columns}
    row[pnl_col] = float(total)
    if label_col and label_col in row:
        row[label_col] = label_value
    return pd.concat([df, pd.DataFrame([row], columns=df.columns)], ignore_index=True)


def _compute_pnl_total(df: pd.DataFrame, pnl_col: str) -> float:
    if pnl_col not in df.columns:
        return 0.0
    total = pd.to_numeric(df[pnl_col], errors="coerce").sum(min_count=1)
    if pd.isna(total):
        return 0.0
    return float(total)


def _pnl_suffix(total_pct: float) -> str:
    sign = "pos" if total_pct >= 0 else "neg"
    return f"pnl_{sign}{abs(total_pct):.2f}pct"


def build_table(
    parquet_path: Path,
    strategy: str,
    out_csv: Path,
    tf: str,
    start: str,
    end: str,
    calc_start: str,
    calc_end: str,
    notional: float,
    fee: float,
    bb_length: int,
    bb_mult: float,
    bb_direction: int,
    bb_price_source: str,
    bb_profile: str,
    st_period: int,
    st_factor: float,
    sl: Optional[float],
    tp: Optional[float],
    expand_sl_tp: bool,
    sl_intrabar_1s: bool,
    readable: bool,
    pnl_in_filename: bool,
) -> int:
    if not parquet_path.exists():
        raise FileNotFoundError(f"No existe parquet: {parquet_path}")

    defaults = _strategy_defaults()
    sl_pct = float(sl if sl is not None else defaults[strategy]["sl"])
    tp_pct_cfg = float(tp if tp is not None else defaults[strategy]["tp"])

    ts_out_start = _parse_local_ts(start)
    ts_out_end = _parse_local_ts(end)
    # Compatibilidad hacia atras:
    # si no se pasa calc_start/calc_end, se usa start/end como ventana de calculo.
    ts_calc_start = _parse_local_ts(calc_start) if calc_start else ts_out_start
    ts_calc_end = _parse_local_ts(calc_end) if calc_end else ts_out_end
    df_ticks = _load_ticks_with_window(parquet_path, ts_calc_start, ts_calc_end)

    df_ticks_calc = _filter_ticks_window(df_ticks, ts_calc_start, ts_calc_end)
    ohlcv_30m_calc = _resample_ohlcv(df_ticks_calc, tf, TZ_BA)
    if ts_calc_start is not None:
        ohlcv_30m_calc = ohlcv_30m_calc[ohlcv_30m_calc.index >= ts_calc_start]
    if ts_calc_end is not None:
        ohlcv_30m_calc = ohlcv_30m_calc[ohlcv_30m_calc.index <= ts_calc_end]
    if ohlcv_30m_calc.empty:
        raise ValueError("No hay velas en el rango de calculo solicitado")

    band_upper, band_lower, signals_30m, tp_pct, reentry_on_tp = _prepare_strategy(
        ohlcv=ohlcv_30m_calc,
        strategy=strategy,
        bb_length=bb_length,
        bb_mult=bb_mult,
        bb_direction=bb_direction,
        bb_price_source=bb_price_source,
        bb_profile=bb_profile,
        st_period=st_period,
        st_factor=st_factor,
        tp_pct_cfg=tp_pct_cfg,
    )

    sl_update_by_ts: dict[pd.Timestamp, float] = {}
    position_by_ts: dict[pd.Timestamp, str] = {}

    ohlcv_1s = None
    if expand_sl_tp or sl_intrabar_1s:
        ohlcv_1s = _resample_ohlcv(df_ticks_calc, "1s", TZ_BA)
        if ts_calc_start is not None:
            ohlcv_1s = ohlcv_1s[ohlcv_1s.index >= ts_calc_start]
        if ts_calc_end is not None:
            ohlcv_1s = ohlcv_1s[ohlcv_1s.index <= ts_calc_end]

    def _on_same_dir_sl(ts: pd.Timestamp, direction: str, new_sl: float) -> None:
        # Registramos solo el último update por vela/timestamp.
        sl_update_by_ts[pd.Timestamp(ts)] = float(new_sl)

    def _on_position_state(ts: pd.Timestamp, direction: Optional[str]) -> None:
        if direction == "long":
            position_by_ts[pd.Timestamp(ts)] = "Long"
        elif direction == "short":
            position_by_ts[pd.Timestamp(ts)] = "Short"
        else:
            position_by_ts[pd.Timestamp(ts)] = ""

    same_dir_hook = _on_same_dir_sl if strategy == "bollinger" else None
    trades_30m, _ = run_backtest(
        ohlcv_30m_calc,
        signals_30m,
        entry_mode="close",
        sl_pct=sl_pct,
        tp_pct=tp_pct,
        notional=notional,
        fee_rate=fee,
        reentry_on_tp=reentry_on_tp,
        same_dir_sl_hook=same_dir_hook,
        position_state_hook=_on_position_state,
        intrabar_ohlcv=ohlcv_1s if sl_intrabar_1s else None,
        sl_tp_intrabar=bool(sl_intrabar_1s),
    )

    entries_by_ts: dict[pd.Timestamp, list[str]] = defaultdict(list)
    exits_by_ts: dict[pd.Timestamp, list[str]] = defaultdict(list)
    pnl_close_by_ts: dict[pd.Timestamp, float] = defaultdict(float)
    for tr in trades_30m:
        entry_ts = _align_ts_to_tf(pd.Timestamp(tr.entry_time), tf)
        exit_ts = _align_ts_to_tf(pd.Timestamp(tr.exit_time), tf)
        entries_by_ts[entry_ts].append(_entry_label(tr.entry_reason))
        exit_lbl = _exit_label(tr.exit_reason)
        exits_by_ts[exit_ts].append(exit_lbl)
        # Toda salida real de posición (SL/TP/estrategia) refleja su PNL%.
        pnl_close_by_ts[exit_ts] += float(tr.pnl_pct) * 100.0

    df_base = pd.DataFrame(index=ohlcv_30m_calc.index)
    df_base["Fecha"] = df_base.index.strftime("%Y-%m-%d")
    df_base["Hora"] = df_base.index.strftime("%H:%M:%S")
    df_base["Precio Maximo Vela"] = ohlcv_30m_calc["High"].astype("float64")
    df_base["Precio minimo Vela"] = ohlcv_30m_calc["Low"].astype("float64")
    df_base["Precio Banda Superior"] = band_upper.reindex(df_base.index).astype("float64")
    df_base["Precio Banda Inferior"] = band_lower.reindex(df_base.index).astype("float64")
    if strategy == "bollinger":
        sl_series = pd.Series(sl_update_by_ts, dtype="float64")
        df_base[NEW_SL_COL] = sl_series.reindex(df_base.index)
    position_series = pd.Series(position_by_ts, dtype="object")
    df_base[POS_TREND_COL] = position_series.reindex(df_base.index).fillna("")
    df_base[ENTRY_COL] = [_collapse_labels(entries_by_ts.get(ts, [])) for ts in df_base.index]
    df_base[EXIT_COL] = [_collapse_labels(exits_by_ts.get(ts, [])) for ts in df_base.index]
    pnl_series = pd.Series(pnl_close_by_ts, dtype="float64")
    df_base[PNL_CLOSE_COL] = pnl_series.reindex(df_base.index)

    # Recorte de salida posterior al calculo (evita cortes de estado/indicadores entre meses).
    df_base_out = df_base
    if ts_out_start is not None:
        df_base_out = df_base_out[df_base_out.index >= ts_out_start]
    if ts_out_end is not None:
        df_base_out = df_base_out[df_base_out.index <= ts_out_end]
    if df_base_out.empty:
        raise ValueError("No hay velas en el rango de salida solicitado")

    touches = []
    if expand_sl_tp:
        if ohlcv_1s is not None and not ohlcv_1s.empty:
            touches = _simulate_intrabar_touches(
                ohlcv_1s=ohlcv_1s,
                signals=signals_30m,
                sl_pct=sl_pct,
                tp_pct=tp_pct,
                reentry_on_tp=reentry_on_tp,
            )
        df_out = _expand_with_touches(df_base=df_base_out, tf=tf, touches=touches)
    else:
        df_out = df_base_out

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pnl_total = _compute_pnl_total(df_out, PNL_CLOSE_COL)
    name_suffix = _pnl_suffix(pnl_total) if pnl_in_filename else ""
    out_csv_effective = out_csv
    if pnl_in_filename:
        out_csv_effective = out_csv.with_name(f"{out_csv.stem}_{name_suffix}{out_csv.suffix}")

    df_out_tech = _append_pnl_total_row(
        df_out,
        PNL_CLOSE_COL,
        label_col="Fecha",
        label_value="TOTAL PNL %",
        total_value=pnl_total,
    )
    df_out_tech.to_csv(out_csv_effective, index=False)
    if readable:
        if pnl_in_filename:
            out_readable = out_csv.with_name(f"{out_csv.stem}_readable_{name_suffix}.csv")
            out_readable_excel = out_csv.with_name(f"{out_csv.stem}_readable_excel_{name_suffix}.csv")
        else:
            out_readable = out_csv.with_name(f"{out_csv.stem}_readable.csv")
            out_readable_excel = out_csv.with_name(f"{out_csv.stem}_readable_excel.csv")
        df_readable = _build_readable_df(df_out, ohlcv_30m_calc["Open"], ohlcv_30m_calc["Close"])
        df_readable = _append_pnl_total_row(
            df_readable,
            PNL_CLOSE_COL,
            label_col="Tipo Fila",
            label_value="TOTAL",
            total_value=round(pnl_total, 2),
        )
        df_readable.to_csv(out_readable, index=False)
        df_readable.to_csv(out_readable_excel, index=False, sep=";", encoding="utf-8-sig")
        print(f"OK (readable) -> {out_readable}")
        print(f"OK (readable excel) -> {out_readable_excel}")

    print(f"OK -> {out_csv_effective}")
    print(f"Estrategia: {strategy} | TF: {tf} | TZ: {TZ_BA}")
    if strategy == "bollinger":
        print(f"Bollinger profile: {bb_profile}")
        print(f"Bollinger fuente de precio: {bb_price_source}")
    print(f"SL={sl_pct:.6f} | TP={tp_pct_cfg:.6f}")
    print(f"Velas calculadas ({tf}): {len(ohlcv_30m_calc)}")
    print(f"Velas salida ({tf}): {len(df_base_out)}")
    print(f"Touches SL/TP: {len(touches)}")
    print(f"Filas exportadas: {len(df_out_tech)}")
    print(f"Trades base (30m): {len(trades_30m)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Exporta una tabla CSV por vela con bandas y señales, con expansión intravela opcional por toques SL/TP."
    )
    parser.add_argument("parquet_path", help="Parquet OHLC fuente")
    parser.add_argument("--strategy", required=True, choices=["bollinger", "supertrend", "supertrend2"])
    parser.add_argument("--out", required=True, help="CSV de salida")
    parser.add_argument("--tf", default="30T", help="Timeframe final (default: 30T)")
    parser.add_argument("--start", default="", help="Inicio de salida opcional (ISO)")
    parser.add_argument("--end", default="", help="Fin de salida opcional (ISO)")
    parser.add_argument(
        "--calc-start",
        default="",
        help="Inicio de calculo opcional (ISO). Si no se pasa, usa --start (compatibilidad).",
    )
    parser.add_argument(
        "--calc-end",
        default="",
        help="Fin de calculo opcional (ISO). Si no se pasa, usa --end (compatibilidad).",
    )
    parser.add_argument("--notional", type=float, default=30.0)
    parser.add_argument("--fee", type=float, default=0.0004)
    parser.add_argument("--sl", type=float, default=None, help="Override de SL (si no, usa default por estrategia)")
    parser.add_argument("--tp", type=float, default=None, help="Override de TP (si no, usa default por estrategia)")
    parser.add_argument("--bb-length", type=int, default=20)
    parser.add_argument("--bb-mult", type=float, default=2.0)
    parser.add_argument("--bb-direction", type=int, default=0)
    parser.add_argument(
        "--bb-price-source",
        choices=["close", "entry_exit"],
        default="close",
        help="Fuente para Bollinger: close (cierre) o entry_exit ((open+close)/2)",
    )
    parser.add_argument(
        "--bb-profile",
        choices=["tradingview", "legacy"],
        default="tradingview",
        help="Perfil Bollinger: tradingview(ddof=0,min_periods=length) o legacy(ddof=1,min_periods=1)",
    )
    parser.add_argument("--st-period", type=int, default=10)
    parser.add_argument("--st-factor", type=float, default=3.0)
    parser.add_argument(
        "--sl-intrabar-1s",
        action="store_true",
        help="Evalúa toques de SL/TP para PnL usando velas de 1s (timestamp exacto intravela).",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--expand-sl-tp", dest="expand_sl_tp", action="store_true", help="Expande filas por toques SL/TP intravela (default)")
    group.add_argument("--no-expand-sl-tp", dest="expand_sl_tp", action="store_false", help="Desactiva expansión intravela")
    parser.set_defaults(expand_sl_tp=True)
    parser.add_argument("--readable", action="store_true", help="Genera CSVs legibles compactos: coma (*_readable.csv) y Excel ';' (*_readable_excel.csv)")
    parser.add_argument("--pnl-in-filename", action="store_true", help="Incluye el PNL total en el nombre de salida")
    args = parser.parse_args()

    try:
        return build_table(
            parquet_path=Path(args.parquet_path),
            strategy=args.strategy,
            out_csv=Path(args.out),
            tf=args.tf,
            start=args.start,
            end=args.end,
            calc_start=args.calc_start,
            calc_end=args.calc_end,
            notional=float(args.notional),
            fee=float(args.fee),
            bb_length=int(args.bb_length),
            bb_mult=float(args.bb_mult),
            bb_direction=int(args.bb_direction),
            bb_price_source=str(args.bb_price_source),
            bb_profile=str(args.bb_profile),
            st_period=int(args.st_period),
            st_factor=float(args.st_factor),
            sl=args.sl,
            tp=args.tp,
            expand_sl_tp=bool(args.expand_sl_tp),
            sl_intrabar_1s=bool(args.sl_intrabar_1s),
            readable=bool(args.readable),
            pnl_in_filename=bool(args.pnl_in_filename),
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

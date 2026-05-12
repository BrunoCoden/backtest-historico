#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from backtest_plotly import _resample_ohlcv


RAW_RE = re.compile(r"^(?P<symbol>[A-Z0-9]+)_(?P<yyyymm>\d{6})_1s_ohlc\.parquet$")

PRICE_COLUMNS = {
    "zone_low",
    "zone_high",
    "channel_value",
    "channel_upper",
    "channel_lower",
    "channel_upper_mid",
    "channel_lower_mid",
    "low_event",
    "high_event",
}

PCT_COLUMNS = {
    "event_share_pct",
    "avg_event_width_pct",
    "width_pct_event",
}

SPANISH_HEADERS = {
    "symbol": "Sym",
    "zone_idx": "Zona ID",
    "zone_low": "Zona Min",
    "zone_high": "Zona Max",
    "event_count": "Cantidad Eventos",
    "event_share_pct": "Porcentaje Eventos",
    "avg_event_width_pct": "Ancho Prom Evento %",
    "avg_event_bars": "Velas Prom Evento",
    "first_event_ts": "Inicio Primer Evento",
    "last_event_ts": "Fin Ultimo Evento",
    "rank_in_symbol": "Ranking Simbolo",
    "event_id": "ID Evento",
    "start_ts": "Inicio Evento",
    "end_ts": "Fin Evento",
    "reset_ts": "Timestamp Reset",
    "reset_reason": "Motivo Reset",
    "bars_event": "N° V",
    "channel_value": "C. Valor",
    "channel_upper": "C. Sup",
    "channel_lower": "C. Inf.",
    "channel_upper_mid": "C. Sup Medio",
    "channel_lower_mid": "C. Inf. Medio",
    "low_event": "Min Evento",
    "high_event": "Max Evento",
    "width_pct_event": "Ancho Evento %",
}


@dataclass(frozen=True)
class Config:
    tf: str
    tz: str
    raw_root: Path
    out_dir: Path
    symbols: list[str]
    start_yyyymm: str
    end_yyyymm: str
    zone_mode: str
    rb_multi: float
    rb_atr_len: int
    rb_atr_sma_len: int
    rb_init_bar_index: int
    rb_max_outside_bars: int


def _month_list(start_yyyymm: str, end_yyyymm: str) -> list[str]:
    sy, sm = int(start_yyyymm[:4]), int(start_yyyymm[4:6])
    ey, em = int(end_yyyymm[:4]), int(end_yyyymm[4:6])
    out: list[str] = []
    y, m = sy, sm
    while (y < ey) or (y == ey and m <= em):
        out.append(f"{y:04d}{m:02d}")
        m += 1
        if m > 12:
            y += 1
            m = 1
    return out


def _discover_monthly_raw(raw_root: Path) -> dict[str, dict[str, Path]]:
    by_symbol: dict[str, dict[str, Path]] = {}
    for p in sorted(raw_root.glob("*/*.parquet")):
        m = RAW_RE.match(p.name)
        if not m:
            continue
        symbol = m.group("symbol")
        yyyymm = m.group("yyyymm")
        by_symbol.setdefault(symbol, {})[yyyymm] = p
    return by_symbol


def _build_ohlcv_symbol(cfg: Config, symbol: str, monthly_paths: dict[str, Path]) -> pd.DataFrame:
    months = _month_list(cfg.start_yyyymm, cfg.end_yyyymm)
    missing = [m for m in months if m not in monthly_paths]
    if missing:
        raise ValueError(f"Faltan parquets para {symbol}: {missing}")

    chunks: list[pd.DataFrame] = []
    for yyyymm in months:
        path = monthly_paths[yyyymm]
        print(f"[{symbol}] leyendo {yyyymm}: {path}", flush=True)
        ticks = pd.read_parquet(path)
        ohlcv = _resample_ohlcv(ticks, cfg.tf, cfg.tz)
        if ohlcv.empty:
            continue
        chunks.append(ohlcv[["Open", "High", "Low", "Close", "Volume"]].copy())

    if not chunks:
        raise ValueError(f"No se pudieron construir velas {cfg.tf} para {symbol}")

    out = pd.concat(chunks, axis=0)
    out = out[~out.index.duplicated(keep="last")]
    out = out.sort_index()
    return out


def _rma(series: pd.Series, length: int) -> pd.Series:
    if length <= 0:
        raise ValueError("length debe ser > 0")
    x = series.to_numpy(dtype="float64")
    out = np.full(len(x), np.nan, dtype="float64")
    if len(x) < length:
        return pd.Series(out, index=series.index)
    seed = float(np.nanmean(x[:length]))
    out[length - 1] = seed
    alpha = 1.0 / float(length)
    for i in range(length, len(x)):
        out[i] = out[i - 1] + alpha * (x[i] - out[i - 1])
    return pd.Series(out, index=series.index)


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def _crossover(prev_a: float, cur_a: float, prev_b: float, cur_b: float) -> bool:
    if np.isnan(prev_a) or np.isnan(cur_a) or np.isnan(prev_b) or np.isnan(cur_b):
        return False
    return (prev_a <= prev_b) and (cur_a > cur_b)


def _crossunder(prev_a: float, cur_a: float, prev_b: float, cur_b: float) -> bool:
    if np.isnan(prev_a) or np.isnan(cur_a) or np.isnan(prev_b) or np.isnan(cur_b):
        return False
    return (prev_a >= prev_b) and (cur_a < cur_b)


def _detect_events_rangebreakout(ohlcv: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    cols = [
        "event_id",
        "start_ts",
        "end_ts",
        "reset_ts",
        "reset_reason",
        "bars_event",
        "channel_value",
        "channel_upper",
        "channel_lower",
        "channel_upper_mid",
        "channel_lower_mid",
        "low_event",
        "high_event",
        "width_pct_event",
    ]
    if ohlcv.empty:
        return pd.DataFrame(columns=cols)

    high = ohlcv["High"].astype(float)
    low = ohlcv["Low"].astype(float)
    close = ohlcv["Close"].astype(float)
    hl2 = (high + low) / 2.0

    tr = _true_range(high, low, close)
    atr = _rma(tr, cfg.rb_atr_len)
    atr_width = atr.rolling(window=cfg.rb_atr_sma_len, min_periods=cfg.rb_atr_sma_len).mean() * float(cfg.rb_multi)

    value = np.nan
    upper = np.nan
    lower = np.nan
    count = 0
    active = False

    event_start_i: int | None = None
    event_value = np.nan
    event_upper = np.nan
    event_lower = np.nan
    event_id = 0
    rows: list[dict] = []

    def _open_event(i: int) -> None:
        nonlocal active, count, event_start_i, event_value, event_upper, event_lower, value, upper, lower
        w = float(atr_width.iloc[i])
        if np.isnan(w):
            active = False
            return
        value = float(hl2.iloc[i])
        upper = value + w
        lower = value - w
        count = 0
        active = True
        event_start_i = i
        event_value = value
        event_upper = upper
        event_lower = lower

    def _close_event(end_i: int, reset_i: int, reason: str) -> None:
        nonlocal event_id
        if event_start_i is None:
            return
        if end_i < event_start_i:
            return
        seg = ohlcv.iloc[event_start_i : end_i + 1]
        low_event = float(seg["Low"].min())
        high_event = float(seg["High"].max())
        width_pct = ((high_event - low_event) / low_event) * 100.0 if low_event > 0 else float("inf")
        event_id += 1
        rows.append(
            {
                "event_id": event_id,
                "start_ts": ohlcv.index[event_start_i],
                "end_ts": ohlcv.index[end_i],
                "reset_ts": ohlcv.index[reset_i],
                "reset_reason": reason,
                "bars_event": int(end_i - event_start_i + 1),
                "channel_value": float(event_value),
                "channel_upper": float(event_upper),
                "channel_lower": float(event_lower),
                "channel_upper_mid": (float(event_value) + float(event_upper)) / 2.0,
                "channel_lower_mid": (float(event_value) + float(event_lower)) / 2.0,
                "low_event": low_event,
                "high_event": high_event,
                "width_pct_event": width_pct,
            }
        )

    upper_prev_bar = np.nan
    lower_prev_bar = np.nan

    for i in range(len(ohlcv)):
        if i == cfg.rb_init_bar_index:
            _open_event(i)

        if not active:
            upper_prev_bar = np.nan
            lower_prev_bar = np.nan
            continue

        if i <= 0:
            upper_prev_bar = float(upper)
            lower_prev_bar = float(lower)
            continue

        cur_upper = float(upper)
        cur_lower = float(lower)
        cross_upper = _crossover(float(low.iloc[i - 1]), float(low.iloc[i]), float(upper_prev_bar), cur_upper)
        cross_lower = _crossunder(float(high.iloc[i - 1]), float(high.iloc[i]), float(lower_prev_bar), cur_lower)

        outside = bool((low.iloc[i] > cur_upper) or (high.iloc[i] < cur_lower))
        if outside:
            count += 1

        reset = bool(cross_upper or cross_lower or (count == cfg.rb_max_outside_bars))
        if reset:
            if cross_upper:
                reason = "breakout_up"
            elif cross_lower:
                reason = "breakout_down"
            else:
                reason = "outside_100"

            _close_event(end_i=i - 1, reset_i=i, reason=reason)
            _open_event(i)

        upper_prev_bar = float(upper) if active else np.nan
        lower_prev_bar = float(lower) if active else np.nan

    if active and event_start_i is not None:
        end_i = len(ohlcv) - 1
        if end_i >= event_start_i:
            seg = ohlcv.iloc[event_start_i : end_i + 1]
            low_event = float(seg["Low"].min())
            high_event = float(seg["High"].max())
            width_pct = ((high_event - low_event) / low_event) * 100.0 if low_event > 0 else float("inf")
            event_id += 1
            rows.append(
                {
                    "event_id": event_id,
                    "start_ts": ohlcv.index[event_start_i],
                    "end_ts": ohlcv.index[end_i],
                    "reset_ts": ohlcv.index[end_i],
                    "reset_reason": "eod",
                    "bars_event": int(end_i - event_start_i + 1),
                    "channel_value": float(event_value),
                    "channel_upper": float(event_upper),
                    "channel_lower": float(event_lower),
                    "channel_upper_mid": (float(event_value) + float(event_upper)) / 2.0,
                    "channel_lower_mid": (float(event_value) + float(event_lower)) / 2.0,
                    "low_event": low_event,
                    "high_event": high_event,
                    "width_pct_event": width_pct,
                }
            )

    return pd.DataFrame(rows, columns=cols)


def _build_ranking(events_z: pd.DataFrame, symbol: str) -> pd.DataFrame:
    cols = [
        "symbol",
        "zone_idx",
        "zone_low",
        "zone_high",
        "event_count",
        "event_share_pct",
        "avg_event_width_pct",
        "avg_event_bars",
        "first_event_ts",
        "last_event_ts",
    ]
    if events_z.empty:
        return pd.DataFrame(columns=cols)

    grp = (
        events_z.groupby("zone_idx", as_index=False)
        .agg(
            zone_low=("zone_low", "first"),
            zone_high=("zone_high", "first"),
            event_count=("event_id", "count"),
            avg_event_width_pct=("width_pct_event", "mean"),
            avg_event_bars=("bars_event", "mean"),
            first_event_ts=("start_ts", "min"),
            last_event_ts=("end_ts", "max"),
        )
        .sort_values(["event_count", "zone_idx"], ascending=[False, True])
        .reset_index(drop=True)
    )
    total_events = float(grp["event_count"].sum())
    grp.insert(0, "symbol", symbol)
    grp["event_share_pct"] = (grp["event_count"] / total_events) * 100.0 if total_events > 0 else 0.0
    return grp[cols]


def _build_discovered_outputs(events: pd.DataFrame, symbol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rank_cols = [
        "symbol",
        "zone_idx",
        "zone_low",
        "zone_high",
        "event_count",
        "event_share_pct",
        "avg_event_width_pct",
        "avg_event_bars",
        "first_event_ts",
        "last_event_ts",
    ]
    detail_cols = [
        "symbol",
        "zone_idx",
        "zone_low",
        "zone_high",
        "event_id",
        "start_ts",
        "end_ts",
        "reset_ts",
        "reset_reason",
        "bars_event",
        "channel_value",
        "channel_upper",
        "channel_lower",
        "channel_upper_mid",
        "channel_lower_mid",
        "low_event",
        "high_event",
        "width_pct_event",
    ]
    if events.empty:
        return pd.DataFrame(columns=rank_cols), pd.DataFrame(columns=detail_cols)

    ev = events.copy()
    ev["center_price"] = (ev["low_event"] + ev["high_event"]) / 2.0
    ev = ev.sort_values(["center_price", "start_ts"]).reset_index(drop=True)

    clusters: list[dict] = []
    cur_idx = [0]
    core_low = float(ev.loc[0, "low_event"])
    core_high = float(ev.loc[0, "high_event"])
    zone_low = core_low
    zone_high = core_high

    for i in range(1, len(ev)):
        lo = float(ev.loc[i, "low_event"])
        hi = float(ev.loc[i, "high_event"])
        if lo <= core_high and hi >= core_low:
            cur_idx.append(i)
            core_low = max(core_low, lo)
            core_high = min(core_high, hi)
            zone_low = min(zone_low, lo)
            zone_high = max(zone_high, hi)
        else:
            clusters.append({"member_idx": list(cur_idx), "zone_low": zone_low, "zone_high": zone_high})
            cur_idx = [i]
            core_low = lo
            core_high = hi
            zone_low = lo
            zone_high = hi

    clusters.append({"member_idx": list(cur_idx), "zone_low": zone_low, "zone_high": zone_high})
    clusters = sorted(clusters, key=lambda c: (-len(c["member_idx"]), c["zone_low"]))

    events_z = ev.copy()
    events_z["zone_idx"] = pd.Series(dtype="int64")
    events_z["zone_low"] = pd.Series(dtype="float64")
    events_z["zone_high"] = pd.Series(dtype="float64")
    for rank, cl in enumerate(clusters, start=1):
        idxs = cl["member_idx"]
        events_z.loc[idxs, "zone_idx"] = rank
        events_z.loc[idxs, "zone_low"] = float(cl["zone_low"])
        events_z.loc[idxs, "zone_high"] = float(cl["zone_high"])
    events_z["zone_idx"] = events_z["zone_idx"].astype("int64")

    ranking = _build_ranking(events_z, symbol)
    detail = events_z[
        [
            "zone_idx",
            "zone_low",
            "zone_high",
            "event_id",
            "start_ts",
            "end_ts",
            "reset_ts",
            "reset_reason",
            "bars_event",
            "channel_value",
            "channel_upper",
            "channel_lower",
            "channel_upper_mid",
            "channel_lower_mid",
            "low_event",
            "high_event",
            "width_pct_event",
        ]
    ].copy()
    detail.insert(0, "symbol", symbol)
    detail = detail.sort_values(["zone_idx", "start_ts", "event_id"]).reset_index(drop=True)
    return ranking, detail[detail_cols]


def _format_ts_series(s: pd.Series, tz: str) -> pd.Series:
    ts = pd.to_datetime(s, errors="coerce")
    if getattr(ts.dt, "tz", None) is not None:
        ts = ts.dt.tz_convert(tz)
    out = ts.dt.strftime("%y-%m-%d %H:%M")
    return out.fillna("")


def _format_ts_columns(df: pd.DataFrame, tz: str) -> pd.DataFrame:
    out = df.copy()
    for col in ["start_ts", "end_ts", "reset_ts", "first_event_ts", "last_event_ts"]:
        if col in out.columns:
            out[col] = _format_ts_series(out[col], tz)
    return out


def _prepare_export_df(df: pd.DataFrame, tz: str) -> pd.DataFrame:
    out = _format_ts_columns(df, tz)

    if "symbol" in out.columns:
        out["symbol"] = out["symbol"].astype(str).str[:3].str.upper()

    for col in PRICE_COLUMNS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").round(2)

    for col in PCT_COLUMNS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").round(2)

    if "avg_event_bars" in out.columns:
        out["avg_event_bars"] = pd.to_numeric(out["avg_event_bars"], errors="coerce").round(2)

    if "zone_low" in out.columns:
        sort_cols = [c for c in ["zone_low", "zone_idx", "event_id"] if c in out.columns]
        out = out.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    out = out.rename(columns=SPANISH_HEADERS)
    out.columns = [
        c.replace("Superior", "Sup").replace("Inferior", "Inf.")
        for c in out.columns
    ]
    return out


def _validate_events(events: pd.DataFrame, symbol: str) -> None:
    if events.empty:
        return
    if not np.isfinite(events["width_pct_event"].to_numpy(dtype="float64")).all():
        raise ValueError(f"{symbol}: width_pct_event contiene valores no finitos")
    if (events["width_pct_event"] < 0).any():
        raise ValueError(f"{symbol}: width_pct_event contiene valores negativos")
    if (events["bars_event"] <= 0).any():
        raise ValueError(f"{symbol}: bars_event <= 0")

    s = events.sort_values("start_ts").reset_index(drop=True)
    if (s["end_ts"] < s["start_ts"]).any():
        raise ValueError(f"{symbol}: hay eventos con end_ts < start_ts")

    for i in range(1, len(s)):
        prev_end = s.loc[i - 1, "end_ts"]
        cur_start = s.loc[i, "start_ts"]
        prev_reset = s.loc[i - 1, "reset_ts"]
        if cur_start <= prev_end:
            raise ValueError(f"{symbol}: eventos superpuestos entre filas {i} y {i+1}")
        if pd.Timestamp(cur_start) != pd.Timestamp(prev_reset):
            raise ValueError(f"{symbol}: discontinuidad temporal entre filas {i} y {i+1}")

    if str(s.loc[len(s) - 1, "reset_reason"]) != "eod":
        raise ValueError(f"{symbol}: último evento no termina con reset_reason=eod")


def run(cfg: Config) -> int:
    if cfg.zone_mode != "discovered":
        raise ValueError("En esta versión RB solo se soporta --zone-mode discovered")

    by_symbol = _discover_monthly_raw(cfg.raw_root)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    all_rankings: list[pd.DataFrame] = []
    months = _month_list(cfg.start_yyyymm, cfg.end_yyyymm)
    print(f"Meses esperados: {months[0]}..{months[-1]} ({len(months)})", flush=True)

    for symbol in cfg.symbols:
        if symbol not in by_symbol:
            raise ValueError(f"No hay parquets para símbolo {symbol}")

        ohlcv = _build_ohlcv_symbol(cfg, symbol, by_symbol[symbol])
        events = _detect_events_rangebreakout(ohlcv, cfg)
        _validate_events(events, symbol)
        ranking, detail = _build_discovered_outputs(events, symbol)

        if not ranking.empty and int(ranking["event_count"].sum()) != len(events):
            raise ValueError(f"{symbol}: sum(event_count) != cantidad de eventos")

        out_symbol = cfg.out_dir / f"rangos_{symbol}.csv"
        out_detail = cfg.out_dir / f"rangos_{symbol}_eventos.csv"
        ranking_out = _prepare_export_df(ranking, cfg.tz)
        detail_out = _prepare_export_df(detail, cfg.tz)
        ranking_out.to_csv(out_symbol, index=False, float_format="%.2f")
        detail_out.to_csv(out_detail, index=False, float_format="%.2f")
        print(f"[OK] {symbol} -> {out_symbol} | eventos={len(events)} | zonas={len(ranking)}", flush=True)
        print(f"[OK] {symbol} detalle -> {out_detail} | filas={len(detail)}", flush=True)
        all_rankings.append(ranking)

    if all_rankings:
        summary = pd.concat(all_rankings, ignore_index=True)
        summary = summary.sort_values(["symbol", "event_count", "zone_idx"], ascending=[True, False, True]).reset_index(drop=True)
        summary["rank_in_symbol"] = summary.groupby("symbol").cumcount().add(1)
        summary = summary.sort_values(["symbol", "rank_in_symbol"]).reset_index(drop=True)
    else:
        summary = pd.DataFrame(
            columns=[
                "symbol",
                "zone_idx",
                "zone_low",
                "zone_high",
                "event_count",
                "event_share_pct",
                "avg_event_width_pct",
                "avg_event_bars",
                "first_event_ts",
                "last_event_ts",
                "rank_in_symbol",
            ]
        )

    out_summary = cfg.out_dir / "rangos_resumen.csv"
    summary_out = _prepare_export_df(summary, cfg.tz)
    summary_out.to_csv(out_summary, index=False, float_format="%.2f")
    print(f"[OK] resumen -> {out_summary}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Estadística de rangos basada en Range Breakout (BigBeluga).")
    parser.add_argument("--tf", default="30T")
    parser.add_argument("--tz", default="America/Argentina/Buenos_Aires")
    parser.add_argument("--raw-root", default="data/velas crudas")
    parser.add_argument("--out-dir", default="data/estadisticas/rb")
    parser.add_argument("--symbols", default="BTCUSDT,ETHUSDT", help="CSV de símbolos")
    parser.add_argument("--start-yyyymm", default="202501")
    parser.add_argument("--end-yyyymm", default="202602")
    parser.add_argument("--zone-mode", choices=["discovered"], default="discovered")
    parser.add_argument("--rb-multi", type=float, default=4.0)
    parser.add_argument("--rb-atr-len", type=int, default=200)
    parser.add_argument("--rb-atr-sma-len", type=int, default=100)
    parser.add_argument("--rb-init-bar-index", type=int, default=301)
    parser.add_argument("--rb-max-outside-bars", type=int, default=100)
    args = parser.parse_args()

    symbols = [s.strip() for s in str(args.symbols).split(",") if s.strip()]
    cfg = Config(
        tf=str(args.tf),
        tz=str(args.tz),
        raw_root=Path(args.raw_root),
        out_dir=Path(args.out_dir),
        symbols=symbols,
        start_yyyymm=str(args.start_yyyymm),
        end_yyyymm=str(args.end_yyyymm),
        zone_mode=str(args.zone_mode),
        rb_multi=float(args.rb_multi),
        rb_atr_len=int(args.rb_atr_len),
        rb_atr_sma_len=int(args.rb_atr_sma_len),
        rb_init_bar_index=int(args.rb_init_bar_index),
        rb_max_outside_bars=int(args.rb_max_outside_bars),
    )

    try:
        return run(cfg)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

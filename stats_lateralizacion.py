#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from backtest_plotly import _resample_ohlcv


RAW_RE = re.compile(r"^(?P<symbol>[A-Z0-9]+)_(?P<yyyymm>\d{6})_1s_ohlc\.parquet$")


@dataclass(frozen=True)
class Config:
    tf: str
    window: int
    range_max_pct: float
    zone_width_pct: float
    zone_mode: str
    tz: str
    raw_root: Path
    out_dir: Path
    symbols: list[str]
    start_yyyymm: str
    end_yyyymm: str


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
        raise ValueError(f"No se pudieron construir velas 30m para {symbol}")

    all_ohlcv = pd.concat(chunks, axis=0)
    all_ohlcv = all_ohlcv[~all_ohlcv.index.duplicated(keep="last")]
    all_ohlcv = all_ohlcv.sort_index()
    return all_ohlcv


def _interval_width_pct(ohlcv: pd.DataFrame, start_i: int, end_i: int) -> float:
    seg = ohlcv.iloc[start_i : end_i + 1]
    low = float(seg["Low"].min())
    high = float(seg["High"].max())
    if low <= 0:
        return float("inf")
    return (high - low) / low


def _detect_events(ohlcv: pd.DataFrame, window: int, range_max_pct: float) -> pd.DataFrame:
    if len(ohlcv) < window:
        return pd.DataFrame(
            columns=["event_id", "start_ts", "end_ts", "low_event", "high_event", "width_pct_event", "bars_event"]
        )

    thr = float(range_max_pct) / 100.0
    high_roll = ohlcv["High"].rolling(window=window, min_periods=window).max()
    low_roll = ohlcv["Low"].rolling(window=window, min_periods=window).min()
    width_roll = (high_roll - low_roll) / low_roll
    ok = (width_roll <= thr) & width_roll.notna()
    idxs = np.flatnonzero(ok.to_numpy())
    if len(idxs) == 0:
        return pd.DataFrame(
            columns=["event_id", "start_ts", "end_ts", "low_event", "high_event", "width_pct_event", "bars_event"]
        )

    candidates = [(int(i - window + 1), int(i)) for i in idxs]
    merged: list[tuple[int, int]] = []
    cur_s, cur_e = candidates[0]

    for s, e in candidates[1:]:
        if s <= cur_e + 1:
            prop_s = cur_s
            prop_e = max(cur_e, e)
            prop_w = _interval_width_pct(ohlcv, prop_s, prop_e)
            if prop_w <= thr:
                cur_s, cur_e = prop_s, prop_e
            else:
                merged.append((cur_s, cur_e))
                cur_s, cur_e = s, e
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))

    rows: list[dict] = []
    for i, (s, e) in enumerate(merged, start=1):
        seg = ohlcv.iloc[s : e + 1]
        low = float(seg["Low"].min())
        high = float(seg["High"].max())
        width = (high - low) / low if low > 0 else float("inf")
        if width > thr:
            # Por seguridad: mantenemos integridad de eventos.
            continue
        rows.append(
            {
                "event_id": i,
                "start_ts": ohlcv.index[s],
                "end_ts": ohlcv.index[e],
                "low_event": low,
                "high_event": high,
                "width_pct_event": width * 100.0,
                "bars_event": int(e - s + 1),
            }
        )
    return pd.DataFrame(rows)


def _assign_zones(events: pd.DataFrame, zone_width_pct: float) -> pd.DataFrame:
    if events.empty:
        return events.assign(zone_idx=pd.Series(dtype="int64"), zone_low=pd.Series(dtype="float64"), zone_high=pd.Series(dtype="float64"))

    zone_factor = 1.0 + float(zone_width_pct) / 100.0
    if zone_factor <= 1.0:
        raise ValueError("zone_width_pct debe ser > 0")

    p0 = float(events["low_event"].min())
    if p0 <= 0:
        raise ValueError("No se puede zonificar con low_event <= 0")

    centers = (events["low_event"].to_numpy(dtype="float64") + events["high_event"].to_numpy(dtype="float64")) / 2.0
    log_base = math.log(zone_factor)
    ratios = np.maximum(centers / p0, 1e-15)
    zone_idx = np.floor(np.log(ratios) / log_base).astype("int64")
    zone_low = p0 * np.power(zone_factor, zone_idx)
    zone_high = zone_low * zone_factor

    out = events.copy()
    out["zone_idx"] = zone_idx
    out["zone_low"] = zone_low
    out["zone_high"] = zone_high
    return out


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
    grp = grp[cols]
    return grp


def _build_discovered_outputs(events: pd.DataFrame, symbol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if events.empty:
        ranking = pd.DataFrame(
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
            ]
        )
        detail = pd.DataFrame(
            columns=[
                "symbol",
                "zone_idx",
                "zone_low",
                "zone_high",
                "event_id",
                "start_ts",
                "end_ts",
                "bars_event",
                "low_event",
                "high_event",
                "width_pct_event",
            ]
        )
        return ranking, detail

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
        # "Rango descubierto": todos los eventos del cluster comparten una intersección no vacía.
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

    # zone_idx por ranking: más eventos primero.
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
            "bars_event",
            "low_event",
            "high_event",
            "width_pct_event",
        ]
    ].copy()
    detail.insert(0, "symbol", symbol)
    detail = detail.sort_values(["zone_idx", "start_ts", "event_id"]).reset_index(drop=True)
    return ranking, detail


def _build_ranking_adaptive(events: pd.DataFrame, symbol: str, zone_width_pct: float) -> pd.DataFrame:
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
    if events.empty:
        return pd.DataFrame(columns=cols)

    zone_factor = 1.0 + float(zone_width_pct) / 100.0
    if zone_factor <= 1.0:
        raise ValueError("zone_width_pct debe ser > 0")

    ev = events.copy().reset_index(drop=True)
    ev["center_price"] = (ev["low_event"] + ev["high_event"]) / 2.0
    remaining = set(ev.index.tolist())
    rows: list[dict] = []
    rank = 1

    while remaining:
        rem = ev.loc[sorted(remaining)]
        candidates = np.unique(rem["low_event"].to_numpy(dtype="float64"))

        best_count = -1
        best_low = None
        best_mask = None
        for low in candidates:
            high = low * zone_factor
            mask = (rem["center_price"] >= low) & (rem["center_price"] < high)
            cnt = int(mask.sum())
            if cnt > best_count:
                best_count = cnt
                best_low = float(low)
                best_mask = mask
            elif cnt == best_count and cnt > 0 and best_low is not None and float(low) < best_low:
                # Empate: priorizamos la zona con límite inferior más bajo para estabilidad.
                best_low = float(low)
                best_mask = mask

        if best_count <= 0 or best_low is None or best_mask is None:
            break

        best_high = best_low * zone_factor
        selected_idx = rem.index[best_mask].tolist()
        sel = ev.loc[selected_idx]

        rows.append(
            {
                "symbol": symbol,
                "zone_idx": rank,
                "zone_low": best_low,
                "zone_high": best_high,
                "event_count": int(len(sel)),
                "avg_event_width_pct": float(sel["width_pct_event"].mean()),
                "avg_event_bars": float(sel["bars_event"].mean()),
                "first_event_ts": sel["start_ts"].min(),
                "last_event_ts": sel["end_ts"].max(),
            }
        )

        remaining -= set(selected_idx)
        rank += 1

    grp = pd.DataFrame(rows)
    if grp.empty:
        return pd.DataFrame(columns=cols)

    total_events = float(grp["event_count"].sum())
    grp["event_share_pct"] = (grp["event_count"] / total_events) * 100.0 if total_events > 0 else 0.0
    grp = grp.sort_values(["event_count", "zone_idx"], ascending=[False, True]).reset_index(drop=True)
    grp = grp[cols]
    return grp


def run(cfg: Config) -> int:
    by_symbol = _discover_monthly_raw(cfg.raw_root)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    all_rankings: list[pd.DataFrame] = []
    expected_months = _month_list(cfg.start_yyyymm, cfg.end_yyyymm)
    print(f"Meses esperados: {expected_months[0]}..{expected_months[-1]} ({len(expected_months)})", flush=True)

    for symbol in cfg.symbols:
        if symbol not in by_symbol:
            raise ValueError(f"No hay parquets para símbolo {symbol}")
        ohlcv = _build_ohlcv_symbol(cfg, symbol, by_symbol[symbol])
        events = _detect_events(ohlcv, cfg.window, cfg.range_max_pct)
        detail_events: pd.DataFrame | None = None
        if cfg.zone_mode == "grid":
            events_z = _assign_zones(events, cfg.zone_width_pct)
            ranking = _build_ranking(events_z, symbol)
        elif cfg.zone_mode == "adaptive":
            ranking = _build_ranking_adaptive(events, symbol, cfg.zone_width_pct)
        else:
            ranking, detail_events = _build_discovered_outputs(events, symbol)

        # Validaciones solicitadas.
        if not events.empty and (events["width_pct_event"] > cfg.range_max_pct + 1e-9).any():
            raise ValueError(f"{symbol}: hay eventos con width_pct_event > {cfg.range_max_pct}")
        if not ranking.empty and int(ranking["event_count"].sum()) != len(events):
            raise ValueError(f"{symbol}: sum(event_count) != cantidad de eventos")

        out_symbol = cfg.out_dir / f"rangos_{symbol}.csv"
        ranking.to_csv(out_symbol, index=False)
        print(f"[OK] {symbol} -> {out_symbol} | eventos={len(events)} | zonas={len(ranking)}", flush=True)

        if detail_events is not None:
            out_detail = cfg.out_dir / f"rangos_{symbol}_eventos.csv"
            detail_events.to_csv(out_detail, index=False)
            print(f"[OK] {symbol} detalle -> {out_detail} | filas={len(detail_events)}", flush=True)
        all_rankings.append(ranking)

    if all_rankings:
        summary = pd.concat(all_rankings, ignore_index=True)
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
            ]
        )
    if not summary.empty:
        summary = summary.sort_values(["symbol", "event_count", "zone_idx"], ascending=[True, False, True]).reset_index(drop=True)
        summary["rank_in_symbol"] = summary.groupby("symbol").cumcount().add(1)
        summary = summary.sort_values(["symbol", "rank_in_symbol"]).reset_index(drop=True)
    out_summary = cfg.out_dir / "rangos_resumen.csv"
    summary.to_csv(out_summary, index=False)
    print(f"[OK] resumen -> {out_summary}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Estadística de zonas de lateralización por símbolo.")
    parser.add_argument("--tf", default="30T")
    parser.add_argument("--window", type=int, default=48)
    parser.add_argument("--range-max-pct", type=float, default=10.0)
    parser.add_argument("--zone-width-pct", type=float, default=10.0)
    parser.add_argument("--zone-mode", choices=["discovered", "adaptive", "grid"], default="discovered")
    parser.add_argument("--tz", default="America/Argentina/Buenos_Aires")
    parser.add_argument("--raw-root", default="data/velas crudas")
    parser.add_argument("--out-dir", default="data/estadisticas")
    parser.add_argument("--symbols", default="BTCUSDT,ETHUSDT", help="CSV de símbolos")
    parser.add_argument("--start-yyyymm", default="202501")
    parser.add_argument("--end-yyyymm", default="202602")
    args = parser.parse_args()

    symbols = [s.strip() for s in str(args.symbols).split(",") if s.strip()]
    cfg = Config(
        tf=str(args.tf),
        window=int(args.window),
        range_max_pct=float(args.range_max_pct),
        zone_width_pct=float(args.zone_width_pct),
        zone_mode=str(args.zone_mode),
        tz=str(args.tz),
        raw_root=Path(args.raw_root),
        out_dir=Path(args.out_dir),
        symbols=symbols,
        start_yyyymm=str(args.start_yyyymm),
        end_yyyymm=str(args.end_yyyymm),
    )

    try:
        return run(cfg)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

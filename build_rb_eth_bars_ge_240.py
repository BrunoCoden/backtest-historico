#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _round2(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").round(2)
    return out


def _to_dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, format="%y-%m-%d %H:%M:%S", errors="coerce")


def run(events_csv: Path, out_dir: Path, bars_min: int, bars_mid_min: int, bars_mid_max: int) -> int:
    if not events_csv.exists():
        raise SystemExit(f"No existe: {events_csv}")
    out_dir.mkdir(parents=True, exist_ok=True)

    ev = pd.read_csv(events_csv)
    if ev.empty:
        raise SystemExit("Archivo de eventos vacío.")

    needed = {
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
    }
    missing = sorted(needed - set(ev.columns))
    if missing:
        raise SystemExit(f"Faltan columnas: {', '.join(missing)}")

    ev["bars_event"] = pd.to_numeric(ev["bars_event"], errors="coerce").fillna(0).astype(int)
    ev["zone_idx"] = pd.to_numeric(ev["zone_idx"], errors="coerce").fillna(-1).astype(int)

    mid = ev[(ev["bars_event"] >= bars_mid_min) & (ev["bars_event"] < bars_mid_max)]
    mid_count = mid.groupby("zone_idx").size().to_dict()

    filtered = ev[ev["bars_event"] >= bars_min].copy()
    filtered = filtered.sort_values(["zone_low", "zone_idx", "start_ts", "event_id"]).reset_index(drop=True)
    if filtered.empty:
        raise SystemExit(f"No hay eventos con bars_event >= {bars_min}.")

    filtered["Eventos 5-10 Dias Zona"] = filtered["zone_idx"].map(mid_count).fillna(0).astype(int)

    # 1) Eventos filtrados (cabeceras en español).
    eventos = pd.DataFrame(
        {
            "Simbolo": filtered["symbol"],
            "Zona": filtered["zone_idx"],
            "Zona Min": filtered["zone_low"],
            "Zona Max": filtered["zone_high"],
            "Evento ID": filtered["event_id"],
            "Inicio": filtered["start_ts"],
            "Fin": filtered["end_ts"],
            "Reset": filtered["reset_ts"],
            "Motivo Reset": filtered["reset_reason"],
            "Velas Evento": filtered["bars_event"],
            "Eventos 5-10 Dias Zona": filtered["Eventos 5-10 Dias Zona"],
            "Canal Medio": filtered["channel_value"],
            "Canal Superior": filtered["channel_upper"],
            "Canal Inferior": filtered["channel_lower"],
            "Canal Superior Medio": filtered["channel_upper_mid"],
            "Canal Inferior Medio": filtered["channel_lower_mid"],
            "Min Evento": filtered["low_event"],
            "Max Evento": filtered["high_event"],
            "Ancho % Evento": filtered["width_pct_event"],
        }
    )
    eventos = _round2(
        eventos,
        [
            "Zona Min",
            "Zona Max",
            "Canal Medio",
            "Canal Superior",
            "Canal Inferior",
            "Canal Superior Medio",
            "Canal Inferior Medio",
            "Min Evento",
            "Max Evento",
            "Ancho % Evento",
        ],
    )

    # 2) Detalle de canales.
    canales = pd.DataFrame(
        {
            "symbol": filtered["symbol"],
            "zone_idx": filtered["zone_idx"],
            "zone_low": filtered["zone_low"],
            "zone_high": filtered["zone_high"],
            "event_id": filtered["event_id"],
            "start_ts": filtered["start_ts"],
            "end_ts": filtered["end_ts"],
            "reset_reason": filtered["reset_reason"],
            "bars_event": filtered["bars_event"],
            "Canal Inferior": filtered["channel_lower"],
            "Canal Inferior Medio": filtered["channel_lower_mid"],
            "Canal Medio": filtered["channel_value"],
            "Canal Superior Medio": filtered["channel_upper_mid"],
            "Canal Superior": filtered["channel_upper"],
            "low_event": filtered["low_event"],
            "high_event": filtered["high_event"],
            "width_pct_event": filtered["width_pct_event"],
        }
    )
    canales = _round2(
        canales,
        [
            "zone_low",
            "zone_high",
            "Canal Inferior",
            "Canal Inferior Medio",
            "Canal Medio",
            "Canal Superior Medio",
            "Canal Superior",
            "low_event",
            "high_event",
            "width_pct_event",
        ],
    )

    # 3) Zonas promedio de canales.
    grp = filtered.groupby("zone_idx", as_index=False).agg(
        **{
            "Eventos Zona": ("event_id", "count"),
            "Zona Min": ("zone_low", "first"),
            "Zona Max": ("zone_high", "first"),
            "C. Inferior Prom": ("channel_lower", "mean"),
            "C. Inferior Medio Prom": ("channel_lower_mid", "mean"),
            "C. Medio Prom": ("channel_value", "mean"),
            "C. Superior Medio Prom": ("channel_upper_mid", "mean"),
            "C. Superior Prom": ("channel_upper", "mean"),
            "Primer Inicio": ("start_ts", "min"),
            "Ultimo Fin": ("end_ts", "max"),
        }
    )
    grp["Eventos 5-10 Dias Zona"] = grp["zone_idx"].map(mid_count).fillna(0).astype(int)
    grp["Rango Unico Min"] = grp["C. Inferior Prom"]
    grp["Rango Unico Max"] = grp["C. Superior Prom"]
    grp = grp.rename(columns={"zone_idx": "Zona"})
    grp = grp[
        [
            "Zona",
            "Eventos Zona",
            "Eventos 5-10 Dias Zona",
            "Zona Min",
            "Zona Max",
            "C. Inferior Prom",
            "C. Inferior Medio Prom",
            "C. Medio Prom",
            "C. Superior Medio Prom",
            "C. Superior Prom",
            "Rango Unico Min",
            "Rango Unico Max",
            "Primer Inicio",
            "Ultimo Fin",
        ]
    ].sort_values("Rango Unico Min", ascending=True)
    grp = _round2(
        grp,
        [
            "Zona Min",
            "Zona Max",
            "C. Inferior Prom",
            "C. Inferior Medio Prom",
            "C. Medio Prom",
            "C. Superior Medio Prom",
            "C. Superior Prom",
            "Rango Unico Min",
            "Rango Unico Max",
        ],
    )

    # Normalizar fechas por orden temporal real.
    for frame, c_in, c_out in [
        (eventos, "Inicio", "Inicio"),
        (eventos, "Fin", "Fin"),
        (eventos, "Reset", "Reset"),
        (canales, "start_ts", "start_ts"),
        (canales, "end_ts", "end_ts"),
        (grp, "Primer Inicio", "Primer Inicio"),
        (grp, "Ultimo Fin", "Ultimo Fin"),
    ]:
        dt = _to_dt(frame[c_in])
        frame[c_out] = dt.dt.strftime("%y-%m-%d %H:%M:%S").fillna(frame[c_in].astype(str))

    suffix = f"bars_ge_{bars_min}"
    out_eventos = out_dir / f"rangos_ETHUSDT_eventos_{suffix}.csv"
    out_canales = out_dir / f"rangos_ETHUSDT_canales_detalle_{suffix}.csv"
    out_zonas = out_dir / f"rangos_ETHUSDT_zonas_promedio_canales_{suffix}.csv"

    eventos.to_csv(out_eventos, index=False)
    canales.to_csv(out_canales, index=False)
    grp.to_csv(out_zonas, index=False)

    print(f"OK -> {out_eventos}")
    print(f"OK -> {out_canales}")
    print(f"OK -> {out_zonas}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Construye derivados de rangos RB para ETHUSDT (bars>=N).")
    parser.add_argument(
        "--events-csv",
        default="data/estadisticas/rb_1h_eth_202101_202602/rangos_ETHUSDT_eventos.csv",
    )
    parser.add_argument(
        "--out-dir",
        default="data/estadisticas/rb_1h_eth_202101_202602",
    )
    parser.add_argument("--bars-min", type=int, default=240)
    parser.add_argument("--bars-mid-min", type=int, default=120, help="Limite inferior para contar 5-10 dias en 1H.")
    parser.add_argument("--bars-mid-max", type=int, default=240, help="Limite superior exclusivo para contar 5-10 dias en 1H.")
    args = parser.parse_args()

    return run(
        events_csv=Path(args.events_csv),
        out_dir=Path(args.out_dir),
        bars_min=int(args.bars_min),
        bars_mid_min=int(args.bars_mid_min),
        bars_mid_max=int(args.bars_mid_max),
    )


if __name__ == "__main__":
    raise SystemExit(main())

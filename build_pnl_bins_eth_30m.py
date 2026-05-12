#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd


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


def _fmt_range(left: float, right: float, closed_right: bool) -> str:
    return f"[{left:.0f},{right:.0f}{']' if closed_right else ')'}"


def run(
    source_dir: Path,
    out_csv: Path,
    start_yyyymm: str,
    end_yyyymm: str,
    step_pct: float,
) -> int:
    months = _month_list(start_yyyymm, end_yyyymm)
    missing: list[str] = []
    values: list[float] = []

    for yyyymm in months:
        file_path = source_dir / f"ETHUSDT_{yyyymm}_bollinger_tv_30m_ctx.csv"
        if not file_path.exists():
            missing.append(str(file_path))
            continue

        df = pd.read_csv(file_path)
        if "PNL Cierre %" not in df.columns:
            raise RuntimeError(f"Falta columna 'PNL Cierre %' en {file_path}")

        pnl = pd.to_numeric(df["PNL Cierre %"], errors="coerce")
        pos = pnl[pnl > 0].dropna()
        values.extend(pos.tolist())

    if missing:
        raise RuntimeError(
            "Faltan archivos mensuales para el período solicitado:\n" + "\n".join(missing)
        )

    if len(months) != 14:
        raise RuntimeError(f"Se esperaban 14 meses y se obtuvieron {len(months)} ({months[0]}..{months[-1]}).")

    if not values:
        out_df = pd.DataFrame(columns=["Rango PNL %", "Cantidad Eventos"])
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out_csv, index=False)
        print("No hay eventos positivos (PNL Cierre % > 0) en el período.")
        print(f"OK -> {out_csv}")
        print("Eventos positivos totales: 0")
        return 0

    max_pos = max(values)
    end = math.ceil(max_pos / step_pct) * step_pct
    # Asegura al menos un bucket.
    if end <= 0:
        end = step_pct
    edges = [0.0]
    cur = 0.0
    while cur < end:
        cur += step_pct
        edges.append(cur)

    counts = [0 for _ in range(len(edges) - 1)]
    for v in values:
        idx = int((v - 0.0) // step_pct)
        if idx >= len(counts):
            idx = len(counts) - 1
        counts[idx] += 1

    rows = []
    for i, c in enumerate(counts):
        left = edges[i]
        right = edges[i + 1]
        closed_right = i == len(counts) - 1
        rows.append({"Rango PNL %": _fmt_range(left, right, closed_right), "Cantidad Eventos": int(c)})

    out_df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)

    print(f"OK -> {out_csv}")
    print(f"Meses procesados: {months[0]}..{months[-1]} ({len(months)})")
    print(f"Eventos positivos totales: {len(values)}")
    print(f"Suma tabla: {int(out_df['Cantidad Eventos'].sum())}")
    print(f"Rango máximo observado: {max_pos:.6f}%")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Distribución de PNL Cierre % positivo por bins de 2% (ETHUSDT 30m).")
    parser.add_argument(
        "--source-dir",
        default="data/velas_30m/ETHUSDT_ctx",
        help="Carpeta con CSV mensuales ETHUSDT 30m.",
    )
    parser.add_argument(
        "--out-csv",
        default="data/estadisticas/pnl_eth_30m_202501_202602_bins_2pct.csv",
        help="CSV de salida con la tabla de bins.",
    )
    parser.add_argument("--start-yyyymm", default="202501")
    parser.add_argument("--end-yyyymm", default="202602")
    parser.add_argument("--step-pct", type=float, default=2.0)
    args = parser.parse_args()

    return run(
        source_dir=Path(args.source_dir),
        out_csv=Path(args.out_csv),
        start_yyyymm=str(args.start_yyyymm),
        end_yyyymm=str(args.end_yyyymm),
        step_pct=float(args.step_pct),
    )


if __name__ == "__main__":
    raise SystemExit(main())

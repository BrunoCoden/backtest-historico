#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _build_ohlcv(df: pd.DataFrame, rule: str, tz: str) -> pd.DataFrame:
    if "timestamp_ms_utc" not in df.columns:
        raise ValueError("Falta columna timestamp_ms_utc en el parquet.")
    if "price" not in df.columns or "qty" not in df.columns:
        raise ValueError("Faltan columnas price/qty en el parquet.")

    dt = pd.to_datetime(df["timestamp_ms_utc"], unit="ms", utc=True)
    if tz:
        dt = dt.dt.tz_convert(tz)
    df = df.copy()
    df["dt"] = dt
    df = df.set_index("dt")

    ohlc = df["price"].resample(rule).ohlc()
    vol = df["qty"].resample(rule).sum()
    out = ohlc.join(vol).dropna()
    out.columns = ["Open", "High", "Low", "Close", "Volume"]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Plotly OHLCV desde parquet de aggTrades")
    parser.add_argument("parquet_path", help="Ruta al .parquet")
    parser.add_argument("timeframe", help="Timeframe pandas (ej: 30T, 1H, 5T)")
    parser.add_argument("out_html", help="Salida HTML (ej: /home/diego/data/eth_30m.html)")
    parser.add_argument("--tz", default="America/Argentina/Buenos_Aires", help="Timezone (default: America/Argentina/Buenos_Aires)")
    parser.add_argument("--out-png", default="", help="Salida PNG (requiere kaleido)")
    args = parser.parse_args()

    parquet_path = Path(args.parquet_path)
    if not parquet_path.exists():
        print(f"No existe: {parquet_path}", file=sys.stderr)
        return 1

    df = pd.read_parquet(parquet_path)
    ohlcv = _build_ohlcv(df, args.timeframe, args.tz)
    if ohlcv.empty:
        print("No hay datos para el rango/timeframe elegido.", file=sys.stderr)
        return 1

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.75, 0.25],
    )

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

    fig.add_trace(
        go.Bar(
            x=ohlcv.index,
            y=ohlcv["Volume"],
            name="Volume",
            marker_color="rgba(100, 149, 237, 0.6)",
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        title=f"OHLCV {args.timeframe}",
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        height=800,
    )

    out_html = Path(args.out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_html)
    print(f"OK HTML -> {out_html}")

    if args.out_png:
        out_png = Path(args.out_png)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        try:
            fig.write_image(out_png)
            print(f"OK PNG -> {out_png}")
        except Exception as exc:
            print(f"No se pudo exportar PNG (necesita kaleido): {exc}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

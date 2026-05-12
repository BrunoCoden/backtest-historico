#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

def _add_hatch(
    fig: go.Figure,
    *,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    color: str,
    slope_up: bool,
) -> None:
    if y1 <= y0 or x1 <= x0:
        return
    h = y1 - y0
    # Mantener ~6-12 lineas por banda.
    n = max(6, min(12, int(math.ceil(h / max(h * 0.12, 1e-9)))))
    dy = h * 0.45
    xs: list[float] = []
    ys: list[float] = []
    for i in range(n):
        ys0 = y0 + (i + 0.5) * (h / n)
        ys1 = ys0 + dy if slope_up else ys0 - dy
        ys1 = min(max(ys1, y0), y1)
        xs.extend([x0, x1, float("nan")])
        ys.extend([ys0, ys1, float("nan")])
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            line={"color": color, "width": 1},
            hoverinfo="skip",
            showlegend=False,
        )
    )


def build_chart(df: pd.DataFrame) -> go.Figure:
    data = df.sort_values("Rango Unico Min").reset_index(drop=True)
    fig = go.Figure()
    y_min = float(data["Rango Unico Min"].min())
    y_max = float(data["Rango Unico Max"].max())
    pad = max((y_max - y_min) * 0.08, 1.0)
    max_events = float(data["Eventos Zona"].max()) if "Eventos Zona" in data.columns else 1.0
    if max_events <= 0:
        max_events = 1.0

    x_cursor = 0.0
    gap = 0.38

    # Dibujar cada zona como bloque cerrado independiente (sin solape visual).
    for i, (_, row) in enumerate(data.iterrows()):
        zona = int(row["Zona"])
        y0 = float(row["Rango Unico Min"])
        y1 = float(row["Rango Unico Max"])
        ev = float(row["Eventos Zona"]) if "Eventos Zona" in row else 1.0
        width = 0.9 + 1.2 * (ev / max_events)  # Ancho relativo por cantidad de eventos.
        x0 = x_cursor
        x1 = x_cursor + width
        x_cursor = x1 + gap

        fig.add_shape(
            type="rect",
            xref="x",
            yref="y",
            x0=x0,
            x1=x1,
            y0=y0,
            y1=y1,
            fillcolor="rgba(0,0,0,0.06)",
            line={"color": "rgba(20,20,20,0.95)", "width": 2},
            layer="below",
        )

        # Sub-bandas de canales dentro de la zona.
        c_inf = float(row["C. Inferior Prom"])
        c_inf_mid = float(row["C. Inferior Medio Prom"])
        c_sup_mid = float(row["C. Superior Medio Prom"])
        c_sup = float(row["C. Superior Prom"])

        low0, low1 = sorted((c_inf, c_inf_mid))
        up0, up1 = sorted((c_sup_mid, c_sup))

        # Clip de seguridad al rango unico.
        low0, low1 = max(low0, y0), min(low1, y1)
        up0, up1 = max(up0, y0), min(up1, y1)

        fig.add_shape(
            type="rect",
            xref="x",
            yref="y",
            x0=x0,
            x1=x1,
            y0=low0,
            y1=low1,
            fillcolor="rgba(40,40,40,0.08)",
            line={"color": "rgba(30,30,30,0.9)", "width": 1},
            layer="above",
        )
        _add_hatch(
            fig,
            x0=x0,
            x1=x1,
            y0=low0,
            y1=low1,
            color="rgba(30,30,30,0.35)",
            slope_up=False,
        )

        fig.add_shape(
            type="rect",
            xref="x",
            yref="y",
            x0=x0,
            x1=x1,
            y0=up0,
            y1=up1,
            fillcolor="rgba(40,40,40,0.08)",
            line={"color": "rgba(30,30,30,0.9)", "width": 1},
            layer="above",
        )
        _add_hatch(
            fig,
            x0=x0,
            x1=x1,
            y0=up0,
            y1=up1,
            color="rgba(30,30,30,0.35)",
            slope_up=True,
        )

        # Marcar el canal medio.
        c_mid = float(row["C. Medio Prom"])
        if y0 <= c_mid <= y1:
            fig.add_shape(
                type="line",
                xref="x",
                yref="y",
                x0=x0,
                x1=x1,
                y0=c_mid,
                y1=c_mid,
                line={"color": "rgba(40,40,40,0.65)", "width": 1, "dash": "dot"},
                layer="above",
            )

        # Etiqueta dentro del bloque.
        fig.add_annotation(
            x=(x0 + x1) / 2.0,
            y=(y0 + y1) / 2.0,
            xref="x",
            yref="y",
            text=f"Z{zona}",
            showarrow=False,
            font={"size": 11, "color": "#1f2d3d"},
            bgcolor="rgba(255,255,255,0.55)",
        )
        # Valores sobre las lineas de precio del evento.
        x_lbl = x1 - max(width * 0.03, 0.02)
        for yv in [y1, c_sup_mid, c_inf_mid, y0]:
            fig.add_annotation(
                x=x_lbl,
                y=yv,
                xref="x",
                yref="y",
                showarrow=False,
                xanchor="right",
                yanchor="middle",
                text=f"{yv:.2f}",
                font={"size": 9, "color": "#111"},
                bgcolor="rgba(255,255,255,0.72)",
                bordercolor="rgba(80,80,80,0.65)",
                borderwidth=1,
            )

    # Eje X "mudo", solo para extender las franjas.
    fig.update_xaxes(
        range=[-0.15, x_cursor + 1.00],
        showticklabels=False,
        showgrid=False,
        zeroline=False,
        title_text="",
    )
    # Traza invisible para forzar autorango/escala visible.
    fig.add_trace(
        go.Scatter(
            x=[0.0, max(x_cursor, 1.0)],
            y=[y_min, y_max],
            mode="markers",
            marker={"opacity": 0},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.update_yaxes(range=[y_min - pad, y_max + pad])

    fig.update_layout(
        title="ETH - Zonas de Precio (Eventos Cerrados)",
        yaxis_title="Precio",
        template="plotly_white",
        margin={"l": 70, "r": 90, "t": 60, "b": 40},
        showlegend=False,
    )

    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grafico de zonas por bandas horizontales (croquis).")
    parser.add_argument(
        "--input",
        default="data/estadisticas/rb_eth_bars_ge_240/rangos_ETHUSDT_zonas_promedio_canales_bars_ge_240.csv",
        help="CSV de zonas promedio.",
    )
    parser.add_argument(
        "--output",
        default="data/estadisticas/rb_eth_bars_ge_240/rangos_ETHUSDT_zonas_croquis.html",
        help="HTML de salida.",
    )
    parser.add_argument("--show", action="store_true", help="Abrir grafico al finalizar.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    in_path = Path(args.input)
    out_path = Path(args.output)

    if not in_path.exists():
        raise SystemExit(f"No existe el input: {in_path}")

    df = pd.read_csv(in_path)
    required = {
        "Zona",
        "Rango Unico Min",
        "Rango Unico Max",
        "C. Inferior Prom",
        "C. Inferior Medio Prom",
        "C. Medio Prom",
        "C. Superior Medio Prom",
        "C. Superior Prom",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"Faltan columnas requeridas: {', '.join(missing)}")

    fig = build_chart(df)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs=True)
    print(f"OK: grafico generado -> {out_path}")
    if args.show:
        fig.show()


if __name__ == "__main__":
    main()

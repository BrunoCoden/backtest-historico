#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parent
RAW_ROOT = ROOT / "data" / "velas crudas"
OUT_STATS_DIR = ROOT / "data" / "estadisticas" / "pnl_mensual_202501_202602"
TMP_CALC_DIR = OUT_STATS_DIR / "_tmp_calc_inputs"

TZ_BA = "America/Argentina/Buenos_Aires"
START_YYYYMM = "202501"
END_YYYYMM = "202602"
PNL_COL = "PNL Cierre %"
TOTAL_LABEL = "TOTAL PNL %"


@dataclass(frozen=True)
class Combo:
    symbol: str
    tf_label: str
    tf_rule: str
    out_dir: Path
    filename_tpl: str


COMBOS = [
    Combo(
        symbol="ETHUSDT",
        tf_label="30m",
        tf_rule="30T",
        out_dir=ROOT / "data" / "velas_30m" / "ETHUSDT_ctx",
        filename_tpl="ETHUSDT_{yyyymm}_bollinger_tv_30m_ctx.csv",
    ),
    Combo(
        symbol="ETHUSDT",
        tf_label="1h",
        tf_rule="1h",
        out_dir=ROOT / "data" / "velas_1h" / "ETHUSDT_202101_202602",
        filename_tpl="ETHUSDT_{yyyymm}_bollinger_tv_1h.csv",
    ),
    Combo(
        symbol="BTCUSDT",
        tf_label="30m",
        tf_rule="30T",
        out_dir=ROOT / "data" / "velas_30m" / "BTCUSDT_ctx",
        filename_tpl="BTCUSDT_{yyyymm}_bollinger_tv_30m_ctx.csv",
    ),
    Combo(
        symbol="BTCUSDT",
        tf_label="1h",
        tf_rule="1h",
        out_dir=ROOT / "data" / "velas_1h" / "BTCUSDT_202501_202602",
        filename_tpl="BTCUSDT_{yyyymm}_bollinger_tv_1h.csv",
    ),
]


def month_list(start_yyyymm: str, end_yyyymm: str) -> list[str]:
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


def prev_month(yyyymm: str) -> str:
    y, m = int(yyyymm[:4]), int(yyyymm[4:6])
    if m == 1:
        return f"{y - 1:04d}12"
    return f"{y:04d}{m - 1:02d}"


def month_bounds_ba(yyyymm: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    y, m = int(yyyymm[:4]), int(yyyymm[4:6])
    start = pd.Timestamp(year=y, month=m, day=1, hour=0, minute=0, second=0, tz=TZ_BA)
    end = (start + pd.offsets.MonthBegin(1)) - pd.Timedelta(seconds=1)
    return start, end


def raw_path(symbol: str, yyyymm: str) -> Path:
    return RAW_ROOT / symbol / f"{symbol}_{yyyymm}_1s_ohlc.parquet"


def has_total_row(csv_path: Path) -> bool:
    if not csv_path.exists():
        return False
    try:
        df = pd.read_csv(csv_path, usecols=["Fecha", PNL_COL])
    except Exception:
        return False
    mask = df["Fecha"].astype(str).eq(TOTAL_LABEL)
    if not mask.any():
        return False
    val = pd.to_numeric(df.loc[mask, PNL_COL], errors="coerce").iloc[-1]
    return pd.notna(val)


def extract_monthly_pnl(csv_path: Path) -> float:
    df = pd.read_csv(csv_path)
    if "Fecha" in df.columns and PNL_COL in df.columns:
        mask = df["Fecha"].astype(str).eq(TOTAL_LABEL)
        if mask.any():
            val = pd.to_numeric(df.loc[mask, PNL_COL], errors="coerce").iloc[-1]
            if pd.notna(val):
                return float(val)
        base = df.loc[~mask, PNL_COL]
        val = pd.to_numeric(base, errors="coerce").sum(min_count=1)
        return float(0.0 if pd.isna(val) else val)
    if PNL_COL in df.columns:
        val = pd.to_numeric(df[PNL_COL], errors="coerce").sum(min_count=1)
        return float(0.0 if pd.isna(val) else val)
    return 0.0


def _pick_ts_col(cols: list[str]) -> str | None:
    lc = {c.lower(): c for c in cols}
    return lc.get("timestamp_ms_utc") or lc.get("bucket_start_ms_utc") or lc.get("bucket_start")


def build_calc_input(symbol: str, yyyymm: str) -> tuple[Path, bool]:
    cur = raw_path(symbol, yyyymm)
    if not cur.exists():
        raise FileNotFoundError(f"Falta parquet mensual: {cur}")

    prev = raw_path(symbol, prev_month(yyyymm))
    if not prev.exists():
        return cur, False

    TMP_CALC_DIR.mkdir(parents=True, exist_ok=True)
    out = TMP_CALC_DIR / f"{symbol}_{prev_month(yyyymm)}_{yyyymm}_calc_ctx.parquet"

    # Leemos columnas relevantes para resample OHLC.
    schema_cols = pq.ParquetFile(cur).schema.names
    keep_candidates = [
        "timestamp_ms_utc",
        "bucket_start_ms_utc",
        "bucket_start",
        "open",
        "high",
        "low",
        "close",
        "Open",
        "High",
        "Low",
        "Close",
        "volume",
        "Volume",
        "qty",
    ]
    keep_cols = [c for c in keep_candidates if c in schema_cols]
    if not keep_cols:
        keep_cols = None

    d1 = pd.read_parquet(prev, columns=keep_cols)
    d2 = pd.read_parquet(cur, columns=keep_cols)
    merged = pd.concat([d1, d2], ignore_index=True)
    del d1, d2

    ts_col = _pick_ts_col(list(merged.columns))
    if ts_col is not None:
        merged = merged.sort_values(ts_col)
        merged = merged.drop_duplicates(subset=[ts_col], keep="last")

    merged.to_parquet(out, index=False)
    return out, True


def iso(ts: pd.Timestamp) -> str:
    return ts.isoformat()


def run_export_month(combo: Combo, yyyymm: str, out_csv: Path, sl_intrabar_1s: bool) -> tuple[str, str, float]:
    start_ts, end_ts = month_bounds_ba(yyyymm)
    prev_ym = prev_month(yyyymm)
    prev_start_ts, _ = month_bounds_ba(prev_ym)
    calc_start_ts = prev_start_ts if raw_path(combo.symbol, prev_ym).exists() else start_ts
    calc_end_ts = end_ts

    t0 = time.monotonic()
    calc_input, is_tmp = build_calc_input(combo.symbol, yyyymm)
    err = ""
    status = "ok"
    try:
        cmd = [
            sys.executable,
            "export_tabla_senales.py",
            str(calc_input),
            "--strategy",
            "bollinger",
            "--bb-profile",
            "tradingview",
            "--bb-price-source",
            "close",
            "--tf",
            combo.tf_rule,
            "--no-expand-sl-tp",
            "--start",
            iso(start_ts),
            "--end",
            iso(end_ts),
            "--calc-start",
            iso(calc_start_ts),
            "--calc-end",
            iso(calc_end_ts),
            "--out",
            str(out_csv),
        ]
        if sl_intrabar_1s:
            cmd.append("--sl-intrabar-1s")
        proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
        if proc.returncode != 0:
            status = "error"
            err = f"exit={proc.returncode}; stderr={proc.stderr.strip()[:600]}"
        elif not has_total_row(out_csv):
            status = "error"
            err = "CSV generado sin fila TOTAL PNL % válida"
    finally:
        if is_tmp and calc_input.exists():
            calc_input.unlink()

    return status, err, (time.monotonic() - t0)


def precheck_raw(months: list[str], symbols: list[str]) -> dict[str, list[str]]:
    missing: dict[str, list[str]] = {}
    for sym in symbols:
        miss = []
        for ym in months:
            if not raw_path(sym, ym).exists():
                miss.append(ym)
        if miss:
            missing[sym] = miss
    return missing


def combo_out_path(combo: Combo, yyyymm: str) -> Path:
    combo.out_dir.mkdir(parents=True, exist_ok=True)
    return combo.out_dir / combo.filename_tpl.format(yyyymm=yyyymm)


def _append_total_row(df: pd.DataFrame, pnl_col: str = "pnl_total_pct") -> pd.DataFrame:
    out = df.copy()
    pnl_sum = float(pd.to_numeric(out[pnl_col], errors="coerce").sum(min_count=1) or 0.0)
    total_row = {c: "" for c in out.columns}
    if "symbol" in total_row and len(out) > 0:
        total_row["symbol"] = str(out.iloc[0]["symbol"])
    if "timeframe" in total_row and len(out) > 0:
        total_row["timeframe"] = str(out.iloc[0]["timeframe"])
    if "yyyymm" in total_row:
        total_row["yyyymm"] = TOTAL_LABEL
    if "status" in total_row:
        total_row["status"] = "total"
    total_row[pnl_col] = pnl_sum
    return pd.concat([out, pd.DataFrame([total_row])], ignore_index=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Construye tablas mensuales de PnL (202501..202602) para ETH/BTC en 30m/1h."
    )
    parser.add_argument("--force", action="store_true", help="Recalcula todos los meses aunque el CSV ya exista con TOTAL PNL %.")
    parser.add_argument(
        "--sl-intrabar-1s",
        action="store_true",
        help="Usa validación intravela 1s para toques SL/TP al calcular PnL.",
    )
    args = parser.parse_args()

    t_all0 = time.monotonic()
    months = month_list(START_YYYYMM, END_YYYYMM)
    symbols = sorted({c.symbol for c in COMBOS})

    missing = precheck_raw(months, symbols)
    if missing:
        print("ERROR: faltan parquets crudos 1s en 202501..202602")
        for sym, miss in missing.items():
            print(f"- {sym}: faltan {len(miss)} meses -> {', '.join(miss)}")
        return 1

    print("Precheck OK: crudos completos para ETHUSDT y BTCUSDT (202501..202602).", flush=True)

    status_map: dict[tuple[str, str, str], str] = {}
    err_map: dict[tuple[str, str, str], str] = {}
    elapsed_by_combo: dict[tuple[str, str], float] = {}

    total_jobs = len(COMBOS) * len(months)
    job_idx = 0

    for combo in COMBOS:
        key_combo = (combo.symbol, combo.tf_label)
        elapsed_by_combo[key_combo] = 0.0
        for ym in months:
            job_idx += 1
            out_csv = combo_out_path(combo, ym)
            key = (combo.symbol, combo.tf_label, ym)

            if (not args.force) and has_total_row(out_csv):
                status_map[key] = "skipped_reuse"
                print(f"[{job_idx}/{total_jobs}] {combo.symbol} {combo.tf_label} {ym} -> skipped_reuse", flush=True)
                continue

            print(f"[{job_idx}/{total_jobs}] {combo.symbol} {combo.tf_label} {ym} -> generating", flush=True)
            st, err, elapsed = run_export_month(combo, ym, out_csv, sl_intrabar_1s=bool(args.sl_intrabar_1s))
            elapsed_by_combo[key_combo] += elapsed
            status_map[key] = st
            if err:
                err_map[key] = err
                print(f"  ERROR: {err}", flush=True)
            else:
                print(f"  OK: {out_csv}", flush=True)

    OUT_STATS_DIR.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict] = []
    all_ok = True

    for combo in COMBOS:
        combo_rows: list[dict] = []
        for ym in months:
            key = (combo.symbol, combo.tf_label, ym)
            out_csv = combo_out_path(combo, ym)
            st = status_map.get(key, "missing")
            pnl_val: float | None = None

            if out_csv.exists():
                try:
                    pnl_val = extract_monthly_pnl(out_csv)
                    if st == "missing":
                        st = "ok"
                except Exception as exc:
                    st = "error"
                    err_map[key] = f"extract_pnl_error={str(exc)[:400]}"

            if st in ("missing", "error"):
                all_ok = False

            start_ts, end_ts = month_bounds_ba(ym)
            combo_rows.append(
                {
                    "symbol": combo.symbol,
                    "timeframe": combo.tf_label,
                    "yyyymm": ym,
                    "period_start_ba": iso(start_ts),
                    "period_end_ba": iso(end_ts),
                    "pnl_total_pct": pnl_val,
                    "source_csv": str(out_csv),
                    "status": st,
                }
            )

        df_combo = pd.DataFrame(combo_rows).sort_values("yyyymm").reset_index(drop=True)
        df_combo_out = _append_total_row(df_combo, pnl_col="pnl_total_pct")
        out_table = OUT_STATS_DIR / f"pnl_mensual_{combo.symbol}_{combo.tf_label}_bollinger.csv"
        df_combo_out.to_csv(out_table, index=False)
        out_table_local = combo.out_dir / f"{combo.symbol}_202501_202602_pnl_mensual_{combo.tf_label}_bollinger.csv"
        df_combo_out.to_csv(out_table_local, index=False)

        ok_count = int(df_combo["status"].isin(["ok", "skipped_reuse"]).sum())
        skip_count = int((df_combo["status"] == "skipped_reuse").sum())
        miss_count = int((df_combo["status"] == "missing").sum())
        err_count = int((df_combo["status"] == "error").sum())
        pnl_sum = float(pd.to_numeric(df_combo["pnl_total_pct"], errors="coerce").sum(min_count=1) or 0.0)
        pnl_series = pd.to_numeric(df_combo["pnl_total_pct"], errors="coerce").dropna()
        pos_count = int((pnl_series > 0).sum())
        neg_count = int((pnl_series < 0).sum())

        manifest_rows.append(
            {
                "combinacion": f"{combo.symbol}_{combo.tf_label}",
                "symbol": combo.symbol,
                "timeframe": combo.tf_label,
                "meses_esperados": len(months),
                "meses_procesados_ok": ok_count,
                "meses_skipped_reuse": skip_count,
                "meses_faltantes": miss_count,
                "meses_error": err_count,
                "pnl_acumulado_pct": pnl_sum,
                "meses_pnl_positivo": pos_count,
                "meses_pnl_negativo": neg_count,
                "elapsed_sec": round(elapsed_by_combo.get((combo.symbol, combo.tf_label), 0.0), 3),
                "sl_intrabar_1s": bool(args.sl_intrabar_1s),
                "force_rebuild": bool(args.force),
                "out_table": str(out_table),
                "out_table_local": str(out_table_local),
            }
        )

        print(
            f"[RESUMEN] {combo.symbol} {combo.tf_label}: "
            f"ok={ok_count}/{len(months)} skip={skip_count} miss={miss_count} err={err_count} "
            f"pnl_acum={pnl_sum:.6f}% pos={pos_count} neg={neg_count}",
            flush=True,
        )

    manifest = pd.DataFrame(manifest_rows).sort_values(["symbol", "timeframe"]).reset_index(drop=True)
    manifest_path = OUT_STATS_DIR / "manifest_pnl_4tablas.csv"
    manifest.to_csv(manifest_path, index=False)

    if TMP_CALC_DIR.exists():
        # Debe quedar vacío porque borramos cada tmp al terminar.
        leftover = list(TMP_CALC_DIR.glob("*.parquet"))
        if not leftover:
            TMP_CALC_DIR.rmdir()

    print(f"Manifest -> {manifest_path}", flush=True)
    print(f"Elapsed total: {time.monotonic() - t_all0:.2f}s", flush=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

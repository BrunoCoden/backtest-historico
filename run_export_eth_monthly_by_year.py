#!/usr/bin/env python3
from __future__ import annotations

import calendar
import csv
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "velas crudas" / "ETHUSDT"
ANUAL_DIR = RAW_DIR / "anual"
OUT_DIR = ROOT / "data" / "velas_30m" / "ETHUSDT_ctx"
TMP_DIR = OUT_DIR / "_tmp_yearly"
MANIFEST_PATH = OUT_DIR / "manifest_eth_ctx_by_year_wm1_202101_202602.csv"

START_YYYYMM = "202101"
END_YYYYMM = "202602"
WARMUP_MONTHS = 1


@dataclass(frozen=True)
class MonthSlot:
    year: int
    month: int

    @property
    def yyyymm(self) -> str:
        return f"{self.year:04d}{self.month:02d}"

    @property
    def start_iso(self) -> str:
        return f"{self.year:04d}-{self.month:02d}-01T00:00:00-03:00"

    @property
    def end_iso(self) -> str:
        last = calendar.monthrange(self.year, self.month)[1]
        return f"{self.year:04d}-{self.month:02d}-{last:02d}T23:59:59-03:00"

    @property
    def start_naive(self) -> pd.Timestamp:
        return pd.Timestamp(f"{self.year:04d}-{self.month:02d}-01 00:00:00")

    @property
    def end_naive(self) -> pd.Timestamp:
        last = calendar.monthrange(self.year, self.month)[1]
        return pd.Timestamp(f"{self.year:04d}-{self.month:02d}-{last:02d} 23:59:59")


def month_iter(start_yyyymm: str, end_yyyymm: str) -> list[MonthSlot]:
    sy, sm = int(start_yyyymm[:4]), int(start_yyyymm[4:6])
    ey, em = int(end_yyyymm[:4]), int(end_yyyymm[4:6])
    out: list[MonthSlot] = []
    y, m = sy, sm
    while (y < ey) or (y == ey and m <= em):
        out.append(MonthSlot(y, m))
        m += 1
        if m > 12:
            y += 1
            m = 1
    return out


def _full_year_span(months: list[MonthSlot]) -> bool:
    return (
        len(months) == 12
        and months[0].month == 1
        and months[-1].month == 12
        and months[0].year == months[-1].year
    )


def merge_monthly_parquets(inputs: list[Path], out_path: Path) -> None:
    missing = [str(p) for p in inputs if not p.exists() or p.stat().st_size <= 0]
    if missing:
        raise RuntimeError(f"Faltan parquets mensuales para merge: {len(missing)}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".parquet.part")
    if tmp.exists():
        tmp.unlink()

    writer = None
    rows = 0
    try:
        for i, p in enumerate(inputs, start=1):
            print(f"[MERGE {i}/{len(inputs)}] {p.name}", flush=True)
            pf = pq.ParquetFile(p)
            for rg in range(pf.num_row_groups):
                t = pf.read_row_group(rg)
                rows += t.num_rows
                if writer is None:
                    writer = pq.ParquetWriter(tmp, t.schema, compression="zstd")
                writer.write_table(t)
        if writer is not None:
            writer.close()
            writer = None
        tmp.replace(out_path)
    finally:
        if writer is not None:
            writer.close()
        if tmp.exists() and not out_path.exists():
            tmp.unlink(missing_ok=True)

    print(f"[MERGE] OK rows={rows} -> {out_path}", flush=True)


def get_calc_input(calc_months: list[MonthSlot]) -> Path:
    if _full_year_span(calc_months):
        annual_path = ANUAL_DIR / f"ETHUSDT_{calc_months[0].year:04d}_1s_ohlc.parquet"
        if annual_path.exists() and annual_path.stat().st_size > 0:
            return annual_path

    start_yyyymm = calc_months[0].yyyymm
    end_yyyymm = calc_months[-1].yyyymm
    tmp_path = TMP_DIR / f"ETHUSDT_calc_{start_yyyymm}_{end_yyyymm}_1s_ohlc.parquet"
    if tmp_path.exists() and tmp_path.stat().st_size > 0:
        return tmp_path

    inputs = [RAW_DIR / f"ETHUSDT_{m.yyyymm}_1s_ohlc.parquet" for m in calc_months]
    merge_monthly_parquets(inputs, tmp_path)
    return tmp_path


def split_yearly_to_monthly(full_csv: Path, months: list[MonthSlot]) -> tuple[int, int]:
    df = pd.read_csv(full_csv)
    if "Fecha" not in df.columns or "Hora" not in df.columns:
        raise RuntimeError(f"CSV inesperado (faltan Fecha/Hora): {full_csv}")

    base = df[df["Fecha"].astype(str) != "TOTAL PNL %"].copy()
    dt = pd.to_datetime(base["Fecha"].astype(str) + " " + base["Hora"].astype(str), errors="coerce")
    base["_dt"] = dt

    written = 0
    for m in months:
        out_csv = OUT_DIR / f"ETHUSDT_{m.yyyymm}_bollinger_tv_30m_ctx.csv"
        mask = (base["_dt"] >= m.start_naive) & (base["_dt"] <= m.end_naive)
        month_df = base.loc[mask].drop(columns=["_dt"]).copy()

        total_row = {c: "" for c in month_df.columns}
        total_row["Fecha"] = "TOTAL PNL %"
        if "PNL Cierre %" in month_df.columns:
            pnl = pd.to_numeric(month_df["PNL Cierre %"], errors="coerce").sum()
            total_row["PNL Cierre %"] = pnl

        final_df = pd.concat([month_df, pd.DataFrame([total_row])], ignore_index=True)
        final_df.to_csv(out_csv, index=False)
        written += 1

    return written, len(base)


def run() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    months = month_iter(START_YYYYMM, END_YYYYMM)

    by_year: dict[int, list[MonthSlot]] = {}
    for m in months:
        by_year.setdefault(m.year, []).append(m)
    month_to_idx = {m.yyyymm: idx for idx, m in enumerate(months)}

    py = sys.executable
    rows: list[dict[str, str]] = []
    t0_all = time.time()
    years = sorted(by_year.keys())

    for i, year in enumerate(years, start=1):
        year_months = by_year[year]
        y_start = year_months[0].start_iso
        y_end = year_months[-1].end_iso
        first_idx = month_to_idx[year_months[0].yyyymm]
        last_idx = month_to_idx[year_months[-1].yyyymm]
        calc_first_idx = max(0, first_idx - WARMUP_MONTHS)
        calc_months = months[calc_first_idx : last_idx + 1]
        calc_start = calc_months[0].start_iso
        calc_end = calc_months[-1].end_iso
        t0 = time.time()
        status = "ok"
        error = ""

        try:
            in_parquet = get_calc_input(calc_months)
            out_full = TMP_DIR / f"ETHUSDT_{year:04d}_bollinger_tv_30m_ctx_full.csv"
            cmd = [
                py,
                "export_tabla_senales.py",
                str(in_parquet),
                "--strategy",
                "bollinger",
                "--bb-profile",
                "tradingview",
                "--bb-price-source",
                "close",
                "--tf",
                "30T",
                "--no-expand-sl-tp",
                "--calc-start",
                calc_start,
                "--calc-end",
                calc_end,
                "--start",
                y_start,
                "--end",
                y_end,
                "--out",
                str(out_full),
            ]

            print(
                f"[YEAR {i}/{len(years)}] RUN {year} ({len(year_months)} meses) "
                f"warmup={WARMUP_MONTHS}m calc_span={calc_months[0].yyyymm}->{calc_months[-1].yyyymm}",
                flush=True,
            )
            proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
            if proc.stdout:
                print(proc.stdout, end="" if proc.stdout.endswith("\n") else "\n")
            if proc.stderr:
                print(proc.stderr, file=sys.stderr, end="" if proc.stderr.endswith("\n") else "\n")
            if proc.returncode != 0:
                raise RuntimeError(f"export rc={proc.returncode}")

            written, base_rows = split_yearly_to_monthly(out_full, year_months)
            print(
                f"[YEAR {i}/{len(years)}] OK {year} -> full={out_full.name} "
                f"meses_escritos={written} filas_base={base_rows}",
                flush=True,
            )
        except Exception as exc:
            status = "error"
            error = str(exc)
            print(f"[YEAR {i}/{len(years)}] ERROR {year}: {error}", flush=True)

        rows.append(
            {
                "year": str(year),
                "months": ",".join(m.yyyymm for m in year_months),
                "calc_start": calc_start,
                "calc_end": calc_end,
                "status": status,
                "error": error,
                "elapsed_sec": f"{time.time() - t0:.3f}",
            }
        )

    with MANIFEST_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["year", "months", "calc_start", "calc_end", "status", "error", "elapsed_sec"],
        )
        w.writeheader()
        w.writerows(rows)

    ok = sum(1 for r in rows if r["status"] == "ok")
    err_n = sum(1 for r in rows if r["status"] == "error")
    print(f"\nDONE in {time.time() - t0_all:.1f}s | years_ok={ok} years_error={err_n}")
    print(f"Manifest -> {MANIFEST_PATH}")
    return 0 if err_n == 0 else 1


if __name__ == "__main__":
    raise SystemExit(run())

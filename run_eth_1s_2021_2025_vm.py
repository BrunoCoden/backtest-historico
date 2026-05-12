#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterator

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


TZ_OFFSET = "-03:00"


@dataclass(frozen=True)
class MonthSlot:
    year: int
    month: int

    @property
    def yyyymm(self) -> str:
        return f"{self.year:04d}{self.month:02d}"

    @property
    def start_iso(self) -> str:
        return f"{self.year:04d}-{self.month:02d}-01T00:00:00{TZ_OFFSET}"

    @property
    def end_iso(self) -> str:
        if self.month == 12:
            nxt = date(self.year + 1, 1, 1)
        else:
            nxt = date(self.year, self.month + 1, 1)
        last = nxt - timedelta(days=1)
        return f"{last:%Y-%m-%d}T23:59:59{TZ_OFFSET}"


def month_iter(year_start: int, year_end: int) -> Iterator[MonthSlot]:
    for year in range(year_start, year_end + 1):
        for month in range(1, 13):
            yield MonthSlot(year, month)


def file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def is_nonempty_file(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def merge_monthly_to_annual(month_paths: list[Path], out_path: Path) -> tuple[int, int]:
    writer: pq.ParquetWriter | None = None
    prev_ts: int | None = None
    total_rows = 0
    dropped_dups = 0

    try:
        for mp in month_paths:
            pf = pq.ParquetFile(mp)
            for rg in range(pf.num_row_groups):
                table = pf.read_row_group(rg)
                if table.num_rows == 0:
                    continue
                if "bucket_start_ms_utc" not in table.column_names:
                    raise RuntimeError(f"Falta columna bucket_start_ms_utc en {mp}")

                arr = np.asarray(table["bucket_start_ms_utc"])
                if arr.size == 0:
                    continue

                # Orden defensivo por timestamp dentro del row-group.
                if np.any(arr[1:] < arr[:-1]):
                    order = np.argsort(arr, kind="stable")
                    table = table.take(pa.array(order, type=pa.int64()))
                    arr = np.asarray(table["bucket_start_ms_utc"])

                keep = np.ones(arr.size, dtype=bool)
                if prev_ts is not None:
                    keep &= arr != prev_ts
                keep[1:] &= arr[1:] != arr[:-1]

                if not keep.all():
                    dropped_dups += int((~keep).sum())
                    table = table.filter(pa.array(keep))
                    arr = arr[keep]

                if arr.size == 0:
                    continue

                first_ts = int(arr[0])
                if prev_ts is not None and first_ts < prev_ts:
                    raise RuntimeError("Orden temporal no monotónico durante merge anual")

                prev_ts = int(arr[-1])
                if writer is None:
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    writer = pq.ParquetWriter(out_path, table.schema)
                writer.write_table(table)
                total_rows += int(table.num_rows)
    finally:
        if writer is not None:
            writer.close()

    if not is_nonempty_file(out_path):
        raise RuntimeError(f"Merge anual produjo archivo vacío: {out_path}")
    return total_rows, dropped_dups


def validate_sorted_unique(path: Path) -> tuple[int, int]:
    pf = pq.ParquetFile(path)
    prev_ts: int | None = None
    rows = 0
    dups = 0
    for rg in range(pf.num_row_groups):
        table = pf.read_row_group(rg, columns=["bucket_start_ms_utc"])
        arr = np.asarray(table["bucket_start_ms_utc"])
        if arr.size == 0:
            continue
        if np.any(arr[1:] < arr[:-1]):
            raise RuntimeError(f"{path}: no está ordenado por bucket_start_ms_utc")
        if prev_ts is not None:
            if int(arr[0]) < prev_ts:
                raise RuntimeError(f"{path}: orden temporal inconsistente entre row-groups")
            if int(arr[0]) == prev_ts:
                dups += 1
        dups += int((arr[1:] == arr[:-1]).sum())
        prev_ts = int(arr[-1])
        rows += int(arr.size)
    return rows, dups


def run(cfg: argparse.Namespace) -> int:
    root = Path(__file__).resolve().parent
    py = sys.executable
    symbol = "ETHUSDT"

    data_dir = root / "data" / "velas crudas" / symbol
    annual_dir = data_dir / "anual"
    data_dir.mkdir(parents=True, exist_ok=True)
    annual_dir.mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(parents=True, exist_ok=True)

    months = list(month_iter(cfg.year_start, cfg.year_end))
    total = len(months)
    monthly_rows: list[dict] = []

    for i, slot in enumerate(months, start=1):
        out_path = data_dir / f"{symbol}_{slot.yyyymm}_1s_ohlc.parquet"
        t0 = time.time()
        status = "ok"
        err = ""

        print(f"[{i}/{total}] {symbol} {slot.yyyymm}", flush=True)
        existed_before = is_nonempty_file(out_path)

        if existed_before:
            status = "skipped"
        else:
            cmd = [
                py,
                "dump_ohlc.py",
                symbol,
                slot.start_iso,
                slot.end_iso,
                str(out_path),
                str(cfg.bucket_seconds),
                str(cfg.workers),
                str(cfg.chunk_days),
            ]
            proc = subprocess.run(cmd, cwd=root, text=True)
            if proc.returncode != 0:
                status = "error"
                err = f"dump_ohlc_rc={proc.returncode}"

        elapsed = round(time.time() - t0, 3)
        exists_now = is_nonempty_file(out_path)
        size = out_path.stat().st_size if exists_now else 0
        sha = file_sha256(out_path) if exists_now else ""

        if status != "error" and not exists_now:
            status = "error"
            err = err or "output_missing_or_empty"

        monthly_rows.append(
            {
                "symbol": symbol,
                "yyyymm": slot.yyyymm,
                "year": slot.year,
                "start_iso_ba": slot.start_iso,
                "end_iso_ba": slot.end_iso,
                "out_monthly_path": str(out_path),
                "out_monthly_exists": int(exists_now),
                "out_monthly_size_bytes": int(size),
                "out_monthly_sha256": sha,
                "status": status,
                "error_msg": err,
                "elapsed_sec": elapsed,
            }
        )

    manifest_monthly = data_dir / "manifest_eth_1s_2021_2025_vm.csv"
    monthly_cols = [
        "symbol",
        "yyyymm",
        "year",
        "start_iso_ba",
        "end_iso_ba",
        "out_monthly_path",
        "out_monthly_exists",
        "out_monthly_size_bytes",
        "out_monthly_sha256",
        "status",
        "error_msg",
        "elapsed_sec",
    ]
    write_csv(manifest_monthly, monthly_rows, monthly_cols)
    print(f"[OK] manifest mensual -> {manifest_monthly}", flush=True)

    annual_rows: list[dict] = []
    for year in range(cfg.year_start, cfg.year_end + 1):
        t0 = time.time()
        y_months = [data_dir / f"{symbol}_{year}{m:02d}_1s_ohlc.parquet" for m in range(1, 13)]
        out_annual = annual_dir / f"{symbol}_{year}_1s_ohlc.parquet"
        status = "ok"
        err = ""
        merged_rows = 0
        dropped_dups = 0

        missing = [str(p) for p in y_months if not is_nonempty_file(p)]
        if missing:
            status = "error"
            err = f"missing_monthly={len(missing)}"
        elif is_nonempty_file(out_annual):
            status = "skipped"
        else:
            try:
                merged_rows, dropped_dups = merge_monthly_to_annual(y_months, out_annual)
            except Exception as exc:
                status = "error"
                err = f"merge_error={exc}"

        exists = is_nonempty_file(out_annual)
        size = out_annual.stat().st_size if exists else 0
        sha = file_sha256(out_annual) if exists else ""
        checked_rows = 0
        checked_dups = 0
        if exists and status != "error":
            try:
                checked_rows, checked_dups = validate_sorted_unique(out_annual)
                if checked_dups > 0:
                    status = "error"
                    err = f"annual_duplicates={checked_dups}"
            except Exception as exc:
                status = "error"
                err = f"annual_validate_error={exc}"

        annual_rows.append(
            {
                "symbol": symbol,
                "year": year,
                "out_annual_path": str(out_annual),
                "out_annual_exists": int(exists),
                "out_annual_size_bytes": int(size),
                "out_annual_sha256": sha,
                "merged_rows": int(merged_rows),
                "dropped_dups": int(dropped_dups),
                "checked_rows": int(checked_rows),
                "checked_dups": int(checked_dups),
                "status": status,
                "error_msg": err,
                "elapsed_sec": round(time.time() - t0, 3),
            }
        )
        print(
            f"[YEAR {year}] status={status} exists={int(exists)} size={size} "
            f"rows={checked_rows} dups={checked_dups}",
            flush=True,
        )

    manifest_annual = data_dir / "manifest_eth_1s_2021_2025_vm_annual.csv"
    annual_cols = [
        "symbol",
        "year",
        "out_annual_path",
        "out_annual_exists",
        "out_annual_size_bytes",
        "out_annual_sha256",
        "merged_rows",
        "dropped_dups",
        "checked_rows",
        "checked_dups",
        "status",
        "error_msg",
        "elapsed_sec",
    ]
    write_csv(manifest_annual, annual_rows, annual_cols)
    print(f"[OK] manifest anual -> {manifest_annual}", flush=True)

    err_month = sum(1 for r in monthly_rows if r["status"] == "error")
    err_year = sum(1 for r in annual_rows if r["status"] == "error")
    print(
        f"Resumen: meses ok/skipped/error = "
        f"{sum(1 for r in monthly_rows if r['status'] == 'ok')}/"
        f"{sum(1 for r in monthly_rows if r['status'] == 'skipped')}/"
        f"{err_month}",
        flush=True,
    )
    print(
        f"Resumen: años ok/skipped/error = "
        f"{sum(1 for r in annual_rows if r['status'] == 'ok')}/"
        f"{sum(1 for r in annual_rows if r['status'] == 'skipped')}/"
        f"{err_year}",
        flush=True,
    )
    return 0 if (err_month + err_year) == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Descarga ETHUSDT 1s mensual (2021-2025) + merge anual.")
    parser.add_argument("--year-start", type=int, default=2021)
    parser.add_argument("--year-end", type=int, default=2025)
    parser.add_argument("--bucket-seconds", type=int, default=1)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk-days", type=int, default=7)
    args = parser.parse_args()
    if args.year_start > args.year_end:
        print("ERROR: year-start > year-end", file=sys.stderr)
        return 2
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())

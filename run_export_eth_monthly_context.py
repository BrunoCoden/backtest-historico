#!/usr/bin/env python3
from __future__ import annotations

import calendar
import csv
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "velas crudas" / "ETHUSDT"
OUT_DIR = ROOT / "data" / "velas_30m" / "ETHUSDT_ctx"
LOG_DIR = ROOT / "logs"
MERGED_PATH = RAW_DIR / "ETHUSDT_202101_202602_merged_tmp.parquet"
MANIFEST_PATH = OUT_DIR / "manifest_eth_ctx_202101_202602.csv"

START_YYYYMM = "202101"
END_YYYYMM = "202602"


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


def ensure_merged(months: list[MonthSlot]) -> None:
    if MERGED_PATH.exists() and MERGED_PATH.stat().st_size > 0:
        print(f"[MERGE] skip (exists): {MERGED_PATH}")
        return

    inputs = [RAW_DIR / f"ETHUSDT_{m.yyyymm}_1s_ohlc.parquet" for m in months]
    missing = [str(p) for p in inputs if not p.exists() or p.stat().st_size <= 0]
    if missing:
        raise RuntimeError(f"Faltan parquets mensuales para merge: {len(missing)}")

    tmp = MERGED_PATH.with_suffix(".parquet.part")
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
        tmp.replace(MERGED_PATH)
    finally:
        if writer is not None:
            writer.close()
        if tmp.exists() and not MERGED_PATH.exists():
            tmp.unlink(missing_ok=True)
    print(f"[MERGE] OK rows={rows} -> {MERGED_PATH}", flush=True)


def run() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    months = month_iter(START_YYYYMM, END_YYYYMM)
    ensure_merged(months)

    py = sys.executable
    rows: list[dict[str, str]] = []
    total = len(months)
    t0_all = time.time()

    for i, slot in enumerate(months, start=1):
        out_csv = OUT_DIR / f"ETHUSDT_{slot.yyyymm}_bollinger_tv_30m_ctx.csv"
        month_t0 = time.time()
        status = "ok"
        err = ""
        if out_csv.exists() and out_csv.stat().st_size > 0:
            print(f"[{i}/{total}] SKIP {slot.yyyymm} (exists)", flush=True)
            rows.append(
                {
                    "yyyymm": slot.yyyymm,
                    "start_out": slot.start_iso,
                    "end_out": slot.end_iso,
                    "calc_start": months[0].start_iso,
                    "calc_end": slot.end_iso,
                    "out_csv": str(out_csv),
                    "status": "skipped",
                    "error": "",
                    "elapsed_sec": f"{time.time() - month_t0:.3f}",
                }
            )
            continue

        cmd = [
            py,
            "export_tabla_senales.py",
            str(MERGED_PATH),
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
            months[0].start_iso,
            "--calc-end",
            slot.end_iso,
            "--start",
            slot.start_iso,
            "--end",
            slot.end_iso,
            "--out",
            str(out_csv),
        ]
        print(f"[{i}/{total}] RUN {slot.yyyymm}", flush=True)
        proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
        if proc.stdout:
            print(proc.stdout, end="" if proc.stdout.endswith("\n") else "\n")
        if proc.stderr:
            print(proc.stderr, file=sys.stderr, end="" if proc.stderr.endswith("\n") else "\n")
        if proc.returncode != 0:
            status = "error"
            err = f"rc={proc.returncode}"
            print(f"[{i}/{total}] ERROR {slot.yyyymm}: {err}", flush=True)
        else:
            print(f"[{i}/{total}] OK {slot.yyyymm} -> {out_csv}", flush=True)

        rows.append(
            {
                "yyyymm": slot.yyyymm,
                "start_out": slot.start_iso,
                "end_out": slot.end_iso,
                "calc_start": months[0].start_iso,
                "calc_end": slot.end_iso,
                "out_csv": str(out_csv),
                "status": status,
                "error": err,
                "elapsed_sec": f"{time.time() - month_t0:.3f}",
            }
        )

    with MANIFEST_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "yyyymm",
                "start_out",
                "end_out",
                "calc_start",
                "calc_end",
                "out_csv",
                "status",
                "error",
                "elapsed_sec",
            ],
        )
        w.writeheader()
        w.writerows(rows)

    ok = sum(1 for r in rows if r["status"] in {"ok", "skipped"})
    err_n = sum(1 for r in rows if r["status"] == "error")
    print(f"\nDONE in {time.time() - t0_all:.1f}s | ok/skipped={ok} error={err_n}")
    print(f"Manifest -> {MANIFEST_PATH}")
    return 0 if err_n == 0 else 1


if __name__ == "__main__":
    raise SystemExit(run())

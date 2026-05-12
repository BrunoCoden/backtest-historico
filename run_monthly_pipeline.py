#!/usr/bin/env python3
from __future__ import annotations

import calendar
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


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
        last_day = calendar.monthrange(self.year, self.month)[1]
        return f"{self.year:04d}-{self.month:02d}-{last_day:02d}T23:59:59-03:00"


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_ROOT = DATA_DIR / "velas crudas"
OUT30_ROOT = DATA_DIR / "velas_30m"

SYMBOLS = ["BTCUSDT", "ETHUSDT"]
START = MonthSlot(2025, 1)
END = MonthSlot(2026, 2)

# Download profile (already agreed).
BUCKET_SECONDS = "1"
WORKERS = "8"
CHUNK_DAYS = "1"


def _month_iter(start: MonthSlot, end: MonthSlot) -> list[MonthSlot]:
    out: list[MonthSlot] = []
    y, m = start.year, start.month
    while (y < end.year) or (y == end.year and m <= end.month):
        out.append(MonthSlot(y, m))
        m += 1
        if m > 12:
            y += 1
            m = 1
    return out


def _run(cmd: list[str]) -> int:
    print(">>", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=ROOT)
    return int(proc.returncode)


def _is_nonempty_file(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def main() -> int:
    months = _month_iter(START, END)
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    OUT30_ROOT.mkdir(parents=True, exist_ok=True)

    failures: list[str] = []
    total_steps = len(SYMBOLS) * len(months) * 2
    step = 0

    print(
        f"Pipeline mensual 1s -> 30m | simbolos={SYMBOLS} | meses={len(months)} | pasos={total_steps}",
        flush=True,
    )

    py = sys.executable
    for symbol in SYMBOLS:
        raw_symbol_dir = RAW_ROOT / symbol
        out30_symbol_dir = OUT30_ROOT / symbol
        raw_symbol_dir.mkdir(parents=True, exist_ok=True)
        out30_symbol_dir.mkdir(parents=True, exist_ok=True)

        for slot in months:
            raw_path = raw_symbol_dir / f"{symbol}_{slot.yyyymm}_1s_ohlc.parquet"
            out30_csv = out30_symbol_dir / f"{symbol}_{slot.yyyymm}_bollinger_tv_30m.csv"
            out30_readable = out30_symbol_dir / f"{symbol}_{slot.yyyymm}_bollinger_tv_30m_readable.csv"

            print(
                f"\n=== {symbol} {slot.yyyymm} | {slot.start_iso} -> {slot.end_iso} ===",
                flush=True,
            )

            # 1) Download 1s monthly parquet (idempotent).
            step += 1
            if _is_nonempty_file(raw_path):
                print(
                    f"[{step}/{total_steps}] SKIP dump (exists): {raw_path}",
                    flush=True,
                )
            else:
                rc = _run(
                    [
                        py,
                        "dump_ohlc.py",
                        symbol,
                        slot.start_iso,
                        slot.end_iso,
                        str(raw_path),
                        BUCKET_SECONDS,
                        WORKERS,
                        CHUNK_DAYS,
                    ]
                )
                if rc != 0:
                    failures.append(f"dump failed: {symbol} {slot.yyyymm}")
                    print(f"[{step}/{total_steps}] ERROR dump rc={rc}", flush=True)
                    # If dump failed, skip export for this month.
                    step += 1
                    print(f"[{step}/{total_steps}] SKIP export because dump failed", flush=True)
                    continue
                print(f"[{step}/{total_steps}] OK dump -> {raw_path}", flush=True)

            # 2) Export 30m bollinger tradingview (technical + readable).
            step += 1
            if _is_nonempty_file(out30_csv) and _is_nonempty_file(out30_readable):
                print(
                    f"[{step}/{total_steps}] SKIP export (exists): {out30_csv}",
                    flush=True,
                )
            else:
                rc = _run(
                    [
                        py,
                        "export_tabla_senales.py",
                        str(raw_path),
                        "--strategy",
                        "bollinger",
                        "--bb-profile",
                        "tradingview",
                        "--bb-price-source",
                        "close",
                        "--tf",
                        "30T",
                        "--expand-sl-tp",
                        "--out",
                        str(out30_csv),
                        "--readable",
                    ]
                )
                if rc != 0:
                    failures.append(f"export failed: {symbol} {slot.yyyymm}")
                    print(f"[{step}/{total_steps}] ERROR export rc={rc}", flush=True)
                    continue
                print(f"[{step}/{total_steps}] OK export -> {out30_csv}", flush=True)

    print("\n=== RESUMEN ===", flush=True)
    if failures:
        print(f"Fallos: {len(failures)}", flush=True)
        for f in failures:
            print(" -", f, flush=True)
        return 1

    print("Sin fallos.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


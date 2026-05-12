#!/usr/bin/env python3
from __future__ import annotations

import calendar
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
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

    def end_iso(self, cap: datetime | None = None) -> str:
        tz = timezone(timedelta(hours=-3))
        last_day = calendar.monthrange(self.year, self.month)[1]
        month_end = datetime(
            self.year,
            self.month,
            last_day,
            23,
            59,
            59,
            tzinfo=tz,
        )
        if cap is not None:
            cap_local = cap.astimezone(tz)
            if cap_local < month_end:
                return cap_local.replace(microsecond=0).isoformat()
        return month_end.isoformat()


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "generated_2026"
INTRABAR_DIR = DATA_DIR / "intrabar_1s"
BACKTEST_DIR = DATA_DIR / "intrabar_backtests"

SYMBOLS = ("ETHUSDT", "BTCUSDT")
START_YEAR = 2026
START_MONTH = 1

# Uses aggTrades from Binance Futures archives via dump_ohlc.py.
# 1s gives the closest reproducible approximation to tick-by-tick with this repo.
BUCKET_SECONDS = "1"
WORKERS = "4"
CHUNK_DAYS = "1"


def _month_iter(start_year: int, start_month: int, end: datetime) -> list[MonthSlot]:
    out: list[MonthSlot] = []
    y, m = start_year, start_month
    end_local = end.astimezone(timezone(timedelta(hours=-3)))
    while (y < end_local.year) or (y == end_local.year and m <= end_local.month):
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


def _nonempty(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def _download_month(py: str, symbol: str, slot: MonthSlot, cap: datetime) -> Path:
    INTRABAR_DIR.mkdir(parents=True, exist_ok=True)
    out_path = INTRABAR_DIR / f"{symbol}_{slot.yyyymm}_1s_ohlc.parquet"
    if _nonempty(out_path):
        print(f"SKIP intrabar exists: {out_path}", flush=True)
        return out_path

    rc = _run(
        [
            py,
            "dump_ohlc.py",
            symbol,
            slot.start_iso,
            slot.end_iso(cap),
            str(out_path),
            BUCKET_SECONDS,
            WORKERS,
            CHUNK_DAYS,
        ]
    )
    if rc != 0:
        raise RuntimeError(f"download failed: {symbol} {slot.yyyymm} rc={rc}")
    return out_path


def _run_month_backtests(py: str, symbol: str, slot: MonthSlot, intrabar_path: Path) -> None:
    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
    base_30m = DATA_DIR / f"{symbol}_2026_ytd_30m_ohlc.parquet"
    base_1h = DATA_DIR / f"{symbol}_2026_ytd_1h_ohlc.parquet"
    if not _nonempty(base_30m) or not _nonempty(base_1h):
        print(f"SKIP backtest {symbol} {slot.yyyymm}: missing 30m/1h base parquet", flush=True)
        return

    for tf, base in (("30min", base_30m), ("1h", base_1h)):
        out_html = BACKTEST_DIR / f"range3_bb_{symbol}_{slot.yyyymm}_{tf}_intrabar_1s.html"
        out_csv = BACKTEST_DIR / f"range3_bb_{symbol}_{slot.yyyymm}_{tf}_intrabar_1s_trades.csv"
        if _nonempty(out_html) and _nonempty(out_csv):
            print(f"SKIP backtest exists: {out_csv}", flush=True)
            continue
        rc = _run(
            [
                py,
                "backtest_plotly.py",
                str(base),
                "--strategy",
                "range3_bb",
                "--tf",
                tf,
                "--intrabar-parquet",
                str(intrabar_path),
                "--intrabar-tf",
                "1s",
                "--out",
                str(out_html),
                "--out-trades",
                str(out_csv),
            ]
        )
        if rc != 0:
            raise RuntimeError(f"backtest failed: {symbol} {slot.yyyymm} {tf} rc={rc}")


def main() -> int:
    now = datetime.now(timezone(timedelta(hours=-3)))
    months = _month_iter(START_YEAR, START_MONTH, now)
    py = sys.executable

    print(
        "Pipeline intrabar 1s 2026 YTD | "
        f"symbols={','.join(SYMBOLS)} | months={len(months)} | "
        f"workers={WORKERS}",
        flush=True,
    )

    failures: list[str] = []
    for symbol in SYMBOLS:
        for slot in months:
            print(f"\n=== {symbol} {slot.yyyymm} ===", flush=True)
            try:
                intrabar = _download_month(py, symbol, slot, now)
                _run_month_backtests(py, symbol, slot, intrabar)
            except Exception as exc:
                failures.append(f"{symbol} {slot.yyyymm}: {exc}")
                print(f"ERROR {failures[-1]}", flush=True)

    if failures:
        print("\nFallos:", flush=True)
        for failure in failures:
            print(f" - {failure}", flush=True)
        return 1

    print("\nOK pipeline intrabar completo.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

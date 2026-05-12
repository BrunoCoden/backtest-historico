#!/usr/bin/env python3
from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
RAW_ROOT = ROOT / "data" / "velas crudas"
OUT_ROOT = ROOT / "data" / "velas_30m"
MANIFEST_PATH = OUT_ROOT / "export_manifest.csv"
PNL_COL = "PNL Cierre %"

RAW_RE = re.compile(r"^(?P<symbol>[A-Z0-9]+)_(?P<yyyymm>\d{6})_1s_ohlc\.parquet$")


@dataclass(frozen=True)
class RawSlot:
    symbol: str
    yyyymm: str
    path: Path


def _discover_raw_slots() -> list[RawSlot]:
    out: list[RawSlot] = []
    for p in sorted(RAW_ROOT.glob("*/*.parquet")):
        name = p.name
        if name.endswith("_merged_tmp.parquet"):
            continue
        m = RAW_RE.match(name)
        if not m:
            continue
        out.append(RawSlot(symbol=m.group("symbol"), yyyymm=m.group("yyyymm"), path=p))
    out.sort(key=lambda x: (x.symbol, x.yyyymm))
    return out


def _parse_out_paths(stdout: str) -> tuple[str, str]:
    out_tech = ""
    out_readable = ""
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith("OK -> "):
            out_tech = line.split("OK -> ", 1)[1].strip()
        elif line.startswith("OK (readable) -> "):
            out_readable = line.split("OK (readable) -> ", 1)[1].strip()
    return out_tech, out_readable


def _read_total_pnl(path_csv: Path) -> float:
    if not path_csv.exists():
        return 0.0
    df = pd.read_csv(path_csv, usecols=[PNL_COL])
    if df.empty:
        return 0.0
    last = pd.to_numeric(df[PNL_COL], errors="coerce").iloc[-1]
    if pd.isna(last):
        return 0.0
    return float(last)


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    slots = _discover_raw_slots()
    if not slots:
        print("No hay parquets crudos mensuales para procesar.")
        return 1

    py = sys.executable
    rows: list[dict] = []
    total = len(slots)

    print(f"Exportando {total} parquets crudos mensuales...", flush=True)
    for i, slot in enumerate(slots, start=1):
        out_dir = OUT_ROOT / slot.symbol
        out_dir.mkdir(parents=True, exist_ok=True)
        out_base = out_dir / f"{slot.symbol}_{slot.yyyymm}_bollinger_tv_30m.csv"

        cmd = [
            py,
            "export_tabla_senales.py",
            str(slot.path),
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
            str(out_base),
            "--readable",
            "--pnl-in-filename",
        ]
        if slot.yyyymm == "202501":
            cmd += ["--start", "2025-01-01T00:00:00-03:00"]

        print(f"[{i}/{total}] {slot.symbol} {slot.yyyymm}", flush=True)
        proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
        if proc.stdout:
            print(proc.stdout, end="" if proc.stdout.endswith("\n") else "\n")
        if proc.stderr:
            print(proc.stderr, file=sys.stderr, end="" if proc.stderr.endswith("\n") else "\n")

        out_tech_txt, out_read_txt = _parse_out_paths(proc.stdout or "")
        out_tech = Path(out_tech_txt) if out_tech_txt else out_base
        pnl_total = _read_total_pnl(out_tech)
        status = "ok" if proc.returncode == 0 else "error"
        rows.append(
            {
                "raw_input": str(slot.path),
                "symbol": slot.symbol,
                "yyyymm": slot.yyyymm,
                "pnl_total_pct": pnl_total,
                "out_tech_final": out_tech_txt,
                "out_readable_final": out_read_txt,
                "status": status,
            }
        )
        if proc.returncode != 0:
            rows[-1]["error_code"] = int(proc.returncode)

    manifest = pd.DataFrame(rows)
    manifest.to_csv(MANIFEST_PATH, index=False)
    ok = int((manifest["status"] == "ok").sum())
    print(f"Manifest -> {MANIFEST_PATH}")
    print(f"OK: {ok}/{len(manifest)}")
    return 0 if ok == len(manifest) else 1


if __name__ == "__main__":
    raise SystemExit(main())

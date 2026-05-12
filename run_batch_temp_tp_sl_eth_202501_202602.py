#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT_ROOT = ROOT / "data" / "pruebas combinaciones TPs y SLs"
MANIFEST = OUT_ROOT / "manifest_batch_eth_202501_202602.csv"
SCRIPT = ROOT / "run_temp_eth_1h_202602_tp_mix.py"

START = "2025-01-01T00:00:00-03:00"
END = "2026-02-28T23:59:59-03:00"
SYMBOL = "ETHUSDT"
TP_FLIP_PCT = 0.03


@dataclass(frozen=True)
class RunSpec:
    run_idx: int
    symbol: str
    tf: str
    sl_pct: float
    tp_signal_pct: float
    tp_flip_pct: float
    calc_start: str
    calc_end: str
    out_start: str
    out_end: str
    run_label: str


def _build_specs(
    *,
    symbol: str,
    start: str,
    end: str,
    sl_pct: float,
    tp_flip_pct: float,
    run_label_prefix: str,
) -> list[RunSpec]:
    tp_values = [3.0 + 0.5 * i for i in range(13)]
    specs: list[RunSpec] = []
    idx = 1
    for tf in ["1H", "30T"]:
        for tp in tp_values:
            specs.append(
                RunSpec(
                    run_idx=idx,
                    symbol=symbol,
                    tf=tf,
                    sl_pct=sl_pct,
                    tp_signal_pct=tp / 100.0,
                    tp_flip_pct=tp_flip_pct,
                    calc_start=start,
                    calc_end=end,
                    out_start=start,
                    out_end=end,
                    run_label=f"{run_label_prefix}{idx:02d}",
                )
            )
            idx += 1
    return specs


def _tp_tag(tp_pct: float) -> str:
    return f"{tp_pct * 100.0:.2f}"


def _load_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, dtype={"run_idx": int})
    except Exception:
        return pd.DataFrame()


def _status_done_from_manifest(df: pd.DataFrame) -> dict[int, str]:
    out: dict[int, str] = {}
    if df.empty or "run_idx" not in df.columns or "status" not in df.columns:
        return out
    for _, r in df.iterrows():
        try:
            ridx = int(r["run_idx"])
        except Exception:
            continue
        st = str(r.get("status", "")).strip().lower()
        if st in {"ok", "skipped_done"}:
            out[ridx] = st
    return out


def _find_existing_completed_folder(out_root: Path, run_label: str) -> Optional[Path]:
    cands = sorted([p for p in out_root.glob(f"*_{run_label}_pnl_*") if p.is_dir()])
    for d in cands:
        tech = [p for p in d.glob("*.csv") if not p.name.endswith("_trades.csv")]
        trds = [p for p in d.glob("*_trades.csv")]
        if len(tech) != 1 or len(trds) != 1:
            continue
        try:
            df_t = pd.read_csv(tech[0])
            df_r = pd.read_csv(trds[0])
        except Exception:
            continue
        ok_t = "Fecha" in df_t.columns and bool(df_t["Fecha"].astype(str).eq("TOTAL PNL %").any())
        ok_r = "entry_time" in df_r.columns and bool(df_r["entry_time"].astype(str).eq("TOTAL PNL %").any())
        if ok_t and ok_r:
            return d
    return None


def _parse_run_output(stdout: str) -> tuple[Optional[str], Optional[str], Optional[float]]:
    out_csv = None
    out_trades = None
    pnl = None
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith("OK -> "):
            out_csv = line.replace("OK -> ", "", 1).strip()
        elif line.startswith("OK (trades) -> "):
            out_trades = line.replace("OK (trades) -> ", "", 1).strip()
        elif "PNL_total_pct=" in line:
            m = re.search(r"PNL_total_pct=([-+]?[0-9]*\.?[0-9]+)", line)
            if m:
                pnl = float(m.group(1))
    return out_csv, out_trades, pnl


def _validate_totals(tech_csv: Path, trades_csv: Path, pnl_stdout: Optional[float]) -> tuple[float, float]:
    df_t = pd.read_csv(tech_csv)
    df_r = pd.read_csv(trades_csv)

    mt = df_t["Fecha"].astype(str).eq("TOTAL PNL %")
    if not mt.any():
        raise ValueError(f"{tech_csv} sin fila TOTAL PNL %")
    tech_total = float(pd.to_numeric(df_t.loc[mt, "PNL Cierre %"], errors="coerce").iloc[-1])

    mr = df_r["entry_time"].astype(str).eq("TOTAL PNL %")
    if not mr.any():
        raise ValueError(f"{trades_csv} sin fila TOTAL PNL %")
    trades_total = float(pd.to_numeric(df_r.loc[mr, "pnl_pct"], errors="coerce").iloc[-1])

    if pnl_stdout is not None and abs(tech_total - pnl_stdout) > 0.05:
        raise ValueError(
            f"Mismatch TOTAL técnico vs stdout: tech={tech_total:.6f} stdout={pnl_stdout:.6f}"
        )
    return tech_total, trades_total


def _append_manifest_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    df = pd.DataFrame([row])
    df.to_csv(path, mode="a", index=False, header=not exists)


def main() -> int:
    parser = argparse.ArgumentParser(description="Lote secuencial de corridas temporales ETH 202501..202602")
    parser.add_argument("--max-runs", type=int, default=0, help="Limita cantidad de corridas a ejecutar (0=todas)")
    parser.add_argument("--resume", dest="resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--symbol", default=SYMBOL)
    parser.add_argument("--start", default=START)
    parser.add_argument("--end", default=END)
    parser.add_argument("--sl-pct", type=float, default=0.02)
    parser.add_argument("--tp-flip-pct", type=float, default=TP_FLIP_PCT)
    parser.add_argument("--run-label-prefix", default="run")
    parser.add_argument(
        "--manifest",
        default=str(MANIFEST.relative_to(ROOT)),
        help="Path del manifest (relativo a ROOT o absoluto)",
    )
    args = parser.parse_args()

    if not SCRIPT.exists():
        print(f"ERROR: no existe script base: {SCRIPT}", file=sys.stderr)
        return 1

    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path

    specs = _build_specs(
        symbol=args.symbol,
        start=args.start,
        end=args.end,
        sl_pct=float(args.sl_pct),
        tp_flip_pct=float(args.tp_flip_pct),
        run_label_prefix=str(args.run_label_prefix),
    )
    manifest_df = _load_manifest(manifest_path)
    done_map = _status_done_from_manifest(manifest_df) if args.resume else {}

    started = 0
    for sp in specs:
        if args.max_runs > 0 and started >= int(args.max_runs):
            break

        run_key = f"[{sp.run_idx:02d}/26] TF={sp.tf} TP_signal={_tp_tag(sp.tp_signal_pct)}%"
        t0 = time.monotonic()

        # Resume por manifest
        if args.resume and sp.run_idx in done_map:
            print(f"{run_key} -> skipped_done (manifest)", flush=True)
            _append_manifest_row(
                manifest_path,
                {
                    "run_idx": sp.run_idx,
                    "symbol": sp.symbol,
                    "tf": sp.tf,
                    "sl_pct": sp.sl_pct,
                    "tp_signal_pct": sp.tp_signal_pct,
                    "tp_flip_pct": sp.tp_flip_pct,
                    "calc_start": sp.calc_start,
                    "calc_end": sp.calc_end,
                    "out_start": sp.out_start,
                    "out_end": sp.out_end,
                    "folder_final": "",
                    "tech_csv": "",
                    "trades_csv": "",
                    "pnl_total_pct": "",
                    "status": "skipped_done",
                    "elapsed_sec": 0.0,
                    "error_msg": "from_manifest",
                },
            )
            continue

        # Resume por carpeta existente completa
        existing = _find_existing_completed_folder(OUT_ROOT, sp.run_label) if args.resume else None
        if existing is not None:
            tech = next(iter([p for p in existing.glob("*.csv") if not p.name.endswith("_trades.csv")]), None)
            trds = next(iter(existing.glob("*_trades.csv")), None)
            pnl_val = ""
            if tech is not None:
                try:
                    df_t = pd.read_csv(tech)
                    m = df_t["Fecha"].astype(str).eq("TOTAL PNL %")
                    if m.any():
                        pnl_val = float(pd.to_numeric(df_t.loc[m, "PNL Cierre %"], errors="coerce").iloc[-1])
                except Exception:
                    pass
            print(f"{run_key} -> skipped_done (folder)", flush=True)
            _append_manifest_row(
                manifest_path,
                {
                    "run_idx": sp.run_idx,
                    "symbol": sp.symbol,
                    "tf": sp.tf,
                    "sl_pct": sp.sl_pct,
                    "tp_signal_pct": sp.tp_signal_pct,
                    "tp_flip_pct": sp.tp_flip_pct,
                    "calc_start": sp.calc_start,
                    "calc_end": sp.calc_end,
                    "out_start": sp.out_start,
                    "out_end": sp.out_end,
                    "folder_final": str(existing),
                    "tech_csv": str(tech) if tech else "",
                    "trades_csv": str(trds) if trds else "",
                    "pnl_total_pct": pnl_val,
                    "status": "skipped_done",
                    "elapsed_sec": round(time.monotonic() - t0, 3),
                    "error_msg": "",
                },
            )
            continue

        print(f"{run_key} -> running", flush=True)
        cmd = [
            sys.executable,
            str(SCRIPT),
            "--symbol",
            sp.symbol,
            "--tf",
            sp.tf,
            "--calc-start",
            sp.calc_start,
            "--calc-end",
            sp.calc_end,
            "--out-start",
            sp.out_start,
            "--out-end",
            sp.out_end,
            "--sl-pct",
            str(sp.sl_pct),
            "--tp-signal-pct",
            str(sp.tp_signal_pct),
            "--tp-flip-pct",
            str(sp.tp_flip_pct),
            "--run-label",
            sp.run_label,
            "--append-pnl-to-folder",
            "--atomic-folder",
            "--sl-intrabar-1s",
        ]

        proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
        elapsed = round(time.monotonic() - t0, 3)
        started += 1

        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "").strip()
            print(f"{run_key} -> ERROR", flush=True)
            _append_manifest_row(
                manifest_path,
                {
                    "run_idx": sp.run_idx,
                    "symbol": sp.symbol,
                    "tf": sp.tf,
                    "sl_pct": sp.sl_pct,
                    "tp_signal_pct": sp.tp_signal_pct,
                    "tp_flip_pct": sp.tp_flip_pct,
                    "calc_start": sp.calc_start,
                    "calc_end": sp.calc_end,
                    "out_start": sp.out_start,
                    "out_end": sp.out_end,
                    "folder_final": "",
                    "tech_csv": "",
                    "trades_csv": "",
                    "pnl_total_pct": "",
                    "status": "error",
                    "elapsed_sec": elapsed,
                    "error_msg": err[:2000],
                },
            )
            if proc.stdout:
                print(proc.stdout, flush=True)
            if proc.stderr:
                print(proc.stderr, file=sys.stderr, flush=True)
            continue

        out_csv_s, out_trades_s, pnl_stdout = _parse_run_output(proc.stdout)
        if not out_csv_s or not out_trades_s:
            _append_manifest_row(
                manifest_path,
                {
                    "run_idx": sp.run_idx,
                    "symbol": sp.symbol,
                    "tf": sp.tf,
                    "sl_pct": sp.sl_pct,
                    "tp_signal_pct": sp.tp_signal_pct,
                    "tp_flip_pct": sp.tp_flip_pct,
                    "calc_start": sp.calc_start,
                    "calc_end": sp.calc_end,
                    "out_start": sp.out_start,
                    "out_end": sp.out_end,
                    "folder_final": "",
                    "tech_csv": "",
                    "trades_csv": "",
                    "pnl_total_pct": "",
                    "status": "error",
                    "elapsed_sec": elapsed,
                    "error_msg": "No se pudieron parsear paths de salida",
                },
            )
            continue

        out_csv = Path(out_csv_s)
        out_trades = Path(out_trades_s)
        folder_final = out_csv.parent

        status = "ok"
        error_msg = ""
        tech_total = ""
        try:
            tech_total, _ = _validate_totals(out_csv, out_trades, pnl_stdout)
        except Exception as exc:
            status = "error"
            error_msg = str(exc)

        print(f"{run_key} -> {status} | folder={folder_final}", flush=True)
        _append_manifest_row(
            manifest_path,
            {
                "run_idx": sp.run_idx,
                "symbol": sp.symbol,
                "tf": sp.tf,
                "sl_pct": sp.sl_pct,
                "tp_signal_pct": sp.tp_signal_pct,
                "tp_flip_pct": sp.tp_flip_pct,
                "calc_start": sp.calc_start,
                "calc_end": sp.calc_end,
                "out_start": sp.out_start,
                "out_end": sp.out_end,
                "folder_final": str(folder_final),
                "tech_csv": str(out_csv),
                "trades_csv": str(out_trades),
                "pnl_total_pct": tech_total,
                "status": status,
                "elapsed_sec": elapsed,
                "error_msg": error_msg,
            },
        )

    print(f"Manifest -> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

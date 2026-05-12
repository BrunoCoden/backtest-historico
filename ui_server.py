#!/usr/bin/env python3
import math
import shutil
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import pyarrow.parquet as pq
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from backtest_plotly import (
    _resample_ohlcv,
    compute_bollinger_bands,
    compute_supertrend,
    generate_bollinger_signals,
    generate_supertrend_signals,
    run_backtest,
    run_range3_bb_backtest,
)

app = FastAPI()

BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "ui"

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

_loaded_path: Optional[Path] = None
_loaded_meta: Optional[dict[str, Any]] = None
_run_jobs: dict[str, dict[str, Any]] = {}
_run_jobs_lock = threading.Lock()

BOT_BOLLINGER_DIR = Path("/home/diego/bot")
BOT_DEX_DIR = Path("/home/diego/botDex")
BOT_ST2_DIR = Path("/home/diego/Supertrend2-0")


def _read_env_value(env_path: Path, key: str) -> Optional[str]:
    if not env_path.exists():
        return None
    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            if k.strip() == key:
                return v.strip().strip('"').strip("'")
    except Exception:
        return None
    return None


def _load_strategy_defaults() -> dict:
    defaults = {
        "bollinger": {"sl": 0.02, "tp": 0.0},
        "supertrend": {"sl": 0.02, "tp": 0.0},
        "supertrend2": {"sl": 0.02, "tp": 0.05},
        "range3_bb": {"sl": 0.02, "tp": 0.0},
    }

    # Bollinger (bot)
    env_boll = BOT_BOLLINGER_DIR / ".env"
    v = _read_env_value(env_boll, "WATCHER_CONTRA_THRESHOLD_PCT")
    if v:
        try:
            defaults["bollinger"]["sl"] = float(v)
        except Exception:
            pass

    # Supertrend (botDex)
    env_dex = BOT_DEX_DIR / ".env"
    v = _read_env_value(env_dex, "STRAT_STOP_LOSS_PCT")
    if v:
        try:
            defaults["supertrend"]["sl"] = float(v)
        except Exception:
            pass

    # Supertrend2 (Supertrend2-0)
    env_st2 = BOT_ST2_DIR / ".env"
    v = _read_env_value(env_st2, "WATCHER_CONTRA_THRESHOLD_PCT")
    if not v:
        v = _read_env_value(env_st2, "STRAT_STOP_LOSS_PCT")
    if v:
        try:
            defaults["supertrend2"]["sl"] = float(v)
        except Exception:
            pass
    v = _read_env_value(env_st2, "WATCHER_TAKE_PROFIT_PCT")
    if v:
        try:
            defaults["supertrend2"]["tp"] = float(v)
        except Exception:
            pass

    return defaults


def _find_col_name(columns: list[str], candidates: list[str]) -> Optional[str]:
    cols_lc = {c.lower(): c for c in columns}
    for cand in candidates:
        hit = cols_lc.get(cand.lower())
        if hit is not None:
            return hit
    return None


def _coerce_ms_value(value: Any) -> int:
    if value is None:
        raise ValueError("timestamp vacío")
    if isinstance(value, (int, float)):
        return int(value)
    if hasattr(value, "item"):
        try:
            return _coerce_ms_value(value.item())
        except Exception:
            pass
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"timestamp inválido: {value!r}")
    return int(ts.value // 1_000_000)


def _series_to_ms(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        return numeric
    dt = pd.to_datetime(series, utc=True, errors="coerce")
    ms = (dt.astype("int64") // 1_000_000).astype("float64")
    ms[dt.isna()] = float("nan")
    return ms


def _extract_bounds(parquet_path: Path, time_col: str) -> tuple[int, int, int]:
    pf = pq.ParquetFile(parquet_path)
    md = pf.metadata
    num_rows = int(md.num_rows) if md else 0
    col_idx = pf.schema.names.index(time_col)

    if md is not None and md.num_row_groups > 0:
        ts_min = None
        ts_max = None
        complete_stats = True
        for i in range(md.num_row_groups):
            stats = md.row_group(i).column(col_idx).statistics
            if not stats or not stats.has_min_max:
                complete_stats = False
                break
            curr_min = _coerce_ms_value(stats.min)
            curr_max = _coerce_ms_value(stats.max)
            ts_min = curr_min if ts_min is None else min(ts_min, curr_min)
            ts_max = curr_max if ts_max is None else max(ts_max, curr_max)
        if complete_stats and ts_min is not None and ts_max is not None:
            return num_rows, ts_min, ts_max

    ts_df = pd.read_parquet(parquet_path, columns=[time_col])
    if ts_df.empty:
        raise ValueError("Parquet sin filas")
    ts_ms = _series_to_ms(ts_df[time_col])
    ts_min = ts_ms.min(skipna=True)
    ts_max = ts_ms.max(skipna=True)
    if pd.isna(ts_min) or pd.isna(ts_max):
        raise ValueError("No se pudo calcular rango temporal del parquet")
    return num_rows if num_rows > 0 else len(ts_df), int(ts_min), int(ts_max)


def _inspect_parquet(parquet_path: Path) -> dict[str, Any]:
    pf = pq.ParquetFile(parquet_path)
    columns = list(pf.schema.names)
    if not columns:
        raise ValueError("Parquet sin columnas")

    time_col = _find_col_name(columns, ["timestamp_ms_utc", "bucket_start_ms_utc"])
    if time_col is None:
        raise ValueError("Falta columna timestamp_ms_utc/bucket_start_ms_utc")

    price_col = _find_col_name(columns, ["price"])
    qty_col = _find_col_name(columns, ["qty"])
    if price_col and qty_col:
        read_columns = [time_col, price_col, qty_col]
        data_format = "ticks"
    else:
        open_col = _find_col_name(columns, ["open"])
        high_col = _find_col_name(columns, ["high"])
        low_col = _find_col_name(columns, ["low"])
        close_col = _find_col_name(columns, ["close"])
        volume_col = _find_col_name(columns, ["volume", "qty"])
        if None in (open_col, high_col, low_col, close_col):
            raise ValueError("Faltan columnas price/qty o open/high/low/close")
        read_columns = [time_col, open_col, high_col, low_col, close_col]
        if volume_col is not None:
            read_columns.append(volume_col)
        data_format = "ohlc"

    rows, ts_min, ts_max = _extract_bounds(parquet_path, time_col)
    return {
        "rows": rows,
        "ts_min": ts_min,
        "ts_max": ts_max,
        "time_col": time_col,
        "read_columns": read_columns,
        "format": data_format,
    }


def _parse_date(value: str, tz: str, is_end: bool) -> Optional[pd.Timestamp]:
    if not value:
        return None
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        # Fallback para entradas tipo DD/MM/YYYY.
        ts = pd.to_datetime(value, errors="coerce", dayfirst=True)
    if pd.isna(ts):
        return None
    if ts.tzinfo is None:
        ts = ts.tz_localize(tz)
    else:
        ts = ts.tz_convert(tz)
    ts = ts.normalize()
    if is_end:
        ts = ts + pd.Timedelta(hours=23, minutes=59, seconds=59)
    return ts


def _to_utc_ms(ts: Optional[pd.Timestamp]) -> Optional[int]:
    if ts is None:
        return None
    return int(ts.tz_convert("UTC").value // 1_000_000)


def _load_ticks_slice(
    parquet_path: Path,
    loaded_meta: dict[str, Any],
    start_ms: Optional[int],
    end_ms: Optional[int],
) -> pd.DataFrame:
    filters = []
    time_col = str(loaded_meta["time_col"])
    if start_ms is not None:
        filters.append((time_col, ">=", int(start_ms)))
    if end_ms is not None:
        filters.append((time_col, "<=", int(end_ms)))

    read_kwargs: dict[str, Any] = {"columns": list(loaded_meta["read_columns"])}
    if filters:
        read_kwargs["filters"] = filters

    try:
        df = pd.read_parquet(parquet_path, **read_kwargs)
    except Exception:
        # Fallback defensivo: si el engine no empuja filtros, filtra en memoria.
        df = pd.read_parquet(parquet_path, columns=list(loaded_meta["read_columns"]))

    if start_ms is not None or end_ms is not None:
        ts_ms = _series_to_ms(df[time_col])
        mask = pd.Series(True, index=df.index)
        if start_ms is not None:
            mask &= ts_ms >= int(start_ms)
        if end_ms is not None:
            mask &= ts_ms <= int(end_ms)
        df = df.loc[mask]

    if "timestamp_ms_utc" not in df.columns and "bucket_start_ms_utc" in df.columns:
        df = df.copy()
        df["timestamp_ms_utc"] = df["bucket_start_ms_utc"]
    return df


@app.get("/")
def index():
    html_path = STATIC_DIR / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.post("/api/load")
async def load_parquet(file: UploadFile = File(...)):
    global _loaded_path, _loaded_meta

    tmp_dir = Path(tempfile.mkdtemp(prefix="backtest_ui_"))
    parquet_name = Path(file.filename or "upload.parquet").name
    parquet_path = tmp_dir / parquet_name

    size = 0
    with parquet_path.open("wb") as f:
        while True:
            chunk = await file.read(8 * 1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            f.write(chunk)
    await file.close()

    if size == 0:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return JSONResponse({"ok": False, "error": "Archivo vacío"}, status_code=400)

    try:
        meta = _inspect_parquet(parquet_path)
    except Exception as exc:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return JSONResponse({"ok": False, "error": f"No se pudo leer parquet: {exc}"}, status_code=400)

    previous_path = _loaded_path
    _loaded_path = parquet_path
    _loaded_meta = meta
    if previous_path is not None and previous_path.exists():
        previous_tmp_dir = previous_path.parent
        if previous_tmp_dir != tmp_dir:
            shutil.rmtree(previous_tmp_dir, ignore_errors=True)

    return {
        "ok": True,
        "rows": int(meta["rows"]),
        "ts_min": int(meta["ts_min"]),
        "ts_max": int(meta["ts_max"]),
        "defaults": _load_strategy_defaults(),
    }


def _safe_list(series: pd.Series) -> list:
    out = []
    for v in series.tolist():
        if v is None:
            out.append(None)
        else:
            try:
                fv = float(v)
                if math.isnan(fv):
                    out.append(None)
                else:
                    out.append(fv)
            except Exception:
                out.append(None)
    return out


def _run_backtest_payload(payload: dict, progress_cb=None) -> dict[str, Any]:
    global _loaded_path, _loaded_meta

    def progress(pct: float, message: str):
        if progress_cb is not None:
            progress_cb(max(0.0, min(100.0, float(pct))), message)

    if _loaded_path is None or _loaded_meta is None:
        raise ValueError("No hay parquet cargado")

    strategy = payload.get("strategy")
    tf = payload.get("tf", "30T")
    entry = payload.get("entry", "close")
    notional = float(payload.get("notional", 30))
    fee = float(payload.get("fee", 0.0004))
    sl = float(payload.get("sl", 0.02))
    tp = float(payload.get("tp", 0.0))
    tz = payload.get("tz", "America/Argentina/Buenos_Aires")

    bb_length = int(payload.get("bb_length", 20))
    bb_mult = float(payload.get("bb_mult", 2.0))
    bb_direction = int(payload.get("bb_direction", 0))
    st_period = int(payload.get("st_period", 10))
    st_factor = float(payload.get("st_factor", 3.0))

    progress(5, "Preparando rango")
    start = payload.get("start")
    end = payload.get("end")
    ts_start = _parse_date(start, tz, is_end=False) if start else None
    ts_end = _parse_date(end, tz, is_end=True) if end else None
    start_ms = _to_utc_ms(ts_start)
    end_ms = _to_utc_ms(ts_end)

    progress(20, "Leyendo parquet")
    df_ticks = _load_ticks_slice(_loaded_path, _loaded_meta, start_ms, end_ms)
    if df_ticks.empty:
        raise ValueError("No hay datos en el rango")

    progress(45, "Resampleando velas")
    ohlcv = _resample_ohlcv(df_ticks, tf, tz)
    if ts_start is not None:
        ohlcv = ohlcv[ohlcv.index >= ts_start]
    if ts_end is not None:
        ohlcv = ohlcv[ohlcv.index <= ts_end]
    if ohlcv.empty:
        raise ValueError("No hay datos en el rango")

    progress(65, "Calculando señales")
    overlays = []
    if strategy == "bollinger":
        bb = compute_bollinger_bands(ohlcv, bb_length, bb_mult)
        overlays.append(("upper", bb["upper"]))
        overlays.append(("lower", bb["lower"]))
        overlays.append(("basis", bb["basis"]))
        signals = generate_bollinger_signals(ohlcv, bb, bb_direction)
        tp_pct = None
        trades, markers = run_backtest(ohlcv, signals, entry, sl, tp_pct, notional, fee, reentry_on_tp=True)
    elif strategy == "supertrend":
        st = compute_supertrend(ohlcv, st_period, st_factor)
        overlays.append(("supertrend", st["supertrend"]))
        signals = generate_supertrend_signals(ohlcv, st)
        tp_pct = None
        trades, markers = run_backtest(ohlcv, signals, entry, sl, tp_pct, notional, fee, reentry_on_tp=True)
    elif strategy == "supertrend2":
        st = compute_supertrend(ohlcv, st_period, st_factor)
        overlays.append(("supertrend", st["supertrend"]))
        signals = generate_supertrend_signals(ohlcv, st)
        tp_pct = tp if tp > 0 else None
        trades, markers = run_backtest(ohlcv, signals, entry, sl, tp_pct, notional, fee, reentry_on_tp=False)
    elif strategy == "range3_bb":
        trades, markers, channels, bb = run_range3_bb_backtest(
            ohlcv=ohlcv,
            notional=notional,
            fee_rate=fee,
            bb_length=bb_length,
            bb_mult=bb_mult,
            stop_loss_pct=sl,
            entry_mode=entry,
        )
        overlays.extend(
            [
                ("range_max", channels["max"]),
                ("range_maxfloor", channels["maxfloor"]),
                ("range_minroof", channels["minroof"]),
                ("range_min", channels["min"]),
                ("bb_upper", bb["upper"]),
                ("bb_lower", bb["lower"]),
                ("bb_basis", bb["basis"]),
            ]
        )
    else:
        raise ValueError("Estrategia inválida")

    progress(85, "Armando respuesta")
    ohlcv_out = {
        "t": [ts.isoformat() for ts in ohlcv.index],
        "open": _safe_list(ohlcv["Open"]),
        "high": _safe_list(ohlcv["High"]),
        "low": _safe_list(ohlcv["Low"]),
        "close": _safe_list(ohlcv["Close"]),
    }
    overlays_out = [
        {"name": name, "t": [ts.isoformat() for ts in series.index], "v": _safe_list(series)}
        for name, series in overlays
    ]
    markers_out = [
        {"t": m["ts"].isoformat(), "price": float(m["price"]), "type": m["type"]}
        for m in markers
    ]
    trades_out = [
        {
            "entry_time": t.entry_time.isoformat(),
            "exit_time": t.exit_time.isoformat(),
            "direction": t.direction,
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "entry_reason": t.entry_reason,
            "exit_reason": t.exit_reason,
            "pnl": t.pnl,
            "pnl_pct": t.pnl_pct,
        }
        for t in trades
    ]

    total_pnl = sum(t.pnl for t in trades)
    total_pct = sum(t.pnl_pct for t in trades)
    wins = sum(1 for t in trades if t.pnl > 0)

    progress(100, "Completado")
    return {
        "ok": True,
        "ohlcv": ohlcv_out,
        "overlays": overlays_out,
        "markers": markers_out,
        "trades": trades_out,
        "metrics": {
            "trades": len(trades),
            "wins": wins,
            "losses": len(trades) - wins,
            "winrate": (wins / len(trades) * 100) if trades else 0.0,
            "pnl": total_pnl,
            "pnl_pct": total_pct,
        },
    }


def _update_job(job_id: str, **kwargs):
    with _run_jobs_lock:
        job = _run_jobs.get(job_id)
        if job is None:
            return
        job.update(kwargs)
        job["updated_at"] = time.time()


def _run_job(job_id: str, payload: dict):
    try:
        result = _run_backtest_payload(payload, progress_cb=lambda p, m: _update_job(job_id, progress=p, message=m))
        _update_job(job_id, status="done", progress=100.0, message="Completado", result=result)
    except Exception as exc:
        _update_job(job_id, status="error", message=str(exc), error=str(exc))


@app.post("/api/run")
async def run_strategy(payload: dict):
    try:
        return _run_backtest_payload(payload)
    except ValueError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
    except Exception as exc:
        return JSONResponse({"ok": False, "error": f"Error interno: {exc}"}, status_code=500)


@app.post("/api/run_async")
async def run_strategy_async(payload: dict):
    global _loaded_path, _loaded_meta
    if _loaded_path is None or _loaded_meta is None:
        return JSONResponse({"ok": False, "error": "No hay parquet cargado"}, status_code=400)

    job_id = uuid.uuid4().hex
    with _run_jobs_lock:
        _run_jobs[job_id] = {
            "status": "running",
            "progress": 0.0,
            "message": "Iniciando",
            "result": None,
            "error": None,
            "created_at": time.time(),
            "updated_at": time.time(),
        }
    th = threading.Thread(target=_run_job, args=(job_id, payload), daemon=True)
    th.start()
    return {"ok": True, "job_id": job_id}


@app.get("/api/run_status/{job_id}")
async def run_status(job_id: str):
    with _run_jobs_lock:
        job = _run_jobs.get(job_id)
        if job is None:
            return JSONResponse({"ok": False, "error": "Job no encontrado"}, status_code=404)
        status = job["status"]
        progress = float(job.get("progress", 0.0))
        message = str(job.get("message", ""))
        result = job.get("result")
        error = job.get("error")
        if status in ("done", "error"):
            del _run_jobs[job_id]

    out: dict[str, Any] = {
        "ok": True,
        "status": status,
        "progress": progress,
        "message": message,
    }
    if status == "done":
        out["result"] = result
    elif status == "error":
        out["error"] = error or "Error desconocido"
    return out


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

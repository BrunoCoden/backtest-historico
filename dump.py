#!/usr/bin/env python3
import os
import sys
import time
import tempfile
import zipfile
import csv
from io import TextIOWrapper
from datetime import datetime, timezone, timedelta
from typing import Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import requests
import pyarrow as pa
import pyarrow.parquet as pq

BINANCE_FAPI = "https://fapi.binance.com/fapi/v1/aggTrades"
BINANCE_ARCHIVE_BASE = "https://data.binance.vision/data/futures/um/daily/aggTrades"
DEFAULT_SAMPLE_SECONDS = 10
DEFAULT_CHUNK_SECONDS = 6 * 3600
DEFAULT_WORKERS = 1
SHOW_PROGRESS = False
ARCHIVE_MAX_MB = float(os.getenv("ARCHIVE_MAX_MB", "0"))  # 0 = sin límite


def _parse_dt(value: str) -> datetime:
    """Parse datetime string. If naive, assume UTC-3."""
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        dt = datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone(timedelta(hours=-3)))
    return dt


def _to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def _ms_to_iso_local(ms: int) -> str:
    tz = timezone(timedelta(hours=-3))
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).astimezone(tz).isoformat()


def _get_with_retry(params: dict, max_retries: int = 10, base_sleep: float = 1.0):
    last_exc = None
    for attempt in range(max_retries):
        try:
            resp = requests.get(BINANCE_FAPI, params=params, timeout=20)
            if resp.status_code in (418, 429):
                sleep_s = min(base_sleep * (2 ** attempt), 30)
                if SHOW_PROGRESS:
                    sys.stdout.write(
                        f"\rRate limit (HTTP {resp.status_code}), retry {attempt+1}/{max_retries} in {sleep_s:.1f}s   "
                    )
                    sys.stdout.flush()
                time.sleep(sleep_s)
                continue
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            last_exc = exc
            time.sleep(min(base_sleep * (2 ** attempt), 10))
    if last_exc:
        raise last_exc
    raise RuntimeError("No se pudo descargar datos (reintentos agotados)")

def _download_file(url: str, dst: str, max_retries: int = 6, max_size_bytes: int | None = None) -> bool:
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, stream=True, timeout=30)
            if max_size_bytes is not None:
                cl = resp.headers.get("Content-Length")
                if cl is not None:
                    try:
                        if int(cl) > max_size_bytes:
                            return False
                    except Exception:
                        pass
            if resp.status_code == 404:
                return False
            if resp.status_code in (418, 429):
                time.sleep(min(2 ** attempt, 30))
                continue
            resp.raise_for_status()
            total = 0
            first_chunk = b""
            with open(dst, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    if chunk:
                        if not first_chunk:
                            first_chunk = chunk[:8]
                        total += len(chunk)
                        if max_size_bytes is not None and total > max_size_bytes:
                            try:
                                os.remove(dst)
                            except OSError:
                                pass
                            return False
                        f.write(chunk)
            # Algunos proxies pueden responder JSON de warning en lugar de ZIP.
            if not first_chunk.startswith(b"PK"):
                try:
                    os.remove(dst)
                except OSError:
                    pass
                return False
            return True
        except Exception:
            time.sleep(min(2 ** attempt, 30))
    return False

def _iter_archive_days(start_dt: datetime, end_dt: datetime):
    cur = start_dt.date()
    end = end_dt.date()
    one = timedelta(days=1)
    while cur <= end:
        yield cur
        cur = cur + one

def _rows_from_archive(symbol: str, start_ms: int, end_ms: int, sample_seconds: int):
    cur_bucket = None
    agg_qty = 0.0
    last_price = None
    last_t = None
    last_side = None
    last_trade_id = None

    start_dt = datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc)
    end_dt = datetime.fromtimestamp(end_ms / 1000, tz=timezone.utc)

    tmp_dir = tempfile.mkdtemp(prefix="aggtrade_zip_")
    days = list(_iter_archive_days(start_dt, end_dt))
    total_days = len(days)

    max_size_bytes = None
    if ARCHIVE_MAX_MB and ARCHIVE_MAX_MB > 0:
        max_size_bytes = int(ARCHIVE_MAX_MB * 1024 * 1024)

    for i, day in enumerate(days, start=1):
        day_str = day.strftime("%Y-%m-%d")
        url = f"{BINANCE_ARCHIVE_BASE}/{symbol}/{symbol}-aggTrades-{day_str}.zip"
        zip_path = os.path.join(tmp_dir, f"{symbol}-{day_str}.zip")
        # Si el archivo supera el límite (ARCHIVE_MAX_MB), hacemos fallback a API para ese día.
        ok = _download_file(url, zip_path, max_size_bytes=max_size_bytes)
        if SHOW_PROGRESS:
            pct = (i / total_days) * 100.0
            sys.stdout.write(f"\rDescargando {i}/{total_days} ({pct:5.1f}%) {day_str}")
            sys.stdout.flush()
        if not ok:
            # Fallback a API para ese día
            if SHOW_PROGRESS:
                sys.stdout.write(f"\rArchivo grande o no disponible, fallback API {day_str}        ")
                sys.stdout.flush()
            day_start = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
            day_end = day_start + timedelta(days=1) - timedelta(milliseconds=1)
            s_ms = max(int(day_start.timestamp() * 1000), start_ms)
            e_ms = min(int(day_end.timestamp() * 1000), end_ms)
            for item in fetch_aggtrades(symbol, s_ms, e_ms, sleep_s=0.05):
                try:
                    trade_id = int(item.get("a"))
                    t_ms = int(item.get("T"))
                    price = float(item.get("p"))
                    qty = float(item.get("q"))
                    buyer_maker = bool(item.get("m"))
                    side = "sell" if buyer_maker else "buy"
                except Exception:
                    continue
                if t_ms < start_ms or t_ms > end_ms:
                    continue
                if sample_seconds and sample_seconds > 0:
                    bucket = t_ms // (sample_seconds * 1000)
                    if cur_bucket is None:
                        cur_bucket = bucket
                    if bucket != cur_bucket:
                        if last_t is not None and last_price is not None and last_side is not None and last_trade_id is not None:
                            yield (last_trade_id, last_t, last_price, agg_qty, last_side)
                        agg_qty = 0.0
                        last_price = None
                        last_t = None
                        last_side = None
                        last_trade_id = None
                        cur_bucket = bucket
                    agg_qty += qty
                    last_price = price
                    last_t = t_ms
                    last_side = side
                    last_trade_id = trade_id
                else:
                    yield (trade_id, t_ms, price, qty, side)
            continue

        try:
            with zipfile.ZipFile(zip_path) as zf:
                # take first csv inside
                names = [n for n in zf.namelist() if n.endswith(".csv")]
                if not names:
                    continue
                with zf.open(names[0]) as f:
                    reader = csv.reader(TextIOWrapper(f, "utf-8"))
                    for row in reader:
                        if not row or row[0].startswith("aggTradeId"):
                            continue
                        try:
                            trade_id = int(row[0])
                            price = float(row[1])
                            qty = float(row[2])
                            t_ms = int(row[5])
                            buyer_maker = row[6].lower() == "true"
                            side = "sell" if buyer_maker else "buy"
                        except Exception:
                            continue
                        if t_ms < start_ms or t_ms > end_ms:
                            continue
                        if sample_seconds and sample_seconds > 0:
                            bucket = t_ms // (sample_seconds * 1000)
                            if cur_bucket is None:
                                cur_bucket = bucket
                            if bucket != cur_bucket:
                                if last_t is not None and last_price is not None and last_side is not None and last_trade_id is not None:
                                    yield (last_trade_id, last_t, last_price, agg_qty, last_side)
                                agg_qty = 0.0
                                last_price = None
                                last_t = None
                                last_side = None
                                last_trade_id = None
                                cur_bucket = bucket
                            agg_qty += qty
                            last_price = price
                            last_t = t_ms
                            last_side = side
                            last_trade_id = trade_id
                        else:
                            yield (trade_id, t_ms, price, qty, side)
        except zipfile.BadZipFile:
            # Fallback a API si un proxy/CDN devolvio payload no-ZIP.
            if SHOW_PROGRESS:
                sys.stdout.write(f"\rZIP invalido, fallback API {day_str}        ")
                sys.stdout.flush()
            day_start = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
            day_end = day_start + timedelta(days=1) - timedelta(milliseconds=1)
            s_ms = max(int(day_start.timestamp() * 1000), start_ms)
            e_ms = min(int(day_end.timestamp() * 1000), end_ms)
            for item in fetch_aggtrades(symbol, s_ms, e_ms, sleep_s=0.05):
                try:
                    trade_id = int(item.get("a"))
                    t_ms = int(item.get("T"))
                    price = float(item.get("p"))
                    qty = float(item.get("q"))
                    buyer_maker = bool(item.get("m"))
                    side = "sell" if buyer_maker else "buy"
                except Exception:
                    continue
                if t_ms < start_ms or t_ms > end_ms:
                    continue
                if sample_seconds and sample_seconds > 0:
                    bucket = t_ms // (sample_seconds * 1000)
                    if cur_bucket is None:
                        cur_bucket = bucket
                    if bucket != cur_bucket:
                        if last_t is not None and last_price is not None and last_side is not None and last_trade_id is not None:
                            yield (last_trade_id, last_t, last_price, agg_qty, last_side)
                        agg_qty = 0.0
                        last_price = None
                        last_t = None
                        last_side = None
                        last_trade_id = None
                        cur_bucket = bucket
                    agg_qty += qty
                    last_price = price
                    last_t = t_ms
                    last_side = side
                    last_trade_id = trade_id
                else:
                    yield (trade_id, t_ms, price, qty, side)

    if sample_seconds and sample_seconds > 0 and last_t is not None:
        yield (last_trade_id, last_t, last_price, agg_qty, last_side)


def fetch_aggtrades(symbol: str, start_ms: int, end_ms: int, sleep_s: float = 0.2):
    cur = start_ms
    while True:
        params = {
            "symbol": symbol,
            "startTime": cur,
            "endTime": end_ms,
            "limit": 1000,
        }
        data = _get_with_retry(params)
        if not isinstance(data, list) or not data:
            break
        last_t = None
        for item in data:
            t = int(item.get("T"))
            if t > end_ms:
                break
            yield item
            last_t = t
        if last_t is None:
            break
        cur = last_t + 1
        if cur > end_ms:
            break
        if SHOW_PROGRESS and last_t is not None:
            _print_progress(last_t, start_ms, end_ms)
        time.sleep(sleep_s)


def _default_out(symbol: str, start: datetime, end: datetime) -> str:
    def fmt(dt: datetime) -> str:
        return dt.astimezone(timezone(timedelta(hours=-3))).strftime("%Y%m%d_%H%M%S")
    return f"data/{symbol}_{fmt(start)}_{fmt(end)}.parquet"


def _print_progress(done_ms: int, start_ms: int, end_ms: int) -> None:
    total = max(end_ms - start_ms, 1)
    done = max(min(done_ms - start_ms, total), 0)
    pct = (done / total) * 100.0
    bar_len = 30
    filled = int(bar_len * pct / 100.0)
    bar = "#" * filled + "-" * (bar_len - filled)
    sys.stdout.write(f"\r[{bar}] {pct:6.2f}%")
    sys.stdout.flush()

def _emit_rows_to_parquet(
    out_path: str,
    symbol: str,
    rows_iter,
) -> int:
    schema = pa.schema([
        ("trade_id", pa.int64()),
        ("symbol", pa.string()),
        ("timestamp_ms_utc", pa.int64()),
        ("timestamp", pa.string()),
        ("price", pa.float64()),
        ("qty", pa.float64()),
        ("side", pa.string()),
    ])
    writer: Optional[pq.ParquetWriter] = None
    total = 0
    batch_trade_id = []
    batch_symbol = []
    batch_ts_ms = []
    batch_ts = []
    batch_price = []
    batch_qty = []
    batch_side = []

    def flush_batch():
        nonlocal writer, total
        if not batch_trade_id:
            return
        table = pa.Table.from_arrays(
            [
                pa.array(batch_trade_id, type=pa.int64()),
                pa.array(batch_symbol, type=pa.string()),
                pa.array(batch_ts_ms, type=pa.int64()),
                pa.array(batch_ts, type=pa.string()),
                pa.array(batch_price, type=pa.float64()),
                pa.array(batch_qty, type=pa.float64()),
                pa.array(batch_side, type=pa.string()),
            ],
            schema=schema,
        )
        if writer is None:
            writer = pq.ParquetWriter(out_path, schema=schema)
        writer.write_table(table)
        total += len(batch_trade_id)
        batch_trade_id.clear()
        batch_symbol.clear()
        batch_ts_ms.clear()
        batch_ts.clear()
        batch_price.clear()
        batch_qty.clear()
        batch_side.clear()

    for trade_id, t_ms, price, qty, side in rows_iter:
        batch_trade_id.append(trade_id)
        batch_symbol.append(symbol)
        batch_ts_ms.append(t_ms)
        batch_ts.append(_ms_to_iso_local(t_ms))
        batch_price.append(price)
        batch_qty.append(qty)
        batch_side.append(side)
        if len(batch_trade_id) >= 50000:
            flush_batch()

    flush_batch()
    if writer is not None:
        writer.close()
    return total


def _process_chunk(args) -> tuple[str, int, int]:
    symbol, start_ms, end_ms, sample_seconds, out_path = args

    def rows():
        cur_bucket = None
        agg_qty = 0.0
        last_price = None
        last_t = None
        last_side = None
        last_trade_id = None
        for item in fetch_aggtrades(symbol, start_ms, end_ms):
            trade_id = int(item.get("a"))
            t_ms = int(item.get("T"))
            price = float(item.get("p"))
            qty = float(item.get("q"))
            buyer_maker = bool(item.get("m"))
            side = "sell" if buyer_maker else "buy"
            if sample_seconds and sample_seconds > 0:
                bucket = t_ms // (sample_seconds * 1000)
                if cur_bucket is None:
                    cur_bucket = bucket
                if bucket != cur_bucket:
                    if last_t is not None and last_price is not None and last_side is not None and last_trade_id is not None:
                        yield (last_trade_id, last_t, last_price, agg_qty, last_side)
                    agg_qty = 0.0
                    last_price = None
                    last_t = None
                    last_side = None
                    last_trade_id = None
                    cur_bucket = bucket
                agg_qty += qty
                last_price = price
                last_t = t_ms
                last_side = side
                last_trade_id = trade_id
            else:
                yield (trade_id, t_ms, price, qty, side)
        if sample_seconds and sample_seconds > 0 and last_t is not None:
            yield (last_trade_id, last_t, last_price, agg_qty, last_side)

    total = _emit_rows_to_parquet(out_path, symbol, rows())
    return out_path, total, end_ms

def main(argv: list[str]) -> int:
    if len(argv) < 4:
        print("Uso: dump.py SYMBOL START END [OUT] [SAMPLE_SECONDS] [WORKERS]", file=sys.stderr)
        print("Ej: dump.py ETHUSDT 2026-02-10T00:00:00-03:00 2026-02-10T06:00:00-03:00", file=sys.stderr)
        return 1

    symbol = argv[1].upper()
    try:
        start_dt = _parse_dt(argv[2])
        end_dt = _parse_dt(argv[3])
    except Exception as exc:
        print(f"Error parseando fechas: {exc}", file=sys.stderr)
        return 1

    if end_dt <= start_dt:
        print("END debe ser mayor que START", file=sys.stderr)
        return 1

    out_path = _default_out(symbol, start_dt, end_dt)
    sample_seconds = DEFAULT_SAMPLE_SECONDS
    workers = DEFAULT_WORKERS
    if len(argv) >= 5:
        if argv[4].isdigit():
            sample_seconds = int(argv[4])
        else:
            out_path = argv[4]
    if len(argv) >= 6:
        sample_seconds = int(argv[5])
    if len(argv) >= 7:
        workers = max(1, int(argv[6]))
    # Ensure output directory exists
    out_dir = out_path.rsplit("/", 1)[0] if "/" in out_path else "."
    if out_dir and out_dir != ".":
        import os
        os.makedirs(out_dir, exist_ok=True)

    start_ms = _to_ms(start_dt)
    end_ms = _to_ms(end_dt)

    # Use archive by default for long ranges to avoid rate limits
    use_archive = (end_ms - start_ms) > 24 * 3600 * 1000
    if use_archive:
        global SHOW_PROGRESS
        SHOW_PROGRESS = True
        total = _emit_rows_to_parquet(out_path, symbol, _rows_from_archive(symbol, start_ms, end_ms, sample_seconds))
        sys.stdout.write("\n")
        print(f"OK: {total} rows -> {out_path}")
        return 0

    # Short range: API with optional multiprocessing
    chunk_ms = DEFAULT_CHUNK_SECONDS * 1000
    chunks = []
    cur = start_ms
    while cur <= end_ms:
        c_end = min(cur + chunk_ms - 1, end_ms)
        chunks.append((symbol, cur, c_end, sample_seconds))
        cur = c_end + 1

    tmp_dir = tempfile.mkdtemp(prefix="aggtrade_")
    tasks = []
    for i, (sym, s_ms, e_ms, samp) in enumerate(chunks):
        tmp_out = os.path.join(tmp_dir, f"chunk_{i:04d}.parquet")
        tasks.append((sym, s_ms, e_ms, samp, tmp_out))

    total = 0
    done = 0
    print(f"Chunks: {len(tasks)} | Workers: {workers} | Sample: {sample_seconds}s")

    results = []
    if workers == 1:
        SHOW_PROGRESS = True
        for t in tasks:
            out_file, rows, _ = _process_chunk(t)
            results.append((out_file, rows))
            total += rows
            done += 1
            pct = (done / len(tasks)) * 100.0
            sys.stdout.write(f"\rChunks completos: {done}/{len(tasks)} ({pct:5.1f}%)")
            sys.stdout.flush()
        sys.stdout.write("\n")
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            fut_map = {ex.submit(_process_chunk, t): t for t in tasks}
            for fut in as_completed(fut_map):
                out_file, rows, _ = fut.result()
                results.append((out_file, rows))
                total += rows
                done += 1
                pct = (done / len(tasks)) * 100.0
                sys.stdout.write(f"\rChunks completos: {done}/{len(tasks)} ({pct:5.1f}%)")
                sys.stdout.flush()
        sys.stdout.write("\n")

    # Merge chunks into final parquet
    results.sort()
    schema = pa.schema([
        ("trade_id", pa.int64()),
        ("symbol", pa.string()),
        ("timestamp_ms_utc", pa.int64()),
        ("timestamp", pa.string()),
        ("price", pa.float64()),
        ("qty", pa.float64()),
        ("side", pa.string()),
    ])
    writer = pq.ParquetWriter(out_path, schema=schema)
    for out_file, rows in results:
        if rows <= 0:
            continue
        pf = pq.ParquetFile(out_file)
        for batch in pf.iter_batches(batch_size=100000):
            writer.write_batch(batch)
    writer.close()

    print(f"OK: {total} rows -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

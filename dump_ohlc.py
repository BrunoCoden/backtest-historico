#!/usr/bin/env python3
import csv
import os
import sys
import tempfile
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from io import TextIOWrapper
from typing import Optional

import pyarrow as pa
import pyarrow.parquet as pq
import requests

BINANCE_FAPI = "https://fapi.binance.com/fapi/v1/aggTrades"
BINANCE_ARCHIVE_BASE = "https://data.binance.vision/data/futures/um/daily/aggTrades"
DEFAULT_BUCKET_SECONDS = 1
DEFAULT_WORKERS = max(1, int(os.getenv("DUMP_OHLC_WORKERS", "4")))
DEFAULT_CHUNK_DAYS = max(1, int(os.getenv("DUMP_OHLC_CHUNK_DAYS", "1")))
SHOW_PROGRESS = True
ARCHIVE_MAX_MB = float(os.getenv("ARCHIVE_MAX_MB", "0"))  # 0 = sin limite
API_MAX_RETRIES = max(1, int(os.getenv("API_MAX_RETRIES", "10")))
API_BASE_SLEEP = max(0.1, float(os.getenv("API_BASE_SLEEP", "1.0")))


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
            resp = requests.get(BINANCE_FAPI, params=params, timeout=25)
            if resp.status_code in (418, 429):
                time.sleep(min(base_sleep * (2 ** attempt), 30))
                continue
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            last_exc = exc
            time.sleep(min(base_sleep * (2 ** attempt), 10))
    if last_exc:
        raise last_exc
    raise RuntimeError("No se pudo descargar datos (reintentos agotados)")


def fetch_aggtrades(symbol: str, start_ms: int, end_ms: int, sleep_s: float = 0.05):
    cur = start_ms
    while True:
        params = {
            "symbol": symbol,
            "startTime": cur,
            "endTime": end_ms,
            "limit": 1000,
        }
        data = _get_with_retry(params, max_retries=API_MAX_RETRIES, base_sleep=API_BASE_SLEEP)
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
        time.sleep(sleep_s)


def _download_file(url: str, dst: str, max_retries: int = 6, max_size_bytes: int | None = None) -> bool:
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, stream=True, timeout=45)
            if resp.status_code == 404:
                return False
            if resp.status_code in (418, 429):
                time.sleep(min(2 ** attempt, 30))
                continue
            resp.raise_for_status()

            cl = resp.headers.get("Content-Length")
            if max_size_bytes is not None and cl is not None:
                try:
                    if int(cl) > max_size_bytes:
                        return False
                except Exception:
                    pass

            total = 0
            first_chunk = b""
            with open(dst, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    if not chunk:
                        continue
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

            # Algunos proxies devuelven JSON con warning de 50MB en lugar del ZIP.
            # Si no arranca con firma ZIP "PK", lo consideramos invalido y hacemos fallback a API.
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


def _default_out(symbol: str, start: datetime, end: datetime, bucket_s: int) -> str:
    def fmt(dt: datetime) -> str:
        return dt.astimezone(timezone(timedelta(hours=-3))).strftime("%Y%m%d_%H%M%S")

    return f"data/{symbol}_{fmt(start)}_{fmt(end)}_{bucket_s}s_ohlc.parquet"


def _emit_rows_to_parquet(out_path: str, rows_iter) -> int:
    schema = pa.schema(
        [
            ("bucket_start_ms_utc", pa.int64()),
            ("bucket_start", pa.string()),
            ("open", pa.float64()),
            ("high", pa.float64()),
            ("low", pa.float64()),
            ("close", pa.float64()),
        ]
    )
    writer: Optional[pq.ParquetWriter] = None
    total = 0
    batch = {
        "bucket_start_ms_utc": [],
        "bucket_start": [],
        "open": [],
        "high": [],
        "low": [],
        "close": [],
    }

    def flush_batch():
        nonlocal writer, total
        if not batch["bucket_start_ms_utc"]:
            return
        table = pa.Table.from_arrays(
            [
                pa.array(batch["bucket_start_ms_utc"], type=pa.int64()),
                pa.array(batch["bucket_start"], type=pa.string()),
                pa.array(batch["open"], type=pa.float64()),
                pa.array(batch["high"], type=pa.float64()),
                pa.array(batch["low"], type=pa.float64()),
                pa.array(batch["close"], type=pa.float64()),
            ],
            schema=schema,
        )
        if writer is None:
            writer = pq.ParquetWriter(out_path, schema=schema)
        writer.write_table(table)
        total += len(batch["bucket_start_ms_utc"])
        for k in batch:
            batch[k].clear()

    for row in rows_iter:
        (bucket_start_ms, o, h, l, c) = row
        batch["bucket_start_ms_utc"].append(bucket_start_ms)
        batch["bucket_start"].append(_ms_to_iso_local(bucket_start_ms))
        batch["open"].append(o)
        batch["high"].append(h)
        batch["low"].append(l)
        batch["close"].append(c)
        if len(batch["bucket_start_ms_utc"]) >= 50000:
            flush_batch()

    flush_batch()
    if writer is not None:
        writer.close()
    return total


def _iter_day_rows_from_archive(zip_path: str, start_ms: int, end_ms: int):
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.endswith(".csv")]
        if not names:
            return
        with zf.open(names[0]) as f:
            reader = csv.reader(TextIOWrapper(f, "utf-8"))
            for row in reader:
                if not row or row[0].startswith("aggTradeId"):
                    continue
                try:
                    t_ms = int(row[5])
                    price = float(row[1])
                except Exception:
                    continue
                if t_ms < start_ms or t_ms > end_ms:
                    continue
                yield (t_ms, price)


def _iter_day_rows_from_api(symbol: str, start_ms: int, end_ms: int):
    for item in fetch_aggtrades(symbol, start_ms, end_ms):
        try:
            t_ms = int(item.get("T"))
            price = float(item.get("p"))
        except Exception:
            continue
        if t_ms < start_ms or t_ms > end_ms:
            continue
        yield (t_ms, price)


def _rows_from_archive(symbol: str, start_ms: int, end_ms: int, bucket_seconds: int):
    bucket_ms = bucket_seconds * 1000
    start_dt = datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc)
    end_dt = datetime.fromtimestamp(end_ms / 1000, tz=timezone.utc)

    tmp_dir = tempfile.mkdtemp(prefix="aggtrade_zip_")
    days = list(_iter_archive_days(start_dt, end_dt))
    total_days = len(days)

    max_size_bytes = None
    if ARCHIVE_MAX_MB and ARCHIVE_MAX_MB > 0:
        max_size_bytes = int(ARCHIVE_MAX_MB * 1024 * 1024)

    cur_bucket = None
    o = h = l = c = None
    def emit_current():
        if cur_bucket is None:
            return None
        return (cur_bucket, o, h, l, c)

    for i, day in enumerate(days, start=1):
        day_start = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
        day_end = day_start + timedelta(days=1) - timedelta(milliseconds=1)
        s_ms = max(start_ms, int(day_start.timestamp() * 1000))
        e_ms = min(end_ms, int(day_end.timestamp() * 1000))
        if s_ms > e_ms:
            continue

        day_str = day.strftime("%Y-%m-%d")
        url = f"{BINANCE_ARCHIVE_BASE}/{symbol}/{symbol}-aggTrades-{day_str}.zip"
        zip_path = os.path.join(tmp_dir, f"{symbol}-{day_str}.zip")

        ok = _download_file(url, zip_path, max_size_bytes=max_size_bytes)
        if SHOW_PROGRESS:
            pct = (i / total_days) * 100.0
            sys.stdout.write(f"\rDescargando {i}/{total_days} ({pct:5.1f}%) {day_str}")
            sys.stdout.flush()

        if ok:
            rows = _iter_day_rows_from_archive(zip_path, s_ms, e_ms)
        else:
            if SHOW_PROGRESS:
                sys.stdout.write(f"\rFallback API {day_str}                                ")
                sys.stdout.flush()
            rows = _iter_day_rows_from_api(symbol, s_ms, e_ms)

        for t_ms, price in rows:
            b = (t_ms // bucket_ms) * bucket_ms
            if cur_bucket is None:
                cur_bucket = b
                o = h = l = c = price
                continue
            if b != cur_bucket:
                out = emit_current()
                if out is not None:
                    yield out
                cur_bucket = b
                o = h = l = c = price
                continue

            # same bucket
            if price > h:
                h = price
            if price < l:
                l = price
            c = price

    out = emit_current()
    if out is not None:
        yield out


def _build_chunks(start_ms: int, end_ms: int, chunk_days: int):
    chunk_ms = max(1, chunk_days) * 24 * 3600 * 1000
    cur = start_ms
    while cur <= end_ms:
        c_end = min(cur + chunk_ms - 1, end_ms)
        yield (cur, c_end)
        cur = c_end + 1


def _process_ohlc_chunk(args) -> tuple[str, int, int]:
    symbol, start_ms, end_ms, bucket_seconds, out_path, show_progress = args
    global SHOW_PROGRESS
    SHOW_PROGRESS = show_progress
    total = _emit_rows_to_parquet(out_path, _rows_from_archive(symbol, start_ms, end_ms, bucket_seconds))
    return out_path, total, start_ms


def _merge_chunk_parquets(
    out_path: str,
    parts: list[tuple[str, int, int]],
    bucket_seconds: int,
    range_start_ms: int,
    range_end_ms: int,
    fill_missing: bool = True,
) -> int:
    schema = pa.schema(
        [
            ("bucket_start_ms_utc", pa.int64()),
            ("bucket_start", pa.string()),
            ("open", pa.float64()),
            ("high", pa.float64()),
            ("low", pa.float64()),
            ("close", pa.float64()),
        ]
    )
    writer: Optional[pq.ParquetWriter] = None
    total_rows = 0
    pending: tuple[int, float, float, float, float] | None = None

    batch = {
        "bucket_start_ms_utc": [],
        "bucket_start": [],
        "open": [],
        "high": [],
        "low": [],
        "close": [],
    }

    def flush_batch():
        nonlocal writer, total_rows
        if not batch["bucket_start_ms_utc"]:
            return
        table = pa.Table.from_arrays(
            [
                pa.array(batch["bucket_start_ms_utc"], type=pa.int64()),
                pa.array(batch["bucket_start"], type=pa.string()),
                pa.array(batch["open"], type=pa.float64()),
                pa.array(batch["high"], type=pa.float64()),
                pa.array(batch["low"], type=pa.float64()),
                pa.array(batch["close"], type=pa.float64()),
            ],
            schema=schema,
        )
        if writer is None:
            writer = pq.ParquetWriter(out_path, schema=schema)
        writer.write_table(table)
        total_rows += len(batch["bucket_start_ms_utc"])
        for k in batch:
            batch[k].clear()

    def push_row(row: tuple[int, float, float, float, float]):
        batch["bucket_start_ms_utc"].append(row[0])
        batch["bucket_start"].append(_ms_to_iso_local(row[0]))
        batch["open"].append(row[1])
        batch["high"].append(row[2])
        batch["low"].append(row[3])
        batch["close"].append(row[4])
        if len(batch["bucket_start_ms_utc"]) >= 50000:
            flush_batch()

    bucket_ms = bucket_seconds * 1000
    start_bucket = (range_start_ms // bucket_ms) * bucket_ms
    end_bucket = (range_end_ms // bucket_ms) * bucket_ms
    last_written: tuple[int, float, float, float, float] | None = None

    def write_row_with_fill(row: tuple[int, float, float, float, float]):
        nonlocal last_written
        if fill_missing:
            if last_written is None:
                b = start_bucket
                seed = row[1]  # primer open observado para backfill inicial
                while b < row[0]:
                    filler = (b, seed, seed, seed, seed)
                    push_row(filler)
                    last_written = filler
                    b += bucket_ms
            else:
                b = last_written[0] + bucket_ms
                while b < row[0]:
                    seed = last_written[4]
                    filler = (b, seed, seed, seed, seed)
                    push_row(filler)
                    last_written = filler
                    b += bucket_ms
        push_row(row)
        last_written = row

    for part_path, rows, _ in sorted(parts, key=lambda x: x[2]):
        if rows <= 0 or not os.path.exists(part_path):
            continue
        pf = pq.ParquetFile(part_path)
        for rb in pf.iter_batches(batch_size=100000):
            ms_col = rb.column(0).to_pylist()
            o_col = rb.column(2).to_pylist()
            h_col = rb.column(3).to_pylist()
            l_col = rb.column(4).to_pylist()
            c_col = rb.column(5).to_pylist()
            for i in range(len(ms_col)):
                cur_row = (
                    int(ms_col[i]),
                    float(o_col[i]),
                    float(h_col[i]),
                    float(l_col[i]),
                    float(c_col[i]),
                )
                if pending is None:
                    pending = cur_row
                    continue
                if cur_row[0] == pending[0]:
                    pending = (
                        pending[0],
                        pending[1],  # open = primero
                        max(pending[2], cur_row[2]),
                        min(pending[3], cur_row[3]),
                        cur_row[4],  # close = ultimo
                    )
                else:
                    write_row_with_fill(pending)
                    pending = cur_row

    if pending is not None:
        write_row_with_fill(pending)

    # Completar hasta el final pedido con ultimo close conocido.
    if fill_missing and last_written is not None:
        b = last_written[0] + bucket_ms
        while b <= end_bucket:
            seed = last_written[4]
            filler = (b, seed, seed, seed, seed)
            push_row(filler)
            last_written = filler
            b += bucket_ms

    flush_batch()

    if writer is not None:
        writer.close()
        return total_rows

    # Sin filas: crear parquet vacio para mantener comportamiento consistente.
    empty = pa.Table.from_arrays(
        [
            pa.array([], type=pa.int64()),
            pa.array([], type=pa.string()),
            pa.array([], type=pa.float64()),
            pa.array([], type=pa.float64()),
            pa.array([], type=pa.float64()),
            pa.array([], type=pa.float64()),
        ],
        schema=schema,
    )
    pq.write_table(empty, out_path)
    return 0


def main(argv: list[str]) -> int:
    if len(argv) < 4:
        print(
            "Uso: dump_ohlc.py SYMBOL START END [OUT] [BUCKET_SECONDS] [WORKERS] [CHUNK_DAYS]",
            file=sys.stderr,
        )
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

    out_path = _default_out(symbol, start_dt, end_dt, DEFAULT_BUCKET_SECONDS)
    bucket_seconds = DEFAULT_BUCKET_SECONDS
    workers = DEFAULT_WORKERS
    chunk_days = DEFAULT_CHUNK_DAYS

    # Compatibilidad:
    # - argv[4] puede ser OUT o BUCKET_SECONDS
    # - si argv[4] es OUT, argv[5] puede ser BUCKET_SECONDS
    if len(argv) >= 5:
        if argv[4].isdigit():
            bucket_seconds = int(argv[4])
            if len(argv) >= 6:
                workers = max(1, int(argv[5]))
            if len(argv) >= 7:
                chunk_days = max(1, int(argv[6]))
        else:
            out_path = argv[4]
            if len(argv) >= 6:
                bucket_seconds = int(argv[5])
            if len(argv) >= 7:
                workers = max(1, int(argv[6]))
            if len(argv) >= 8:
                chunk_days = max(1, int(argv[7]))

    out_dir = out_path.rsplit("/", 1)[0] if "/" in out_path else "."
    if out_dir and out_dir != ".":
        os.makedirs(out_dir, exist_ok=True)

    start_ms = _to_ms(start_dt)
    end_ms = _to_ms(end_dt)
    chunks = list(_build_chunks(start_ms, end_ms, chunk_days))

    tmp_dir = tempfile.mkdtemp(prefix="ohlc_parts_")
    tasks = []
    for i, (s_ms, e_ms) in enumerate(chunks):
        part_out = os.path.join(tmp_dir, f"part_{i:05d}.parquet")
        tasks.append((symbol, s_ms, e_ms, bucket_seconds, part_out))

    print(
        f"Chunks: {len(tasks)} | Workers: {workers} | Bucket: {bucket_seconds}s | Chunk days: {chunk_days}"
    )

    done = 0
    results: list[tuple[str, int, int]] = []

    if workers <= 1 or len(tasks) == 1:
        for task in tasks:
            out_file, rows, s_ms = _process_ohlc_chunk((*task, True))
            results.append((out_file, rows, s_ms))
            done += 1
            pct = (done / len(tasks)) * 100.0
            sys.stdout.write(f"\rChunks completos: {done}/{len(tasks)} ({pct:5.1f}%)")
            sys.stdout.flush()
        sys.stdout.write("\n")
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            fut_map = {
                ex.submit(_process_ohlc_chunk, (*task, False)): task for task in tasks
            }
            for fut in as_completed(fut_map):
                out_file, rows, s_ms = fut.result()
                results.append((out_file, rows, s_ms))
                done += 1
                pct = (done / len(tasks)) * 100.0
                sys.stdout.write(f"\rChunks completos: {done}/{len(tasks)} ({pct:5.1f}%)")
                sys.stdout.flush()
        sys.stdout.write("\n")

    total = _merge_chunk_parquets(
        out_path,
        results,
        bucket_seconds=bucket_seconds,
        range_start_ms=start_ms,
        range_end_ms=end_ms,
        fill_missing=True,
    )
    print(f"OK: {total} rows -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

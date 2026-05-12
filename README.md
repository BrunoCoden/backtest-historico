# Backtest historico (Binance Futures aggTrades)

Descarga aggTrades de Binance Futures (USDT-M) y los guarda en **Parquet**.

## Instalacion
```bash
pip install -r requirements.txt
```

## Uso (simple)
```bash
python dump.py ETHUSDT 2026-02-10T00:00:00-03:00 2026-02-10T06:00:00-03:00
```

## Generar OHLC por bucket (recomendado para backtest rápido con SL/TP real)
```bash
python dump_ohlc.py ETHUSDT 2025-02-10T00:00:00-03:00 2026-02-10T06:00:00-03:00
```

Opcional: ruta de salida
```bash
python dump_ohlc.py ETHUSDT 2025-02-10T00:00:00-03:00 2026-02-10T06:00:00-03:00 /home/diego/data/ETHUSDT_5s_ohlc.parquet
```

Opcional: bucket en segundos (default 1s)
```bash
python dump_ohlc.py ETHUSDT 2025-02-10T00:00:00-03:00 2026-02-10T06:00:00-03:00 /home/diego/data/ETHUSDT_10s_ohlc.parquet 10
```

Opcional: paralelizar por chunks diarios (`workers=4`, `chunk_days=1`)
```bash
python dump_ohlc.py ETHUSDT 2025-02-10T00:00:00-03:00 2026-02-10T06:00:00-03:00 /home/diego/data/ETHUSDT_1s_ohlc.parquet 1 4 1
```

Si no pasas `workers/chunk_days`, usa defaults:
- `DUMP_OHLC_WORKERS` (default `4`)
- `DUMP_OHLC_CHUNK_DAYS` (default `1`)

Opcional: ruta de salida
```bash
python dump.py ETHUSDT 2026-02-10T00:00:00-03:00 2026-02-10T06:00:00-03:00 /home/diego/data/ETHUSDT_20260210.parquet
```

Opcional: muestrear cada N segundos (default 10s)
```bash
python dump.py ETHUSDT 2026-02-10T00:00:00-03:00 2026-02-10T06:00:00-03:00 /home/diego/data/ETHUSDT_20260210.parquet 1
```

Opcional: workers (default 1)
```bash
python dump.py ETHUSDT 2026-02-10T00:00:00-03:00 2026-02-10T06:00:00-03:00 /home/diego/data/ETHUSDT_20260210.parquet 10 1
```

### Notas
- Si START/END no tienen timezone, se asume **UTC-3**.
- Si no pasas OUT, guarda en `data/` con nombre automatico.
- Para rangos largos (mas de 1 dia) usa el archivo historico de Binance (data.binance.vision) y evita rate limit.
- Para rangos cortos usa API y opcional multiproceso (default workers=1).
- `ARCHIVE_MAX_MB=0` (default) desactiva límite de tamaño al descargar ZIPs del archivo.
- Si querés limitar tamaño y forzar fallback a API por día: `ARCHIVE_MAX_MB=50`.
- Si un proxy/CDN devuelve JSON (ej: `{"warning":"file size exceeds 50MB limit"}`) en vez de ZIP, `dump_ohlc.py` hace fallback automático a API para ese día.
- En `dump_ohlc.py`, si faltan segundos sin trades, se completan con `OHLC=ultimo close` y `volume=0` para mantener grilla temporal continua.

## Columnas
- trade_id
- symbol
- timestamp_ms_utc
- timestamp (UTC-3)
- price
- qty
- side

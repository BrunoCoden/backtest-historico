# Contexto del repo `/home/diego/backtest historico`

Este archivo resume exclusivamente el estado, propósito, scripts y uso del repo **backtest historico**.  
No incluye información de otros repos/bots.

## Objetivo del repo
- Descargar datos históricos de Binance Futures y generar **parquet** para backtest.
- Visualizar velas y ejecutar backtests con **Plotly**.
- Proveer una UI local (HTML/JS) para cargar parquet, elegir rango y estrategia, y ver PnL.

## Estructura principal
- `dump.py`: descarga aggTrades históricos y genera parquet (tick/agregado).
- `dump_ohlc.py`: genera parquet **OHLC** en buckets (p.ej. 5s) a partir de data histórica.
- `plot_candles.py` / `plot_candles_open.py`: dibuja velas desde parquet (Plotly).
- `backtest_plotly.py`: backtest + generación de HTML + señales sobre el gráfico.
- `ui_server.py`: servidor FastAPI que sirve `ui/index.html`.
- `ui/index.html`: UI web para cargar parquet y ejecutar backtest desde el browser.
- `run_ui.sh`: script para levantar la UI.
- `requirements.txt`: dependencias (pandas, plotly, fastapi, uvicorn, pyarrow, etc).

## Scripts clave y uso

### 1) Descargar trades (aggTrades) a parquet
```bash
cd "/home/diego/backtest historico"
source .venv/bin/activate
python dump.py ETHUSDT 2025-02-10T00:00:00-03:00 2026-02-10T06:00:00-03:00
```

Notas:
- `dump.py` soporta multiproceso por chunks.
- Hay manejo de rate limits (429/418) con reintentos.
- Puede tardar mucho para rangos de 1 año si se usa tick‑level.

### 2) Generar parquet OHLC por bucket (recomendado)
```bash
cd "/home/diego/backtest historico"
source .venv/bin/activate
python dump_ohlc.py ETHUSDT 2025-02-10T00:00:00-03:00 2026-02-10T06:00:00-03:00 5
```
- El último parámetro es el bucket en segundos (ej: 5s).

Ejemplo generado:
```
data/ETHUSDT_20250210_000000_20260210_060000_5s_ohlc.parquet
```
Rango detectado (local -03): 2025-02-10 00:00:00 → 2026-02-10 05:59:55.

### 3) Graficar velas desde parquet
```bash
cd "/home/diego/backtest historico"
source .venv/bin/activate
python plot_candles_open.py /home/diego/data/ETHUSDT.parquet 30min
```
Notas:
- El parámetro de timeframe usa formatos tipo `30min`, `1h`, etc (pandas>=2.2).

### 4) Backtest + HTML (CLI)
```bash
cd "/home/diego/backtest historico"
source .venv/bin/activate
python backtest_plotly.py /home/diego/data/ETHUSDT.parquet \
  --strategy bollinger --tf 30min --entry close \
  --notional 30 --fee 0.0004 --sl 0.02 \
  --out bollinger_30m.html --out-trades bollinger_30m_trades.csv
```

## UI Web (FastAPI + HTML)
### Arranque:
```bash
cd "/home/diego/backtest historico"
source .venv/bin/activate
./run_ui.sh
```

### Comportamiento:
- Carga un `.parquet` desde el navegador.
- Permite elegir **rango** (inputs tipo `date` con TZ -03).
- Estrategia, TF, entrada (close/next open), SL/TP, notional, fee.
- Dibuja velas + señales en gráfico Plotly.
- Botón “Ir a última vela”.
- Por defecto, el rango usa todo el parquet si no se toca.

## Dependencias importantes
- `pandas`, `plotly`, `pyarrow`, `fastapi`, `uvicorn`, `python-multipart`, `requests`, `numpy`.
- Se instala en `.venv` local:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Notas operativas
- El repo no toca otros bots.
- El UI sirve solo local (no producción).
- Se usa timezone `America/Argentina/Buenos_Aires` para rangos y etiquetas cuando corresponde.

## Conexión SSH a la VM (OCI)

### Datos de conexión
- Host/IP: `167.126.0.127`
- Usuario: `ubuntu`
- Comando directo (si ya hay key cargada en tu máquina):
```bash
ssh ubuntu@167.126.0.127
```

### Si necesitás forzar una key específica
```bash
ssh -i /ruta/a/tu/key -o IdentitiesOnly=yes ubuntu@167.126.0.127
```

Ejemplo típico usado en la VM para `git` por SSH:
```bash
GIT_SSH_COMMAND='ssh -i /home/ubuntu/.ssh/id_ed25519 -o IdentitiesOnly=yes' git pull
```

### Verificaciones rápidas después de entrar
```bash
whoami
hostname
pwd
```

### Paths útiles relacionados a este flujo
- Repo local de backtest: `/home/diego/backtest historico`
- Archivo de contexto: `/home/diego/backtest historico/CONTEXT.md`
- Repo de bot4 en VM (si necesitás cruzar datos de señales): `/home/ubuntu/bot4BBBtc`

### Logs frecuentes en VM (referencia)
- `bot4BBBtc watcher`: `/var/log/bot4bbb/watcher.log`
- `bot4BBBtc telegram`: `/var/log/bot4bbb/telegram_commands.log`
- `bot4BBBtc heartbeat`: `/var/log/bot4bbb/heartbeat.log`

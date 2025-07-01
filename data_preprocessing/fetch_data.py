

import asyncio
import datetime
import logging
import sqlite3
from typing import List, Tuple

import ccxt.async_support as ccxt
import aiosqlite
import pandas as pd

from dateutil import parser
from tenacity import retry, stop_after_attempt, wait_exponential

# ----------------------------------------------------------------------------
# CONFIGURATION & LOGGING SETUP
# ----------------------------------------------------------------------------
DB_PATH        = "ohlcv_cache.db"
TIMEFRAME      = "1m"
BATCH_LIMIT    = 1000
CONCURRENT_CHUNKS = 10
RETRY_STOP     = stop_after_attempt(50)
RETRY_WAIT     = wait_exponential(multiplier=1.1, min=1, max=30)

# Configure root logger
logging.basicConfig(
    level=logging.ERROR,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ohlcv_fetcher")

# ----------------------------------------------------------------------------
# DATABASE INITIALIZATION
# ----------------------------------------------------------------------------
def init_db():
    logger.info("Initializing database at %s", DB_PATH)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ohlcv_backup (
                symbol TEXT,
                ts     INTEGER,
                open   REAL,
                high   REAL,
                low    REAL,
                close  REAL,
                volume REAL,
                PRIMARY KEY(symbol, ts)
            )""")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_symbol_ts ON ohlcv_backup(symbol, ts)")
        conn.commit()
    logger.info("Database initialized")

# ----------------------------------------------------------------------------
# TIME CONVERSION
# ----------------------------------------------------------------------------
def to_milliseconds(dt_str: str) -> int:
    dt = parser.isoparse(dt_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    else:
        dt = dt.astimezone(datetime.timezone.utc)
    ms = int(dt.timestamp() * 1000)
    logger.debug("Parsed %s into %d ms", dt_str, ms)
    return ms

# ----------------------------------------------------------------------------
# FETCHING WITH RETRIES
# ----------------------------------------------------------------------------
@retry(stop=RETRY_STOP, wait=RETRY_WAIT)
async def fetch_batch(exchange: ccxt.Exchange, symbol: str, since_ms: int) -> List[List]:
    since_dt = datetime.datetime.utcfromtimestamp(since_ms / 1000).isoformat() + "Z"
    logger.info("Fetching batch for %s since %s", symbol, since_dt)
    data = await exchange.fetch_ohlcv(
        symbol=symbol,
        timeframe=TIMEFRAME,
        since=since_ms,
        limit=BATCH_LIMIT,
    )
    if data is None:
        logger.warning("Received None for batch since %s, treating as empty", since_dt)
        return []
    logger.info("Fetched %d rows for %s since %s", len(data), symbol, since_dt)
    return data

# ----------------------------------------------------------------------------
# DATABASE CHECKPOINTS
# ----------------------------------------------------------------------------
async def get_last_timestamp(symbol: str) -> int:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT COALESCE(MAX(ts), -1) FROM ohlcv_backup WHERE symbol = ?",
            (symbol,)
        ) as cur:
            row = await cur.fetchone()
            last = row[0] or -1
    logger.info("Last saved timestamp for %s is %d", symbol, last)
    return last

async def save_to_db(symbol: str, records: List[Tuple[int, float, float, float, float, float]]):
    if not records:
        logger.info("No records to save for %s", symbol)
        return
    logger.info("Saving %d records for %s to database", len(records), symbol)
    async with aiosqlite.connect(DB_PATH) as db:
        await db.executemany(
            "INSERT OR IGNORE INTO ohlcv_backup(symbol, ts, open, high, low, close, volume) VALUES (?, ?, ?, ?, ?, ?, ?)",
            [(symbol, *r) for r in records]
        )
        await db.commit()
    logger.info("Database commit complete for %s", symbol)

# ----------------------------------------------------------------------------
# MAIN FETCH + CLEAN
# ----------------------------------------------------------------------------
async def fetch_ohlcv_to_df(symbol: str, start: str, end: str) -> pd.DataFrame:
    logger.info("Starting fetch for %s from %s to %s", symbol, start, end)
    init_db()

    start_ms = to_milliseconds(start)
    end_ms   = to_milliseconds(end)

    exchange = ccxt.binance({"enableRateLimit": True})
    await exchange.load_markets()
    logger.info("Connected to Binance, markets loaded")

    last_ms = await get_last_timestamp(symbol)
    next_ms = max(last_ms + 60_000, start_ms)
    if next_ms > end_ms:
        logger.info("All data already fetched up to %s, skipping fetch", end)
    else:
        # build batch start list
        all_starts = list(range(next_ms, end_ms, BATCH_LIMIT * 60_000))
        logger.info("Total batches to fetch: %d", len(all_starts))

        # iterate in chunks
        for i in range(0, len(all_starts), CONCURRENT_CHUNKS):
            group = all_starts[i:i + CONCURRENT_CHUNKS]
            logger.info("Fetching batch group %d – %d", i, i + len(group) - 1)
            tasks = [fetch_batch(exchange, symbol, ts) for ts in group]
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            for idx, resp in enumerate(responses):
                ts = group[idx]
                if isinstance(resp, Exception):
                    logger.error("Batch starting %d failed: %s", ts, resp)
                    continue
                records = [(r[0], r[1], r[2], r[3], r[4], r[5]) for r in resp]
                await save_to_db(symbol, records)
        logger.info("All batches fetched and saved")

    await exchange.close()
    logger.info("Exchange connection closed")

    # load into pandas
    logger.info("Loading data from SQLite into DataFrame")
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT ts, open, high, low, close, volume FROM ohlcv_backup "
        "WHERE symbol=? AND ts BETWEEN ? AND ? ORDER BY ts ASC",
        conn, params=(symbol, start_ms, end_ms)
    )
    conn.close()
    logger.info("Loaded %d rows from database", len(df))

    df["datetime"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df.set_index("datetime", inplace=True)
    df = df[["open", "high", "low", "close", "volume"]]
    df.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume',
    }, inplace=True)

    # fill gaps
    logger.info("Reindexing to full 1-min timeline and forward-filling gaps")
    full_idx = pd.date_range(start, end, freq="1min", tz="UTC")
    df = df.reindex(full_idx).ffill()

    # correct OHLC bounds
    logger.info("Correcting any OHLC inconsistencies")
    df["High"] = df[["High", "Open", "Close"]].max(axis=1)
    df["Low"]  = df[["Low", "Open", "Close"]].min(axis=1)

    logger.info("Fetch and cleanup complete; returning DataFrame")
    return df.iloc[:-1, :]

def fetch_ohlcv(symbol: str, start: str, end: str) -> pd.DataFrame:
    return asyncio.get_event_loop().run_until_complete(
        fetch_ohlcv_to_df(symbol, start, end)
    )

"""
Time-series database interface and TimescaleDB implementation.

This module provides a high-level interface for time-series data storage and retrieval,
with a concrete implementation for TimescaleDB.
"""

import logging
from abc import ABC, abstractmethod
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Type, TypeVar, Generic, Iterator, Union

import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_values, Json
import pandas as pd
import numpy as np

from .models import (
    AggregationType,
    MarketDepth,
    OHLCV,
    OrderBookSnapshot,
    TickDataPoint,
    TimeRange,
    Trade,
)

logger = logging.getLogger(__name__)
T = TypeVar('T')


class TimeSeriesQuery:
    """Builder for time-series queries."""
    
    def __init__(self):
        self._filters = {}
        self._time_range = None
        self._limit = None
        self._offset = 0
        self._order_by = None
        self._group_by = None
        self._aggregation = None
        self._select_fields = None
    
    def filter(self, **filters) -> 'TimeSeriesQuery':
        """Add equality filters to the query."""
        self._filters.update(filters)
        return self
    
    def time_range(self, start: datetime, end: datetime) -> 'TimeSeriesQuery':
        """Set the time range for the query."""
        self._time_range = (start, end)
        return self
    
    def limit(self, n: int) -> 'TimeSeriesQuery':
        """Limit the number of results."""
        self._limit = n
        return self
    
    def offset(self, n: int) -> 'TimeSeriesQuery':
        """Set the offset for pagination."""
        self._offset = n
        return self
    
    def order_by(self, field: str, desc: bool = False) -> 'TimeSeriesQuery':
        """Set the sort order."""
        self._order_by = (field, 'DESC' if desc else 'ASC')
        return self
    
    def group_by(self, *fields: str) -> 'TimeSeriesQuery':
        """Set the group by clause."""
        self._group_by = fields
        return self
    
    def aggregate(self, agg_type: AggregationType) -> 'TimeSeriesQuery':
        """Set the aggregation type."""
        self._aggregation = agg_type
        return self
    
    def select(self, *fields: str) -> 'TimeSeriesQuery':
        """Select specific fields to return."""
        self._select_fields = fields
        return self
    
    def build(self) -> tuple:
        """Build the SQL query and parameters."""
        query_parts = []
        params = []
        
        # SELECT clause
        if self._select_fields:
            fields = [sql.Identifier(f) for f in self._select_fields]
            query_parts.append(sql.SQL("SELECT ") + sql.SQL(", ").join(fields))
        else:
            query_parts.append(sql.SQL("SELECT *"))
        
        # FROM clause (handled by the model)
        
        # WHERE clause
        conditions = []
        
        # Time range
        if self._time_range:
            start, end = self._time_range
            conditions.append(sql.SQL("timestamp BETWEEN %s AND %s"))
            params.extend([start, end])
        
        # Other filters
        for field, value in self._filters.items():
            if value is not None:
                conditions.append(sql.Identifier(field) == sql.Placeholder())
                params.append(value)
        
        if conditions:
            query_parts.append(sql.SQL("WHERE ") + sql.SQL(" AND ").join(conditions))
        
        # GROUP BY
        if self._group_by:
            group_fields = [sql.Identifier(f) for f in self._group_by]
            query_parts.append(
                sql.SQL("GROUP BY ") + sql.SQL(", ").join(group_fields)
            )
        
        # ORDER BY
        if self._order_by:
            field, direction = self._order_by
            query_parts.append(
                sql.SQL("ORDER BY {field} {direction}").format(
                    field=sql.Identifier(field),
                    direction=sql.SQL(direction)
                )
            )
        
        # LIMIT and OFFSET
        if self._limit is not None:
            query_parts.append(sql.SQL("LIMIT %s"))
            params.append(self._limit)
            
            if self._offset > 0:
                query_parts.append(sql.SQL("OFFSET %s"))
                params.append(self._offset)
        
        return sql.SQL(" ").join(query_parts), params


class TimeSeriesDB(ABC):
    """Abstract base class for time-series database operations."""
    
    @abstractmethod
    def connect(self):
        """Establish a connection to the database."""
        pass
    
    @abstractmethod
    def disconnect(self):
        """Close the database connection."""
        pass
    
    @abstractmethod
    def initialize_schema(self):
        """Initialize the database schema."""
        pass
    
    @abstractmethod
    def insert_ticks(self, ticks: List[TickDataPoint]):
        """Insert multiple tick data points."""
        pass
    
    @abstractmethod
    def query_ticks(self, query: TimeSeriesQuery) -> List[TickDataPoint]:
        """Query tick data."""
        pass
    
    @abstractmethod
    def get_ohlcv(
        self,
        symbol: str,
        time_range: TimeRange,
        aggregation: AggregationType = AggregationType.MINUTE
    ) -> List[OHLCV]:
        """Get OHLCV data for a symbol and time range."""
        pass
    
    @abstractmethod
    def insert_order_book_snapshot(self, snapshot: OrderBookSnapshot):
        """Insert an order book snapshot."""
        pass
    
    @abstractmethod
    def get_order_book_history(
        self,
        symbol: str,
        time_range: TimeRange,
        limit: int = 100
    ) -> List[OrderBookSnapshot]:
        """Get historical order book snapshots."""
        pass
    
    @abstractmethod
    def insert_trades(self, trades: List[Trade]):
        """Insert multiple trade records."""
        pass
    
    @abstractmethod
    def get_trades(
        self,
        symbol: str,
        time_range: TimeRange,
        limit: int = 1000
    ) -> List[Trade]:
        """Get trade history."""
        pass


class TimescaleDB(TimeSeriesDB):
    """TimescaleDB implementation of TimeSeriesDB."""
    
    def __init__(self, dsn: str):
        """Initialize with a connection string."""
        self.dsn = dsn
        self.conn = None
    
    def connect(self):
        """Establish a connection to TimescaleDB."""
        if self.conn is None or self.conn.closed != 0:
            self.conn = psycopg2.connect(self.dsn)
            self.conn.autocommit = False
        return self.conn
    
    def disconnect(self):
        """Close the database connection."""
        if self.conn is not None:
            self.conn.close()
            self.conn = None
    
    @contextmanager
    def _get_cursor(self):
        """Context manager for database cursors."""
        conn = self.connect()
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            cursor.close()
    
    def initialize_schema(self):
        """Initialize the TimescaleDB schema and create hypertables."""
        with self._get_cursor() as cur:
            # Enable TimescaleDB extension
            cur.execute("""
                CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;
            """)
            
            # Create tick_data table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS tick_data (
                    time TIMESTAMPTZ NOT NULL,
                    symbol TEXT NOT NULL,
                    price DOUBLE PRECISION NOT NULL,
                    volume DOUBLE PRECISION NOT NULL,
                    exchange TEXT NOT NULL,
                    bid DOUBLE PRECISION,
                    ask DOUBLE PRECISION,
                    trade_id TEXT,
                    metadata JSONB,
                    
                    -- Indexes for common query patterns
                    CONSTRAINT tick_data_pkey PRIMARY KEY (time, symbol, exchange)
                );
                
                -- Create hypertable for time-series data
                SELECT create_hypertable('tick_data', 'time', 
                    if_not_exists => TRUE,
                    chunk_time_interval => INTERVAL '1 day');
                
                -- Indexes for common query patterns
                CREATE INDEX IF NOT EXISTS idx_tick_data_symbol_time 
                    ON tick_data (symbol, time DESC);
                CREATE INDEX IF NOT EXISTS idx_tick_data_exchange_symbol_time 
                    ON tick_data (exchange, symbol, time DESC);
                
                -- Enable compression for older data
                ALTER TABLE tick_data SET (
                    timescaledb.compress,
                    timescaledb.compress_segmentby = 'symbol,exchange'
                );
                
                -- Compression policy: compress chunks older than 7 days
                SELECT add_compression_policy('tick_data', INTERVAL '7 days');
            """)
            
            # Create order_books table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS order_books (
                    time TIMESTAMPTZ NOT NULL,
                    symbol TEXT NOT NULL,
                    exchange TEXT NOT NULL,
                    bids JSONB NOT NULL,
                    asks JSONB NOT NULL,
                    
                    CONSTRAINT order_books_pkey PRIMARY KEY (time, symbol, exchange)
                );
                
                -- Create hypertable for order book data
                SELECT create_hypertable('order_books', 'time', 
                    if_not_exists => TRUE,
                    chunk_time_interval => INTERVAL '1 day');
                
                -- Indexes for order book queries
                CREATE INDEX IF NOT EXISTS idx_order_books_symbol_time 
                    ON order_books (symbol, time DESC);
                
                -- Compression for order book data
                ALTER TABLE order_books SET (
                    timescaledb.compress,
                    timescaledb.compress_segmentby = 'symbol,exchange'
                );
            """)
            
            # Create trades table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    trade_id TEXT NOT NULL,
                    time TIMESTAMPTZ NOT NULL,
                    symbol TEXT NOT NULL,
                    price DOUBLE PRECISION NOT NULL,
                    quantity DOUBLE PRECISION NOT NULL,
                    side TEXT NOT NULL,
                    exchange TEXT NOT NULL,
                    taker_side TEXT,
                    trade_condition TEXT,
                    is_snapshot BOOLEAN DEFAULT FALSE,
                    
                    CONSTRAINT trades_pkey PRIMARY KEY (trade_id, exchange)
                );
                
                -- Create hypertable for trades
                SELECT create_hypertable('trades', 'time', 
                    if_not_exists => TRUE,
                    chunk_time_interval => INTERVAL '1 day',
                    create_default_indexes => FALSE);
                
                -- Indexes for trade queries
                CREATE INDEX IF NOT EXISTS idx_trades_symbol_time 
                    ON trades (symbol, time DESC);
                CREATE INDEX IF NOT EXISTS idx_trades_exchange_symbol_time 
                    ON trades (exchange, symbol, time DESC);
                
                -- Compression for trades
                ALTER TABLE trades SET (
                    timescaledb.compress,
                    timescaledb.compress_segmentby = 'symbol,exchange,side'
                );
            """)
            
            # Create continuous aggregates for OHLCV data
            cur.execute("""
                -- Materialized view for 1-minute OHLCV
                CREATE MATERIALIZED VIEW IF NOT EXISTS ohlcv_1m
                WITH (timescaledb.continuous) AS
                SELECT 
                    time_bucket('1 minute', time) AS bucket,
                    symbol,
                    exchange,
                    first(price, time) AS open,
                    max(price) AS high,
                    min(price) AS low,
                    last(price, time) AS close,
                    sum(volume) AS volume,
                    sum(price * volume) / NULLIF(sum(volume), 0) AS vwap,
                    count(*) AS trade_count
                FROM tick_data
                GROUP BY bucket, symbol, exchange
                WITH NO DATA;
                
                -- Refresh policy for 1-minute OHLCV
                SELECT add_continuous_aggregate_policy('ohlcv_1m',
                    start_offset => INTERVAL '1 hour',
                    end_offset => INTERVAL '1 minute',
                    schedule_interval => INTERVAL '1 minute');
                
                -- Materialized view for 1-hour OHLCV (aggregated from 1m)
                CREATE MATERIALIZED VIEW IF NOT EXISTS ohlcv_1h
                WITH (timescaledb.continuous) AS
                SELECT 
                    time_bucket('1 hour', bucket) AS bucket,
                    symbol,
                    exchange,
                    first(open, bucket) AS open,
                    max(high) AS high,
                    min(low) AS low,
                    last(close, bucket) AS close,
                    sum(volume) AS volume,
                    sum(vwap * volume) / NULLIF(sum(volume), 0) AS vwap,
                    sum(trade_count) AS trade_count
                FROM ohlcv_1m
                GROUP BY bucket, symbol, exchange
                WITH NO DATA;
                
                -- Refresh policy for 1-hour OHLCV
                SELECT add_continuous_aggregate_policy('ohlcv_1h',
                    start_offset => INTERVAL '1 day',
                    end_offset => INTERVAL '1 hour',
                    schedule_interval => INTERVAL '1 hour');
            """)
    
    def insert_ticks(self, ticks: List[TickDataPoint]):
        """Insert multiple tick data points."""
        if not ticks:
            return
            
        with self._get_cursor() as cur:
            # Prepare the data for batch insertion
            data = [(
                tick.timestamp,
                tick.symbol,
                tick.price,
                tick.volume,
                tick.exchange,
                tick.bid,
                tick.ask,
                tick.trade_id,
                Json(tick.metadata) if tick.metadata else None
            ) for tick in ticks]
            
            # Use execute_values for efficient batch insertion
            execute_values(
                cur,
                """
                INSERT INTO tick_data 
                    (time, symbol, price, volume, exchange, bid, ask, trade_id, metadata)
                VALUES %s
                ON CONFLICT (time, symbol, exchange) DO UPDATE SET
                    price = EXCLUDED.price,
                    volume = EXCLUDED.volume,
                    bid = EXCLUDED.bid,
                    ask = EXCLUDED.ask,
                    trade_id = EXCLUDED.trade_id,
                    metadata = EXCLUDED.metadata
                """,
                data,
                template="(%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                page_size=1000
            )
    
    def query_ticks(self, query: TimeSeriesQuery) -> List[TickDataPoint]:
        """Query tick data."""
        query_str, params = query.build()
        query_str = sql.SQL("SELECT * FROM tick_data ").join([query_str])
        
        with self._get_cursor() as cur:
            cur.execute(query_str, params)
            columns = [desc[0] for desc in cur.description]
            
            results = []
            for row in cur.fetchall():
                row_dict = dict(zip(columns, row))
                # Convert JSONB to dict
                if 'metadata' in row_dict and row_dict['metadata'] is not None:
                    row_dict['metadata'] = dict(row_dict['metadata'])
                results.append(TickDataPoint(**row_dict))
            
            return results
    
    def get_ohlcv(
        self,
        symbol: str,
        time_range: TimeRange,
        aggregation: AggregationType = AggregationType.MINUTE
    ) -> List[OHLCV]:
        """Get OHLCV data for a symbol and time range."""
        # Map aggregation to the appropriate materialized view
        view_map = {
            AggregationType.MINUTE: 'ohlcv_1m',
            AggregationType.FIVE_MINUTES: 'ohlcv_5m',
            AggregationType.FIFTEEN_MINUTES: 'ohlcv_15m',
            AggregationType.HOUR: 'ohlcv_1h',
            AggregationType.DAY: 'ohlcv_1d'
        }
        
        view_name = view_map.get(aggregation, 'ohlcv_1m')
        
        with self._get_cursor() as cur:
            # First, refresh the materialized view to ensure we have the latest data
            cur.execute(
                f"CALL refresh_continuous_aggregate('{view_name}', %s, %s)",
                (time_range.start_time, time_range.end_time)
            )
            
            # Then query the data
            cur.execute("""
                SELECT 
                    bucket as timestamp,
                    symbol,
                    open,
                    high,
                    low,
                    close,
                    volume,
                    vwap,
                    trade_count,
                    exchange
                FROM {view}
                WHERE symbol = %s 
                AND bucket BETWEEN %s AND %s
                ORDER BY bucket ASC
            """.format(view=sql.Identifier(view_name)),
                (symbol, time_range.start_time, time_range.end_time)
            )
            
            columns = [desc[0] for desc in cur.description]
            return [
                OHLCV(**dict(zip(columns, row)))
                for row in cur.fetchall()
            ]
    
    def insert_order_book_snapshot(self, snapshot: OrderBookSnapshot):
        """Insert an order book snapshot."""
        with self._get_cursor() as cur:
            cur.execute("""
                INSERT INTO order_books (time, symbol, exchange, bids, asks)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (time, symbol, exchange) DO UPDATE SET
                    bids = EXCLUDED.bids,
                    asks = EXCLUDED.asks
            """, (
                snapshot.timestamp,
                snapshot.symbol,
                snapshot.exchange,
                Json([{str(k): v for k, v in bid.items()} for bid in snapshot.bids]),
                Json([{str(k): v for k, v in ask.items()} for ask in snapshot.asks])
            ))
    
    def get_order_book_history(
        self,
        symbol: str,
        time_range: TimeRange,
        limit: int = 100
    ) -> List[OrderBookSnapshot]:
        """Get historical order book snapshots."""
        with self._get_cursor() as cur:
            cur.execute("""
                SELECT time, symbol, exchange, bids, asks
                FROM order_books
                WHERE symbol = %s
                AND time BETWEEN %s AND %s
                ORDER BY time DESC
                LIMIT %s
            """, (symbol, time_range.start_time, time_range.end_time, limit))
            
            results = []
            for row in cur.fetchall():
                time, symbol, exchange, bids, asks = row
                results.append(OrderBookSnapshot(
                    timestamp=time,
                    symbol=symbol,
                    exchange=exchange,
                    bids=[{float(k): v for item in bids for k, v in item.items()}],
                    asks=[{float(k): v for item in asks for k, v in item.items()}]
                ))
            return results
    
    def insert_trades(self, trades: List[Trade]):
        """Insert multiple trade records."""
        if not trades:
            return
            
        with self._get_cursor() as cur:
            # Prepare the data for batch insertion
            data = [(
                trade.trade_id,
                trade.timestamp,
                trade.symbol,
                trade.price,
                trade.quantity,
                trade.side,
                trade.exchange,
                trade.taker_side,
                trade.trade_condition,
                trade.is_snapshot
            ) for trade in trades]
            
            # Use execute_values for efficient batch insertion
            execute_values(
                cur,
                """
                INSERT INTO trades 
                    (trade_id, time, symbol, price, quantity, side, exchange, 
                     taker_side, trade_condition, is_snapshot)
                VALUES %s
                ON CONFLICT (trade_id, exchange) DO UPDATE SET
                    time = EXCLUDED.time,
                    price = EXCLUDED.price,
                    quantity = EXCLUDED.quantity,
                    side = EXCLUDED.side,
                    taker_side = EXCLUDED.taker_side,
                    trade_condition = EXCLUDED.trade_condition,
                    is_snapshot = EXCLUDED.is_snapshot
                """,
                data,
                template="(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                page_size=1000
            )
    
    def get_trades(
        self,
        symbol: str,
        time_range: TimeRange,
        limit: int = 1000
    ) -> List[Trade]:
        """Get trade history."""
        with self._get_cursor() as cur:
            cur.execute("""
                SELECT 
                    trade_id, time, symbol, price, quantity, 
                    side, exchange, taker_side, trade_condition, is_snapshot
                FROM trades
                WHERE symbol = %s
                AND time BETWEEN %s AND %s
                ORDER BY time DESC
                LIMIT %s
            """, (symbol, time_range.start_time, time_range.end_time, limit))
            
            columns = [desc[0] for desc in cur.description]
            return [
                Trade(**dict(zip(columns, row)))
                for row in cur.fetchall()
            ]

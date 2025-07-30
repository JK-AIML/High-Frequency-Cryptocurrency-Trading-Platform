"""Migration 1: Initial database schema"""

def upgrade(cursor):
    """Run the migration."""
    # Create tick_data table
    cursor.execute("""
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
        
        CONSTRAINT tick_data_pkey PRIMARY KEY (time, symbol, exchange)
    )
    """)
    
    # Create hypertable for tick data
    cursor.execute("""
    SELECT create_hypertable(
        'tick_data', 
        'time', 
        if_not_exists => TRUE,
        chunk_time_interval => INTERVAL '1 day'
    )
    """)
    
    # Create indexes
    cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_tick_data_symbol_time 
        ON tick_data (symbol, time DESC)
    """)
    
    cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_tick_data_exchange_symbol_time 
        ON tick_data (exchange, symbol, time DESC)
    """)
    
    # Create order_books table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS order_books (
        time TIMESTAMPTZ NOT NULL,
        symbol TEXT NOT NULL,
        exchange TEXT NOT NULL,
        bids JSONB NOT NULL,
        asks JSONB NOT NULL,
        
        CONSTRAINT order_books_pkey PRIMARY KEY (time, symbol, exchange)
    )
    """)
    
    # Create hypertable for order books
    cursor.execute("""
    SELECT create_hypertable(
        'order_books', 
        'time', 
        if_not_exists => TRUE,
        chunk_time_interval => INTERVAL '1 day'
    )
    """)
    
    # Create trades table
    cursor.execute("""
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
    )
    """)
    
    # Create hypertable for trades
    cursor.execute("""
    SELECT create_hypertable(
        'trades', 
        'time', 
        if_not_exists => TRUE,
        chunk_time_interval => INTERVAL '1 day'
    )
    """)
    
    # Create indexes for trades
    cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_trades_symbol_time 
        ON trades (symbol, time DESC)
    """)
    
    cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_trades_exchange_symbol_time 
        ON trades (exchange, symbol, time DESC)
    """)
    
    # Create materialized view for 1-minute OHLCV
    cursor.execute("""
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
    WITH NO DATA
    """)
    
    # Add continuous aggregate policy
    cursor.execute("""
    SELECT add_continuous_aggregate_policy(
        'ohlcv_1m',
        start_offset => INTERVAL '1 hour',
        end_offset => INTERVAL '1 minute',
        schedule_interval => INTERVAL '1 minute'
    )
    """)
    
    # Create materialized view for 1-hour OHLCV
    cursor.execute("""
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
    WITH NO DATA
    """)
    
    # Add continuous aggregate policy for hourly data
    cursor.execute("""
    SELECT add_continuous_aggregate_policy(
        'ohlcv_1h',
        start_offset => INTERVAL '1 day',
        end_offset => INTERVAL '1 hour',
        schedule_interval => INTERVAL '1 hour'
    )
    """)
    
    # Enable compression for all hypertables
    for table in ['tick_data', 'order_books', 'trades']:
        cursor.execute(f"""
        ALTER TABLE {table} SET (
            timescaledb.compress,
            timescaledb.compress_segmentby = 'symbol,exchange'
        )
        """)
        
        # Add compression policy (compress chunks older than 7 days)
        cursor.execute(f"""
        SELECT add_compression_policy(
            '{table}',
            compress_after => INTERVAL '7 days',
            if_not_exists => TRUE
        )
        """)

# Mark this function as a migration
upgrade.__migration__ = True

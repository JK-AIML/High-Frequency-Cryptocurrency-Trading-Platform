# Time-Series Database Module

This module provides a high-performance time-series data storage and retrieval system 
built on top of TimescaleDB, a PostgreSQL extension optimized for time-series data.

## Features

- **Efficient Time-Series Storage**: Optimized for high-frequency tick data storage
- **Schema Management**: Automatic schema creation and migration system
- **Connection Pooling**: Efficient database connection management
- **Data Models**: Strongly-typed data models for tick data, order books, and trades
- **Query Builder**: Fluent interface for building complex time-series queries
- **Migrations**: Versioned database migrations

## Installation

1. Install TimescaleDB:
   ```bash
   # On Ubuntu/Debian
   sudo apt-get install -y timescaledb-2-postgresql-14
   
   # Or using Docker
   docker run -d --name timescaledb -p 5432:5432 -e POSTGRES_PASSWORD=password timescale/timescaledb:latest-pg14
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

## Configuration

Create a `.env` file in your project root:

```env
# Database Configuration
DB_TIMESCALEDB_HOST=localhost
DB_TIMESCALEDB_PORT=5432
DB_TIMESCALEDB_USER=postgres
DB_TIMESCALEDB_PASSWORD=yourpassword
DB_TIMESCALEDB_DB=tickdata
DB_TIMESCALEDB_SCHEMA=public
DB_TIMESCALEDB_SSLMODE=prefer
```

## Usage

### Initialize the Database

```python
from tick_analysis.storage import TimeSeriesDB, TimescaleDB
from tick_analysis.config.db import get_db_config

# Initialize with default configuration
db = TimescaleDB(get_db_config().timescaledb_dsn)
db.initialize_schema()
```

### Store Tick Data

```python
from datetime import datetime
from tick_analysis.storage.models import TickDataPoint

# Create some tick data
ticks = [
    TickDataPoint(
        timestamp=datetime.utcnow(),
        symbol="BTC-USD",
        price=50000.0,
        volume=1.0,
        exchange="binance",
        bid=49999.0,
        ask=50001.0
    )
]

# Store the ticks
db.insert_ticks(ticks)
```

### Query Data

```python
from datetime import datetime, timedelta
from tick_analysis.storage import TimeRange

# Query ticks from the last hour
end_time = datetime.utcnow()
start_time = end_time - timedelta(hours=1)
time_range = TimeRange(start_time=start_time, end_time=end_time)

ticks = db.query_ticks(
    symbol="BTC-USD",
    time_range=time_range,
    limit=1000
)
```

### Run Migrations

```bash
# Create a new migration
python -m tick_analysis.cli.db new_migration add_order_books_table

# Run pending migrations
python -m tick_analysis.cli.db migrate

# Show migration status
python -m tick_analysis.cli.db status
```

## CLI Reference

### Database Management

```bash
# Initialize database
python -m tick_analysis.cli.db init

# Create a new migration
python -m tick_analysis.cli.db new_migration <name>

# Run migrations
python -m tick_analysis.cli.db migrate

# Show migration status
python -m tick_analysis.cli.db status

# Reset database (DANGER: drops all data)
python -m tick_analysis.cli.db reset

# Execute raw SQL query
python -m tick_analysis.cli.db query "SELECT * FROM tick_data LIMIT 10"
```

## Performance Tuning

### Indexing Strategy

- Time-based partitioning with hypertables
- Composite indexes on (symbol, time) for common query patterns
- Separate indexes for exchange-based queries

### Compression

- Time-series compression for older data
- Segment by symbol and exchange for optimal compression
- Automatic compression policies

### Retention Policies

```sql
-- Example: Drop data older than 1 year
SELECT add_retention_policy('tick_data', INTERVAL '1 year');
```

## Troubleshooting

### Common Issues

1. **Connection Issues**
   - Verify TimescaleDB is running
   - Check connection parameters in `.env`
   - Ensure the database and user exist

2. **Performance Problems**
   - Check for missing indexes with `EXPLAIN ANALYZE`
   - Monitor query performance with `pg_stat_statements`
   - Adjust chunk time intervals if needed

3. **Migration Failures**
   - Check the migration logs
   - Ensure no other processes are modifying the schema
   - Verify database user has sufficient permissions

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

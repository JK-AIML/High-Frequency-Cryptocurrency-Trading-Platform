"""
Flink Stream Processor for Real-time Data Processing

This module provides a high-level interface for processing financial market data streams
using Apache Flink. It includes features for windowed aggregations, state management,
and integration with drift detection for monitoring data quality.
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional, Union
import json
import logging
import time
import uuid
from dataclasses import dataclass, asdict, field
from enum import Enum

from pyflink.datastream import (
    StreamExecutionEnvironment,
    CheckpointingMode
)
from pyflink.table import (
    StreamTableEnvironment,
    DataTypes,
    expressions as F,
    TableDescriptor,
    Schema
)
from pyflink.table.udf import udf, udtf, ScalarFunction, TableFunction
from pyflink.table.window import Tumble, Session, Slide
from pyflink.common.serialization import SimpleStringSchema
from pyflink.common.typeinfo import Types
from pyflink.datastream.connectors.kafka import (
    FlinkKafkaConsumer,
    FlinkKafkaProducer
)
# Import drift detection
from tick_analysis.monitoring.unified_drift import UnifiedDriftDetector

logger = logging.getLogger(__name__)

class ProcessingMode(Enum):
    """Processing modes for the stream processor."""
    STREAMING = "streaming"
    BATCH = "batch"

@dataclass
class WindowConfig:
    """Configuration for windowed operations."""
    window_size: int = 60  # seconds
    slide_size: int = 10   # seconds
    window_type: str = 'tumbling'  # 'tumbling', 'sliding', or 'session'
    session_gap: int = 30  # seconds, for session windows
    
    def get_window(self, time_col: str):
        """Get the appropriate window type based on configuration."""
        if self.window_type == 'tumbling':
            return Tumble.over(F.lit(self.window_size).seconds).on(F.col(time_col)).alias('w')
        elif self.window_type == 'sliding':
            return Slide.over(F.lit(self.window_size).seconds) \
                       .every(F.lit(self.slide_size).seconds) \
                       .on(F.col(time_col)).alias('w')
        elif self.window_type == 'session':
            return Session.with_gap(F.lit(self.session_gap).seconds) \
                         .on(F.col(time_col)).alias('w')
        else:
            raise ValueError(f"Unsupported window type: {self.window_type}")

class FlinkStreamProcessor:
    """
    A Flink-based stream processor for real-time financial data analysis.
    
    Features:
    - Windowed aggregations (tumbling, sliding, session)
    - State management with checkpoints
    - Exactly-once processing semantics
    - Integration with drift detection
    - Custom UDF support
    """
    
    def __init__(self, config: Dict[str, Any], processing_mode: ProcessingMode = ProcessingMode.STREAMING):
        """
        Initialize the Flink stream processor.
        
        Args:
            config: Configuration dictionary with required parameters
            processing_mode: Processing mode (streaming or batch)
        """
        self.config = config
        self.processing_mode = processing_mode
        self._setup_environment()
        self._register_udfs()
        self._initialize_drift_detector()
    
    def _setup_environment(self):
        """Set up the Flink execution environment."""
        try:
            # Set up the execution environment
            self.env = StreamExecutionEnvironment.get_execution_environment()
            self.table_env = StreamTableEnvironment.create(self.env)
            
            # Set processing time characteristic (TimeCharacteristic removed in newer PyFlink versions)
            # self.env.set_stream_time_characteristic(TimeCharacteristic.EventTime)
            
            # Configure checkpointing for fault tolerance
            self.env.enable_checkpointing(10000)  # Checkpoint every 10 seconds
            checkpoint_config = self.env.get_checkpoint_config()
            checkpoint_config.set_checkpointing_mode(CheckpointingMode.EXACTLY_ONCE)
            checkpoint_config.set_min_pause_between_checkpoints(5000)
            checkpoint_config.set_checkpoint_timeout(60000)
            checkpoint_config.set_max_concurrent_checkpoints(1)
            
            # Add required JARs
            self._add_required_jars()
            
            # Set parallelism
            if 'parallelism' in self.config.get('flink', {}):
                self.env.set_parallelism(self.config['flink']['parallelism'])
        except Exception as e:
            logger.error(f"Error setting up Flink environment: {e}")
            raise
    
    def _add_required_jars(self):
        """Add required Flink connector JARs."""
        # This should be configured based on your Flink setup
        jars = [
            "flink-connector-kafka_2.12-1.15.0.jar",
            "flink-sql-connector-kafka_2.12-1.15.0.jar",
            "flink-connector-jdbc_2.12-1.15.0.jar",
            "flink-table-planner-blink_2.12-1.15.0.jar"
        ]
        
        for jar in jars:
            try:
                self.env.add_jars(f"file:///path/to/flink/lib/{jar}")
            except Exception as e:
                logger.warning(f"Failed to add JAR {jar}: {e}")
    
    def _register_udfs(self):
        """Register custom UDFs."""
        # Example UDF for calculating VWAP
        @udf(result_type=DataTypes.DOUBLE())
        def calculate_vwap(prices, volumes):
            if not prices or not volumes or len(prices) != len(volumes):
                return None
            try:
                total_volume = sum(volumes)
                if total_volume <= 0:
                    return None
                return sum(p * v for p, v in zip(prices, volumes)) / total_volume
            except Exception as e:
                logger.error(f"Error in VWAP calculation: {e}")
                return None
        
        self.table_env.create_temporary_function("calculate_vwap", calculate_vwap)
    
    def _initialize_drift_detector(self):
        """Initialize the drift detector if configured."""
        self.drift_detector = None
        if self.config.get('enable_drift_detection', False):
            # Initialize with reference data if provided
            ref_data = self.config.get('drift_detection', {}).get('reference_data')
            if ref_data:
                self.drift_detector = UnifiedDriftDetector(
                    reference_data=ref_data,
                    **self.config.get('drift_detection', {}).get('params', {})
                )
                logger.info("Drift detector initialized with reference data")
            else:
                logger.warning("Drift detection enabled but no reference data provided")
    
    def create_kafka_source(self, topic: str, group_id: str = None, schema: Optional[Dict] = None):
        """
        Create a Kafka source table with the given schema.
        
        Args:
            topic: Kafka topic to consume from
            group_id: Consumer group ID
            schema: Dictionary defining the table schema
            
        Returns:
            SQL string to create the source table
        """
        if schema is None:
            schema = {
                'event_time': 'TIMESTAMP(3) METADATA FROM \'timestamp\' ROW METADATA VIRTUAL',
                'symbol': 'STRING',
                'price': 'DOUBLE',
                'volume': 'DOUBLE',
                'exchange': 'STRING',
                'timestamp': 'BIGINT',
                'WATERMARK': 'FOR event_time AS event_time - INTERVAL \'5\' SECOND'
            }
        
        columns = []
        watermark = None
        
        for col, col_type in schema.items():
            if col.upper() == 'WATERMARK':
                watermark = col_type
            else:
                columns.append(f'`{col}` {col_type}')
        
        sql = f"""
            CREATE TABLE {topic}_source (
                {', '.join(columns)},
                {watermark if watermark else ''}
            ) WITH (
                'connector' = 'kafka',
                'topic' = '{topic}',
                'properties.bootstrap.servers' = '{bootstrap_servers}',
                'properties.group.id' = '{group_id or self.config['kafka'].get('group_id', f'flink-{topic}-consumer')}',
                'format' = 'json',
                'json.fail-on-missing-field' = 'false',
                'json.ignore-parse-errors' = 'true',
                'scan.startup.mode' = '{self.config['kafka'].get('startup_mode', 'latest-offset')}',
                'properties.auto.offset.reset' = '{self.config['kafka'].get('auto_offset_reset', 'latest')}'
            )
        """.format(
            bootstrap_servers=self.config['kafka']['bootstrap_servers'],
            group_id=group_id or self.config['kafka'].get('group_id', f'flink-{topic}-consumer'),
            **self.config['kafka']
        )
        
        return sql
    
    def create_kafka_sink(self, topic: str, schema: Optional[Dict] = None, primary_key: Optional[List[str]] = None):
        """
        Create a Kafka sink table with the given schema.
        
        Args:
            topic: Kafka topic to write to
            schema: Dictionary defining the table schema
            primary_key: List of columns that form the primary key
            
        Returns:
            SQL string to create the sink table
        """
        if schema is None:
            schema = {
                'symbol': 'STRING',
                'event_time': 'TIMESTAMP(3)',
                'feature_name': 'STRING',
                'feature_value': 'DOUBLE',
                'metadata': 'STRING'
            }
            primary_key = primary_key or ['symbol', 'event_time', 'feature_name']
        
        columns = [f'`{col}` {col_type}' for col, col_type in schema.items()]
        pk_constraint = f",\n    PRIMARY KEY ({', '.join(primary_key)}) NOT ENFORCED" if primary_key else ""
        
        return f"""
            CREATE TABLE {topic}_sink (
                {', '.join(columns)}{pk_constraint}
            ) WITH (
                'connector' = 'upsert-kafka',
                'topic' = '{topic}',
                'properties.bootstrap.servers' = '{self.config['kafka']['bootstrap_servers']}',
                'key.format' = 'json',
                'value.format' = 'json',
                'sink.parallelism' = '{self.config['kafka'].get('sink_parallelism', 3)}',
                'sink.buffer-flush.interval' = '{self.config['kafka'].get('sink_flush_interval', '1s')}',
                'sink.buffer-flush.max-rows' = '{self.config['kafka'].get('sink_buffer_max_rows', 1000)}',
                'sink.delivery-guarantee' = '{self.config['kafka'].get('delivery_guarantee', 'exactly-once')}'
            )
        """
    
    def process_tick_data(self, source_topic: str = "market_data", sink_topic: str = "processed_features",
                         window_config: Optional[WindowConfig] = None):
        """
        Process tick data stream with windowed aggregations and feature extraction.
        
        Args:
            source_topic: Kafka topic to consume market data from
            sink_topic: Kafka topic to write processed features to
            window_config: Configuration for windowed operations
        """
        if window_config is None:
            window_config = WindowConfig()
        
        # Create source and sink tables
        self.table_env.execute_sql(self.create_kafka_source(source_topic))
        self.table_env.execute_sql(self.create_kafka_sink(sink_topic))
        
        # Get table references
        source_table = self.table_env.from_path(f"{source_topic}_source")
        
        # Define windowed aggregations
        window = window_config.get_window("event_time")
        
        # Process data with windowed aggregations
        result = source_table.window(window) \
            .group_by(
                F.col('symbol'),
                F.col('w')
            ) \
            .select(
                F.col('symbol'),
                F.col('w').end.alias('window_end'),
                F.col('price').avg.alias('avg_price'),
                F.col('volume').sum.alias('total_volume'),
                (F.col('price') * F.col('volume')).sum.cast(DataTypes.DOUBLE()) / 
                F.func('nullif', F.col('volume').sum, 0).cast(DataTypes.DOUBLE()).alias('vwap'),
                F.col('price').stddev_pop.alias('price_volatility'),
                F.col('price').max.alias('high'),
                F.col('price').min.alias('low')
            )
        
        # Add drift detection if enabled
        if self.drift_detector:
            # Convert to Pandas DataFrame for drift detection
            drift_result = result.to_pandas()
            
            # Check for drift on numerical features
            drift_features = ['avg_price', 'price_volatility', 'total_volume']
            drift_scores = self.drift_detector.detect_drift(
                current_data=drift_result[drift_features],
                drift_type='covariate',
                method='ks_test'
            )
            
            # Add drift scores to result
            for i, feature in enumerate(drift_features):
                result = result.add_columns(
                    F.lit(drift_scores[i].statistic).alias(f'{feature}_drift_score'),
                    F.lit(drift_scores[i].p_value).alias(f'{feature}_p_value'),
                    F.lit(drift_scores[i].is_drifted).alias(f'{feature}_drifted')
                )
        
        # Execute the query
        result.execute_insert(f"{sink_topic}_sink").wait()
        
        # Start the Flink job
        self.env.execute("tick_data_processing")
    
    def process_with_sql(self, sql_query: str, source_tables: Dict[str, str] = None):
        """
        Process data using a custom SQL query.
        
        Args:
            sql_query: SQL query to execute
            source_tables: Dictionary of {table_name: topic} for source tables
        """
        if source_tables:
            for table_name, topic in source_tables.items():
                self.table_env.execute_sql(self.create_kafka_source(topic, table_name))
        
        # Execute the query
        result = self.table_env.sql_query(sql_query)
        return result
    
    def process_with_python_udf(self, process_func, input_schema, output_schema, 
                              source_topic: str, sink_topic: str):
        """
        Process data using a Python UDF.
        
        Args:
            process_func: Function to process the data
            input_schema: Schema of the input data
            output_schema: Schema of the output data
            source_topic: Source Kafka topic
            sink_topic: Sink Kafka topic
        """
        # Register the UDF
        self.table_env.create_temporary_function("process_data", udf(process_func, output_schema, input_schema))
        
        # Create source and sink tables
        self.table_env.execute_sql(self.create_kafka_source(source_topic))
        self.table_env.execute_sql(self.create_kafka_sink(sink_topic))
        
        # Process data
        source_table = self.table_env.from_path(f"{source_topic}_source")
        result = source_table.select("process_data(*)")
        
        # Write to sink
        result.execute_insert(f"{sink_topic}_sink")
        
        # Start the Flink job
        self.env.execute("python_udf_processing")
    
    def run(self):
        """Run the Flink job."""
        try:
            logger.info("Starting Flink job")
            self.env.execute("tick_data_processing")
        except Exception as e:
            logger.error(f"Error in Flink job: {e}")
            raise


def create_default_config() -> Dict[str, Any]:
    """Create a default configuration dictionary."""
    return {
        'flink': {
            'parallelism': 4,
            'checkpoint_interval': 10000,  # ms
            'checkpoint_timeout': 60000,   # ms
        },
        'kafka': {
            'bootstrap_servers': 'localhost:9092',
            'group_id': 'flink-consumer',
            'auto_offset_reset': 'latest',
            'enable_auto_commit': True,
            'sink_parallelism': 3,
            'sink_flush_interval': '1s',
            'sink_buffer_max_rows': 1000,
            'delivery_guarantee': 'exactly-once'
        },
        'enable_drift_detection': False,
        'drift_detection': {
            'reference_data': None,  # Should be a pandas DataFrame
            'params': {
                'alpha': 0.05,
                'n_significant': 1,
                'correction': 'bonferroni',
                'n_jobs': -1
            }
        }
    }

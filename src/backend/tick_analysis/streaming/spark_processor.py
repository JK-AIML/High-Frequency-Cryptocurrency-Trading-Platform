"""
Spark Stream Processor for Real-time Data Processing

This module provides a high-level interface for processing financial market data streams
using Apache Spark Structured Streaming. It includes features for windowed aggregations,
state management, and integration with drift detection for monitoring data quality.
"""

from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable, Union, Tuple
import json
import logging
from dataclasses import dataclass, asdict, field
from enum import Enum

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark.sql.streaming import DataStreamWriter, StreamingQuery
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation

# Import drift detection
from tick_analysis.monitoring.unified_drift import UnifiedDriftDetector, DriftResult, DriftType, DetectionMethod

logger = logging.getLogger(__name__)

class ProcessingMode(Enum):
    """Processing modes for the stream processor."""
    STREAMING = "streaming"
    BATCH = "batch"

@dataclass
class WindowConfig:
    """Configuration for windowed operations."""
    window_duration: str = "1 minute"  # e.g., "5 minutes", "1 hour"
    slide_duration: Optional[str] = None  # e.g., "1 minute" for sliding windows
    watermark_delay: str = "5 minutes"  # watermark delay threshold
    
    def __post_init__(self):
        if self.slide_duration is None:
            self.slide_duration = self.window_duration

class SparkStreamProcessor:
    """
    A Spark Structured Streaming processor for real-time financial data analysis.
    
    Features:
    - Windowed aggregations with watermarking
    - Stateful processing with checkpointing
    - Exactly-once processing semantics
    - Integration with drift detection
    - Custom UDF support
    - Multiple sink types (Kafka, Iceberg, Feature Store)
    """
    
    def __init__(self, config: Dict[str, Any], app_name: str = "TickDataProcessor"):
        """
        Initialize the Spark stream processor.
        
        Args:
            config: Configuration dictionary with required parameters
            app_name: Name for the Spark application
        """
        self.config = config
        self.app_name = app_name
        self.spark = self._create_spark_session()
        self.checkpoint_location = config.get("checkpoint_location", "/tmp/checkpoints")
        self.drift_detector = None
        self._initialize_drift_detector()
    
    def _create_spark_session(self) -> SparkSession:
        """Create and configure a Spark session with appropriate configurations."""
        builder = SparkSession.builder \
            .appName(self.app_name) \
            .config("spark.sql.shuffle.partitions", 
                   self.config.get("spark", {}).get("shuffle_partitions", "8")) \
            .config("spark.sql.streaming.checkpointLocation", self.checkpoint_location) \
            .config("spark.sql.streaming.statefulOperator.checkCorrectness.enabled", "true")
        
        # Add Iceberg configuration if enabled
        if self.config.get("iceberg", {}).get("enabled", False):
            builder = builder \
                .config("spark.sql.catalog.spark_catalog", "org.apache.iceberg.spark.SparkSessionCatalog") \
                .config("spark.sql.catalog.spark_catalog.type", "hive") \
                .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions")
        
        # Add Kafka package
        builder = builder.config("spark.jars.packages", 
            "org.apache.spark:spark-sql-kafka-0-10_2.12:3.2.0,"
            "org.apache.iceberg:iceberg-spark3-runtime:0.13.2"
        )
        
        return builder.getOrCreate()
    
    def _initialize_drift_detector(self):
        """Initialize the drift detector if configured."""
        if self.config.get('enable_drift_detection', False):
            # Initialize with reference data if provided
            ref_data = self.config.get('drift_detection', {}).get('reference_data')
            if ref_data is not None:
                self.drift_detector = UnifiedDriftDetector(
                    reference_data=ref_data,
                    **self.config.get('drift_detection', {}).get('params', {})
                )
                logger.info("Drift detector initialized with reference data")
            else:
                logger.warning("Drift detection enabled but no reference data provided")
    
    def read_from_kafka(self, topics: Union[str, List[str]], 
                       starting_offsets: str = "latest",
                       include_headers: bool = False) -> DataFrame:
        """
        Read stream from Kafka.
        
        Args:
            topics: Kafka topic(s) to subscribe to
            starting_offsets: Starting offset ("earliest", "latest", or JSON string)
            include_headers: Whether to include Kafka headers in the output
            
        Returns:
            DataFrame with Kafka message data
        """
        read_stream = self.spark.readStream.format("kafka") \
            .option("kafka.bootstrap.servers", self.config["kafka"]["bootstrap_servers"]) \
            .option("subscribe", ",".join(topics) if isinstance(topics, list) else topics) \
            .option("startingOffsets", starting_offsets) \
            .option("failOnDataLoss", "false")
        
        # Add headers if requested
        if include_headers:
            read_stream = read_stream.option("includeHeaders", "true")
            
        return read_stream.load()
    
    def write_to_kafka(self, 
                      df: DataFrame, 
                      topic: str, 
                      output_mode: str = "update",
                      trigger: str = "10 seconds",
                      query_name: str = None) -> StreamingQuery:
        """
        Write stream to Kafka.
        
        Args:
            df: Input DataFrame to write
            topic: Kafka topic to write to
            output_mode: Output mode ("append", "complete", "update")
            trigger: Processing time trigger interval
            query_name: Name for the streaming query (for monitoring)
            
        Returns:
            StreamingQuery object for managing the streaming query
        """
        writer = df.writeStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", self.config["kafka"]["bootstrap_servers"]) \
            .option("topic", topic) \
            .option("checkpointLocation", f"{self.checkpoint_location}/{query_name or 'kafka_sink'}") \
            .outputMode(output_mode) \
            .trigger(processingTime=trigger)
            
        if query_name:
            writer = writer.queryName(query_name)
            
        return writer.start()
    
    def process_tick_data(self, 
                         input_topic: str = "market_data",
                         window_config: Optional[WindowConfig] = None) -> DataFrame:
        """
        Process tick data stream with windowed aggregations and feature extraction.
        
        Args:
            input_topic: Kafka topic to read market data from
            window_config: Configuration for windowed operations
            
        Returns:
            DataFrame with processed features
        """
        if window_config is None:
            window_config = WindowConfig()
            
        # Read from Kafka
        kafka_df = self.read_from_kafka(input_topic)
        
        # Define schema for parsing JSON messages
        schema = StructType([
            StructField("symbol", StringType()),
            StructField("price", DoubleType()),
            StructField("volume", DoubleType()),
            StructField("exchange", StringType()),
            StructField("timestamp", TimestampType()),
            StructField("trade_id", StringType()),
            StructField("side", StringType()),
            StructField("conditions", ArrayType(StringType()))
        ])
        
        # Parse JSON and add processing timestamp
        parsed_df = kafka_df \
            .select(
                F.from_json(
                    F.col("value").cast("string"), 
                    schema
                ).alias("data"),
                F.col("timestamp").alias("kafka_timestamp")
            ).select(
                "data.*",
                F.col("kafka_timestamp").alias("processing_time")
            )
        
        # Add watermark for late data
        watermarked_df = parsed_df.withWatermark("timestamp", window_config.watermark_delay)
        
        # Calculate features with windowed aggregations
        window = F.window(
            F.col("timestamp"),
            window_config.window_duration,
            window_config.slide_duration
        )
        
        features_df = watermarked_df \
            .withColumn("vwap_component", F.col("price") * F.col("volume")) \
            .groupBy(
                window,
                "symbol"
            ).agg(
                F.avg("price").alias("avg_price"),
                F.sum("volume").alias("total_volume"),
                (F.sum("vwap_component") / F.sum("volume")).alias("vwap"),
                F.stddev("price").alias("price_volatility"),
                F.max("price").alias("high"),
                F.min("price").alias("low"),
                F.count("*").alias("trade_count")
            ) \
            .withColumn("window_start", F.col("window.start")) \
            .withColumn("window_end", F.col("window.end")) \
            .drop("window")
        
        # Add drift detection if enabled
        if self.drift_detector:
            features_df = self._add_drift_detection(features_df)
        
        return features_df
    
    def _add_drift_detection(self, df: DataFrame) -> DataFrame:
        """
        Add drift detection metrics to the DataFrame.
        
        Args:
            df: Input DataFrame with features
            
        Returns:
            DataFrame with added drift detection columns
        """
        # Define numerical features to monitor for drift
        numerical_features = ['avg_price', 'price_volatility', 'total_volume']
        
        # Convert to Pandas for drift detection
        def detect_drift(batch_df, batch_id):
            if batch_df.rdd.isEmpty():
                return batch_df
                
            # Convert to Pandas for drift detection
            pdf = batch_df.toPandas()
            
            # Check for drift on numerical features
            drift_results = {}
            for feature in numerical_features:
                if feature in pdf.columns:
                    result = self.drift_detector.detect_drift(
                        current_data=pdf[[feature]],
                        drift_type=DriftType.COVARIATE,
                        method=DetectionMethod.KOLMOGOROV_SMIRNOV
                    )[0]
                    drift_results[f"{feature}_drift_score"] = result.statistic
                    drift_results[f"{feature}_p_value"] = result.p_value
                    drift_results[f"{feature}_drifted"] = result.is_drifted
            
            # Add drift results to the DataFrame
            for col, values in drift_results.items():
                pdf[col] = values
                
            return self.spark.createDataFrame(pdf)
        
        # Apply drift detection to each micro-batch
        return df.mapInPandas(
            detect_drift,
            schema=df.schema.add(
                *[StructField(f"{f}_drift_score", DoubleType()) for f in numerical_features] +
                [StructField(f"{f}_p_value", DoubleType()) for f in numerical_features] +
                [StructField(f"{f}_drifted", BooleanType()) for f in numerical_features]
            )
        )
    
    def start(self, 
              input_topic: str = "market_data",
              output_topic: str = "processed_features",
              output_mode: str = "update",
              trigger: str = "10 seconds") -> None:
        """
        Start the streaming job with the given configuration.
        
        Args:
            input_topic: Kafka topic to read from
            output_topic: Kafka topic to write to
            output_mode: Output mode for the sink
            trigger: Processing trigger interval
        """
        logger.info(f"Starting Spark streaming job: {self.app_name}")
        try:
            # Process the data
            features_df = self.process_tick_data(input_topic)
            
            # Prepare output DataFrame
            output_df = features_df.select(
                F.to_json(F.struct(
                    *[col for col in features_df.columns if col != "window"]
                )).alias("value")
            )
            
            # Start the streaming query
            query = self.write_to_kafka(
                df=output_df,
                topic=output_topic,
                output_mode=output_mode,
                trigger=trigger,
                query_name=f"{self.app_name}_kafka_sink"
            )
            
            # Wait for the query to terminate
            query.awaitTermination()
            
        except Exception as e:
            logger.error(f"Error in streaming job: {e}")
            raise
        finally:
            self.spark.stop()

class IcebergSink:
    """Iceberg sink for Spark Structured Streaming."""
    
    def __init__(self, spark: SparkSession, table_path: str):
        self.spark = spark
        self.table_path = table_path
    
    def write_stream(self, df: 'DataFrame', trigger: str = "60 seconds") -> DataStreamWriter:
        """Write stream to Iceberg table."""
        return df.writeStream \
            .format("iceberg") \
            .outputMode("append") \
            .trigger(processingTime=trigger) \
            .option("path", self.table_path) \
            .option("checkpointLocation", f"{self.table_path}/checkpoints") \
            .option("write-format", "parquet")

class FeatureStoreSink:
    """Feature store sink for Spark Structured Streaming."""
    
    def __init__(self, feature_store_uri: str):
        self.feature_store_uri = feature_store_uri
    
    def write_stream(self, df: 'DataFrame', feature_set: str) -> DataStreamWriter:
        """Write stream to feature store."""
        return df.writeStream \
            .format("feature-store") \
            .outputMode("update") \
            .option("feature_set", feature_set) \
            .option("checkpointLocation", f"/tmp/checkpoints/{feature_set}") \
            .trigger(processingTime="60 seconds")

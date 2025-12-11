from json import loads, dumps
import requests
import threading
from time import sleep, time
import sys
from datetime import datetime
from pathlib import Path
import logging
import csv
import io


sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.config import Config


try:
    from src.monitoring.kafka_monitor import KafkaMonitoringConsumer
    MONITORING_AVAILABLE = True
except ImportError:
    from kafka import KafkaConsumer
    MONITORING_AVAILABLE = False
    print("Warning: Monitoring not available, using standard consumer")

logger = logging.getLogger(__name__)

class StreamReddit:
    def __init__(
            self, subreddit, limit=50, author=True,
            comments=False, url=True, name=False,
            num_comments=False, score=False,
            title=True, created_utc=False, edited=False,
            spoiler=False, data_file='reddit_data.csv'
    ):
        self.args = locals()
        del self.args['self']
        self.limit = limit  
        self.data_file = data_file
        self.data_file_path = Config.DATA_FILE_PATH
        self.records_consumed = 0  
        self.timeout_seconds = 300  
        self.csv_writer = None
        self.csv_file = None
        self.fieldnames = None  
        
        
        if MONITORING_AVAILABLE:
            self.monitoring_consumer = KafkaMonitoringConsumer(
                bootstrap_servers=Config.get_kafka_servers(),
                topic=Config.KAFKA_TOPIC
            )
        else:
            self.monitoring_consumer = None
            from kafka import KafkaConsumer
            self.standard_consumer = KafkaConsumer(
                bootstrap_servers=Config.get_kafka_servers(),
                auto_offset_reset='earliest',  # Changed from 'latest' to consume all messages
                enable_auto_commit=True,
                value_deserializer=lambda x: loads(x.decode('utf-8'))
            )
            self.standard_consumer.subscribe([Config.KAFKA_TOPIC])

    def send_data(self):
        r = requests.post(
            Config.PRODUCER_URL,
            data=dumps(self.args).encode('utf-8')
        )
        return r.status_code

    def get_stream(self):    
        logger.info(f"Starting to consume records. Target limit: {self.limit}")
        start_time = time()
        
        if self.monitoring_consumer:
            
            for message in self.monitoring_consumer.consume_with_monitoring():
                
                if time() - start_time > self.timeout_seconds:
                    logger.warning(f"Timeout reached ({self.timeout_seconds}s). Stopping consumption.")
                    logger.info(f"Consumed {self.records_consumed} records out of {self.limit} requested.")
                    break
                
                # Check if we've reached the limit
                if self.records_consumed >= self.limit:
                    logger.info(f"Reached limit of {self.limit} records. Stopping consumption.")
                    break
                
                record = message.value
                
                record['_kafka_timestamp'] = datetime.now().isoformat()
                
                self._write_csv_record(record)
                
                self.records_consumed += 1
                logger.info(f"Saved record {self.records_consumed}/{self.limit} to {self.data_file_path}")
                
                
                if self.records_consumed % 10 == 0:
                    metrics = self.monitoring_consumer.get_metrics()
                    self.monitoring_consumer.monitor.print_metrics()
        else:
            
            for message in self.standard_consumer:
                #  timeout
                if time() - start_time > self.timeout_seconds:
                    logger.warning(f"Timeout reached ({self.timeout_seconds}s). Stopping consumption.")
                    logger.info(f"Consumed {self.records_consumed} records out of {self.limit} requested.")
                    break
                
                
                if self.records_consumed >= self.limit:
                    logger.info(f"Reached limit of {self.limit} records. Stopping consumption.")
                    break
                
                record = message.value
                record['_kafka_timestamp'] = datetime.now().isoformat()
                
                self._write_csv_record(record)
                
                self.records_consumed += 1
                logger.info(f"Saved record {self.records_consumed}/{self.limit} to {self.data_file_path}")
        
        
        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None
        
        elapsed_time = time() - start_time
        logger.info(f"Finished consuming. Total records saved: {self.records_consumed}/{self.limit} in {elapsed_time:.2f} seconds")
    
    def _write_csv_record(self, record):
        """Write a record to CSV file, handling header on first write"""
        
        def flatten_value(value):
            if value is None:
                return ''
            elif isinstance(value, (dict, list)):
                return dumps(value)
            else:
                return str(value)
        
        
        flat_record = {k: flatten_value(v) for k, v in record.items()}
        
        file_exists = self.data_file_path.exists()
        
        
        if file_exists and self.fieldnames is None:
            
            with open(self.data_file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                self.fieldnames = list(reader.fieldnames) if reader.fieldnames else list(flat_record.keys())
        elif not file_exists and self.fieldnames is None:
            
            self.fieldnames = list(flat_record.keys())
        
        
        complete_record = {field: flat_record.get(field, '') for field in self.fieldnames}
        
        
        with open(self.data_file_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames, extrasaction='ignore')
            
    
            if not file_exists:
                writer.writeheader()
            
            writer.writerow(complete_record)

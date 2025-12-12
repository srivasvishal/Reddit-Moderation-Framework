"""
Kafka Monitoring System
Monitors Kafka streaming requests, latency, and throughput
"""

import time
import json
from datetime import datetime
from typing import Dict, List
from kafka import KafkaConsumer, KafkaProducer
from kafka.admin import KafkaAdminClient, ConfigResource, ConfigResourceType
import logging
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)


class KafkaMonitor:
    """Monitor Kafka performance metrics"""
    
    def __init__(self, bootstrap_servers: List[str], topic: str):
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.metrics = {
            'messages_sent': 0,
            'messages_received': 0,
            'total_latency_ms': 0,
            'min_latency_ms': float('inf'),
            'max_latency_ms': 0,
            'throughput_msgs_per_sec': 0,
            'errors': 0,
            'start_time': datetime.now().isoformat()
        }
        self.message_timestamps = {}
        self.latency_history = []
        
    def record_message_sent(self, message_key: str):
        """Record when a message is sent"""
        self.metrics['messages_sent'] += 1
        self.message_timestamps[message_key] = time.time()
    
    def record_message_received(self, message_key: str):
        """Record when a message is received and calculate latency"""
        if message_key in self.message_timestamps:
            send_time = self.message_timestamps[message_key]
            receive_time = time.time()
            latency_ms = (receive_time - send_time) * 1000
            
            self.metrics['messages_received'] += 1
            self.metrics['total_latency_ms'] += latency_ms
            self.metrics['min_latency_ms'] = min(self.metrics['min_latency_ms'], latency_ms)
            self.metrics['max_latency_ms'] = max(self.metrics['max_latency_ms'], latency_ms)
            
            self.latency_history.append({
                'timestamp': datetime.now().isoformat(),
                'latency_ms': latency_ms,
                'message_key': message_key
            })
            
            # Calculate throughput
            elapsed = time.time() - time.mktime(
                datetime.fromisoformat(self.metrics['start_time']).timetuple()
            )
            if elapsed > 0:
                self.metrics['throughput_msgs_per_sec'] = (
                    self.metrics['messages_received'] / elapsed
                )
            
            del self.message_timestamps[message_key]
        else:
            self.metrics['messages_received'] += 1
    
    def record_error(self):
        """Record an error"""
        self.metrics['errors'] += 1
    
    def get_metrics(self) -> Dict:
        """Get current metrics"""
        metrics = self.metrics.copy()
        if metrics['messages_received'] > 0:
            metrics['avg_latency_ms'] = (
                metrics['total_latency_ms'] / metrics['messages_received']
            )
        else:
            metrics['avg_latency_ms'] = 0
        
        return metrics
    
    def get_latency_history(self) -> List[Dict]:
        """Get latency history"""
        return self.latency_history
    
    def save_metrics(self, filepath: str):
        """Save metrics to JSON file"""
        output_dir = Path(filepath).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump({
                'metrics': self.get_metrics(),
                'latency_history': self.latency_history[-100:]  # Last 100 entries
            }, f, indent=2)
    
    def print_metrics(self):
        """Print current metrics"""
        metrics = self.get_metrics()
        print("\n" + "="*60)
        print("Kafka Monitoring Metrics")
        print("="*60)
        print(f"Topic: {self.topic}")
        print(f"Messages Sent: {metrics['messages_sent']}")
        print(f"Messages Received: {metrics['messages_received']}")
        print(f"Errors: {metrics['errors']}")
        print(f"Average Latency: {metrics['avg_latency_ms']:.2f} ms")
        print(f"Min Latency: {metrics['min_latency_ms']:.2f} ms")
        print(f"Max Latency: {metrics['max_latency_ms']:.2f} ms")
        print(f"Throughput: {metrics['throughput_msgs_per_sec']:.2f} msgs/sec")
        print("="*60 + "\n")


class KafkaMonitoringConsumer:
    """Consumer with built-in monitoring"""
    
    def __init__(self, bootstrap_servers: List[str], topic: str):
        self.consumer = KafkaConsumer(
            bootstrap_servers=bootstrap_servers,
            auto_offset_reset='earliest',  # Changed from 'latest' to consume all messages
            enable_auto_commit=True,
            value_deserializer=lambda x: json.loads(x.decode('utf-8'))
        )
        self.consumer.subscribe([topic])
        self.monitor = KafkaMonitor(bootstrap_servers, topic)
    
    def consume_with_monitoring(self):
        """Consume messages with monitoring"""
        for message in self.consumer:
            try:
                # Extract message key or use offset
                message_key = str(message.offset)
                self.monitor.record_message_received(message_key)
                
                yield message
            except Exception as e:
                logger.error(f"Error consuming message: {e}")
                self.monitor.record_error()
    
    def get_metrics(self) -> Dict:
        """Get monitoring metrics"""
        return self.monitor.get_metrics()


class KafkaMonitoringProducer:
    """Producer with built-in monitoring"""
    
    def __init__(self, bootstrap_servers: List[str], topic: str):
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda x: json.dumps(x).encode('utf-8')
        )
        self.topic = topic
        self.monitor = KafkaMonitor(bootstrap_servers, topic)
    
    def send_with_monitoring(self, value: Dict, key: str = None):
        """Send message with monitoring"""
        message_key = key or str(time.time())
        self.monitor.record_message_sent(message_key)
        
        future = self.producer.send(self.topic, value=value, key=message_key.encode() if key else None)
        return future, message_key
    
    def get_metrics(self) -> Dict:
        """Get monitoring metrics"""
        return self.monitor.get_metrics()


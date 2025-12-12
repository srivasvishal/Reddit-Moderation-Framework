"""Entry point for Consumer service"""

import sys
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.consumer.consumer import StreamReddit
import threading
from time import sleep
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    subreddit = os.getenv('SUBREDDIT', 'UCI')
    limit = int(os.getenv('LIMIT', '200'))
    
    stream = StreamReddit(
        subreddit=subreddit,
        comments="True",
        limit=limit,
        num_comments="True"
    )

    # Send request to producer to fetch Reddit data
    send_thread = threading.Thread(target=stream.send_data)
    send_thread.start()
    send_thread.join()  # Wait for request to complete

    # Wait a bit for producer to start processing
    sleep(5)

    # Start consuming records up to the limit
    stream_thread = threading.Thread(target=stream.get_stream)
    stream_thread.start()
    stream_thread.join()  # Wait for all records to be consumed
    
    logger.info(f"Consumer finished. Consumed {stream.records_consumed} records out of {limit} requested.")


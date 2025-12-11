from flask import Flask, Response, jsonify
import sys
import os
from datetime import datetime
from pathlib import Path
import csv

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.config import Config

app = Flask(__name__)

DATA_FILE = Config.DATA_FILE_PATH

@app.route('/', methods=['GET'])
def index():
    """Root endpoint with API information"""
    return jsonify({
        'service': 'Reddit Stream Data API',
        'version': '1.0.0',
        'endpoints': {
            'health': '/health',
            'records': '/api/records',
            'records_info': '/api/records/info'
        },
        'usage': {
            'health_check': 'GET /health',
            'get_all_records': 'GET /api/records',
            'get_records_info': 'GET /api/records/info'
        }
    }), 200

@app.route('/api/records', methods=['GET'])
def get_records():
    """
    API endpoint that returns all records as a CSV file.
    This endpoint is consumed by the TRIAGE AI Agent.
    """
    if not os.path.exists(DATA_FILE):
        return jsonify({'error': 'No data file found. Consumer may not have started yet.'}), 404
    
    try:
        def generate():
            with open(DATA_FILE, 'r', encoding='utf-8') as f:
                for line in f:
                        yield line
        
        return Response(
            generate(),
            mimetype='text/csv',
            headers={
                'Content-Disposition': f'attachment; filename=reddit_records_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                'Content-Type': 'text/csv; charset=utf-8'
            }
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/records/info', methods=['GET'])
def get_records_info():
    """
    Get information about the records file (count, size, etc.)
    """
    if not os.path.exists(DATA_FILE):
        return jsonify({'error': 'No data file found'}), 404
    
    try:
        record_count = 0
        file_size = os.path.getsize(DATA_FILE)
        
        # Count CSV records (excluding header)
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            record_count = sum(1 for row in reader)
        
        return jsonify({
            'file_path': str(DATA_FILE),
            'record_count': record_count,
            'file_size_bytes': file_size,
            'file_size_mb': round(file_size / (1024 * 1024), 2)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'service': 'reddit-data-api'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=Config.API_PORT, debug=True)


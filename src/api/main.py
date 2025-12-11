"""Entry point for API service"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.api.api import app

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Serve Puter.js Web Demo

This script serves the Puter.js web demo using Flask.
"""

import os
import sys
import logging
from pathlib import Path
from flask import Flask, send_from_directory

# Add the parent directory to the path so we can import from src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)

@app.route('/')
def serve_demo():
    """Serve the Puter.js web demo."""
    return send_from_directory(os.path.dirname(__file__), 'puter_web_demo.html')

def main():
    """Run the Flask server."""
    logger.info("Starting Puter.js web demo server")
    logger.info("Open your browser and navigate to http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)

if __name__ == '__main__':
    main()
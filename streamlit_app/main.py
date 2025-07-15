"""
Flask wrapper for Streamlit application deployment
"""

from flask import Flask, redirect
import subprocess
import threading
import time
import os

app = Flask(__name__)

# Start Streamlit in a separate thread
def start_streamlit():
    """Start the Streamlit application"""
    time.sleep(2)  # Give Flask time to start
    subprocess.run([
        'streamlit', 'run', 'src/app.py', 
        '--server.port', '8502',
        '--server.address', '0.0.0.0',
        '--server.headless', 'true'
    ])

@app.route('/')
def index():
    """Redirect to Streamlit app"""
    return redirect('http://localhost:8502')

@app.route('/health')
def health():
    """Health check endpoint"""
    return {'status': 'healthy', 'service': 'Census Income Prediction'}

if __name__ == '__main__':
    # Start Streamlit in background
    streamlit_thread = threading.Thread(target=start_streamlit, daemon=True)
    streamlit_thread.start()
    
    # Start Flask
    app.run(host='0.0.0.0', port=5000, debug=False)


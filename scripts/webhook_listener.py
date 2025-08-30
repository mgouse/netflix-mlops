#!/usr/bin/env python3
# Shebang line - tells the OS to use Python 3 interpreter to run this script

from flask import Flask, request, jsonify     # Flask is used to build web APIs
import subprocess                             # Used to run shell commands from Python
import json                                   # For handling JSON data
import logging                                # For logging output to console with timestamps
from datetime import datetime                 # Used in health check to return current time
import os                                     # Used for file path operations and checking file existence

# Create the Flask web application
app = Flask(__name__)

# Configure logging to show timestamps and message level
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Define the /webhook route which listens for POST requests
@app.route('/webhook', methods=['POST'])
def webhook():
    try:
        logger.info("=== Webhook Received ===")  # Log when the webhook is triggered

        data = request.json  # Parse JSON payload from the request body

        # Log the full JSON payload nicely formatted
        logger.info(f"Full payload: {json.dumps(data, indent=2)}")

        # Extract specific values from the JSON (or provide default values if not found)
        commit_sha = data.get('commit_sha', 'unknown')
        image_tag = data.get('image_tag', 'latest')

        # Log what commit and image tag are being processed
        logger.info(f"Processing commit: {commit_sha}, image tag: {image_tag}")

        # Build the absolute path to the workflow YAML file (relative to this script)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        workflow_path = os.path.join(script_dir, "..", "pipelines", "argo", "netflix_ml_pipeline_phase8.yaml")

        logger.info(f"Using workflow path: {workflow_path}")  # Log the resolved file path

        # Check if the YAML workflow file exists
        if not os.path.exists(workflow_path):
            error_msg = f"Workflow file not found: {workflow_path}"
            logger.error(error_msg)
            return jsonify({"status": "error", "error": error_msg}), 500

        # Prepare the kubectl command to trigger the Argo workflow
        cmd = ["kubectl", "create", "-f", workflow_path]

        # Run the kubectl command and capture output
        result = subprocess.run(cmd, capture_output=True, text=True, shell=True)

        # Check if the command was successful
        if result.returncode == 0:
            logger.info(f"✓ Workflow triggered: {result.stdout.strip()}")
            return jsonify({"status": "success", "output": result.stdout}), 200
        else:
            # If kubectl failed, log the error and return a 500 response
            logger.error(f"✗ Failed: {result.stderr}")
            return jsonify({"status": "error", "error": result.stderr}), 500

    except Exception as e:
        # Handle any unexpected exceptions during processing
        logger.error(f"✗ Error: {str(e)}")
        return jsonify({"status": "error", "error": str(e)}), 500

# Define a health check endpoint
@app.route('/health', methods=['GET'])
def health():
    # Returns app status and current server time
    return jsonify({"status": "healthy", "time": datetime.now().isoformat()}), 200

# Define a debug endpoint to help test environment and file paths
@app.route('/debug', methods=['GET'])
def debug():
    return jsonify({
        "working_dir": os.getcwd(),  # Shows the current working directory
        "script_path": os.path.abspath(__file__),  # Shows the full path to this script
        "workflow_exists": os.path.exists(os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "pipelines", "argo", "netflix_ml_pipeline_phase8.yaml"
        ))  # Checks whether the YAML workflow file exists
    })

# This block runs the app only when the script is executed directly
if __name__ == '__main__':
    logger.info("Starting webhook listener on port 5002...")  # Log that the server is starting
    app.run(host='0.0.0.0', port=5002, debug=True)  # Start Flask server on port 5002 and listen on all IPs

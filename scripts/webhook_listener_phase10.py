#!/usr/bin/env python3
import os
import sys
import json
import logging
import subprocess
from datetime import datetime
from flask import Flask, request, jsonify

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration - UPDATE THESE PATHS
WORKFLOW_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pipelines", "argo")
KUBECONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kubeconfig-github.yaml")

@app.route('/webhook', methods=['POST'])
def webhook():
    try:
        logger.info("=== Webhook Received ===")
        data = request.json
        logger.info(f"Payload: {json.dumps(data, indent=2)}")

        # Extract GitHub data
        commit_sha = data.get('commit_sha', 'unknown')[:7]
        image_tag = data.get('image_tag', 'latest')
        repo_name = data.get('repository', 'unknown')
        
        logger.info(f"Processing: {repo_name} @ {commit_sha}, Image: {image_tag}")

        # Determine which workflow to use
        # Option A: Use phase10 fixed workflow (recommended)
        workflow_file = "netflix-ml-pipeline-fixed_phase10.yml"
        
        # Option B: Create dynamic workflow with image tag injected
        workflow_path = os.path.join(WORKFLOW_DIR, workflow_file)
        
        if not os.path.exists(workflow_path):
            logger.error(f"Workflow not found: {workflow_path}")
            return jsonify({"status": "error", "error": f"File not found: {workflow_path}"}), 500

        logger.info(f"Submitting workflow: {workflow_path}")

        # Set environment for kubectl
        env = os.environ.copy()
        if os.path.exists(KUBECONFIG):
            env['KUBECONFIG'] = KUBECONFIG
            logger.info(f"Using kubeconfig: {KUBECONFIG}")

        # Build kubectl command - CRITICAL FIX: shell=False for list args
        cmd = [
            "kubectl", 
            "create", 
            "-f", 
            workflow_path,
            "-n", 
            "argo"
        ]

        # Execute
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            shell=False,  # FIXED: Must be False when using list
            env=env
        )

        if result.returncode == 0:
            logger.info(f"✓ SUCCESS: {result.stdout.strip()}")
            return jsonify({
                "status": "success", 
                "workflow": workflow_file,
                "commit": commit_sha,
                "kubectl_output": result.stdout.strip()
            }), 200
        else:
            logger.error(f"✗ FAILED: {result.stderr}")
            return jsonify({
                "status": "error", 
                "error": result.stderr,
                "command": " ".join(cmd)
            }), 500

    except Exception as e:
        logger.error(f"✗ EXCEPTION: {str(e)}")
        return jsonify({"status": "error", "error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    # Check kubectl connectivity
    try:
        env = os.environ.copy()
        if os.path.exists(KUBECONFIG):
            env['KUBECONFIG'] = KUBECONFIG
        
        result = subprocess.run(
            ["kubectl", "version", "--client"], 
            capture_output=True, 
            text=True, 
            shell=False,
            env=env
        )
        kubectl_status = "connected" if result.returncode == 0 else "disconnected"
    except Exception as e:
        kubectl_status = f"error: {str(e)}"

    return jsonify({
        "status": "healthy",
        "time": datetime.now().isoformat(),
        "kubectl": kubectl_status,
        "workflow_dir_exists": os.path.exists(WORKFLOW_DIR)
    }), 200

if __name__ == '__main__':
    logger.info(f"Starting webhook server...")
    logger.info(f"Workflow dir: {WORKFLOW_DIR}")
    logger.info(f"Kubeconfig: {KUBECONFIG}")
    logger.info(f"Server running on http://0.0.0.0:5002")
    app.run(host='0.0.0.0', port=5002, debug=False)  # debug=False for production
import subprocess
import logging
import json
import os
from flask import Flask, request, jsonify
from datetime import datetime

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('D:/Netflix/logs/webhook.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
WORKFLOW_YAML = "D:/Netflix/pipelines/argo/netflix-ml-pipeline-fixed_phase10.yml"
NAMESPACE = "argo"


def check_argo_health():
    """Check if Argo controller is running"""
    try:
        result = subprocess.run(
            ["kubectl", "get", "pods", "-n", NAMESPACE, "-l", "app=workflow-controller"],
            capture_output=True,
            text=True,
            timeout=10
        )
        return "Running" in result.stdout
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return False


def cleanup_old_workflows(keep_last=10):
    """Delete old completed workflows to save space"""
    try:
        result = subprocess.run(
            ["kubectl", "get", "workflows", "-n", NAMESPACE, "-o", "json"],
            capture_output=True,
            text=True,
            timeout=15
        )
        
        if result.returncode == 0:
            workflows = json.loads(result.stdout)
            items = workflows.get("items", [])
            
            # Sort by creation time
            items_sorted = sorted(items, key=lambda x: x["metadata"]["creationTimestamp"])
            
            # Keep only last N
            if len(items_sorted) > keep_last:
                for wf in items_sorted[:-keep_last]:
                    phase = wf.get("status", {}).get("phase", "")
                    if phase in ["Succeeded", "Failed", "Error"]:
                        name = wf["metadata"]["name"]
                        logger.info(f"Deleting old workflow: {name}")
                        subprocess.run(
                            ["kubectl", "delete", "workflow", name, "-n", NAMESPACE],
                            capture_output=True,
                            timeout=10
                        )
    except Exception as e:
        logger.warning(f"Cleanup failed: {e}")


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    argo_healthy = check_argo_health()
    return jsonify({
        "status": "healthy" if argo_healthy else "unhealthy",
        "argo_controller": "running" if argo_healthy else "not running",
        "workflow_yaml": WORKFLOW_YAML,
        "timestamp": datetime.utcnow().isoformat()
    }), 200 if argo_healthy else 503


@app.route('/webhook', methods=['POST'])
def webhook():
    """GitHub Actions webhook handler"""
    try:
        payload = request.json or {}
        
        logger.info("=" * 80)
        logger.info("📨 WEBHOOK RECEIVED FROM GITHUB ACTIONS")
        logger.info(f"Repository: {payload.get('repository', 'unknown')}")
        logger.info(f"Event: {payload.get('event', 'unknown')}")
        logger.info(f"Commit SHA: {payload.get('commit_sha', 'unknown')}")
        logger.info(f"Triggered by: {payload.get('triggered_by', 'unknown')}")
        logger.info(f"Image: {payload.get('image_name', '')}:{payload.get('image_tag', '')}")
        
        workflow_run = payload.get('workflow_run', {})
        if workflow_run:
            logger.info(f"GitHub Workflow: {workflow_run.get('name')}")
            logger.info(f"Run #: {workflow_run.get('run_number')}")
        logger.info("=" * 80)
        
        # Health check
        if not check_argo_health():
            logger.error("❌ Argo Workflows is not healthy")
            return jsonify({
                "status": "error",
                "message": "Argo controller not running",
                "action": "Check: kubectl get pods -n argo"
            }), 503
        
        # Cleanup old workflows
        cleanup_old_workflows(keep_last=10)
        
        # Verify workflow file exists
        if not os.path.exists(WORKFLOW_YAML):
            logger.error(f"❌ Workflow file not found: {WORKFLOW_YAML}")
            return jsonify({
                "status": "error",
                "message": f"Workflow YAML not found: {WORKFLOW_YAML}"
            }), 500
        
        # Submit workflow
        logger.info(f"📤 Submitting workflow: {WORKFLOW_YAML}")
        result = subprocess.run(
            ["kubectl", "create", "-f", WORKFLOW_YAML, "-n", NAMESPACE],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            # Parse workflow name from output: "workflow.argoproj.io/name created"
            workflow_name = result.stdout.strip().split("/")[-1].replace(" created", "")
            
            logger.info(f"✅ Workflow submitted successfully: {workflow_name}")
            
            return jsonify({
                "status": "success",
                "message": "ML pipeline triggered successfully",
                "workflow_name": workflow_name,
                "namespace": NAMESPACE,
                "commit_sha": payload.get("commit_sha", ""),
                "image_tag": payload.get("image_tag", ""),
                "timestamp": datetime.utcnow().isoformat(),
                "argo_ui": f"http://localhost:2746/workflows/{NAMESPACE}/{workflow_name}"
            }), 200
        else:
            logger.error(f"❌ Workflow submission failed: {result.stderr}")
            return jsonify({
                "status": "error",
                "message": "Failed to submit workflow",
                "error": result.stderr,
                "stdout": result.stdout
            }), 500
            
    except Exception as e:
        logger.exception(f"❌ Webhook handler error: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@app.route('/workflows', methods=['GET'])
def list_workflows():
    """List recent workflows"""
    try:
        result = subprocess.run(
            ["kubectl", "get", "workflows", "-n", NAMESPACE, "-o", "json"],
            capture_output=True,
            text=True,
            timeout=15
        )
        
        if result.returncode == 0:
            workflows = json.loads(result.stdout)
            items = workflows.get("items", [])
            
            workflow_list = []
            for wf in reversed(items[-10:]):  # Last 10
                workflow_list.append({
                    "name": wf["metadata"]["name"],
                    "status": wf.get("status", {}).get("phase", "Unknown"),
                    "created": wf["metadata"].get("creationTimestamp"),
                    "started": wf.get("status", {}).get("startedAt"),
                    "finished": wf.get("status", {}).get("finishedAt")
                })
            
            return jsonify({
                "workflows": workflow_list,
                "total": len(workflow_list)
            }), 200
        else:
            return jsonify({"error": "Failed to list workflows"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    logger.info("=" * 80)
    logger.info("🚀 Netflix MLOps Webhook Listener Starting")
    logger.info(f"📁 Workflow YAML: {WORKFLOW_YAML}")
    logger.info(f"📦 Namespace: {NAMESPACE}")
    logger.info(f"🌐 Port: 5002")
    logger.info("=" * 80)
    
    if check_argo_health():
        logger.info("✅ Argo Workflows is healthy")
    else:
        logger.warning("⚠️  Argo Workflows health check failed")
    
    app.run(host='0.0.0.0', port=5002, debug=False)
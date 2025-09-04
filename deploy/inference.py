"""
SageMaker-compatible inference server for Flavor RAG.
Exposes:
  - GET  /ping          -> 200 OK (health)
  - POST /invocations   -> JSON {query, top_k, bitter_threshold}
Emits CloudWatch EMF metrics (LatencyMs, ShortlistSize).
"""
import os, sys, json, time
from flask import Flask, request, Response

# Make sure we can import from ./src
HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
SRC  = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.append(SRC)

import genai_flavor_rag as rag  # your pipeline

# Ensure index dir exists (supports either rag.INDEX_DIR or a local "faiss_index/")
INDEX_DIR = getattr(rag, "INDEX_DIR", os.path.join(ROOT, "faiss_index"))
os.makedirs(INDEX_DIR, exist_ok=True)

app = Flask(__name__)

@app.route("/ping", methods=["GET"])
def ping():
    # Optional: try loading something lightweight here if you want deeper health checks
    return Response("ok", 200)

@app.route("/invocations", methods=["POST"])
def invocations():
    t0 = time.time()
    try:
        body = request.get_json(force=True) or {}
        query = (body.get("query") or "").strip()
        top_k = int(body.get("top_k", 12))

        # Let caller override BITTER_THRESH at runtime
        if "bitter_threshold" in body:
            rag.BITTER_THRESH = float(body["bitter_threshold"])

        if not query:
            return Response(json.dumps({"ok": False, "message": "query is required"}), 400, mimetype="application/json")

        out = rag.answer(query, top_k=top_k)
        latency_ms = int((time.time() - t0) * 1000)

        # CloudWatch Embedded Metric Format (no SDK calls needed)
        try:
            emf = {
                "_aws": {
                    "Timestamp": int(time.time() * 1000),
                    "CloudWatchMetrics": [{
                        "Namespace": "FlavorRAG/Online",
                        "Dimensions": [["Stage"]],
                        "Metrics": [
                            {"Name": "LatencyMs", "Unit": "Milliseconds"},
                            {"Name": "ShortlistSize", "Unit": "Count"}
                        ]
                    }]
                },
                "Stage": os.getenv("STAGE", "dev"),
                "LatencyMs": latency_ms,
                "ShortlistSize": len(out.get("shortlist", [])) if out and out.get("ok") else 0
            }
            print(json.dumps(emf), flush=True)
        except Exception:
            pass

        return Response(json.dumps(out), 200, mimetype="application/json")
    except Exception as e:
        err = {"ok": False, "message": f"server error: {e.__class__.__name__}: {e}"}
        return Response(json.dumps(err), 500, mimetype="application/json")

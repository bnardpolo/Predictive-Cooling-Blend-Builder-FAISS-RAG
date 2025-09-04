"""
Offline eval to compute Precision@K / Recall@K for the retriever
(before and after Bitter-Guard) and optionally log to MLflow.

It works even if your `answer()` hasn't been patched to return
retrieved_* ids yet — in that case it prints a notice and skips metrics.
"""
import os, sys, json
from typing import List, Dict, Tuple

# Import pipeline from ../src
HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
SRC  = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.append(SRC)

from genai_flavor_rag import answer  # uses your local data/models
USE_MLFLOW = bool(os.getenv("MLFLOW_TRACKING_URI"))

if USE_MLFLOW:
    import mlflow
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT", "FlavorRAG"))

# ---------------- Helpers ----------------
def pr_at_k(retrieved: List[str], truth: List[str], k: int = 12) -> Tuple[float, float]:
    if not retrieved: return 0.0, 0.0
    top = retrieved[:k]
    hits = sum(1 for x in top if x in truth)
    precision = hits / max(len(top), 1)
    recall    = hits / max(len(truth), 1) if truth else 0.0
    return precision, recall

def safe_get(obj: Dict, key: str) -> List[str]:
    v = obj.get(key)
    return v if isinstance(v, list) else []

# --------------- Evaluation set ---------------
# Fill with ~50 examples for a proper run.
EVAL: List[Dict] = [
    {
        "query": "Clean minty cooling with low bitterness",
        # IDs should match how your pipeline formats doc IDs:
        # "Name||SMILES" (e.g., "L-Menthol||CC1CCC(C(C)C)C(O)C1")
        "truth": ["L-Menthol||CC1CCC(C(C)C)C(O)C1"]
    },
]

def main():
    run_ctx = None
    if USE_MLFLOW:
        run_ctx = mlflow.start_run(run_name=os.getenv("MODEL_TAG", "local"))

    try:
        for ex in EVAL:
            q = ex["query"]
            out = answer(q, top_k=12)

            before_ids = safe_get(out, "retrieved_before_guard_ids")
            after_ids  = safe_get(out, "retrieved_after_guard_ids")
            truth_ids  = ex.get("truth", [])

            # If your pipeline hasn't been patched yet, tell the user once.
            if not before_ids and not after_ids:
                print(f"[WARN] answer() did not return retrieved_* ids for query: {q}")
                print("       Patch answer() to include 'retrieved_before_guard_ids' and 'retrieved_after_guard_ids'.")
            else:
                p_b, r_b = pr_at_k(before_ids, truth_ids, k=12)
                p_a, r_a = pr_at_k(after_ids,  truth_ids, k=12)
                print(f"[{q}]  precision@12(before)={p_b:.3f}  recall@12(before)={r_b:.3f}  "
                      f"precision@12(after)={p_a:.3f}  recall@12(after)={r_a:.3f}")

                if USE_MLFLOW:
                    mlflow.log_metric("precision@12_before", p_b)
                    mlflow.log_metric("recall@12_before",    r_b)
                    mlflow.log_metric("precision@12_after",  p_a)
                    mlflow.log_metric("recall@12_after",     r_a)

            # Always keep artifacts for traceability (context, blend text)
            ctx = out.get("context_blocks", "")
            blend = out.get("blend_text", "")
            if USE_MLFLOW:
                mlflow.log_text(ctx, artifact_file=f"context/{q}.txt")
                mlflow.log_text(blend, artifact_file=f"blends/{q}.txt")
            else:
                # Write to local files if MLflow isn't configured
                os.makedirs(os.path.join(ROOT, "eval_artifacts", "context"), exist_ok=True)
                os.makedirs(os.path.join(ROOT, "eval_artifacts", "blends"),  exist_ok=True)
                open(os.path.join(ROOT, "eval_artifacts", "context", f"{q}.txt"), "w", encoding="utf-8").write(ctx)
                open(os.path.join(ROOT, "eval_artifacts", "blends",  f"{q}.txt"), "w", encoding="utf-8").write(blend)

    finally:
        if run_ctx and USE_MLFLOW:
            mlflow.end_run()

if __name__ == "__main__":
    main()

# =========================================================
# genai_flavor_rag.py — S3-aware loaders + ranking helpers
# - Supports local paths OR s3:// URIs via .env
# - Downloads S3 files/prefixes into .cache/ before loading
# - Shortlists candidates by cooling/bitterness/keyword match
# - Provides fallback suggestions + preset choices
# - Back-compat globals (INDEX_DIR, DATASET, etc.) for existing UI
# =========================================================

import os
import re
from pathlib import Path
from urllib.parse import urlparse
from typing import Tuple, Optional, Dict, Any, List

import joblib
import pandas as pd

# ----- Optional S3 deps (only needed if using s3:// paths) -----
try:
    import boto3
    from botocore.exceptions import NoCredentialsError, ClientError
except Exception:
    boto3 = None  # We'll raise only if an s3:// path is actually used

from dotenv import load_dotenv, dotenv_values

# ---------- Project root + .env ----------
HERE = Path(__file__).resolve()
ROOT = HERE.parents[1] if HERE.parent.name == "src" else HERE.parent
ENV_FILE = ROOT / ".env"

if ENV_FILE.exists():
    load_dotenv(ENV_FILE, override=True)
    for k, v in dotenv_values(ENV_FILE).items():
        if v is not None:
            os.environ[k] = v

# ---------- Config (local path or s3://...) ----------
TASTE_MODEL_PATH      = os.getenv("TASTE_MODEL_PATH", "")
TASTE_LABEL_BINARIZER = os.getenv("TASTE_LABEL_BINARIZER", "")
FLAVOR_DATASET        = os.getenv("FLAVOR_DATASET", "")
FAISS_INDEX_DIR       = os.getenv("FAISS_INDEX_DIR", "")

CACHE_DIR = (ROOT / ".cache").resolve()
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ---------- S3 helpers ----------
def _is_s3_uri(uri: str) -> bool:
    return isinstance(uri, str) and uri.strip().lower().startswith("s3://")

def _parse_s3_uri(uri: str) -> Tuple[str, str]:
    parsed = urlparse(uri)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    if not bucket or not key:
        raise ValueError(f"Invalid S3 URI: {uri}")
    return bucket, key

def _require_boto3():
    if boto3 is None:
        raise RuntimeError(
            "boto3 not available. Install it (pip install boto3) and configure AWS credentials "
            "(e.g., run 'aws configure')."
        )

def _s3_client():
    _require_boto3()
    try:
        return boto3.client("s3")
    except NoCredentialsError:
        raise RuntimeError(
            "AWS credentials not found. Run 'aws configure' or set AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY."
        )

def ensure_local_file(path_or_s3: str, prefer_name: Optional[str] = None) -> Path:
    """
    If path is s3://bucket/key, download to .cache/<basename or prefer_name> and return local Path.
    If path is local, verify it exists and return the resolved Path.
    """
    if not path_or_s3:
        raise FileNotFoundError("Empty path provided.")

    if _is_s3_uri(path_or_s3):
        bucket, key = _parse_s3_uri(path_or_s3)
        filename = prefer_name or Path(key).name
        local_path = CACHE_DIR / filename
        if not local_path.exists():
            s3 = _s3_client()
            local_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                s3.download_file(bucket, key, str(local_path))
            except ClientError as e:
                raise FileNotFoundError(f"Failed to download s3://{bucket}/{key}: {e}")
        return local_path

    p = Path(path_or_s3).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Local file not found: {p}")
    return p

def ensure_local_dir(dir_or_s3_prefix: str) -> Path:
    """
    If s3://bucket/prefix, mirror objects under prefix into .cache/<last-segment>/ and return that dir.
    If local dir, verify and return.
    """
    if not dir_or_s3_prefix:
        raise FileNotFoundError("Empty directory/prefix provided.")

    if _is_s3_uri(dir_or_s3_prefix):
        bucket, prefix = _parse_s3_uri(dir_or_s3_prefix)
        folder_name = Path(prefix.rstrip("/")).name or "s3_dir"
        local_dir = CACHE_DIR / folder_name

        if not local_dir.exists() or not any(local_dir.iterdir()):
            local_dir.mkdir(parents=True, exist_ok=True)
            s3 = _s3_client()
            paginator = s3.get_paginator("list_objects_v2")
            found = False
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix.rstrip("/") + "/"):
                for obj in page.get("Contents", []):
                    found = True
                    key = obj["Key"]
                    rel = Path(key[len(prefix):].lstrip("/"))
                    target = local_dir / rel
                    target.parent.mkdir(parents=True, exist_ok=True)
                    s3.download_file(bucket, key, str(target))
            if not found:
                # allow "prefix is actually a single file" case
                try:
                    file_target = local_dir / Path(prefix).name
                    s3.download_file(bucket, prefix, str(file_target))
                except ClientError as e:
                    raise FileNotFoundError(
                        f"No objects under s3://{bucket}/{prefix} and single-file download failed: {e}"
                    )
        return local_dir

    d = Path(dir_or_s3_prefix).expanduser().resolve()
    if not d.exists():
        raise FileNotFoundError(f"Local directory not found: {d}")
    return d

# ---------- Artifact loaders ----------
def load_taste_model():
    """
    Load RandomForest model + MultiLabelBinarizer from local/S3 (download first if S3).
    Returns: (rf_model, mlb)
    """
    model_local = ensure_local_file(TASTE_MODEL_PATH, prefer_name="taste_model_randomforest.joblib")
    mlb_local   = ensure_local_file(TASTE_LABEL_BINARIZER, prefer_name="taste_label_binarizer.joblib")
    rf  = joblib.load(str(model_local))
    mlb = joblib.load(str(mlb_local))
    return rf, mlb

def load_flavor_dataset() -> pd.DataFrame:
    """
    Load CSV/XLSX dataset; normalize expected columns if missing.
    Expected: name, aroma, taste, cool_score, bitter_prob (others OK).
    """
    local = ensure_local_file(FLAVOR_DATASET)
    if local.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(local)
    else:
        df = pd.read_csv(local)

    # Normalize text columns (best-effort aliases)
    def _ensure_text(col: str, candidates: List[str]):
        if col in df.columns:
            return
        for alt in candidates:
            if alt in df.columns:
                df[col] = df[alt]
                return
        df[col] = ""

    _ensure_text("name",  ["Name", "compound", "Compound"])
    _ensure_text("aroma", ["Aroma", "aroma_notes", "aroma_list"])
    _ensure_text("taste", ["Taste", "taste_notes", "taste_list"])

    # Numeric defaults if missing
    if "cool_score" not in df.columns:
        def _cool_from_text(a, t):
            text = f"{str(a)} {str(t)}".lower()
            return 1.0 if any(w in text for w in ["cool", "menthol", "mint", "peppermint"]) else 0.0
        df["cool_score"] = [ _cool_from_text(a, t) for a, t in zip(df["aroma"], df["taste"]) ]

    if "bitter_prob" not in df.columns:
        df["bitter_prob"] = 0.5  # unknown → neutral middle

    # Coerce numeric types safely
    df["cool_score"]  = pd.to_numeric(df["cool_score"],  errors="coerce").fillna(0.0)
    df["bitter_prob"] = pd.to_numeric(df["bitter_prob"], errors="coerce").fillna(0.5)

    return df

def get_faiss_index_dir() -> Path:
    return ensure_local_dir(FAISS_INDEX_DIR)

# ---------- Intent / scope helpers ----------
_SCOPE_KEYWORDS = {
    "dessert": ["dessert", "ice cream", "cake", "frosting", "creamy", "vanilla", "sweet"],
    "herbal":  ["herbal", "herb", "camphor", "eucalyptus", "rosemary", "sage"],
    "mint":    ["mint", "menthol", "peppermint", "menthyl", "cool"],
}

def _infer_scope(goal_text: str) -> str:
    text = (goal_text or "").lower()
    for scope, kws in _SCOPE_KEYWORDS.items():
        if any(k in text for k in kws):
            return scope
    return "generic"

def _keyword_score(text: str, goal_text: str) -> float:
    """Very simple overlap score between free text and goal words."""
    base = str(text or "").lower()
    words = [w for w in re.findall(r"[a-z]+", (goal_text or "").lower()) if len(w) > 2]
    if not words:
        return 0.0
    hits = sum(1 for w in words if w in base)
    return hits / max(4, len(words))  # dampen long prompts

# ---------- Ranking / shortlisting ----------
def shortlist_candidates(
    df: pd.DataFrame,
    goal_text: str,
    bitter_thresh: float = 0.40,
    min_cool: float = 0.15,
    top_k: int = 12,
) -> List[Dict[str, Any]]:
    """
    Rank compounds for a goal_text by favoring cooling, penalizing bitterness,
    and rewarding keyword alignment. Returns up to top_k candidates.
    """
    if df is None or len(df) == 0:
        return []

    # Ensure required cols exist/safe
    for col in ["name", "aroma", "taste"]:
        if col not in df.columns:
            df[col] = ""
    for col in ["cool_score", "bitter_prob"]:
        if col not in df.columns:
            df[col] = 0.0

    df = df.copy()
    df["cool_score"]  = pd.to_numeric(df["cool_score"],  errors="coerce").fillna(0.0)
    df["bitter_prob"] = pd.to_numeric(df["bitter_prob"], errors="coerce").fillna(0.5)

    scope = _infer_scope(goal_text)
    extra_terms: List[str] = []
    if scope == "dessert":
        extra_terms = ["vanilla", "creamy", "sweet", "buttery", "cake", "icing"]
    elif scope == "herbal":
        extra_terms = ["herbal", "camphor", "eucalyptus", "sage", "rosemary"]
    elif scope == "mint":
        extra_terms = ["mint", "menthol", "menthyl", "peppermint", "cool"]

    def _row_score(row: pd.Series) -> float:
        aroma = str(row.get("aroma", "") or "")
        taste = str(row.get("taste", "") or "")
        text_for_match = f"{aroma} {taste}"

        kcore  = _keyword_score(text_for_match, goal_text)
        kextra = _keyword_score(text_for_match, " ".join(extra_terms)) if extra_terms else 0.0
        cool   = float(row.get("cool_score", 0.0) or 0.0)
        bitter = float(row.get("bitter_prob", 0.5) or 0.5)

        # Tune weights as you learn
        return (0.60 * cool) - (0.50 * bitter) + (0.60 * kcore) + (0.30 * kextra)

    # Primary thresholding
    filtered = df[(df["bitter_prob"] <= bitter_thresh) & (df["cool_score"] >= min_cool)].copy()
    target = filtered if len(filtered) else df

    # Rank
    target["__score__"] = target.apply(_row_score, axis=1)
    ranked = target.sort_values("__score__", ascending=False).head(int(top_k))

    # Assemble results
    results: List[Dict[str, Any]] = []
    for _, r in ranked.iterrows():
        name   = str(r.get("name", "") or "")
        aroma  = str(r.get("aroma", "") or "")
        taste  = str(r.get("taste", "") or "")
        cool   = float(r.get("cool_score", 0.0) or 0.0)
        bitter = float(r.get("bitter_prob", 0.5) or 0.5)
        score  = float(r.get("__score__", 0.0) or 0.0)

        aroma_taste = f"{aroma} {taste}"
        match_core  = _keyword_score(aroma_taste, goal_text)

        results.append({
            "name": name,
            "aroma": aroma,
            "taste": taste,
            "cool_score": cool,
            "bitter_prob": bitter,
            "score": score,
            "reason": f"cool={cool:.2f}, bitter={bitter:.2f}, match≈{match_core:.2f}",
        })

    return results

# ---------- Preset (“pretrained”) choices for UI ----------
def preset_choices(df: pd.DataFrame, limit: int = 25) -> List[str]:
    if df is None or len(df) == 0:
        return []
    df2 = df.copy()
    df2["cool_score"]  = pd.to_numeric(df2.get("cool_score", 0.0), errors="coerce").fillna(0.0)
    df2["bitter_prob"] = pd.to_numeric(df2.get("bitter_prob", 0.5), errors="coerce").fillna(0.5)
    df2["__rank__"] = 0.8 * df2["cool_score"] - 0.6 * df2["bitter_prob"]
    return (
        df2.sort_values("__rank__", ascending=False)
           .head(int(limit))
           .get("name", pd.Series([], dtype=str))
           .dropna()
           .astype(str)
           .tolist()
    )

# ---------- Environment check ----------
def environment_check() -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    for key, val in {
        "TASTE_MODEL_PATH": TASTE_MODEL_PATH,
        "TASTE_LABEL_BINARIZER": TASTE_LABEL_BINARIZER,
        "FLAVOR_DATASET": FLAVOR_DATASET,
        "FAISS_INDEX_DIR": FAISS_INDEX_DIR,
    }.items():
        try:
            if key == "FAISS_INDEX_DIR":
                path = ensure_local_dir(val)
            else:
                path = ensure_local_file(val)
            results[key] = {"configured": val, "local_resolved": str(path), "exists": path.exists()}
        except Exception as e:
            results[key] = {"configured": val, "error": str(e), "exists": False}
    return results

# ---------- Main facade for UI ----------
def answer(
    goal_text: str,
    top_k: int = 12,
    bitter_thresh: float = 0.40,
    min_cool: float = 0.15,
    exact_compounds: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Returns shortlist + suggestions + helpful messages.
    - If no items meet thresholds, offers nearest alternatives.
    - If user picks exact compounds, restricts to those first.
    """
    df = load_flavor_dataset()

    scope = _infer_scope(goal_text)
    scope_msg = ""
    low = goal_text.lower()
    if ("ice cream" in low) or ("cake" in low):
        scope_msg = "Dessert-like profile requested; prioritizing creamy/vanilla/sweet notes."

    if exact_compounds:
        names = [str(x) for x in exact_compounds]
        df_view = df[df["name"].astype(str).isin(names)].copy()
        if df_view.empty:
            return {
                "shortlist": [],
                "suggestions": [],
                "message": "Selected compounds were not found in the dataset.",
                "scope": scope,
                "scope_note": scope_msg,
                "preset_choices": preset_choices(df),
            }
    else:
        df_view = df

    picks = shortlist_candidates(
        df=df_view,
        goal_text=goal_text,
        bitter_thresh=bitter_thresh,
        min_cool=min_cool,
        top_k=top_k,
    )

    if picks:
        return {
            "shortlist": picks,
            "suggestions": [],
            "message": f"Found {len(picks)} candidates meeting (cool ≥ {min_cool}, bitter ≤ {bitter_thresh}).",
            "scope": scope,
            "scope_note": scope_msg,
            "preset_choices": preset_choices(df),
        }

    # Fallback: relax thresholds to suggest nearest options
    fallback = shortlist_candidates(
        df=df,
        goal_text=goal_text,
        bitter_thresh=1.0,
        min_cool=0.0,
        top_k=top_k,
    )
    msg = (
        "No compounds matched your thresholds. "
        "Here are the closest alternatives—consider raising Minimum Cool or relaxing Bitter threshold."
    )
    return {
        "shortlist": [],
        "suggestions": fallback,
        "message": msg,
        "scope": scope,
        "scope_note": scope_msg,
        "preset_choices": preset_choices(df),
    }

# ---------------- Back-compat shim so older UI code keeps working ----------------
def _safe_file(p: str) -> str:
    try:
        return str(ensure_local_file(p))
    except Exception:
        return "(unresolved)"

try:
    INDEX_DIR = str(ensure_local_dir(FAISS_INDEX_DIR)) if FAISS_INDEX_DIR else "(unresolved)"
except Exception:
    INDEX_DIR = "(unresolved)"

DATASET = _safe_file(FLAVOR_DATASET) if FLAVOR_DATASET else "(unresolved)"
MODEL_PATH = _safe_file(TASTE_MODEL_PATH) if TASTE_MODEL_PATH else "(unresolved)"
LABEL_BINARIZER_PATH = _safe_file(TASTE_LABEL_BINARIZER) if TASTE_LABEL_BINARIZER else "(unresolved)"

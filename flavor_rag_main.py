
import os, re
from pathlib import Path
from urllib.parse import urlparse
from typing import Tuple, Optional, Dict, Any, List

import joblib
import pandas as pd

try:
    import boto3
    from botocore.exceptions import NoCredentialsError, ClientError
except Exception:
    boto3 = None

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

# ---------- Config ----------
TASTE_MODEL_PATH       = os.getenv("TASTE_MODEL_PATH",       "")
TASTE_LABEL_BINARIZER  = os.getenv("TASTE_LABEL_BINARIZER",  "")
FLAVOR_DATASET         = os.getenv("FLAVOR_DATASET",         "")
FAISS_INDEX_DIR        = os.getenv("FAISS_INDEX_DIR",        "")

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
        raise RuntimeError("boto3 not available. pip install boto3; configure creds via 'aws configure'.")

def _s3_client():
    _require_boto3()
    try:
        return boto3.client("s3")
    except NoCredentialsError:
        raise RuntimeError("AWS credentials not found. Run 'aws configure'.")

def ensure_local_file(path_or_s3: str, prefer_name: Optional[str] = None) -> Path:
    if not path_or_s3:
        raise FileNotFoundError("Empty path provided for artifact.")
    if _is_s3_uri(path_or_s3):
        bucket, key = _parse_s3_uri(path_or_s3)
        filename = prefer_name or Path(key).name
        local_path = CACHE_DIR / filename
        if not local_path.exists():
            s3 = _s3_client()
            local_path.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(bucket, key, str(local_path))
        return local_path
    p = Path(path_or_s3).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Local file not found: {p}")
    return p

def ensure_local_dir(dir_or_s3_prefix: str) -> Path:
    if not dir_or_s3_prefix:
        raise FileNotFoundError("Empty directory/prefix provided for index.")
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
                # prefix might actually be a single file
                file_target = local_dir / Path(prefix).name
                s3.download_file(bucket, prefix, str(file_target))
        return local_dir
    d = Path(dir_or_s3_prefix).expanduser().resolve()
    if not d.exists():
        raise FileNotFoundError(f"Local directory not found: {d}")
    return d

# ---------- Loaders ----------
def load_taste_model():
    model_local = ensure_local_file(TASTE_MODEL_PATH, prefer_name="taste_model_randomforest.joblib")
    mlb_local   = ensure_local_file(TASTE_LABEL_BINARIZER, prefer_name="taste_label_binarizer.joblib")
    rf  = joblib.load(str(model_local))
    mlb = joblib.load(str(mlb_local))
    return rf, mlb

def load_flavor_dataset() -> pd.DataFrame:
    local = ensure_local_file(FLAVOR_DATASET)
    if local.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(local)
    else:
        df = pd.read_csv(local)
    # Normalize common columns
    # Expected columns: name, aroma (string), taste (string), cool_score, bitter_prob (float-ish)
    # If arrays are stored as JSON-ish strings, keep as raw text for regex matching.
    for col in ["name", "aroma", "taste"]:
        if col not in df.columns:
            # best-effort aliases
            if col == "name":
                for alt in ["Name", "compound", "Compound"]:
                    if alt in df.columns:
                        df[col] = df[alt]
                        break
            elif col == "aroma":
                for alt in ["Aroma", "aroma_notes", "aroma_list"]:
                    if alt in df.columns:
                        df[col] = df[alt]
                        break
            elif col == "taste":
                for alt in ["Taste", "taste_notes", "taste_list"]:
                    if alt in df.columns:
                        df[col] = df[alt]
                        break
            if col not in df.columns:
                df[col] = ""
    if "cool_score" not in df.columns:
        # derive a naive cool score from keywords if missing
        def _cool_from_text(a, t):
            text = f"{a} {t}".lower()
            return 1.0 if "cool" in text or "menthol" in text or "mint" in text else 0.0
        df["cool_score"] = [ _cool_from_text(a, t) for a, t in zip(df["aroma"].astype(str), df["taste"].astype(str)) ]
    if "bitter_prob" not in df.columns:
        # default unknown bitterness to mid value
        df["bitter_prob"] = 0.5
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
    text = goal_text.lower()
    for scope, kws in _SCOPE_KEYWORDS.items():
        if any(k in text for k in kws):
            return scope
    return "generic"

# ---------- Candidate scoring ----------
def _keyword_score(text: str, goal_text: str) -> float:
    if not text:
        return 0.0
    text = str(text).lower()
    words = [w for w in re.findall(r"[a-z]+", goal_text.lower()) if len(w) > 2]
    if not words:
        return 0.0
    hits = sum(1 for w in words if w in text)
    return hits / max(4, len(words))  # dampen long prompts

def shortlist_candidates(
    df: pd.DataFrame,
    goal_text: str,
    bitter_thresh: float = 0.40,
    min_cool: float = 0.15,
    top_k: int = 12,
) -> List[Dict[str, Any]]:
    scope = _infer_scope(goal_text)
    # bias keywords per scope
    extra = []
    if scope == "dessert":
        extra = ["vanilla", "creamy", "sweet", "buttery", "cake"]
    elif scope == "herbal":
        extra = ["herbal", "camphor", "eucalyptus", "sage", "rosemary"]
    elif scope == "mint":
        extra = ["mint", "menthol", "menthyl", "peppermint", "cool"]

    def score_row(row):
        aroma = str(row["aroma"])
        taste = str(row["taste"])
        txt = f"{aroma} {taste}"
        kcore = _keyword_score(txt, goal_text)
        kextra = _keyword_score(txt, " ".join(extra)) if extra else 0.0
        cool = float(row.get("cool_score", 0.0) or 0.0)
        bitter = float(row.get("bitter_prob", 0.5) or 0.5)
        # score: prefer cool, penalize bitter, plus keyword alignment
        return 0.6 * cool - 0.5 * bitter + 0.6 * kcore + 0.3 * kextra

    # filter first by thresholds
    filtered = df[
        (df["bitter_prob"].astype(float) <= bitter_thresh) &
        (df["cool_score"].astype(float) >= min_cool)
    ].copy()

    # If filtered is empty, consider full df for suggestions later
    target = filtered if len(filtered) else df

    target["__score__"] = target.apply(score_row, axis=1)
    ranked = target.sort_values("__score__", ascending=False).head(int(top_k))

    results = []
    for _, r in ranked.iterrows():
        results.append({
            "name": r.get("name", ""),
            "aroma": r.get("aroma", ""),
            "taste": r.get("taste", ""),
            "cool_score": float(r.get("cool_score", 0.0) or 0.0),
            "bitter_prob": float(r.get("bitter_prob", 0.5) or 0.5),
            "score": float(r["__score__"]),
            "reason": f"cool={float(r.get('cool_score',0)):0.2f}, bitter={float(r.get('bitter_prob',0.5)):0.2f}, match≈{_keyword_score(f'{r.get('aroma','')} {r.get('taste','')}', goal_text):0.2f}",
        })
    return results

# ---------- Preset (“pretrained”) choices ----------
def preset_choices(df: pd.DataFrame, limit: int = 25) -> List[str]:
    # top cool / low bitter “safe” starting points
    df2 = df.copy()
    df2["cool_score"] = df2["cool_score"].astype(float)
    df2["bitter_prob"] = df2["bitter_prob"].astype(float)
    df2["__rank__"] = (0.8 * df2["cool_score"] - 0.6 * df2["bitter_prob"])
    out = (
        df2.sort_values("__rank__", ascending=False)
           .head(int(limit))
           .get("name", pd.Series([], dtype=str))
           .dropna()
           .astype(str)
           .tolist()
    )
    return out

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

# ---------- Main facade the UI calls ----------
def answer(
    goal_text: str,
    top_k: int = 12,
    bitter_thresh: float = 0.40,
    min_cool: float = 0.15,
    exact_compounds: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Returns a shortlist and helpful messages. Falls back to suggestions if none pass thresholds.
    """
    df = load_flavor_dataset()

    scope = _infer_scope(goal_text)
    scope_msg = ""
    if "ice cream" in goal_text.lower() or "cake" in goal_text.lower():
        scope_msg = "Dessert-like profile requested; prioritizing creamy/vanilla/sweet notes."

    # If user picked exact compounds, filter df first to those
    if exact_compounds:
        df_view = df[df["name"].astype(str).isin([str(x) for x in exact_compounds])].copy()
        if df_view.empty:
            return {
                "shortlist": [],
                "suggestions": [],
                "message": "Selected compounds were not found in the dataset.",
                "scope": scope,
                "scope_note": scope_msg,
                "preset_choices": preset_choices(df)
            }
    else:
        df_view = df

    picks = shortlist_candidates(df_view, goal_text, bitter_thresh, min_cool, top_k)

    if picks:
        return {
            "shortlist": picks,
            "suggestions": [],
            "message": f"Found {len(picks)} candidates meeting (cool ≥ {min_cool}, bitter ≤ {bitter_thresh}).",
            "scope": scope,
            "scope_note": scope_msg,
            "preset_choices": preset_choices(df)
        }

    # No items met thresholds → provide suggestions (best-effort)
    fallback = shortlist_candidates(df, goal_text, bitter_thresh=1.0, min_cool=0.0, top_k=top_k)
    msg = (
        "No compounds matched your thresholds. "
        "Here are the nearest options—consider raising min cool or relaxing bitterness."
    )
    return {
        "shortlist": [],
        "suggestions": fallback,
        "message": msg,
        "scope": scope,
        "scope_note": scope_msg,
        "preset_choices": preset_choices(df)
    }

# ---------------- Back-compat shim for existing UI ----------------
def _safe(path: str) -> str:
    try:
        return str(ensure_local_file(path))
    except Exception:
        return "(unresolved)"

try:
    INDEX_DIR = str(ensure_local_dir(FAISS_INDEX_DIR)) if FAISS_INDEX_DIR else "(unresolved)"
except Exception:
    INDEX_DIR = "(unresolved)"
DATASET = _safe(FLAVOR_DATASET) if FLAVOR_DATASET else "(unresolved)"
MODEL_PATH = _safe(TASTE_MODEL_PATH) if TASTE_MODEL_PATH else "(unresolved)"
LABEL_BINARIZER_PATH = _safe(TASTE_LABEL_BINARIZER) if TASTE_LABEL_BINARIZER else "(unresolved)"

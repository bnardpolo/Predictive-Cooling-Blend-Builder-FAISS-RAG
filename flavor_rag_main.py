# =========================================================
# genai_flavor_rag.py  — Retrieval + Bitter Guard + LLM
# =========================================================
import os, re, ast, json
from typing import List, Dict, Any, Tuple

import pandas as pd
import numpy as np
import joblib

from dotenv import load_dotenv
load_dotenv()  # for OPENAI_API_KEY

# -------------------- CONFIG --------------------
BASE_DIR      = r"C:\Users\enhan\Desktop\Procter & Gamble"
DATA_BASENAME = os.path.join(BASE_DIR, "flavor_kb_master")  # extension auto-detected
MODEL_PATH    = os.path.join(BASE_DIR, "taste_model_randomforest.joblib")
MLB_PATH      = os.path.join(BASE_DIR, "taste_label_binarizer.joblib")
INDEX_DIR     = os.path.join(BASE_DIR, "faiss_index")
BITTER_THRESH = 0.40

os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

# ---------------- File helpers ----------------
def _resolve_data_path(basename: str) -> str:
    candidates = [basename]
    root, ext = os.path.splitext(basename)
    if ext == "":
        candidates += [root + ".csv", root + ".xlsx", root + ".xls"]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Data file not found. Tried: {candidates}")

def _read_table_auto(basename: str) -> pd.DataFrame:
    path = _resolve_data_path(basename)
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path)  # requires: pip install openpyxl

# --------------- Normalization helpers ---------------
_ALIAS_MAP = {
    "name":        ["name", "compound", "molecule", "ingredient", "material", "compound_name"],
    "aroma":       ["aroma", "aroma_clean", "aromas", "descriptor", "descriptors", "notes", "note"],
    "smiles":      ["smiles", "smile", "smiles_string"],
    "description": ["description", "desc", "details", "summary"],
    "taste":       ["taste", "tastes", "taste_labels", "labels", "label", "tag", "tags"],
}

def _to_lower_cols(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    df2.columns = [str(c).strip().lower() for c in df2.columns]
    return df2

def _ensure_canonical_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = _to_lower_cols(df)
    for canon, aliases in _ALIAS_MAP.items():
        if canon in df.columns:
            continue
        for a in aliases:
            if a in df.columns:
                df[canon] = df[a]
                break
        if canon not in df.columns:
            df[canon] = ""  # empty fallback
    return df

def _safe_str(x: Any) -> str:
    return "" if x is None or (isinstance(x, float) and pd.isna(x)) else str(x)

# --------------- Taste model helpers ---------------
def _parse_list_like(x) -> List[str]:
    if x is None or (isinstance(x, float) and pd.isna(x)): return []
    if isinstance(x, list): return [str(i).strip() for i in x if str(i).strip()]
    if isinstance(x, str):
        s = x.strip()
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
            try:
                val = ast.literal_eval(s)
                if isinstance(val, (list, tuple)):
                    return [str(i).strip() for i in val if str(i).strip()]
            except Exception:
                pass
        parts = re.split(r"[;,]\s*", s)
        return [p.strip() for p in parts if p.strip()]
    return [str(x).strip()]

def load_taste_model() -> Tuple[Any, Any]:
    rf = joblib.load(MODEL_PATH)
    mlb = joblib.load(MLB_PATH)
    return rf, mlb

def predict_taste_probs(model, mlb, name: str, aroma: str, smiles: str) -> Dict[str, float]:
    text = f"{_safe_str(name)} {_safe_str(aroma)} {_safe_str(smiles)}".strip()
    proba_matrix = model.predict_proba([text])
    probs = [float(p) for p in proba_matrix[0]]
    return dict(zip(mlb.classes_.tolist(), probs))

def passes_bitter_guard(probs: Dict[str, float], threshold: float = BITTER_THRESH) -> bool:
    return probs.get("Bitter", 0.0) < threshold

# ---------------- LangChain / FAISS ----------------
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.schema import SystemMessage, HumanMessage, AIMessage

def _row_to_doc(row: Dict[str, Any]) -> Document:
    name   = _safe_str(row.get("name", ""))
    aromaV = row.get("aroma", "")
    aroma  = " ".join(_parse_list_like(aromaV)) if not isinstance(aromaV, str) else _safe_str(aromaV)
    smiles = _safe_str(row.get("smiles", ""))
    desc   = _safe_str(row.get("description", ""))
    taste  = row.get("taste", "")
    taste_list = _parse_list_like(taste) if not isinstance(taste, list) else taste

    blocks = []
    if name:   blocks.append(f"Name: {name}")
    if aroma:  blocks.append(f"Aroma: {aroma}")
    if smiles: blocks.append(f"SMILES: {smiles}")
    if desc:   blocks.append(f"Description: {desc}")
    if taste_list: blocks.append(f"Known tastes: {', '.join(taste_list)}")
    text = "\n".join(blocks).strip() or json.dumps({k: _safe_str(v) for k,v in row.items()}, ensure_ascii=False)

    meta = {"name": name, "aroma": aroma, "smiles": smiles, "description": desc, "taste_labels": taste_list}
    return Document(page_content=text, metadata=meta)

def build_or_load_vectorstore(df: pd.DataFrame) -> FAISS:
    df = _ensure_canonical_columns(df)
    total = len(df)
    print(f"[Index] rows={total} | name>0: {int((df['name'].astype(str).str.strip()!='').sum())} | "
          f"aroma>0: {int((df['aroma'].astype(str).str.strip()!='').sum())} | smiles>0: {int((df['smiles'].astype(str).str.strip()!='').sum())}")
    docs = [_row_to_doc(r) for _, r in df.iterrows()]
    docs = [d for d in docs if d.page_content.strip()]
    if not docs: raise ValueError("No documents to index. Check dataset columns.")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vs = FAISS.from_documents(docs, embeddings)
    vs.save_local(INDEX_DIR)
    return vs

def load_vectorstore() -> FAISS:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)

def retrieve_mmr(vs: FAISS, query: str, k: int = 12, fetch_k: int = 24) -> List[Document]:
    retriever = vs.as_retriever(search_type="mmr", search_kwargs={"k": k, "fetch_k": fetch_k, "lambda_mult": 0.5})
    return retriever.invoke(query)

# ---------------- LLM compose ----------------
SYSTEM = """You are a flavor scientist. Use ONLY the provided SAFE candidates (already filtered for Bitter).
Combine 1–3 candidates to match the user's goal. Be concise and pragmatic.
Return bullets with % ranges and a one-line rationale. Do not warn about bitterness."""

def compose_with_llm(query: str, safe_items: List[Dict[str, Any]], threshold: float) -> str:
    # Build context (works for both LLM and fallback)
    ctx_lines = [f"User goal: {query}", f"Bitter threshold: {threshold}"]
    for i, item in enumerate(safe_items[:8], 1):
        m = item["doc"].metadata
        ctx_lines.append(
            f"\nCandidate {i}:\n"
            f"- Name: {m.get('name','')}\n"
            f"- Aroma: {m.get('aroma','')}\n"
            f"- SMILES: {m.get('smiles','')}\n"
            f"- Known tastes: {', '.join(m.get('taste_labels', []))}\n"
            f"- Predicted probs: {json.dumps(item['probs'])}"
        )
    context = "\n".join(ctx_lines)

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        names = [it["doc"].metadata.get("name","") for it in safe_items[:3]]
        return (
            f"(No LLM key) Suggested blend from safe picks: {', '.join([n for n in names if n]) or 'N/A'}.\n"
            f"Rationale: chosen for the target profile while keeping Bitter<{threshold}. "
            f"Set OPENAI_API_KEY to get a polished composition."
        )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    msgs = [SystemMessage(content=SYSTEM),
            HumanMessage(content=f"{context}\n\nPropose the blend now: bullets + brief rationale.")]
    resp = llm.invoke(msgs)
    return resp.content if isinstance(resp, AIMessage) else str(resp)

# ---------------- Public API ----------------
def answer(query: str, top_k: int = 12) -> Dict[str, Any]:
    # Load model + KB (build index if missing)
    rf, mlb = load_taste_model()
    try:
        vs = load_vectorstore()
    except Exception:
        df = _read_table_auto(DATA_BASENAME)
        vs = build_or_load_vectorstore(df)

    # Retrieve + guard
    docs = retrieve_mmr(vs, query, k=top_k, fetch_k=max(2*top_k, 24))
    safe = []
    for d in docs:
        m = d.metadata or {}
        probs = predict_taste_probs(rf, mlb, m.get("name",""), m.get("aroma",""), m.get("smiles",""))
        if passes_bitter_guard(probs, BITTER_THRESH):
            safe.append({"doc": d, "probs": probs})

    if not safe:
        # Diagnostics: show nearest with their bitter scores
        closest = []
        for d in docs[:5]:
            m = d.metadata or {}
            p = predict_taste_probs(rf, mlb, m.get("name",""), m.get("aroma",""), m.get("smiles",""))
            closest.append({"name": m.get("name",""), "aroma": m.get("aroma",""), "bitter": p.get("Bitter", 0.0), "probs": p})
        return {"ok": False, "message": f"No candidates passed Bitter<{BITTER_THRESH}. Try raising threshold or revising the query.", "closest": closest}

    blend_text = compose_with_llm(query, safe, BITTER_THRESH)
    shortlist = [{
        "name": it["doc"].metadata.get("name",""),
        "aroma": it["doc"].metadata.get("aroma",""),
        "smiles": it["doc"].metadata.get("smiles",""),
        "probs": it["probs"]
    } for it in safe[:8]]

    return {"ok": True, "blend_text": blend_text, "shortlist": shortlist}

# ---------------- CLI demo ----------------
if __name__ == "__main__":
    print(f"Resolving dataset from: {_resolve_data_path(DATA_BASENAME)}")
    demo_query = "Design a cooling, minty mouthfeel for a sugar-free gum with a clean aftertaste."
    out = answer(demo_query, top_k=12)
    print(json.dumps(out, indent=2))

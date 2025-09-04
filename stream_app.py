# stream_app.py
# Streamlit UI for genai_flavor_rag.py — clean look, no logos/icons.
# - Intent picker + freeform goal
# - Working "Add to Blend" (session_state + on_click)
# - Weighted blend metrics
# - Save (CSV+JSON) and Download; Load a saved recipe
# - Filters: hide sweeteners, min Cool
# - Reindex / Retrain controls
import genai_flavor_rag as rag

import os
import json
import math
import time
import shutil
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
import streamlit as st

# ----- import the core pipeline -----
try:
    import genai_flavor_rag as rag
except Exception as e:
    st.error("Couldn't import genai_flavor_rag.py. Place this app in the same folder.")
    st.exception(e)
    st.stop()

APP_DIR = Path.cwd()
BLENDS_DIR = APP_DIR / "blends"
BLENDS_DIR.mkdir(parents=True, exist_ok=True)

# ===== Helpers =====
def ensure_state():
    ss = st.session_state
    ss.setdefault("builder_rows", [])         # list of dicts: name, aroma, smiles, probs, pct
    ss.setdefault("last_out", None)           # last pipeline response
    ss.setdefault("saved_name", "")           # default save name
    ss.setdefault("intent", "Max cooling")    # default intent

def _item_key(item: Dict[str, Any]) -> str:
    return f"{(item.get('name') or '').strip()}||{(item.get('smiles') or '').strip()}"

def add_to_builder_cb(item: Dict[str, Any], key: str):
    ensure_state()
    # avoid duplicates using composite key
    keys = {_item_key(r) for r in st.session_state.builder_rows}
    if key not in keys:
        st.session_state.builder_rows.append({**item, "pct": 0.0})

def clear_builder():
    st.session_state.builder_rows = []

def total_pct() -> float:
    return float(sum(float(r.get("pct", 0) or 0) for r in st.session_state.builder_rows))

def normalize_builder_to_100():
    s = total_pct() or 1.0
    for r in st.session_state.builder_rows:
        r["pct"] = round((r.get("pct", 0.0) or 0.0) * 100.0 / s, 2)

def export_builder_df() -> pd.DataFrame:
    rows = []
    for r in st.session_state.builder_rows:
        probs = r.get("probs", {})
        rows.append({
            "Name": r.get("name",""),
            "Aroma": r.get("aroma",""),
            "SMILES": r.get("smiles",""),
            "Pct": r.get("pct", 0.0),
            "Bitter": probs.get("Bitter", 0.0),
            "Cool": probs.get("Cool", 0.0),
            "Sweet": probs.get("Sweet", 0.0),
            "Pungent": probs.get("Pungent", 0.0),
        })
    return pd.DataFrame(rows)

def weighted_metric(metric: str) -> float:
    # percentage-weighted mean of a metric across builder rows
    tot = total_pct()
    if tot <= 0:
        return 0.0
    acc = 0.0
    for r in st.session_state.builder_rows:
        pct = float(r.get("pct", 0) or 0)
        acc += pct * float(r.get("probs", {}).get(metric, 0.0))
    return acc / max(tot, 1e-9)

def human_pct(x: float) -> str:
    try:
        return f"{float(x):.0f}%"
    except Exception:
        return "-"

def save_recipe_files(base_name: str) -> Dict[str, str]:
    base = base_name.strip() or f"blend_{int(time.time())}"
    safe = "".join(c for c in base if c.isalnum() or c in "-_ ").strip().replace(" ", "_")
    csv_path = BLENDS_DIR / f"{safe}.csv"
    json_path = BLENDS_DIR / f"{safe}.json"

    df = export_builder_df()
    df.to_csv(csv_path, index=False)

    payload = {
        "name": safe,
        "created_ts": int(time.time()),
        "items": st.session_state.builder_rows,
        "metrics": {
            "Bitter": weighted_metric("Bitter"),
            "Cool": weighted_metric("Cool"),
            "Sweet": weighted_metric("Sweet"),
            "Pungent": weighted_metric("Pungent"),
            "TotalPct": total_pct(),
        }
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {"csv": str(csv_path), "json": str(json_path)}

def load_recipe_file(path: Path):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        items = data.get("items", [])
        # scrub keys
        cleaned = []
        for it in items:
            cleaned.append({
                "name": it.get("name",""),
                "aroma": it.get("aroma",""),
                "smiles": it.get("smiles",""),
                "probs": it.get("probs", {}),
                "pct": float(it.get("pct", 0.0) or 0.0),
            })
        st.session_state.builder_rows = cleaned
        st.success(f"Loaded {path.name}")
    except Exception as e:
        st.error(f"Failed to load {path.name}")
        st.exception(e)

# ===== Page setup =====
st.set_page_config(page_title="Flavor Builder", layout="wide")
ensure_state()

# ===== Sidebar =====
st.sidebar.title("Settings")

try:
    resolved_path = rag._resolve_data_path(rag.DATA_BASENAME)
except Exception:
    resolved_path = "(unresolved)"

st.sidebar.markdown(f"**Dataset**: `{resolved_path}`")
st.sidebar.markdown(f"**Index dir**: `{rag.INDEX_DIR}`")

# Core controls
bitter_threshold = st.sidebar.slider(
    "Bitter threshold", 0.0, 1.0,
    float(getattr(rag, "BITTER_THRESH", 0.40)), 0.01
)
top_k = st.sidebar.slider("Top-K retrieval", 4, 40, 12, 1)

# Filters
hide_sweeteners = st.sidebar.checkbox("Hide sweeteners", value=True)
min_cool = st.sidebar.slider("Minimum Cool score (filter display)", 0.0, 1.0, 0.15, 0.01)
EXCLUDE_NAMES = ["xylitol", "sucralose", "stevia", "acesulfame", "aspartame", "saccharin"]

# Maintenance
colA, colB = st.sidebar.columns(2)
reindex_clicked = colA.button("Reindex")
retrain_clicked = colB.button("Retrain")

if reindex_clicked:
    try:
        shutil.rmtree(rag.INDEX_DIR, ignore_errors=True)
        st.sidebar.success("Index cleared. It will rebuild on next run.")
    except Exception as e:
        st.sidebar.error("Couldn't clear index.")
        st.sidebar.exception(e)

if retrain_clicked:
    try:
        if os.path.exists(rag.MODEL_PATH): os.remove(rag.MODEL_PATH)
        if os.path.exists(rag.MLB_PATH): os.remove(rag.MLB_PATH)
        shutil.rmtree(rag.INDEX_DIR, ignore_errors=True)
        st.sidebar.success("Model + index cleared. They will retrain/rebuild on next run.")
    except Exception as e:
        st.sidebar.error("Couldn't clear model/index.")
        st.sidebar.exception(e)

# apply live threshold to pipeline
rag.BITTER_THRESH = float(bitter_threshold)

# ===== Main header =====
st.title("Flavor Blend Builder")

# Intent picker + freeform goal
INTENTS = {
    "Max cooling": "Maximize cooling intensity while keeping bitterness low; emphasize menthol/menthyl esters.",
    "Balanced mint": "Mint-forward profile with balanced cooling and low aftertaste harshness.",
    "Clean aftertaste": "Fresh mint with a very clean finish; avoid lingering bitterness and harsh notes.",
    "Herbal mint": "Minty base with herbal/camphor facets; keep bitterness controlled.",
}
intent = st.selectbox("Blend intent", list(INTENTS.keys()), index=list(INTENTS.keys()).index(st.session_state.intent))
st.session_state.intent = intent

default_goal = INTENTS[intent]
goal = st.text_area("Design goal (editable)", value=default_goal, height=80)

go = st.button("Suggest Blend", type="primary")
st.markdown("---")

def _passes_display_filters(item) -> bool:
    name = (item.get("name","") or "").lower()
    if hide_sweeteners and any(t in name for t in EXCLUDE_NAMES):
        return False
    probs = item.get("probs", {})
    return probs.get("Cool", 0.0) >= float(min_cool)

# ===== Pipeline trigger =====
if go:
    with st.spinner("Retrieving candidates and composing a blend…"):
        try:
            # combine intent + goal
            query = f"{goal}".strip()
            out = rag.answer(query, top_k=int(top_k))
            st.session_state.last_out = out
        except Exception as e:
            st.error("Pipeline error while generating suggestion.")
            st.exception(e)
            st.stop()

out = st.session_state.last_out

# ===== Results =====
if not out:
    st.info("Enter your goal and click Suggest Blend.")
else:
    if not out.get("ok"):
        st.warning(out.get("message", "No result"))
        closest = out.get("closest", [])
        if closest:
            st.subheader("Nearest candidates (diagnostics)")
            df = pd.DataFrame([{
                "Name": c.get("name",""),
                "Aroma": c.get("aroma",""),
                "Bitter": c.get("bitter", 0.0),
                "Cool": c.get("probs",{}).get("Cool", 0.0),
                "Sweet": c.get("probs",{}).get("Sweet", 0.0),
            } for c in closest])
            st.dataframe(df, use_container_width=True, height=240)
    else:
        st.subheader("Suggested Blend")
        st.markdown(out.get("blend_text","").strip())

        st.subheader("Shortlist")
        raw_items = out.get("shortlist", [])
        items = [it for it in raw_items if _passes_display_filters(it)]

        if not items:
            st.info("No shortlist items after applying filters. Try lowering the minimum Cool score or unchecking Hide sweeteners.")
        else:
            cards_per_row = 3
            rows = math.ceil(len(items) / cards_per_row)
            for r in range(rows):
                cols = st.columns(cards_per_row, gap="large")
                for i in range(cards_per_row):
                    idx = r * cards_per_row + i
                    if idx >= len(items):
                        break
                    it = items[idx]
                    with cols[i]:
                        nm = it.get("name") or "Unnamed candidate"
                        aroma = it.get("aroma","")
                        if isinstance(aroma, list):
                            aroma = ", ".join(aroma)
                        st.markdown(f"**{nm}**")
                        st.caption(aroma if aroma else "—")
                        probs = it.get("probs", {})
                        mcols = st.columns(4)
                        mcols[0].metric("Bitter", f"{probs.get('Bitter',0.0):.2f}")
                        mcols[1].metric("Cool",   f"{probs.get('Cool',0.0):.2f}")
                        mcols[2].metric("Sweet",  f"{probs.get('Sweet',0.0):.2f}")
                        mcols[3].metric("Pungent",f"{probs.get('Pungent',0.0):.2f}")

                        # robust add-to-blend button via on_click callback
                        candidate_payload = {
                            "name": it.get("name",""),
                            "aroma": aroma,
                            "smiles": it.get("smiles",""),
                            "probs": probs
                        }
                        key = _item_key(candidate_payload) or f"idx_{idx}"
                        st.button(
                            "Add to Blend",
                            key=f"add_{key}",
                            on_click=add_to_builder_cb,
                            args=(candidate_payload, key)
                        )

st.markdown("---")

# ===== Builder =====
st.subheader("Your Blend")
ensure_state()

if not st.session_state.builder_rows:
    st.info("Click Add to Blend on any shortlist item to start building.")
else:
    # header
    header_cols = st.columns([3, 5, 2, 2, 1, 1])
    header_cols[0].markdown("**Name**")
    header_cols[1].markdown("**Aroma**")
    header_cols[2].markdown("**Bitter**")
    header_cols[3].markdown("**Cool**")
    header_cols[4].markdown("**%**")
    header_cols[5].markdown("**Remove**")

    # rows
    for i, r in enumerate(st.session_state.builder_rows):
        c = st.columns([3, 5, 2, 2, 1, 1])
        c[0].markdown(r.get("name") or "Unnamed")
        c[1].caption(r.get("aroma") or "—")
        c[2].markdown(f"{r.get('probs',{}).get('Bitter',0.0):.2f}")
        c[3].markdown(f"{r.get('probs',{}).get('Cool',0.0):.2f}")
        st.session_state.builder_rows[i]["pct"] = c[4].number_input(
            label=f"%_{i}",
            label_visibility="collapsed",
            min_value=0.0, max_value=100.0, step=1.0,
            value=float(r.get("pct", 0.0) or 0.0),
            key=f"pct_{i}"
        )
        if c[5].button("X", key=f"rm_{i}"):
            st.session_state.builder_rows.pop(i)
            st.experimental_rerun()

    # totals + metrics
    tot = total_pct()
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total", human_pct(tot))
    m2.metric("Blend Bitter", f"{weighted_metric('Bitter'):.2f}")
    m3.metric("Blend Cool",   f"{weighted_metric('Cool'):.2f}")
    m4.metric("Blend Sweet",  f"{weighted_metric('Sweet'):.2f}")
    m5.metric("Blend Pungent",f"{weighted_metric('Pungent'):.2f}")

    if abs(tot - 100.0) > 0.01:
        st.warning("Target total is 100%. Adjust or click Normalize.")

    a1, a2, a3, a4 = st.columns(4)
    if a1.button("Normalize to 100%"):
        normalize_builder_to_100()
        st.experimental_rerun()
    if a2.button("Clear Blend"):
        clear_builder()
        st.experimental_rerun()

    # save / download / load
    st.markdown("#### Save or Load")
    st.session_state.saved_name = st.text_input("Blend name", value=st.session_state.saved_name or "my_blend")

    bcols = st.columns(3)
    if bcols[0].button("Save to server"):
        paths = save_recipe_files(st.session_state.saved_name)
        st.success(f"Saved CSV and JSON in {BLENDS_DIR}")
        st.caption(paths["csv"])
        st.caption(paths["json"])

    # on-demand download
    df_export = export_builder_df()
    if df_export is not None:
        bcols[1].download_button(
            "Download CSV",
            df_export.to_csv(index=False).encode("utf-8"),
            file_name=f"{(st.session_state.saved_name or 'blend')}.csv",
            mime="text/csv"
        )
        # download JSON payload too
        payload = {
            "name": st.session_state.saved_name or "blend",
            "items": st.session_state.builder_rows,
            "metrics": {
                "Bitter": weighted_metric("Bitter"),
                "Cool": weighted_metric("Cool"),
                "Sweet": weighted_metric("Sweet"),
                "Pungent": weighted_metric("Pungent"),
                "TotalPct": total_pct(),
            }
        }
        bcols[2].download_button(
            "Download JSON",
            json.dumps(payload, indent=2).encode("utf-8"),
            file_name=f"{(st.session_state.saved_name or 'blend')}.json",
            mime="application/json"
        )

    # load from saved JSON
    with st.expander("Load a saved recipe"):
        files = sorted(BLENDS_DIR.glob("*.json"))
        if not files:
            st.caption("No saved recipes yet.")
        else:
            sel = st.selectbox("Pick a saved recipe", [f.name for f in files])
            if st.button("Load recipe"):
                load_recipe_file(BLENDS_DIR / sel)

# ===== Footer =====
st.caption("Local prototype — data remains on this machine.")

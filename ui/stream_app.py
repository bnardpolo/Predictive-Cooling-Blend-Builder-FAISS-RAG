# ui/stream_app.py
import os
import sys
from pathlib import Path
import importlib.util
import streamlit as st

# ---------- import pipeline (src/genai_flavor_rag.py) ----------
ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
PIPELINE_FILE = SRC_DIR / "genai_flavor_rag.py"

def _import_pipeline():
    if not PIPELINE_FILE.exists():
        raise FileNotFoundError(f"Pipeline file not found: {PIPELINE_FILE}")
    spec = importlib.util.spec_from_file_location("genai_flavor_rag", str(PIPELINE_FILE))
    mod = importlib.util.module_from_spec(spec)  # type: ignore
    sys.modules["genai_flavor_rag"] = mod
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore
    return mod

try:
    rag = _import_pipeline()
except Exception as e:
    st.error(f"Failed to load pipeline: {e}")
    st.stop()

st.set_page_config(page_title="Flavor Blend Builder", page_icon="🧪", layout="wide")

# ---------- light custom styling ----------
st.markdown("""
<style>
/* hide streamlit branding bits */
#MainMenu, header {visibility: hidden;}
footer {visibility: hidden;}
/* page width + typography */
.block-container {max-width: 1100px;}
h1, h2, h3 {font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial;}
/* card-ish feel for result rows */
.result-card {
  padding: .6rem .8rem; border: 1px solid #E8EAED; border-radius: 14px; margin-bottom: .5rem;
  background: #fff;
}
.smallcap {color:#6b7280; font-size: 0.86rem;}
.addbtn button {width:100%;}
.badge {display:inline-block; padding:2px 8px; border-radius:999px; background:#F3F4F6; margin-right:6px;}
</style>
""", unsafe_allow_html=True)

# ---------- state ----------
if "blend_items" not in st.session_state:
    st.session_state["blend_items"] = []
if "last_out" not in st.session_state:
    st.session_state["last_out"] = None
if "picked_compounds" not in st.session_state:
    st.session_state["picked_compounds"] = []

# ---------- sidebar (Settings) ----------
with st.sidebar:
    st.markdown("### Settings")

    # thresholds
    st.session_state["bitter_thresh"] = st.slider("Bitter threshold", 0.0, 1.0, 0.40, 0.01)
    st.session_state["top_k"] = st.slider("Top-K retrieval", 4, 40, 12, 1)
    st.session_state["min_cool"] = st.slider("Minimum Cool score", 0.0, 1.0, 0.15, 0.01)

    # presets (optional exact compounds)
    st.markdown("### Pick exact compounds (optional)")
    try:
        _df_for_presets = rag.load_flavor_dataset()
        _presets = rag.preset_choices(_df_for_presets, limit=50)
    except Exception:
        _presets = []
    st.multiselect("Start from presets", _presets, key="picked_compounds")

# ---------- header ----------
st.title("Flavor Blend Builder")
st.caption("Suggest cooling flavor candidates, filter by thresholds, and assemble a blend.")

# ---------- intent + goal ----------
col1, col2 = st.columns([2, 3], gap="large")
with col1:
    intent = st.selectbox(
        "Blend intent",
        ["Max cooling", "Herbal mint", "Dessert profile", "Custom"],
        index=0,
    )
with col2:
    default_goal = "Minty base with herbal/camphor facets; keep bitterness controlled."
    goal_text = st.text_area("Design goal (editable)", value=default_goal, key="saved_goal")

# ---------- action ----------
if st.button("Suggest Blend", type="primary", use_container_width=True):
    try:
        out = rag.answer(
            goal_text=st.session_state.get("saved_goal", ""),
            top_k=int(st.session_state.get("top_k", 12)),
            bitter_thresh=float(st.session_state.get("bitter_thresh", 0.40)),
            min_cool=float(st.session_state.get("min_cool", 0.15)),
            exact_compounds=st.session_state.get("picked_compounds") or None,
        )
        st.session_state["last_out"] = out
    except Exception as e:
        st.error(f"Pipeline error: {e}")

# ---------- results ----------
out = st.session_state.get("last_out")
if out:
    if out.get("scope_note"):
        st.info(out["scope_note"])
    if out.get("message"):
        st.success(out["message"])

    shortlist = out.get("shortlist") or []
    suggestions = out.get("suggestions") or []

    if shortlist:
        st.subheader("Shortlist")
        for i, r in enumerate(shortlist):
            name = r.get("name", "")
            cool = float(r.get("cool_score", 0.0))
            bitter = float(r.get("bitter_prob", 0.0))
            aroma = r.get("aroma") or ""
            taste = r.get("taste") or ""
            reason = r.get("reason") or ""
            c1, c2 = st.columns([6, 1], vertical_alignment="center")
            with c1:
                st.markdown(f"""
<div class="result-card">
  <div><strong>{name}</strong></div>
  <div class="smallcap">
    <span class="badge">cool {cool:.2f}</span>
    <span class="badge">bitter {bitter:.2f}</span>
  </div>
  <div class="smallcap">Aroma: {aroma} &nbsp;|&nbsp; Taste: {taste}</div>
  <div class="smallcap">{reason}</div>
</div>
""", unsafe_allow_html=True)
            with c2:
                if st.button("➕ Add", key=f"add_{i}"):
                    if name and name not in st.session_state["blend_items"]:
                        st.session_state["blend_items"].append(name)
    else:
        if suggestions:
            st.subheader("Closest alternatives")
            for i, r in enumerate(suggestions):
                name = r.get("name", "")
                cool = float(r.get("cool_score", 0.0))
                bitter = float(r.get("bitter_prob", 0.0))
                reason = r.get("reason") or ""
                c1, c2 = st.columns([6, 1], vertical_alignment="center")
                with c1:
                    st.markdown(f"""
<div class="result-card">
  <div><strong>{name}</strong></div>
  <div class="smallcap">
    <span class="badge">cool {cool:.2f}</span>
    <span class="badge">bitter {bitter:.2f}</span>
  </div>
  <div class="smallcap">{reason}</div>
</div>
""", unsafe_allow_html=True)
                with c2:
                    if st.button("➕ Add", key=f"sugg_add_{i}"):
                        if name and name not in st.session_state["blend_items"]:
                            st.session_state["blend_items"].append(name)
        else:
            st.info("No candidates available in dataset.")

st.divider()

# ---------- your blend ----------
st.subheader("Your Blend")
blend = [x for x in st.session_state["blend_items"] if x]
if blend:
    # chip-style list
    st.markdown(" ".join([f"<span class='badge'>{x}</span>" for x in blend]), unsafe_allow_html=True)
    colx, coly = st.columns([1,1])
    with colx:
        if st.button("Clear blend"):
            st.session_state["blend_items"] = []
    with coly:
        st.download_button(
            "Export names (txt)",
            data="\n".join(blend).encode("utf-8"),
            file_name="blend.txt",
            mime="text/plain"
        )
else:
    st.caption("Click **Add** on any shortlist item to start building.")

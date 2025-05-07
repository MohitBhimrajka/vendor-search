# app.py  – final polish + Gemini fix (2025‑05‑07 rev‑D)

import os
import json
import logging
import base64
from datetime import datetime
from pathlib import Path

import streamlit as st
import pandas as pd
import requests
import pycountry
from PIL import Image
from google import genai
from google.genai import types

# ───────────────────────── PAGE CONFIG ───────────────────────── #
st.set_page_config(
    page_title="Vendor Search & Recommendation",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ───────────────────────── THEME / GLOBAL CSS ───────────────────────── #
st.markdown(
    """
    <style>
    html,body,[data-testid="stAppViewContainer"],.stApp{background:#fff!important;}
    header,footer,[data-testid="stToolbar"]{display:none!important;}
    .main .block-container{padding:1rem!important;max-width:100%!important;}

    h1,h2,h3,h4,h5{font-family:'Segoe UI',sans-serif;color:#1976D2;margin:0 0 1rem;}
    h1:after,h2:after,h3:after,hr,.stMarkdown hr{display:none!important;}

    [data-testid="stTextInput"]>div>div,
    [data-testid="stSelectbox"]>div,
    [data-testid="stRadio"]>div{border:none!important;box-shadow:none!important;}

    .wizard-step{background:#f8f9fa;border:none!important;border-radius:8px;
                 padding:1.5rem;margin-bottom:1.25rem;}
    .disamb-box{background:#F1F8FF;border:none!important;border-radius:.6rem;padding:1.25rem;}

    .stButton>button,.stForm button{
        background:#1976D2;color:#fff;font-weight:600;border:none;border-radius:.5rem;
        padding:.55rem 1.3rem;transition:all .25s ease;}
    .stButton>button:hover,.stForm button:hover{background:#1565C0;}

    .results-summary{background:#f1f8e9;border-left:4px solid #689f38;margin-bottom:1.25rem;
                     border-radius:8px;padding:1rem;}
    .recommendations-box{background:#fff;border:1px solid #bbdefb;border-radius:8px;
                         padding:1.5rem;margin-bottom:1rem;box-shadow:0 1px 4px rgba(0,0,0,.05);}

    .fixed-table{height:420px;overflow:auto;width:100%;border:none!important;border-radius:6px;}
    .fixed-table table{border-collapse:collapse;width:100%;}
    .fixed-table th,.fixed-table td{border:none!important;padding:8px 10px;white-space:nowrap;
                                    text-overflow:ellipsis;max-width:340px;overflow:hidden;}
    .fixed-table th{position:sticky;top:0;background:#f3f3f3;font-weight:600;color:#1976D2;z-index:5;}
    .fixed-table tr:nth-child(even){background:#fafafa;}
    .fixed-table tr:hover{background:#f5f5f5;}
    
    .status-message{padding:10px 15px;border-radius:6px;margin-bottom:1rem;}
    .loading-spinner{display:flex;align-items:center;gap:10px;}
    .loading-spinner .spinner-text{font-style:italic;color:#555;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ───────────────────────── LOGGING ───────────────────────── #
log_file = f"vendor_search_{datetime.now().strftime('%Y%m%d')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# ───────────────────────── HELPERS ───────────────────────── #
def countries_list():
    c = [x.name for x in pycountry.countries]
    c.remove("United States")
    return ["United States"] + c


def query_string(product, location, country):
    if not location or location.lower() == "any location":
        return f"{product} vendors in {country}"
    return f"{product} vendors in {location}, {country}"


def google_cse(query, target, country):
    out = pd.DataFrame(columns=["title", "link", "snippet", "displayLink"])
    try:
        for start in range(1, target + 1, 10):
            p = {
                "key": os.environ["GOOGLE_CSE_API_KEY"],
                "cx": os.environ["GOOGLE_CSE_ID"],
                "q": query,
                "num": 10,
                "start": start,
                "gl": "us",
                "hl": "en",
            }
            if country == "United States":
                p["cr"] = "countryUS"
            j = requests.get("https://www.googleapis.com/customsearch/v1", params=p).json()
            items = j.get("items", [])
            if not items:
                break
            out = pd.concat(
                [
                    out,
                    pd.DataFrame(
                        [
                            {
                                "title": i.get("title", ""),
                                "link": i.get("link", ""),
                                "snippet": i.get("snippet", ""),
                                "displayLink": i.get("displayLink", ""),
                            }
                            for i in items
                        ]
                    ),
                ],
                ignore_index=True,
            )
            if len(out) >= target or len(items) < 10:
                break
        return out.head(target)
    except Exception as e:
        st.error(f"Search error: {e}")
        logger.exception(e)
        return pd.DataFrame()


# ▸▸▸▸▸  FIXED GEMINI WRAPPER  ▸▸▸▸▸
def gemini(prompt_text: str) -> str:
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    cfg = types.GenerateContentConfig(response_mime_type="text/plain")
    try:
        return "".join(
            chunk.text
            for chunk in client.models.generate_content_stream(
                model="gemini-2.5-flash-preview-04-17",
                contents=[
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=prompt_text)],  # << correct signature
                    )
                ],
                config=cfg,
            )
        )
    except Exception as e:
        logger.exception(e)
        return f"Error generating recommendations: {e}"


def disambiguate(term: str) -> dict:
    prompt = f"""
    As a search assistant, analyse whether the term "{term}" is ambiguous for finding vendors.
    If ambiguous, respond ONLY with JSON: {{is_ambiguous:true, interpretations:[...]}} with is_ambiguous true and up to 4 common interpretations
    (id, term, description). If unambiguous set is_ambiguous to false and include the single
    interpretation. Else {{is_ambiguous:false, interpretations:[{{id:1,term:"{term}",description:"Original term"}}]}} Respond ONLY with JSON.
    """
    try:
        t = gemini(prompt)
        j = t[t.find("{") : t.rfind("}") + 1]
        return json.loads(j)
    except Exception:
        return {"is_ambiguous": False, "interpretations": [{"id": 1, "term": term, "description": ""}]}


def llm_prompt(df, product, location, country):
    return (
        "No vendors found."
        if df.empty
        else f"""
You are a vendor recommendation expert. Provide 5‑7 quick, categorized recommendations
for "{product}" vendors in {location}, {country}.

Format your response as follows:
## Top Recommendations for {product}

### Best Overall
- **[Vendor Name]** – one‑sentence reason

### Best for [Specific Need]
- **[Vendor Name]** – one‑sentence reason

(Continue with 2‑4 more categories)

All my search results:
{df.to_string(index=False)}
"""
    )


def to_html_table(df: pd.DataFrame) -> str:
    html = '<div class="fixed-table"><table><tr>'
    html += "".join(f"<th>{c}</th>" for c in df.columns)
    html += "</tr>"
    for _, r in df.iterrows():
        html += "<tr>" + "".join(f"<td>{r[c]}</td>" for c in df.columns) + "</tr>"
    return html + "</table></div>"


def logo_b64() -> str | None:
    path = Path(__file__).with_name("logo.png")
    return base64.b64encode(path.read_bytes()).decode() if path.exists() else None


# ─────────────────── SESSION STATE DEFAULTS ─────────────────── #
defaults = dict(
    wizard_step=1,
    product_term="",
    disamb_json=None,
    product_final="",
    country="United States",
    location="",
    count_choice="1 – 50",
    search_completed=False,
    results_df=pd.DataFrame(),
    recommendations="",
    search_query="",
    form_submit_clicked=False,
    processing=False,
)
for k, v in defaults.items():
    st.session_state.setdefault(k, v)

# ───────────────────────── HEADER ───────────────────────── #
logo64 = logo_b64()
st.markdown(
    f"""
<div style="display:flex;align-items:center;padding-bottom:1.5rem;margin-bottom:1rem;">
  <div style="margin-right:1rem;">
    {f'<img src="data:image/png;base64,{logo64}" style="height:60px;width:auto;">' if logo64 else '<div style="height:60px;width:60px;"></div>'}
  </div>
  <h1>🔍 Vendor Search & Recommendation</h1>
</div>
""",
    unsafe_allow_html=True,
)


# ──────────────────── NAV HELPERS ──────────────────── #
def next_step(): st.session_state.wizard_step += 1
def prev_step(): st.session_state.wizard_step -= 1


# ──────────────────── STEP 1 ──────────────────── #
if st.session_state.wizard_step == 1:
    st.subheader("Step 1 · Product or Service")

    with st.container():
        st.markdown('<div class="wizard-step">', unsafe_allow_html=True)
        st.text_input(
            "What product or service are you looking for?",
            key="product_term",
            value=st.session_state.product_term,
            placeholder="e.g. Computer Keyboard, Cloud Storage, IT Security Services",
        )

        dj = st.session_state.disamb_json
        if dj and dj.get("is_ambiguous"):
            st.markdown('<div class="disamb-box">', unsafe_allow_html=True)
            st.write("Multiple meanings found – choose one:")
            opts = {o["term"]: f'{o["term"]} – {o["description"]}' for o in dj["interpretations"]}
            sel = st.radio("", list(opts.values()), key="disamb_radio")
            st.session_state.product_final = sel.split(" – ")[0]
            st.button("Continue", on_click=next_step)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            if st.button("Check Term"):
                with st.spinner("Analyzing your product term..."):
                    st.session_state.disamb_json = disambiguate(st.session_state.product_term.strip())
                    if not st.session_state.disamb_json.get("is_ambiguous"):
                        st.session_state.product_final = st.session_state.product_term.strip()
                        next_step()
        st.markdown("</div>", unsafe_allow_html=True)

# ──────────────────── STEP 2 ──────────────────── #
elif st.session_state.wizard_step == 2:
    st.subheader(f"Step 2 · Location for \"{st.session_state.product_final}\"")

    st.markdown('<div class="wizard-step">', unsafe_allow_html=True)
    with st.form("loc_form"):
        st.selectbox(
            "Country",
            countries_list(),
            key="form_country",
            index=countries_list().index(st.session_state.country),
        )
        st.text_input(
            "Location in country",
            key="form_location",
            value=st.session_state.location,
            placeholder="City / region – leave blank for nationwide",
        )
        st.radio(
            "How many vendors?",
            ["1 – 50", "50 – 100", "100 +"],
            key="form_count_choice",
            index=["1 – 50", "50 – 100", "100 +"].index(st.session_state.count_choice),
            horizontal=True,
        )

        col_back, col_go = st.columns([1, 3])
        with col_back:
            st.form_submit_button("Back", on_click=prev_step, use_container_width=True)
        with col_go:
            submitted = st.form_submit_button(
                "🔍 Search Vendors",
                use_container_width=True,
            )
            if submitted:
                st.session_state.form_submit_clicked = True
                st.session_state.country = st.session_state.form_country
                st.session_state.location = st.session_state.form_location
                st.session_state.count_choice = st.session_state.form_count_choice
    st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.form_submit_clicked:
        st.session_state.form_submit_clicked = False
        st.session_state.processing = True
        
        term = st.session_state.product_final
        q = query_string(term, st.session_state.location or "Any location", st.session_state.country)
        st.session_state.search_query = q
        tgt = {"1 – 50": 50, "50 – 100": 100}.get(st.session_state.count_choice, 150)

        with st.spinner(f"🔍 Searching for {term} vendors..."):
            progress_text = st.empty()
            progress_text.markdown('<div class="loading-spinner"><div class="spinner-text">Retrieving vendors from the web...</div></div>', unsafe_allow_html=True)
            st.session_state.results_df = google_cse(q, tgt, st.session_state.country)
            progress_text.empty()

        if st.session_state.results_df.empty:
            st.warning("No vendors matched – try broadening the terms.")
            st.session_state.processing = False
        else:
            with st.spinner("✨ Analyzing vendors and generating recommendations..."):
                progress_bar = st.progress(0)
                progress_text = st.empty()
                
                progress_text.markdown('<div class="loading-spinner"><div class="spinner-text">Processing vendor data (25%)...</div></div>', unsafe_allow_html=True)
                progress_bar.progress(25)
                
                progress_text.markdown('<div class="loading-spinner"><div class="spinner-text">Analyzing market positioning (50%)...</div></div>', unsafe_allow_html=True)
                progress_bar.progress(50)
                
                progress_text.markdown('<div class="loading-spinner"><div class="spinner-text">Comparing vendor capabilities (75%)...</div></div>', unsafe_allow_html=True)
                progress_bar.progress(75)
                
                st.session_state.recommendations = gemini(
                    llm_prompt(
                        st.session_state.results_df,
                        term,
                        st.session_state.location or "Any location",
                        st.session_state.country,
                    )
                )
                
                progress_text.markdown('<div class="loading-spinner"><div class="spinner-text">Finalizing recommendations (100%)...</div></div>', unsafe_allow_html=True)
                progress_bar.progress(100)
                
                progress_bar.empty()
                progress_text.empty()
                
            st.session_state.search_completed = True
            st.session_state.processing = False
            st.session_state.wizard_step = 3
            st.experimental_rerun()

# ──────────────────── STEP 3 – RESULTS ──────────────────── #
elif st.session_state.wizard_step == 3 and st.session_state.search_completed:
    st.markdown('<div class="results-summary">', unsafe_allow_html=True)
    loc_disp = st.session_state.location if st.session_state.location else "Nationwide"
    st.markdown(
        f"""
**Product / Service:** {st.session_state.product_final}  
**Location:** {loc_disp}, {st.session_state.country}  
**Vendor count:** {len(st.session_state.results_df)} vendors found
""",
        unsafe_allow_html=True,
    )
    if st.button("New Search"):
        for k, v in defaults.items():
            if k not in ["wizard_step"]:
                st.session_state[k] = v
        st.session_state.wizard_step = 1
        st.experimental_rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    left, right = st.columns([1, 2], gap="large")

    with left:
        st.markdown('<div class="recommendations-box">', unsafe_allow_html=True)
        st.markdown(st.session_state.recommendations, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        if not st.session_state.results_df.empty:
            col1, col2 = st.columns([1, 5])
            with col1:
                st.download_button(
                    "Download CSV",
                    data=st.session_state.results_df.to_csv(index=False).encode(),
                    file_name=f"vendor_search_{st.session_state.product_final.replace(' ','_').lower()}.csv",
                    mime="text/csv",
                )
            disp = st.session_state.results_df.copy()
            disp["title"] = disp.apply(
                lambda r: f'<a href="{r["link"]}" target="_blank">{r["title"]}</a>', axis=1
            )
            st.markdown(to_html_table(disp), unsafe_allow_html=True)
            st.caption(f"Showing {len(disp)} vendors for **{st.session_state.search_query}**")
        else:
            st.warning("No results – please try a different search.")

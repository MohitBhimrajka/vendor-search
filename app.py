# app.py  – final polish + Gemini fix (2025‑05‑07 rev‑D)

import os
import json
import logging
import base64
from datetime import datetime
from pathlib import Path
import time
import re

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

    .results-summary{background:#f5f5f5;border-left:4px solid #1976D2;margin-bottom:1.25rem;
                     border-radius:8px;padding:1rem;}
    .recommendations-box{background:#fff;border:1px solid #e0e0e0;border-radius:8px;
                         padding:1.5rem;margin-bottom:1.5rem;box-shadow:0 1px 4px rgba(0,0,0,.05);}

    .fixed-table{height:480px;overflow:auto;width:100%;border:1px solid #e0e0e0!important;border-radius:6px;margin-top:1rem;}
    .fixed-table table{border-collapse:collapse;width:100%;}
    .fixed-table th,.fixed-table td{border:none!important;padding:10px 12px;white-space:nowrap;
                                    text-overflow:ellipsis;max-width:340px;overflow:hidden;}
    .fixed-table th{position:sticky;top:0;background:#1976D2;color:white;font-weight:600;z-index:5;text-transform:uppercase;font-size:0.9rem;}
    .fixed-table tr:nth-child(even){background:#f9f9f9;}
    .fixed-table tr:hover{background:#f0f7ff;}
    
    .status-message{padding:10px 15px;border-radius:6px;margin-bottom:1rem;}
    .loading-spinner{display:flex;align-items:center;gap:10px;}
    .loading-spinner .spinner-text{font-style:italic;color:#555;}
    
    .action-bar{display:flex;gap:10px;margin-bottom:1rem;align-items:center;}
    .action-bar button{margin:0!important;}
    .vendor-count{color:#666;font-style:italic;margin-left:auto;}
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
    """Build a targeted vendor search query with industry-specific terms."""
    # Create base query with industry terms
    base_query = f"{product} vendors suppliers providers"
    
    # Add location specifics if provided
    if not location or location.lower() == "any location":
        location_query = f"in {country}"
    else:
        location_query = f"in {location}, {country}"
    
    # Combine with specific business-related terms
    return f"{base_query} {location_query} business directory"


def optimize_search_query(term: str) -> str:
    """Use LLM to optimize the search query for better vendor results."""
    prompt = f"""
    As a B2B sourcing expert, create a precise search query to find vendors for: "{term}"
    
    TASK:
    Create the most effective search query to find quality vendors and suppliers for this product/service.
    
    FORMAT GUIDELINES:
    - Return ONLY the search query terms (3-6 words)
    - Include industry-specific terminology
    - Add terms like: distributors, manufacturers, suppliers, vendors, or providers
    - For products: include relevant categories, types, or specifications
    - For services: include relevant expertise, certifications, or specializations
    - NO explanation text, quotes, or other content
    
    EXAMPLE SUCCESSFUL QUERIES:
    - For "Computer Keyboard": mechanical keyboard manufacturers distributors ergonomic
    - For "Cloud Storage": enterprise cloud storage solution providers secure
    - For "Accounting": certified accounting services tax specialists professional
    - For "Office Furniture": commercial office furniture suppliers ergonomic
    """
    
    max_attempts = 2
    
    for attempt in range(max_attempts):
        try:
            result = gemini(prompt).strip()
            
            # Validate the result
            if not result or len(result) < 5:
                raise ValueError("Response too short")
                
            # Clean up the result (remove quotes, punctuation, extra spaces)
            result = result.replace('"', '').replace("'", "").strip()
            result = re.sub(r'[^\w\s-]', '', result)  # Remove punctuation except hyphens
            
            # If the result is too long, truncate it to a reasonable length
            words = result.split()
            if len(words) > 8:
                result = " ".join(words[:8])
                
            return result
        except Exception as e:
            logger.warning(f"Query optimization attempt {attempt+1}/{max_attempts} failed: {e}")
            if attempt == max_attempts - 1:
                logger.exception(f"All query optimization attempts failed: {e}")
                # Return enhanced version of the original term as fallback
                return f"{term} vendors suppliers"
            # Wait before retry
            time.sleep(1)


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
        
        # Rename columns for better display
        if not out.empty:
            out = out.rename(columns={"snippet": "description"})
        
        return out.head(target)
    except Exception as e:
        st.error(f"Search error: {e}")
        logger.exception(e)
        return pd.DataFrame()


# ▸▸▸▸▸  FIXED GEMINI WRAPPER  ▸▸▸▸▸
def gemini(prompt_text: str, max_retries=2) -> str:
    """Enhanced Gemini wrapper with retry logic and improved error handling."""
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    cfg = types.GenerateContentConfig(response_mime_type="text/plain")
    
    for attempt in range(max_retries + 1):
        try:
            response_chunks = client.models.generate_content_stream(
                model="gemini-2.5-flash-preview-04-17",
                contents=[
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=prompt_text)],
                    )
                ],
                config=cfg,
            )
            
            # Safely join chunks, filtering out None values
            result = ""
            for chunk in response_chunks:
                if chunk and hasattr(chunk, 'text') and chunk.text is not None:
                    result += chunk.text
            
            if not result:
                raise ValueError("Empty response from Gemini")
                
            return result
            
        except Exception as e:
            logger.warning(f"Gemini API attempt {attempt+1}/{max_retries+1} failed: {e}")
            if attempt == max_retries:
                logger.exception(f"All Gemini API attempts failed: {e}")
                return f"Error: Could not generate response after {max_retries+1} attempts."
            # Wait briefly before retry (exponential backoff)
            time.sleep(1 * (2 ** attempt))


def extract_json(text: str) -> str:
    """Extract JSON from a text that might contain additional content.
    More robust version that handles various edge cases."""
    if not text:
        raise ValueError("Empty response")
    
    # Try to find JSON with standard markers first
    start = text.find('{')
    end = text.rfind('}')
    
    if start >= 0 and end > start:
        # Found potential JSON, try to extract and validate it
        potential_json = text[start:end+1]
        try:
            # Verify it's valid JSON
            json.loads(potential_json)
            return potential_json
        except json.JSONDecodeError:
            # Not valid JSON, continue with fallbacks
            pass
    
    # Try more aggressive extraction - look for any JSON-like content
    import re
    json_pattern = r'({[\s\S]*?})'
    matches = re.findall(json_pattern, text)
    
    for match in matches:
        try:
            json.loads(match)
            return match
        except json.JSONDecodeError:
            continue
    
    # No valid JSON found, create a fallback JSON
    fallback = {"error": "No valid JSON found in response", "raw_text": text[:200] + "..."}
    return json.dumps(fallback)


def disambiguate(term: str) -> dict:
    """Get potential interpretations for an ambiguous product/service term."""
    if not term.strip():
        return {"is_ambiguous": False, "interpretations": [{"id": 1, "term": term, "description": "Empty term"}]}
    
    prompt = f"""
    You are analyzing a search term to help find vendors. Your task is to determine if "{term}" has multiple business meanings.
    
    Respond ONLY with a JSON object in this exact format:
    {{
      "is_ambiguous": true/false,
      "interpretations": [
        {{ "id": 1, "term": "term1", "description": "description1" }},
        {{ "id": 2, "term": "term2", "description": "description2" }}
      ]
    }}
    
    Rules:
    1. Set is_ambiguous to true ONLY if the term has multiple distinct business interpretations
    2. Include 1-4 interpretations with unique id, term, and description
    3. For unambiguous terms (is_ambiguous = false), include ONLY ONE interpretation
    4. NEVER explain your reasoning, ONLY return valid parseable JSON
    5. NEVER include quotes, notes, or anything outside the JSON structure
    
    IMPORTANT: Your entire response must be ONLY valid JSON that can be parsed with json.loads().
    """
    
    max_attempts = 3
    
    for attempt in range(max_attempts):
        try:
            response_text = gemini(prompt)
            # Find JSON in the response
            json_text = extract_json(response_text)
            result = json.loads(json_text)
            
            # Validate the response structure
            if not isinstance(result, dict):
                raise ValueError("Response is not a dictionary")
            if "is_ambiguous" not in result:
                raise ValueError("Missing 'is_ambiguous' field")
            if "interpretations" not in result or not isinstance(result["interpretations"], list):
                raise ValueError("Missing or invalid 'interpretations' field")
                
            # Ensure interpretations have required fields and are not empty
            valid_interpretations = []
            for interp in result["interpretations"]:
                if all(k in interp for k in ["id", "term", "description"]) and interp["term"].strip():
                    valid_interpretations.append(interp)
            
            if not valid_interpretations:
                raise ValueError("No valid interpretations found")
            
            # Update with valid interpretations only
            result["interpretations"] = valid_interpretations
            return result
            
        except Exception as e:
            logger.warning(f"Disambiguation attempt {attempt+1}/{max_attempts} failed: {e}")
            if attempt == max_attempts - 1:
                logger.exception(f"All disambiguation attempts failed for term '{term}': {e}")
                # Fallback to non-ambiguous with original term
                return {
                    "is_ambiguous": False, 
                    "interpretations": [{"id": 1, "term": term, "description": "Original search term"}]
                }
            # Wait briefly before retry (exponential backoff)
            time.sleep(1 * (2 ** attempt))


def llm_prompt(df, product, location, country):
    """Create a prompt for generating vendor recommendations with website links."""
    if df.empty:
        return "No vendors found."
    
    # Create a version of the dataframe that includes only the necessary columns for recommendations
    # and ensure website URLs are properly formatted
    rec_df = df.copy()
    
    # Create a simple mapping of display names to their URLs for the LLM to reference
    vendor_links = {}
    for _, row in rec_df.iterrows():
        if 'displayLink' in row and 'title' in row:
            # Extract company name from title (simplified approach)
            company_name = row['title'].split(' - ')[0].strip() if ' - ' in row['title'] else row['title']
            # Clean up common suffixes in company names
            company_name = re.sub(r'\s+(Inc\.?|LLC|Ltd\.?|Corporation|Corp\.?|Co\.?)$', '', company_name, flags=re.IGNORECASE)
            
            # Add the display link with protocol if missing
            display_link = row['displayLink']
            if display_link and not display_link.startswith(('http://', 'https://')):
                full_url = f"https://{display_link}"
            else:
                full_url = display_link
                
            vendor_links[company_name] = full_url
    
    # Convert to a simple text format the LLM can use
    vendor_links_text = "\n".join([f"{name}: {url}" for name, url in vendor_links.items()])
    
    return f"""
You are a vendor recommendation specialist writing for a B2B audience. Based on the search results, create structured recommendations for "{product}" vendors in {location}, {country}.

RESPONSE FORMAT:
```markdown
## Top Recommendations for {product}

### Best Overall
- **[Company Name]** – [One sentence about their key strength and differentiation] [Website: URL]

### Best for [Specific Category 1]
- **[Company Name]** – [One sentence about why they excel in this category] [Website: URL]

### Best for [Specific Category 2]
- **[Company Name]** – [One sentence about why they excel in this category] [Website: URL]

[Continue with 2-3 more categories that are relevant to this product/service]
```

INSTRUCTIONS:
1. Use ONLY vendors that appear in the search results
2. Create 4-5 distinct, relevant categories (like "Best Enterprise Solution", "Best Budget Option", etc.)
3. For each category, recommend 1-2 vendors with clear justification
4. Include the vendor's website URL with each recommendation using format: [Website: URL]
5. Write in a factual, business-appropriate tone
6. If specific local vendors aren't found, focus on nationwide/online providers
7. ONLY use markdown format as shown above

SEARCH RESULTS:
{df.to_string(index=False)}

VENDOR WEBSITES (use these exact URLs in your recommendations):
{vendor_links_text}
"""


def to_html_table(df: pd.DataFrame) -> str:
    # Define the desired column order and display names
    column_mapping = {
        "title": "Title",
        "description": "Description", 
        "displayLink": "Website"
    }
    
    # Filter to only show desired columns in the specified order
    columns_to_display = [col for col in column_mapping.keys() if col in df.columns]
    
    html = '<div class="fixed-table"><table><tr>'
    for col in columns_to_display:
        html += f"<th>{column_mapping[col]}</th>"
    html += "</tr>"
    
    for _, r in df.iterrows():
        html += "<tr>"
        for col in columns_to_display:
            if col == "title":
                # Make title a clickable link
                html += f'<td><a href="{r["link"]}" target="_blank">{r[col]}</a></td>'
            else:
                html += f"<td>{r[col]}</td>"
        html += "</tr>"
    
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
    check_term_clicked=False,   # Track if Check Term was clicked
    button_disabled=False,      # Disable buttons during processing
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
def next_step(): 
    st.session_state.wizard_step += 1
    st.session_state.button_disabled = False
    
def prev_step(): 
    st.session_state.wizard_step -= 1
    st.session_state.button_disabled = False

def check_term_handler():
    st.session_state.check_term_clicked = True
    st.session_state.button_disabled = True
    
def disamb_selection_handler():
    next_step()


# ──────────────────── STEP 1 ──────────────────── #
if st.session_state.wizard_step == 1:
    st.subheader("Step 1 · Product or Service")

    with st.container():
        st.markdown('<div class="wizard-step">', unsafe_allow_html=True)
        
        term_input = st.text_input(
            "What product or service are you looking for?",
            key="product_term",
            value=st.session_state.product_term,
            placeholder="e.g. Computer Keyboard, Cloud Storage, IT Security Services",
            on_change=lambda: setattr(st.session_state, 'check_term_clicked', False)
        )

        # Only proceed if the term is not empty
        if term_input and term_input.strip():
            dj = st.session_state.disamb_json
            
            # Show disambiguation options if term is ambiguous
            if dj and dj.get("is_ambiguous") and dj.get("interpretations"):
                st.markdown('<div class="disamb-box">', unsafe_allow_html=True)
                st.write("This term has multiple potential meanings. Please select one:")
                
                # Ensure interpretations exist and have required fields
                valid_options = []
                for option in dj.get("interpretations", []):
                    if all(k in option for k in ["term", "description"]):
                        label = f"{option['term']} – {option['description']}"
                        valid_options.append((option['term'], label))
                
                if valid_options:
                    # Create radio options with descriptions
                    options_labels = [opt[1] for opt in valid_options]
                    selected = st.radio("Select an interpretation:", options_labels, key="disamb_radio")
                    
                    # Get the selected term without the description
                    selected_term = next((opt[0] for opt in valid_options if opt[1] == selected), term_input)
                    st.session_state.product_final = selected_term
                    
                    st.button(
                        "Continue with this selection", 
                        key="disamb_continue",
                        on_click=disamb_selection_handler,
                        disabled=st.session_state.button_disabled
                    )
                else:
                    st.warning("No valid interpretations found. Please try a different search term.")
                    if st.button("Try with original term", key="use_original", disabled=st.session_state.button_disabled):
                        st.session_state.product_final = term_input
                        next_step()
                
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                # Button to check term or proceed
                check_col, info_col = st.columns([1, 4])
                with check_col:
                    st.button(
                        "Check Term", 
                        key="check_term",
                        on_click=check_term_handler,
                        disabled=st.session_state.button_disabled or st.session_state.processing
                    )
                
                with info_col:
                    if st.session_state.processing:
                        st.markdown('<div class="loading-spinner"><div class="spinner-text">Analyzing term...</div></div>', unsafe_allow_html=True)
                
                # Process check term click immediately
                if st.session_state.check_term_clicked and not st.session_state.processing:
                    # Clear flag first to prevent multiple processing
                    st.session_state.check_term_clicked = False
                    st.session_state.processing = True
                    
                    with st.spinner("Analyzing your product term..."):
                        st.session_state.disamb_json = disambiguate(term_input.strip())
                        
                        # If not ambiguous or error occurred, set final term and proceed
                        if not st.session_state.disamb_json.get("is_ambiguous", False):
                            # Get the term from the first interpretation or use input
                            interps = st.session_state.disamb_json.get("interpretations", [])
                            if interps and "term" in interps[0]:
                                st.session_state.product_final = interps[0]["term"]
                            else:
                                st.session_state.product_final = term_input.strip()
                            
                            st.session_state.processing = False
                            st.session_state.button_disabled = False
                            next_step()
                            st.rerun()
                        else:
                            st.session_state.processing = False
                            st.session_state.button_disabled = False
                            st.rerun()
        else:
            st.info("Please enter a product or service to continue.")
        
        st.markdown("</div>", unsafe_allow_html=True)

# ──────────────────── STEP 2 ──────────────────── #
elif st.session_state.wizard_step == 2:
    st.subheader(f"Step 2 · Location for \"{st.session_state.product_final}\"")

    st.markdown('<div class="wizard-step">', unsafe_allow_html=True)
    with st.form("loc_form", clear_on_submit=False):
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
            back_pressed = st.form_submit_button(
                "Back", 
                use_container_width=True,
                disabled=st.session_state.processing
            )
            if back_pressed:
                prev_step()
                st.rerun()
                
        with col_go:
            submitted = st.form_submit_button(
                "🔍 Search Vendors",
                use_container_width=True,
                disabled=st.session_state.processing
            )
            if submitted:
                st.session_state.form_submit_clicked = True
                # Update state immediately
                st.session_state.country = st.session_state.form_country
                st.session_state.location = st.session_state.form_location
                st.session_state.count_choice = st.session_state.form_count_choice
                st.session_state.button_disabled = True
                st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    # Process form submission
    if st.session_state.form_submit_clicked:
        st.session_state.form_submit_clicked = False  # Reset for next use
        st.session_state.processing = True
        
        # First optimize the query terms
        term = st.session_state.product_final
        
        with st.spinner(f"🔍 Optimizing search query..."):
            optimized_term = optimize_search_query(term)
            logger.info(f"Optimized query: '{term}' → '{optimized_term}'")
        
        # Then construct the full query with location
        q = query_string(optimized_term, st.session_state.location or "Any location", st.session_state.country)
        st.session_state.search_query = q
        logger.info(f"Final search query: '{q}'")
        tgt = {"1 – 50": 50, "50 – 100": 100}.get(st.session_state.count_choice, 150)

        with st.spinner(f"🔍 Searching for {term} vendors..."):
            progress_text = st.empty()
            progress_text.markdown('<div class="loading-spinner"><div class="spinner-text">Retrieving vendors from the web...</div></div>', unsafe_allow_html=True)
            st.session_state.results_df = google_cse(q, tgt, st.session_state.country)
            progress_text.empty()

        if st.session_state.results_df.empty:
            st.warning("No vendors matched – try broadening the terms.")
            st.session_state.processing = False
            st.session_state.button_disabled = False
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
            st.session_state.button_disabled = False
            st.session_state.wizard_step = 3
            st.rerun()

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
    if st.button("New Search", disabled=st.session_state.button_disabled):
        # Reset all necessary values for a new search
        st.session_state.button_disabled = True
        for k, v in defaults.items():
            if k not in ["wizard_step", "button_disabled"]:  # Keep some values
                st.session_state[k] = v
        st.session_state.wizard_step = 1
        st.session_state.button_disabled = False
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    # Only show recommendations if we have results
    if not st.session_state.results_df.empty:
        # Recommendations section
        st.markdown('<div class="recommendations-box">', unsafe_allow_html=True)
        st.markdown(st.session_state.recommendations, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Results section
        st.markdown('<div class="action-bar">', unsafe_allow_html=True)
        st.download_button(
            "📥 Download CSV",
            data=st.session_state.results_df.to_csv(index=False).encode(),
            file_name=f"vendor_search_{st.session_state.product_final.replace(' ','_').lower()}.csv",
            mime="text/csv",
            disabled=st.session_state.button_disabled,
        )
        st.markdown(f'<div class="vendor-count">Showing {len(st.session_state.results_df)} results for "{st.session_state.search_query}"</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        disp = st.session_state.results_df.copy()
        st.markdown(to_html_table(disp), unsafe_allow_html=True)
    else:
        st.warning("No results – please try a different search.")
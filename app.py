import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from pathlib import Path

# --------------------------
# PAGE + THEME
# --------------------------
st.set_page_config(page_title="Groundwater Stage Dashboard", layout="wide")

# NOTE: For a fully light theme and to remove the top black bar, 
# ensure you have a .streamlit/config.toml file with:
# [theme]
# base="light"

# Light theme with improved contrast and card styling
st.markdown(
    """
    <style>
    /* 1. Global background and main text contrast */
    body, .stApp { 
        background-color: #F8F9FB !important; 
        color: #1F2937; 
    }
    
    /* Ensure all container content has dark text */
    .stApp, .block-container, p, li, .st-dh, .st-dp, .st-bo { 
        color: #1F2937 !important;
    }
    
    .block-container { 
        padding-top: 1rem; padding-bottom: 1.25rem; 
    }
    
    /* Headings */
    h1, h2, h3, h4 { 
        color: #0F2B46; 
    }
    
    /* 2. Tab Styling Fix - Ensure tabs are clearly visible */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        margin-top: 20px; 
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #ECEFF1; 
        color: #0F2B46; 
        border-radius: 8px 8px 0 0;
        font-weight: 600;
        padding: 10px 18px;
        transition: background-color 0.2s;
        border: none;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #CFD8DC;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF; 
        color: #B71C1C; 
        border-bottom: 3px solid #B71C1C; 
    }

    /* 3. Card style */
    .card {
        background: #FFFFFF; 
        border-radius: 12px; 
        padding: 18px 20px; 
        box-shadow: 0 4px 12px rgba(15, 43, 70, 0.08); 
        border: 1px solid rgba(15, 43, 70, 0.04);
        transition: transform 0.2s;
    }
    .card:hover {
        transform: translateY(-1px);
    }

    /* KPI (Key Performance Indicator) styling */
    .kpi { 
        font-size: 32px; 
        font-weight: 700; 
        color: #0F2B46; 
    }
    .kpi-label { 
        color: #4B5563; 
        font-size: 14px; 
        margin-top: -4px; 
        font-weight: 500;
    }
    
    /* Explanation text */
    .explain { 
        color: #37474F; 
        font-size: 1rem; 
        line-height: 1.6rem; 
        margin-bottom: 1rem;
    }
    
    /* Tag/Badge styling (used in stage_badge) */
    .tag {
        display:inline-block; 
        padding:5px 12px; 
        border-radius:999px; 
        font-size:13px; 
        font-weight:700;
        text-transform: uppercase; 
        letter-spacing: 0.5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Plotly defaults (white bg, light grids)
def set_plotly_clean(fig, height=340):
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=40, b=10),
        height=height,
        legend_title_text="",
        font=dict(color="#1F2937") 
    )
    fig.update_xaxes(showgrid=False, title_font=dict(size=14), tickfont=dict(size=12))
    fig.update_yaxes(gridcolor="rgba(0,0,0,0.08)", title_font=dict(size=14), tickfont=dict(size=12))
    return fig

# --------------------------
# PATHS
# --------------------------
BASE = Path(__file__).resolve().parent
DATA_PATH = BASE / "data" / "Dynamic_2017_2_0.csv" 
MODEL_PATH = BASE / "groundwater_stage_model.pkl"

# --------------------------
# LOAD
# --------------------------
@st.cache_data
def load_df():
    df = pd.read_csv(DATA_PATH, low_memory=False) 
    df.columns = [c.strip() for c in df.columns]
    return df

@st.cache_resource
def load_model():
    try:
        return joblib.load(MODEL_PATH) 
    except Exception:
        return None

try:
    df = load_df()
    model = load_model()
except FileNotFoundError:
    st.error(f"Data file not found at {DATA_PATH.absolute()}. Please check the path and file name.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred during data or model loading: {e}")
    st.stop()


# --------------------------
# LABEL & HELPERS
# --------------------------
PERCENT_CANDIDATES = [
    "Stage of Ground Water Extraction (%)",
    "Stage of Ground Water Extraction (%) - 2017",
    "Stage of Ground Water Extraction (%) - 2011",
    "Stage of GW Extraction (%)",
    "Stage of groundwater extraction (%)",
]

STATE_COL = "Name of State"
DISTRICT_COL = "Name of District"

def detect_label(_df: pd.DataFrame):
    for c in _df.columns:
        if c.strip().lower() == "stage_label":
            return c
    for c in _df.columns:
        cl = c.lower()
        if any(k in cl for k in ["stage","label","status","category","class","level"]):
            if 2 <= _df[c].nunique(dropna=True) <= 20: 
                return c
    return None

def classify_percent(x: float) -> str:
    if pd.isna(x): return "unknown"
    if x < 70: return "safe"
    if x < 90: return "semi-critical"
    if x < 100: return "critical"
    return "over-exploited"

def ensure_stage_label(_df: pd.DataFrame):
    label = detect_label(_df)
    used_pc = None
    if label:
        return _df, label, used_pc
    
    for pc in PERCENT_CANDIDATES:
        if pc in _df.columns and pd.api.types.is_numeric_dtype(_df[pc]):
            tmp = _df.copy()
            tmp["stage_label"] = tmp[pc].apply(classify_percent) 
            return tmp, "stage_label", pc
            
    return _df, None, None

df, label_col, used_percent_col = ensure_stage_label(df.copy()) 

num_cols = df.select_dtypes(include=np.number).columns.tolist()
cat_cols = [c for c in df.columns if c not in num_cols and c != label_col]

# Palette 
COLOR_MAP = {
    "safe": "#42A5F5",        # Blue
    "semi-critical": "#66BB6A", # Green
    "critical": "#FFA726",      # Orange
    "over-exploited": "#EF5350", # Red
    "unknown": "#9E9E9E"      # Gray
}

def donut_from_counts(counts_df: pd.DataFrame, xcol: str, ycol: str, title: str):
    order = ["safe","semi-critical","critical","over-exploited","unknown"]
    counts_df = counts_df.copy()
    counts_df["__ord__"] = counts_df[xcol].map({v:i for i,v in enumerate(order)})
    counts_df = counts_df.sort_values("__ord__").drop(columns="__ord__", errors="ignore")
    
    fig = px.pie(
        counts_df, names=xcol, values=ycol, hole=0.6, 
        color=xcol, color_discrete_map=COLOR_MAP, title=title
    )
    fig.update_traces(textposition='inside', textinfo='percent+label', 
                      marker=dict(line=dict(color='#FFFFFF', width=1))) 
    set_plotly_clean(fig, height=350)
    return fig

def stage_badge(stage: str):
    stage = str(stage).lower().replace('-', '')
    colors = {
        "safe": ("#E3F2FD", "#1565C0"), 
        "semicritical": ("#E8F5E9", "#388E3C"), 
        "critical": ("#FFF3E0", "#E65100"), 
        "overexploited": ("#FBE9E7", "#C62828"), 
        "unknown": ("#ECEFF1", "#37474F") 
    }
    bg, fg = colors.get(stage, ("#ECEFF1", "#37474F")) 
    display_name = stage.replace('semi', 'Semi-').replace('over', 'Over-').title()
    return f"<span class='tag' style='background:{bg}; color:{fg}; border-color:{bg};'>{display_name}</span>"


# --------------------------
# HANDLER FUNCTION FOR DEFAULTS
# --------------------------
def update_defaults_handler(df, feature_cols, state_val, dist_val, used_percent_col):
    """Calculates median/mode for selected location and stores in session state."""
    
    # 1. Calculate the filtered DataFrame
    df_filtered_by_state = df[df[STATE_COL].astype(str) == state_val]
    df_final_filtered = df_filtered_by_state[df_filtered_by_state[DISTRICT_COL].astype(str) == dist_val]
    
    # 2. Calculate new dynamic defaults
    dynamic_defaults = {}
    for c in feature_cols:
        if pd.api.types.is_numeric_dtype(df_final_filtered[c]):
            median_val = pd.to_numeric(df_final_filtered[c], errors="coerce").median()
            dynamic_defaults[c] = float(median_val) if not pd.isna(median_val) else 0.0
        elif c == STATE_COL:
            dynamic_defaults[c] = state_val
        elif c == DISTRICT_COL:
            dynamic_defaults[c] = dist_val
        else: 
            m = df_final_filtered[c].dropna().astype(str).mode() 
            dynamic_defaults[c] = str(m.iloc[0]) if not m.empty else ""

    # 3. Get the actual percentage value (if column exists)
    percentage = np.nan
    if used_percent_col in df_final_filtered.columns:
        percentage = pd.to_numeric(df_final_filtered[used_percent_col], errors="coerce").median()
        
    # 4. Store the full dictionary, location data, and percentage in session state
    st.session_state['dynamic_defaults_dict'] = dynamic_defaults
    st.session_state['current_state_val'] = state_val
    st.session_state['current_dist_val'] = dist_val
    st.session_state['current_percentage'] = percentage

# --------------------------
# TABS (Prediction is now the default first tab)
# --------------------------
tabs = st.tabs(["🧠 Prediction", "🏠 Overview", "📊 Insights", "🧭 Recommendations"]) 

# --------------------------
# TAB 1: PREDICTION (Now Default)
# --------------------------
with tabs[0]:
    st.markdown("## Groundwater Stage Prediction")
    st.markdown(
        "<div class='explain'>Select a location and click **Load Defaults** to view the area's current metrics "
        "and auto-fill the numeric drivers with median values.</div>",
        unsafe_allow_html=True
    )

    if model is None:
        st.warning("Prediction Model not found. Please run `python train.py` to create `groundwater_stage_model.pkl`.")
    else:
        excluded = set(filter(None, [label_col, used_percent_col]))
        feature_cols = [c for c in df.columns if c not in excluded]

        # --- Initial Defaults (Fallback to overall median/mode) ---
        initial_defaults = {}
        for c in feature_cols:
            if pd.api.types.is_numeric_dtype(df[c]):
                initial_defaults[c] = float(pd.to_numeric(df[c], errors="coerce").median()) 
            else:
                m = df[c].dropna().astype(str).mode() 
                initial_defaults[c] = str(m.iloc[0]) if not m.empty else ""
        
        # Initialize session state for the defaults if it's the first run
        if 'dynamic_defaults_dict' not in st.session_state:
             st.session_state['dynamic_defaults_dict'] = initial_defaults
             st.session_state['current_state_val'] = initial_defaults.get(STATE_COL)
             st.session_state['current_dist_val'] = initial_defaults.get(DISTRICT_COL)
             st.session_state['current_percentage'] = np.nan


        KEY_WORDS = ["recharge", "extraction", "use", "allocation", "withdrawal", "draft"]
        key_num = [c for c in feature_cols if (c in df.columns and pd.api.types.is_numeric_dtype(df[c]) and any(k in c.lower() for k in KEY_WORDS))]
        if len(key_num) < 6:
            extra = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c]) and c not in key_num]
            key_num = key_num + extra[: max(0, 6 - len(key_num))]
        key_num = key_num[:8] 
        
        # --- LOCATION SELECTION (Interactive) ---
        st.subheader("Location Selection")
        c_loc1, c_loc2 = st.columns([0.4, 0.6])

        # --- State Selection ---
        state_opts = sorted([str(x) for x in df[STATE_COL].dropna().unique().tolist()])
        initial_selected_state = st.session_state.get('current_state_val', state_opts[0] if state_opts else "")
        initial_state_idx = state_opts.index(initial_selected_state) if initial_selected_state in state_opts else 0
        
        state_val = c_loc1.selectbox(
            STATE_COL, 
            state_opts, 
            index=initial_state_idx,
            key="pred_state_key"
        )
        
        # --- District Selection (Dynamically filtered) ---
        df_filtered_by_state = df[df[STATE_COL].astype(str) == state_val]
        dist_opts = sorted([str(x) for x in df_filtered_by_state[DISTRICT_COL].dropna().unique().tolist()])
        
        current_dist_in_new_state = st.session_state.get('current_dist_val', dist_opts[0] if dist_opts else "")
        if current_dist_in_new_state not in dist_opts:
            selected_district_mode = df_filtered_by_state[DISTRICT_COL].dropna().astype(str).mode()
            current_dist_in_new_state = str(selected_district_mode.iloc[0]) if not selected_district_mode.empty else dist_opts[0] if dist_opts else ""
            
        initial_dist_idx = dist_opts.index(current_dist_in_new_state) if current_dist_in_new_state in dist_opts else 0
        
        dist_val = c_loc2.selectbox(
            DISTRICT_COL, 
            dist_opts, 
            index=initial_dist_idx,
            key="pred_district_key"
        )
        
        # --- LOAD DEFAULTS BUTTON ---
        st.button(
            "Load Defaults for Selected Location", 
            on_click=update_defaults_handler,
            args=(df, feature_cols, state_val, dist_val, used_percent_col),
            type="secondary"
        )
        
        # -----------------------------------------------
        # ✅ NEW SECTION: DISPLAY CURRENT LOCATION METRICS
        # -----------------------------------------------
        st.write("---")
        
        current_defaults = st.session_state['dynamic_defaults_dict']
        current_percentage = st.session_state.get('current_percentage')
        current_display_district = st.session_state.get('current_dist_val')
        
        # Display the classification
        if not pd.isna(current_percentage):
            current_stage = classify_percent(current_percentage)
            st.subheader(f"Current Status of {current_display_district}")
            
            col_perc, col_stage = st.columns(2)
            
            with col_perc:
                st.markdown(
                    f"<div class='card'>"
                    f"<div class='kpi'>{current_percentage:.2f}%</div>"
                    f"<div class='kpi-label'>{used_percent_col} (Median)</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )
            
            with col_stage:
                st.markdown(
                    f"<div class='card'>"
                    f"<h4>Current Stress Stage:</h4>"
                    f"{stage_badge(current_stage)}"
                    f"</div>",
                    unsafe_allow_html=True
                )
            
            st.write("---")
            
            # Display Loaded Numeric Drivers (excluding State/District)
            st.subheader("Loaded Numeric Drivers (Median Values)")
            
            # Filter the defaults to only include the numeric drivers we care about
            driver_data = {
                c: f"{current_defaults.get(c, 0.0):.2f}"
                for c in key_num
            }
            
            # Create a simple DataFrame for display
            display_df = pd.DataFrame(
                list(driver_data.items()),
                columns=["Feature", f"Median Value in {current_display_district}"]
            )
            
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            st.caption("These are the values pre-loaded into the fields below. Change them to predict a future scenario.")
            
            st.write("---")
            
        else:
            st.info("Select a State and District, then click 'Load Defaults' above to display the current metrics and populate the numeric fields.")

        # --- INPUT FORM (Only for Numeric Drivers and Submit Button) ---
        with st.form("prediction_form", clear_on_submit=False):
            
            st.subheader("Adjust Drivers for Prediction")
            col1, col2 = st.columns(2)
            edits = {}
            for i, c in enumerate(key_num):
                with (col1 if i % 2 == 0 else col2):
                    # Get value from the session state dictionary
                    default_value = float(current_defaults.get(c, 0.0)) 
                    edits[c] = st.number_input(
                        c, 
                        # Use the session state value
                        value=default_value, 
                        format="%.2f", 
                        help=f"Loaded Median: {default_value:.2f}",
                        key=f"num_input_{c}" 
                    )

            go = st.form_submit_button("Predict Groundwater Stage", type="primary")

        if go:
            # Re-fetch the current defaults and then update with user edits
            row = current_defaults.copy() 
            row.update(edits) 
            Xnew = pd.DataFrame([row], columns=feature_cols) 
            
            try:
                pred = model.predict(Xnew)[0]
                pred_proba = model.predict_proba(Xnew)[0]
                
                st.subheader("✅ Prediction Result (Based on Adjusted Drivers)")
                st.markdown(f"**Predicted Stage:** {stage_badge(pred)}", unsafe_allow_html=True)
                
                st.write("**Confidence Score (Probabilities):**")
                proba_df = pd.DataFrame({
                    "Stage": model.classes_,
                    "Probability": pred_proba
                }).sort_values("Probability", ascending=False).reset_index(drop=True)

                proba_df['Stage Badge'] = proba_df['Stage'].apply(lambda x: stage_badge(x))
                proba_df['Probability (%)'] = (proba_df['Probability'] * 100).round(2)
                
                st.markdown(proba_df[['Stage Badge', 'Probability (%)']].to_markdown(index=False), unsafe_allow_html=True)
                
                st.caption(
                    "All non-visible categorical and numeric features were automatically filled "
                    "with the selected location's mode/median to ensure a complete input row."
                )
            except Exception as e:
                if "tabulate" in str(e).lower():
                    st.error("Prediction failed. An error occurred: Missing optional dependency 'tabulate'. Please run 'pip install tabulate' and restart the app.")
                else:
                    st.error(f"Prediction failed. An error occurred: {e}")
                st.info("Check if your `train.py` model pipeline is correctly configured for the input features.")

# --------------------------
# TAB 2: OVERVIEW
# --------------------------
with tabs[1]:
    st.markdown("## Groundwater Stage Dashboard (Overview)")
    st.markdown(
        "<div class='explain'>Overview of groundwater stress categories. "
        "Categories are derived from the **Groundwater Extraction Percent** "
        "(<b style='color:#1565C0;'>&lt;70%</b> Safe, <b style='color:#388E3C;'>70–90%</b> Semi-critical, "
        "<b style='color:#E65100;'>90–100%</b> Critical, <b style='color:#C62828;'>&gt;100%</b> Over-exploited).</div>",
        unsafe_allow_html=True
    )

    k1, k2, k3, k4 = st.columns(4)
    if label_col:
        vc = df[label_col].astype(str).str.lower().value_counts()
        with k1: 
            st.markdown(
                f"<div class='card'><div class='kpi'>{len(df):,}</div>"
                f"<div class='kpi-label'>Total Data Points</div></div>", 
                unsafe_allow_html=True
            )
        with k2: 
            st.markdown(
                f"<div class='card'><div class='kpi'>{int(vc.get('safe',0)):,}</div>"
                f"<div class='kpi-label'>Safe Blocks/Units</div></div>", 
                unsafe_allow_html=True
            )
        with k3: 
            st.markdown(
                f"<div class='card'><div class='kpi'>{int(vc.get('semi-critical',0)):,}</div>"
                f"<div class='kpi-label'>Semi-critical Blocks/Units</div></div>", 
                unsafe_allow_html=True
            )
        with k4: 
            high_stress_count = int(vc.get('critical',0) + vc.get('over-exploited',0))
            st.markdown(
                f"<div class='card'><div class='kpi'>{high_stress_count:,}</div>"
                f"<div class='kpi-label'>High-Stress (Critical + Over)</div></div>", 
                unsafe_allow_html=True
            )

    st.write("---") 

    if label_col:
        donut_df = df[label_col].astype(str).str.lower().value_counts().rename_axis("stage").reset_index(name="count")
        fig = donut_from_counts(donut_df, "stage", "count", "Overall Category Mix (Block/Unit Count)")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Bigger coral/orange slices indicate a higher share of high-stress groundwater areas.")
    else:
        st.info("Category label column could not be identified or generated from percentage data.")


# --------------------------
# TAB 3: INSIGHTS
# --------------------------
with tabs[2]:
    st.markdown("## Regional Insights")
    st.markdown("<div class='explain'>Visual patterns of groundwater stress across States and Districts.</div>", unsafe_allow_html=True)

    st.write("---")

    if label_col and (STATE_COL in df.columns) and (DISTRICT_COL in df.columns):
        # State stack
        st.subheader("State-wise Category Distribution")
        grp_state = df.groupby([STATE_COL, label_col]).size().reset_index(name="count")
        order = ["safe", "semi-critical", "critical", "over-exploited", "unknown"]
        fig1 = px.bar(
            grp_state, x=STATE_COL, y="count", color=label_col,
            category_orders={label_col: order}, 
            color_discrete_map=COLOR_MAP, 
            title="Category Distribution by State", barmode="stack"
        )
        fig1.update_layout(xaxis_title="State", yaxis_title="Number of Blocks/Units")
        set_plotly_clean(fig1, height=450)
        st.plotly_chart(fig1, use_container_width=True)

        st.write("---")
        
        # District top risky
        st.subheader("Top Districts by High-Stress Count")
        g = df.groupby([DISTRICT_COL, label_col]).size().reset_index(name="count")
        risky = g[g[label_col].astype(str).str.lower().str.contains("over|critical")] 
        top_d = risky.groupby(DISTRICT_COL)["count"].sum().reset_index().sort_values("count", ascending=False).head(15)
        
        if not top_d.empty:
            fig2 = px.bar(
                top_d, x=DISTRICT_COL, y="count", 
                title="Top 15 Districts with Critical or Over-Exploited Groundwater",
                color_discrete_sequence=["#C62822"] 
            )
            fig2.update_layout(xaxis_title="District", yaxis_title="Number of High-Stress Blocks/Units")
            set_plotly_clean(fig2, height=400)
            st.plotly_chart(fig2, use_container_width=True)
        else:
             st.info("No high-stress (Critical/Over) data found in the dataset to display a top list.")
             
    else:
        st.info("Insights need columns: 'Name of State', 'Name of District', and a category label or % column to be present.")

# --------------------------
# TAB 4: RECOMMENDATIONS
# --------------------------
with tabs[3]:
    st.markdown("## Management Recommendations")
    st.markdown(
        "<div class='explain'>Strategic actions tailored to each groundwater stress category for sustainable management.</div>",
        unsafe_allow_html=True
    )
    st.write("---")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            f"<div class='card' style='border-left: 5px solid {COLOR_MAP['safe']};'>"
            f"<h4>Safe {stage_badge('safe')}</h4>"
            "<ul>"
            "<li>Continue monitoring & publish seasonal dashboards.</li>"
            "<li>Mandate **rooftop rainwater harvesting** in new buildings.</li>"
            "<li>Community awareness on sustainable use and recharge.</li>"
            "</ul></div>",
            unsafe_allow_html=True
        )
        st.write("")
        st.markdown(
            f"<div class='card' style='border-left: 5px solid {COLOR_MAP['semi-critical']};'>"
            f"<h4>Semi-critical {stage_badge('semi-critical')}</h4>"
            "<ul>"
            "<li>Promote **drip/sprinkler irrigation** and crop diversification.</li>"
            "<li>Implement small recharge structures (contour trenches, farm ponds).</li>"
            "<li>Begin soft abstractions control + metering pilots.</li>"
            "</ul></div>",
            unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            f"<div class='card' style='border-left: 5px solid {COLOR_MAP['critical']};'>"
            f"<h4>Critical {stage_badge('critical')}</h4>"
            "<ul>"
            "<li>Limit new deep borewells; **mandatory metering** for large users.</li>"
            "<li>Scale up major recharge structures (check dams, percolation tanks).</li>"
            "<li>Enforce seasonal advisories and water budgeting.</li>"
            "<li>Enforce seasonal advisories and water budgeting.</li>"
            "</ul></div>",
            unsafe_allow_html=True
        )
        st.write("")
        st.markdown(
            f"<div class='card' style='border-left: 5px solid {COLOR_MAP['over-exploited']};'>"
            f"<h4>Over-exploited {stage_badge('over-exploited')}</h4>"
            "<ul>"
            "<li>**Strict abstraction controls** and permits for all users.</li>"
            "<li>Substitute sources: treated wastewater for non-potable uses.</li>"
            "<li>Incentivize demand reduction and reuse at scale with subsidies.</li>"
            "</ul></div>",
            unsafe_allow_html=True
        )
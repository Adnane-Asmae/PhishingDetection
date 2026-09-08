
import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Phishing Detection",
    page_icon="🛡️",
    layout="wide"
)


# ============================================================
# FILE PATHS
# ============================================================

MODEL_PATH = "notebooks/random_forest_phishing_model.pkl"
FEATURES_PATH = "notebooks/feature_names.pkl"

DATASET_PATHS = [
    "dataset_B_05_2020.csv",
    "data/dataset_B_05_2020.csv",
    "datasets/dataset_B_05_2020.csv",
    "notebooks/dataset_B_05_2020.csv",
]


# ============================================================
# IMPORTANT FEATURES FOR SECTION 1
# ============================================================

important_features = [
    "google_index",
    "page_rank",
    "length_url",
    "length_hostname",
    "nb_dots",
    "nb_hyphens",
    "phish_hints",
    "domain_in_title"
]


# ============================================================
# LOAD MODEL AND FEATURE NAMES
# ============================================================

try:
    model = joblib.load(MODEL_PATH)
    feature_names = joblib.load(FEATURES_PATH)

except Exception as e:
    st.error("❌ Error loading the model or feature names.")
    st.code(str(e))
    st.stop()


# ============================================================
# THEME INITIALISATION
# ============================================================

if "theme" not in st.session_state:
    st.session_state.theme = "Light"

if "theme_picker" not in st.session_state:
    st.session_state.theme_picker = st.session_state.theme


THEME_OPTIONS = ["Light", "Dark"]
THEME_LABELS = {
    "Light": "Light",
    "Dark":  "Dark",
}


def get_theme_css(theme):
    if theme == "Dark":
        palette = dict(
            bg="#0e1117",
            surface="#1a1d23",
            surface_alt="#262930",
            text="#fafafa",
            text_muted="#9ca3af",
            border="#2e323a",
            accent="#3b82f6",
            info_bg="#0f172a",
            info_bd="#1d4ed8",
            success_bg="#052e16",
            success_bd="#15803d",
            warning_bg="#27270a",
            warning_bd="#a16207",
            error_bg="#3b0d0d",
            error_bd="#b91c1c",
            progress_track="#262930",
            color_scheme="dark",
        )
    else:
        palette = dict(
            bg="#ffffff",
            surface="#f8fafc",
            surface_alt="#f1f5f9",
            text="#111827",
            text_muted="#6b7280",
            border="#e5e7eb",
            accent="#2563eb",
            info_bg="#eff6ff",
            info_bd="#93c5fd",
            success_bg="#f0fdf4",
            success_bd="#86efac",
            warning_bg="#fefce8",
            warning_bd="#fde047",
            error_bg="#fef2f2",
            error_bd="#fecaca",
            progress_track="#e5e7eb",
            color_scheme="light",
        )

    p = palette
    return f"""
    <style>
        :root {{
            --bg: {p['bg']};
            --surface: {p['surface']};
            --surface-alt: {p['surface_alt']};
            --text: {p['text']};
            --text-muted: {p['text_muted']};
            --border: {p['border']};
            --accent: {p['accent']};
        }}

        html, body, #root,
        [data-testid="stAppViewContainer"],
        [data-testid="stApp"],
        .stApp,
        div[data-stale="false"],
        [data-testid="stMain"] {{
            background-color: {p['bg']} !important;
            background: {p['bg']} !important;
            color: {p['text']} !important;
        }}

        [data-testid="stApp"],
        [data-testid="stAppViewContainer"] {{
            color-scheme: {p['color_scheme']};
        }}

        /* Block container (main scroll area) */
        .block-container,
        [data-testid="stMainBlockContainer"],
        section.main {{
            background-color: {p['bg']} !important;
            background: {p['bg']} !important;
            color: {p['text']} !important;
        }}

        /* Titles, headers, paragraphs */
        .main-title,
        h1, h2, h3, h4, h5, h6, p, span,
        div[data-testid="stMarkdownContainer"] > div,
        div[data-testid="stMarkdownContainer"] p,
        div[data-testid="stMarkdownContainer"] h1,
        div[data-testid="stMarkdownContainer"] h2,
        div[data-testid="stMarkdownContainer"] h3,
        div[data-testid="stMarkdownContainer"] span,
        [data-testid="stCaptionContainer"],
        [data-testid="stCaptionContainer"] *,
        li, ol, ul,
        label,
        div[class*="StyledWidgetLabel"] {{
            color: {p['text']} !important;
        }}

        .subtitle, small, .caption,
        [data-testid="stCaptionContainer"] {{
            color: {p['text_muted']} !important;
        }}

        /* Cards / expanders / inputs surfaces */
        .section-card,
        [data-testid="stExpander"] details,
        [data-testid="stExpander"] summary,
        [data-testid="stExpander"] ul {{
            background-color: {p['surface']} !important;
            background: {p['surface']} !important;
            border: 1px solid {p['border']} !important;
            color: {p['text']} !important;
        }}

        /* Sidebar */
        [data-testid="stSidebar"],
        [data-testid="stSidebarContent"],
        [data-testid="stSidebarNav"],
        [data-testid="stSidebarUserContent"] {{
            background-color: {p['surface_alt']} !important;
            background: {p['surface_alt']} !important;
        }}
        section[data-testid="stSidebar"] {{
            background-color: {p['surface_alt']} !important;
            border-right: 1px solid {p['border']} !important;
        }}
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] li,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] small,
        [data-testid="stSidebar"] label {{
            color: {p['text']} !important;
        }}

        /* Widget labels */
        div[data-testid="stWidgetLabel"] p,
        div[data-testid="stWidgetLabel"] label {{
            color: {p['text']} !important;
        }}

        /* Number / text / select inputs (BaseWeb) */
        div[data-baseweb="input"] input,
        div[data-baseweb="input"] div,
        div[data-baseweb="input"]::before,
        div[data-baseweb="select"] > div,
        div[data-baseweb="select"] ul,
        div[data-baseweb="select"] li,
        div[data-baseweb="base-input"],
        div[data-baseweb="number-input"] input,
        textarea,
        div[class*="stNumberInput"] input,
        div[class*="stTextInput"] input,
        div[class*="stSelectbox"] div[data-baseweb="select"] > div,
        div[data-baseweb="textarea"] textarea {{
            background-color: {p['surface_alt']} !important;
            background: {p['surface_alt']} !important;
            color: {p['text']} !important;
            border-color: {p['border']} !important;
            caret-color: {p['text']} !important;
        }}

        /* BaseWeb select popout list */
        ul[role="listbox"],
        div[class*="popoverPopupContainer"] ul {{
            background-color: {p['surface_alt']} !important;
            color: {p['text']} !important;
            border: 1px solid {p['border']} !important;
        }}
        ul[role="listbox"] li,
        ul[role="listbox"] span {{
            color: {p['text']} !important;
        }}

        /* Buttons */
        div[data-testid="stButton"] button,
        div[data-testid="stFormSubmitButton"] button,
        button[kind="primary"],
        button[kind="secondary"],
        button[kind="tertiary"] {{
            color: {p['text']} !important;
            border-color: {p['border']} !important;
        }}

        /* SVG icons (select/radio) */
        div[data-baseweb="select"] svg,
        div[data-baseweb="radio"] svg,
        div[data-baseweb="checkbox"] svg {{
            fill: {p['text']} !important;
            color: {p['text']} !important;
        }}

        /* Radio text */
        div[data-baseweb="radio"] label span {{
            color: {p['text']} !important;
        }}

        /* Code blocks */
        code, pre,
        [data-testid="stCodeBlock"] pre,
        [data-testid="stCodeBlock"] code,
        .stCodeBlock,
        .stCodeBlock pre {{
            background-color: {p['surface_alt']} !important;
            background: {p['surface_alt']} !important;
            color: {p['text']} !important;
        }}

        /* Separators */
        hr, div[role="separator"] {{
            border-color: {p['border']} !important;
            background-color: {p['border']} !important;
        }}
        [data-testid="stVerticalBlock"] > div[style*="height: 1px"],
        [data-testid="stHorizontalBlock"] > div[style*="height: 1px"] {{
            background-color: {p['border']} !important;
        }}

        /* Metrics */
        [data-testid="stMetricLabel"],
        [data-testid="stMetricDelta"],
        [data-testid="stMetricValue"] > div,
        [data-testid="stMetricValue"] span {{
            color: {p['text']} !important;
        }}

        /* Progress bar track */
        [data-testid="stProgress"] > div {{
            background-color: {p['progress_track']} !important;
        }}

        /* Info / Success / Warning / Error boxes */
        div[data-testid="stAlert"] {{
            background-color: {p['surface']} !important;
            border: 1px solid {p['border']} !important;
            color: {p['text']} !important;
        }}
        div[data-testid="stAlertContainer"] > div,
        div[data-testid="stAlert"] div,
        div[data-testid="stAlert"] p,
        div[data-testid="stAlert"] span,
        div[data-testid="stAlert"] ul,
        div[data-testid="stAlert"] li {{
            color: {p['text']} !important;
        }}

        /* Toast notifications */
        div[class*="ToastView"],
        [data-testid="stToast"] {{
            background-color: {p['surface_alt']} !important;
            color: {p['text']} !important;
            border: 1px solid {p['border']} !important;
        }}

        /* Tooltips */
        div[role="tooltip"] {{
            background-color: {p['surface_alt']} !important;
            color: {p['text']} !important;
            border: 1px solid {p['border']} !important;
        }}

        /* Tabs */
        div[data-testid="stTabs"] [role="tab"] {{
            color: {p['text']} !important;
            background-color: transparent !important;
        }}
        div[data-testid="stTabs"] [role="tab"][aria-selected="true"] {{
            border-bottom: 2px solid {p['accent']} !important;
        }}

        /* =========================================================
           REMOVE "VIDE RECTANGLES" (empty dark horizontal stripes)
           - Keep Streamlit's real widget backgrounds (radios, buttons,
             st.info / st.success boxes, etc.) — do NOT blanket-transparent
             every child div, because that hides the radio circles.
           - Only target Streamlit's auto-inserted WRAPPER divs that apply
             alternating-row or "spacer" backgrounds between sections.
           - Also remove empty 1-2px gap divs Streamlit injects.
           ========================================================= */

        /* Reset Streamlit's per-wrapper alternating row background */
        [data-testid="stVerticalBlockBorderWrapper"],
        [data-testid="stVerticalBlockBorderWrapper"] > div,
        [data-testid="stColumn"] > div:first-child,
        [data-testid="stHorizontalBlock"] > div:first-child {{
            background-color: transparent !important;
            background: transparent !important;
            box-shadow: none !important;
            border: none !important;
        }}

        /* Hide genuinely empty divs (Streamlit's separator injects <div></div>) */
        section[data-testid="stSidebar"] div:empty,
        [data-testid="stSidebarUserContent"] div:empty,
        [data-testid="stVerticalBlock"] > div:empty,
        [data-testid="stHorizontalBlock"] > div:empty {{
            display: none !important;
            height: 0 !important;
            width: 0 !important;
            margin: 0 !important;
            padding: 0 !important;
            border: 0 !important;
        }}

        /* Collapse excessive top padding on sidebar user-content block
           (this often creates a dark gap above the first widget) */
        [data-testid="stSidebarUserContent"] {{
            padding-top: 0.25rem !important;
        }}

        /* Collapse excessive vertical spacing around st.markdown("---") HR */
        hr {{
            margin-top: 0.5rem !important;
            margin-bottom: 0.5rem !important;
        }}

        /* Remove box-shadows on alert / info boxes (keeps borders clean,
           but preserves their colored backgrounds) */
        div[data-testid="stAlert"] {{
            box-shadow: none !important;
        }}

        /* --- RADIO CIRCLES VISIBILITY (must NOT be transparent) --- */
        div[data-baseweb="radio"] {{
            background-color: transparent !important;
        }}
        /* Make the radio circle itself pop — outer ring + inner dot
           inherit BaseWeb defaults but remove any conflicting bg override */
        div[data-baseweb="radio"] div[role="radio"] {{
            background-color: transparent !important;
        }}
        div[data-baseweb="radio"] div[role="radio"][aria-checked="true"]::after,
        div[data-baseweb="radio"] div[role="radio"] svg {{
            /* keep checked dot visible */
            opacity: 1 !important;
        }}

        /* Ensure horizontal radio wrapper doesn't add a dark band */
        div[role="radiogroup"] {{
            background-color: transparent !important;
        }}

    </style>
    """


def apply_theme():
    st.markdown(
        get_theme_css(st.session_state.theme),
        unsafe_allow_html=True
    )


def on_theme_change():
    picked = st.session_state.theme_picker
    st.session_state.theme = picked


# Always apply the current theme CSS BEFORE any widgets render
apply_theme()

EXPECTED_FEATURES = len(feature_names)


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_dataset():

    dataset_path = None

    for path in DATASET_PATHS:
        if os.path.exists(path):
            dataset_path = path
            break

    if dataset_path is None:
        return None, None, None, None, None

    dataset = pd.read_csv(dataset_path)

    # Remove target and raw URL
    dataset_features = dataset.drop(
        columns=["status", "url"],
        errors="ignore"
    )

    # Check if all model features exist
    missing_features = [
        feature
        for feature in feature_names
        if feature not in dataset_features.columns
    ]

    if missing_features:
        raise ValueError(
            "The dataset is missing the following model features: "
            + ", ".join(missing_features)
        )

    # Keep exactly the features used by the trained model
    dataset_features = dataset_features[feature_names]

    # Pre-compute correct classifications for Section 2
    y_true = dataset["status"].values

    # Batch predict in chunks to avoid memory issues
    chunk_size = 1000
    all_preds = []
    all_probs = []

    for start in range(0, len(dataset_features), chunk_size):
        end = min(start + chunk_size, len(dataset_features))
        chunk = dataset_features.iloc[start:end]
        all_preds.append(model.predict(chunk))
        all_probs.append(model.predict_proba(chunk))

    all_preds = np.concatenate(all_preds)
    all_probs = np.vstack(all_probs)

    correct_mask = all_preds == y_true

    classes = list(model.classes_)
    legit_idx = classes.index("legitimate")
    phish_idx = classes.index("phishing")

    legit_correct_indices = np.where(
        (y_true == "legitimate") & correct_mask
    )[0].tolist()

    phish_correct_indices = np.where(
        (y_true == "phishing") & correct_mask
    )[0].tolist()

    return (
        dataset,
        dataset_features,
        dataset_path,
        legit_correct_indices,
        phish_correct_indices
    )


try:

    (
        dataset,
        dataset_features,
        dataset_path,
        legit_correct_indices,
        phish_correct_indices
    ) = load_dataset()

except Exception as e:

    st.error("❌ Error loading the dataset.")
    st.code(str(e))
    st.stop()


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #6b7280;
        margin-bottom: 30px;
    }

    .section-card {
        padding: 25px;
        border-radius: 15px;
        background-color: #f8fafc;
        border: 1px solid #e5e7eb;
        margin-bottom: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🛡️ Phishing Detector")

    st.markdown("---")

    st.write("### Appearance")

    st.radio(
        "Theme",
        options=THEME_OPTIONS,
        format_func=lambda opt: THEME_LABELS[opt],
        key="theme_picker",
        on_change=on_theme_change,
        horizontal=True,
        label_visibility="collapsed"
    )

    # Defensive sync: if user changed radio, force a rerun so CSS
    # re-applies IMMEDIATELY without waiting for next interaction.
    if st.session_state.theme_picker != st.session_state.theme:
        st.session_state.theme = st.session_state.theme_picker
        st.rerun()

    st.markdown("---")

    st.write("### About")

    st.write(
        "This application uses a Machine Learning "
        "Random Forest classifier to detect potentially "
        "phishing websites based on web page features."
    )

    st.markdown("---")

    st.write("### Model")

    st.info("🌲 Random Forest")

    st.write("Accuracy")

    st.success("96.5%")

    st.markdown("---")

    st.write("### Model Input")

    st.info(f"{len(feature_names)} numerical features")


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    '🛡️ Phishing Website Detection'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine Learning based website security analysis'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SECTION 1
# MANUAL TEST WITH 8 FEATURES
# ============================================================

st.markdown("---")

st.header("🔍 Test a Website")

st.write(
    "Enter the characteristics of the website below "
    "and let the model analyze it."
)

st.info(
    "📋 **Simplified Manual Test** — This demonstration interface "
    "shows 8 key features for quick manual testing. The remaining "
    f"{len(feature_names) - len(important_features)} model features "
    "are set to zero for this section. This is a simplified demo, "
    "not a complete representation of all model inputs. "
    "Use Section 2 below for a complete test with all real features."
)


# ============================================================
# FEATURE INPUTS — 8 IMPORTANT FEATURES IN 2 COLUMNS
# ============================================================

st.subheader("🔢 Website Characteristics")

user_values_section1 = {}

col1, col2 = st.columns(2)

for i, feature in enumerate(important_features):

    target_col = col1 if (i % 2 == 0) else col2

    with target_col:

        user_values_section1[feature] = st.number_input(
            f"**{feature}**",
            value=0.0,
            format="%.6f",
            key=f"s1_feature_{i}"
        )


# ============================================================
# PREDICTION BUTTON — SECTION 1
# ============================================================

st.markdown("---")

predict_button_1 = st.button(
    "🔍 Analyze Website",
    use_container_width=True,
    type="primary"
)


if predict_button_1:

    # Build the full {EXPECTED_FEATURES}-feature vector
    # User enters 8 features, rest are filled with 0
    section1_input_dict = {}

    for feature in feature_names:

        if feature in user_values_section1:
            section1_input_dict[feature] = user_values_section1[feature]
        else:
            section1_input_dict[feature] = 0.0

    # Create input DataFrame
    input_data_1 = pd.DataFrame(
        [
            [
                section1_input_dict[feature]
                for feature in feature_names
            ]
        ],
        columns=feature_names
    )

    # Make prediction
    prediction_1 = model.predict(input_data_1)[0]

    # Get probabilities
    probabilities_1 = model.predict_proba(input_data_1)[0]

    classes_1 = list(model.classes_)

    phishing_probability_1 = probabilities_1[
        classes_1.index("phishing")
    ]

    legitimate_probability_1 = probabilities_1[
        classes_1.index("legitimate")
    ]


    # ========================================================
    # DISPLAY PREDICTION RESULT — SECTION 1
    # ========================================================

    st.markdown("---")

    st.subheader("📊 Prediction Result")

    result_col1_1, result_col2_1 = st.columns(2)


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    with result_col1_1:

        if prediction_1 == "phishing":

            st.error("⚠️ PHISHING")

            st.write(
                "The model classifies this observation "
                "as potentially phishing."
            )

        else:

            st.success("✅ LEGITIMATE")

            st.write(
                "The model classifies this observation "
                "as legitimate."
            )


    # --------------------------------------------------------
    # Probabilities
    # --------------------------------------------------------

    with result_col2_1:

        st.metric(
            "Phishing Probability",
            f"{phishing_probability_1:.2%}"
        )

        st.metric(
            "Legitimate Probability",
            f"{legitimate_probability_1:.2%}"
        )


    st.progress(
        float(phishing_probability_1)
    )

    st.caption(
        "💡 Interpretation: Higher phishing probability means "
        "the model is more confident the site is malicious. "
        "A very high legitimate probability indicates the model "
        "considers the site safe."
    )


# ============================================================
# SECTION 2
# TEST WITH A REAL DATASET OBSERVATION
# ============================================================

st.markdown("---")

st.header("🧪 Test with a Dataset Observation")

st.write(
    "Test the existing Random Forest model using "
    "a real observation from the dataset. "
    f"All {len(feature_names)} real features are passed "
    "to the model exactly as trained."
)

st.info(
    "💡 This section is the important demonstration mode. "
    "It uses real values from the actual dataset and allows "
    "you to show both a correctly classified legitimate "
    "example and a correctly classified phishing example."
)


# ============================================================
# DATASET AVAILABLE
# ============================================================

if dataset is None:

    st.warning(
        "⚠️ The dataset could not be found."
    )

    st.write(
        "Please make sure that "
        "`dataset_B_05_2020.csv` is located in the project "
        "directory or in one of the configured dataset folders."
    )

else:

    # --------------------------------------------------------
    # Dataset information
    # --------------------------------------------------------

    st.caption(
        f"Dataset loaded: {len(dataset):,} observations | "
        f"{len(feature_names)} model features | "
        f"Classes: Legitimate / Phishing balanced"
    )


    # ========================================================
    # CLASS SELECTOR + OBSERVATION SELECTOR
    # ========================================================

    st.subheader("📌 Select an Observation")

    selector_col1, selector_col2 = st.columns(2)

    with selector_col1:
        selected_class = st.selectbox(
            "Select class to demonstrate:",
            options=["Legitimate", "Phishing"],
            index=0,
            key="s2_class_selector",
            help="Choose Legitimate or Phishing class to find "
                 "a correctly classified real example."
        )

    if selected_class == "Legitimate":
        available_indices = legit_correct_indices
    else:
        available_indices = phish_correct_indices

    # Build display labels
    observation_labels = [
        f"Observation {idx + 1}"
        for idx in available_indices
    ]

    with selector_col2:
        selected_label = st.selectbox(
            "Select observation:",
            options=observation_labels,
            index=0,
            key="s2_obs_selector",
            help="Choose from real dataset observations that the "
                 "Random Forest model correctly classifies."
        )

    # Extract selected 0-based index
    selected_label_idx = observation_labels.index(selected_label)
    selected_dataset_index = available_indices[selected_label_idx]


    # ========================================================
    # REAL CLASS — BEFORE TESTING
    # ========================================================

    st.markdown("---")

    real_class_2 = dataset.iloc[selected_dataset_index]["status"]

    real_label = str(real_class_2).upper()

    if real_class_2 == "legitimate":
        st.success(f"Real class: ✅ {real_label}")
    else:
        st.error(f"Real class: ⚠️ {real_label}")


    # ========================================================
    # SHOW 8 IMPORTANT FEATURE VALUES (preview)
    # ========================================================

    with st.expander("👀 View key feature values for this observation"):
        obs_row = dataset.iloc[selected_dataset_index]
        prev_col1, prev_col2 = st.columns(2)
        for i, feat in enumerate(important_features):
            val = obs_row[feat]
            target = prev_col1 if (i % 2 == 0) else prev_col2
            with target:
                st.write(f"**{feat}**: `{val}`")


    # ========================================================
    # TEST BUTTON
    # ========================================================

    st.markdown("")

    test_button_2 = st.button(
        "🧪 Test Observation",
        use_container_width=True,
        type="primary"
    )


    if test_button_2:

        # Get the selected observation with exact feature order
        input_data_2 = dataset_features.iloc[
            [selected_dataset_index]
        ].copy()

        input_data_2 = input_data_2[feature_names]


        # ====================================================
        # MODEL PREDICTION
        # ====================================================

        prediction_2 = model.predict(input_data_2)[0]

        probabilities_2 = model.predict_proba(input_data_2)[0]

        classes_2 = list(model.classes_)

        phishing_probability_2 = probabilities_2[
            classes_2.index("phishing")
        ]

        legitimate_probability_2 = probabilities_2[
            classes_2.index("legitimate")
        ]

        predicted_label = str(prediction_2).upper()


        # ====================================================
        # RESULT
        # ====================================================

        st.markdown("---")

        st.subheader("📊 Observation Test Result")

        st.caption(
            f"Analysis of **{selected_label}** — "
            f"real values from the dataset with all "
            f"{len(feature_names)} features."
        )

        st.markdown("")

        result_col1_2, result_col2_2 = st.columns(2)


        # ----------------------------------------------------
        # Real class and Predicted class
        # ----------------------------------------------------

        with result_col1_2:

            if real_class_2 == "legitimate":
                st.info(f"Real class: ✅ {real_label}")
            else:
                st.info(f"Real class: ⚠️ {real_label}")

            st.markdown("")

            if prediction_2 == "phishing":

                st.error(f"Predicted class: ⚠️ {predicted_label}")

                st.write(
                    "The Random Forest model predicts this "
                    "observation is **phishing**."
                )

            else:

                st.success(f"Predicted class: ✅ {predicted_label}")

                st.write(
                    "The Random Forest model predicts this "
                    "observation is **legitimate**."
                )


        # ----------------------------------------------------
        # Probabilities
        # ----------------------------------------------------

        with result_col2_2:

            st.metric(
                "Phishing Probability",
                f"{phishing_probability_2:.2%}"
            )

            st.metric(
                "Legitimate Probability",
                f"{legitimate_probability_2:.2%}"
            )


        st.markdown("")

        st.progress(
            float(phishing_probability_2)
        )

        st.caption(
            "The progress bar visualizes the phishing probability "
            "from 0% (far left, safe) to 100% (far right, malicious)."
        )


        # ====================================================
        # COMPARE REAL CLASS WITH PREDICTION
        # ====================================================

        st.markdown("")

        if prediction_2 == real_class_2:

            st.success(
                "✅ Correct classification — "
                "the Random Forest prediction matches the "
                "real dataset label. This is a reliable example "
                "for the presentation."
            )

        else:

            st.warning(
                "⚠️ Misclassification — "
                "the model prediction does not match "
                "the real dataset label. Try a different observation."
            )

        st.markdown("")

        # Model confidence explanation
        confidence = max(phishing_probability_2, legitimate_probability_2)
        if confidence >= 0.9:
            conf_label = "Very High Confidence"
        elif confidence >= 0.75:
            conf_label = "High Confidence"
        elif confidence >= 0.6:
            conf_label = "Moderate Confidence"
        else:
            conf_label = "Low / Uncertain Confidence"

        with st.expander(
            f"📖 Model Confidence Explanation — {conf_label}"
        ):
            st.write(
                f"The model assigns a confidence of **{confidence:.2%}** "
                f"to its {predicted_label.lower()} prediction. "
                "Confidence corresponds to how strongly the Random Forest "
                "trees collectively vote for the winning class. "
                f"Since this observation is from the pre-screened "
                f"correctly-classified list, you can use it as a clear "
                f"demonstration of the model working correctly on a real "
                f"{real_class_2.lower()} example."
            )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Phishing Website Detection | "
    "Random Forest | "
    f"{len(feature_names)} Features"
)

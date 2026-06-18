import re
import unicodedata
from collections import defaultdict

import pandas as pd
import streamlit as st
from rapidfuzz import fuzz, process


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Supplier Normalization Workbench",
    layout="wide",
)


# ============================================================
# STYLING
# ============================================================

def apply_styles():
    st.markdown(
        """
        <style>
        .hero-card {
            background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 60%, #312e81 100%);
            padding: 2rem;
            border-radius: 1.2rem;
            color: white;
            margin-bottom: 1.2rem;
            box-shadow: 0 18px 45px rgba(15, 23, 42, 0.25);
        }

        .hero-label {
            text-transform: uppercase;
            letter-spacing: 0.25em;
            font-size: 0.75rem;
            color: #bfdbfe;
            margin-bottom: 0.8rem;
            font-weight: 700;
        }

        .hero-title {
            color: white;
            margin-bottom: 0.6rem;
            font-size: 2.1rem;
            font-weight: 850;
            line-height: 1.1;
        }

        .hero-subtitle {
            color: #dbeafe;
            font-size: 1rem;
            line-height: 1.6;
            max-width: 1100px;
        }

        .info-card {
            background-color: white;
            padding: 1rem 1.1rem;
            border-radius: 0.9rem;
            border: 1px solid #e2e8f0;
            box-shadow: 0 8px 22px rgba(15, 23, 42, 0.06);
            margin-bottom: 1rem;
            color: #1a1a1a;
        }

        div[data-testid="stMetricValue"] {
            font-size: 1.55rem;
            font-weight: 800;
        }

        div[data-testid="stMetricLabel"] {
            color: #475569;
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# FORMAT HELPERS
# ============================================================

def format_currency(value):
    try:
        value = float(value)
    except Exception:
        return "Not available"

    if abs(value) >= 1_000_000_000:
        return f"${value / 1_000_000_000:.1f}B"
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:.1f}M"
    if abs(value) >= 1_000:
        return f"${value / 1_000:.1f}K"
    return f"${value:,.0f}"


def format_number(value):
    try:
        return f"{float(value):,.0f}"
    except Exception:
        return "Not available"


def clean_display_columns(df):
    display = df.copy()
    display.columns = [str(col).replace("_", " ").title() for col in display.columns]
    return display


# ============================================================
# CONFIG
# ============================================================

COMMON_SUFFIXES = [
    "inc", "incorporated", "llc", "ltd", "limited", "corp", "corporation",
    "co", "company", "plc", "lp", "llp", "gmbh", "ag", "sa", "sarl",
    "services", "service", "group", "holdings", "holding", "the",
    "bv", "nv", "srl", "ltda", "pty", "pte", "sdn", "bhd", "ab",
]

ABBREVIATION_MAP = {
    "mfg": "manufacturing",
    "mfr": "manufacturing",
    "intl": "international",
    "int": "international",
    "svcs": "services",
    "svc": "service",
    "mgmt": "management",
    "dist": "distribution",
    "tech": "technologies",
    "sys": "systems",
    "assoc": "associates",
    "bros": "brothers",
    "grp": "group",
    "soln": "solution",
    "solns": "solutions",
    "dept": "department",
    "hldgs": "holdings",
    "hlgs": "holdings",
    "ctr": "center",
    "natl": "national",
    "amer": "american",
    "engr": "engineering",
    "engg": "engineering",
    "inds": "industries",
    "corp": "corporation",
    "co": "company",
}

METADATA_PATTERNS = [
    r"\(old\)",
    r"\(inactive\)",
    r"\(duplicate\)",
    r"\(blocked\)",
    r"\(merged\)",
    r"\(legacy\)",
    r"\[old\]",
    r"\[inactive\]",
    r"\[duplicate\]",
    r"\[blocked\]",
    r"\[merged\]",
    r"\[legacy\]",
    r"\bdo not use\b",
    r"\binactive\b",
    r"\bblocked\b",
    r"\blegacy\b",
    r"\bduplicate of\b.*",
    r"\bformerly known as\b.*",
    r"\bfka\b.*",
]

KNOWN_ALIASES = {
    "ibm": "IBM",
    "i b m": "IBM",
    "international business machines": "IBM",

    "aws": "Amazon / AWS",
    "amazon web services": "Amazon / AWS",
    "amazon": "Amazon / AWS",
    "amazon com": "Amazon / AWS",

    "microsoft": "Microsoft",
    "microsoft azure": "Microsoft",
    "msft": "Microsoft",

    "google": "Google / Alphabet",
    "google cloud": "Google / Alphabet",
    "alphabet": "Google / Alphabet",
    "youtube": "Google / Alphabet",

    "dhl": "DHL",
    "dhl express": "DHL",
    "dhl global forwarding": "DHL",

    "fedex": "FedEx",
    "federal express": "FedEx",

    "ups": "UPS",
    "united parcel service": "UPS",

    "oracle": "Oracle",
    "oracle america": "Oracle",

    "sap": "SAP",
    "sap america": "SAP",

    "grainger": "Grainger",
    "ww grainger": "Grainger",
    "w w grainger": "Grainger",

    "fastenal": "Fastenal",
    "staples": "Staples",
    "office depot": "Office Depot",

    "3m": "3M",
    "3 m": "3M",
}


# ============================================================
# CLEANING FUNCTIONS
# ============================================================

def normalize_unicode_text(value):
    if pd.isna(value):
        return ""

    text = str(value)
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    return text


def strip_metadata_annotations(text):
    cleaned = text

    for pattern in METADATA_PATTERNS:
        cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)

    return cleaned


def expand_abbreviations(tokens):
    return [ABBREVIATION_MAP.get(token, token) for token in tokens]


def clean_supplier_name(name):
    if pd.isna(name):
        return ""

    cleaned = normalize_unicode_text(name)
    cleaned = cleaned.lower().strip()

    cleaned = strip_metadata_annotations(cleaned)

    cleaned = re.sub(r"&", " and ", cleaned)
    cleaned = re.sub(r"\b3[\s\-]?m\b", "3m", cleaned)
    cleaned = re.sub(r"[^a-z0-9\s]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    tokens = cleaned.split()
    tokens = expand_abbreviations(tokens)
    tokens = [token for token in tokens if token not in COMMON_SUFFIXES]

    return " ".join(tokens).strip()


def alias_lookup(name):
    cleaned = clean_supplier_name(name)
    return KNOWN_ALIASES.get(cleaned)


def choose_canonical_name(original_names, spend_lookup):
    valid_names = [str(name).strip() for name in original_names if str(name).strip()]

    if not valid_names:
        return "Unknown Supplier"

    return sorted(
        valid_names,
        key=lambda supplier: spend_lookup.get(supplier, 0),
        reverse=True,
    )[0]


def confidence_tier(score, method, variant_count=1):
    """
    Converts raw fuzzy score into a business-readable confidence tier.
    Standalone suppliers are not duplicates.
    """
    if variant_count == 1:
        return "Standalone Supplier"

    if method == "Known alias":
        return "Confirmed Duplicate"

    if score >= 95:
        return "Confirmed Duplicate"

    if score >= 88:
        return "Probable Duplicate"

    if score >= 75:
        return "Possible Duplicate"

    return "Not a Match"


def false_positive_risk(score, variant_count, category_conflict=False, country_conflict=False):
    risk = "Low"

    if variant_count == 1:
        return "Low"

    if score < 90:
        risk = "Medium"

    if score < 85:
        risk = "High"

    if category_conflict or country_conflict:
        risk = "High"

    return risk


def recommended_action(score, method, false_positive_risk_value, variant_count):
    if variant_count == 1:
        return "No duplicate action needed"

    if method == "Known alias":
        return "Suggested merge"

    if false_positive_risk_value == "High":
        return "Human review required"

    if score >= 95:
        return "Suggested merge"

    if score >= 88:
        return "Review before merge"

    if score >= 75:
        return "Investigate only"

    return "Do not merge"


def build_match_explanation(row):
    variant_count = row.get("variant_count", 0)

    if variant_count <= 1:
        return (
            "This supplier was not grouped with another supplier name. "
            "No duplicate match was identified based on the current matching threshold."
        )

    reasons = []

    if "KNOWN_ALIAS" in str(row.get("reason_codes", "")):
        reasons.append("a known supplier alias rule matched")

    if "FUZZY_NAME_MATCH" in str(row.get("reason_codes", "")):
        reasons.append(
            f"supplier names were similar after cleaning with an average match score of {row.get('average_match_score')}"
        )

    if "EXACT_CLEANED_NAME" in str(row.get("reason_codes", "")):
        reasons.append("supplier names matched exactly after cleaning")

    reasons.append(f"{variant_count} supplier-name variants were grouped together")

    categories = str(row.get("categories_detected", ""))
    countries = str(row.get("countries_detected", ""))

    if "," in categories:
        reasons.append("multiple categories were detected, increasing false-positive review risk")

    if "," in countries:
        reasons.append("multiple countries were detected, increasing hierarchy/subsidiary review risk")

    if row.get("false_positive_risk") == "High":
        reasons.append("false-positive risk is high, so this should not be auto-merged")

    return "This group was flagged because " + "; ".join(reasons) + "."


# ============================================================
# DEMO DATA
# ============================================================

def build_demo_data():
    rows = [
        ["IBM", 900000, "IT Services", "United States", "V001"],
        ["I.B.M. Corp", 450000, "IT Services", "United States", "V002"],
        ["International Business Machines", 700000, "IT Services", "United States", "V003"],

        ["Amazon Web Services", 1250000, "Cloud", "United States", "V004"],
        ["AWS", 540000, "Cloud", "United States", "V005"],
        ["Amazon.com", 210000, "Office Supplies", "United States", "V006"],

        ["Microsoft Corp", 1800000, "Software", "United States", "V007"],
        ["MSFT", 350000, "Software", "United States", "V008"],
        ["Microsoft Azure", 950000, "Cloud", "United States", "V009"],

        ["DHL Express", 760000, "Logistics", "United States", "V010"],
        ["D.H.L.", 310000, "Logistics", "United States", "V011"],
        ["DHL Global Forwarding", 420000, "Logistics", "United States", "V012"],

        ["Cascade Mfg Co.", 180000, "MRO", "United States", "V013"],
        ["Cascade Manufacturing Company", 95000, "MRO", "United States", "V014"],

        ["Muller Industrial GmbH", 260000, "Industrial Supplies", "Germany", "V015"],
        ["Müller Industrial GmbH", 180000, "Industrial Supplies", "Germany", "V016"],

        ["Apex Tech Inc - DO NOT USE", 125000, "IT Services", "United States", "V017"],
        ["Apex Technologies Incorporated", 140000, "IT Services", "United States", "V018"],

        ["ABC Logistics LLC", 180000, "Logistics", "United States", "V019"],
        ["ABC Logistic Services", 95000, "Logistics", "United States", "V020"],
        ["ABC Consulting LLC", 220000, "Professional Services", "United States", "V021"],

        ["Delta Air Lines", 500000, "Travel", "United States", "V022"],
        ["Delta Dental", 150000, "Benefits", "United States", "V023"],

        ["National Freight Services", 620000, "Logistics", "United States", "V024"],
        ["National Office Supplies", 120000, "Office Supplies", "United States", "V025"],

        ["Local HVAC Repair Co", 85000, "Facilities", "United States", "V026"],
        ["Local H.V.A.C. Repair", 92000, "Facilities", "United States", "V027"],

        ["Staples Inc", 260000, "Office Supplies", "United States", "V028"],
        ["Staples", 190000, "Office Supplies", "United States", "V029"],
        ["Office Depot", 240000, "Office Supplies", "United States", "V030"],
    ]

    return pd.DataFrame(
        rows,
        columns=["supplier_name", "spend", "category", "country", "supplier_id"],
    )


# ============================================================
# FILE LOADING
# ============================================================

def load_file(uploaded_file):
    if uploaded_file is None:
        return None

    file_name = uploaded_file.name.lower()

    if file_name.endswith(".csv"):
        return pd.read_csv(uploaded_file)

    if file_name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)

    raise ValueError("Unsupported file type. Please upload CSV or Excel.")


# ============================================================
# MATCHING ENGINE
# ============================================================

def normalize_suppliers(
    df,
    supplier_col,
    spend_col=None,
    category_col=None,
    country_col=None,
    threshold=88,
):
    data = df.copy()

    data["original_supplier_name"] = data[supplier_col].fillna("Unknown Supplier").astype(str)
    data["clean_supplier_name"] = data["original_supplier_name"].apply(clean_supplier_name)

    if spend_col and spend_col in data.columns:
        data["spend_value"] = (
            data[spend_col]
            .astype(str)
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
        )
        data["spend_value"] = pd.to_numeric(data["spend_value"], errors="coerce").fillna(0)
    else:
        data["spend_value"] = 0

    if category_col and category_col in data.columns:
        data["category_value"] = data[category_col].fillna("Unknown").astype(str)
    else:
        data["category_value"] = "Unknown"

    if country_col and country_col in data.columns:
        data["country_value"] = data[country_col].fillna("Unknown").astype(str)
    else:
        data["country_value"] = "Unknown"

    spend_lookup = (
        data.groupby("original_supplier_name")["spend_value"]
        .sum()
        .to_dict()
    )

    original_suppliers = sorted(data["original_supplier_name"].dropna().astype(str).unique())

    mapping = {}
    scores = {}
    methods = {}
    reason_codes = {}

    unresolved = []

    # Step 1: known alias matching
    for supplier in original_suppliers:
        alias = alias_lookup(supplier)

        if alias:
            mapping[supplier] = alias
            scores[supplier] = 100
            methods[supplier] = "Known alias"
            reason_codes[supplier] = "KNOWN_ALIAS"
        else:
            unresolved.append(supplier)

    # Step 2: exact cleaned-name match
    cleaned_to_originals = defaultdict(list)

    for supplier in unresolved:
        cleaned = clean_supplier_name(supplier)

        if not cleaned:
            mapping[supplier] = "Unknown Supplier"
            scores[supplier] = 0
            methods[supplier] = "Missing supplier name"
            reason_codes[supplier] = "MISSING_NAME"
        else:
            cleaned_to_originals[cleaned].append(supplier)

    fuzzy_candidates = []

    for cleaned_name, originals in cleaned_to_originals.items():
        if len(originals) > 1:
            canonical = choose_canonical_name(originals, spend_lookup)

            for supplier in originals:
                mapping[supplier] = canonical
                scores[supplier] = 100
                methods[supplier] = "Exact cleaned match"
                reason_codes[supplier] = "EXACT_CLEANED_NAME"
        else:
            fuzzy_candidates.append(cleaned_name)

    # Step 3: fuzzy matching among unresolved cleaned names
    assigned_cleaned_names = set()

    for cleaned_name in fuzzy_candidates:
        if cleaned_name in assigned_cleaned_names:
            continue

        matches = process.extract(
            cleaned_name,
            fuzzy_candidates,
            scorer=fuzz.token_set_ratio,
            score_cutoff=threshold,
            limit=None,
        )

        matched_cleaned_names = [match[0] for match in matches]
        matched_scores = [match[1] for match in matches]

        for matched_name in matched_cleaned_names:
            assigned_cleaned_names.add(matched_name)

        original_group = []

        for matched_name in matched_cleaned_names:
            original_group.extend(cleaned_to_originals[matched_name])

        canonical = choose_canonical_name(original_group, spend_lookup)

        if len(original_group) > 1:
            avg_score = round(sum(matched_scores) / len(matched_scores), 1)
        else:
            avg_score = 0

        for supplier in original_group:
            mapping[supplier] = canonical
            scores[supplier] = avg_score

            if len(original_group) > 1:
                methods[supplier] = "Fuzzy match"
                reason_codes[supplier] = "FUZZY_NAME_MATCH"
            else:
                methods[supplier] = "No duplicate found"
                reason_codes[supplier] = "STANDALONE_SUPPLIER"

    data["normalized_supplier_name"] = data["original_supplier_name"].map(mapping)
    data["match_score"] = data["original_supplier_name"].map(scores)
    data["match_method"] = data["original_supplier_name"].map(methods)
    data["reason_code"] = data["original_supplier_name"].map(reason_codes)

    data["variant_count_for_normalized_supplier"] = data.groupby(
        "normalized_supplier_name"
    )["original_supplier_name"].transform("nunique")

    data["confidence_tier"] = data.apply(
        lambda row: confidence_tier(
            row["match_score"],
            row["match_method"],
            row["variant_count_for_normalized_supplier"],
        ),
        axis=1,
    )

    # Group-level summary
    group_rows = []

    for normalized_name, group_df in data.groupby("normalized_supplier_name", dropna=False):
        variants = sorted(group_df["original_supplier_name"].dropna().astype(str).unique())
        total_spend = group_df["spend_value"].sum()
        avg_score = group_df["match_score"].mean()

        categories = sorted(group_df["category_value"].dropna().astype(str).unique())
        countries = sorted(group_df["country_value"].dropna().astype(str).unique())

        category_conflict = len([cat for cat in categories if cat.lower() != "unknown"]) > 1
        country_conflict = len([country for country in countries if country.lower() != "unknown"]) > 1

        fp_risk = false_positive_risk(
            avg_score,
            len(variants),
            category_conflict=category_conflict,
            country_conflict=country_conflict,
        )

        if len(variants) == 1:
            group_method = "Standalone"
        elif "Known alias" in group_df["match_method"].values:
            group_method = "Known alias"
        else:
            group_method = "Fuzzy match"

        tier = confidence_tier(avg_score, group_method, len(variants))

        action = recommended_action(
            avg_score,
            group_method,
            fp_risk,
            len(variants),
        )

        if len(variants) == 1:
            review_status = "No Review Needed"
        elif fp_risk == "High":
            review_status = "Needs Review"
        elif avg_score < 90:
            review_status = "Needs Review"
        else:
            review_status = "Suggested Merge"

        group_rows.append(
            {
                "match_group_id": f"MG-{len(group_rows) + 1:04d}",
                "normalized_supplier_name": normalized_name,
                "original_supplier_variants": ", ".join(variants),
                "variant_count": len(variants),
                "total_spend": total_spend,
                "average_match_score": round(avg_score, 1),
                "confidence_tier": tier,
                "recommended_action": action,
                "match_methods": ", ".join(sorted(group_df["match_method"].dropna().unique())),
                "reason_codes": ", ".join(sorted(group_df["reason_code"].dropna().unique())),
                "categories_detected": ", ".join(categories),
                "countries_detected": ", ".join(countries),
                "false_positive_risk": fp_risk,
                "review_status": review_status,
            }
        )

    group_summary = pd.DataFrame(group_rows)

    if not group_summary.empty:
        group_summary = group_summary.sort_values(
            ["variant_count", "total_spend"],
            ascending=[False, False],
        )

        group_summary["match_explanation"] = group_summary.apply(build_match_explanation, axis=1)

    return data, group_summary


# ============================================================
# GOLDEN RECORD BUILDER
# ============================================================

def build_golden_records(normalized_data):
    rows = []

    for normalized_name, group_df in normalized_data.groupby("normalized_supplier_name", dropna=False):
        variants = sorted(group_df["original_supplier_name"].dropna().astype(str).unique())
        highest_spend_row = group_df.sort_values("spend_value", ascending=False).iloc[0]

        categories = group_df["category_value"].dropna().astype(str)
        countries = group_df["country_value"].dropna().astype(str)

        best_category = categories.mode().iloc[0] if not categories.empty else "Unknown"
        best_country = countries.mode().iloc[0] if not countries.empty else "Unknown"

        rows.append(
            {
                "normalized_supplier_name": normalized_name,
                "recommended_golden_name": normalized_name,
                "source_name_with_highest_spend": highest_spend_row["original_supplier_name"],
                "best_category": best_category,
                "best_country": best_country,
                "variant_count": len(variants),
                "total_spend": group_df["spend_value"].sum(),
                "survivorship_logic": (
                    "Name from normalized family; category/country selected by most frequent value; "
                    "spend aggregated across variants."
                ),
                "review_required": "Yes" if len(variants) > 1 else "No",
            }
        )

    golden = pd.DataFrame(rows)

    if not golden.empty:
        golden = golden.sort_values("total_spend", ascending=False)

    return golden


# ============================================================
# DATA QUALITY SCORE
# ============================================================

def calculate_data_quality(
    df,
    group_summary,
    supplier_col,
    spend_col=None,
    category_col=None,
    country_col=None,
):
    supplier_completeness = df[supplier_col].notna().mean() * 100 if supplier_col in df.columns else 0

    if spend_col and spend_col in df.columns:
        spend_numeric = pd.to_numeric(
            df[spend_col]
            .astype(str)
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False),
            errors="coerce",
        )
        spend_usability = spend_numeric.notna().mean() * 100
    else:
        spend_usability = 0

    category_completeness = (
        df[category_col].notna().mean() * 100
        if category_col and category_col in df.columns
        else 0
    )

    country_completeness = (
        df[country_col].notna().mean() * 100
        if country_col and country_col in df.columns
        else 0
    )

    duplicate_groups = int((group_summary["variant_count"] > 1).sum()) if not group_summary.empty else 0
    review_groups = int((group_summary["review_status"] == "Needs Review").sum()) if not group_summary.empty else 0

    duplicate_score = max(50, 100 - duplicate_groups * 3)
    review_burden_score = max(50, 100 - review_groups * 5)

    dimensions = [
        ("Supplier name completeness", supplier_completeness),
        ("Spend field usability", spend_usability),
        ("Category completeness", category_completeness),
        ("Country completeness", country_completeness),
        ("Duplicate supplier risk", duplicate_score),
        ("Manual review burden", review_burden_score),
    ]

    overall_score = round(sum(score for _, score in dimensions) / len(dimensions), 1)

    return {
        "overall_score": overall_score,
        "dimensions": pd.DataFrame(dimensions, columns=["dimension", "score"]),
        "duplicate_groups": duplicate_groups,
        "review_groups": review_groups,
        "total_records": len(df),
    }


# ============================================================
# MAIN APP
# ============================================================

def main():
    apply_styles()

    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-label">Sid's AI Portfolio - Procurement Data Quality Accelerator</div>
            <div class="hero-title">Supplier Normalization & Duplicate Detection Workbench</div>
            <div class="hero-subtitle">
                Upload messy supplier data, detect duplicate vendor records, normalize supplier families,
                generate a human review queue, build golden record recommendations, and export cleaner supplier
                data for downstream spend analytics.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ========================================================
    # SIDEBAR: INPUT
    # ========================================================

    st.sidebar.header("Data Input")

    input_mode = st.sidebar.radio(
        "Choose input mode",
        ["Use demo data", "Upload file"],
    )

    if input_mode == "Use demo data":
        raw_df = build_demo_data()
        st.sidebar.success("Demo data loaded.")
    else:
        uploaded_file = st.sidebar.file_uploader(
            "Upload supplier file",
            type=["csv", "xlsx", "xls"],
        )

        if uploaded_file is None:
            st.info("Upload a CSV/Excel file or switch to demo data.")
            return

        try:
            raw_df = load_file(uploaded_file)
        except Exception as e:
            st.error(f"Could not load file: {e}")
            return

    if raw_df is None or raw_df.empty:
        st.warning("No data available.")
        return

    # ========================================================
    # SIDEBAR: COLUMN MAPPING
    # ========================================================

    st.sidebar.header("Column Mapping")

    columns = list(raw_df.columns)

    default_supplier = columns.index("supplier_name") if "supplier_name" in columns else 0

    supplier_col = st.sidebar.selectbox(
        "Supplier name column",
        columns,
        index=default_supplier,
    )

    optional_columns = ["None"] + columns

    spend_default = optional_columns.index("spend") if "spend" in optional_columns else 0
    spend_col = st.sidebar.selectbox(
        "Spend column optional",
        optional_columns,
        index=spend_default,
    )
    spend_col = None if spend_col == "None" else spend_col

    category_default = optional_columns.index("category") if "category" in optional_columns else 0
    category_col = st.sidebar.selectbox(
        "Category column optional",
        optional_columns,
        index=category_default,
    )
    category_col = None if category_col == "None" else category_col

    country_default = optional_columns.index("country") if "country" in optional_columns else 0
    country_col = st.sidebar.selectbox(
        "Country column optional",
        optional_columns,
        index=country_default,
    )
    country_col = None if country_col == "None" else country_col

    # ========================================================
    # SIDEBAR: MATCH SETTINGS
    # ========================================================

    st.sidebar.header("Matching Settings")

    matching_mode = st.sidebar.radio(
        "Matching mode",
        ["Conservative", "Balanced", "Aggressive"],
        index=1,
    )

    default_thresholds = {
        "Conservative": 92,
        "Balanced": 88,
        "Aggressive": 82,
    }

    threshold = st.sidebar.slider(
        "Fuzzy matching threshold",
        min_value=75,
        max_value=98,
        value=default_thresholds[matching_mode],
        step=1,
    )

    st.sidebar.caption(
        "Higher threshold reduces false positives. Lower threshold finds more possible duplicates but increases review burden."
    )

    # ========================================================
    # RUN DIAGNOSTIC
    # ========================================================

    normalized_data, group_summary = normalize_suppliers(
        raw_df,
        supplier_col=supplier_col,
        spend_col=spend_col,
        category_col=category_col,
        country_col=country_col,
        threshold=threshold,
    )

    golden_records = build_golden_records(normalized_data)

    data_quality = calculate_data_quality(
        raw_df,
        group_summary=group_summary,
        supplier_col=supplier_col,
        spend_col=spend_col,
        category_col=category_col,
        country_col=country_col,
    )

    before_supplier_count = raw_df[supplier_col].nunique()
    after_supplier_count = normalized_data["normalized_supplier_name"].nunique()
    duplicate_groups = data_quality["duplicate_groups"]
    review_groups = data_quality["review_groups"]

    spend_affected = 0
    if not group_summary.empty:
        spend_affected = group_summary[group_summary["variant_count"] > 1]["total_spend"].sum()

    # ========================================================
    # TABS
    # ========================================================

    tab1, tab2, tab3 = st.tabs(
        [
            "Executive Summary",
            "Match Review",
            "Output & Methodology",
        ]
    )

    # ========================================================
    # TAB 1: EXECUTIVE SUMMARY
    # ========================================================

    with tab1:
        st.subheader("Executive Data Quality Summary")

        metric_cols = st.columns(6)
        metric_cols[0].metric("Input Records", format_number(len(raw_df)))
        metric_cols[1].metric("Supplier Names Before", format_number(before_supplier_count))
        metric_cols[2].metric("Supplier Families After", format_number(after_supplier_count))
        metric_cols[3].metric("Potential Duplicate Groups", format_number(duplicate_groups))
        metric_cols[4].metric("Review Groups", format_number(review_groups))
        metric_cols[5].metric("Data Quality Score", f"{data_quality['overall_score']} / 100")

        st.markdown(
            f"""
            <div class="info-card">
                <strong>Executive interpretation:</strong> The file contains {format_number(before_supplier_count)} unique supplier
                names before normalization and {format_number(after_supplier_count)} normalized supplier families after standardization.
                The tool identified {format_number(duplicate_groups)} potential duplicate groups and {format_number(review_groups)}
                groups requiring human review. Spend affected by potential duplicate families is {format_currency(spend_affected)}.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### Normalization Impact Summary")

        impact_cols = st.columns(3)

        impact_cols[0].metric(
            "Supplier Names Reduced",
            format_number(before_supplier_count - after_supplier_count),
        )

        impact_cols[1].metric(
            "Reduction Rate",
            f"{((before_supplier_count - after_supplier_count) / before_supplier_count * 100):.1f}%"
            if before_supplier_count > 0
            else "0.0%",
        )

        impact_cols[2].metric(
            "Spend in Duplicate Groups",
            format_currency(spend_affected),
        )

        st.markdown("### Supplier Count Before vs. After")

        supplier_count_summary = pd.DataFrame(
            [
                {
                    "view": "Raw supplier names",
                    "supplier_count": before_supplier_count,
                    "interpretation": "Unique supplier names exactly as they appeared in the uploaded file.",
                },
                {
                    "view": "Normalized supplier families",
                    "supplier_count": after_supplier_count,
                    "interpretation": "Supplier families after alias matching, cleaning, and fuzzy grouping.",
                },
            ]
        )

        st.dataframe(
            clean_display_columns(supplier_count_summary),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("### Diagnostic Findings")

        diagnostic_summary = pd.DataFrame(
            [
                {
                    "finding": "Potential duplicate groups",
                    "value": duplicate_groups,
                    "meaning": "Supplier groups where multiple raw names may refer to the same supplier family.",
                },
                {
                    "finding": "Groups requiring review",
                    "value": review_groups,
                    "meaning": "Groups that should be manually reviewed before any vendor-master cleanup decision.",
                },
                {
                    "finding": "Spend affected by duplicate groups",
                    "value": format_currency(spend_affected),
                    "meaning": "Spend tied to supplier groups with more than one raw supplier-name variant.",
                },
            ]
        )

        st.dataframe(
            clean_display_columns(diagnostic_summary),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("### Data Quality Dimensions")

        quality_display = data_quality["dimensions"].copy()
        quality_display["score"] = quality_display["score"].round(1)

        st.dataframe(
            clean_display_columns(quality_display),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("### Executive Takeaway")

        if duplicate_groups > 0:
            st.markdown(
                f"""
                The supplier file has a meaningful normalization opportunity. The tool detected **{duplicate_groups}**
                potential duplicate supplier groups and **{review_groups}** groups requiring manual review.
                Cleaning these records can improve supplier visibility, spend concentration analysis, category strategy,
                and downstream sourcing opportunity identification.
                """
            )
        else:
            st.markdown(
                """
                The supplier file appears relatively clean from a duplicate-name perspective based on the current
                matching threshold. Consider lowering the threshold or using aggressive mode if the goal is broader
                duplicate discovery.
                """
            )

    # ========================================================
    # TAB 2: MATCH REVIEW
    # ========================================================

    with tab2:
        st.subheader("Supplier Match Review")

        st.markdown(
            """
            This section shows proposed supplier-family groupings, highlights records requiring human review,
            explains why records were grouped, and recommends golden supplier records. Groups marked as
            **Needs Review** should not be automatically merged without validation.
            """
        )

        review_queue = group_summary[
            (group_summary["review_status"] == "Needs Review")
            | (group_summary["false_positive_risk"] == "High")
        ].copy()

        review_cols = st.columns(4)
        review_cols[0].metric("Total Supplier Families", format_number(len(group_summary)))
        review_cols[1].metric("Duplicate Groups", format_number(duplicate_groups))
        review_cols[2].metric("Needs Review", format_number(len(review_queue)))
        review_cols[3].metric("Spend Affected", format_currency(spend_affected))

        st.markdown("### Potential Duplicate Match Groups")

        match_groups_only = group_summary[group_summary["variant_count"] > 1].copy()

        if match_groups_only.empty:
            st.success("No potential duplicate supplier groups were found at the current matching threshold.")
        else:
            display_match_groups = match_groups_only.copy()

            if "total_spend" in display_match_groups.columns:
                display_match_groups["total_spend"] = display_match_groups["total_spend"].apply(format_currency)

            preferred_columns = [
                "match_group_id",
                "normalized_supplier_name",
                "original_supplier_variants",
                "variant_count",
                "total_spend",
                "average_match_score",
                "confidence_tier",
                "recommended_action",
                "false_positive_risk",
                "review_status",
                "categories_detected",
                "countries_detected",
            ]

            display_match_groups = display_match_groups[
                [col for col in preferred_columns if col in display_match_groups.columns]
            ]

            st.caption("Select a row below to view the match explanation.")

            selected_event = st.dataframe(
                clean_display_columns(display_match_groups),
                use_container_width=True,
                hide_index=True,
                on_select="rerun",
                selection_mode="single-row",
                key="match_group_selection",
            )

            selected_rows = selected_event.selection.rows

            if selected_rows:
                selected_position = selected_rows[0]
            else:
                selected_position = 0

            selected_row = match_groups_only.iloc[selected_position]

            st.markdown("### Match Explanation")

            st.markdown(
                f"""
                <div class="info-card">
                    <strong>Normalized Supplier Family:</strong> {selected_row["normalized_supplier_name"]}<br><br>
                    <strong>Supplier Variants:</strong> {selected_row["original_supplier_variants"]}<br><br>
                    <strong>Average Match Score:</strong> {selected_row["average_match_score"]}<br><br>
                    <strong>Confidence Tier:</strong> {selected_row["confidence_tier"]}<br><br>
                    <strong>Recommended Action:</strong> {selected_row["recommended_action"]}<br><br>
                    <strong>False Positive Risk:</strong> {selected_row["false_positive_risk"]}<br><br>
                    <strong>Explanation:</strong> {selected_row["match_explanation"]}
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("### Human Review Queue")

        if review_queue.empty:
            st.success("No high-risk match groups require manual review.")
        else:
            display_review = review_queue.copy()

            if "total_spend" in display_review.columns:
                display_review["total_spend"] = display_review["total_spend"].apply(format_currency)

            st.dataframe(
                clean_display_columns(display_review),
                use_container_width=True,
                hide_index=True,
            )

        st.markdown("### Golden Record Recommendations")

        display_golden = golden_records.copy()

        if not display_golden.empty and "total_spend" in display_golden.columns:
            display_golden["total_spend"] = display_golden["total_spend"].apply(format_currency)

        st.dataframe(
            clean_display_columns(display_golden),
            use_container_width=True,
            hide_index=True,
        )

    # ========================================================
    # TAB 3: OUTPUT & METHODOLOGY
    # ========================================================

    with tab3:
        st.subheader("Output & Methodology")

        st.markdown(
            """
            This section provides the normalized supplier-level output, export files, and methodology notes.
            The normalized output can be used as a cleaner supplier master input for downstream spend analytics,
            sourcing opportunity analysis, or vendor-master cleanup review.
            """
        )

        st.markdown("### Download Outputs")

        normalized_export = normalized_data.to_csv(index=False).encode("utf-8")
        mapping_export = group_summary.to_csv(index=False).encode("utf-8")
        golden_export = golden_records.to_csv(index=False).encode("utf-8")

        col1, col2, col3 = st.columns(3)

        col1.download_button(
            label="Download Normalized Data",
            data=normalized_export,
            file_name="normalized_supplier_data.csv",
            mime="text/csv",
        )

        col2.download_button(
            label="Download Match Groups",
            data=mapping_export,
            file_name="supplier_match_groups.csv",
            mime="text/csv",
        )

        col3.download_button(
            label="Download Golden Records",
            data=golden_export,
            file_name="supplier_golden_records.csv",
            mime="text/csv",
        )

        st.markdown("### Normalized Supplier-Level Data")

        st.markdown(
            """
            This is the row-level output with original supplier names, cleaned supplier names,
            normalized supplier names, match confidence, reason codes, and supporting fields.
            """
        )

        display_normalized = normalized_data.copy()

        if "spend_value" in display_normalized.columns:
            display_normalized["spend_value"] = display_normalized["spend_value"].apply(format_currency)

        st.dataframe(
            clean_display_columns(display_normalized),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("### Methodology")

        st.markdown(
            """
            1. Supplier names are normalized using unicode cleanup, lowercasing, whitespace cleanup, punctuation removal, legal suffix removal, abbreviation expansion, and ERP metadata stripping.
            2. Known aliases are applied first for common supplier families such as AWS, IBM, Microsoft, DHL, FedEx, and others.
            3. Exact cleaned-name matches are grouped before fuzzy matching.
            4. Remaining supplier names are grouped using RapidFuzz fuzzy matching.
            5. Standalone suppliers are separated from duplicate groups and are not treated as confirmed duplicates.
            6. Match confidence is translated into business-readable tiers: Confirmed Duplicate, Probable Duplicate, Possible Duplicate, Not a Match, and Standalone Supplier.
            7. Groups with lower confidence, category conflicts, country conflicts, or high false-positive risk are routed to the human review queue.
            8. Each duplicate match group includes a deterministic explanation showing why the supplier variants were grouped.
            9. Golden records are recommended using normalized supplier family names and simple survivorship logic.
            """
        )

        st.markdown("### Limitations")

        st.markdown(
            """
            - This tool does not verify legal entities against external databases.
            - Fuzzy matching is directional and requires human review before ERP/vendor-master updates.
            - Parent-company and corporate hierarchy mapping are not included in this MVP.
            - Tax ID, address, domain, LEI, and D-U-N-S matching can be added in a future version.
            - Supplier consolidation decisions should be validated with contracts, business owners, tax/legal data, and category strategy.
            """
        )


if __name__ == "__main__":
    main()
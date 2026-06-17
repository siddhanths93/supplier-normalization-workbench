import re
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
        }

        .warning-card {
            background: #fffbeb;
            border: 1px solid #fde68a;
            padding: 1rem 1.1rem;
            border-radius: 0.9rem;
            color: #92400e;
            margin-bottom: 1rem;
        }

        div[data-testid="stMetricValue"] {
            font-size: 1.55rem;
            font-weight: 800;
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


def format_percent(value):
    try:
        return f"{float(value):.1f}%"
    except Exception:
        return "Not available"


def clean_display_columns(df):
    display = df.copy()
    display.columns = [str(col).replace("_", " ").title() for col in display.columns]
    return display


# ============================================================
# SUPPLIER CLEANING / MATCHING CONFIG
# ============================================================

COMMON_SUFFIXES = [
    "inc", "incorporated", "llc", "ltd", "limited", "corp", "corporation",
    "co", "company", "plc", "lp", "llp", "gmbh", "ag", "sa", "sarl",
    "services", "service", "group", "holdings", "holding", "the",
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
    "microsoft corp": "Microsoft",
    "microsoft corporation": "Microsoft",
    "msft": "Microsoft",
    "microsoft azure": "Microsoft",

    "google": "Google / Alphabet",
    "google cloud": "Google / Alphabet",
    "alphabet": "Google / Alphabet",
    "youtube": "Google / Alphabet",

    "dhl": "DHL",
    "dhl express": "DHL",
    "dhl global forwarding": "DHL",

    "fedex": "FedEx",
    "fedex corp": "FedEx",
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
    "staples inc": "Staples",
    "office depot": "Office Depot",
}


def clean_supplier_name(name):
    """
    Standardizes supplier names before matching.
    This does not create the final name; it creates the comparison string.
    """
    if pd.isna(name):
        return ""

    cleaned = str(name).lower().strip()
    cleaned = re.sub(r"&", " and ", cleaned)
    cleaned = re.sub(r"[^a-z0-9\s]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    words = [word for word in cleaned.split() if word not in COMMON_SUFFIXES]
    return " ".join(words).strip()


def alias_lookup(name):
    cleaned = clean_supplier_name(name)
    return KNOWN_ALIASES.get(cleaned)


def choose_canonical_name(original_names, spend_lookup):
    """
    Chooses the canonical display name for a fuzzy group.
    For non-alias fuzzy groups, use the highest-spend variant.
    """
    valid_names = [str(name).strip() for name in original_names if str(name).strip()]

    if not valid_names:
        return "Unknown Supplier"

    return sorted(
        valid_names,
        key=lambda supplier: spend_lookup.get(supplier, 0),
        reverse=True,
    )[0]


def confidence_band(score, method):
    if method == "Known alias":
        return "High"
    if score >= 95:
        return "High"
    if score >= 90:
        return "Medium-High"
    if score >= 85:
        return "Medium / Review"
    if score >= 75:
        return "Low / Review"
    return "Low"


def false_positive_risk(score, variant_count, category_conflict=False, country_conflict=False):
    risk = "Low"

    if score < 90:
        risk = "Medium"

    if score < 85:
        risk = "High"

    if category_conflict or country_conflict:
        risk = "High"

    if variant_count == 1:
        risk = "Low"

    return risk


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

        ["ABC Logistics LLC", 180000, "Logistics", "United States", "V013"],
        ["ABC Logistic Services", 95000, "Logistics", "United States", "V014"],
        ["ABC Consulting LLC", 220000, "Professional Services", "United States", "V015"],

        ["Delta Air Lines", 500000, "Travel", "United States", "V016"],
        ["Delta Dental", 150000, "Benefits", "United States", "V017"],

        ["Local HVAC Repair Co", 85000, "Facilities", "United States", "V018"],
        ["Local H.V.A.C. Repair", 92000, "Facilities", "United States", "V019"],

        ["Staples Inc", 260000, "Office Supplies", "United States", "V020"],
        ["Staples", 190000, "Office Supplies", "United States", "V021"],
        ["Office Depot", 240000, "Office Supplies", "United States", "V022"],
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

    # Step 1: alias matching
    for supplier in original_suppliers:
        alias = alias_lookup(supplier)

        if alias:
            mapping[supplier] = alias
            scores[supplier] = 100
            methods[supplier] = "Known alias"
            reason_codes[supplier] = "KNOWN_ALIAS"
        else:
            unresolved.append(supplier)

    # Step 2: fuzzy matching among unresolved suppliers
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

    cleaned_names = list(cleaned_to_originals.keys())
    assigned_cleaned_names = set()

    for cleaned_name in cleaned_names:
        if cleaned_name in assigned_cleaned_names:
            continue

        matches = process.extract(
            cleaned_name,
            cleaned_names,
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
        avg_score = round(sum(matched_scores) / len(matched_scores), 1) if matched_scores else 0

        for supplier in original_group:
            mapping[supplier] = canonical
            scores[supplier] = avg_score
            if len(original_group) > 1:
                methods[supplier] = "Fuzzy match"
                reason_codes[supplier] = "FUZZY_NAME_MATCH"
            else:
                methods[supplier] = "No close match"
                reason_codes[supplier] = "NO_CLOSE_MATCH"

    data["normalized_supplier_name"] = data["original_supplier_name"].map(mapping)
    data["match_score"] = data["original_supplier_name"].map(scores)
    data["match_method"] = data["original_supplier_name"].map(methods)
    data["reason_code"] = data["original_supplier_name"].map(reason_codes)
    data["confidence_band"] = data.apply(
        lambda row: confidence_band(row["match_score"], row["match_method"]),
        axis=1,
    )

    # Build group-level summary
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
                "survivorship_logic": "Name from normalized family; category/country selected by most frequent value; spend aggregated across variants.",
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

def calculate_data_quality(df, normalized_data, group_summary, supplier_col, spend_col=None, category_col=None, country_col=None):
    total_records = len(df)

    supplier_completeness = df[supplier_col].notna().mean() * 100 if supplier_col in df.columns else 0

    if spend_col and spend_col in df.columns:
        spend_numeric = pd.to_numeric(
            df[spend_col].astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False),
            errors="coerce",
        )
        spend_usability = spend_numeric.notna().mean() * 100
    else:
        spend_usability = 0

    category_completeness = df[category_col].notna().mean() * 100 if category_col and category_col in df.columns else 0
    country_completeness = df[country_col].notna().mean() * 100 if country_col and country_col in df.columns else 0

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
        "total_records": total_records,
    }


# ============================================================
# RENDER APP
# ============================================================

def main():
    apply_styles()

    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-label">Procurement Data Quality Accelerator</div>
            <div class="hero-title">Supplier Normalization & Duplicate Detection Workbench</div>
            <div class="hero-subtitle">
                Upload messy supplier data, detect duplicate vendor records, normalize supplier families,
                generate a review queue, build golden record recommendations, and export a cleaner supplier
                master for spend analytics.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

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

    st.sidebar.header("Column Mapping")

    columns = list(raw_df.columns)

    default_supplier = columns.index("supplier_name") if "supplier_name" in columns else 0
    supplier_col = st.sidebar.selectbox("Supplier name column", columns, index=default_supplier)

    optional_columns = ["None"] + columns

    spend_default = optional_columns.index("spend") if "spend" in optional_columns else 0
    spend_col = st.sidebar.selectbox("Spend column optional", optional_columns, index=spend_default)
    spend_col = None if spend_col == "None" else spend_col

    category_default = optional_columns.index("category") if "category" in optional_columns else 0
    category_col = st.sidebar.selectbox("Category column optional", optional_columns, index=category_default)
    category_col = None if category_col == "None" else category_col

    country_default = optional_columns.index("country") if "country" in optional_columns else 0
    country_col = st.sidebar.selectbox("Country column optional", optional_columns, index=country_default)
    country_col = None if country_col == "None" else country_col

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
        normalized_data,
        group_summary,
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

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [
            "Executive Summary",
            "Match Groups",
            "Review Queue",
            "Golden Records",
            "Normalized Data",
            "Export & Methodology",
        ]
    )

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

        st.markdown("### Data Quality Dimensions")
        quality_display = data_quality["dimensions"].copy()
        quality_display["score"] = quality_display["score"].round(1)
        st.dataframe(clean_display_columns(quality_display), use_container_width=True, hide_index=True)

        st.markdown("### Before / After Impact")
        impact = pd.DataFrame(
            [
                {"metric": "Unique supplier names", "before": before_supplier_count, "after": after_supplier_count},
                {"metric": "Potential duplicate groups", "before": "Not available", "after": duplicate_groups},
                {"metric": "Groups requiring review", "before": "Not available", "after": review_groups},
                {"metric": "Spend affected by duplicate groups", "before": "Not available", "after": format_currency(spend_affected)},
            ]
        )
        st.dataframe(clean_display_columns(impact), use_container_width=True, hide_index=True)

    with tab2:
        st.subheader("Supplier Match Groups")

        st.markdown(
            """
            This tab shows how messy supplier names were grouped into normalized supplier families.
            Groups marked as **Needs Review** should not be automatically merged without human validation.
            """
        )

        display = group_summary.copy()
        if not display.empty and "total_spend" in display.columns:
            display["total_spend"] = display["total_spend"].apply(format_currency)

        st.dataframe(clean_display_columns(display), use_container_width=True, hide_index=True)

    with tab3:
        st.subheader("Human Review Queue")

        review_queue = group_summary[
            (group_summary["review_status"] == "Needs Review")
            | (group_summary["false_positive_risk"] == "High")
        ].copy()

        if review_queue.empty:
            st.success("No high-risk match groups require manual review.")
        else:
            display = review_queue.copy()
            display["total_spend"] = display["total_spend"].apply(format_currency)
            st.dataframe(clean_display_columns(display), use_container_width=True, hide_index=True)

        st.markdown("### Why Human Review Matters")
        st.markdown(
            """
            Supplier standardization should not blindly merge every similar name. Similar supplier names may represent
            different legal entities, business units, categories, countries, or buying relationships. This review queue
            helps reduce false positives before downstream spend analytics or sourcing decisions.
            """
        )

    with tab4:
        st.subheader("Golden Record Recommendations")

        st.markdown(
            """
            Golden records represent the recommended supplier-family view after normalization.
            In a full supplier master data program, these records would be reviewed and approved before updating ERP or vendor-master data.
            """
        )

        display = golden_records.copy()
        if not display.empty and "total_spend" in display.columns:
            display["total_spend"] = display["total_spend"].apply(format_currency)

        st.dataframe(clean_display_columns(display), use_container_width=True, hide_index=True)

    with tab5:
        st.subheader("Normalized Supplier-Level Data")

        display = normalized_data.copy()
        if "spend_value" in display.columns:
            display["spend_value"] = display["spend_value"].apply(format_currency)

        st.dataframe(clean_display_columns(display), use_container_width=True, hide_index=True)

    with tab6:
        st.subheader("Export & Methodology")

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

        st.markdown("### Methodology")
        st.markdown(
            """
            1. Supplier names are cleaned by lowercasing, removing punctuation, and removing common legal suffixes.
            2. Known aliases are applied first for common supplier families such as AWS, IBM, Microsoft, DHL, FedEx, and others.
            3. Remaining supplier names are grouped using RapidFuzz fuzzy matching.
            4. Match confidence is calculated using the fuzzy score and match method.
            5. Groups with lower confidence, category conflicts, or country conflicts are sent to the human review queue.
            6. Golden records are recommended using normalized supplier family names and simple survivorship logic.
            """
        )

        st.markdown("### Limitations")
        st.markdown(
            """
            - This tool does not verify legal entities against external databases.
            - Fuzzy matching is directional and requires human review before ERP/vendor-master updates.
            - Parent-company and corporate hierarchy mapping are not included in this MVP.
            - Tax ID, address, domain, and D-U-N-S matching can be added in a future version.
            - Supplier consolidation decisions should be validated with contracts, business owners, tax/legal data, and category strategy.
            """
        )


if __name__ == "__main__":
    main()
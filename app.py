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
    "bv", "nv", "srl", "ltda", "pty", "pte", "sdn", "bhd", "ab",
]

BUSINESS_NOISE_WORDS = [
    "services", "service", "group", "holdings", "holding", "company",
    "corporation", "corp", "inc", "llc", "ltd", "limited", "plc", "lp",
    "llp", "ag", "gmbh", "sa", "america", "american",
]

ERP_NOISE_WORDS = [
    "ap", "a p", "accounts payable", "payable", "ach", "wire",
    "pcard", "p-card", "p card", "credit card", "credit", "card",
    "po", "purchasing", "procurement", "indirect", "direct",
    "marketplace", "online", "store", "vendor", "old vendor",
    "legacy", "blocked", "inactive", "duplicate", "do not use",
    "remit", "remit to", "ap hold", "hold", "old", "new", "regional",
    "local", "global", "national", "international", "intl",
    "north america", "america", "americas", "emea", "apac", "latam",
    "na", "us", "u s", "usa", "u s a", "canada", "mexico",
    "new york", "ny", "nyc", "atlanta", "dallas", "chicago",
    "tor", "toronto", "mia", "miami", "houston", "boston",
    "san francisco", "sf", "los angeles", "la",
]

GENERIC_MATCH_TOKENS = {
    "consulting", "services", "service", "group", "company", "companies",
    "industries", "industry", "corporation", "corp", "inc", "llc",
    "technologies", "technology", "systems", "solutions", "solution",
    "international", "global", "national", "america", "american",
    "business", "advantage", "partners", "partner", "management",
    "enterprise", "enterprises", "supply", "supplies", "industrial",
    "products", "product", "energy", "automation", "express",
}

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
    r"\b/remit to\b",
    r"\bremit to\b",
]

CANONICAL_SUPPLIER_LIBRARY = [
    "3M Company",
    "ABC Supply",
    "ABM Industries",
    "ADP",
    "Accenture",
    "Adobe Inc",
    "Agilent Technologies",
    "Airgas USA",
    "Amazon Web Services",
    "American Express",
    "Apple Inc",
    "AppleOne Employment Services",
    "BASF Corporation",
    "BP Energy",
    "Bain & Company",
    "Bank of America",
    "Boston Consulting Group",
    "C.H. Robinson",
    "CBRE Group",
    "CDW Corporation",
    "Cardinal Health",
    "Chevron Products",
    "Cisco Systems",
    "Constellation Energy",
    "CrowdStrike",
    "DHL Express",
    "Dell Technologies",
    "Deloitte Consulting",
    "Delta Air Lines",
    "Delta Dental",
    "Dow Chemical",
    "DuPont",
    "Duke Energy",
    "Ernst & Young",
    "ExxonMobil Fuels",
    "Expeditors International",
    "Fastenal",
    "FedEx Corporation",
    "Fortinet",
    "Grainger",
    "Graybar Electric",
    "HP Inc",
    "Honeywell International",
    "IBM Corporation",
    "Insight Enterprises",
    "Insight Global",
    "International Paper",
    "JLL",
    "Johnson Controls",
    "Kelly Services",
    "Kone Inc",
    "LinkedIn Corporation",
    "Linde Gas & Equipment",
    "Matheson Tri-Gas",
    "McKinsey & Company",
    "Medline Industries",
    "Microsoft Corporation",
    "Motion Industries",
    "MSC Industrial Supply",
    "Okta",
    "Oracle Corporation",
    "Otis Elevator",
    "Palo Alto Networks",
    "Peachtree Consulting",
    "PricewaterhouseCoopers",
    "Randstad",
    "Republic Services",
    "Robert Half",
    "Rockwell Automation",
    "SAP America",
    "Salesforce Inc",
    "SHI International",
    "Siemens Industry",
    "Slack Technologies",
    "Staples Business Advantage",
    "United Airlines",
    "United Parcel Service",
    "VWR International",
    "Waste Management",
    "Workday",
    "XPO Logistics",
    "Zoom Video Communications",
]

KNOWN_ALIASES = {
    "ibm": "IBM Corporation",
    "i b m": "IBM Corporation",
    "international business machines": "IBM Corporation",

    "aws": "Amazon Web Services",
    "amazon web services": "Amazon Web Services",
    "amazon": "Amazon Web Services",

    "microsoft": "Microsoft Corporation",
    "microsoft corporation": "Microsoft Corporation",
    "microsoft company": "Microsoft Corporation",
    "mcrsft corporation": "Microsoft Corporation",
    "micro soft corporation": "Microsoft Corporation",
    "msft": "Microsoft Corporation",

    "fedex": "FedEx Corporation",
    "fedex corporation": "FedEx Corporation",
    "fedex corp": "FedEx Corporation",
    "fdx corporation": "FedEx Corporation",
    "fc": "FedEx Corporation",
    "federal express": "FedEx Corporation",

    "ups": "United Parcel Service",
    "united parcel service": "United Parcel Service",

    "dhl": "DHL Express",
    "dhl express": "DHL Express",
    "dhl experss": "DHL Express",
    "dhl expre ss": "DHL Express",

    "dell technologies": "Dell Technologies",
    "dell tech": "Dell Technologies",
    "dll technologies": "Dell Technologies",
    "deell technologies": "Dell Technologies",
    "dt": "Dell Technologies",

    "accenture": "Accenture",
    "accennture": "Accenture",

    "duke energy": "Duke Energy",
    "dske energy": "Duke Energy",
    "duke nrgy": "Duke Energy",

    "international paper": "International Paper",
    "intl paper": "International Paper",
    "ip": "International Paper",
    "internatiobal paper": "International Paper",
    "interantional paper": "International Paper",

    "rockwell automation": "Rockwell Automation",
    "rockwell tmtn": "Rockwell Automation",

    "kone": "Kone Inc",
    "kone inc": "Kone Inc",
    "kn inc": "Kone Inc",
    "ki": "Kone Inc",

    "zoom video communications": "Zoom Video Communications",
    "zoom video comm": "Zoom Video Communications",
    "zoom video cmmnctns": "Zoom Video Communications",
    "zvc": "Zoom Video Communications",

    "adobe": "Adobe Inc",
    "adobe inc": "Adobe Inc",

    "salesforce": "Salesforce Inc",
    "salesforce inc": "Salesforce Inc",

    "sap": "SAP America",
    "sap america": "SAP America",
    "sa": "SAP America",

    "staples": "Staples Business Advantage",
    "staples business advantage": "Staples Business Advantage",
    "sba": "Staples Business Advantage",

    "workday": "Workday",
    "woruday": "Workday",
    "wrkday": "Workday",

    "slack": "Slack Technologies",
    "slack technologies": "Slack Technologies",

    "linkedin": "LinkedIn Corporation",
    "linkedin corporation": "LinkedIn Corporation",
    "lc": "LinkedIn Corporation",

    "linde gas": "Linde Gas & Equipment",
    "linde gas equipment": "Linde Gas & Equipment",
    "lge": "Linde Gas & Equipment",

    "mckinsey": "McKinsey & Company",
    "mckinsey company": "McKinsey & Company",
    "mc": "McKinsey & Company",

    "siemens": "Siemens Industry",
    "siemens industry": "Siemens Industry",
    "si": "Siemens Industry",

    "medline": "Medline Industries",
    "medline industries": "Medline Industries",
    "mi": "Medline Industries",

    "kelly services": "Kelly Services",
    "ks": "Kelly Services",

    "palo alto networks": "Palo Alto Networks",
    "pan": "Palo Alto Networks",

    "united airlines": "United Airlines",
    "ua": "United Airlines",

    "hp": "HP Inc",
    "hp inc": "HP Inc",

    "adp": "ADP",
    "jll": "JLL",

    "msc industrial": "MSC Industrial Supply",
    "msc industrial supply": "MSC Industrial Supply",
    "mis": "MSC Industrial Supply",

    "bcg": "Boston Consulting Group",
    "boston consulting": "Boston Consulting Group",
    "boston consulting group": "Boston Consulting Group",
    "boston connsulting group": "Boston Consulting Group",

    "bain": "Bain & Company",
    "bain company": "Bain & Company",

    "ey": "Ernst & Young",
    "ernst young": "Ernst & Young",

    "3m": "3M Company",
    "3 m": "3M Company",
    "3m company": "3M Company",

    "grainger": "Grainger",
    "ww grainger": "Grainger",
    "w w grainger": "Grainger",
    "graingre": "Grainger",

    "republic services": "Republic Services",
    "rs": "Republic Services",

    "shi": "SHI International",
    "shi international": "SHI International",

    "oracle": "Oracle Corporation",
    "oracle corporation": "Oracle Corporation",

    "waste management": "Waste Management",
    "wastte management": "Waste Management",
    "wafte management": "Waste Management",

    "chevron products": "Chevron Products",
    "chvrn products": "Chevron Products",

    "pricewaterhousecoopers": "PricewaterhouseCoopers",
    "pricewaterhosuecoopers": "PricewaterhouseCoopers",
    "pwc": "PricewaterhouseCoopers",

    "cisco systems": "Cisco Systems",
    "cicso systems": "Cisco Systems",

    "randstad": "Randstad",
    "r stad": "Randstad",

    "abc supply": "ABC Supply",
    "abc spply": "ABC Supply",
    "abc supplly": "ABC Supply",

    "agilent technologies": "Agilent Technologies",

    "american express": "American Express",
    "americanexpress": "American Express",

    "dow chemical": "Dow Chemical",
    "dow chomical": "Dow Chemical",

    "dupont": "DuPont",
    "dupoont": "DuPont",

    "exxonmobil fuels": "ExxonMobil Fuels",
    "exxonmobil fls": "ExxonMobil Fuels",

    "fastenal": "Fastenal",
    "fastennal": "Fastenal",
    "fasthnal": "Fastenal",

    "graybar electric": "Graybar Electric",
    "graybar lctrc": "Graybar Electric",
    "grybr electric": "Graybar Electric",

    "peachtree consulting": "Peachtree Consulting",
    "peachtree connsulting": "Peachtree Consulting",

    "robert half": "Robert Half",
    "robert hlaf": "Robert Half",

    "expeditors international": "Expeditors International",
    "xpdtrs international": "Expeditors International",

    "xpo logistics": "XPO Logistics",
    "xpo lgstcs": "XPO Logistics",

    "insight global": "Insight Global",
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
    cleaned = cleaned.strip(" -_#.*!@$/\\")

    tokens = cleaned.split()
    tokens = expand_abbreviations(tokens)
    tokens = [token for token in tokens if token not in COMMON_SUFFIXES]

    return " ".join(tokens).strip(" -_#.*!@$/\\")


def strip_erp_noise_tokens(cleaned_name):
    text = f" {cleaned_name} "

    text = re.sub(r"\b\d{4}\b", " ", text)
    text = re.sub(r"\b\d+\b", " ", text)
    text = re.sub(r"\b[a-z]{2,5}\d{2,6}\b", " ", text)

    multi_word_noise = sorted(
        [word for word in ERP_NOISE_WORDS if " " in word],
        key=len,
        reverse=True,
    )

    for phrase in multi_word_noise:
        text = re.sub(rf"\b{re.escape(phrase)}\b", " ", text)

    tokens = text.split()
    single_word_noise = set(word for word in ERP_NOISE_WORDS if " " not in word)
    tokens = [token for token in tokens if token not in single_word_noise]

    return " ".join(tokens).strip(" -_#.*!@$/\\")


def extract_core_supplier_name(name):
    cleaned = clean_supplier_name(name)

    core = strip_erp_noise_tokens(cleaned)

    tokens = core.split()
    tokens = [token for token in tokens if token not in BUSINESS_NOISE_WORDS]

    core = " ".join(tokens).strip(" -_#.*!@$/\\")

    if not core:
        core = cleaned

    return core


def has_meaningful_token_overlap(core_a, core_b):
    tokens_a = set(core_a.split())
    tokens_b = set(core_b.split())

    shared = tokens_a.intersection(tokens_b)

    if not shared:
        return False

    meaningful_shared = shared.difference(GENERIC_MATCH_TOKENS)

    if len(meaningful_shared) == 0:
        return False

    return True


def alias_lookup(name):
    core = extract_core_supplier_name(name)
    return KNOWN_ALIASES.get(core)


def canonical_library_lookup(name, threshold=90):
    core = extract_core_supplier_name(name)

    if not core:
        return None, 0

    canonical_clean_map = {
        extract_core_supplier_name(canonical): canonical
        for canonical in CANONICAL_SUPPLIER_LIBRARY
    }

    if core in canonical_clean_map:
        return canonical_clean_map[core], 100

    match = process.extractOne(
        core,
        list(canonical_clean_map.keys()),
        scorer=fuzz.token_set_ratio,
        score_cutoff=threshold,
    )

    if match:
        matched_core, score, _ = match

        if not has_meaningful_token_overlap(core, matched_core):
            return None, 0

        return canonical_clean_map[matched_core], score

    return None, 0


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
    if variant_count == 1:
        return "Standalone Supplier"

    if method in ["Known alias", "Canonical library match"]:
        return "Confirmed Duplicate"

    if score >= 95:
        return "Confirmed Duplicate"

    if score >= 88:
        return "Probable Duplicate"

    if score >= 75:
        return "Possible Duplicate"

    return "Not a Match"


def false_positive_risk(score, variant_count, category_conflict=False, country_conflict=False):
    if variant_count == 1:
        return "Low"

    risk = "Low"

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

    if method in ["Known alias", "Canonical library match"]:
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

    if "CANONICAL_LIBRARY_MATCH" in str(row.get("reason_codes", "")):
        reasons.append("the cleaned supplier name matched the canonical supplier library")

    if "FUZZY_CORE_NAME_MATCH" in str(row.get("reason_codes", "")):
        reasons.append(
            f"supplier names were similar after core-name extraction with an average match score of {row.get('average_match_score')}"
        )

    if "EXACT_CORE_NAME" in str(row.get("reason_codes", "")):
        reasons.append("supplier names matched exactly after core-name extraction")

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
        ["#Wastte Management", 500000, "Facilities", "United States", "V001"],
        [".Cicso Systems (do not use)-", 450000, "IT Hardware", "United States", "V002"],
        ["AMERICAN EXPRESS", 300000, "Travel", "United States", "V003"],
        ["Agilent Technologies, L.L.C.", 250000, "Lab Supplies", "United States", "V004"],
        ["Peachtree Connsulting", 200000, "Consulting", "United States", "V005"],
        ["Wrkday", 750000, "Software", "United States", "V006"],
        ["xpdtrs International", 600000, "Logistics", "United States", "V007"],
        ["#INTL Insight Global-", 550000, "Professional Services", "United States", "V008"],
        ["DHL Express PCard", 760000, "Logistics", "United States", "V009"],
        ["D.H.L. Express - DO NOT USE", 310000, "Logistics", "United States", "V010"],
        ["SAP America AP HOLD", 330000, "Software", "United States", "V011"],
        ["SA", 110000, "Software", "United States", "V012"],
        ["Boston Consulting Group", 500000, "Consulting", "United States", "V013"],
        ["Deloitte Consulting", 800000, "Consulting", "United States", "V014"],
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
    threshold=82,
    canonical_threshold=88,
):
    data = df.copy()

    data["original_supplier_name"] = data[supplier_col].fillna("Unknown Supplier").astype(str)
    data["clean_supplier_name"] = data["original_supplier_name"].apply(clean_supplier_name)
    data["core_supplier_name"] = data["original_supplier_name"].apply(extract_core_supplier_name)

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

    for supplier in original_suppliers:
        alias = alias_lookup(supplier)

        if alias:
            mapping[supplier] = alias
            scores[supplier] = 100
            methods[supplier] = "Known alias"
            reason_codes[supplier] = "KNOWN_ALIAS"
            continue

        canonical_match, canonical_score = canonical_library_lookup(
            supplier,
            threshold=canonical_threshold,
        )

        if canonical_match:
            mapping[supplier] = canonical_match
            scores[supplier] = canonical_score
            methods[supplier] = "Canonical library match"
            reason_codes[supplier] = "CANONICAL_LIBRARY_MATCH"
            continue

        unresolved.append(supplier)

    core_to_originals = defaultdict(list)

    for supplier in unresolved:
        core = extract_core_supplier_name(supplier)

        if not core:
            mapping[supplier] = "Unknown Supplier"
            scores[supplier] = 0
            methods[supplier] = "Missing supplier name"
            reason_codes[supplier] = "MISSING_NAME"
        else:
            core_to_originals[core].append(supplier)

    fuzzy_candidates = []

    for core_name, originals in core_to_originals.items():
        if len(originals) > 1:
            canonical = choose_canonical_name(originals, spend_lookup)

            for supplier in originals:
                mapping[supplier] = canonical
                scores[supplier] = 100
                methods[supplier] = "Exact core-name match"
                reason_codes[supplier] = "EXACT_CORE_NAME"
        else:
            fuzzy_candidates.append(core_name)

    assigned_core_names = set()

    for core_name in fuzzy_candidates:
        if core_name in assigned_core_names:
            continue

        matches = process.extract(
            core_name,
            fuzzy_candidates,
            scorer=fuzz.token_set_ratio,
            score_cutoff=threshold,
            limit=None,
        )

        matched_core_names = [match[0] for match in matches]
        matched_scores = [match[1] for match in matches]

        for matched_name in matched_core_names:
            assigned_core_names.add(matched_name)

        original_group = []

        for matched_name in matched_core_names:
            original_group.extend(core_to_originals[matched_name])

        canonical = choose_canonical_name(original_group, spend_lookup)

        if len(original_group) > 1:
            avg_score = round(sum(matched_scores) / len(matched_scores), 1)
        else:
            avg_score = 0

        for supplier in original_group:
            mapping[supplier] = canonical
            scores[supplier] = avg_score

            if len(original_group) > 1:
                methods[supplier] = "Fuzzy core-name match"
                reason_codes[supplier] = "FUZZY_CORE_NAME_MATCH"
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

    group_rows = []

    for normalized_name, group_df in data.groupby("normalized_supplier_name", dropna=False):
        variants = sorted(group_df["original_supplier_name"].dropna().astype(str).unique())
        core_names = sorted(group_df["core_supplier_name"].dropna().astype(str).unique())

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
        elif "Canonical library match" in group_df["match_method"].values:
            group_method = "Canonical library match"
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
                "core_supplier_names_detected": ", ".join(core_names),
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
            <div class="hero-label">Procurement Data Quality Accelerator</div>
            <div class="hero-title">Supplier Normalization & Duplicate Detection Workbench</div>
            <div class="hero-subtitle">
                Upload messy supplier data, extract core supplier identities, detect duplicate vendor records,
                normalize supplier families, generate a human review queue, build golden record recommendations,
                and export cleaner supplier data for downstream spend analytics.
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

    if raw_df is None or raw_df.empty:
        st.warning("No data available.")
        return

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

    st.sidebar.header("Matching Settings")

    matching_mode = st.sidebar.radio(
        "Matching mode",
        ["Conservative", "Balanced", "Aggressive"],
        index=2,
    )

    fuzzy_thresholds = {
        "Conservative": 90,
        "Balanced": 84,
        "Aggressive": 78,
    }

    canonical_thresholds = {
        "Conservative": 92,
        "Balanced": 88,
        "Aggressive": 84,
    }

    threshold = st.sidebar.slider(
        "Fuzzy matching threshold",
        min_value=70,
        max_value=98,
        value=fuzzy_thresholds[matching_mode],
        step=1,
    )

    canonical_threshold = st.sidebar.slider(
        "Canonical library threshold",
        min_value=75,
        max_value=98,
        value=canonical_thresholds[matching_mode],
        step=1,
    )

    st.sidebar.caption(
        "Lower thresholds group more supplier variants but may increase false positives. "
        "For messy ERP test data, start with Aggressive mode."
    )

    normalized_data, group_summary = normalize_suppliers(
        raw_df,
        supplier_col=supplier_col,
        spend_col=spend_col,
        category_col=category_col,
        country_col=country_col,
        threshold=threshold,
        canonical_threshold=canonical_threshold,
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

    tab1, tab2, tab3 = st.tabs(
        [
            "Executive Summary",
            "Match Review",
            "Output & Methodology",
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
                    "interpretation": "Supplier families after alias matching, core-name extraction, canonical library matching, and fuzzy grouping.",
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

    with tab2:
        st.subheader("Supplier Match Review")

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
                "core_supplier_names_detected",
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

            try:
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

            except TypeError:
                st.dataframe(
                    clean_display_columns(display_match_groups),
                    use_container_width=True,
                    hide_index=True,
                )
                selected_position = 0
                st.caption(
                    "Interactive row selection requires Streamlit 1.35+. Showing the first match group explanation."
                )

            selected_row = match_groups_only.iloc[selected_position]

            st.markdown("### Match Explanation")

            st.markdown(
                f"""
                <div class="info-card">
                    <strong>Normalized Supplier Family:</strong> {selected_row["normalized_supplier_name"]}<br><br>
                    <strong>Supplier Variants:</strong> {selected_row["original_supplier_variants"]}<br><br>
                    <strong>Core Supplier Names Detected:</strong> {selected_row["core_supplier_names_detected"]}<br><br>
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

    with tab3:
        st.subheader("Output & Methodology")

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
            2. The tool extracts a core supplier identity by removing transaction IDs, store numbers, payment terms, location noise, region labels, and ERP-process words.
            3. Known aliases are applied first for common supplier families and high-confidence abbreviations.
            4. Supplier names are then matched against a canonical supplier library.
            5. A generic-token guardrail prevents weak matches where the only shared words are generic business terms such as consulting, services, group, company, industries, technologies, international, or express.
            6. Exact core-name matches are grouped before fuzzy matching.
            7. Remaining supplier names are grouped using RapidFuzz fuzzy matching on extracted core supplier names.
            8. Standalone suppliers are separated from duplicate groups and are not treated as confirmed duplicates.
            9. Each duplicate match group includes a deterministic explanation showing why the supplier variants were grouped.
            """
        )

        st.markdown("### Limitations")

        st.markdown(
            """
            - This tool does not verify legal entities against external databases.
            - The built-in canonical supplier library is intentionally limited and should be expanded for production use.
            - Acronym aliases can be powerful but risky; very short aliases should be curated carefully.
            - Fuzzy matching is directional and requires human review before ERP/vendor-master updates.
            - Parent-company and corporate hierarchy mapping are not included in this MVP.
            - Tax ID, address, domain, LEI, and D-U-N-S matching can be added in a future version.
            """
        )


if __name__ == "__main__":
    main()
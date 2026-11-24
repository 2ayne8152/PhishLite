"""
app.py - PhishLite lightweight GUI prototype (Streamlit).

Run after `python main.py part_b` (which creates artifacts/models.joblib):
    streamlit run app.py
"""
from __future__ import annotations

import ipaddress
import re
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import joblib
import pandas as pd
import streamlit as st

# Keep these imports so joblib can unpickle custom wrappers from b_ensemble / b_tuning
from b_ensemble import EnsembleModel  # noqa: F401
from b_tuning import TabNetSklearn  # noqa: F401

BUNDLE_PATH = Path("artifacts/models.joblib")

# 17 special characters counted across url, domain, directory, file, and params (Tables 1-5)
CHAR_MAP = {
    "dot": ".",
    "hyphen": "-",
    "underline": "_",
    "slash": "/",
    "questionmark": "?",
    "equal": "=",
    "at": "@",
    "and": "&",
    "exclamation": "!",
    "space": " ",
    "tilde": "~",
    "comma": ",",
    "plus": "+",
    "asterisk": "*",
    "hashtag": "#",
    "dollar": "$",
    "percent": "%",
}

COMMON_TLDS = {
    "com", "org", "net", "edu", "gov", "mil", "int", "info", "biz", "co", "io",
    "ai", "app", "dev", "xyz", "online", "site", "top", "club", "shop", "my",
    "uk", "us", "ca", "au", "de", "fr", "jp", "cn", "in", "sg", "ru", "br",
}

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


# --------------------------------------------------------------------------- #
# Local lexical URL feature extractor (Tables 1-5, zero network lookups)
# --------------------------------------------------------------------------- #
def _is_ip(host: str) -> int:
    try:
        ipaddress.ip_address(host.strip("[]"))
        return 1
    except ValueError:
        return 0


def _count_chars(text: str | None, suffix: str) -> dict[str, int]:
    if text is None:
        return {f"qty_{name}_{suffix}": -1 for name in CHAR_MAP}
    return {f"qty_{name}_{suffix}": text.count(ch) for name, ch in CHAR_MAP.items()}


def extract_url_features(raw_url: str) -> dict[str, int]:
    """Parse a raw URL into the 97 lexical features from the Vrbancic et al. dataset."""
    url = raw_url.strip()
    if "://" not in url:
        parse_target = "http://" + url
    else:
        parse_target = url

    parts = urlsplit(parse_target)
    domain = (parts.hostname or "").lower()
    path = parts.path or ""
    query = parts.query or ""

    # Strip scheme for whole-URL lexical counts (matches dataset convention)
    no_scheme = re.sub(r"^https?://", "", url, flags=re.IGNORECASE)
    if path == "/" and not query:
        no_scheme = no_scheme.rstrip("/")
        path = ""

    # Split path into directory (up to last '/') and file (after last '/')
    if "/" in path:
        last_slash = path.rfind("/")
        directory = path[: last_slash + 1]
        filename = path[last_slash + 1:] if last_slash + 1 < len(path) else ""
    else:
        directory = None
        filename = None

    # In the dataset, absent directory/file/params are encoded as -1
    has_dir = directory is not None
    has_file = filename is not None and len(filename) > 0
    has_params = len(query) > 0

    tld = domain.rsplit(".", 1)[-1] if "." in domain and not _is_ip(domain) else ""

    feats: dict[str, int] = {}

    # Table 1: Whole URL
    feats.update(_count_chars(no_scheme, "url"))
    feats["qty_tld_url"] = len(tld)
    feats["length_url"] = len(no_scheme)
    feats["email_in_url"] = int(bool(EMAIL_RE.search(no_scheme)))

    # Table 2: Domain
    feats.update(_count_chars(domain, "domain"))
    feats["qty_vowels_domain"] = sum(domain.count(v) for v in "aeiou")
    feats["domain_length"] = len(domain)
    feats["domain_in_ip"] = _is_ip(domain)
    feats["server_client_domain"] = int("server" in domain or "client" in domain)

    # Table 3: Directory
    feats.update(_count_chars(directory if has_dir else None, "directory"))
    feats["directory_length"] = len(directory) if has_dir else -1

    # Table 4: File
    feats.update(_count_chars(filename if has_file else None, "file"))
    feats["file_length"] = len(filename) if has_file else -1

    # Table 5: Query parameters
    feats.update(_count_chars(query if has_params else None, "params"))
    feats["params_length"] = len(query) if has_params else -1
    if has_params:
        tokens = re.findall(r"[a-z]{2,6}", query.lower())
        feats["tld_present_params"] = int(any(tok in COMMON_TLDS for tok in tokens))
        feats["qty_params"] = max(1, len(parse_qs(query, keep_blank_values=True)))
    else:
        feats["tld_present_params"] = -1
        feats["qty_params"] = -1

    return feats


# --------------------------------------------------------------------------- #
# Streamlit UI setup
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="PhishLite — URL Phishing Detector", layout="centered")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Montserrat', sans-serif; }
    .verdict-box {
        padding: 1.1rem 1.25rem;
        border-radius: 6px;
        border-left: 4px solid #111111;
        background-color: #f6f6f6;
        margin: 0.75rem 0 1.25rem 0;
    }
    .verdict-phish { border-left-color: #c0392b; background-color: #fdf2f2; }
    .verdict-legit { border-left-color: #1e8449; background-color: #f2faf5; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_bundle():
    if not BUNDLE_PATH.exists():
        return None
    return joblib.load(BUNDLE_PATH)


def render_verdict(prob: float, threshold: float, subtitle: str) -> None:
    is_phish = prob >= threshold
    label = "PHISHING DETECTED" if is_phish else "LIKELY LEGITIMATE"
    css = "verdict-phish" if is_phish else "verdict-legit"
    st.markdown(
        f"""
        <div class="verdict-box {css}">
            <div style="font-size:0.78rem;letter-spacing:0.08em;color:#000000;">{subtitle}</div>
            <div style="font-size:1.35rem;font-weight:700;margin:0.2rem 0;color:#000000">{label}</div>
            <div style="font-size:0.9rem;color:#333;">
                Phishing probability: <b>{prob:.2%}</b> &nbsp;|&nbsp; Decision threshold: <b>{threshold:.2f}</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(float(min(max(prob, 0.0), 1.0)))


# --------------------------------------------------------------------------- #
# Main App
# --------------------------------------------------------------------------- #
st.title("PhishLite")
st.caption("Lightweight machine learning phishing URL detector — live lexical screening & full-ensemble inspector.")

bundle = load_bundle()
if bundle is None or "url_only_model" not in bundle:
    st.error(
        f"`{BUNDLE_PATH}` is missing or incomplete. Run `python main.py part_b` "
        "(or `python main.py ensemble` followed by `python main.py eval`) first."
    )
    st.stop()

meta = bundle["meta"]
url_meta = bundle["url_only_meta"]
url_model = bundle["url_only_model"]
ensemble = bundle["ensemble"]
test_sample: pd.DataFrame = bundle["test_sample"]

tab_live, tab_full = st.tabs(["Live URL Scanner (URL-Only)", "Full-Feature Inspector (Test Samples)"])

# --- TAB 1: Live URL Scanner ---
with tab_live:
    st.write(
        "Enter any URL below. Features are extracted **locally from the URL string only** "
        "(no external DNS/WHOIS requests or page fetches), then scored by the deployable "
        f"**{url_meta['model_name']}** model trained on **{len(url_meta['feature_names'])}** lexical features."
    )

    sample_urls = {
        "Custom input": "",
        "Example legitimate (IEEE)": "https://www.ieee.org",
        "Example legitimate (GitHub)": "https://github.com",
        "Example suspicious (long path + params)": "http://secure-login-verify.example-account.com/auth/update/confirm.php?user=12345&redirect=paypal.com",
    }
    preset = st.selectbox("Quick test presets", list(sample_urls.keys()))
    url_input = st.text_input(
        "Target URL",
        value=sample_urls[preset],
        placeholder="https://example.com/login",
    )

    if st.button("Scan URL", type="primary", use_container_width=True):
        if not url_input.strip():
            st.warning("Please enter a URL to scan.")
        else:
            raw_feats = extract_url_features(url_input)
            X_row = pd.DataFrame([[raw_feats.get(c, -1) for c in url_meta["feature_names"]]],
                                 columns=url_meta["feature_names"])
            prob = float(url_model.predict_proba(X_row)[:, 1][0])
            render_verdict(prob, float(url_meta["threshold"]), f"URL-ONLY MODEL ({url_meta['model_name'].upper()})")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("URL Length", raw_feats["length_url"])
            c2.metric("Domain Length", raw_feats["domain_length"])
            c3.metric("Directory Length", raw_feats["directory_length"])
            c4.metric("Params Length", raw_feats["params_length"])

            with st.expander("View extracted model input features"):
                feat_df = pd.DataFrame({"Feature": url_meta["feature_names"], "Value": X_row.iloc[0].values})
                st.dataframe(feat_df, use_container_width=True, hide_index=True)

    st.caption(
        f"URL-only test performance — F1: {url_meta['test_metrics']['F1']:.4f} | "
        f"FPR: {url_meta['test_metrics']['FPR']:.4f} | ROC-AUC: {url_meta['test_metrics']['ROC_AUC']:.4f}"
    )

# --- TAB 2: Full-Feature Inspector ---
with tab_full:
    st.write(
        "Inspect held-out test records that include both lexical and external network-lookup features "
        f"(DNS, WHOIS, TTL). Scored by the winning **{meta['ensemble_method'].upper()}** ensemble "
        f"({', '.join(meta['ensemble_models'])})."
    )

    row_idx = st.slider("Select held-out test sample index", 0, len(test_sample) - 1, 0)
    row = test_sample.iloc[row_idx]
    true_label = int(row["phishing"])
    X_full_row = test_sample.iloc[[row_idx]][meta["feature_names"]]

    prob_ens = float(ensemble.predict_proba(X_full_row)[:, 1][0])
    render_verdict(
        prob_ens,
        float(meta["ensemble_threshold"]),
        f"FULL ENSEMBLE ({meta['ensemble_method'].upper()}) — TRUE LABEL: {'PHISHING' if true_label == 1 else 'LEGITIMATE'}",
    )

    # Show individual base model votes
    indiv = ensemble.individual_probas(X_full_row)
    vote_df = pd.DataFrame([
        {"Base Model": name, "Phishing Probability": round(float(p[0]), 4)}
        for name, p in indiv.items()
    ])
    st.subheader("Base Model Breakdown")
    st.dataframe(vote_df, use_container_width=True, hide_index=True)

    with st.expander("View all feature values for this sample"):
        f_df = pd.DataFrame({"Feature": meta["feature_names"], "Value": X_full_row.iloc[0].values})
        st.dataframe(f_df, use_container_width=True, hide_index=True)
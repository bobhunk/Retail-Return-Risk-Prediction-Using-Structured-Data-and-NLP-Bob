"""
Streamlit UI for Retail Return Prediction.

Uses FastAPI backend when available.
Falls back to local in-app model inference when API is unavailable.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "Tumushiime best model.joblib"
DATA_PATH = BASE_DIR / "Tumushiime.csv"
REQUEST_TIMEOUT = 12


def _default_api_url() -> str:
    """Pick API URL from Streamlit secrets/env; empty means local fallback mode."""
    try:
        if "API_URL" in st.secrets and str(st.secrets["API_URL"]).strip():
            return str(st.secrets["API_URL"]).strip()
    except Exception:
        pass
    return os.getenv("API_URL", "").strip()


def _normalize_base_url(url: str) -> str:
    cleaned = str(url).strip().rstrip("/")
    if not cleaned:
        return ""
    if not cleaned.startswith(("http://", "https://")):
        cleaned = f"http://{cleaned}"
    return cleaned


def _parse_json_response(resp: requests.Response) -> Tuple[Optional[Any], str]:
    try:
        return resp.json(), ""
    except ValueError:
        text = (resp.text or "").strip().replace("\n", " ")
        return None, text[:300]


def _check_api(api_base_url: str) -> Tuple[bool, Dict[str, Any], str]:
    """Validate that api_base_url is a compatible FastAPI backend."""
    if not api_base_url:
        return False, {}, "No FastAPI URL configured. Running in local fallback mode."

    try:
        health_resp = requests.get(f"{api_base_url}/health", timeout=REQUEST_TIMEOUT)
        health_json, raw = _parse_json_response(health_resp)

        if health_resp.status_code != 200:
            return False, {}, f"Health endpoint returned HTTP {health_resp.status_code}."

        if not isinstance(health_json, dict) or health_json.get("status") != "ok":
            if raw:
                return False, {}, "Endpoint does not look like the expected FastAPI service."
            return False, {}, "Invalid health response format from API."

        info_resp = requests.get(f"{api_base_url}/model-info", timeout=REQUEST_TIMEOUT)
        info_json, _ = _parse_json_response(info_resp)
        model_info = info_json if isinstance(info_json, dict) else {}
        return True, model_info, ""
    except requests.RequestException as exc:
        return False, {}, f"Network error: {exc}"


@st.cache_resource(show_spinner=False)
def _load_local_resources() -> Tuple[Optional[Dict[str, Any]], str]:
    """Load model and preprocessing assets for local fallback inference."""
    if not MODEL_PATH.exists():
        return None, f"Missing model file: {MODEL_PATH.name}"
    if not DATA_PATH.exists():
        return None, f"Missing data file: {DATA_PATH.name}"

    try:
        bundle = joblib.load(MODEL_PATH)
        if isinstance(bundle, dict):
            model = bundle["model"]
            threshold = float(bundle.get("threshold", 0.5))
            numeric_features = list(bundle.get("numeric_features", []))
            text_features = int(bundle.get("tfidf_vocab_size", 200))
        else:
            model = bundle
            threshold = 0.5
            numeric_features = [
                "UnitPrice", "AbsQuantity", "LineValueAbs", "InvoiceMonth",
                "InvoiceWeekday", "InvoiceHour", "CustomerIDMissing",
                "DescriptionLength", "CountryEncoded",
            ]
            text_features = 200

        ref = pd.read_csv(DATA_PATH, usecols=["Country", "DescriptionClean"], nrows=50000)
        ref["Country"] = ref["Country"].fillna("Unknown")
        ref["DescriptionClean"] = ref["DescriptionClean"].fillna("unknown")

        le = LabelEncoder()
        le.fit(ref["Country"])

        tfidf = TfidfVectorizer(max_features=text_features, ngram_range=(1, 2), min_df=5)
        tfidf.fit(ref["DescriptionClean"])

        return {
            "model": model,
            "threshold": threshold,
            "numeric_features": numeric_features,
            "label_encoder": le,
            "tfidf": tfidf,
        }, ""
    except Exception as exc:
        return None, f"Failed to load local resources: {exc}"


def _predict_local(payload: Dict[str, Any], resources: Dict[str, Any]) -> Dict[str, Any]:
    model = resources["model"]
    threshold = resources["threshold"]
    le = resources["label_encoder"]
    tfidf = resources["tfidf"]
    numeric_features: List[str] = resources["numeric_features"]

    country = str(payload.get("Country", "United Kingdom"))
    if country not in le.classes_:
        country = "United Kingdom" if "United Kingdom" in le.classes_ else str(le.classes_[0])
    country_encoded = int(le.transform([country])[0])

    unit_price = float(payload.get("UnitPrice", 0.0))
    quantity = float(payload.get("Quantity", 0.0))
    abs_qty = abs(quantity)
    description = str(payload.get("Description", ""))

    feature_map = {
        "UnitPrice": unit_price,
        "AbsQuantity": abs_qty,
        "LineValueAbs": unit_price * abs_qty,
        "InvoiceMonth": int(payload.get("InvoiceMonth", 6)),
        "InvoiceWeekday": int(payload.get("InvoiceWeekday", 2)),
        "InvoiceHour": int(payload.get("InvoiceHour", 12)),
        "CustomerIDMissing": 1 if not str(payload.get("CustomerID", "")).strip() else 0,
        "DescriptionLength": len(description),
        "CountryEncoded": country_encoded,
    }

    if not numeric_features:
        numeric_features = list(feature_map.keys())

    numeric_values = []
    for name in numeric_features:
        if name not in feature_map:
            raise KeyError(f"Unsupported feature required by model: {name}")
        numeric_values.append(float(feature_map[name]))

    x_num = csr_matrix([numeric_values])
    x_text = tfidf.transform([description.lower()])
    x_all = hstack([x_num, x_text])

    # HistGradientBoosting requires dense matrix input.
    if "HistGradient" in type(model).__name__:
        x_all = x_all.toarray()

    probs = model.predict_proba(x_all)[0]
    return_prob = float(probs[1])
    pred = int(return_prob >= threshold)

    return {
        "prediction": pred,
        "label": "RETURN" if pred == 1 else "Normal Sale",
        "return_probability": round(return_prob, 4),
        "normal_probability": round(float(probs[0]), 4),
    }

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Retail Return Predictor",
    page_icon="🛒",
    layout="wide"
)

st.title("🛒 Retail Return Risk Predictor")
st.caption("Powered by RandomForest | Tumushiime Final Exam Model")

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("API Settings")
    api_url = _normalize_base_url(
        st.text_input("FastAPI URL (optional)", value=_default_api_url(), placeholder="https://your-fastapi-service.onrender.com")
    )
    api_ok, api_info, api_error = _check_api(api_url)
    local_resources, local_error = _load_local_resources()
    use_local_fallback = (not api_ok) and (local_resources is not None)

    st.divider()
    if api_ok:
        st.success("FastAPI backend connected ✅")
        if api_info:
            st.json(api_info)
    else:
        if "streamlit.app" in api_url:
            st.warning("This URL looks like a Streamlit frontend, not a FastAPI backend.")

        if use_local_fallback:
            st.info("Using local in-app model fallback ✅")
        else:
            st.error("Cannot reach a valid FastAPI backend.")

        if api_error and api_url:
            st.caption(api_error)
        if local_error:
            st.caption(local_error)

# ── Tabs ───────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction (CSV)"])

# ── Tab 1: Single prediction ───────────────────────────────────────────────
with tab1:
    st.subheader("Enter Transaction Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        description = st.text_input("Product Description", value="WHITE HANGING HEART T-LIGHT HOLDER")
        unit_price  = st.number_input("Unit Price (£)", min_value=0.0, value=2.55, step=0.01)
        quantity    = st.number_input("Quantity", value=6, step=1)

    with col2:
        country    = st.selectbox("Country", [
            "United Kingdom", "Germany", "France", "Spain", "Netherlands",
            "Belgium", "Switzerland", "Australia", "Norway", "EIRE", "Other"
        ])
        customer_id = st.text_input("Customer ID (leave blank if unknown)", value="")

    with col3:
        inv_month   = st.slider("Invoice Month", 1, 12, 6)
        inv_weekday = st.slider("Invoice Weekday (0=Mon, 6=Sun)", 0, 6, 2)
        inv_hour    = st.slider("Invoice Hour", 6, 20, 12)

    if st.button("🔍 Predict Return Risk", type="primary"):
        payload = {
            "UnitPrice":      unit_price,
            "Quantity":       quantity,
            "Description":    description,
            "Country":        country,
            "InvoiceMonth":   inv_month,
            "InvoiceWeekday": inv_weekday,
            "InvoiceHour":    inv_hour,
            "CustomerID":     customer_id if customer_id.strip() else None,
        }

        result = None
        try:
            if api_ok:
                resp = requests.post(f"{api_url}/predict", json=payload, timeout=REQUEST_TIMEOUT)
                parsed, raw = _parse_json_response(resp)

                if parsed is None:
                    st.error("API returned non-JSON response. Check FastAPI URL in sidebar.")
                    if raw:
                        st.code(raw, language="text")
                elif resp.status_code != 200:
                    st.error(f"API error ({resp.status_code}): {parsed}")
                else:
                    result = parsed
            else:
                if local_resources is None:
                    st.error("No valid API and no local fallback model available.")
                else:
                    result = _predict_local(payload, local_resources)

            if result is not None:
                st.divider()
                label = result["label"]
                prob = float(result["return_probability"])

                if int(result["prediction"]) == 1:
                    st.error(f"⚠️ Prediction: **{label}**")
                else:
                    st.success(f"✅ Prediction: **{label}**")

                c1, c2, c3 = st.columns(3)
                c1.metric("Return Probability", f"{prob*100:.1f}%")
                c2.metric("Normal Probability", f"{float(result['normal_probability'])*100:.1f}%")
                c3.metric("Risk Level", "HIGH" if prob > 0.5 else "LOW")

                with st.expander("Raw JSON response"):
                    st.json(result)
        except Exception as e:
            st.error(f"Request failed: {e}")

# ── Tab 2: Batch prediction ────────────────────────────────────────────────
with tab2:
    st.subheader("Upload a CSV for Batch Prediction")
    st.caption("CSV must have columns: Description, UnitPrice, Quantity, Country")

    uploaded = st.file_uploader("Choose CSV file", type="csv")

    if uploaded:
        df = pd.read_csv(uploaded)
        st.write("Preview:", df.head())

        if st.button("Run Batch Prediction", type="primary"):
            transactions = []
            for _, row in df.iterrows():
                transactions.append({
                    "Description":    str(row.get("Description", "unknown")),
                    "UnitPrice":      float(row.get("UnitPrice", 1.0)),
                    "Quantity":       float(row.get("Quantity", 1)),
                    "Country":        str(row.get("Country", "United Kingdom")),
                    "InvoiceMonth":   int(row.get("InvoiceMonth", 6)),
                    "InvoiceWeekday": int(row.get("InvoiceWeekday", 2)),
                    "InvoiceHour":    int(row.get("InvoiceHour", 12)),
                    "CustomerID":     str(row.get("CustomerID", "")) or None,
                })

            try:
                if api_ok:
                    resp = requests.post(
                        f"{api_url}/batch-predict",
                        json={"transactions": transactions},
                        timeout=REQUEST_TIMEOUT * 2,
                    )
                    parsed, raw = _parse_json_response(resp)

                    if parsed is None:
                        st.error("API returned non-JSON response. Check FastAPI URL in sidebar.")
                        if raw:
                            st.code(raw, language="text")
                        st.stop()
                    if resp.status_code != 200:
                        st.error(f"API error ({resp.status_code}): {parsed}")
                        st.stop()

                    results = parsed.get("results", [])
                else:
                    if local_resources is None:
                        st.error("No valid API and no local fallback model available.")
                        st.stop()

                    results = []
                    for t in transactions:
                        try:
                            pred = _predict_local(t, local_resources)
                            results.append({
                                "description": t["Description"],
                                "prediction": pred["prediction"],
                                "label": pred["label"],
                                "return_probability": pred["return_probability"],
                            })
                        except Exception as exc:
                            results.append({"description": t["Description"], "error": str(exc)})

                result_df = pd.DataFrame(results)
                st.dataframe(result_df)

                csv_out = result_df.to_csv(index=False).encode()
                st.download_button("Download Results CSV", csv_out, "predictions.csv", "text/csv")
            except Exception as e:
                st.error(f"Batch request failed: {e}")
    else:
        st.info("Or try a quick demo with sample data:")
        if st.button("Run Demo Batch"):
            demo = [
                {"Description": "WHITE HANGING HEART T-LIGHT HOLDER", "UnitPrice": 2.55,
                 "Quantity": 6, "Country": "United Kingdom", "InvoiceMonth": 12,
                 "InvoiceWeekday": 1, "InvoiceHour": 8, "CustomerID": "17850"},
                {"Description": "damaged broken item refund", "UnitPrice": 3.39,
                 "Quantity": -2, "Country": "Germany", "InvoiceMonth": 11,
                 "InvoiceWeekday": 3, "InvoiceHour": 14, "CustomerID": None},
                {"Description": "CREAM CUPID HEARTS COAT HANGER", "UnitPrice": 2.75,
                 "Quantity": 8, "Country": "France", "InvoiceMonth": 6,
                 "InvoiceWeekday": 0, "InvoiceHour": 11, "CustomerID": "15311"},
            ]
            try:
                if api_ok:
                    resp = requests.post(
                        f"{api_url}/batch-predict",
                        json={"transactions": demo},
                        timeout=REQUEST_TIMEOUT,
                    )
                    parsed, raw = _parse_json_response(resp)
                    if parsed is None:
                        st.error("API returned non-JSON response. Check FastAPI URL in sidebar.")
                        if raw:
                            st.code(raw, language="text")
                    elif resp.status_code != 200:
                        st.error(f"API error ({resp.status_code}): {parsed}")
                    else:
                        st.dataframe(pd.DataFrame(parsed.get("results", [])))
                else:
                    if local_resources is None:
                        st.error("No valid API and no local fallback model available.")
                    else:
                        rows = []
                        for t in demo:
                            try:
                                pred = _predict_local(t, local_resources)
                                rows.append({
                                    "description": t["Description"],
                                    "prediction": pred["prediction"],
                                    "label": pred["label"],
                                    "return_probability": pred["return_probability"],
                                })
                            except Exception as exc:
                                rows.append({"description": t["Description"], "error": str(exc)})
                        st.dataframe(pd.DataFrame(rows))
            except Exception as e:
                st.error(f"Demo failed: {e}")

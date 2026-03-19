"""
Streamlit UI for Retail Return Prediction
Calls FastAPI backend at http://127.0.0.1:8000
"""

import streamlit as st
import requests
import pandas as pd
import json
import os


def _default_api_url() -> str:
    """Pick API URL from Streamlit secrets, env var, then local fallback."""
    try:
        if "API_URL" in st.secrets:
            return str(st.secrets["API_URL"])
    except Exception:
        pass
    return os.getenv("API_URL", "http://127.0.0.1:8000")

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
    api_url = st.text_input("FastAPI URL", value=_default_api_url())

    st.divider()
    try:
        r = requests.get(f"{api_url}/health", timeout=3)
        if r.status_code == 200:
            st.success("API is running ✅")
            info = requests.get(f"{api_url}/model-info", timeout=3).json()
            st.json(info)
        else:
            st.error("API error")
    except Exception:
        st.error("Cannot reach API — is uvicorn running?")

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
        try:
            resp = requests.post(f"{api_url}/predict", json=payload, timeout=10)
            result = resp.json()

            if resp.status_code == 200:
                st.divider()
                label = result["label"]
                prob  = result["return_probability"]

                if result["prediction"] == 1:
                    st.error(f"⚠️ Prediction: **{label}**")
                else:
                    st.success(f"✅ Prediction: **{label}**")

                c1, c2, c3 = st.columns(3)
                c1.metric("Return Probability",  f"{prob*100:.1f}%")
                c2.metric("Normal Probability",  f"{result['normal_probability']*100:.1f}%")
                c3.metric("Risk Level", "HIGH" if prob > 0.5 else "LOW")

                with st.expander("Raw JSON response"):
                    st.json(result)
            else:
                st.error(f"API error: {result}")
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
                resp = requests.post(
                    f"{api_url}/batch-predict",
                    json={"transactions": transactions},
                    timeout=30
                )
                results = resp.json()["results"]
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
                resp = requests.post(
                    f"{api_url}/batch-predict",
                    json={"transactions": demo},
                    timeout=15
                )
                st.dataframe(pd.DataFrame(resp.json()["results"]))
            except Exception as e:
                st.error(f"Demo failed: {e}")

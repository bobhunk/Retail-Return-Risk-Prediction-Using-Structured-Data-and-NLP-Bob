"""
FastAPI service for Retail Return Prediction model (Tumushiime)
Endpoints: /health, /model-info, /predict, /batch-predict
"""

import os
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "Tumushiime best model.joblib")
DATA_PATH  = os.path.join(BASE_DIR, "Tumushiime.csv")

# ── Globals set at startup ──────────────────────────────────────────────────
model            = None
tfidf            = None
country_map      = {}
N_NUMERIC        = 9
THRESHOLD        = 0.5
NUMERIC_FEATURES = []

app = FastAPI(title="Retail Return Predictor", version="1.0")


# ── Startup ────────────────────────────────────────────────────────────────
@app.on_event("startup")
def load_model():
    global model, tfidf, country_map, N_NUMERIC, THRESHOLD, NUMERIC_FEATURES

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model file not found: {MODEL_PATH}")

    bundle = joblib.load(MODEL_PATH)

    # Support both a plain model and the saved dict bundle
    if isinstance(bundle, dict):
        model            = bundle["model"]
        THRESHOLD        = float(bundle.get("threshold", 0.5))
        NUMERIC_FEATURES = list(bundle.get("numeric_features", []))
        N_NUMERIC        = len(NUMERIC_FEATURES)
    else:
        model            = bundle
        THRESHOLD        = 0.5
        N_NUMERIC        = model.n_features_in_ - 200
        NUMERIC_FEATURES = []

    # Fit TF-IDF on reference data
    if os.path.exists(DATA_PATH):
        ref = pd.read_csv(DATA_PATH, usecols=["DescriptionClean"], nrows=50000)
        ref["DescriptionClean"] = ref["DescriptionClean"].fillna("unknown")
        tfidf = TfidfVectorizer(max_features=200, ngram_range=(1, 2), min_df=5)
        tfidf.fit(ref["DescriptionClean"])

        # Build country encoding map
        ref2 = pd.read_csv(DATA_PATH, usecols=["Country"], nrows=50000)
        le = LabelEncoder()
        le.fit(ref2["Country"].fillna("Unknown"))
        country_map = {c: i for i, c in enumerate(le.classes_)}
    else:
        # Minimal fallback TF-IDF
        tfidf = TfidfVectorizer(max_features=200, ngram_range=(1, 2), min_df=1)
        tfidf.fit(["unknown item"])

    print(f"Model loaded. Expects {model.n_features_in_} features ({N_NUMERIC} numeric + 200 TF-IDF). Threshold={THRESHOLD:.4f}")


# ── Pydantic schemas ───────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    UnitPrice:       float
    Quantity:        float
    Description:     str
    Country:         Optional[str] = "United Kingdom"
    InvoiceMonth:    Optional[int] = 6
    InvoiceWeekday:  Optional[int] = 2
    InvoiceHour:     Optional[int] = 12
    CustomerID:      Optional[str] = None

class PredictResponse(BaseModel):
    prediction:        int
    label:             str
    return_probability: float
    normal_probability: float

class BatchPredictRequest(BaseModel):
    transactions: List[PredictRequest]


# ── Feature builder ────────────────────────────────────────────────────────
def build_features(req: PredictRequest) -> np.ndarray:
    abs_qty      = abs(req.Quantity)
    line_val     = req.UnitPrice * abs_qty
    cust_missing = 1 if (req.CustomerID is None or str(req.CustomerID).strip() == "") else 0
    desc_len     = len(req.Description)
    country_enc  = country_map.get(req.Country, 0)

    if N_NUMERIC == 9:
        numeric = np.array([[
            req.UnitPrice, abs_qty, line_val,
            req.InvoiceMonth, req.InvoiceWeekday, req.InvoiceHour,
            cust_missing, desc_len, country_enc
        ]])
    else:  # 12 features with cyclic encoding
        month_sin  = np.sin(2 * np.pi * req.InvoiceMonth / 12)
        month_cos  = np.cos(2 * np.pi * req.InvoiceMonth / 12)
        wday_sin   = np.sin(2 * np.pi * req.InvoiceWeekday / 7)
        wday_cos   = np.cos(2 * np.pi * req.InvoiceWeekday / 7)
        numeric = np.array([[
            req.UnitPrice, abs_qty, line_val,
            month_sin, month_cos, wday_sin, wday_cos,
            req.InvoiceHour, cust_missing, desc_len, country_enc,
            line_val  # duplicate filler
        ]])

    text_vec = tfidf.transform([req.Description.lower()])
    combined = hstack([csr_matrix(numeric), text_vec])
    return combined.toarray()  # HistGradientBoosting requires dense input


# ── Endpoints ──────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/model-info")
def model_info():
    return {
        "model_type":       type(model).__name__,
        "n_features":       int(model.n_features_in_),
        "n_numeric":        N_NUMERIC,
        "numeric_features": NUMERIC_FEATURES,
        "n_text":           200,
        "threshold":        THRESHOLD,
        "tfidf_fitted":     tfidf is not None,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        X    = build_features(req)
        prob = model.predict_proba(X)[0]
        pred = int(prob[1] >= THRESHOLD)   # use saved optimal threshold
        return PredictResponse(
            prediction=pred,
            label="RETURN" if pred == 1 else "Normal Sale",
            return_probability=round(float(prob[1]), 4),
            normal_probability=round(float(prob[0]), 4),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch-predict")
def batch_predict(req: BatchPredictRequest):
    results = []
    for t in req.transactions:
        try:
            X    = build_features(t)
            prob = model.predict_proba(X)[0]
            pred = int(prob[1] >= THRESHOLD)
            results.append({
                "description":        t.Description,
                "prediction":         pred,
                "label":              "RETURN" if pred == 1 else "Normal Sale",
                "return_probability": round(float(prob[1]), 4),
            })
        except Exception as e:
            results.append({"description": t.Description, "error": str(e)})
    return {"results": results, "count": len(results)}

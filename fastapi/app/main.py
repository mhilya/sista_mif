#!/usr/bin/env python3
"""
TRACER STUDY - FASTAPI CLASSIFICATION WORKER
Phase 3: Hybrid Rule + ML Fallback + Confidence Routing
Sumber: Internal MIF (klasifikasi) & Kemendiktisaintek (dashboard only)
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
import joblib
import re
import io
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Dict, Any

# ──────────────────────────────────────────────────────────────
# KONFIGURASI & KONSTANTA
# ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
ML_DIR = BASE_DIR / "ml_assets"
PIPELINE_PATH = ML_DIR / "ml_pipeline_internal.pkl"
CONFIDENCE_THRESHOLD = 0.75
TARGET_CLASSES = ["Programmer", "Data Analyst", "Wirausaha Informatika", "Non-IT"]

# Keyword Rules (Prioritas: IT Spesifik → Wirausaha → Fallback)
KEYWORD_RULES = {
    "Programmer": ["programmer", "developer", "engineer", "fullstack", "backend", "frontend", "mobile", "android", "ios", "software", "web dev", "coding", "it staff", "teknisi", "sistem informasi", "application", "network", "devops", "qa", "tester", "ui", "ux", "swe"],
    "Data Analyst": ["data analyst", "analis data", "data science", "business analyst", "research", "statistik", "bi analyst", "reporting", "database", "sql", "etl", "data engineer", "big data", "analyst", "data mining", "machine learning", "data visual", "power bi", "tableau", "looker", "business intelligence", "bi developer", "data warehouse"],
    "Wirausaha Informatika": ["founder", "owner", "ceo", "wiraswasta", "startup", "freelance", "freelancer", "wirausaha", "bisnis", "usaha mandiri", "konsultan", "co founder", "entrepreneur", "self employed", "owner toko", "usaha", "dagang online", "tokopedia", "shopee", "dropship", "reseller"]
}

# Mapping Kode Kemendik → Keyword
F5C_MAP = {"1": "founder owner wirausaha startup", "2": "co-founder partner wirausaha", "3": "staff karyawan pegawai", "4": "freelance kerja lepas lepasan"}
F1101_MAP = {"1": "instansi pemerintah dinas kementerian", "2": "non-profit lsm yayasan", "3": "perusahaan swasta corporate", "4": "wiraswasta usaha mandiri", "6": "bumn bumd pemerintah", "7": "multilateral internasional"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("tracer_worker")


# ──────────────────────────────────────────────────────────────
# LIFESPAN & MODEL LOADING
# ──────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.pipeline = joblib.load(PIPELINE_PATH)
        logger.info(f"✅ ML Pipeline loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load ML pipeline: {e}")
        app.state.pipeline = None
    yield

app = FastAPI(title="Tracer Study Classification Worker", lifespan=lifespan)


# ──────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ──────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    if pd.isna(text) or str(text).strip().lower() in ["nan", "none", "null", "-", "0", "", "tidak diisi"]:
        return ""
    text = str(text).strip().lower()
    text = text.replace('-', ' ').replace('/', ' ').replace('_', ' ').replace(',', ' ')
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    tokens = [w for w in text.split() if len(w) >= 3 and w.isalpha()]
    return " ".join(tokens)

def detect_source(df: pd.DataFrame) -> str:
    """Deteksi sumber data berdasarkan pola nama kolom (case-insensitive, substring match)"""
    cols_lower = [c.lower().strip() for c in df.columns]
    
    # Kemendik codes: cek apakah ADA kode ini di dalam nama kolom (substring match)
    kemendik_codes = ["f5b", "f5c", "f8", "f1101", "nimhsmsmh", "nmmhsmsmh"]
    for col in cols_lower:
        if any(code in col for code in kemendik_codes):
            return "kemendik"
    
    # Internal MIF: cek kolom 'jabatan'
    if any("jabatan" in c for c in cols_lower):
        return "internal_mif"
    
    return "unknown"

def classify_rule(job_text: str) -> dict:
    clean = clean_text(job_text)
    if not clean or len(clean) < 3: return {"profile": "Non-IT", "confidence": 1.0, "method": "rule_based_fallback"}
    for profile, keywords in KEYWORD_RULES.items():
        if any(kw in clean for kw in keywords):
            return {"profile": profile, "confidence": 1.0, "method": "rule_based"}
    return None

def classify_ml(job_text: str, pipeline) -> dict:
    clean = clean_text(job_text)
    if not clean: return {"profile": "Non-IT", "confidence": 0.0, "method": "ml_fallback"}
    try:
        proba = pipeline.predict_proba([clean])[0]
        max_conf = float(np.max(proba))
        pred_class = pipeline.classes_[np.argmax(proba)]
        method = "ml_fallback" if max_conf >= CONFIDENCE_THRESHOLD else "manual_review"
        return {"profile": pred_class, "confidence": round(max_conf, 4), "method": method}
    except Exception as e:
        logger.error(f"ML Inference error: {e}")
        return {"profile": "Non-IT", "confidence": 0.0, "method": "ml_error"}

def extract_kemendik_text(row: pd.Series, df_columns: list) -> str:
    """Ekstrak teks Kemendik dengan fuzzy column matching"""
    # Helper: cari kolom yang mengandung pattern (case-insensitive)
    def find_col(pattern: str) -> str | None:
        for col in df_columns:
            if pattern.lower() in col.lower():
                return col
        return None
    
    # Temukan kolom aktual
    f5b_col = find_col("f5b")
    f5c_col = find_col("f5c") 
    f1101_col = find_col("f1101")
    f1102_col = find_col("f1102")
    
    # Ekstrak nilai dengan fallback aman
    f5b = clean_text(row.get(f5b_col, "") if f5b_col else "")
    f1102 = clean_text(row.get(f1102_col, "") if f1102_col else "")
    
    # Parse kode F5c & F1101 (format: "1 - Founder" atau "3")
    f5c_raw = str(row.get(f5c_col, "")).strip() if f5c_col else ""
    f5c_code = re.match(r'^(\d+)', f5c_raw)
    f5c_text = F5C_MAP.get(f5c_code.group(1), "") if f5c_code else ""
    
    f1101_raw = str(row.get(f1101_col, "")).strip() if f1101_col else ""
    f1101_code = re.match(r'^(\d+)', f1101_raw)
    f1101_text = F1101_MAP.get(f1101_code.group(1), "") if f1101_code else ""
    
    return " ".join(p for p in [f5b, f5c_text, f1101_text, f1102] if p).strip()


# ──────────────────────────────────────────────────────────────
# ROUTES
# ──────────────────────────────────────────────────────────────
@app.get("/health")
def health_check():
    return {"status": "healthy", "pipeline_loaded": app.state.pipeline is not None}

@app.post("/api/v1/classify")
async def classify_tracer(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    try:
        contents = await file.read()
        
        # Parse file dengan fallback encoding
        if file.filename.endswith(".xlsx") or file.filename.endswith(".xls"):
            try:
                df = pd.read_excel(io.BytesIO(contents), dtype=str, engine="openpyxl")
            except ImportError:
                logger.error("openpyxl not installed")
                raise HTTPException(status_code=500, detail="Server misconfiguration: openpyxl missing")
            except Exception as excel_err:
                logger.error(f"Excel parsing error: {excel_err}")
                # Fallback ke CSV
                try:
                    df = pd.read_csv(io.BytesIO(contents), sep=";", encoding="utf-8-sig", dtype=str)
                except:
                    df = pd.read_csv(io.BytesIO(contents), sep=";", encoding="latin-1", dtype=str)
        else:
            try:
                df = pd.read_csv(io.BytesIO(contents), sep=";", encoding="utf-8-sig", dtype=str)
            except UnicodeDecodeError:
                df = pd.read_csv(io.BytesIO(contents), sep=";", encoding="latin-1", dtype=str)
        
        df.columns = df.columns.str.strip()
        
    except Exception as e:
        logger.error(f"File parsing error: {type(e).__name__}: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to parse file: {type(e).__name__}: {str(e)[:200]}")

    source_type = detect_source(df)
    if source_type == "unknown":
        raise HTTPException(status_code=400, detail="Unrecognized file format.")

    if source_type == "internal_mif" and app.state.pipeline is None:
        logger.error("ML pipeline not loaded")
        raise HTTPException(status_code=503, detail="ML model not available.")

    results = []
    pipeline = app.state.pipeline
    logger.info(f"Processing {len(df)} rows | Source: {source_type}")

    for _, row in df.iterrows():
        try:
            # Safe string extraction: handle NaN/None/float
            def safe_str(val):
                if pd.isna(val) or val is None:
                    return ""
                return str(val).strip()
            
            nim = safe_str(row.get("nimhsmsmh" if source_type=="kemendik" else "nim"))
            nama = safe_str(row.get("nmmhsmsmh" if source_type=="kemendik" else "nama_lengkap"))
            tahun = safe_str(row.get("tahun_lulus"))

            if source_type == "kemendik":
                job_text = extract_kemendik_text(row, df.columns.tolist())
                results.append({
                    "nim": nim, "nama": nama, "tahun_lulus": tahun,
                    "job_text_raw": job_text, "source_type": "kemendik",
                    "predicted_profile": None, "confidence_score": None,
                    "classification_method": "dashboard_only", "status": "processed"
                })
            else:
                # Internal MIF classification
                col_jabatan = next((c for c in df.columns if "jabatan" in c.lower()), None)
                col_perusahaan = next((c for c in df.columns if "nama_perusahaan" in c.lower() or "perusahaan" in c.lower()), None)
                col_deskripsi = next((c for c in df.columns if "deskripsi" in c.lower()), None)

                job_text = clean_text(safe_str(row.get(col_jabatan))) if col_jabatan else ""
                if not job_text and col_perusahaan and col_deskripsi:
                    perusahaan = safe_str(row.get(col_perusahaan))
                    deskripsi = safe_str(row.get(col_deskripsi))
                    job_text = clean_text(f"{perusahaan} {deskripsi}".strip())
                elif not job_text and col_perusahaan:  # Fallback hanya perusahaan
                    job_text = clean_text(safe_str(row.get(col_perusahaan)))

                # Debug log jika masih kosong
                if not job_text:
                    logger.warning(f"Empty job_text for NIM {nim} | Available cols: {[c for c in df.columns if 'jabatan' in c.lower() or 'perusahaan' in c.lower() or 'deskripsi' in c.lower()]}")

                # Classification logic
                rule_res = classify_rule(job_text)
                if rule_res:
                    res = rule_res
                elif pipeline:
                    res = classify_ml(job_text, pipeline)
                else:
                    res = {"profile": "Non-IT", "confidence": 0.0, "method": "ml_unavailable"}

                status = "auto_classified" if res["method"] in ["rule_based", "ml_fallback"] else "needs_review"
                
                results.append({
                    "nim": nim, "nama": nama, "tahun_lulus": tahun,
                    "job_text_raw": job_text, "source_type": "internal_mif",
                    "predicted_profile": res["profile"], "confidence_score": res["confidence"],
                    "classification_method": res["method"], "status": status
                })
        except Exception as row_err:
            logger.warning(f"Row processing error: {row_err}")
            # Still include row with error status untuk audit trail
            results.append({
                "nim": nim if 'nim' in locals() else "", 
                "nama": nama if 'nama' in locals() else "", 
                "tahun_lulus": tahun if 'tahun' in locals() else "",
                "job_text_raw": "", "source_type": source_type,
                "predicted_profile": None, "confidence_score": None,
                "classification_method": "error", "status": "failed",
                "error_detail": str(row_err)[:100]
            })

    return JSONResponse(content={
        "status": "success", 
        "total_rows": len(df), 
        "processed_rows": len(results),
        "source_type": source_type, 
        "results": results
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
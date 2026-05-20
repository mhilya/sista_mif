#!/usr/bin/env python3
"""
TRACER STUDY - DATA PREPARATION (INTERNAL MIF ONLY) - STRICT JOB-ONLY
Perbaikan: Menghindari kontaminasi nama perusahaan atau orang pada job_text_raw.
Hanya menggunakan kolom 'jabatan'. Fallback disesuaikan dari 'klasifikasi_pekerjaan'.
"""

import pandas as pd
import numpy as np
import re
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR
OUTPUT_DIR = DATA_DIR / "processed"
OUTPUT_DIR.mkdir(exist_ok=True)

FILE_INTERNAL = DATA_DIR / "ts_internal_mif.xlsx"
SEPARATOR = ";"
ENCODING = "utf-8-sig"

TARGET_CLASSES = ["Programmer", "Data Analyst", "Wirausaha Informatika", "Non-IT"]

KLASIFIKASI_MAP = {
    "Programmer": "Programmer",
    "Data Analyst": "Data Analyst",
    "Wirausaha IT": "Wirausaha Informatika",
    "Wirausaha": "Wirausaha Informatika",
    "Non IT": "Non-IT",
    "Infokom": None, "Pelajar": None, "Tidak Bekerja": None, "TIdah diketahui": None
}

KEYWORD_RULES = {
    "Programmer": ["programmer", "developer", "engineer", "fullstack", "backend", "frontend", "mobile", "android", "ios", "software", "web dev", "coding", "it staff", "teknisi", "sistem informasi", "application", "network", "devops", "qa", "tester", "ui", "ux", "swe"],
    "Data Analyst": ["data analyst", "analis data", "data science", "business analyst", "research", "statistik", "bi analyst", "reporting", "database", "sql", "etl", "data engineer", "big data", "analyst", "data mining", "machine learning", "data visual", "power bi", "tableau", "looker", "business intelligence", "bi developer", "data warehouse"],
    "Wirausaha Informatika": ["founder", "owner", "ceo", "wiraswasta", "startup", "freelance", "freelancer", "wirausaha", "bisnis", "usaha mandiri", "konsultan", "co founder", "entrepreneur", "self employed", "owner toko", "usaha", "dagang online", "tokopedia", "shopee", "dropship", "reseller"]
}

# Daftar kata institusi atau perusahaan yang akan diabaikan dalam fitur jabatan
COMPANY_STOPWORDS = {
    "pt", "cv", "ud", "tbk", "persero", "corp", "inc", "ltd", "koperasi", "bumn", "bumd",
    "dinas", "kantor", "pemkab", "pemprov", "politeknik", "universitas", "sekolah", "sma", "smk", "sd",
    "bank", "bpr", "rs", "rumah sakit", "klinik", "apotek", "hotel", "restoran", "cafe", "toko", "konter",
    "foundation", "yayasan", "perkumpulan", "organisasi", "agency", "studio", "consulting", "group", "holding"
}

def load_data_file(file_path: str) -> pd.DataFrame:
    ext = Path(file_path).suffix.lower()
    if ext == '.csv':
        try:
            return pd.read_csv(file_path, sep=SEPARATOR, encoding=ENCODING, dtype=str, on_bad_lines='skip', engine='python')
        except UnicodeDecodeError:
            return pd.read_csv(file_path, sep=SEPARATOR, encoding='latin1', dtype=str, on_bad_lines='skip', engine='python')
    elif ext == '.xlsx':
        return pd.read_excel(file_path, dtype=str)
    else:
        raise ValueError(f"Format tidak didukung: {ext}")

def find_col(df: pd.DataFrame, keywords: list) -> str | None:
    for col in df.columns:
        col_clean = str(col).strip().lower()
        if any(k.strip().lower() in col_clean for k in keywords):
            return col
    return None

def clean_text(text: str) -> str:
    if pd.isna(text) or str(text).strip().lower() in ["nan", "none", "null", "-", "0", "", "tidak diisi", "tidak diketahui"]:
        return ""
    text = str(text).strip().lower()
    text = re.sub(r'^\d+\s*[-:/]\s*', '', text)
    text = text.replace('-', ' ').replace('/', ' ').replace('_', ' ').replace(',', ' ')
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    
    # Penyaringan token: menghapus stopwords, indikator perusahaan, dan kata di bawah 3 karakter
    tokens = [w for w in text.split() if w not in COMPANY_STOPWORDS and len(w) >= 3]
    return " ".join(tokens)

def is_likely_name(text: str, full_name: str) -> bool:
    """Pendeteksian teks jika lebih mengarah pada nama orang alih-alih jabatan"""
    if not text or not full_name: return False
    name_parts = set(str(full_name).lower().split())
    text_parts = set(text.split())
    # Mengindikasi kebocoran nama jika lebih dari 50% token job_text terdapat pada nama lengkap
    if len(text_parts & name_parts) / max(len(text_parts), 1) > 0.5:
        return True
    return False

def classify_rule_based(text: str) -> str:
    if not text or len(text) < 3: return "Non-IT"
    for profile in ["Programmer", "Data Analyst", "Wirausaha Informatika"]:
        if any(kw in text for kw in KEYWORD_RULES[profile]):
            return profile
    return "Non-IT"

def main():
    print("=== TRACER STUDY: INTERNAL MIF STRICT JOB-ONLY PIPELINE ===")
    if not FILE_INTERNAL.exists():
        print(f" Berkas tidak ditemukan: {FILE_INTERNAL}"); sys.exit(1)

    print("[1/4] Memuat & Membersihkan Data (Strict Cleaning)...")
    df = load_data_file(str(FILE_INTERNAL))
    df.columns = df.columns.str.strip()

    col_nim = find_col(df, ["nim"])
    col_nama = find_col(df, ["nama", "lengkap"])
    col_jabatan = find_col(df, ["jabatan"])
    col_klasifikasi = find_col(df, ["klasifikasi"])
    col_status = find_col(df, ["status", "kerja"])

    if not all([col_nim, col_nama, col_jabatan]):
        print(" Kolom esensial tidak ditemukan"); print(df.columns.tolist()[:10]); sys.exit(1)

    # Menyaring status pekerjaan
    if col_status:
        blacklist = ["tidah diketahui", "tidak bekerja", "pelajar", "melanjutkan pendidikan", "nan", ""]
        mask = ~df[col_status].str.lower().str.strip().isin(blacklist)
        df = df[mask].copy()

    # STRICT: Menjadikan kolom 'jabatan' sebagai acuan utama, tanpa fallback ke kolom perusahaan.
    df["job_text_raw"] = df[col_jabatan].apply(clean_text)
    
    # Menggunakan kata kunci generik dari klasifikasi_pekerjaan jika jabatan kosong
    if col_klasifikasi:
        fallback_map = {
            "Programmer": "programmer developer",
            "Data Analyst": "data analyst",
            "Wirausaha IT": "wirausaha founder",
            "Wirausaha": "wirausaha founder",
            "Non IT": "staff admin",
            "Infokom": "it staff teknisi",
            "TIdah diketahui": "", "Pelajar": "", "Tidak Bekerja": ""
        }
        empty_mask = df["job_text_raw"] == ""
        if empty_mask.any():
            df.loc[empty_mask, "job_text_raw"] = df.loc[empty_mask, col_klasifikasi].map(fallback_map).fillna("")

    # Validasi: Menghapus baris yang terindikasi sebagai nama orang atau institusi
    if col_nama:
        name_leak_mask = df.apply(lambda row: is_likely_name(row["job_text_raw"], row[col_nama]), axis=1)
        leaked_count = name_leak_mask.sum()
        if leaked_count > 0:
            print(f"  Mengabaikan {leaked_count} baris yang terindikasi mengandung nama pribadi.")
        df = df[~name_leak_mask].copy()

    # Pelabelan Data
    if col_klasifikasi:
        df["label"] = df[col_klasifikasi].str.strip().map(KLASIFIKASI_MAP)
        missing = df["label"].isna()
        if missing.any():
            df.loc[missing, "label"] = df.loc[missing, "job_text_raw"].apply(classify_rule_based)
    else:
        df["label"] = df["job_text_raw"].apply(classify_rule_based)

    df = df[df["job_text_raw"].str.len() >= 3].copy()
    result = df[[col_nim, col_jabatan, "job_text_raw", "label"]].rename(columns={col_nim: "nim"}).dropna(subset=["nim", "label"]).drop_duplicates(subset=["nim"], keep="first")
    print(f"  Memuat {len(result)} data valid (berdasarkan teks pekerjaan)\n")

    # Pembagian Dataset
    min_count = result["label"].value_counts().min()
    if min_count < 2:
        print("Data minoritas kurang dari 2 sampel. Melakukan pembagian secara acak.")
        train_df, test_df = train_test_split(result, test_size=0.30, random_state=42)
    else:
        train_df, test_df = train_test_split(result, test_size=0.30, stratify=result["label"], random_state=42)

    cols_out = ["nim", "job_text_raw", "label"]
    train_df[cols_out].to_csv(OUTPUT_DIR / "training_corpus_3.csv", index=False, sep=";", encoding=ENCODING)
    test_df[cols_out].to_csv(OUTPUT_DIR / "test_set_3.csv", index=False, sep=";", encoding=ENCODING)

    print("📊 DATA PELATIHAN (TRAINING SET):")
    print(train_df["label"].value_counts().to_string())
    print("\n📊 DATA PENGUJIAN (TEST SET):")
    for cls in TARGET_CLASSES:
        c = test_df["label"].value_counts().get(cls, 0)
        print(f"  {cls:25} : {c}{'<5' if c<5 else ''}")
    print(f"\n Data berhasil disimpan ke {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
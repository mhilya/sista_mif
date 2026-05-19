#!/usr/bin/env python3

import pandas as pd
import numpy as np
import re
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR
OUTPUT_DIR = DATA_DIR / "processed"
OUTPUT_DIR.mkdir(exist_ok=True)

FILE_INTERNAL = DATA_DIR / "ts_internal_mif.xlsx"
FILE_KEMENDIK = DATA_DIR / "ts_kemendiktisaintek_mif.xlsx"
FILE_JOBSTREET = DATA_DIR / "external" / "jobs_crawling" / "mergeFile.csv"

SEPARATOR = ";"
ENCODING = "utf-8-sig"

TARGET_CLASSES = ["Programmer", "Data Analyst", "Wirausaha Informatika", "Non-IT"]

KLASIFIKASI_MAP = {
    "Programmer"  : "Programmer",
    "Data Analyst": "Data Analyst",
    "Wirausaha"   : "Wirausaha Informatika",
    "Wirausaha IT": "Wirausaha Informatika",
    "Non IT"      : "Non-IT",
}

KEYWORD_RULES = {
    "Programmer": [
        "programmer", "developer", "engineer", "fullstack", "backend", "frontend",
        "mobile dev", "android", "ios", "software", "web dev", "coding", "it staff",
        "teknisi", "sistem informasi", "application", "network engineer", "devops",
        "qa", "tester", "ui/ux", "web developer", "mobile developer", "swe",
        "software engineer", "full stack", "back end", "front end", "app developer"
    ],
    "Data Analyst": [
        "data analyst", "analis data", "data science", "data scientist", "business analyst",
        "research", "statistik", "bi analyst", "reporting", "database", "sql", "etl",
        "data engineer", "big data", "analyst", "data mining", "machine learning", 
        "data visual", "data visualization", "power bi", "tableau", "looker",
        "business intelligence", "bi developer", "data warehouse"
    ],
    "Wirausaha Informatika": [
        "founder", "owner", "ceo", "wiraswasta", "startup", "freelance", "freelancer",
        "wirausaha", "bisnis", "usaha mandiri", "konsultan independen", "co-founder",
        "entrepreneur", "self-employed", "owner toko", "usaha", "dagang online",
        "tokopedia", "shopee seller", "dropship", "reseller", "affiliate"
    ]
}

F5C_MAP = {
    "1": "founder owner wirausaha startup",
    "2": "co-founder partner wirausaha", 
    "3": "staff karyawan pegawai",
    "4": "freelance kerja lepas lepasan"
}

F1101_MAP = {
    "1": "instansi pemerintah dinas kementerian",
    "2": "non-profit lsm yayasan",
    "3": "perusahaan swasta corporate",
    "4": "wiraswasta usaha mandiri", 
    "6": "bumn bumd pemerintah",
    "7": "multilateral internasional"
}

IT_KEYWORDS = [
    'programmer', 'developer', 'engineer', 'software', 'web', 'mobile', 'android', 'ios',
    'data analyst', 'data science', 'business intelligence', 'bi developer', 'bi developer',
    'etl', 'data warehouse', 'founder', 'startup', 'freelance', 'it staff', 'it support',
    'system analyst', 'network', 'cloud', 'devops', 'qa', 'tester', 'ui', 'ux',
    'fullstack', 'backend', 'frontend', 'application', 'database', 'sql', 'python',
    'java', 'javascript', 'php', 'laravel', 'react', 'vue', 'angular', 'flutter'
]

def load_data_file(file_path: str) -> pd.DataFrame:
    ext = Path(file_path).suffix.lower()
    if ext == '.csv':
        return pd.read_csv(
            file_path, 
            sep=SEPARATOR, 
            encoding=ENCODING, 
            dtype=str, 
            on_bad_lines='skip',
            engine='python'
        )
    elif ext in ['.xlsx', '.xls']:
        return pd.read_excel(
            file_path, 
            engine='openpyxl', 
            dtype=str, 
            header=0
        )
    else:
        raise ValueError(f"Format tidak didukung: {ext}")

def find_col(df: pd.DataFrame, keywords: list) -> str | None:
    for col in df.columns:
        col_clean = str(col).strip().lower()
        if any(k.strip().lower() in col_clean for k in keywords):
            return col
    return None

def clean_text(text) -> str:
    if pd.isna(text) or str(text).strip().lower() in ["nan", "none", "null", "-", "0", "", "tidak diisi", "tidak diketahui"]:
        return ""
    return re.sub(r'\s+', ' ', str(text).strip().lower())

def extract_code(value: str) -> str:
    val = str(value).strip()
    match = re.match(r'^(\d+)', val)
    return match.group(1) if match else val

def map_kemendik_to_text(row: pd.Series, col_f5b: str, col_f5c: str, col_f1101: str, col_f1102: str) -> str:
    company = clean_text(row.get(col_f5b, ""))
    f1102 = clean_text(row.get(col_f1102, ""))
    
    f5c_code = extract_code(row.get(col_f5c, ""))
    f5c_text = F5C_MAP.get(f5c_code, "")
    
    f1101_code = extract_code(row.get(col_f1101, ""))
    f1101_text = F1101_MAP.get(f1101_code, "")
    
    parts = [company, f5c_text, f1101_text, f1102]
    return " ".join([p for p in parts if p]).strip()

def classify_rule_based(text: str) -> str:
    if not text or len(text) < 3:
        return "Non-IT"
    text_lower = text.lower()
    
    for profile in ["Programmer", "Data Analyst", "Wirausaha Informatika"]:
        if any(kw in text_lower for kw in KEYWORD_RULES[profile]):
            return profile
    
    return "Non-IT"

def process_internal_mif(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        print(f"  File not found: {file_path}")
        return pd.DataFrame()
    
    print("[1/4] Processing Internal MIF...")
    df = load_data_file(str(file_path))
    df.columns = df.columns.str.strip()
    
    col_nim = find_col(df, ["nim"])
    col_nama = find_col(df, ["nama", "lengkap"])
    col_tahun = find_col(df, ["tahun", "lulus"])
    col_status = find_col(df, ["status", "kerja"])
    col_jabatan = find_col(df, ["jabatan", "posisi"])
    col_perusahaan = find_col(df, ["nama", "perusahaan"])
    col_deskripsi = find_col(df, ["deskripsi", "perusahaan"])
    col_klasifikasi = find_col(df, ["klasifikasi"])
    
    if not all([col_nim, col_jabatan]):
        print("  Missing critical columns (nim/jabatan) in Internal MIF")
        print("  Headers found:", df.columns.tolist()[:10], "...")
        return pd.DataFrame()
    
    if col_status:
        mask_work = df[col_status].str.lower().str.contains("bekerja|kerja|wir|wira", na=False, regex=True)
        df = df[mask_work].copy()
    
    df["job_text_raw"] = df[col_jabatan].apply(clean_text)
    if col_perusahaan and col_deskripsi:
        df.loc[df["job_text_raw"] == "", "job_text_raw"] = (
            df[col_perusahaan].apply(clean_text) + " " + df[col_deskripsi].apply(clean_text)
        ).str.strip()
    
    df["source"] = "internal_mif"
    
    cols = [col_nim, col_nama, col_tahun, "job_text_raw", "source"]
    rename_map = {col_nim: "nim", col_nama: "nama", col_tahun: "tahun_lulus"}

    if col_klasifikasi:
        cols.append(col_klasifikasi)
        rename_map[col_klasifikasi] = "klasifikasi_raw"

    result = df[cols].rename(columns=rename_map).dropna(subset=["nim"]).drop_duplicates(subset=["nim"], keep="first")

    if "klasifikasi_raw" in result.columns:
        result["label_verified"] = result["klasifikasi_raw"].map(KLASIFIKASI_MAP)
        result = result.drop(columns=["klasifikasi_raw"])
        verified_count = result["label_verified"].notna().sum()
        print(f"  {verified_count} records dengan label terverifikasi → akan dijadikan test set")

    print(f"  ✅ Loaded {len(result)} records from Internal MIF")
    return result

def process_kemendik(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        print(f"  File not found: {file_path}")
        return pd.DataFrame()
    
    print("[2/4] Processing Kemendiktisaintek...")
    df = load_data_file(str(file_path))
    df.columns = df.columns.str.strip()
    
    col_f8 = find_col(df, ["f8", "status"])
    col_f5b = find_col(df, ["f5b", "nama perusahaan"])
    col_f5c = find_col(df, ["f5c", "posisi wira"])
    col_f1101 = find_col(df, ["f1101", "jenis instansi"])
    col_f1102 = find_col(df, ["f1102", "lainnya"])
    col_nim = find_col(df, ["nim", "mahasiswa"])
    col_nama = find_col(df, ["nama"])
    col_tahun = find_col(df, ["tahun", "lulus"])
    
    if not all([col_f8, col_nim, col_f5b]):
        print("  Missing critical columns in Kemendik")
        return pd.DataFrame()
    
    f8_code = df[col_f8].str.extract(r'^(\d+)', expand=False).str.strip()
    df = df[f8_code.isin(["1", "3"])].copy()
    
    df["job_text_raw"] = df.apply(
        lambda r: map_kemendik_to_text(r, col_f5b, col_f5c, col_f1101, col_f1102), 
        axis=1
    )
    df["source"] = "kemendik"
    
    result = df[[col_nim, col_nama, col_tahun, "job_text_raw", "source"]].rename(columns={
        col_nim: "nim", col_nama: "nama", col_tahun: "tahun_lulus"
    }).dropna(subset=["nim"]).drop_duplicates(subset=["nim"], keep="first")
    
    print(f"  Loaded {len(result)} records from Kemendik")
    return result

def process_jobstreet_kaggle(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        print(f"  Jobstreet dataset not found: {file_path}")
        return pd.DataFrame()
    
    print("[3/4] Processing Jobstreet Kaggle Dataset...")
    
    chunks = []
    for chunk in pd.read_csv(str(file_path), encoding='utf-8-sig', dtype=str, chunksize=50000, on_bad_lines='skip'):
        text_col = 'jobTitle' if 'jobTitle' in chunk.columns else (find_col(chunk, ['job', 'title']) or 'jobTitle')
        desc_col = 'description' if 'description' in chunk.columns else None
        
        if text_col not in chunk.columns:
            continue
            
        search_text = chunk[text_col].fillna('')
        if desc_col and desc_col in chunk.columns:
            search_text = search_text + ' ' + chunk[desc_col].fillna('')
        
        it_pattern = '|'.join(IT_KEYWORDS)
        chunk_filtered = chunk[search_text.str.contains(it_pattern, case=False, na=False, regex=True)].copy()
        
        if len(chunk_filtered) > 0:
            chunks.append(chunk_filtered)
    
    if not chunks:
        print("  No IT jobs found after filtering.")
        return pd.DataFrame()
    
    df = pd.concat(chunks, ignore_index=True)
    print(f"  Filtered to {len(df)} IT jobs from {len(chunks)*50000}+ raw rows")
    
    text_col = 'jobTitle' if 'jobTitle' in df.columns else (find_col(df, ['job', 'title']) or 'jobTitle')
    desc_col = 'description' if 'description' in df.columns else None
    
    df['job_text_raw'] = df[text_col].apply(clean_text)
    if desc_col and desc_col in df.columns:
        df['job_text_raw'] = df['job_text_raw'] + ' ' + df[desc_col].apply(clean_text)
        df['job_text_raw'] = df['job_text_raw'].apply(lambda x: re.sub(r'\s+', ' ', x).strip())
    
    df['label'] = df['job_text_raw'].apply(classify_rule_based)
    df['source'] = 'jobstreet_kaggle'
    df['nim'] = 'JOB-' + df.index.astype(str)
    df['nama'] = 'Jobstreet Sample'
    df['tahun_lulus'] = ''
    
    df = df.drop_duplicates(subset=['job_text_raw'], keep='first')
    
    result = df[['nim', 'nama', 'tahun_lulus', 'job_text_raw', 'label', 'source']]
    print(f"  Loaded {len(result)} IT jobs from Jobstreet")
    return result

def main():
    print("Starting data preparation pipeline...")
    print(f"Target classes: {TARGET_CLASSES}\n")

    df_int = process_internal_mif(FILE_INTERNAL)
    df_kem = process_kemendik(FILE_KEMENDIK)
    df_job = process_jobstreet_kaggle(FILE_JOBSTREET)

    # Export test set dari internal_mif SEBELUM masuk training
    if not df_int.empty and "label_verified" in df_int.columns:
        df_test = df_int[df_int["label_verified"].notna()].copy()
        df_test = df_test.rename(columns={"label_verified": "label"})
        test_output = OUTPUT_DIR / "test_set_internal.csv"
        cols_test = ["nim", "nama", "tahun_lulus", "job_text_raw", "label", "source"]
        df_test[cols_test].to_csv(test_output, index=False, sep=";", encoding=ENCODING)
        dist_test = df_test["label"].value_counts()
        print(f"\n📋 TEST SET TERSIMPAN: {len(df_test)} records → test_set_internal.csv")
        for cls in TARGET_CLASSES:
            print(f"  {cls:25} : {dist_test.get(cls, 0)}")

    # Training hanya dari kemendik + jobstreet (internal_mif TIDAK masuk training)
    dfs = [df for df in [df_kem, df_job] if not df.empty]
    if not dfs:
        print("No data loaded. Check file paths and column names.")
        sys.exit(1)

    print(f"\n[4/4] Merging & processing {len(dfs)} sources (training only)...")
    df_combined = pd.concat(dfs, ignore_index=True)

    df_combined = df_combined.sort_values('source', key=lambda x: x.map({'internal_mif': 0, 'kemendik': 1, 'jobstreet_kaggle': 2}))
    df_combined = df_combined.drop_duplicates(subset=["nim"], keep="first")

    df_combined = df_combined[df_combined["job_text_raw"].str.len() >= 3]
    df_combined = df_combined[~df_combined["job_text_raw"].str.match(r'^[\d\-\s\./]+$')]
    df_combined["job_text_raw"] = df_combined["job_text_raw"].str.replace(r'^\d+\s*-\s*', '', regex=True).str.strip()

    if 'label' not in df_combined.columns:
        print("  Label column missing, applying classification now.")
        df_combined["label"] = df_combined["job_text_raw"].apply(classify_rule_based)
    else:
        nan_mask = df_combined['label'].isna()
        if nan_mask.any():
            print(f"  Filling {nan_mask.sum()} missing labels using rule-based classification.")
            df_combined.loc[nan_mask, "label"] = df_combined.loc[nan_mask, "job_text_raw"].apply(classify_rule_based)

    print("  Applying stratified downsampling to balance classes...")
    target_classes = TARGET_CLASSES
    max_per_class = 5000
    balanced_dfs = []
    for cls in target_classes:
        class_df = df_combined[df_combined['label'] == cls]
        if len(class_df) > max_per_class:
            class_df = class_df.sample(n=max_per_class, random_state=42)
        balanced_dfs.append(class_df)
    df_combined = pd.concat(balanced_dfs, ignore_index=True)

    print("\nFINAL CLASS DISTRIBUTION:")
    dist = df_combined["label"].value_counts()
    for cls in TARGET_CLASSES:
        count = dist.get(cls, 0)
        pct = count / len(df_combined) * 100
        print(f"  {cls:25} : {count:5} ({pct:5.1f}%)")

    print(f"\nTOTAL TRAINING RECORDS: {len(df_combined)}")

    if len(dist) > 1:
        ratio = dist.max() / dist.min()
        if ratio > 5:
            print(f"  WARNING: Class imbalance ({ratio:.1f}:1).")

    output_file = OUTPUT_DIR / "training_corpus.csv"
    cols_out = ["nim", "nama", "tahun_lulus", "job_text_raw", "label", "source"]
    df_combined[cols_out].to_csv(output_file, index=False, sep=";", encoding=ENCODING)

    print(f"\nSaved to: {output_file}")
    print("Phase 1 completed.")

    print("\nSAMPLE PREVIEWS:")
    for cls in TARGET_CLASSES:
        samples = df_combined[df_combined['label'] == cls].head(3)
        if not samples.empty:
            print(f"\n  [{cls}]")
            for _, row in samples.iterrows():
                text = row['job_text_raw'][:80] + "..." if len(row['job_text_raw']) > 80 else row['job_text_raw']
                print(f"    - {text}")

if __name__ == "__main__":
    main()
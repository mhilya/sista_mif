import pandas as pd
import numpy as np
import re
import joblib
import json
import warnings
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).parent
TRAIN_FILE = BASE_DIR / "processed" / "training_corpus_3.csv"
TEST_FILE = BASE_DIR / "processed" / "test_set_3.csv"
ML_DIR = BASE_DIR.parent / "fastapi" / "ml_assets"
ML_DIR.mkdir(parents=True, exist_ok=True)

STOPWORDS = {
    "yang", "di", "ke", "dari", "dan", "atau", "dengan", "untuk", "pada", "dalam", 
    "adalah", "ini", "itu", "tidak", "juga", "sudah", "akan", "bisa", "ada", "oleh", 
    "karena", "secara", "serta", "sebagai", "bagi", "telah", "maka", "namun", "sehingga", 
    "jika", "agar", "ketika", "saat", "sebelum", "sesudah", "hingga", "sampai", "antara", 
    "sekitar", "hanya", "saja", "belum", "masih", "lagi", "pun", "justru", "walaupun", 
    "meskipun", "bahkan", "cukup", "sangat", "paling", "lebih", "kurang", "lain", 
    "macam", "cara", "hal", "tentang", "mengenai", "terhadap", "kepada", "menuju", 
    "kecuali", "selain", "tanpa", "demi", "guna", "khususnya", "umumnya", "kebanyakan", 
    "sebagian", "beberapa", "semua", "setiap", "tiap", "satu", "dua", "tiga", "empat", 
    "lima", "enam", "tujuh", "delapan", "sembilan", "sepuluh", "ratus", "ribu", "juta"
}

stemmer = StemmerFactory().create_stemmer()
TARGET_CLASSES = ["Programmer", "Data Analyst", "Wirausaha Informatika", "Non-IT"]
CONFIDENCE_THRESHOLD = 0.75

def preprocess_text(text):
    if pd.isna(text) or not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    words = [w for w in text.split() if w not in STOPWORDS and len(w) > 2]
    return " ".join([stemmer.stem(w) for w in words])

def main():
    print("Fase 2: Pelatihan model khusus data internal MIF\n")

    if not TRAIN_FILE.exists():
        print("Berkas training_corpus_3.csv tidak ditemukan. Harap jalankan prepare_corpus_3.py terlebih dahulu.")
        return
    df_train = pd.read_csv(TRAIN_FILE, sep=";", dtype=str).dropna(subset=["job_text_raw", "label"])
    
    df_test = None
    if TEST_FILE.exists():
        df_test = pd.read_csv(TEST_FILE, sep=";", dtype=str).dropna(subset=["job_text_raw", "label"])
        print(f"Berhasil memuat data pengujian: {len(df_test)} baris\n")
    else:
        print("Berkas test_set_3.csv tidak ditemukan. Evaluasi hanya akan menggunakan metode K-Fold.\n")

    print("Sedang memproses teks (stemming dengan Sastrawi dan penghapusan stopwords)...")
    X_train = df_train["job_text_raw"].apply(preprocess_text)
    y_train = df_train["label"]
    mask = X_train.str.len() > 0
    X_train, y_train = X_train[mask], y_train[mask]
    print(f"  Data pelatihan: {len(X_train)} baris teks bersih\n")

    X_test, y_test = pd.Series(dtype=str), pd.Series(dtype=str)
    if df_test is not None:
        X_test = df_test["job_text_raw"].apply(preprocess_text)
        y_test = df_test["label"]
        mask_test = X_test.str.len() > 0
        X_test, y_test = X_test[mask_test], y_test[mask_test]
        print(f"  Data pengujian: {len(X_test)} baris teks bersih\n")

    print("Memulai evaluasi Stratified K-Fold (k=5) pada data pelatihan...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_metrics = []

    for i, (train_idx, test_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr, X_te = X_train.iloc[train_idx], X_train.iloc[test_idx]
        y_tr, y_te = y_train.iloc[train_idx], y_train.iloc[test_idx]

        model = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=3000, ngram_range=(1,2), sublinear_tf=True, min_df=1)),
            ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', solver='lbfgs'))
        ])

        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        report = classification_report(y_te, y_pred, output_dict=True, zero_division=0)
        fold_metrics.append(report)
        print(f"  Akurasi Fold {i+1}/5: {report['accuracy']:.4f}")

    print("\nMETRIK K-FOLD (Data Pelatihan - Khusus Internal MIF):")
    avg_acc = np.mean([m['accuracy'] for m in fold_metrics])
    print(f"  Akurasi  : {avg_acc:.4f} ± {np.std([m['accuracy'] for m in fold_metrics]):.4f}")
    for cls in TARGET_CLASSES:
        avg_p = np.mean([m.get(cls, {}).get('precision', 0) for m in fold_metrics])
        avg_r = np.mean([m.get(cls, {}).get('recall', 0) for m in fold_metrics])
        avg_f1 = np.mean([m.get(cls, {}).get('f1-score', 0) for m in fold_metrics])
        print(f"  {cls:25} | P: {avg_p:.3f} | R: {avg_r:.3f} | F1: {avg_f1:.3f}")

    print("\nMelatih model final menggunakan seluruh data pelatihan internal MIF...")
    final_model = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=3000, ngram_range=(1,2), sublinear_tf=True, min_df=1)),
        ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', solver='lbfgs'))
    ])
    final_model.fit(X_train, y_train)

    if df_test is not None and len(X_test) > 0:
        print("\nMETRIK TEST HOLD-OUT (Internal MIF - rasio 30%, label terverifikasi):")
        y_pred_test = final_model.predict(X_test)
        report_test = classification_report(y_test, y_pred_test, output_dict=True, zero_division=0)
        print(classification_report(y_test, y_pred_test, zero_division=0))
        
        for cls in TARGET_CLASSES:
            support = report_test[cls]['support'] if cls in report_test else 0
            if support < 10:
                print(f"PERINGATAN: Kelas '{cls}' hanya memiliki {support} sampel pengujian. Nilai F1-score mungkin kurang stabil.")

        cm = confusion_matrix(y_test, y_pred_test, labels=TARGET_CLASSES)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=TARGET_CLASSES)
        disp.plot(cmap='Blues', values_format='d', xticks_rotation=45, colorbar=False)
        plt.title('Confusion Matrix - Hold-Out Test (Internal MIF)')
        plt.tight_layout()
        plt.savefig(ML_DIR / 'confusion_matrix_test.png', dpi=300)
        plt.close()
        print(f" Tersimpan: confusion_matrix_test.png")

        vectorizer = final_model.named_steps['tfidf']
        clf = final_model.named_steps['clf']
        feature_names = vectorizer.get_feature_names_out()
        coeffs = clf.coef_
        
        print("\n TOP 5 FITUR TF-IDF PER KELAS:")
        for i, cls in enumerate(TARGET_CLASSES):
            top_idx = coeffs[i].argsort()[-5:][::-1]
            top_features = [(feature_names[j], coeffs[i][j]) for j in top_idx if coeffs[i][j] > 0]
            print(f"\n  [{cls}]")
            for feat, weight in top_features:
                print(f"    • {feat:20} (weight: {weight:.3f})")

        y_proba = final_model.predict_proba(X_test)
        max_proba = np.max(y_proba, axis=1)
        
        plt.figure(figsize=(8, 5))
        plt.hist(max_proba, bins=15, edgecolor='black', alpha=0.7, color='teal')
        plt.axvline(x=CONFIDENCE_THRESHOLD, color='red', linestyle='--', linewidth=2, label=f'Threshold {CONFIDENCE_THRESHOLD}')
        plt.xlabel('Skor Kepercayaan Prediksi')
        plt.ylabel('Frekuensi')
        plt.title('Distribusi Skor Kepercayaan - Hold-Out Test')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(ML_DIR / 'confidence_distribution.png', dpi=300)
        plt.close()
        below_thresh = np.sum(max_proba < CONFIDENCE_THRESHOLD)
        print(f"\n Analisis Threshold: {below_thresh}/{len(max_proba)} ({below_thresh/len(max_proba)*100:.1f}%) prediksi di bawah {CONFIDENCE_THRESHOLD} → akan dialihkan untuk pemeriksaan manual")

        print("\n🔍 ANALISIS SAMPEL TIDAK SESUAI (Misclassified):")
        mis_idx = np.where(y_test != y_pred_test)[0]
        if len(mis_idx) > 0:
            for idx in mis_idx[:min(8, len(mis_idx))]:
                text = df_test.iloc[idx]['job_text_raw'][:90]
                print(f"  • Asli: {y_test.iloc[idx]:20} | Prediksi: {y_pred_test[idx]:20} | Teks: '{text}...'")
        else:
            print("  Tidak ditemukan kesalahan prediksi pada data pengujian.")
    else:
        print("\nData pengujian hold-out tidak tersedia. Metrik di atas merupakan estimasi berdasarkan evaluasi K-Fold.")
        report_test = None

    model_path = ML_DIR / "ml_pipeline_internal.pkl"
    joblib.dump(final_model, model_path)

    metrics = {
        "methodology": "Internal MIF only (no external augmentation)",
        "k_fold": {
            "accuracy_mean": float(avg_acc),
            "accuracy_std": float(np.std([m['accuracy'] for m in fold_metrics])),
            "folds": fold_metrics
        },
        "threshold_config": CONFIDENCE_THRESHOLD
    }
    if report_test:
        metrics["hold_out_test"] = report_test

    metrics_path = ML_DIR / "metrics_internal_only.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\nPipeline model telah disimpan pada: {model_path}")
    print(f"Metrik performa telah disimpan pada: {metrics_path}")

if __name__ == "__main__":
    main()
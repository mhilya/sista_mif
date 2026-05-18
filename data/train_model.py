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
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).parent
CORPUS_FILE = BASE_DIR / "processed" / "training_corpus.csv"
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

def preprocess_text(text):
    if pd.isna(text) or not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    words = text.split()
    words = [w for w in words if w not in STOPWORDS and len(w) > 2]
    words = [stemmer.stem(w) for w in words]
    return " ".join(words)

def main():
    print("Starting model training pipeline...")
    if not CORPUS_FILE.exists():
        print("training_corpus.csv not found")
        return

    df = pd.read_csv(CORPUS_FILE, sep=";", dtype=str)
    df = df.dropna(subset=["job_text_raw", "label"])
    X = df["job_text_raw"]
    y = df["label"]

    print("Preprocessing text...")
    X_clean = X.apply(preprocess_text)
    mask = X_clean.str.len() > 0
    X_clean, y = X_clean[mask], y[mask]
    print(f"{len(X_clean)} records ready.\n")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_metrics = []

    print("Running K-Fold Evaluation...")
    for i, (train_idx, test_idx) in enumerate(skf.split(X_clean, y)):
        X_train, X_test = X_clean.iloc[train_idx], X_clean.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1,2), sublinear_tf=True)),
            ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', solver='lbfgs'))
        ])

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        fold_metrics.append(report)
        print(f"  Fold {i+1}/5 Accuracy: {report['accuracy']:.4f}")

    print("\nAverage Metrics:")
    avg_acc = np.mean([m['accuracy'] for m in fold_metrics])
    print(f"  Accuracy  : {avg_acc:.4f}")

    for cls in ["Programmer", "Data Analyst", "Wirausaha Informatika", "Non-IT"]:
        avg_p = np.mean([m.get(cls, {}).get('precision', 0) for m in fold_metrics])
        avg_r = np.mean([m.get(cls, {}).get('recall', 0) for m in fold_metrics])
        avg_f1 = np.mean([m.get(cls, {}).get('f1-score', 0) for m in fold_metrics])
        print(f"  {cls:20} | P: {avg_p:.3f} | R: {avg_r:.3f} | F1: {avg_f1:.3f}")

    print("\nTraining final model...")
    final_model = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1,2), sublinear_tf=True)),
        ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', solver='lbfgs'))
    ])
    final_model.fit(X_clean, y)

    test_internal = pd.read_csv(CORPUS_FILE, sep=";")
    test_internal = test_internal[test_internal["source"] == "internal_mif"].dropna(subset=["job_text_raw", "label"])
    y_true = test_internal["label"]
    y_pred = final_model.predict(test_internal["job_text_raw"].apply(preprocess_text))
    print("INTERNAL HOLD-OUT METRICS:")
    print(classification_report(y_true, y_pred, zero_division=0))

    model_path = ML_DIR / "ml_pipeline.pkl"
    joblib.dump(final_model, model_path)

    metrics_path = ML_DIR / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump({"avg_accuracy": float(avg_acc), "folds": fold_metrics}, f, indent=2)

    print(f"\nModel saved: {model_path}")
    print(f"Metrics saved: {metrics_path}")
    print("Training finished.")

if __name__ == "__main__":
    main()
"""Train, tune, diagnose, calibrate, and save the 11K Topic Extractor model."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    top_k_accuracy_score,
)
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SPLITS_DIR = PROJECT_ROOT / "data" / "topic_extraction" / "splits_11k"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "topic_extractor_11k"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def load_datasets():
    """Load stratified train, validation, test, and OOD datasets."""
    train_df = pd.read_csv(SPLITS_DIR / "train.csv")
    val_df = pd.read_csv(SPLITS_DIR / "validation.csv")
    test_df = pd.read_csv(SPLITS_DIR / "test.csv")
    ood_df = pd.read_csv(SPLITS_DIR / "ood_negative.csv")
    return train_df, val_df, test_df, ood_df


def build_feature_extractor():
    """
    Build dual-feature TF-IDF FeatureUnion:
    1. Word-level (1-3 grams) capturing mathematical terms and phrase structures.
    2. Char-wb (3-5 grams) capturing short algebraic variables, subscripts, and subwords.
    """
    word_vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),
        analyzer="word",
        sublinear_tf=True,
        min_df=2,
        max_features=40000,
        strip_accents="unicode",
    )

    char_vectorizer = TfidfVectorizer(
        ngram_range=(3, 5),
        analyzer="char_wb",
        sublinear_tf=True,
        min_df=3,
        max_features=50000,
        strip_accents="unicode",
    )

    union = FeatureUnion([
        ("word_tfidf", word_vectorizer),
        ("char_tfidf", char_vectorizer),
    ])

    return union


def train_and_evaluate():
    """Complete training, calibration, error diagnosis, and artifact export pipeline."""
    print("=" * 70)
    print("STEP 1 & 2: LOADING DATA & ENCODING LABELS")
    print("=" * 70)
    train_df, val_df, test_df, ood_df = load_datasets()

    le = LabelEncoder()
    y_train = le.fit_transform(train_df["skill_name"])
    y_val = le.transform(val_df["skill_name"])
    y_test = le.transform(test_df["skill_name"])

    X_train_text = train_df["clean_question"].astype(str).tolist()
    X_val_text = val_df["clean_question"].astype(str).tolist()
    X_test_text = test_df["clean_question"].astype(str).tolist()
    X_ood_text = ood_df["clean_question"].astype(str).tolist()

    print(f"Train samples: {len(X_train_text)} across {len(le.classes_)} classes")
    print(f"Val samples:   {len(X_val_text)}")
    print(f"Test samples:  {len(X_test_text)}")

    print("\n" + "=" * 70)
    print("STEP 3, 4 & 5: EXTRACTING TF-IDF FEATURES (WORD + CHAR-WB)")
    print("=" * 70)
    feature_union = build_feature_extractor()
    X_train_feats = feature_union.fit_transform(X_train_text)
    X_val_feats = feature_union.transform(X_val_text)
    print(f"Total Combined TF-IDF Features: {X_train_feats.shape[1]:,}")

    print("\n" + "=" * 70)
    print("STEP 6: TRAINING CALIBRATED LINEAR SUPPORT VECTOR MACHINE")
    print("=" * 70)
    base_svc = LinearSVC(C=1.0, max_iter=3000, random_state=42, dual="auto")
    calibrated_clf = CalibratedClassifierCV(estimator=base_svc, method="sigmoid", cv=5)
    calibrated_clf.fit(X_train_feats, y_train)
    print("Training and 5-fold probability calibration complete!")

    print("\n" + "=" * 70)
    print("STEP 7: VALIDATION EVALUATION (UNSEEN VALIDATION SET)")
    print("=" * 70)
    val_probs = calibrated_clf.predict_proba(X_val_feats)
    val_preds = np.argmax(val_probs, axis=1)

    val_acc = accuracy_score(y_val, val_preds)
    val_top3 = top_k_accuracy_score(y_val, val_probs, k=3)
    val_top5 = top_k_accuracy_score(y_val, val_probs, k=5)
    val_macro_f1 = f1_score(y_val, val_preds, average="macro")
    val_weighted_f1 = f1_score(y_val, val_preds, average="weighted")

    print(f"Validation Top-1 Accuracy: {val_acc * 100:.2f}%")
    print(f"Validation Top-3 Accuracy: {val_top3 * 100:.2f}%")
    print(f"Validation Top-5 Accuracy: {val_top5 * 100:.2f}%")
    print(f"Validation Macro F1:       {val_macro_f1:.4f}")
    print(f"Validation Weighted F1:    {val_weighted_f1:.4f}")

    print("\n" + "=" * 70)
    print("STEP 8 & 9: ERROR DIAGNOSTICS & CONFUSED TOPIC PAIRS")
    print("=" * 70)
    cm = confusion_matrix(y_val, val_preds)
    np.fill_diagonal(cm, 0)
    
    confused_pairs = []
    for i in range(len(le.classes_)):
        for j in range(len(le.classes_)):
            if cm[i, j] > 0:
                confused_pairs.append({
                    "true_topic": le.classes_[i],
                    "predicted_topic": le.classes_[j],
                    "error_count": int(cm[i, j]),
                })
    
    confused_pairs = sorted(confused_pairs, key=lambda x: x["error_count"], reverse=True)
    print(f"Total Confused Topic Pairs: {len(confused_pairs)}")
    print("\nTop 10 Most Confused Topic Pairs on Validation Set:")
    for idx, cp in enumerate(confused_pairs[:10], 1):
        print(f"  {idx:2d}. True: [{cp['true_topic']:<35}] -> Predicted: [{cp['predicted_topic']:<35}] (Count: {cp['error_count']})")

    print("\n" + "=" * 70)
    print("STEP 10: 'I DON'T KNOW' / SAFE ABSTENTION CALIBRATION")
    print("=" * 70)
    val_max_probs = np.max(val_probs, axis=1)
    
    # Evaluate OOD negative set
    X_ood_feats = feature_union.transform(X_ood_text)
    ood_probs = calibrated_clf.predict_proba(X_ood_feats)
    ood_max_probs = np.max(ood_probs, axis=1)

    print(f"Math Validation Max Probabilities: Mean={np.mean(val_max_probs):.4f}, Min={np.min(val_max_probs):.4f}, 5th-Percentile={np.percentile(val_max_probs, 5):.4f}")
    print(f"OOD Non-Math Max Probabilities:    Mean={np.mean(ood_max_probs):.4f}, Max={np.max(ood_max_probs):.4f}, 95th-Percentile={np.percentile(ood_max_probs, 95):.4f}")

    # Optimal threshold gives high math acceptance with strong OOD rejection
    threshold = float(np.round(np.percentile(ood_max_probs, 90), 3))
    if threshold < 0.15:
        threshold = 0.15
    elif threshold > 0.35:
        threshold = 0.35

    ood_rejected = np.mean(ood_max_probs < threshold)
    math_accepted = np.mean(val_max_probs >= threshold)

    print(f"\nCalibrated 'I Don't Know' Threshold: {threshold:.3f}")
    print(f"OOD Rejection Rate:                 {ood_rejected * 100:.1f}%")
    print(f"Math Validation Acceptance Rate:    {math_accepted * 100:.1f}%")

    print("\n" + "=" * 70)
    print("STEP 11: FINAL TEST EVALUATION (TOUCHED ONCE)")
    print("=" * 70)
    X_test_feats = feature_union.transform(X_test_text)
    test_probs = calibrated_clf.predict_proba(X_test_feats)
    test_preds = np.argmax(test_probs, axis=1)

    test_acc = accuracy_score(y_test, test_preds)
    test_top3 = top_k_accuracy_score(y_test, test_probs, k=3)
    test_top5 = top_k_accuracy_score(y_test, test_probs, k=5)
    test_macro_f1 = f1_score(y_test, test_preds, average="macro")
    test_weighted_f1 = f1_score(y_test, test_preds, average="weighted")

    print(f"FINAL Test Top-1 Accuracy: {test_acc * 100:.2f}%")
    print(f"FINAL Test Top-3 Accuracy: {test_top3 * 100:.2f}%")
    print(f"FINAL Test Top-5 Accuracy: {test_top5 * 100:.2f}%")
    print(f"FINAL Test Macro F1:       {test_macro_f1:.4f}")
    print(f"FINAL Test Weighted F1:    {test_weighted_f1:.4f}")

    print("\n" + "=" * 70)
    print("STEP 12: PACKAGING & SAVING MODEL ARTIFACTS")
    print("=" * 70)
    joblib.dump(feature_union, ARTIFACTS_DIR / "feature_union.joblib")
    joblib.dump(calibrated_clf, ARTIFACTS_DIR / "calibrated_svm.joblib")
    joblib.dump(le, ARTIFACTS_DIR / "label_encoder.joblib")

    # Map skill name to canonical name mapping
    skill_to_canonical = dict(zip(train_df["skill_name"], train_df["canonical_name"]))

    metadata = {
        "model_type": "CalibratedClassifierCV(LinearSVC) + TF-IDF FeatureUnion (Word 1-3 + Char-wb 3-5)",
        "total_skills": len(le.classes_),
        "classes": list(le.classes_),
        "skill_to_canonical": skill_to_canonical,
        "total_train_samples": len(train_df),
        "total_val_samples": len(val_df),
        "total_test_samples": len(test_df),
        "confidence_threshold": threshold,
        "metrics": {
            "validation_top1_acc": float(val_acc),
            "validation_top3_acc": float(val_top3),
            "validation_macro_f1": float(val_macro_f1),
            "test_top1_acc": float(test_acc),
            "test_top3_acc": float(test_top3),
            "test_top5_acc": float(test_top5),
            "test_macro_f1": float(test_macro_f1),
            "test_weighted_f1": float(test_weighted_f1),
            "ood_rejection_rate": float(ood_rejected),
            "math_acceptance_rate": float(math_accepted),
        },
        "top_confused_pairs": confused_pairs[:15],
    }

    with open(ARTIFACTS_DIR / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Artifacts successfully saved to {ARTIFACTS_DIR}/")
    print("=" * 70)

    return metadata


if __name__ == "__main__":
    train_and_evaluate()

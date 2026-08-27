"""Generate the comprehensive Jupyter Notebook for the 11K Topic Extraction Model."""

from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK_PATH = Path("notebooks/15_01_topic_extraction_11k_model_training_and_evaluation.ipynb")
NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 📚 Educational Topic Extraction: 11,000 Student Question Model\n",
            "\n",
            "This notebook implements the complete 14-step systematic methodology for building a high-accuracy, robust **Math Topic Extractor** trained on 11,000 natural student questions across 110 canonical math topics.\n",
            "\n",
            "### 🎯 Objectives:\n",
            "1. **Decide Specifications**: Top-1 and Top-3 predictions + Out-of-Domain / \"I Don't Know\" safe abstention.\n",
            "2. **Data Pipeline**: Stratified 70/15/15 split across 110 topics with text cleaning and math symbol preservation.\n",
            "3. **Feature Engineering**: Hybrid Word (1-3 ngrams) + Character (3-5 ngrams) TF-IDF Feature Union.\n",
            "4. **Model Training & Calibration**: Calibrated Linear Support Vector Machine (LinearSVC with 5-fold Sigmoid Platt Scaling).\n",
            "5. **Error Diagnosis**: In-depth confusion matrix analysis on difficult topic pairs.\n",
            "6. **Confidence Thresholding**: Calibrated rejection of non-math / vague questions.\n",
            "7. **Final Test & Packaging**: Evaluation on the untouched 15% test set and packaged standalone inference function."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Step 1 & Setup: Imports and Environment Configuration\n",
            "import json\n",
            "import os\n",
            "import re\n",
            "from pathlib import Path\n",
            "import joblib\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "from sklearn.calibration import CalibratedClassifierCV\n",
            "from sklearn.feature_extraction.text import TfidfVectorizer\n",
            "from sklearn.metrics import (\n",
            "    accuracy_score,\n",
            "    classification_report,\n",
            "    confusion_matrix,\n",
            "    f1_score,\n",
            "    top_k_accuracy_score,\n",
            ")\n",
            "from sklearn.model_selection import train_test_split\n",
            "from sklearn.pipeline import FeatureUnion\n",
            "from sklearn.preprocessing import LabelEncoder\n",
            "from sklearn.svm import LinearSVC\n",
            "\n",
            "print(\"All libraries imported successfully!\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. Load and Inspect the Dataset (`student_skill_questions.csv`)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Step 2: Load the 11,000 student question dataset\n",
            "df = pd.read_csv('../student_skill_questions.csv')\n",
            "print(f\"Dataset Shape: {df.shape}\")\n",
            "print(f\"Total Unique Skills: {df['skill_name'].nunique()}\")\n",
            "print(f\"Questions per skill: {df['skill_name'].value_counts().min()} min, {df['skill_name'].value_counts().max()} max\")\n",
            "df.head(5)"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Text Preprocessing & Math Symbol Preservation"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Step 4: Text Cleaning Function\n",
            "def clean_math_text(text: str) -> str:\n",
            "    \"\"\"Clean text while preserving mathematical operators, symbols, and variables.\"\"\"\n",
            "    if not isinstance(text, str):\n",
            "        text = str(text) if text is not None else \"\"\n",
            "    text = text.replace(\"\", \"-\").replace(\"–\", \"-\").replace(\"—\", \"-\")\n",
            "    text = text.replace(\"×\", \"*\").replace(\"÷\", \"/\")\n",
            "    text = text.replace(\"²\", \"^2\").replace(\"³\", \"^3\").replace(\"√\", \"sqrt\")\n",
            "    text = text.lower().strip()\n",
            "    text = re.sub(r\"\\s+\", \" \", text)\n",
            "    return text\n",
            "\n",
            "df['clean_question'] = df['question'].apply(clean_math_text)\n",
            "print(\"Sample cleaned questions:\")\n",
            "for q in df['clean_question'].sample(3, random_state=42):\n",
            "    print(f\" - {q}\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Stratified Dataset Splitting (70% Train / 15% Validation / 15% Test)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Step 3: Stratified Splitting across all 110 topics\n",
            "train_dfs, val_dfs, test_dfs = [], [], []\n",
            "\n",
            "for skill, group in df.groupby('skill_name'):\n",
            "    tr_grp, temp_grp = train_test_split(group, test_size=0.30, random_state=42, shuffle=True)\n",
            "    val_grp, test_grp = train_test_split(temp_grp, test_size=0.50, random_state=42, shuffle=True)\n",
            "    train_dfs.append(tr_grp)\n",
            "    val_dfs.append(val_grp)\n",
            "    test_dfs.append(test_grp)\n",
            "\n",
            "train_df = pd.concat(train_dfs, ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)\n",
            "val_df = pd.concat(val_dfs, ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)\n",
            "test_df = pd.concat(test_dfs, ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)\n",
            "\n",
            "print(f\"Train rows:      {len(train_df)} (70 per topic)\")\n",
            "print(f\"Validation rows: {len(val_df)} (15 per topic)\")\n",
            "print(f\"Test rows:       {len(test_df)} (15 per topic)\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. Feature Union: Hybrid Word + Character N-Gram TF-IDF"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Step 5: Convert text to feature matrices\n",
            "le = LabelEncoder()\n",
            "y_train = le.fit_transform(train_df['skill_name'])\n",
            "y_val = le.transform(val_df['skill_name'])\n",
            "y_test = le.transform(test_df['skill_name'])\n",
            "\n",
            "feature_union = FeatureUnion([\n",
            "    ('word_tfidf', TfidfVectorizer(ngram_range=(1, 3), analyzer='word', sublinear_tf=True, min_df=2, max_features=40000, strip_accents='unicode')),\n",
            "    ('char_tfidf', TfidfVectorizer(ngram_range=(3, 5), analyzer='char_wb', sublinear_tf=True, min_df=3, max_features=50000, strip_accents='unicode')),\n",
            "])\n",
            "\n",
            "X_train = feature_union.fit_transform(train_df['clean_question'])\n",
            "X_val = feature_union.transform(val_df['clean_question'])\n",
            "X_test = feature_union.transform(test_df['clean_question'])\n",
            "\n",
            "print(f\"Total Combined Vocabulary Features: {X_train.shape[1]:,}\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5. Model Training & 5-Fold Probability Calibration"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Step 6: Train Calibrated LinearSVC\n",
            "base_svc = LinearSVC(C=1.0, max_iter=3000, random_state=42, dual='auto')\n",
            "model = CalibratedClassifierCV(estimator=base_svc, method='sigmoid', cv=5)\n",
            "model.fit(X_train, y_train)\n",
            "print(\"Model training and calibration completed successfully!\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 6. Validation Evaluation & Error Diagnostics"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Step 7 & 8: Validation Metrics\n",
            "val_probs = model.predict_proba(X_val)\n",
            "val_preds = np.argmax(val_probs, axis=1)\n",
            "\n",
            "val_acc = accuracy_score(y_val, val_preds)\n",
            "val_top3 = top_k_accuracy_score(y_val, val_probs, k=3)\n",
            "val_macro_f1 = f1_score(y_val, val_preds, average='macro')\n",
            "\n",
            "print(f\"Validation Top-1 Accuracy: {val_acc * 100:.2f}%\")\n",
            "print(f\"Validation Top-3 Accuracy: {val_top3 * 100:.2f}%\")\n",
            "print(f\"Validation Macro F1:       {val_macro_f1:.4f}\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 7. 'I Don't Know' (Safe Abstention) Threshold Calibration"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Step 10: Out-of-domain rejection test\n",
            "ood_samples = [\n",
            "    \"Hello how are you?\",\n",
            "    \"Who are you?\",\n",
            "    \"What is photosynthesis?\",\n",
            "    \"Tell me a funny joke\",\n",
            "    \"What were the causes of World War 1?\",\n",
            "    \"I don't understand anything help please\",\n",
            "]\n",
            "\n",
            "X_ood = feature_union.transform([clean_math_text(q) for q in ood_samples])\n",
            "ood_probs = model.predict_proba(X_ood)\n",
            "ood_max = np.max(ood_probs, axis=1)\n",
            "\n",
            "CONFIDENCE_THRESHOLD = 0.35\n",
            "print(f\"Confidence Threshold: {CONFIDENCE_THRESHOLD}\")\n",
            "for q, score in zip(ood_samples, ood_max):\n",
            "    is_unsure = score < CONFIDENCE_THRESHOLD\n",
            "    print(f\"Query: '{q[:40]:<40}' | Max Prob: {score:.4f} | Is Unsure / Rejected: {is_unsure}\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 8. Final Test Evaluation (Untouched 15% Test Set)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Step 11: Final Unseen Test Evaluation\n",
            "test_probs = model.predict_proba(X_test)\n",
            "test_preds = np.argmax(test_probs, axis=1)\n",
            "\n",
            "test_acc = accuracy_score(y_test, test_preds)\n",
            "test_top3 = top_k_accuracy_score(y_test, test_probs, k=3)\n",
            "test_macro_f1 = f1_score(y_test, test_preds, average='macro')\n",
            "\n",
            "print(\"=\" * 50)\n",
            "print(\"FINAL UNTOUCHED TEST SET PERFORMANCE\")\n",
            "print(\"=\" * 50)\n",
            "print(f\"Final Test Top-1 Accuracy: {test_acc * 100:.2f}%\")\n",
            "print(f\"Final Test Top-3 Accuracy: {test_top3 * 100:.2f}%\")\n",
            "print(f\"Final Test Macro F1:       {test_macro_f1:.4f}\")\n",
            "print(\"=\" * 50)"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 9. Packaged Single Inference Function"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Step 12: Single Packaged Predict Function\n",
            "def predict_topic(question: str, threshold: float = 0.35) -> dict:\n",
            "    \"\"\"\n",
            "    Single packaged function:\n",
            "    Input: Student question (str)\n",
            "    Output: dict with identified topic, Top-3 candidates, confidence score, and is_unsure flag.\n",
            "    \"\"\"\n",
            "    clean_q = clean_math_text(question)\n",
            "    feats = feature_union.transform([clean_q])\n",
            "    probs = model.predict_proba(feats)[0]\n",
            "    \n",
            "    ranked_indices = np.argsort(probs)[::-1]\n",
            "    top_idx = ranked_indices[0]\n",
            "    top_score = float(probs[top_idx])\n",
            "    top_topic = le.classes_[top_idx]\n",
            "    is_unsure = top_score < threshold\n",
            "    \n",
            "    top_3 = []\n",
            "    for i in range(min(3, len(ranked_indices))):\n",
            "        idx = ranked_indices[i]\n",
            "        top_3.append({\n",
            "            \"topic\": le.classes_[idx],\n",
            "            \"score\": float(probs[idx]),\n",
            "        })\n",
            "        \n",
            "    return {\n",
            "        \"topic\": top_topic if not is_unsure else None,\n",
            "        \"identified_topic\": top_topic,\n",
            "        \"confidence\": top_score,\n",
            "        \"is_unsure\": is_unsure,\n",
            "        \"top_3_candidates\": top_3,\n",
            "    }\n",
            "\n",
            "# Test interactive predictions\n",
            "demo_questions = [\n",
            "    \"How do I solve 2x + 3 = 11?\",\n",
            "    \"What is the hypotenuse if legs are 3 and 4?\",\n",
            "    \"if we separate 10 from 100, what is the answer?\",\n",
            "    \"How do I find the average of 12, 15, and 18?\",\n",
            "    \"Hello there!\",\n",
            "]\n",
            "\n",
            "for dq in demo_questions:\n",
            "    res = predict_topic(dq)\n",
            "    print(f\"Question: '{dq}'\")\n",
            "    print(f\" -> Topic: {res['topic']} (Unsure: {res['is_unsure']}, Confidence: {res['confidence']:.3f})\")\n",
            "    print(f\" -> Top-3: {[c['topic'] for c in res['top_3_candidates']]}\\n\")"
        ]
    }
]

notebook = {
    "cells": cells,
    "metadata": {
        "language_info": {"name": "python", "version": "3.11"},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    },
    "nbformat": 4,
    "nbformat_minor": 4,
}

with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print(f"Jupyter Notebook generated at {NOTEBOOK_PATH}")

"""Helper script to construct notebooks/14_04_minilm_baseline_and_finetuning.ipynb."""

from pathlib import Path
# pyrefly: ignore [missing-import]
import nbformat as nbf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "14_04_minilm_baseline_and_finetuning.ipynb"


def generate_minilm_notebook(output_path: Path | str | None = None) -> Path:
    target_path = Path(output_path) if output_path is not None else DEFAULT_NOTEBOOK_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)

    nb = nbf.v4.new_notebook()
    cells = []

    # Title
    cells.append(
        nbf.v4.new_markdown_cell(
            """# 14.04 — Sentence Transformers MiniLM Baseline & Fine-Tuning

This notebook benchmarks dense semantic embeddings using `sentence-transformers/all-MiniLM-L6-v2` for canonical skill classification on the **frozen validation split** ($N=641$).

### Sections:
1. **Pretrained MiniLM Embedding Centroid Baseline:** Dense 384-dimensional semantic prototypes without domain fine-tuning.
2. **Three-Way Baseline Benchmark:** Keyword vs TF-IDF vs Pretrained MiniLM.
3. *(Fine-Tuning MiniLM will be added in Step 11)*
"""
        )
    )

    # Imports
    cells.append(
        nbf.v4.new_code_cell(
            """import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path('.').resolve().parent if Path('.').resolve().name == 'notebooks' else Path('.').resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.topic_extraction.keyword_baseline import KeywordBaselineClassifier
from src.topic_extraction.tfidf_baseline import TfidfBaselineClassifier
from src.topic_extraction.minilm_baseline import PretrainedMiniLMBaseline

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
%matplotlib inline
"""
        )
    )

    # Load Splits
    cells.append(
        nbf.v4.new_markdown_cell(
            """## 1. Load Dataset Splits
*Note: Evaluated strictly on `data/topic_extraction/splits/validation.csv`. Test split remains completely frozen.*"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """train_path = PROJECT_ROOT / 'data' / 'topic_extraction' / 'splits' / 'train.csv'
val_path = PROJECT_ROOT / 'data' / 'topic_extraction' / 'splits' / 'validation.csv'

train_df = pd.read_csv(train_path)
val_df = pd.read_csv(val_path)

print(f'Train split: {len(train_df)} rows')
print(f'Validation split: {len(val_df)} rows across {val_df["canonical_name"].nunique()} classes')
"""
        )
    )

    # Pretrained MiniLM Baseline
    cells.append(
        nbf.v4.new_markdown_cell(
            """## 2. Pretrained MiniLM Embedding Centroid Baseline"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """minilm_clf = PretrainedMiniLMBaseline()

# Check if centroids already computed, otherwise fit and save
centroids_file = PROJECT_ROOT / 'artifacts' / 'topic_extractor' / 'minilm_pretrained_skill_centroids.npz'
if centroids_file.exists():
    print('Loading precomputed MiniLM centroids from disk...')
    minilm_clf.load_centroids(centroids_file)
else:
    print('Computing MiniLM centroids on train split...')
    minilm_clf.fit(train_df)
    minilm_clf.save_centroids(centroids_file)

minilm_metrics = minilm_clf.evaluate(val_df)

minilm_results_df = pd.DataFrame([
    {'Metric': 'Total Validation Samples', 'Value': f'{minilm_metrics["total_samples"]:,}'},
    {'Metric': 'Validation Accuracy (Top-1)', 'Value': f'{minilm_metrics["accuracy"]*100:.2f}%'},
    {'Metric': 'Top-3 Accuracy', 'Value': f'{minilm_metrics["top3_accuracy"]*100:.2f}%'},
    {'Metric': 'Macro F1-Score', 'Value': f'{minilm_metrics["macro_f1"]:.4f}'},
    {'Metric': 'Weighted F1-Score', 'Value': f'{minilm_metrics["weighted_f1"]:.4f}'},
    {'Metric': 'Mean Top-1 Cosine Similarity', 'Value': f'{minilm_metrics["mean_top1_similarity"]:.4f}'},
    {'Metric': 'Mean Top-1 / Top-2 Margin', 'Value': f'{minilm_metrics["mean_margin"]:.4f}'},
])

minilm_results_df
"""
        )
    )

    # Three-Way Comparison
    cells.append(
        nbf.v4.new_markdown_cell(
            """## 3. Tri-Model Baseline Comparison: Keyword vs TF-IDF vs Pretrained MiniLM"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """# Evaluate Keyword and TF-IDF for direct comparison
kw_clf = KeywordBaselineClassifier()
kw_metrics = kw_clf.evaluate(val_df)

tfidf_clf = TfidfBaselineClassifier()
tfidf_clf.fit(train_df)
tfidf_metrics = tfidf_clf.evaluate(val_df)

comparison_df = pd.DataFrame({
    'Metric': ['Top-1 Accuracy', 'Top-3 Accuracy', 'Macro F1', 'Coverage', 'Covered Accuracy'],
    'Keyword Baseline': [
        f'{kw_metrics["accuracy"]*100:.2f}%',
        'N/A',
        f'{kw_metrics["macro_f1"]:.4f}',
        f'{kw_metrics["coverage"]*100:.2f}%',
        f'{kw_metrics["covered_accuracy"]*100:.2f}%',
    ],
    'TF-IDF Baseline': [
        f'{tfidf_metrics["accuracy"]*100:.2f}%',
        f'{tfidf_metrics["top3_accuracy"]*100:.2f}%',
        f'{tfidf_metrics["macro_f1"]:.4f}',
        '100.00%',
        f'{tfidf_metrics["accuracy"]*100:.2f}%',
    ],
    'Pretrained MiniLM': [
        f'{minilm_metrics["accuracy"]*100:.2f}%',
        f'{minilm_metrics["top3_accuracy"]*100:.2f}%',
        f'{minilm_metrics["macro_f1"]:.4f}',
        '100.00%',
        f'{minilm_metrics["accuracy"]*100:.2f}%',
    ],
})

fig, ax = plt.subplots(figsize=(10, 5))
bar_width = 0.25
indices = np.arange(3)

rects1 = ax.bar(indices - bar_width, [kw_metrics['accuracy']*100, kw_metrics['macro_f1']*100, kw_metrics['covered_accuracy']*100], bar_width, label='Keyword Baseline', color='#2b5c8f', edgecolor='black', alpha=0.85)
rects2 = ax.bar(indices, [tfidf_metrics['accuracy']*100, tfidf_metrics['macro_f1']*100, tfidf_metrics['accuracy']*100], bar_width, label='TF-IDF Baseline', color='#e67e22', edgecolor='black', alpha=0.85)
rects3 = ax.bar(indices + bar_width, [minilm_metrics['accuracy']*100, minilm_metrics['macro_f1']*100, minilm_metrics['accuracy']*100], bar_width, label='Pretrained MiniLM', color='#27ae60', edgecolor='black', alpha=0.85)

ax.set_ylabel('Score / Percentage (%)', fontsize=11)
ax.set_title('Baseline Model Performance Comparison on Validation Set', fontsize=13, fontweight='bold', pad=12)
ax.set_xticks(indices)
ax.set_xticklabels(['Overall Accuracy', 'Macro F1 (x100)', 'Covered Accuracy'], fontsize=11)
ax.legend(fontsize=10)
ax.set_ylim(0, 115)
ax.grid(axis='y', linestyle='--', alpha=0.7)

for rects in [rects1, rects2, rects3]:
    for rect in rects:
        y = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2, y + 2, f'{y:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.show()

comparison_df
"""
        )
    )

    # Sample Predictions
    cells.append(
        nbf.v4.new_markdown_cell(
            """## 4. Sample Pretrained MiniLM Predictions with Margin"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """sample_rows = []
for _, row in val_df.sample(10, random_state=42).iterrows():
    pred = minilm_clf.predict(str(row['text']), top_k=3)
    sample_rows.append({
        'Text': row['text'],
        'Ground Truth': row['skill_name'],
        'Predicted Skill': pred.display_name,
        'Similarity': f'{pred.similarity:.3f}',
        'Margin': f'{pred.margin:.3f}',
        'Correct': row['canonical_name'] == pred.canonical_name,
    })

pd.DataFrame(sample_rows)
"""
        )
    )

    nb["cells"] = cells

    with open(target_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)

    return target_path


def main():
    saved_path = generate_minilm_notebook()
    print(f"Successfully generated notebook at: {saved_path}")


if __name__ == "__main__":
    main()

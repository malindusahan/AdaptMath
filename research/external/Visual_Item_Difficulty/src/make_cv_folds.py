"""Generate the frozen 5-fold CV assignment on the 580 train items.

Stratified by difficulty quintile x has_figure.
Writes data/cv_folds.json with fold QuestionIds, split SHA256,
and per-fold composition stats. Idempotent: refuses to overwrite unless
--force (folds are FROZEN once written).
"""
import os, sys, json, argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import KBinsDiscretizer

sys.path.insert(0, os.path.dirname(__file__))
from vdiff_utils import load_data
from run_utils import FOLDS_JSON, split_sha256, verify_split

FOLD_SEED = 20260724  # fixed, independent of run seeds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()

    if os.path.exists(FOLDS_JSON) and not args.force:
        print(f"FROZEN folds already exist: {FOLDS_JSON} — not overwriting.")
        return

    train_df, test_df = load_data()
    sha = verify_split(train_df, test_df)

    disc  = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
    dbins = disc.fit_transform(train_df[['difficulty']]).ravel().astype(int)
    hasf  = (train_df['visual_description'].fillna('') != '').astype(int).to_numpy()
    strata = dbins * 2 + hasf  # 10 strata

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=FOLD_SEED)
    folds = []
    for k, (_, val_idx) in enumerate(skf.split(train_df, strata)):
        sub = train_df.iloc[val_idx]
        folds.append({
            'fold': k,
            'val_qids': sorted(int(q) for q in sub['QuestionId']),
            'n': len(val_idx),
            'beta_mean': float(sub['difficulty'].mean()),
            'beta_std': float(sub['difficulty'].std()),
            'figure_frac': float((sub['visual_description'].fillna('') != '').mean()),
        })
        print(f"fold {k}: n={len(val_idx)}  beta={folds[-1]['beta_mean']:.3f}"
              f"±{folds[-1]['beta_std']:.3f}  fig={folds[-1]['figure_frac']:.3f}")

    all_val = [q for f in folds for q in f['val_qids']]
    assert sorted(all_val) == sorted(int(q) for q in train_df['QuestionId'])
    assert len(all_val) == len(set(all_val)) == 580

    payload = {
        'split_sha256': sha,
        'fold_seed': FOLD_SEED,
        'stratify': 'difficulty_quintile_x_has_figure',
        'n_train': 580, 'n_test': 145,
        'folds': folds,
    }
    with open(FOLDS_JSON, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"\nsplit_sha256 = {sha}")
    print(f"Frozen folds written: {FOLDS_JSON}")


if __name__ == '__main__':
    main()

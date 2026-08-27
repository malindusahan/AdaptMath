"""Shared utilities for vDiff experiments."""
import os, json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import KBinsDiscretizer

# Repository root = parent of src/. Override any path with an environment variable.
BASE_DIR    = os.environ.get('VDIFF_ROOT',
                             os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR    = os.environ.get('VDIFF_DATA',    os.path.join(BASE_DIR, 'data'))
RESULTS_DIR = os.environ.get('VDIFF_RESULTS', os.path.join(BASE_DIR, 'results'))
LOGS_DIR    = os.environ.get('VDIFF_LOGS',    os.path.join(BASE_DIR, 'logs'))
DATA_CSV    = os.environ.get('VDIFF_ITEMS',   os.path.join(DATA_DIR, 'items.csv'))
# Image directory: the Eedi question images are NOT redistributed here.
# See data/README.md for how to obtain them, then point IMAGE_DIR at them.
IMAGE_DIR   = os.environ.get('VDIFF_IMAGES',  os.path.join(DATA_DIR, 'images'))


def resolve_image(rel_path):
    """Map an `image_path` entry from items.csv onto the local Eedi image directory.

    The Eedi question images are not redistributed with this repository; download
    them separately (see data/README.md) and set VDIFF_IMAGES to their location.
    """
    return os.path.join(IMAGE_DIR, os.path.basename(str(rel_path)))


def load_data():
    """Return (train_df, test_df) with `text_vd` column added."""
    df = pd.read_csv(DATA_CSV)
    df['visual_description'] = df['visual_description'].fillna('')
    df['text_vd'] = df.apply(
        lambda r: (r['text'] + ' [Figure: ' + r['visual_description'] + ']')
                   if r['visual_description'] else r['text'],
        axis=1,
    )
    train = df[df['split'] == 'train'].reset_index(drop=True)
    test  = df[df['split'] == 'test'].reset_index(drop=True)
    return train, test


def make_folds(df, n_splits=5, seed=42):
    """Stratified 5-fold by difficulty quantile bins."""
    disc = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
    bins = disc.fit_transform(df[['difficulty']]).ravel().astype(int)
    skf  = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(df, bins))


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        'rmse':     float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'mae':      float(mean_absolute_error(y_true, y_pred)),
        'pearson':  float(stats.pearsonr(y_true, y_pred)[0]),
        'spearman': float(stats.spearmanr(y_true, y_pred)[0]),
    }


def save_results(results, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    def _clean(o):
        # NaN/Inf are not valid JSON -> emit null so standard parsers don't choke
        if isinstance(o, float):
            return o if np.isfinite(o) else None
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_clean(v) for v in o]
        if isinstance(o, np.floating):
            f = float(o)
            return f if np.isfinite(f) else None
        if isinstance(o, np.integer):
            return int(o)
        return o

    with open(path, 'w') as f:
        json.dump(_clean(results), f, indent=2, allow_nan=False, default=float)
    print(f"Results saved: {path}")
